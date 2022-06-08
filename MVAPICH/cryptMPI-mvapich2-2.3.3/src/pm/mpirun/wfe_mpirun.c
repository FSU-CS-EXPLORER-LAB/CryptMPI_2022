/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include <src/pm/mpirun/mpirun_rsh.h>
#include <mpirun_ckpt.h>
#include <wfe_mpirun.h>
#include <debug_utils.h>
#include <mpispawn_error_codes.h>
#include <m_state.h>
#include <process.h>

#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef CR_FTB
#include <libftb.h>
#endif


static pthread_t wfe_tid;
static pthread_cond_t wfe_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t wfe_mutex = PTHREAD_MUTEX_INITIALIZER;

static int is_joined = 1;
static int is_running = 0;

static int
make_sfd_block (int sfd)
{
    int flags = fcntl(sfd, F_GETFL);

    if (-1 == flags) {
        /*
         * We shouldn't need the following perror call but PRINT_ERROR_ERRNO
         * doesn't seem to be working in the calling function.
         */
        perror("make_sfd_block (fcntl)");
        return -1;
    }

    flags &= ~O_NONBLOCK;

    if(fcntl(sfd, F_SETFL, flags)) {
        /*
         * We shouldn't need the following perror call but PRINT_ERROR_ERRNO
         * doesn't seem to be working in the calling function.
         */
        perror("make_sfd_block (fcntl)");
        return -1;
    }

    return 0;
}

static void
process_mpispawn_connection (int sfd)
{
    int code, id;
    extern int dpm_cnt;
    extern process_groups * pglist;

    /*
     * The next two read_socket calls should probably be preceded by the
     * reading of some magic value to reduce the chances of processing
     * messages from unrelated processes.
     */
    if (read_socket(sfd, &code, sizeof(int))) {
        PRINT_ERROR("failed to read mpispawn code");
        return;
    }

    if (read_socket(sfd, &id, sizeof(int))) {
        PRINT_ERROR("failed to read mpispawn id");
        return;
    }

    switch (code) {
        case MPISPAWN_DPM_REQ:
            /*
             * Handle DPM request
             */
            PRINT_DEBUG(DEBUG_Fork_verbose, "Dynamic spawn request from %d\n",
                    id);
            if (handle_spawn_req(sfd)) {
                PRINT_ERROR("Unable to process DPM spawn request\n");
                m_state_fail();
            }
            dpm_cnt++;
            break;
        case MPISPAWN_MPIPROCESS_NONZEROEXIT:
            /*
             * Maybe we should read the code from mpispawn and assign
             * this via `exit_code |= code' instead.
             */
            PRINT_DEBUG(DEBUG_Fork_verbose, "mpispawn %d reported an MPI "
                    "process exit with non-zero code\n", id);
            m_state_fail();
            break;
        case MPISPAWN_TRIGGER_MIGRATION:

#ifdef CR_FTB
            /*
             * Handle the migration request mpispawn
             */
            PRINT_DEBUG(DEBUG_FT_verbose,"Migration request from: "
                                        "%s\n", pglist->index[id]->hostname);
            FTB_receive_event_t mig_recv_event;
            memset(&mig_recv_event, 0, sizeof(FTB_receive_event_t));
            snprintf(mig_recv_event.event_name,FTB_MAX_EVENT_NAME,
                                            "FTB_MIGRATE_TRIGGER");
            snprintf(mig_recv_event.event_payload,
                                sizeof(pglist->index[id]->hostname),
                                pglist->index[id]->hostname);
            cr_ftb_callback(&mig_recv_event,NULL);
#else
            PRINT_ERROR("MVAPICH2 has not been configured with"
                                    " Process-Migration support!\n");
#endif
            break;
        default:
            /*
             * Assume abort from mpispawn
             */
            PRINT_ERROR("mpispawn_%d from node %s aborted: %s (%d)\n", id,
                                        pglist->index[id]->hostname,
                                        get_mpispawn_error_str(code), code);
            m_state_fail();
            pthread_exit(NULL);
            break;
    }
}

static void
cleanup_wfe_thread (void * arg)
{
    extern int is_running;
    is_running = 0;
}

static void *
wfe_thread (void * arg)
{
    struct wfe_params * params = arg;
    int wfe_socket;

    /*
     * Let start_wfe_thread know that it may continue. 
     */
    pthread_mutex_lock(&wfe_mutex);
    is_running = 1;
    pthread_cond_signal(&wfe_cond);
    pthread_mutex_unlock(&wfe_mutex);

    pthread_cleanup_push(&cleanup_wfe_thread, NULL);

    /*
     * First let the mpispawn processes checkin before waiting for errors
     */
    mpispawn_checkin(params->s);

    /*
     * Do not continue if something prevented a smooth transition from the
     * launch state to the run state.
     */
    if (M_RUN != m_state_transition(M_LAUNCH, M_RUN)) {
        PRINT_ERROR("Internal error: transition failed\n");
        m_state_fail();
        pthread_exit(NULL);
    }

    /*
     * Make sure socket is blocking as this thread only needs to be active
     * once connections are established.
     */
    if (make_sfd_block(params->s)) {
        /*
         * This error should be very unlikely unless there was a problem
         * with the file descriptor passed into this thread.
         */
        PRINT_ERROR_ERRNO("could not make socket block on i/o", errno);
        m_state_fail();
        pthread_exit(NULL);
    }

    for (;;) {
        /*
         * Wait for connection from any mpispawn process.  Opened
         * connections will be used to read any abnormal operation detected
         * by the mpispawn process.
         */
        wfe_socket = accept(params->s, (struct sockaddr *)params->sockaddr,
                &params->sockaddr_len); 

        if (0 > wfe_socket) {
#ifdef CKPT
            PRINT_DEBUG(DEBUG_Fork_verbose, "got wfe_socket %d\n", wfe_socket);

            if (M_RESTART == m_state_get()) {
                /*
                 * Ignore accept errors if we're in the restart phase of a
                 * checkpoint/restart.  Instead we will shutdown this thread
                 * so that the main thread can proceed with restarting.
                 */
                pthread_exit(NULL);
            }
#endif

            switch (errno) {
                case EINTR:
                    /*
                     * This indicates that accept was interrupted by a
                     * signal and can safely be resumed.
                     */
                    continue;
                default:
                    /*
                     * Unexpected error.  Print error message and shutdown
                     * this thread so that the main thread can exit the mpi
                     * job gracefully.
                     */
                    PRINT_ERROR_ERRNO("accept", errno);
                    m_state_fail();
                    pthread_exit(NULL);
            }
        } /* 0 > wfe_socket */

        process_mpispawn_connection(wfe_socket);
        close(wfe_socket);

#ifdef CKPT
        if (M_RESTART == m_state_get()) {
            pthread_exit(NULL);
        }
#endif /* CKPT */
    }

    pthread_cleanup_pop(0);
    return NULL;
}

void
start_wfe_thread (struct wfe_params * params)
{
    extern pthread_t wfe_tid;
    extern int is_joined;

    int error = pthread_create(&wfe_tid, NULL, &wfe_thread, params);

    if (error) {
        PRINT_ERROR_ERRNO("could not create wfe thread", error);
        m_state_fail();
        return;
    }

    /*
     * Don't return until the wfe_thread confirms that it is running.  This is
     * to prevent a race condition where stop_wfe_thread can be called before
     * is_running is set to 1.
     */
    pthread_mutex_lock(&wfe_mutex);
    while (!is_running) pthread_cond_wait(&wfe_cond, &wfe_mutex);
    pthread_mutex_unlock(&wfe_mutex);

    is_joined = 0;

    return;
}

extern void
stop_wfe_thread (void)
{
    extern pthread_t wfe_tid;
    extern int is_joined;
    extern int is_running;
    void * return_value;

    if (is_running) pthread_cancel(wfe_tid);
    if (!is_joined) pthread_join(wfe_tid, &return_value);
    is_joined = 1;
}

/* vi:set sw=4 sts=4 tw=80 expandtab: */
