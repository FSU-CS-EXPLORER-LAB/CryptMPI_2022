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

#include "mpirun_ckpt.h"

#ifdef CKPT


#include <sys/time.h>
#include <libcr.h>
#include <pthread.h>
#include "debug_utils.h"
#include "m_state.h"
#include "mpirun_params.h"


static pthread_t cr_tid = 0;

static int num_procs = 0;



static void *CR_Loop(void *);
static int CR_Callback(void *);



// ================================================
// Helpers to manage internal state for CR thread
// ================================================

typedef enum {
    CR_UNDEFINED = 0,
    CR_INIT = 1,
    CR_READY = 2,
    CR_REQUEST_CHECKPOINT = 3,
    CR_CHECKPOINT = 4,
    CR_RESTART = 5,
    CR_STOPPED = 6,
    CR_FINALIZED = 7,
} CR_state_t;

static CR_state_t cr_state = CR_UNDEFINED;
static pthread_mutex_t cr_state_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline void CR_state_lock() 
{
    int rv = pthread_mutex_lock(&cr_state_mutex);
    CHECK_ERROR_ERRNO( rv == 0, "pthread_mutex_lock() failed", errno );
}

static inline void CR_state_unlock() 
{
    int rv = pthread_mutex_unlock(&cr_state_mutex);
    CHECK_ERROR_ERRNO( rv == 0, "pthread_mutex_unlock() failed", errno );
}

static inline CR_state_t CR_state_transition_nolock( CR_state_t state ) 
{
    CR_state_t old_state = cr_state;
    cr_state = state;
    PRINT_DEBUG( DEBUG_CR_verbose>1, "cr_state transition: %d -> %d\n", old_state, state );
    return old_state;
}

static inline CR_state_t CR_state_transition( CR_state_t state ) 
{
    CR_state_lock();
    CR_state_t old_state = CR_state_transition_nolock( state );
    CR_state_unlock();
    return old_state;
}

// ================================================



static int restart_version = -1;

static unsigned int nspawns = 0;

#ifndef CR_FTB
static int *mpirun_fd;
static int mpirun_port;
static int mpirun_listen_fd = 0;
static int create_connections();
static int accept_connections();
static void clean_connections();
#endif

static unsigned long starting_time = 0;
static unsigned long last_ckpt = 0;

// TODO Change to static???
char sessionid[CR_SESSION_MAX];

static int checkpoint_version = 1;
static int checkpoint_interval = 0;
static int max_save_ckpts = 0;
static int max_ckpts = 0;

static char ckpt_filename[CR_MAX_FILENAME];

#ifdef CR_FTB

char *current_spare_host;

struct spawn_info_s *spawninfo;

static FTB_client_t ftb_cinfo;
static FTB_client_handle_t ftb_handle;
static FTB_event_info_t cr_ftb_events[] = CR_FTB_EVENT_INFO;
static FTB_subscribe_handle_t shandle;
static int ftb_init_done;

static pthread_cond_t cr_ftb_ckpt_req_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t cr_ftb_ckpt_req_mutex = PTHREAD_MUTEX_INITIALIZER;
static int cr_ftb_ckpt_req;
static int cr_ftb_app_ckpt_req;
static int cr_ftb_finalize_ckpt;

static int cr_ftb_init(int nprocs);
static void cr_ftb_finalize();
static int cr_ftb_wait_for_resp(int);

// =====================================================
// For Migration

int sparehosts_on;
char sparehostfile[HOSTFILE_LEN + 1];
char **sparehosts;
int nsparehosts;
static int sparehosts_idx;

static char cr_mig_src_host[32];
static char cr_mig_tgt_host[32];

static int get_src_tgt(char *, char *, char *);

// =====================================================

#endif                          /* CR_FTB */


#if defined(CKPT) && defined(CR_AGGRE)
extern int use_aggre;           // by default we use CR-aggregation
extern int use_aggre_mig;       // by default, enable aggre-mig
#endif


void set_ckpt_nprocs(int nprocs)
{
    num_procs = nprocs;
}

// In CR_initialize(), put only code that must be called once
// because CR_initialize() won't be called at restart
// Code that needs to be run after each restart should go in CR_thread_start() or CR_Loop()
int CR_initialize()
{
    time_t tm;
    struct tm *stm;

    int rv = pthread_mutex_init( &cr_state_mutex, NULL );
    if ( rv != 0 ) {
        PRINT_ERROR_ERRNO( "pthread_mutex_init() failed", errno );
        return -1;
    }

    CR_state_transition( CR_INIT );

    cr_client_id_t cr_id = cr_init();
    if (cr_id < 0) {
        PRINT_ERROR("BLCR call cr_init() failed\n");
        return -2;
    }

    if (cr_register_callback(CR_Callback, (void *) NULL, CR_THREAD_CONTEXT) == -1) {
        PRINT_ERROR("BLCR call cr_register_callback() failed with error %d: %s\n", errno, cr_strerror(errno));
        return -3;
    }

    strncpy(ckpt_filename, DEFAULT_CHECKPOINT_FILENAME, CR_MAX_FILENAME);

    tm = time(NULL);
    if ((time_t) tm == -1) {
        PRINT_ERROR("time() failed\n");
        return -4;
    }

    stm = localtime(&tm);
    if (!stm) {
        PRINT_ERROR("localtime() failed\n");
        return -5;
    }

    snprintf(sessionid, CR_SESSION_MAX, "%d%d%d%d%d", stm->tm_yday, stm->tm_hour, stm->tm_min, stm->tm_sec, getpid());
    sessionid[CR_SESSION_MAX - 1] = '\0';

    return 0;
}


int CR_finalize()
{
    CR_state_transition( CR_FINALIZED );
    return 0;
}


int CR_thread_start( unsigned int n )
{
    nspawns = n;

    if ( m_state_get() == M_RESTART ) {
        ASSERT_MSG( restart_version >= 0, "Internal error" );
    }

#ifndef CR_FTB
    // This must be called before mpispawn are started
    // Do not move this to CR_Loop()
    if (USE_LINEAR_SSH) {
        if (!show_on) {
            int rv = create_connections();
            if ( rv != 0 ) {
                return -1;
            }
        }
    }
#endif

    struct timeval starting;
    gettimeofday(&starting, NULL);
    starting_time = last_ckpt = starting.tv_sec;

    // Check and set CR state
    CR_state_lock();
    ASSERT_MSG( cr_state == CR_INIT || cr_state == CR_STOPPED, "Internal Error\n");
    CR_state_transition_nolock( CR_INIT );
    CR_state_unlock();

    if (pthread_create(&cr_tid, NULL, CR_Loop, NULL) < 0) {
        PRINT_ERROR_ERRNO("pthread_create() failed", errno);
        cr_tid = 0;
        return -1;
    }

    return 0;
}


int CR_thread_stop( int blocking )
{
    PRINT_DEBUG( DEBUG_CR_verbose>1, "CR_thread_stop(blocking=%d) called\n", blocking);
    CR_state_transition( CR_STOPPED );
    nspawns = 0;
    if (cr_tid) {
        if (blocking) {
            pthread_join(cr_tid, NULL);
        }
        cr_tid = 0;
    }

#ifdef CR_FTB
    /* Nothing to be done */
#else
    clean_connections();
#endif

    return 0;
}


#ifndef CR_FTB
int create_connections()
{
    struct sockaddr_in cr_sa;
    int val;

    mpirun_fd = malloc(nspawns * sizeof(int));
    if (!mpirun_fd) {
        PRINT_ERROR_ERRNO("malloc() failed",errno);
        return -1;
    }

    /* Create connections for the mpispawns to connect back */
    mpirun_listen_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (mpirun_listen_fd < 0) {
        PRINT_ERROR_ERRNO("socket() failed",errno);
        return -1;
    }

    memset(&cr_sa, 0, sizeof(cr_sa));
    cr_sa.sin_family = AF_INET;
    cr_sa.sin_addr.s_addr = INADDR_ANY;
    cr_sa.sin_port = 0;

    if (bind(mpirun_listen_fd, (struct sockaddr *) &cr_sa, sizeof(cr_sa)) < 0) {
        PRINT_ERROR_ERRNO("bind() failed",errno);
        return -1;
    }

    val = sizeof(cr_sa);
    if (getsockname(mpirun_listen_fd, &cr_sa, (socklen_t *) & val) < 0) {
        PRINT_ERROR_ERRNO("getsockname() failed",errno);
        close(mpirun_listen_fd);
        return -1;
    }

    mpirun_port = ntohs(cr_sa.sin_port);

    if (listen(mpirun_listen_fd, nspawns) < 0) {
        PRINT_ERROR_ERRNO("listen() failed",errno);
        close(mpirun_listen_fd);
        return -1;
    }
    
    return 0;
}


int accept_connections()
{
    int i;
    for (i = 0; i < nspawns; i++) {
        // Ignore all EINTR 'Interrupted system call' errors
        do {
            mpirun_fd[i] = accept(mpirun_listen_fd, NULL, NULL);
        } while (mpirun_fd[i] < 0 && errno == EINTR);
        if (mpirun_fd[i] < 0) {
            PRINT_ERROR_ERRNO( "accept(mpirun_fd[%d]) failed", errno, i );
            close(mpirun_listen_fd);
            return -1;
        }
    }

    close(mpirun_listen_fd);
    mpirun_listen_fd = 0;
    return 0;
}

void clean_connections()
{
    free(mpirun_fd);
}
#endif

char *create_mpispawn_vars(char *mpispawn_env)
{

    char *tmp = NULL;
#ifdef CR_FTB
    /* Keep mpispawn happy. Pass some junk value */
    tmp = mkstr("%s MPISPAWN_MPIRUN_CR_PORT=%d", mpispawn_env, 0);
#else
    tmp = mkstr("%s MPISPAWN_MPIRUN_CR_PORT=%d", mpispawn_env, mpirun_port);
#endif
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        PRINT_ERROR("[1] mkstr() failed\n");
        goto allocation_error;
    }

    // TODO Should be useless, MPISPAWN_CR_CKPT_CNT should be enough
    // Remove this in the future
    int restart_context = 0;
    if ( restart_version >= 0 ) restart_context = 1;
    tmp = mkstr("%s MPISPAWN_CR_CONTEXT=%d", mpispawn_env, restart_context);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        PRINT_ERROR("[2] mkstr() failed\n");
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_CR_SESSIONID=%s", mpispawn_env, sessionid);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        PRINT_ERROR("[3] mkstr() failed\n");
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_CR_CKPT_CNT=%d", mpispawn_env, restart_version);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        PRINT_ERROR("[4] mkstr() failed\n");
        goto allocation_error;
    }

    return tmp;

  allocation_error:
    if (mpispawn_env) {
        PRINT_ERROR("Error: current mpispawn_env = '%s'\n", mpispawn_env);
        free(mpispawn_env);
    }
    exit(EXIT_FAILURE);

}

// Request a checkpoint of the local process
// The return value is
// - negative in case of error
// - zero when successfully resuming after the checkpoint
// - positive when restarting from the checkpoint
static int request_checkpoint( const char* filename ) 
{
    cr_checkpoint_args_t cr_file_args;
    cr_checkpoint_handle_t cr_handle;
    int cr_fd = -1;
    int return_code = 0;

    // Check current state
    CR_state_lock();
    if ( cr_state != CR_READY ) {
        switch( cr_state ) {
            case CR_REQUEST_CHECKPOINT:
            case CR_CHECKPOINT:
            {
                PRINT_ERROR("Error: Already checkpointing... (cr_state=%d)\n", cr_state);
                return_code = -10;
                break;
            }
            default:
            {
                PRINT_ERROR("Error: Not ready to checkpoint... (cr_state=%d)\n", cr_state);
                return_code = -11;
                break;
            }
        }
        CR_state_unlock();
        goto error;
    } else {
        /* Everything fine, proceed to checkpoint request */
        CR_state_transition_nolock( CR_REQUEST_CHECKPOINT );
    }
    CR_state_unlock();

    fprintf(stderr,"mpirun_rsh opening file %s\n", filename);
    cr_fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if ( cr_fd < 0 ) {
        PRINT_ERROR_ERRNO("Failed to open checkpoint file '%s'", errno, filename);
        return_code = -1;
        goto error;
    }

    int ret = cr_initialize_checkpoint_args_t(&cr_file_args);
    if (ret < 0) {
        PRINT_ERROR("BLCR call cr_initialize_checkpoint_args_t() failed\n");
        return_code = -2;
        goto error;
    }

    cr_file_args.cr_scope = CR_SCOPE_PROC;
    cr_file_args.cr_target = getpid();
    cr_file_args.cr_fd = cr_fd;
    cr_file_args.cr_signal = 0;
    cr_file_args.cr_timeout = 0;
    cr_file_args.cr_flags &= ~CR_CHKPT_DUMP_ALL; /* Save None */

    /* Request a checkpoint */
    PRINT_DEBUG( DEBUG_CR_verbose, "mpirun_rsh cr_request_checkpoint() with file '%s'\n", filename );
    ret = cr_request_checkpoint(&cr_file_args, &cr_handle);
    PRINT_DEBUG( DEBUG_CR_verbose>1, "cr_request_checkpoint() returned %d\n", ret );
    if (ret < 0) {
        PRINT_ERROR("BLCR call cr_request_checkpoint() failed with error %d: %s\n", errno, cr_strerror(errno));
        return_code = -3;
        goto error;
    }

    // Wait for the end of the checkpoint, and retry while interrupted
    PRINT_DEBUG( DEBUG_CR_verbose, "cr_poll_checkpoint()\n" );
    do {
        ret = cr_poll_checkpoint(&cr_handle, NULL);
    } while (ret == CR_POLL_CHKPT_ERR_PRE && errno == EINTR);
    PRINT_DEBUG( DEBUG_CR_verbose>1, "cr_poll_checkpoint() returned %d\n", ret );

    // Check the result of the checkpoint
    if (ret == CR_POLL_CHKPT_ERR_POST && errno == CR_ERESTARTED) { 
        // We are restarting, ignore this error code

        // The checkpoint file is not opened at restart
        cr_fd = -1;

        // Positive value means restart
        return_code = 1;
        return return_code;
    } else if (ret < 0) {
        // Checkpoint failed
        PRINT_ERROR("BLCR call cr_poll_checkpoint() failed with error %d: %s\n", errno, cr_strerror(errno));

        // Negative value for failure
        return_code = -4;
        goto error;
    } else if (ret == 0) {
        // 0 means that the checkpoint is in progress
        // It should never happen because we don't specify any timeout when calling cr_poll_checkpoint()
        ASSERT_MSG( 0==1, "Internal error\n");
    }

    // Close the checkpoint file
    ASSERT_MSG( cr_fd>=0, "Internal error\n");
    ret = close(cr_fd);
    cr_fd = -1;
    PRINT_DEBUG( DEBUG_CR_verbose, "close() returned %d\n", ret );
    if (ret < 0) {
        PRINT_ERROR_ERRNO("Failed to close file '%s'", errno, filename);
        return_code = -5;
        goto error;
    }

    // If we are here, it means that everything went good
    ASSERT_MSG( return_code==0, "Internal error\n");
    return return_code;

error:
    // An error happened, cleanup and return properly
    if ( cr_fd >= 0 ) {
        close( cr_fd );
        cr_fd = -1;
    }

    // If the request failed, ie not the checkpoint itself
    // Restore the CR_READY state
    CR_state_lock();
    if ( cr_state == CR_REQUEST_CHECKPOINT ) {
        CR_state_transition_nolock( CR_READY );
    }
    CR_state_unlock();

    return return_code;
}


static void *CR_Loop(void *arg)
{

#ifdef CR_FTB
    if (cr_ftb_init(nprocs))
        exit(EXIT_FAILURE);
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    char valstr[CRU_MAX_VAL_LEN];
    fd_set set;
    int i, n, nfd = 0, ret;

    if (USE_LINEAR_SSH) {
        if (!show_on) {
            // This call is blocking (in case of error)
            // It must be kept in the CR thread
            int rv = accept_connections();
            if ( rv != 0 ) {
                m_state_fail();
                pthread_exit(NULL);
            }
        }
    }
#endif

    CR_state_transition( CR_READY );

#ifdef CR_FTB
    // The main thread of mpirun_rsh is waiting for the CR thread to connect to FTB
    // before starting the mpispawn processes
    // Make the transition to the M_LAUNCH state
    // Use to signal the main thread of mpirun_rsh
    // This should be removed once we remove the use of FTB for this
    if (M_LAUNCH != m_state_transition(M_INITIALIZE|M_RESTART, M_LAUNCH)) {
        PRINT_ERROR("Internal error: transition failed\n");
        m_state_fail();
        pthread_exit(NULL);
    }
#endif

    if ( checkpoint_interval > 0 ) {
        PRINT_DEBUG( DEBUG_CR_verbose, "Checkpoint interval = %d s\n", checkpoint_interval );
    }

    while (1) {

        // Check if CR_thread_stop() has been called
        CR_state_lock();
        if (cr_state == CR_STOPPED) {
            CR_state_unlock();
            PRINT_DEBUG( DEBUG_CR_verbose, "Exit CR thread\n" );
            pthread_exit(NULL);
        }
        CR_state_unlock();

#ifdef CR_FTB
        sleep(1);
        if (cr_ftb_app_ckpt_req || cr_ftb_finalize_ckpt)
#else
        nfd = 0;
        FD_ZERO(&set);
        for (i = 0; i < nspawns; i++) {
            FD_SET(mpirun_fd[i], &set);
            nfd = (nfd >= mpirun_fd[i]) ? nfd : mpirun_fd[i];
        }
        nfd += 1;

        struct timeval tv;
        do {
            tv.tv_sec = 1;
            tv.tv_usec = 0;
            ret = select(nfd, &set, NULL, NULL, &tv);
        } while ( ret==-1 && errno==EINTR );
            
        if (ret < 0) {
            PRINT_ERROR_ERRNO("select(nfd=%d, set, NULL, NULL, tv={%lu,%lu}) failed", errno, nfd, tv.tv_sec, tv.tv_usec);
            return ((void *) -1);
        } else if (ret > 0)
#endif
        {

            // Do not go further if not ready
            // This avoid bad interactions on file descriptors with the CR_Callback thread
            CR_state_lock();
            if (cr_state != CR_READY) {
                CR_state_unlock();
                continue;
            }
            CR_state_unlock();

#ifdef CR_FTB
            if (cr_ftb_app_ckpt_req)
#else
            for (i = 0; i < nspawns; i++) {

                if (!FD_ISSET(mpirun_fd[i], &set))
                    continue;

                n = CR_MPDU_readline(mpirun_fd[i], cr_msg_buf, MAX_CR_MSG_LEN);
                if (n == 0)
                    continue;

                if (CR_MPDU_parse_keyvals(cr_msg_buf) < 0)
                    break;

                CR_MPDU_getval("cmd", valstr, CRU_MAX_VAL_LEN);

                if (strcmp(valstr, "app_ckpt_req") == 0)
#endif
                {
#ifdef CR_FTB
                    cr_ftb_app_ckpt_req = 0;
#endif
                    unsigned int current_version = checkpoint_version;
                    char buf[CR_MAX_FILENAME];
                    sprintf(buf, "%s.%d.sync", ckpt_filename, current_version);
                    PRINT_DEBUG( DEBUG_CR_verbose, "Checkpoint request from the application\n" );
                    int rv = request_checkpoint( buf );
                    if ( rv < 0 ) {
                        PRINT_ERROR( "Checkpoint failed\n" );
                    } else if ( rv > 0 ) {
                        PRINT_DEBUG( DEBUG_CR_verbose, "Restarting from checkpoint\n" );
                        // Terminate the thread
                        pthread_exit(NULL);
                    }
                }
#ifdef CR_FTB
                else if (cr_ftb_finalize_ckpt)
#else
                else if (strcmp(valstr, "finalize_ckpt") == 0)
#endif
                {
                    // One process called MPI_finalize()
#ifdef CR_FTB
                    cr_ftb_finalize_ckpt = 0;
#endif
                    // Terminate the CR_thread
                    CR_state_transition( CR_STOPPED );
                    pthread_exit(NULL);
                }
#ifndef CR_FTB
            }
#endif
        } else {

            struct timeval now;
            gettimeofday(&now, NULL);
            unsigned long time_counter = (now.tv_sec - starting_time);

            if ((checkpoint_interval > 0) && (now.tv_sec > last_ckpt) && (time_counter % checkpoint_interval == 0)) {
                // Inject a checkpoint
                unsigned int current_version = checkpoint_version;
                char buf[CR_MAX_FILENAME];
                sprintf(buf, "%s.%d.auto", ckpt_filename, current_version);
                PRINT_DEBUG( DEBUG_CR_verbose, "Automatic checkpoint request\n" );
                int rv = request_checkpoint( buf );
                if ( rv < 0 ) {
                    // Checkpoint failed
                    PRINT_ERROR( "Checkpoint failed\n" );
                } else if ( rv > 0 ) {
                    // Restarting from checkpoint
                    PRINT_DEBUG( DEBUG_CR_verbose, "Restarting from checkpoint\n" );
                    // Terminate the thread
                    pthread_exit(NULL);
                } else {
                    // Resume after a successful checkpoint

                    // Remove the ealier checkpoints
                    if ( (max_save_ckpts > 0) && (max_save_ckpts < current_version) ) {
                        sprintf(buf, "%s.%d.auto", ckpt_filename, current_version - max_save_ckpts);
                        PRINT_DEBUG( DEBUG_CR_verbose, "[2] unlink() file '%s' \n", buf );
                        int ret = unlink(buf);
                        if ( ret != 0) {
                            PRINT_ERROR_ERRNO("unlink() failed", errno);
                        }
                    }
                }
            }
        }
    }

    return (0);
}

/*
 * This stuff may be in the environment as well.  Call this function before
 * save_ckpt_vars is called so the old way takes precedence.
 */
void save_ckpt_vars_env(void)
{
    if (getenv("MV2_CKPT_FILE")) {
        strncpy(ckpt_filename, getenv("MV2_CKPT_FILE"), CR_MAX_FILENAME);
    }

    if (getenv("MV2_CKPT_INTERVAL")) {
        checkpoint_interval = atof(getenv("MV2_CKPT_INTERVAL")) * 60.0;
    }

    if (getenv("MV2_CKPT_MAX_SAVE_CKPTS")) {
        max_save_ckpts = atoi(getenv("MV2_CKPT_MAX_SAVE_CKPTS"));
    }

    if (getenv("MV2_CKPT_MAX_CKPTS")) {
        max_ckpts = atoi(getenv("MV2_CKPT_MAX_CKPTS"));
    }
}

void save_ckpt_vars(char *name, char *value)
{
    if (strcmp(name, "MV2_CKPT_FILE") == 0) {
        strncpy(ckpt_filename, value, CR_MAX_FILENAME);
    } else if (strcmp(name, "MV2_CKPT_INTERVAL") == 0) {
        checkpoint_interval = atof(value) * 60.0;
    } else if (strcmp(name, "MV2_CKPT_MAX_SAVE_CKPTS") == 0) {
        max_save_ckpts = atoi(value);
    } else if (strcmp(name, "MV2_CKPT_MAX_CKPTS") == 0) {
        max_ckpts = atoi(value);
    }
}

// This CR_Callback actually proceed to the checkpoint
// It is called automatically by BLCR when:
// - the user run cr_checkpoint BLCR command on mpirun_rsh
// - when a the request_checkpoint() function is called (automatic or application checkpoint)
static int CR_Callback(void *arg)
{
    int ret;
    PRINT_DEBUG( DEBUG_CR_verbose, "CR_Callback() called\n" );

    // Check current state
    CR_state_lock();
    if ( cr_state == CR_READY ) {
        // It is a user checkpoint
        // Proceed to checkpoint
        CR_state_transition_nolock( CR_CHECKPOINT );
        CR_state_unlock();
    } else if ( cr_state == CR_REQUEST_CHECKPOINT ) {
        // Checkpoint has been trigger by request_checkpoint()
        // Proceed to checkpoint
        CR_state_transition_nolock( CR_CHECKPOINT );
        CR_state_unlock();
    } else {
        // Not ready for checkpoint
        switch( cr_state ) {
            case CR_REQUEST_CHECKPOINT:
            case CR_CHECKPOINT:
            {
                PRINT_ERROR("Error: Already checkpointing... (cr_state=%d)\n", cr_state);
                break;
            }
            default:
            {
                PRINT_ERROR("Error: Not ready to checkpoint... (cr_state=%d)\n", cr_state);
                break;
            }
        }
        CR_state_unlock();
        // Abort current checkpoint
        int ret = cr_checkpoint(CR_CHECKPOINT_TEMP_FAILURE);
        if (ret != -CR_ETEMPFAIL) {
            PRINT_ERROR("BLCR call cr_checkpoint() return unexpected value %d: %s\n", ret, cr_strerror(-ret));
        }
        // Exit callback
        return 0;
    }

    struct timeval now;
    gettimeofday(&now, NULL);
    last_ckpt = now.tv_sec;


    // TODO send this version number to the MPI processes
    unsigned int current_version = checkpoint_version;
    unsigned int checkpoint_failure = 0;

    checkpoint_version++;

    PRINT_DEBUG( DEBUG_CR_verbose, "Proceed to checkpoint version %d\n", current_version );


#ifdef CR_FTB
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;

    SET_EVENT(eprop, FTB_EVENT_NORMAL, " ");
    ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_CHECKPOINT), &eprop, &ehandle);
    if (ret != FTB_SUCCESS) {
        PRINT_ERROR("FTB_Publish() failed with %d\n", ret);
        checkpoint_failure = 1;
    } else {
        ret = cr_ftb_wait_for_resp(num_procs);
        if (ret) {
            PRINT_ERROR("cr_ftb_wait_for_resp() failed\n");
            checkpoint_failure = 1;
        }
    }

#else
    int Progressing, nfd = 0;
    char val[CRU_MAX_VAL_LEN];
    fd_set set;
    int i;
    char buf[MAX_CR_MSG_LEN];

    // Request all processes to take a checkpoint 
    sprintf(buf, "cmd=ckpt_req file=%s\n", ckpt_filename);
    for (i = 0; i < nspawns; i++) {
        CR_MPDU_writeline(mpirun_fd[i], buf);
    }

    // Wait for Checkpoint to finish
    Progressing = num_procs;
    while (Progressing && !checkpoint_failure) {

        FD_ZERO(&set);
        for (i = 0; i < nspawns; i++) {
            FD_SET(mpirun_fd[i], &set);
            nfd = (nfd >= mpirun_fd[i]) ? nfd : mpirun_fd[i];
        }
        nfd += 1;

        int ret = select(nfd, &set, NULL, NULL, NULL);
        if (ret < 0) {
            PRINT_ERROR_ERRNO("select() failed", errno);
            checkpoint_failure = 1;
            break;
        }

        for (i = 0; i < nspawns; i++) {

            if (!FD_ISSET(mpirun_fd[i], &set))
                continue;

            CR_MPDU_readline(mpirun_fd[i], buf, MAX_CR_MSG_LEN);
            PRINT_DEBUG( DEBUG_CR_verbose>2, "CR_MPDU_readline: i=%d, buf=\"%s\"\n", i, buf );

            if (CR_MPDU_parse_keyvals(buf) < 0) {
                PRINT_ERROR("CR_MPDU_parse_keyvals() failed\n");
                checkpoint_failure = 1;
                break;
            }

            CR_MPDU_getval("result", val, CRU_MAX_VAL_LEN);
            PRINT_DEBUG( DEBUG_CR_verbose>2, "CR_MPDU_getval: i=%d, result=\"%s\"\n", i, buf );

            if (strcmp(val, "succeed") == 0) {
                --Progressing;
                continue;
            } else if (strcmp(val, "finalize_ckpt") == 0) {
                // One process called MPI_finalize()
                PRINT_ERROR("End of the application. Aborting checkpoint...\n");

                // Terminate the CR_thread
                CR_state_transition( CR_STOPPED );

                checkpoint_failure = 1;
                break;
            } if (strcmp(val, "fail") == 0) {
                PRINT_ERROR("Checkpoint of a process failed\n");
                checkpoint_failure = 1;
                break;
            } else {
                PRINT_ERROR("Unexpected results from %d: \"%s\"\n", i, val);
                checkpoint_failure = 1;
                break;
            }
        }
    }                           /* while(Progressing) */

#endif                          /* CR_FTB */



    if ( checkpoint_failure == 1 ) {
        PRINT_ERROR("Some processes failed to checkpoint. Abort checkpoint...\n");
        int ret = cr_checkpoint(CR_CHECKPOINT_TEMP_FAILURE);
        if (ret != -CR_ETEMPFAIL) {
            PRINT_ERROR("BLCR call cr_checkpoint() returned unexpected value %d: %s\n", ret, cr_strerror(-ret));
        }

        // TODO mark this checkpoint as failed

        // Go back to the CR_READY state if not stopped
        CR_state_lock();
        if ( cr_state == CR_CHECKPOINT ) {
            CR_state_transition_nolock( CR_READY );
        }
        CR_state_unlock();

        return 0;
    } 


    // At this point, all the MPI processes have succesfully checkpointed
    // Then, proceed to local checkpoint of mpirun_rsh

#if defined(CKPT) && defined(CR_FTB)
    cr_ftb_finalize();
#endif

    ret = cr_checkpoint(CR_CHECKPOINT_READY);
    PRINT_DEBUG( DEBUG_CR_verbose, "cr_checkpoint() returned %d\n", ret );

    if (ret < 0) {
        // The checkpoint failed
        PRINT_ERROR("BLCR call cr_checkpoint() failed with error %d: %s\n", ret, cr_strerror(-ret));

        // TODO mark this checkpoint as failed

        // Go back to the CR_READY state
        CR_state_transition( CR_READY );

    } else if (ret == 0) {
        // The checkpoint was successful
        PRINT_DEBUG( DEBUG_CR_verbose, " Returned from cr_checkpoint(): resume execution normally\n" );

#if defined(CKPT) && defined(CR_FTB)
        // Reconnect to FTB
        cr_ftb_init(num_procs);
#endif
        // TODO mark this checkpoint as successful

        // Go back to the CR_READY state
        CR_state_transition( CR_READY );

    } else if (ret) {
        // We are restarting from a checkpoint
        PRINT_DEBUG( DEBUG_CR_verbose, " Returned from cr_checkpoint(: restart execution\n" );

        // Set the restart version, which will be used to restart the processes
        restart_version = current_version;

        // Change to CR_RESTART state
        CR_state_transition( CR_RESTART );

        // Change mpirun_rsh state
        m_state_transition(M_RUN, M_RESTART);

    }

    return 0;
}

#ifdef CR_FTB

//static int cr_ftb_init(int nprocs, char *sessionid)
int cr_ftb_init(int nprocs)
{
    int ret;
    char *subscription_str;
    if (ftb_init_done)
        return 0;
    ftb_init_done = 1;

    memset(&ftb_cinfo, 0, sizeof(ftb_cinfo));
    strcpy(ftb_cinfo.client_schema_ver, "0.5");
    strcpy(ftb_cinfo.event_space, "FTB.STARTUP.MV2_MPIRUN");
    strcpy(ftb_cinfo.client_name, "MV2_MPIRUN");

    /* sessionid should be <= 16 bytes since client_jobid is 16 bytes. */
    snprintf(ftb_cinfo.client_jobid, FTB_MAX_CLIENT_JOBID, "%s", sessionid);

    strcpy(ftb_cinfo.client_subscription_style, "FTB_SUBSCRIPTION_BOTH");
    ftb_cinfo.client_polling_queue_len = 64;    //nprocs;

    ret = FTB_Connect(&ftb_cinfo, &ftb_handle);
    if (ret != FTB_SUCCESS) {
        PRINT_ERROR("FTB_Connect() failed with %d\n", ret);
        goto err_connect;
    }
    
    ret = FTB_Declare_publishable_events(ftb_handle, NULL, cr_ftb_events, CR_FTB_EVENTS_MAX);
    if (ret != FTB_SUCCESS) {
        PRINT_ERROR("FTB_Declare_publishable_events() failed with %d\n", ret);
        goto err_declare_events;
    }

    subscription_str = malloc(sizeof(char) * FTB_MAX_SUBSCRIPTION_STR);
    if (!subscription_str) {
        PRINT_ERROR_ERRNO("Failed allocate subscription_str: malloc() failed", errno);
        goto err_malloc;
    }

    snprintf(subscription_str, FTB_MAX_SUBSCRIPTION_STR, "event_space=FTB.MPI.MVAPICH2 , jobid=%s", sessionid);
    ret = FTB_Subscribe(&shandle, ftb_handle, subscription_str, cr_ftb_callback, NULL);

    snprintf(subscription_str, FTB_MAX_SUBSCRIPTION_STR, "event_space=FTB.STARTUP.MV2_MPISPAWN , jobid=%s", sessionid);
    //"event_space=FTB.STARTUP.MV2_MPISPAWN" );
    ret = FTB_Subscribe(&shandle, ftb_handle, subscription_str, cr_ftb_callback, NULL);

    /// subscribe to migration trigger
    snprintf(subscription_str, FTB_MAX_SUBSCRIPTION_STR, "event_space=FTB.MPI.MIG_TRIGGER");
    ret = FTB_Subscribe(&shandle, ftb_handle, subscription_str, cr_ftb_callback, NULL);

    free(subscription_str);
    if (ret != FTB_SUCCESS) {
        PRINT_ERROR("FTB_Subscribe() failed with %d\n", ret);
        goto err_subscribe;
    }

    ftb_init_done = 1;

    return (0);

  err_connect:
    ret = -1;
    goto exit_connect;

  err_declare_events:
    ret = -2;
    goto exit_declare_events;

  err_malloc:
    ret = -3;
    goto exit_malloc;

  err_subscribe:
    ret = -4;
    goto exit_subscribe;

  exit_subscribe:
  exit_malloc:
  exit_declare_events:
    FTB_Disconnect(ftb_handle);

  exit_connect:
    return (ret);
}

static void cr_ftb_finalize()
{
    int ret = 0;
    if (ftb_init_done) {
        ftb_init_done = 0;
        ret = FTB_Unsubscribe(&shandle);
        usleep(20000);
        ret = FTB_Disconnect(ftb_handle);
    }
    PRINT_DEBUG(DEBUG_CR_verbose, "Has close FTB: ftb_init_done=%d, ftb-disconnect ret %d\n", ftb_init_done, ret);
}

static int cr_ftb_wait_for_resp(int nprocs)
{
    pthread_mutex_lock(&cr_ftb_ckpt_req_mutex);
    PRINT_DEBUG(DEBUG_CR_verbose, "wait for nprocs %d \n", nprocs);
    cr_ftb_ckpt_req += nprocs;
    while (cr_ftb_ckpt_req > 0)
        pthread_cond_wait(&cr_ftb_ckpt_req_cond, &cr_ftb_ckpt_req_mutex);
    pthread_mutex_unlock(&cr_ftb_ckpt_req_mutex);

    if (cr_ftb_ckpt_req == 0) {
        return (0);
    } else {
        PRINT_DEBUG(DEBUG_CR_verbose, "cr_ftb_wait_for_resp() returned %d\n", cr_ftb_ckpt_req);
        return (-1);
    }
}

#ifdef CR_AGGRE
int cr_ftb_aggre_based_mig(char *src)
{
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    char buf[MAX_CR_MSG_LEN];
    char tmpbuf[16];
    int i, ret;
    int isrc = -1, itgt = -1;
    char *tgt;

    tgt = sparehosts[sparehosts_idx];
    PRINT_DEBUG(DEBUG_CR_verbose, "enter: src=%s, tgt=%s, tgt-idx=%d\n", src, tgt, sparehosts_idx);

    //// find src and tgt node idx
    for (i = 0; i < pglist->npgs; i++) {
        if (strcmp(pglist->data[i].hostname, src) == 0)
            isrc = i;
        if (strcmp(pglist->data[i].hostname, tgt) == 0)
            itgt = i;
    }
    if (isrc == -1) {
        PRINT_ERROR("Source node '%s' not found\n", src);
        return -1;
    }
    if (itgt == -1) {
        PRINT_ERROR("Target node '%s' not found\n", tgt);
        return -1;
    }

    snprintf(buf, MAX_CR_MSG_LEN, "%s %s %ld ", src, tgt, pglist->data[isrc].npids);

    /// find all proc-ranks to be migrated
    for (i = 0; i < pglist->data[isrc].npids; i++) {
        sprintf(tmpbuf, "%d ", pglist->data[isrc].plist_indices[i]);
        strncat(buf, tmpbuf, MAX_CR_MSG_LEN);
    }
    PRINT_DEBUG(DEBUG_CR_verbose, "[Aggre-Based Mig]: init a mig: \"%s\"\n", buf);

    sparehosts_idx++;

    SET_EVENT(eprop, FTB_EVENT_NORMAL, buf);
    ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_MIGRATE), &eprop, &ehandle);
    if (ret != FTB_SUCCESS) {
        PRINT_ERROR("FTB_Publish failed with %d\n", ret);
        return -1;
    }
    return 0;
}
#endif

int cr_ftb_callback(FTB_receive_event_t * revent, void *arg)
{
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    char buf[MAX_CR_MSG_LEN];
    int ret;
    char cnum[16];
    process_group tmp_pg;
    PRINT_DEBUG(DEBUG_CR_verbose, "Got event %s from %s: payload=\"%s\"\n", revent->event_name, revent->client_name, revent->event_payload);

    /* TODO: Do some sanity checking to see if this is the intended target */

    if (!strcmp(revent->event_name, EVENT(MPI_PROCS_CKPTED))) {
        pthread_mutex_lock(&cr_ftb_ckpt_req_mutex);
        if (cr_ftb_ckpt_req <= 0) {
            PRINT_ERROR("Internal error: Got CR_FTB_CKPT_DONE but cr_ftb_ckpt_req not set\n");
            cr_ftb_ckpt_req = -1;
            pthread_cond_signal(&cr_ftb_ckpt_req_cond);
            pthread_mutex_unlock(&cr_ftb_ckpt_req_mutex);
            return (0);
        }
        --cr_ftb_ckpt_req;
        if (!cr_ftb_ckpt_req) {
            pthread_cond_signal(&cr_ftb_ckpt_req_cond);
            pthread_mutex_unlock(&cr_ftb_ckpt_req_mutex);
            return (0);
        }
        pthread_mutex_unlock(&cr_ftb_ckpt_req_mutex);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_RSRT_DONE))) {
        PRINT_DEBUG(DEBUG_CR_verbose, "a proc has been migrated/restarted...\n");
        return 0;
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_CKPT_FAIL))) {
        pthread_mutex_lock(&cr_ftb_ckpt_req_mutex);
        PRINT_ERROR("Got CR_FTB_CKPT_FAIL\n");
        cr_ftb_ckpt_req = -2;
        pthread_cond_signal(&cr_ftb_ckpt_req_cond);
        pthread_mutex_unlock(&cr_ftb_ckpt_req_mutex);
        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_APP_CKPT_REQ))) {
        cr_ftb_app_ckpt_req = 1;
        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_CKPT_FINALIZE))) {
        cr_ftb_finalize_ckpt = 1;
        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(FTB_MIGRATE_TRIGGER))) {
        if (sparehosts_on) {
            if (sparehosts_idx >= nsparehosts) {
                PRINT_ERROR("[Migration] Out of Spares\n");
                return (0);
            }
#ifdef CR_AGGRE
            if (use_aggre > 0 && use_aggre_mig > 0) {
                ret = cr_ftb_aggre_based_mig(revent->event_payload);
                if (ret != 0) {
                    PRINT_ERROR("Error: RDMA-based migration failed\n");
                }
                return 0;
            }
#endif
            snprintf(buf, MAX_CR_MSG_LEN, "%s %s", revent->event_payload, sparehosts[sparehosts_idx++]);

            PRINT_DEBUG(DEBUG_CR_verbose, "[Migration]: init a mig: \"%s\"\n", buf);
            SET_EVENT(eprop, FTB_EVENT_NORMAL, buf);
            ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_MIGRATE), &eprop, &ehandle);
            if (ret != FTB_SUCCESS) {
                PRINT_ERROR("FTB_Publish(CR_FTB_MIGRATE) failed with %d\n", ret);
            }
        } else {
            PRINT_ERROR("[Migration] No spare nodes (see the -sparehosts option of mpirun_rsh)\n");
        }

        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_MIGRATE_PIC))) {
        int isrc = -1, itgt = -1, i;

        /* Find Source & Target in the pglist */
        get_src_tgt(revent->event_payload, cr_mig_src_host, cr_mig_tgt_host);

        PRINT_DEBUG(DEBUG_MIG_verbose, " src_tgt payload=%s, src=%s:tgt=%s\n", revent->event_payload, cr_mig_src_host, cr_mig_tgt_host);

        for (i = 0; i < pglist->npgs; i++) {
            if (strcmp(pglist->data[i].hostname, cr_mig_src_host) == 0)
                isrc = i;
            if (strcmp(pglist->data[i].hostname, cr_mig_tgt_host) == 0)
                itgt = i;
        }
        ASSERT_MSG(isrc != -1, "source node '%s' not found\n", cr_mig_src_host);
        ASSERT_MSG(itgt != -1, "target node '%s' not found\n", cr_mig_tgt_host);

        /* Get the list of ranks */
        buf[0] = '\0';
        for (i = 0; i < pglist->data[isrc].npids; i++) {
            sprintf(cnum, "%d ", pglist->data[isrc].plist_indices[i]);
            strncat(buf, cnum, MAX_CR_MSG_LEN);
        }
        i = strlen(buf);
        if (buf[i - 1] == ' ')
            buf[i - 1] = '\0';

        PRINT_DEBUG(DEBUG_CR_verbose, "list of procs to migrate: %s\n", buf);
#ifdef SPAWN_DEBUG
        pglist_print();
#endif
        /* Fixup the pglist */
        // swap
        const char *src_hostname = pglist->data[isrc].hostname;
        const char *tgt_hostname = pglist->data[itgt].hostname;
        pid_t src_pid = pglist->data[isrc].pid;
        pid_t tgt_pid = pglist->data[itgt].pid;
        pid_t local_src = pglist->data[isrc].local_pid;
        pid_t local_tgt = pglist->data[itgt].local_pid;

        memcpy(&tmp_pg, &pglist->data[isrc], sizeof(process_group));
        memcpy(&pglist->data[isrc], &pglist->data[itgt], sizeof(process_group));
        memcpy(&pglist->data[itgt], &tmp_pg, sizeof(process_group));

        pglist->data[isrc].hostname = src_hostname;
        pglist->data[itgt].hostname = tgt_hostname;
        pglist->data[isrc].pid = src_pid;
        pglist->data[itgt].pid = tgt_pid;
        pglist->data[isrc].local_pid = local_src;
        pglist->data[itgt].local_pid = local_tgt;
        //I need to change also the plist_indice[itgt].hostname
        int index;
        for (index = 0; index < pglist->data[itgt].npids; index++) {
            plist[pglist->data[itgt].plist_indices[index]].hostname = (char *) strdup(tgt_hostname);
        }

        PRINT_DEBUG(DEBUG_MIG_verbose, "mpirun_rsh: will do migrate...\n");
#ifdef SPAWN_DEBUG
        pglist_print();
        dump_pgrps();
#endif
        /* Copy checkpointed image */
        char syscmd[256];
        char ckptdir[256], *tp;
        strncpy(ckptdir, ckpt_filename, 256);
        ckptdir[255] = 0;
        tp = ckptdir + strlen(ckptdir) - 1;
        while (*tp != '/' && tp >= ckptdir)
            tp--;
        if (tp >= ckptdir)
            *(tp + 1) = 0;
        sprintf(syscmd, "scp %s:%s.0* %s:%s", cr_mig_src_host, ckpt_filename, cr_mig_tgt_host, ckptdir);
        PRINT_DEBUG(DEBUG_MIG_verbose, "syscmd=%s\n", syscmd);
        system(syscmd);

        /* Initiate Phase II */
        PRINT_DEBUG(DEBUG_MIG_verbose, "move ckpt img complete...started phase II: send: \"%s\"\n", buf);
        SET_EVENT(eprop, FTB_EVENT_NORMAL, buf);
        ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_MIGRATE_PIIC), &eprop, &ehandle);
        if (ret != FTB_SUCCESS) {
            PRINT_ERROR("FTB_Publish(CR_FTB_MIGRATE_PIIC) failed with %d\n", ret);
        }

        return (0);
    }
    return 0;
}

int read_sparehosts(char *hostfile, char ***hostarr, int *nhosts)
{

    FILE *fp;
    char line[HOSTFILE_LEN + 1];
    int ret, line_len, n = 0;
    char **hosts;

    if (!hostfile || !hostarr || !nhosts) {
        PRINT_ERROR("Invalid parameters in read_sparehosts()\n");
        return (-1);
    }

    fp = fopen(hostfile, "r");
    if (!fp) {
        PRINT_ERROR_ERRNO("Error opening hostfile: fopen() failed:", errno);
        goto err_fopen;
    }
    
    /* Figure out the number of hosts */
    while (fgets(line, HOSTFILE_LEN, fp) != NULL) {

        line_len = strlen(line);
        if (line[line_len - 1] == '\n')
            line[line_len - 1] = '\0';

        line_len = strlen(line);
        if (line_len == 0)
            continue;           /* Blank Lines */
        if (line[0] == '#')
            continue;           /* Comments    */

        ++n;
    }

    *nhosts = n;

    hosts = (char **) malloc(n * sizeof(char *));
    if (!hosts) {
        PRINT_ERROR_ERRNO("Error allocating hosts array: malloc() failed", errno);
        goto err_malloc_hosts;
    }

    /* Reset File Pointer */
    rewind(fp);
    /* Store the list of hosts */
    n = 0;
    while (fgets(line, HOSTFILE_LEN, fp) != NULL) {

        line_len = strlen(line);
        if (line[line_len - 1] == '\n')
            line[line_len - 1] = '\0';

        line_len = strlen(line);
        if (line_len == 0)
            continue;           /* Blank Lines */
        if (line[0] == '#')
            continue;           /* Comments    */

        hosts[n] = (char *) malloc((line_len + 1) * sizeof(char));
        if (!hosts[n]) {
            PRINT_ERROR_ERRNO("Error allocating host[%d]: malloc() failed", errno, n);
            goto err_malloc_hostn;
        }

        strncpy(hosts[n], line, line_len + 1);
        ++n;
    }

    *hostarr = hosts;

    ret = 0;

  exit_malloc_hostn:
  exit_malloc_hosts:
    fclose(fp);

  exit_fopen:
    return (ret);

  err_malloc_hostn:
    while (n > 0) {
        --n;
        free(hosts[n]);
    }
    free(hosts);
    ret = -4;
    goto exit_malloc_hostn;

  err_malloc_hosts:
    ret = -3;
    goto exit_malloc_hosts;

  err_fopen:
    ret = -2;
    goto exit_fopen;
}

/* FIXME: Need to fix possible overrun flaw */
static int get_src_tgt(char *str, char *src, char *tgt)
{
    int i, j, tgt_start;

    if (!str || !src || !tgt)
        return (-1);

    i = j = tgt_start = 0;

    while (str[i]) {

        if (str[i] == ' ') {
            tgt_start = 1;
            src[j] = '\0';
            j = 0;
            ++i;
            continue;
        }

        if (tgt_start)
            tgt[j++] = str[i++];
        else
            src[j++] = str[i++];
    }

    tgt[j] = '\0';

    return (0);
}
#endif                          /* CR_FTB */

#endif                          /* CKPT */
