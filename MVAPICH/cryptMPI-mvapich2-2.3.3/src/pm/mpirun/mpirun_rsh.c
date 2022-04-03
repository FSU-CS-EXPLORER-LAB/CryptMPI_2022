
/*RAM
 * Copyright (C) 1999-2001 The Regents of the University of California
 * (through E.O. Lawrence Berkeley National Laboratory), subject to
 * approval by the U.S. Department of Energy.
 *
 * Use of this software is under license. The license agreement is included
 * in the file MVICH_LICENSE.TXT.
 *
 * Developed at Berkeley Lab as part of MVICH.
 *
 * Authors: Bill Saphir      <wcsaphir@lbl.gov>
 *          Michael Welcome  <mlwelcome@lbl.gov>
 */

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

#include <mpichconf.h>
#include <src/pm/mpirun/mpirun_rsh.h>
#include <mpispawn_tree.h>
#include <mpirun_util.h>
#include <mpmd.h>
#include <mpirun_dbg.h>
#include <mpirun_params.h>
#include <mpirun_ckpt.h>
#include <param.h>
#include <mv2_config.h>
#include <error_handling.h>
#include <debug_utils.h>
#include <signal_processor.h>
#include <wfe_mpirun.h>
#include <m_state.h>
#include <process.h>
#include <spawn_info.h>
#include <read_specfile.h>
#include <gethostip.h>
#include <mpirun_environ.h>

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <linux/limits.h>

#include <src/db/text.h>

/*
 * When an error occurs in the init phase, mpirun_rsh doesn't have the pid
 * of all the mpispawns and in the cleanup it doesn't kill the mpispawns. To
 * solve this problem, we need to wait that it has all the right pids before
 * cleaning up the mpispawn. These two variables wait_socks_succ and
 * socket_error are used to keep trace of this situation.
 */

int wait_socks_succ = 0;
int socket_error = 0;
pthread_mutex_t wait_socks_succ_lock = PTHREAD_MUTEX_INITIALIZER;

void spawn_one(int argc, char *argv[], char *totalview_cmd, char *env, int fastssh_nprocs_thres);

process_groups *pglist = NULL;
int port;
char *wd;                       /* working directory of current process */
#define MAX_HOST_LEN 256
#define DEFAULT_FAST_SSH_THRESHOLD 800
char mpirun_host[MAX_HOST_LEN + 1]; /* hostname of current process */
char mpirun_hostip[MAX_HOST_LEN + 1]; /* ip address of current process */
char mpirun_path[PATH_MAX] = { '\0' }; /* PATH of mpirun_rsh binary */

/* xxx need to add checking for string overflow, do this more carefully ... */

int NSPAWNS;
char *dpmenv;
int dpmenvlen;
int dpm_cnt = 0;
/*List of pids of the mpirun started by mpirun_rsh with dpm.*/
list_pid_mpirun_t *dpm_mpirun_pids = NULL;

#if defined(CKPT) && defined(CR_AGGRE)
int use_aggre = 1;              // by default we use CR-aggregation
int use_aggre_mig = 1;
#endif

/* #define dbg(fmt, args...)   do{ \
    fprintf(stderr,"%s: [mpirun]: "fmt, __func__, ##args); fflush(stderr); } while(0) */
#define dbg(fmt, args...)

/*struct spawn_info_t {
    int totspawns;
    int spawnsdone;
    int dpmtot;
    int dpmindex;
    int launch_num;
    char buf[MAXLINE];
    char linebuf[MAXLINE];
    char runbuf[MAXLINE];
    char argbuf[MAXLINE];
    char *spawnfile;
};
struct spawn_info_t spinf;
*/
spawn_info_t spinf;

/*
 * Message notifying user of what timed out
 */
static const char *alarm_msg = NULL;

void free_memory(void);
void pglist_print(void);
void pglist_insert(const char *const, const int);
void rkill_fast(void);
void rkill_linear(void);
void spawn_fast(int, char *[], char *, char *);
void cleanup_handler(int);
void sigtstp_handler(int);
void send_signal(char const sig[16]);
void remote_signal(char const host[256], char const signal[16],
        pid_t const pid[], size_t npids);
void alarm_handler(int);
void child_handler(int);
void cleanup(void);
char *skip_white(char *s);
int read_param_file(char *paramfile, char **env);
int set_fds(fd_set * rfds, fd_set * efds);
void make_command_strings(int argc, char *argv[], char *totalview_cmd, char *command_name, char *command_name_tv);
void launch_newmpirun(int total);
static void get_line(void *buf, char *fill, int buf_or_file);
void dpm_add_env(char *, char *);

/*
#define SH_NAME_LEN    (128)
char sh_cmd[SH_NAME_LEN];
    if (tmp) {
        free (mpispawn_env);
        mpispawn_env = tmp;
    }
    else
    {
        goto allocation_error;
    }
*/

int server_socket;

char *binary_name;

#define DISPLAY_STR_LEN 200
char display[DISPLAY_STR_LEN];

static void get_display_str()
{
    char *p;
    /* char str[DISPLAY_STR_LEN]; */

    p = getenv("DISPLAY");
    if (p != NULL) {
        /* For X11 programs */
        assert(DISPLAY_STR_LEN > strlen("DISPLAY=") + strlen(p));
        snprintf(display, DISPLAY_STR_LEN, "DISPLAY=%s", p);
    }
}

//Define the max len of a pid
#define MAX_PID_LEN 22

/* The name of the file where to write the host_list. Used when the number
 * of processes is beyond NPROCS_THRES */
char *host_list_file = NULL;

/* This function is called at exit and remove the temporary file with the
 * host_list information*/
void remove_host_list_file()
{
    if (host_list_file) {
        if (unlink(host_list_file) < 0) {
            fprintf(stderr, "ERROR removing the %s\n", host_list_file);
        }
        free(host_list_file);
        host_list_file = NULL;
    }
}

void dump_pgrps()
{
    int i;
    for (i = 0; i < pglist->npgs; i++) {
        dbg("pg_%d@%s: local_proc %d : rem_proc %d, has %d pids\n", i, pglist->data[i].hostname, pglist->data[i].local_pid, pglist->data[i].pid, pglist->data[i].npids);
    }
}

int lookup_exit_pid(pid_t pid)
{
    int rv = -1;
    if (pid < 0) {
        dbg("invalid pid %d\n", pid);
        return rv;
    }
    int i;
    for (i = 0; i < pglist->npgs; i++) {
        if (pid == pglist->data[i].local_pid) {
            dbg("proc exit: local_proc %d:%d for %s\n", i, pid, pglist->data[i].hostname);
            return i;
        }
    }
    dbg("exit pid %d cannot found\n", pid);
    return rv;
}

void init_debug()
{
    // Set coresize limit
    char *coresize = getenv("MV2_DEBUG_CORESIZE");
    set_coresize_limit(coresize);
    // ignore error code, failure if not fatal

    // Set prefix for debug output
    const int MAX_LENGTH = 256;
    char hostname[MAX_LENGTH];
    gethostname(hostname, MAX_LENGTH);
    hostname[MAX_LENGTH - 1] = '\0';
    char output_prefix[MAX_LENGTH];
    snprintf(output_prefix, MAX_LENGTH, "%s:mpirun_rsh", hostname);
    set_output_prefix(output_prefix);

    // Set an error signal handler
    char *bt = getenv("MV2_DEBUG_SHOW_BACKTRACE");
    int backtrace = 0;
    if (bt != NULL) {
        backtrace = ! !atoi(bt);
    }
    setup_error_sighandler(backtrace);
    // ignore error code, failure if not fatal

    // Initialize DEBUG variables
    initialize_debug_variables();
}

static void
signal_processor (int signal)
{
    switch (signal) {
        case SIGHUP:
        case SIGINT:
        case SIGTERM:
            PRINT_ERROR("Caught signal %d, killing job\n", signal);
            cleanup_handler(signal);
            break;
        case SIGTSTP:
            sigtstp_handler(signal);
            break;
        case SIGALRM:
            alarm_handler(signal);
            break;
        case SIGCHLD:
            child_handler(signal);
            break;
        default:
            PRINT_ERROR("Caught unexpected signal %d\n, killing job", signal);
            cleanup_handler(signal);
            break;
    }
}

void
setup_signal_handling_thread (void)
{
    sigset_t sigmask;

    sigemptyset(&sigmask);
    sigaddset(&sigmask, SIGHUP);
    sigaddset(&sigmask, SIGINT);
    sigaddset(&sigmask, SIGTERM);
    sigaddset(&sigmask, SIGTSTP);
    sigaddset(&sigmask, SIGALRM);
    sigaddset(&sigmask, SIGCHLD);

    start_sp_thread(sigmask, signal_processor, 0);
}

int main(int argc, char *argv[])
{
    int i, s;
    struct sockaddr_in sockaddr;
    unsigned int sockaddr_len = sizeof(sockaddr);
    unsigned long crc;

    char *env = "\0";

    char totalview_cmd[TOTALVIEW_CMD_LEN];

    int timeout, fastssh_threshold;
    struct wfe_params wfe_params;
    M_STATE state;

    if (read_configuration_files(&crc)) {
        fprintf(stderr, "mpirun_rsh: error reading configuration file\n");
        return EXIT_FAILURE;
    }

    init_debug();
    totalview_cmd[TOTALVIEW_CMD_LEN - 1] = '\0';
    display[0] = '\0';

    setup_signal_handling_thread();

#ifdef CKPT
    int ret_ckpt;

    ret_ckpt = CR_initialize();
    if ( ret_ckpt  < 0 ) {
        m_state_fail();
        goto exit_main;
    }
  restart_from_ckpt:
#endif                          /* CKPT */

    commandLine(argc, argv, totalview_cmd, &env);

#ifdef CKPT
    set_ckpt_nprocs(nprocs);
#endif

    if (use_totalview) {
        MPIR_proctable = (struct MPIR_PROCDESC *)
            malloc(MPIR_PROCDESC_s * nprocs);
        MPIR_proctable_size = nprocs;

        for (i = 0; i < nprocs; ++i) {
            MPIR_proctable[i].host_name = plist[i].hostname;
        }
    }

    wd = get_current_dir_name();
    gethostname(mpirun_host, sizeof(mpirun_host));
    gethostip(mpirun_hostip, sizeof(mpirun_hostip));

    get_display_str();

    server_socket = s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    sockaddr.sin_family = AF_INET;  
    sockaddr.sin_addr.s_addr = INADDR_ANY;
    sockaddr.sin_port = 0;
    timeout = env2int("MV2_MPIRUN_BIND_TIMEOUT");

    if (timeout <= 0) {
        timeout = 60;
    }

    alarm_msg = "Timeout during attempt to bind mpirun_rsh socket.\n";
    alarm(timeout);

    while (bind(s, (struct sockaddr *) &sockaddr, sockaddr_len)) {
        static int backoff = 1;
        struct timeval tv;

        tv.tv_sec = backoff;
        tv.tv_usec = 0;

        switch (errno) {
            case EADDRINUSE:
                /*
                 * I can't use sleep because that may interfere with SIGARLM
                 */
                select(0, NULL, NULL, NULL, &tv);
                backoff <<= 1;
                continue;
            default:
                perror("bind");
                exit(EXIT_FAILURE);
        }
    }

    alarm(0);

    if (getsockname(s, (struct sockaddr *) &sockaddr, &sockaddr_len) < 0) {
        perror("getsockname");
        exit(EXIT_FAILURE);
    }

    port = (int) ntohs(sockaddr.sin_port);

    listen(s, nprocs);

    for (i = 0; i < nprocs; i++) {
        /* I should probably do some sort of hostname lookup to account for
         * situations where people might use two different hostnames for the
         * same host.
         */
        pglist_insert(plist[i].hostname, i);
    }

#if defined(CKPT) && defined(CR_FTB)
    if (sparehosts_on) {
        /* Read Hot Spares */
        int ret = read_sparehosts(sparehostfile, &sparehosts, &nsparehosts);
        if (ret) {
            DBG(fprintf(stderr, "Error Reading Spare Hosts (%d)\n", ret));
            exit(EXIT_FAILURE);
        }
        /* Add Hot Spares to pglist */
        for (i = 0; i < nsparehosts; i++) {
            pglist_insert(sparehosts[i], -1);
            dbg("sparehosts[%d] = %s\n", i, sparehosts[i]);
        }
    }
#endif                          /* CR_FTB */

    /*
     * Set alarm for mpirun initialization on MV2_MPIRUN_TIMEOUT if set.
     * Otherwise set to default value based on size of job and whether
     * debugging is set.
     */
    timeout = env2int("MV2_MPIRUN_TIMEOUT");

    if (timeout <= 0) {
        timeout = pglist ? pglist->npgs : nprocs;
        if (timeout < 30) {
            timeout = 30;
        } else if (timeout > 1000) {
            timeout = 1000;
        }
    }

    alarm_msg = "Timeout during client startup.\n";
    if (!debug_on) {
        alarm(timeout);
    }

    fastssh_threshold = env2int("MV2_FASTSSH_THRESHOLD");

    if (!fastssh_threshold)
        fastssh_threshold = DEFAULT_FAST_SSH_THRESHOLD;

    //Another way to activate hiearachical ssh is having a number of nodes
    //beyond a threshold
    if (pglist->npgs >= fastssh_threshold) {
        USE_LINEAR_SSH = 0;
    }

    USE_LINEAR_SSH = dpm ? 1 : USE_LINEAR_SSH;

#ifdef CKPT

#ifdef CR_FTB
    /* force the use of linear startup if process migration is enabled */
    USE_LINEAR_SSH = 1;
#endif

    ret_ckpt = CR_thread_start(pglist->npgs);
    if ( ret_ckpt  < 0 ) {
        m_state_fail();
        goto exit_main;
    }

    /*
     * Check to see of ckpt variables are in the environment
     */
    save_ckpt_vars_env();
#ifdef CR_AGGRE
    if (getenv("MV2_CKPT_USE_AGGREGATION")) {
        use_aggre = atoi(getenv("MV2_CKPT_USE_AGGREGATION"));
    }
#endif
#endif


#if defined(CKPT) && defined(CR_FTB) 
    // Wait for CR thread to connect to FTB before proceeding
    // The CR thread will change the state to M_LAUNCH when ready
    // This should be removed once we remove the use of FTB for this
    state = m_state_wait_while(M_INITIALIZE|M_RESTART);
#else
    // We are ready to launch, make the transition
    state = m_state_transition(M_INITIALIZE|M_RESTART, M_LAUNCH);
#endif

    // If an error happened, skip launching and exit
    if (M_LAUNCH != state) {
        goto exit_main;
    }

    if (USE_LINEAR_SSH) {
        NSPAWNS = pglist->npgs;
        DBG(fprintf(stderr, "USE_LINEAR = %d \n", USE_LINEAR_SSH));

        spawn_fast(argc, argv, totalview_cmd, env);

    } else {
        NSPAWNS = 1;
        //Search if the number of processes is set up as a environment
        int fastssh_nprocs_thres = env2int("MV2_NPROCS_THRESHOLD");
        if (!fastssh_nprocs_thres)
            fastssh_nprocs_thres = 1 << 13;

        spawn_one(argc, argv, totalview_cmd, env, fastssh_nprocs_thres);
    }

    if (show_on)
        exit(EXIT_SUCCESS);

    /*
     * Create Communication Thread
     */
    wfe_params.s            = server_socket;
    wfe_params.sockaddr     = &sockaddr;
    wfe_params.sockaddr_len = sockaddr_len;

    start_wfe_thread(&wfe_params);

    state = m_state_wait_while(M_LAUNCH);

    /*
     * Disable alarm since mpirun is no longer launching.
     */
    alarm(0);
    dump_pgrps();

    if (M_RUN != state) {
        goto exit_main;
    }

    /*
     * Activate new alarm for mpi job based on MPIEXEC_TIMEOUT if set.
     */
    timeout = env2int("MPIEXEC_TIMEOUT");
    if (timeout > 0 && !debug_on) {
        alarm_msg = mkstr("Timeout [TIMEOUT = %d seconds].\n", timeout);
        alarm(timeout);
    }

    /*
     * Wait until a transition from the M_RUN to M_RESTART or M_EXIT occurs.
     */
    state = m_state_wait_until(M_RESTART|M_EXIT);

exit_main:
    stop_wfe_thread();
    stop_sp_thread();

    switch (state) {
        case M_RESTART:
#ifdef CKPT
            // Restart mpirun_rsh: cleanup and re-initialize
            PRINT_DEBUG( DEBUG_FT_verbose, "Restarting mpirun_rsh...\n" );

            // Cleanup
            CR_thread_stop(1);
            remove_host_list_file();
            free_memory();

            // Reset the parsing function
            optind = 1;

            // Restart
            setup_signal_handling_thread();
            goto restart_from_ckpt;
#else
            ASSERT_MSG( 0==1, "Internal Error\n");
#endif /* CKPT */
            break;
        case M_EXIT:
            break;
        default:
            /*
             * This should only happen due to a coding error.
             */
            PRINT_ERROR("invalid state");
            m_state_fail();
            break;
    }

#ifdef CKPT
    CR_thread_stop(0);
    CR_finalize();
#endif

    int exit_code = m_state_get_exit_code();
    int run_cleanup = (exit_code != EXIT_SUCCESS);

#if defined(CKPT) && defined(CR_FTB)
    // When migration is enabled, we need to run cleanup() if sparehosts has been specified.
    // Because, even if we have terminated properly, mpispawn processes still run on the spare hosts.
    run_cleanup = run_cleanup || sparehosts_on;
#endif

    if (run_cleanup) {
        cleanup();
    }

    remove_host_list_file();
    free_memory();

    return exit_code;
}

#if defined(CKPT) && defined(CR_AGGRE)
static void rkill_aggregation()
{
    int i;
    char cmd[256];
    if (!use_aggre)
        return;
    for (i = 0; i < NSPAWNS; i++) {
        dbg("before umnt for pg_%d @ %s...\n", i, pglist->index[i]->hostname);
        snprintf(cmd, 256, "%s %s fusermount -u /tmp/cr-%s/wa > /dev/null 2>&1", SSH_CMD, pglist->index[i]->hostname, sessionid);
        system(cmd);
        snprintf(cmd, 256, "%s %s fusermount -u /tmp/cr-%s/mig > /dev/null 2>&1", SSH_CMD, pglist->index[i]->hostname, sessionid);
        system(cmd);
        dbg("has finished umnt pg_%d @ %s\n", i, pglist->index[i]->hostname);
    }
}
#endif

void cleanup_handler(int sig)
{
    m_state_fail();
}

void pglist_print(void)
{
    if (pglist) {
        size_t i, j, npids = 0, npids_allocated = 0;

        fprintf(stderr, "\n--------------pglist-------------\ndata:\n");
        for (i = 0; i < pglist->npgs; i++) {
            fprintf(stderr, "%p - %s:", &pglist->data[i], pglist->data[i].hostname);
            fprintf(stderr, " %d (", pglist->data[i].pid);

            for (j = 0; j < pglist->data[i].npids; fprintf(stderr, ", "), j++) {
                fprintf(stderr, "%d", pglist->data[i].plist_indices[j]);
            }

            fprintf(stderr, ")\n");
            npids += pglist->data[i].npids;
            npids_allocated += pglist->data[i].npids_allocated;
        }

        fprintf(stderr, "\nindex:");
        for (i = 0; i < pglist->npgs; i++) {
            fprintf(stderr, " %p", pglist->index[i]);
        }

        fprintf(stderr, "\nnpgs/allocated: %d/%d (%d%%)\n", (int) pglist->npgs, (int) pglist->npgs_allocated, (int) (pglist->npgs_allocated ? 100. * pglist->npgs / pglist->npgs_allocated : 100.));
        fprintf(stderr, "npids/allocated: %d/%d (%d%%)\n", (int) npids, (int) npids_allocated, (int) (npids_allocated ? 100. * npids / npids_allocated : 100.));
        fprintf(stderr, "--pglist-end--\n\n");

        /// show all procs
        for (i = 0; i < nprocs; i++) {
            fprintf(stderr, "proc_%ld: at %s: pid=%d, remote_pid=%d\n", i, plist[i].hostname, plist[i].pid, plist[i].remote_pid);
        }
    }

}

int
pglist_cmp (char const * hostname, int index, spawn_info_t const * si)
{
    int result = strcmp(hostname, pglist->index[index]->hostname);


    if (!result) {
        if (si) {
            result = si - pglist->index[index]->si;
        }
    }

    return result;
}

void pglist_insert(const char *const hostname, const int plist_index)
{
    const size_t increment = nprocs > 4 ? nprocs / 4 : 1;
    size_t i, index = 0;
    static size_t alloc_error = 0;
    int result, bottom = 0, top;
    process_group *pg;
    void *backup_ptr;
    static int read_dpmfile = 1;
    static spawn_info_t * si = NULL;
    static int si_count = 0;

    /**
     * Only read the dpm file once in the beginning
     */
    if (read_dpmfile) {
        if (dpm) {
            si = read_dpm_specfile(spinf.totspawns);
        }
        read_dpmfile = 0;
    }

    if (si) {
        if (si_count++ == si->nprocs) {
            si = si->next;
            si_count = 1;
        }
    }

    if (alloc_error) {
        return;
    }

    if (pglist == NULL) {
        goto init_pglist;
    }

    top = pglist->npgs - 1;
    index = (top + bottom) / 2;

    while ((result = pglist_cmp(hostname, index, si))) {
        if (result > 0) {
            bottom = index + 1;
        }

        else {
            top = index - 1;
        }

        if (bottom > top) {
            break;
        }

        index = (top + bottom) / 2;
    }

    /*
     * We need to add another control (we need to understand if the exe is
     * different from the others inserted)
     */
    if (configfile_on && result == 0) {
        int index_previous;
        /*
         * Check if the previous name of exexutable and args in the pglist are
         * equal.  If they are different we need to add this exe to another
         * group.
         */

        index_previous = pglist->index[index]->plist_indices[0];
        result = strcmp(plist[plist_index].executable_name,
                plist[index_previous].executable_name);

        if (0 == result) {
            /*
             * Make sure both args are not NULL before comparing.
             */
            if (plist[plist_index].executable_args
                    && plist[index_previous].executable_args) {
                result = strcmp(plist[plist_index].executable_args,
                        plist[index_previous].executable_args);
            }

            /*
             * At least one is NULL, do a simple pointer comparison to see if
             * they are the same or not.
             */
            else {
                result = plist[plist_index].executable_args !=
                    plist[index_previous].executable_args;
            }
        }
    }

    if (!result)
        goto insert_pid;

    if (result > 0) {
        index++;
    }

    goto add_process_group;

init_pglist:
    pglist = malloc(sizeof(process_groups));

    if (pglist) {
        pglist->data = NULL;
        pglist->index = NULL;
        pglist->npgs = 0;
        pglist->npgs_allocated = 0;
    } else {
        goto register_alloc_error;
    }

add_process_group:
    if (pglist->npgs == pglist->npgs_allocated) {
        process_group *pglist_data_backup = pglist->data;
        ptrdiff_t offset;

        pglist->npgs_allocated += increment;

        backup_ptr = pglist->data;
        pglist->data = realloc(pglist->data, sizeof(process_group) *
                pglist->npgs_allocated);

        if (pglist->data == NULL) {
            pglist->data = backup_ptr;
            goto register_alloc_error;
        }

        backup_ptr = pglist->index;
        pglist->index = realloc(pglist->index, sizeof(process_group *) *
                pglist->npgs_allocated);

        if (pglist->index == NULL) {
            pglist->index = backup_ptr;
            goto register_alloc_error;
        }

        offset = (size_t) pglist->data - (size_t) pglist_data_backup;
        if (offset) {
            for (i = 0; i < pglist->npgs; i++) {
                pglist->index[i] = (process_group *) ((size_t) pglist->index[i]
                        + offset);
            }
        }
    }

    for (i = pglist->npgs; i > index; i--) {
        pglist->index[i] = pglist->index[i - 1];
    }

    pglist->data[pglist->npgs].hostname = hostname;
    pglist->data[pglist->npgs].pid = -1;
    pglist->data[pglist->npgs].plist_indices = NULL;
    pglist->data[pglist->npgs].npids = 0;
    pglist->data[pglist->npgs].npids_allocated = 0;
    pglist->data[pglist->npgs].si = si;

    pglist->index[index] = &pglist->data[pglist->npgs++];

insert_pid:
#if defined(CKPT) && defined(CR_FTB)
    /* This is a spare host. Create a PG but do not insert a PID */
    if (plist_index == -1)
        return;
#endif
    pg = pglist->index[index];

    if (pg->npids == pg->npids_allocated) {
        if (pg->npids_allocated) {
            pg->npids_allocated <<= 1;

            if (pg->npids_allocated < pg->npids)
                pg->npids_allocated = SIZE_MAX;
            if (pg->npids_allocated > (size_t) nprocs)
                pg->npids_allocated = nprocs;
        } else {
            pg->npids_allocated = 1;
        }

        backup_ptr = pg->plist_indices;
        pg->plist_indices = realloc(pg->plist_indices, pg->npids_allocated *
                sizeof(int));

        if (pg->plist_indices == NULL) {
            pg->plist_indices = backup_ptr;
            goto register_alloc_error;
        }
    }

    pg->plist_indices[pg->npids++] = plist_index;

    return;

register_alloc_error:
    if (pglist) {
        if (pglist->data) {
            for (pg = pglist->data; pglist->npgs--; pg++) {
                if (pg->plist_indices) {
                    free(pg->plist_indices);
                    pg->plist_indices = NULL;
                }
            }
            free(pglist->data);
            pglist->data = NULL;
        }

        if (pglist->index) {
            free(pglist->index);
            pglist->index = NULL;
        }

        free(pglist);
        pglist = NULL;
    }

    alloc_error = 1;
}

void free_memory(void)
{
    if (pglist) {
        if (pglist->data) {
            process_group *pg = pglist->data;

            while (pglist->npgs--) {
                if (pg->plist_indices) {
                    free(pg->plist_indices);
                    pg->plist_indices = NULL;
                }
                pg++;
            }

            free(pglist->data);
            pglist->data = NULL;
        }

        if (pglist->index) {
            free(pglist->index);
            pglist->index = NULL;
        }

        free(pglist);
        pglist = NULL;
    }

    if (plist) {
        free(plist);
        plist = NULL;
    }

    db_clear();
}

void cleanup(void)
{
    int i;

    if (use_totalview) {
        fprintf(stderr, "Cleaning up all processes ...\n");
        MPIR_debug_state = MPIR_DEBUG_ABORTING;
    }

    if (pglist) {
        dbg("will do rkill_fast...\n");
        rkill_fast();
    }

    else {
        for (i = 0; i < nprocs; i++) {
            if (RUNNING(i)) {
                /* send terminal interrupt, which will hopefully
                   propagate to the other side. (not sure what xterm will
                   do here.
                 */
                PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGINT to pid %d\n", plist[i].pid);
                int rv = kill(plist[i].pid, SIGINT);
                if (rv == 0) {
                    PRINT_DEBUG(DEBUG_Fork_verbose, "kill SIGINT to pid %d returned successfully\n", plist[i].pid);
                } else {
                    PRINT_ERROR_ERRNO("kill pid %d with SIGINT returned %d", errno, plist[i].pid, rv);
                }
            }
        }

        sleep(1);

        for (i = 0; i < nprocs; i++) {
            if (plist[i].state != P_NOTSTARTED) {
                /* send regular interrupt to rsh */
                PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGTERM to pid %d\n", plist[i].pid);
                int rv = kill(plist[i].pid, SIGTERM);
                if (rv == 0) {
                    PRINT_DEBUG(DEBUG_Fork_verbose, "kill SIGTERM to pid %d returned successfully\n", plist[i].pid);
                } else {
                    PRINT_ERROR_ERRNO("kill pid %d with SIGTERM returned %d", errno, plist[i].pid, rv);
                }
            }
        }

        sleep(1);

        for (i = 0; i < nprocs; i++) {
            if (plist[i].state != P_NOTSTARTED) {
                /* Kill the processes */
                PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGKILL to pid %d\n", plist[i].pid);
                int rv = kill(plist[i].pid, SIGKILL);
                if (rv == 0) {
                    PRINT_DEBUG(DEBUG_Fork_verbose, "kill SIGKILL to pid %d returned successfully\n", plist[i].pid);
                } else {
                    PRINT_ERROR_ERRNO("kill pid %d with SIGKILL returned %d", errno, plist[i].pid, rv);
                }
            }
        }

        rkill_linear();
    }
}

void remote_signal(char const host[256], char const signal[16],
        pid_t const pid[], size_t npids)
{
    char remote_command[1024];
    char * pid_start;
    pid_t const * next_pid = pid;

    if (use_rsh) {
        sprintf(remote_command, "%s %s kill -s %s", RSH_CMD, host, signal);
    } else {
        sprintf(remote_command, "%s %s -x %s kill -s %s", SSH_CMD, SSH_ARG,
                host, signal);
    }

    pid_start = remote_command + strlen(remote_command);

    while (npids--) {
        char pid_string[256];
        int pid_len;

        pid_len = sprintf(pid_string, " %lu", (unsigned long)*next_pid);

        if (1000 < pid_len + strlen(remote_command)) {
            strcat(remote_command, " >/dev/null 2>&1");

            PRINT_DEBUG(DEBUG_Fork_verbose, "%s", remote_command);
            system(remote_command);

            *pid_start = '\0';
        }

        strcat(remote_command, pid_string);
        next_pid++;
    }

    strcat(remote_command, " >/dev/null 2>&1");

    PRINT_DEBUG(DEBUG_Fork_verbose, "%s", remote_command);
    system(remote_command);
}

void send_signal(char const sig[16])
{
    int i, j;

    if (pglist) {
        for (i = 0; i < NSPAWNS; i++) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to signal a specific instance of mpispawn.  No exit
             * codes or state transitions should be called here.
             */
            if (pglist->index[i]->pid != -1) {
                const process_group *pg = pglist->index[i];

                if (legacy_startup) {
                    pid_t pids[pg->npids];

                    for (j = 0; j < pg->npids; j++) {
                        pids[j] = plist[pg->plist_indices[j]].remote_pid;
                    }

                    remote_signal(pg->hostname, sig, pids, pg->npids);
                } else {
                    remote_signal(pg->hostname, sig, &pg->pid, 1);
                }
            } 
        }
    } else {
        for (i = 0; i < nprocs; i++) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to signal a specific instance of mpispawn.  No exit
             * codes or state transitions should be called here.
             */
            if (plist[i].remote_pid) {
                remote_signal(plist[i].hostname, sig, &plist[i].remote_pid,
                        1);
            }
        }
    }
}

void rkill_fast(void)
{
    int tryagain, spawned_pid[pglist->npgs];
    size_t i, j;

    for (i = 0; i < NSPAWNS; i++) {
        if (0 == (spawned_pid[i] = fork())) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to kill a specific instance of mpispawn.  No exit codes
             * or state transitions should be called here.
             */
            if (pglist->index[i]->pid != -1) {
                const process_group *pg = pglist->index[i];

                if (legacy_startup) {
                    pid_t pids[pg->npids];

                    for (j = 0; j < pg->npids; j++) {
                        pids[j] = plist[pg->plist_indices[j]].remote_pid;
                    }

                    remote_signal(pg->hostname, "15", pids, pg->npids);
                } else {
                    remote_signal(pg->hostname, "15", &pg->pid, 1);
                }
            } 

            exit(EXIT_SUCCESS);
        }
    }
#if defined(CKPT) && defined(CR_AGGRE)
    rkill_aggregation();
#endif
    while (1) {
        static int iteration = 0;
        tryagain = 0;

        sleep(1 << iteration);

        for (i = 0; i < pglist->npgs; i++) {
            if (spawned_pid[i]) {
                int tmp = waitpid(spawned_pid[i], NULL, WNOHANG); 

                if (tmp == spawned_pid[i]) {
                    spawned_pid[i] = 0;
                }

                else {
                    tryagain = 1;
                }
            }
        }

        if (++iteration == 5 || !tryagain) {
            break;
        }
    }

    if (tryagain) {
        fprintf(stderr, "The following processes may have not been killed:\n");
        for (i = 0; i < pglist->npgs; i++) {
            if (spawned_pid[i]) {
                const process_group *pg = pglist->index[i];

                fprintf(stderr, "%s:", pg->hostname);

                for (j = 0; j < pg->npids; j++) {
                    fprintf(stderr, " %d", plist[pg->plist_indices[j]].remote_pid);
                }

                fprintf(stderr, "\n");
            }
        }
    }
}

void rkill_linear(void)
{
    int i, tryagain, spawned_pid[nprocs];

    fprintf(stderr, "Killing remote processes...");

    for (i = 0; i < nprocs; i++) {
        if (0 == (spawned_pid[i] = fork())) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to kill a specific instance of mpispawn.  No exit codes
             * or state transitions should be called here.
             */
            if (plist[i].remote_pid) {
                remote_signal(plist[i].hostname, "15", &plist[i].remote_pid, 1);
            }

            exit(EXIT_SUCCESS);
        }
    }

    while (1) {
        static int iteration = 0;
        tryagain = 0;

        sleep(1 << iteration);

        for (i = 0; i < nprocs; i++) {
            if (spawned_pid[i]) {
                if (!(spawned_pid[i] = waitpid(spawned_pid[i], NULL, WNOHANG))) {
                    tryagain = 1;
                }
            }
        }

        if (++iteration == 5 || !tryagain) {
            break;
        }
    }

    if (tryagain) {
        fprintf(stderr, "The following processes may have not been killed:\n");
        for (i = 0; i < nprocs; i++) {
            if (spawned_pid[i]) {
                fprintf(stderr, "%s [%d]\n", plist[i].hostname,
                        plist[i].remote_pid);
            }
        }
    }
}

int getpath(char *buf, int buf_len)
{
    char link[32];
    pid_t pid;
    unsigned len;
    pid = getpid();
    snprintf(&link[0], sizeof(link), "/proc/%i/exe", pid);

    if (mpirun_path[0] != '\0') {
        strcpy(buf, mpirun_path);
        return strlen(mpirun_path);
    }

    len = readlink(&link[0], buf, buf_len);
    if (len == -1) {
        buf[0] = 0;
        return 0;
    } else {
        buf[len] = 0;
        while (len && buf[--len] != '/') ;
        if (buf[len] == '/')
            buf[len] = 0;
        strcpy(mpirun_path, buf);
        return len;
    }
}

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/*
#define CHECK_ALLOC() do { \
    if (tmp) { \
        free (mpispawn_env); \
        mpispawn_env = tmp; \
    } \
    else goto allocation_error; \
} while (0);
*/
// Append "name=value" into env string.
// the old env-string is freed, the return value shall be assigned to the env-string
static inline char *append2env(char *env, char *name, char *value)
{
    if (!env || !name || !value) {
        fprintf(stderr, "%s: Invalid params:  env=%s, name=%s, val=%s\n", __func__, env, name, value);
        return NULL;
    }
    char *tmp = mkstr("%s %s=%s", env, name, value);
    free(env);
    return tmp;
}

/**
* Spawn the processes using a linear method.
*/
void spawn_fast(int argc, char *argv[], char *totalview_cmd, char *env)
{
    char *mpispawn_env, *tmp, *ld_library_path;
    char *name, *value;
    int i, tmp_i, getpath_status;
    char pathbuf[PATH_MAX];

    if ((ld_library_path = getenv("LD_LIBRARY_PATH"))) {
        mpispawn_env = mkstr("LD_LIBRARY_PATH=%s", ld_library_path);
    } else {
        mpispawn_env = mkstr("");
    }

    if (!mpispawn_env)
        goto allocation_error;

    /*
     * Forward mpirun parameters to mpispawn
     */
    mpispawn_env = append_mpirun_parameters(mpispawn_env);

    if (!mpispawn_env) {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_MPD=0", mpispawn_env);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s USE_LINEAR_SSH=%d", mpispawn_env, USE_LINEAR_SSH);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_HOST=%s", mpispawn_env, mpirun_host);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_HOSTIP=%s", mpispawn_env, mpirun_hostip);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPIRUN_RSH_LAUNCH=1", mpispawn_env);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_CHECKIN_PORT=%d", mpispawn_env, port);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_PORT=%d", mpispawn_env, port);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_NNODES=%d", mpispawn_env, pglist->npgs);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_GLOBAL_NPROCS=%d", mpispawn_env, nprocs);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_ID=%d", mpispawn_env, getpid());
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    if (use_totalview) {
        tmp = mkstr("%s MPISPAWN_USE_TOTALVIEW=1", mpispawn_env);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }
    }
#ifdef CKPT
    mpispawn_env = create_mpispawn_vars(mpispawn_env);

#endif                          /* CKPT */

    /*
     * mpirun_rsh allows env variables to be set on the commandline
     */
    if (!mpispawn_param_env) {
        mpispawn_param_env = mkstr("");
        if (!mpispawn_param_env)
            goto allocation_error;
    }

    while (aout_index != argc && strchr(argv[aout_index], '=')) {
        name = strdup(argv[aout_index++]);
        value = strchr(name, '=');
        value[0] = '\0';
        value++;
        dpm_add_env(name, value);

#ifdef CKPT
        save_ckpt_vars(name, value);
#ifdef CR_AGGRE
        if (strcmp(name, "MV2_CKPT_FILE") == 0) {
            mpispawn_env = append2env(mpispawn_env, name, value);
            if (!mpispawn_env)
                goto allocation_error;
        } else if (strcmp(name, "MV2_CKPT_USE_AGGREGATION") == 0) {
            use_aggre = atoi(value);
        } else if (strcmp(name, "MV2_CKPT_USE_AGGREGATION_MIGRATION") == 0) {
            use_aggre_mig = atoi(value);
        } else if (strcmp(name, "MV2_CKPT_AGGREGATION_BUFPOOL_SIZE") == 0) {
            mpispawn_env = append2env(mpispawn_env, name, value);
            if (!mpispawn_env)
                goto allocation_error;
        } else if (strcmp(name, "MV2_CKPT_AGGREGATION_CHUNK_SIZE") == 0) {
            mpispawn_env = append2env(mpispawn_env, name, value);
            if (!mpispawn_env)
                goto allocation_error;
        }
#endif
#endif                          /* CKPT */

        tmp = mkstr("%s MPISPAWN_GENERIC_NAME_%d=%s" " MPISPAWN_GENERIC_VALUE_%d=%s", mpispawn_param_env, param_count, name, param_count, value);

        free(name);
        free(mpispawn_param_env);

        if (tmp) {
            mpispawn_param_env = tmp;
            param_count++;
        }

        else {
            goto allocation_error;
        }
    }
#if defined(CKPT) && defined(CR_AGGRE)
    if (use_aggre > 0)
        mpispawn_env = append2env(mpispawn_env, "MV2_CKPT_USE_AGGREGATION", "1");
    else
        mpispawn_env = append2env(mpispawn_env, "MV2_CKPT_USE_AGGREGATION", "0");
    if (!mpispawn_env)
        goto allocation_error;
#endif
    if (!configfile_on) {
        if (!dpm && aout_index == argc) {
            fprintf(stderr, "Incorrect number of arguments.\n");
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    i = argc - aout_index;
    if (debug_on && !use_totalview)
        i++;

    if (dpm == 0 && !configfile_on) {
        tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, argc - aout_index);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }

    }

    i = 0;

    if (debug_on && !use_totalview) {
        tmp = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env, i++, DBG_CMD);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }
    }

    if (use_totalview) {
        int j;
        if (!configfile_on) {
            for (j = 0; j < MPIR_proctable_size; j++) {
                MPIR_proctable[j].executable_name = argv[aout_index];
            }
        } else {
            for (j = 0; j < MPIR_proctable_size; j++) {
                MPIR_proctable[j].executable_name = plist[j].executable_name;
            }

        }
    }

    tmp_i = i;

    /* to make sure all tasks generate same pg-id we send a kvs_template */
    srand(getpid());
    i = rand() % MAXLINE;
    tmp = mkstr("%s MPDMAN_KVS_TEMPLATE=kvs_%d_%s_%d", mpispawn_env, i, mpirun_host, getpid());
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    getpath_status = (getpath(pathbuf, PATH_MAX) && file_exists(pathbuf));

    dbg("%d forks to be done, with env:=  %s\n", pglist->npgs, mpispawn_env);

    for (i = 0; i < pglist->npgs; i++) {
        if (!(pglist->data[i].pid = fork())) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to launch a specific instance of mpispawn.  No exit codes
             * or state transitions should be called here.
             */
            size_t arg_offset = 0;
            const char *nargv[7];
            char const * command = NULL, * custpath = NULL, * custwd = NULL;
            int n = 0;
            arg_list * arg = dpm ? &pglist->data[i].si->command : NULL;

            tmp = mkstr("%s MPISPAWN_LOCAL_NPROCS=%d", mpispawn_env,
                    pglist->data[i].npids);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

#if defined(CKPT) && defined(CR_FTB) && defined(CR_AGGRE)
            if (use_aggre > 0 && use_aggre_mig > 0) {   // use Aggre-based migration
                if (pglist->data[i].npids > 0)  // will be src of mig
                    mpispawn_env = append2env(mpispawn_env, "MV2_CKPT_AGGRE_MIG_ROLE", "1");
                else            // will be target of mig
                    mpispawn_env = append2env(mpispawn_env, "MV2_CKPT_AGGRE_MIG_ROLE", "2");
                if (!mpispawn_env)
                    goto allocation_error;
            } else {            // not use Aggre-based migration
                mpispawn_env = append2env(mpispawn_env, "MV2_CKPT_AGGRE_MIG_ROLE", "0");
            }
#endif
            if (dpm) {
                tmp = mkstr("%s PMI_SPAWNED=1", mpispawn_env);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }

                tmp = mkstr("%s MPIRUN_COMM_MULTIPLE=%d", mpispawn_env,
                        spinf.totspawns != 1);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }

                if (pglist->data[i].si->wdir) {
                    custwd = pglist->data[i].si->wdir;
                }

                if (pglist->data[i].si->path) {
                    custpath = pglist->data[i].si->path;
                }

                if (pglist->data[i].si->port) {
                    tmp = mkstr("%s %s", mpispawn_env,
                            pglist->data[i].si->port);
                    if (tmp) {
                        free(mpispawn_env);
                        mpispawn_env = tmp;
                    } else {
                        goto allocation_error;
                    }
                }

                while (arg) {
                    tmp = mkstr("%s MPISPAWN_ARGV_%d='%s'", mpispawn_env, n,
                            arg->arg);
                    if (tmp) {
                        free(mpispawn_env);
                        mpispawn_env = tmp;
                    } else {
                        goto allocation_error;
                    }

                    n++;
                    arg = arg->next;
                }

                tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, n);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }

            //If the config option is activated, the executable information are taken from the pglist.
            else if (configfile_on) {
                int index_plist = pglist->data[i].plist_indices[0];
                /* When the config option is activated we need to put in the mpispawn the number of argument of the exe. */
                tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, plist[index_plist].argc);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
                /*Add the executable name and args in the list of arguments from the pglist. */
                tmp = add_argv(mpispawn_env, plist[index_plist].executable_name, plist[index_plist].executable_args, tmp_i);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }

            }

            else {
                while (aout_index < argc) {
                    tmp = mkstr("%s MPISPAWN_ARGV_%d='%s'", mpispawn_env,
                            tmp_i++, argv[aout_index++]);
                    if (tmp) {
                        free(mpispawn_env);
                        mpispawn_env = tmp;
                    } else {
                        goto allocation_error;
                    }
                }

                tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, tmp_i);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }

            if (mpispawn_param_env) {
                tmp = mkstr("%s MPISPAWN_GENERIC_ENV_COUNT=%d %s", mpispawn_env, param_count, mpispawn_param_env);

                free(mpispawn_param_env);
                free(mpispawn_env);

                if (tmp) {
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }

            tmp = mkstr("%s MPISPAWN_ID=%d", mpispawn_env, i);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            /* user may specify custom binary path via MPI_Info */
            if (custpath) {
                tmp = mkstr("%s MPISPAWN_BINARY_PATH=%s", mpispawn_env, custpath);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }

            /* user may specifiy custom working dir via MPI_Info */
            if (custwd) {
                tmp = mkstr("%s MPISPAWN_WORKING_DIR=%s", mpispawn_env, custwd);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            } else {
                tmp = mkstr("%s MPISPAWN_WORKING_DIR=%s", mpispawn_env, wd);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }

            }

            for (n = 0; n < pglist->data[i].npids; n++) {
                tmp = mkstr("%s MPISPAWN_MPIRUN_RANK_%d=%d", mpispawn_env, n, pglist->data[i].plist_indices[n]);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }

                if (plist[pglist->data[i].plist_indices[n]].device != NULL) {
                    tmp = mkstr("%s MPISPAWN_MV2_IBA_HCA_%d=%s", mpispawn_env, n, plist[pglist->data[i].plist_indices[n]].device);
                    if (tmp) {
                        free(mpispawn_env);
                        mpispawn_env = tmp;
                    } else {
                        goto allocation_error;
                    }

                }
            }

            int local_hostname = 0;
            if ((strcmp(pglist->data[i].hostname, mpirun_host) == 0) && (!xterm_on)) {
                local_hostname = 1;
                nargv[arg_offset++] = SHELL_CMD;
                nargv[arg_offset++] = SHELL_ARG;
            } else {

                if (xterm_on) {
                    nargv[arg_offset++] = XTERM_CMD;
                    nargv[arg_offset++] = "-e";
                }

                if (use_rsh) {
                    nargv[arg_offset++] = RSH_CMD;
                } else {
                    nargv[arg_offset++] = SSH_CMD;
                    nargv[arg_offset++] = SSH_ARG;
                }
            }
            if (getpath_status) {
                command = mkstr("cd %s; %s %s %s %s/mpispawn 0", wd, ENV_CMD, mpispawn_env, env, pathbuf);
            } else if (use_dirname) {
                command = mkstr("cd %s; %s %s %s %s/mpispawn 0", wd, ENV_CMD, mpispawn_env, env, binary_dirname);
            } else {
                command = mkstr("cd %s; %s %s %s mpispawn 0", wd, ENV_CMD, mpispawn_env, env);
            }

            /* If the user request an execution with an alternate group
               use the 'sg' command to run mpispawn */
            if (change_group != NULL) {
                command = mkstr("sg %s -c '%s'", change_group, command);
            }

            if (!command) {
                fprintf(stderr, "Couldn't allocate string for remote command!\n");
                exit(EXIT_FAILURE);
            }
            if (local_hostname == 0)
                nargv[arg_offset++] = pglist->data[i].hostname;

            nargv[arg_offset++] = command;
            nargv[arg_offset++] = NULL;

            if (show_on) {
                size_t arg = 0;
                fprintf(stdout, "\n");
                while (nargv[arg] != NULL)
                    fprintf(stdout, "%s ", nargv[arg++]);
                fprintf(stdout, "\n");

                exit(EXIT_SUCCESS);
            }

            if (strcmp(pglist->data[i].hostname, plist[0].hostname)) {
                int fd = open("/dev/null", O_RDWR, 0);
                dup2(fd, STDIN_FILENO);
            }
            /*
               int myti = 0;
               for(myti=0; myti<arg_offset; myti++)
               printf("%s: before exec-%d:  argv[%d] = %s\n", __func__, i, myti, nargv[myti] );
             */
            PRINT_DEBUG(DEBUG_Fork_verbose, "FORK mpispawn (pid=%d)\n", getpid());
            PRINT_DEBUG(DEBUG_Fork_verbose > 1, "mpispawn command line: %s\n", mpispawn_env);
            execv(nargv[0], (char *const *) nargv);
            perror("execv");

            for (i = 0; i < argc; i++) {
                fprintf(stderr, "%s ", nargv[i]);
            }

            fprintf(stderr, "\n");

            exit(EXIT_FAILURE);
        }
        pglist->data[i].local_pid = pglist->data[i].pid;
    }

    if (tmp) {
        free(tmp);
    }
    if (spawnfile) {
        unlink(spawnfile);
    }
    return;

  allocation_error:
    perror("spawn_fast");
    if (mpispawn_env) {
        fprintf(stderr, "%s\n", mpispawn_env);
        free(mpispawn_env);
    }

    exit(EXIT_FAILURE);
}

/*
 * Spawn the processes using the hierarchical way.
 */
void spawn_one(int argc, char *argv[], char *totalview_cmd, char *env, int fastssh_nprocs_thres)
{
    char *mpispawn_env, *tmp, *ld_library_path;
    char *name, *value;
    int i, j, k, n, tmp_i;
    int numBytes = 0, getpath_status;
    FILE *host_list_file_fp;
    char pathbuf[PATH_MAX];
    char *host_list = NULL;
    char *ptr;
    size_t max_npids = 0, record_sz;

    if ((ld_library_path = getenv("LD_LIBRARY_PATH"))) {
        mpispawn_env = mkstr("LD_LIBRARY_PATH=%s", ld_library_path);
    } else {
        mpispawn_env = mkstr("");
    }

    if (!mpispawn_env)
        goto allocation_error;

    /*
     * Forward mpirun parameters to mpispawn
     */
    mpispawn_env = append_mpirun_parameters(mpispawn_env);

    if (!mpispawn_env) {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_MPD=0", mpispawn_env);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s USE_LINEAR_SSH=%d", mpispawn_env, USE_LINEAR_SSH);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_HOST=%s", mpispawn_env, mpirun_host);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_HOSTIP=%s", mpispawn_env, mpirun_hostip);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPIRUN_RSH_LAUNCH=1", mpispawn_env);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_CHECKIN_PORT=%d", mpispawn_env, port);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_PORT=%d", mpispawn_env, port);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_GLOBAL_NPROCS=%d", mpispawn_env, nprocs);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    tmp = mkstr("%s MPISPAWN_MPIRUN_ID=%d", mpispawn_env, getpid());
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    if (!configfile_on) {
        for (k = 0; k < pglist->npgs; k++) {
            if (pglist->data[k].npids > max_npids)
                max_npids = pglist->data[k].npids;
        }
        record_sz = (MAX_HOST_LEN+2) + (max_npids * pglist->npgs)*MAX_PID_LEN;
        host_list = (char*)malloc(record_sz * sizeof(char));
        if (!host_list)
            goto allocation_error;
        ptr = host_list;
        for (k = 0; k < pglist->npgs; k++) {
            /* Make a list of hosts and the number of processes on each host */
            /* NOTE: RFCs do not allow : or ; in hostnames */
            ptr += sprintf(ptr, "%s:%zu:", pglist->data[k].hostname, pglist->data[k].npids);
            for (n = 0; n < pglist->data[k].npids; n++) {
                ptr += sprintf(ptr, "%d:", pglist->data[k].plist_indices[n]);
            }
        }
        *ptr = '\0';
    } else {
        /*In case of mpmd activated we need to pass to mpispawn the different names and arguments of executables */
        host_list = create_host_list_mpmd(pglist, plist);
        tmp = mkstr("%s MPISPAWN_MPMD=%d", mpispawn_env, 1);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }

    }

    //If we have a number of processes >= PROCS_THRES we use the file approach
    //Write the hostlist in a file and send the filename and the number of
    //bytes to the other procs
    if (nprocs >= fastssh_nprocs_thres) {

        int pathlen = strlen(wd);
        host_list_file = (char *) malloc(sizeof(char) * (pathlen + strlen("host_list_file.tmp") + MAX_PID_LEN + 1));
        sprintf(host_list_file, "%s/host_list_file%d.tmp", wd, getpid());

        /* open the host list file and write the host_list in it */
        DBG(fprintf(stderr, "OPEN FILE %s\n", host_list_file));
        host_list_file_fp = fopen(host_list_file, "w");

        if (host_list_file_fp == NULL) {
            fprintf(stderr, "host list temp file could not be created\n");
            goto allocation_error;
        }

        fprintf(host_list_file_fp, "%s", host_list);
        fclose(host_list_file_fp);

        tmp = mkstr("%s HOST_LIST_FILE=%s", mpispawn_env, host_list_file);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }

        numBytes = (strlen(host_list) + 1) * sizeof(char);
        DBG(fprintf(stderr, "WRITTEN %d bytes\n", numBytes));

        tmp = mkstr("%s HOST_LIST_NBYTES=%d", mpispawn_env, numBytes);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }

    } else {

        /*This is the standard approach used when the number of processes < PROCS_THRES */
        tmp = mkstr("%s MPISPAWN_HOSTLIST=%s", mpispawn_env, host_list);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }
    }

    tmp = mkstr("%s MPISPAWN_NNODES=%d", mpispawn_env, pglist->npgs);
    if (tmp) {
        free(mpispawn_env);
        mpispawn_env = tmp;
    } else {
        goto allocation_error;
    }

    if (use_totalview) {
        tmp = mkstr("%s MPISPAWN_USE_TOTALVIEW=1", mpispawn_env);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }
    }

    /*
     * mpirun_rsh allows env variables to be set on the commandline
     */
    if (!mpispawn_param_env) {
        mpispawn_param_env = mkstr("");
        if (!mpispawn_param_env)
            goto allocation_error;
    }

    while (aout_index != argc && strchr(argv[aout_index], '=')) {
        name = strdup(argv[aout_index++]);
        value = strchr(name, '=');
        value[0] = '\0';
        value++;

        tmp = mkstr("%s MPISPAWN_GENERIC_NAME_%d=%s" " MPISPAWN_GENERIC_VALUE_%d=%s", mpispawn_param_env, param_count, name, param_count, value);

        free(name);
        free(mpispawn_param_env);

        if (tmp) {
            mpispawn_param_env = tmp;
            param_count++;
        }

        else {
            goto allocation_error;
        }
    }

    if (!(configfile_on || dpm)) {
        if (aout_index == argc) {
            fprintf(stderr, "Incorrect number of arguments.\n");
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }

        i = argc - aout_index;
        if (debug_on && !use_totalview)
            i++;

        tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, argc - aout_index);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }

    }
    i = 0;

    if (debug_on && !use_totalview) {
        tmp = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env, i++, DBG_CMD);
        if (tmp) {
            free(mpispawn_env);
            mpispawn_env = tmp;
        } else {
            goto allocation_error;
        }
    }

    if (use_totalview) {
        int j;
        if (!configfile_on) {
            for (j = 0; j < MPIR_proctable_size; j++) {
                MPIR_proctable[j].executable_name = argv[aout_index];
            }
        } else {
            for (j = 0; j < MPIR_proctable_size; j++) {
                MPIR_proctable[j].executable_name = plist[j].executable_name;
            }
        }
    }

    tmp_i = i;

    i = 0;                      /* Spawn root mpispawn */
    {
        getpath_status = (getpath(pathbuf, PATH_MAX) && file_exists(pathbuf));

        if (!(pglist->data[i].pid = fork())) {
            /*
             * We're no longer the mpirun_rsh process but a child process
             * used to launch a specific instance of mpispawn.  No exit codes
             * or state transitions should be called here.
             */
            size_t arg_offset = 0;
            const char *nargv[7];
            char *command;
            //If the config option is activated, the executable information are taken from the pglist.
            if (configfile_on) {
                int index_plist = pglist->data[i].plist_indices[0];

                /* When the config option is activated we need to put in the mpispawn the number of argument of the exe. */
                tmp = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env, plist[index_plist].argc);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
                /*Add the executable name and args in the list of arguments from the pglist. */
                tmp = add_argv(mpispawn_env, plist[index_plist].executable_name, plist[index_plist].executable_args, tmp_i);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            } else {
                while (aout_index < argc) {
                    tmp = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env, tmp_i++, argv[aout_index++]);
                    if (tmp) {
                        free(mpispawn_env);
                        mpispawn_env = tmp;
                    } else {
                        goto allocation_error;
                    }

                }
            }

            if (mpispawn_param_env) {
                tmp = mkstr("%s MPISPAWN_GENERIC_ENV_COUNT=%d %s", mpispawn_env, param_count, mpispawn_param_env);

                free(mpispawn_param_env);
                free(mpispawn_env);

                if (tmp) {
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }

            tmp = mkstr("%s MPISPAWN_ID=%d", mpispawn_env, i);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            tmp = mkstr("%s MPISPAWN_LOCAL_NPROCS=%d", mpispawn_env, pglist->data[i].npids);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            tmp = mkstr("%s MPISPAWN_WORKING_DIR=%s", mpispawn_env, wd);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            if (xterm_on) {
                nargv[arg_offset++] = XTERM_CMD;
                nargv[arg_offset++] = "-e";
            }

            if (use_rsh) {
                nargv[arg_offset++] = RSH_CMD;
            }

            else {
                nargv[arg_offset++] = SSH_CMD;
                nargv[arg_offset++] = SSH_ARG;
            }
            tmp = mkstr("%s MPISPAWN_WD=%s", mpispawn_env, wd);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            for (j = 0; j < arg_offset; j++) {
                tmp = mkstr("%s MPISPAWN_NARGV_%d=%s", mpispawn_env, j, nargv[j]);
                if (tmp) {
                    free(mpispawn_env);
                    mpispawn_env = tmp;
                } else {
                    goto allocation_error;
                }
            }
            tmp = mkstr("%s MPISPAWN_NARGC=%d", mpispawn_env, arg_offset);
            if (tmp) {
                free(mpispawn_env);
                mpispawn_env = tmp;
            } else {
                goto allocation_error;
            }

            if (getpath_status) {
                command = mkstr("cd %s; %s %s %s %s/mpispawn %d", wd, ENV_CMD, mpispawn_env, env, pathbuf, pglist->npgs);
            } else if (use_dirname) {
                command = mkstr("cd %s; %s %s %s %s/mpispawn %d", wd, ENV_CMD, mpispawn_env, env, binary_dirname, pglist->npgs);
            } else {
                command = mkstr("cd %s; %s %s %s mpispawn %d", wd, ENV_CMD, mpispawn_env, env, pglist->npgs);
            }

            if (!command) {
                fprintf(stderr, "Couldn't allocate string for remote command!\n");
                exit(EXIT_FAILURE);
            }

            nargv[arg_offset++] = pglist->data[i].hostname;
            nargv[arg_offset++] = command;
            nargv[arg_offset++] = NULL;

            if (show_on) {
                size_t arg = 0;
                fprintf(stdout, "\n");
                while (nargv[arg] != NULL)
                    fprintf(stdout, "%s ", nargv[arg++]);
                fprintf(stdout, "\n");

                exit(EXIT_SUCCESS);
            }

            if (strcmp(pglist->data[i].hostname, plist[0].hostname)) {
                int fd = open("/dev/null", O_RDWR, 0);
                dup2(fd, STDIN_FILENO);
            }
            PRINT_DEBUG(DEBUG_Fork_verbose, "FORK mpispawn (pid=%d)\n", getpid());
            PRINT_DEBUG(DEBUG_Fork_verbose > 1, "mpispawn command line: %s\n", mpispawn_env);
            execv(nargv[0], (char *const *) nargv);
            perror("execv");

            for (i = 0; i < argc; i++) {
                fprintf(stderr, "%s ", nargv[i]);
            }

            fprintf(stderr, "\n");

            exit(EXIT_FAILURE);
        }
    }

    if (spawnfile) {
        unlink(spawnfile);
    }
    return;

  allocation_error:
    perror("spawn_one");
    if (mpispawn_env) {
        fprintf(stderr, "%s\n", mpispawn_env);
        free(mpispawn_env);
    }

    exit(EXIT_FAILURE);
}

/* #undef CHECK_ALLOC */

void make_command_strings(int argc, char *argv[], char *totalview_cmd, char *command_name, char *command_name_tv)
{
    int i;
    if (debug_on) {
        fprintf(stderr, "debug enabled !\n");
        char keyval_list[COMMAND_LEN];
        sprintf(keyval_list, "%s", " ");
        /* Take more env variables if present */
        while (strchr(argv[aout_index], '=')) {
            strcat(keyval_list, argv[aout_index]);
            strcat(keyval_list, " ");
            aout_index++;
        }
        if (use_totalview) {
            sprintf(command_name_tv, "%s %s %s", keyval_list, totalview_cmd, argv[aout_index]);
            sprintf(command_name, "%s %s ", keyval_list, argv[aout_index]);
        } else {
            sprintf(command_name, "%s %s %s", keyval_list, DBG_CMD, argv[aout_index]);
        }
    } else {
        sprintf(command_name, "%s", argv[aout_index]);
    }

    if (use_totalview) {
        /* Only needed for root */
        strcat(command_name_tv, " -a ");
    }

    /* add the arguments */
    for (i = aout_index + 1; i < argc; i++) {
        strcat(command_name, " ");
        strcat(command_name, argv[i]);
    }

    if (use_totalview) {
        /* Complete the command for non-root processes */
        strcat(command_name, " -mpichtv");

        /* Complete the command for root process */
        for (i = aout_index + 1; i < argc; i++) {
            strcat(command_name_tv, " ");
            strcat(command_name_tv, argv[i]);
        }
        strcat(command_name_tv, " -mpichtv");
    }
}

void sigtstp_handler(int signal)
{
    send_signal("SIGTSTP");
    raise(SIGSTOP);
    send_signal("SIGCONT");
}

void alarm_handler(int signal)
{
    extern const char *alarm_msg;

    if (use_totalview) {
        fprintf(stderr, "Timeout alarm signaled\n");
    }

    if (alarm_msg) {
        fprintf(stderr, "%s", alarm_msg);
    }

    m_state_fail();
}

void child_handler(int signal)
{
    int status, pid;
    static int count = 0;
    int is_dpm_signal = 0;

    int check_mpirun = 1;
    list_pid_mpirun_t *curr = dpm_mpirun_pids;
    list_pid_mpirun_t *prev = NULL;

    while (1) {
        dbg("pid count =%d num_exited %d\n", count, num_exited);

        // Retry while system call is interrupted
        do {
            pid = waitpid(-1, &status, WNOHANG);
        } while (pid == -1 && errno == EINTR);

        is_dpm_signal = 0;

        // Debug output
        PRINT_DEBUG(DEBUG_Fork_verbose, "waitpid return pid = %d\n", pid);
        if (pid >= 0) {
            if (WIFEXITED(status)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "process %d exited with status %d\n", pid, WEXITSTATUS(status));
            } else if (WIFSIGNALED(status)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "process %d terminated with signal %d\n", pid, WTERMSIG(status));
            } else if (WIFSTOPPED(status)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "process %d stopped with signal %d\n", pid, WSTOPSIG(status));
            } else if (WIFCONTINUED(status)) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "process %d continued\n", pid);
            }
        }

        if (pid < 0) {
            if (errno == ECHILD) {
                // No more unwaited-for child -> end mpirun_rsh
                if (legacy_startup) {
                    close(server_socket);
                }

                /*
                 * Tell main thread to exit mpirun_rsh cleanly.
                 */
                m_state_exit();

                return;
            } else {
                // Unhandled cases -> error
                PRINT_ERROR_ERRNO("Error: waitpid returned %d", errno, pid);

                /*
                 * Tell main thread to abort mpirun_rsh.
                 */
                m_state_fail();

                return;
            }
        } else if (pid == 0) {
            // No more exited child -> end handler
            return;
        }

        count++;
        /*With DPM if a SIGCHLD is sent by a child mpirun the mpirun
         *root must not end. It must continue its run and terminate only
         *when a signal is sent by a different child.*/
        if (check_mpirun) {
            while (curr != NULL) {
                if (pid == curr->pid) {
                    is_dpm_signal = 1;
                    if (prev == NULL)
                        dpm_mpirun_pids = (list_pid_mpirun_t *) curr->next;
                    else
                        prev->next = (list_pid_mpirun_t *) curr->next;
                    free(curr);
                    // Try with next exited child
                    continue;
                }
                prev = curr;
                curr = curr->next;
            }
        }

        /*If this is the root mpirun and the signal was sent by another child (not mpirun)
         *it can wait the termination of the mpirun childs too.*/
        if (dpm_mpirun_pids != NULL && curr == NULL)
            check_mpirun = 0;
        dbg("wait pid =%d count =%d, errno=%d\n", pid, count, errno);

#ifdef CKPT
        static int num_exited = 0;
        if (lookup_exit_pid(pid) >= 0)
            num_exited++;
        // TODO update
        dbg(" pid =%d count =%d, num_exited=%d,nspawn=%d, ckptcnt=%d, wait_socks_succ=%d\n", pid, count, num_exited, NSPAWNS, checkpoint_count, wait_socks_succ);
        // If number of active nodes have exited, ensure all 
        // other spare nodes are killed too.

#ifdef CR_FTB
        dbg("nsparehosts=%d\n", nsparehosts);
        if (sparehosts_on && (num_exited >= (NSPAWNS - nsparehosts))) {
            dbg(" num_exit=%d, NSPAWNS=%d, nsparehosts=%d, will kill-fast\n",
                    num_exited, NSPAWNS, nsparehosts);

            /*
             * Tell main thread to exit mpirun_rsh cleanly.
             */
            m_state_exit();

            continue;
        }
#endif

        if (num_exited < NSPAWNS) {
            // Try with next exited child
            continue;
        }
#endif
        if (!is_dpm_signal && (!WIFEXITED(status) || WEXITSTATUS(status))) {
            /*
             * mpirun_rsh did not successfully accept the connections of each
             * mpispawn before one of the processes terminated abnormally or
             * with a bad exit status.
             */
            pthread_mutex_lock(&wait_socks_succ_lock);
            if (wait_socks_succ < NSPAWNS) {
                PRINT_ERROR("Error in init phase, aborting! "
                            "(%d/%d mpispawn connections)\n", wait_socks_succ, NSPAWNS);
                alarm_msg = "Failed in initilization phase, cleaned up all the mpispawn!\n";
                socket_error = 1;
            }
            pthread_mutex_unlock(&wait_socks_succ_lock);

            /*
             * Tell main thread to abort mpirun_rsh.
             */
            m_state_fail();

            continue;
        }
    }
    dbg("mpirun_rsh EXIT FROM child_handler\n");
}

void mpispawn_checkin(int s)
{
    int sock, id, i, n, mpispawn_root = -1;
    socklen_t addrlen;
    struct sockaddr_storage addr;
    int mt_degree;
    struct {
        char hostname[MAX_HOST_LEN + 1];
        char port[MAX_PORT_LEN + 1];
    } mpispawn_info[NSPAWNS];

    mt_degree = env2int("MV2_MT_DEGREE");
    if (!mt_degree) {
        mt_degree = MT_DEFAULT_DEGREE;
    }

    else if (mt_degree < MT_MIN_DEGREE) {
        /*
         * Shouldn't we detect this error much earlier in this process?
         */
        PRINT_ERROR("MV2_MT_DEGREE must be >= %d\n", MT_MIN_DEGREE);
        m_state_fail();

        return;
    }

    else if (mt_degree > MT_MAX_DEGREE) {
        /*
         * Shouldn't we detect this error much earlier in this process?
         */
        PRINT_ERROR("MV2_MT_DEGREE must be <= %d\n", MT_MAX_DEGREE);
        m_state_fail();

        return;
    }

#if defined(CKPT) && defined(CR_FTB)
    mt_degree = MT_MAX_DEGREE;
    spawninfo = (struct spawn_info_s *) malloc(NSPAWNS * sizeof(struct spawn_info_s));
    if (!spawninfo) {
        PRINT_ERROR_ERRNO("malloc() failed", errno);
        m_state_fail();

        return;
    }
#endif

    for (i = 0; i < NSPAWNS; i++) {
        addrlen = sizeof(addr);

        while ((sock = accept(s, (struct sockaddr *) &addr, &addrlen)) < 0) {
            if (errno == EINTR || errno == EAGAIN)
                continue;
            socket_error = 1;
        }

        if (read_socket(sock, &id, sizeof(int))
            || read_socket(sock, &pglist->data[id].pid, sizeof(pid_t))
            || read_socket(sock, &mpispawn_info[id].port, MAX_PORT_LEN + 1)) {
            socket_error = 1;
        }

        if (send_environ(sock)) {
            socket_error = 1;
        }

        if (!(id == 0 && use_totalview))
            close(sock);
        else
            mpispawn_root = sock;

        for (n = 0; n < pglist->data[id].npids; n++) {
            plist[pglist->data[id].plist_indices[n]].state = P_STARTED;
        }

        pthread_mutex_lock(&wait_socks_succ_lock);
        wait_socks_succ++;
        pthread_mutex_unlock(&wait_socks_succ_lock);

#if defined(CKPT) && defined(CR_FTB)
        strncpy(spawninfo[i].spawnhost, pglist->data[i].hostname, 31);
        spawninfo[i].sparenode = (pglist->data[i].npids == 0) ? 1 : 0;
#endif
    }

    /*
     *  If there was some errors in the socket mpirun_rsh can cleanup all the
     * mpispawns.    
     */
    if (socket_error) {
        PRINT_ERROR( "Error on the socket\n" );
        m_state_fail();

        return;
    }

    if (USE_LINEAR_SSH) {
        for (i = 0; i < NSPAWNS; i++) {
            strncpy(mpispawn_info[i].hostname, pglist->data[i].hostname,
                    MAX_HOST_LEN);
            mpispawn_info[i].hostname[MAX_HOST_LEN] = '\0';
        }

        sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock < 0) {
            PRINT_ERROR_ERRNO("socket() failed", errno);
            m_state_fail();

            return;
        }

        sock = connect_socket(mpispawn_info[0].hostname, mpispawn_info[0].port);
        if (0 == sock) {
            m_state_fail();
            return;
        }

        /*
         * Send mpispawn_info array to mpispawn_info[0] (mpispawn with id 0).
         * The mpispawn processes will propagate this information to each other
         * after connecting in a tree like structure.
         */
        if (write_socket(sock, &mpispawn_info, sizeof(mpispawn_info))
#if defined(CKPT) && defined(CR_FTB)
            || write_socket(sock, spawninfo, sizeof(struct spawn_info_s) * (pglist->npgs))
#endif
            ) {
            m_state_fail();

            return;
        }

        close(sock);
    }
    if (use_totalview) {
        int id, j;
        process_info_t *pinfo = (process_info_t *) malloc(process_info_s * nprocs);
        read_socket(mpispawn_root, pinfo, process_info_s * nprocs);
        for (j = 0; j < nprocs; j++) {
            MPIR_proctable[pinfo[j].rank].pid = pinfo[j].pid;
        }
        free(pinfo);
        /* We're ready for totalview */
        MPIR_debug_state = MPIR_DEBUG_SPAWNED;
        MPIR_Breakpoint();

        /* MPI processes can proceed now */
        id = 0;
        write_socket(mpispawn_root, &id, sizeof(int));
        if (USE_LINEAR_SSH)
            close(mpispawn_root);
    }
}

int check_token(FILE * fp, char *tmpbuf)
{
    char *tokptr, *val, *key;
    if (strncmp(tmpbuf, "preput_val_", 11) == 0 || strncmp(tmpbuf, "info_val_", 9) == 0) {
        /* info_val_ */
        tokptr = strtok(tmpbuf, "=");
        if (!tokptr) {
            return -1;
        }

        key = strtok(NULL, "=");
        if (!key) {
            return -1;
        }

        fprintf(fp, "%s\n", key);
    } else if (strncmp(tmpbuf, "execname", 8) == 0 || strncmp(tmpbuf, "preput_key_", 11) == 0 || strncmp(tmpbuf, "info_key_", 9) == 0) {
        /* execname, preput.. */
        /* info_key_ */
        tokptr = strtok(tmpbuf, "=");
        if (!tokptr) {
            return -1;
        }

        val = strtok(NULL, "=");
        if (!val) {
            return -1;
        }

        fprintf(fp, "%s\n", val);
    } else if (strncmp(tmpbuf, "nprocs", 6) == 0) {
        tokptr = strtok(tmpbuf, "=");
        if (!tokptr) {
            return -1;
        }

        val = strtok(NULL, "=");
        if (!val) {
            return -1;
        }

        fprintf(fp, "%s\n", val);
        spinf.launch_num = atoi(val);
    } else if (strncmp(tmpbuf, "totspawns", 9) == 0) {
        /* totspawns */
        tokptr = strtok(tmpbuf, "=");
        if (!tokptr) {
            return -1;
        }

        spinf.totspawns = atoi(strtok(NULL, "="));
    } else if (strncmp(tmpbuf, "endcmd", 6) == 0) {
        /* endcmd */
        fprintf(fp, "endcmd\n");
        spinf.spawnsdone = spinf.spawnsdone + 1;
        if (spinf.spawnsdone >= spinf.totspawns) {
            return 1;
        }
    } else if (strncmp(tmpbuf, "info_num", 8) == 0) {
        /* info num */
        fprintf(fp, "%s\n", tmpbuf);
    } else if (strncmp(tmpbuf, "argcnt", 6) == 0) {
        /* args end */
        fprintf(fp, "endarg\n");
    } else if (strncmp(tmpbuf, "arg", 3) == 0) {
        /* arguments */
        fprintf(fp, "%s\n", tmpbuf);
    }

    return 0;
}

int handle_spawn_req(int readsock)
{
    FILE *fp;
    int done = 0;
    char *chptr, *hdptr;
    uint32_t size, spcnt;
    static int totalprocs = 0;
    static char *TOTALPROCS;

    memset(&spinf, 0, sizeof(spawn_info_t));

    sprintf(spinf.linebuf, TMP_PFX "%d_%d_spawn_spec.file", getpid(), dpm_cnt);
    spinf.spawnfile = strdup(spinf.linebuf);

    fp = fopen(spinf.linebuf, "w");
    if (NULL == fp) {
        fprintf(stderr, "temp file could not be created\n");
    }

    read_socket(readsock, &spcnt, sizeof(uint32_t));
    read_socket(readsock, &size, sizeof(uint32_t));
    hdptr = chptr = malloc(size);
    read_socket(readsock, chptr, size);

    do {
        get_line(chptr, spinf.linebuf, 0);
        chptr = chptr + strlen(spinf.linebuf) + 1;
        done = check_token(fp, spinf.linebuf);
    }
    while (!done);

    fsync(fileno(fp));
    fclose(fp);
    free(hdptr);

    /*
     * Check to see if there was an error returned by check_token
     */
    if (0 > done) {
        return 1;
    }

    if (totalprocs == 0) {
        totalprocs = nprocs;
    }

    TOTALPROCS = mkstr("TOTALPROCS=%d", totalprocs);
    putenv(TOTALPROCS);
    totalprocs = totalprocs + spcnt;

    launch_newmpirun(spcnt);
    return 0;
}

static void get_line(void *ptr, char *fill, int is_file)
{
    int i = 0, ch;
    FILE *fp;
    char *buf;

    if (is_file) {
        fp = ptr;
        while ((ch = fgetc(fp)) != '\n') {
            fill[i] = ch;
            ++i;
        }
    } else {
        buf = ptr;
        while (buf[i] != '\n') {
            fill[i] = buf[i];
            ++i;
        }
    }
    fill[i] = '\0';
}

void launch_newmpirun(int total)
{
    FILE *fp;
    int i, j;
    char *newbuf;

    if (!hostfile_on) {
        sprintf(spinf.linebuf, TMP_PFX "%d_hostlist.file", getpid());
        fp = fopen(spinf.linebuf, "w");
        for (j = 0, i = 0; i < total; i++) {
            fprintf(fp, "%s\n", plist[j].hostname);
            j = (j + 1) % nprocs;
        }
        fclose(fp);
    } else {
        strncpy(spinf.linebuf, hostfile, MAXLINE);
    }

    sprintf(spinf.buf, "%d", total);

    /*It needs to maintain the list of mpirun pids to handle the termination. */
    list_pid_mpirun_t *curr;
    curr = (list_pid_mpirun_t *) malloc(sizeof(list_pid_mpirun_t));
    curr->next = (list_pid_mpirun_t *) dpm_mpirun_pids;
    dpm_mpirun_pids = curr;
    if ((curr->pid = fork()))
        return;

    newbuf = (char *) malloc(PATH_MAX + MAXLINE);
    if (use_dirname) {
        strcpy(newbuf, binary_dirname);
        strcat(newbuf, "/mpirun_rsh");
    } else {
        getpath(newbuf, PATH_MAX);
        strcat(newbuf, "/mpirun_rsh");
    }
    DBG(fprintf(stderr, "launching %s\n", newbuf));
    DBG(fprintf(stderr, "numhosts = %s, hostfile = %s, spawnfile = %s\n", spinf.buf, spinf.linebuf, spinf.spawnfile));
    char nspawns[MAXLINE];
    sprintf(nspawns, "%d", spinf.totspawns);
    PRINT_DEBUG(DEBUG_Fork_verbose, "FORK new mpirun (pid=%d)\n", getpid());
    PRINT_DEBUG(DEBUG_Fork_verbose > 1, "new mpirun command line: %s\n", newbuf);
    if (dpmenv) {
        execl(newbuf, newbuf, "-np", spinf.buf, "-hostfile", spinf.linebuf, "-spawnfile", spinf.spawnfile, "-dpmspawn", nspawns, "-dpm", dpmenv, NULL);
    } else {
        execl(newbuf, newbuf, "-np", spinf.buf, "-hostfile", spinf.linebuf, "-spawnfile", spinf.spawnfile, "-dpmspawn", nspawns, "-dpm", NULL);
    }

    perror("execl failed\n");
    exit(EXIT_FAILURE);
}

#define ENVLEN 20480
void dpm_add_env(char *buf, char *optval)
{
    if (dpmenv == NULL) {
        dpmenv = (char *) malloc(ENVLEN);
        dpmenv[0] = '\0';
        dpmenvlen = ENVLEN;
    }

    if (optval == NULL) {
        strcat(dpmenv, buf);
        strcat(dpmenv, " ");
    } else {
        strcat(dpmenv, buf);
        strcat(dpmenv, "=");
        strcat(dpmenv, optval);
        strcat(dpmenv, " ");
    }
}

#undef ENVLEN

/* vi:set sw=4 sts=4 tw=80 expandtab: */
