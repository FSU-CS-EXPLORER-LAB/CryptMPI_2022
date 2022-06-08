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
#include <mpirun_util.h>
#include <mpispawn_tree.h>
#include <pmi_tree.h>
#include <mpmd.h>
#include <error_handling.h>
#include <debug_utils.h>
#include <crfs.h>
#include <mpispawn_ckpt.h>
#include <signal_processor.h>
#include <mpispawn_error_codes.h>
#include <gethostip.h>
#include <mpirun_environ.h>

#include <signal.h>
#include <stdio.h>
#include <sys/socket.h>
#include <netdb.h>
#include <errno.h>
#include <pthread.h>
#include <wait.h>
#include <string.h>
#include <math.h>
#include <sys/select.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/time.h>

#define DBG(_stmt_)

// Static variables
static int USE_LINEAR_SSH;
static int checkin_sock;
static int **ranks;
static pid_t *mpispawn_pids;
static size_t npids = 0;

// Used in with pmi_tree.c
int NCHILD;
int N;
int *mpispawn_fds;
int mpirun_socket;
process_info_t *local_processes;

// Used in with pmi_tree.c and mpispawn_tree.c
int MPISPAWN_HAS_PARENT;

// Used in with pmi_tree.c, mpispawn_tree.c and mpispawn_ckpt.c
int MPISPAWN_NCHILD;
child_t *children;

// Used in pmi_tree.c, opt.c and debug.h
int mt_id;

// Used in mpispawn_ckpt.c
int c_socket;
in_port_t c_port;
struct sockaddr_in c_sockaddr;
unsigned int sockaddr_len = sizeof(c_sockaddr);

// Shouldn't this be defined by unistd.h?
extern char **environ;

#define ENV_CMD         "/usr/bin/env"
#define MAX_HOST_LEN 256

// Remove/update debugging statement that use my_hostname variable

/* #define dbg(fmt, args...)   do{ \
    fprintf(stderr, "%s: [spawn_%d]: "fmt, __func__, mt_id, ##args );fflush(stderr);} while(0) */
#define dbg(fmt, args...)

// Keep track of MPISPAWN state
typedef enum {
    MPISPAWN_STATE_UNDEF,                   // Undefined
    MPISPAWN_STATE_INITIALIZING,            // Initializing MPISPAWN
    MPISPAWN_STATE_STARTING,                // Starting MPI processes
    MPISPAWN_STATE_RUNNING,                 // Running MPI processes, waiting for end
    MPISPAWN_STATE_FINALIZING,              // Cleaning the state before exiting

} mpispawn_state_t;

// Normal state changes are:
// INITIALIZING -> STARTING -> RUNNING -> FINALIZING

static mpispawn_state_t mpispawn_state = MPISPAWN_STATE_UNDEF;



void process_cleanup(void);
#ifdef CKPT
void cr_cleanup(void);
#endif

void report_error(int abort_code)
{
    PRINT_DEBUG(DEBUG_Fork_verbose, "Report error %d to mpirun_rsh\n", abort_code);
    int sock, id = env2int("MPISPAWN_ID");
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int connect_attempt = 0, max_connect_attempts = 5;
    struct sockaddr_in sockaddr;
    struct hostent *mpirun_hostent;
    if (sock < 0) {
        /* Oops! */
        PRINT_ERROR_ERRNO("socket() failed", errno);
        exit(EXIT_FAILURE);
    }

    mpirun_hostent = gethostbyname(env2str("MPISPAWN_MPIRUN_HOST"));
    if (NULL == mpirun_hostent) {
        mpirun_hostent = gethostbyname(env2str("MPISPAWN_MPIRUN_HOSTIP"));
        if (NULL == mpirun_hostent) {
            /* Oops! */
            PRINT_ERROR("gethostbyname() failed: %s (%d)\n",
                    hstrerror(h_errno), h_errno);
            exit(EXIT_FAILURE);
        }
    }

    sockaddr.sin_family = AF_INET;
    sockaddr.sin_addr = *(struct in_addr *) (*mpirun_hostent->h_addr_list);
    sockaddr.sin_port = htons(env2int("MPISPAWN_CHECKIN_PORT"));

    while (connect(sock, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0) {
        if (++connect_attempt > max_connect_attempts) {
            PRINT_ERROR_ERRNO("connect() failed", errno);
            exit(EXIT_FAILURE);
        }
    }
    if (sock) {
        write_socket(sock, &abort_code, sizeof(int));
        write_socket(sock, &id, sizeof(int));
        close(sock);
    }
}

void mpispawn_abort(int abort_code)
{
    PRINT_DEBUG(DEBUG_Fork_verbose, "MPISPAWN ABORT with code %d\n", abort_code);
    mpispawn_state = MPISPAWN_STATE_FINALIZING;
#ifdef CKPT
    cr_cleanup();
#endif
    process_cleanup();
    report_error(abort_code);
    PRINT_DEBUG(DEBUG_Fork_verbose, "exit(EXIT_FAILURE)\n");
    exit(EXIT_FAILURE);
}

static inline void get_lvalues(int index) {
    char buffer[32], param_value[32];
    sprintf(buffer, "MPISPAWN_MPIRUN_RANK_%d", index);

    if (USE_LINEAR_SSH) {
        local_processes[index].rank = env2int(buffer);
    } else {
        local_processes[index].rank = ranks[mt_id][index];
        sprintf(param_value, "%d", local_processes[index].rank);
        setenv(buffer, param_value, 1);
    }
}

static inline int setup_global_environment()
{
    char my_host_name[MAX_HOST_LEN + MAX_PORT_LEN];
    char tmp[MAX_HOST_LEN + 1];

    int i = env2int("MPISPAWN_GENERIC_ENV_COUNT");

    setenv("MPIRUN_MPD", "0", 1);
    setenv("MPIRUN_NPROCS", getenv("MPISPAWN_GLOBAL_NPROCS"), 1);
    setenv("MPIRUN_ID", getenv("MPISPAWN_MPIRUN_ID"), 1);
    setenv("MV2_NUM_NODES_IN_JOB", getenv("MPISPAWN_NNODES"), 1);
    setenv("MV2_NODE_ID", getenv("MPISPAWN_ID"), 1);

    /* Ranks now connect to mpispawn */
    int rv = gethostname(tmp, MAX_HOST_LEN);
    tmp[MAX_HOST_LEN] = '\0';

    if ( rv == -1 ) {
        PRINT_ERROR_ERRNO("gethostname() failed", errno);
        return -1;
    }

    sprintf(my_host_name, "%s:%d", tmp, c_port);

    setenv("PMI_PORT", my_host_name, 2);

    if (env2int("MPISPAWN_USE_TOTALVIEW")) {
        setenv("USE_TOTALVIEW", "1", 1);
    } else {
        setenv("USE_TOTALVIEW", "0", 1);
    }

    while (i--) {
        char *buffer, *name, *value;

        buffer = mkstr("MPISPAWN_GENERIC_NAME_%d", i);
        if (!buffer) {
            fprintf(stderr, "%s:%d Insufficient memory\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        name = env2str(buffer);
        if (!name) {
            fprintf(stderr, "%s:%d Insufficient memory\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        free(buffer);

        buffer = mkstr("MPISPAWN_GENERIC_VALUE_%d", i);
        if (!buffer) {
            fprintf(stderr, "%s:%d Insufficient memory\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        value = env2str(buffer);
        if (!value) {
            fprintf(stderr, "%s:%d Insufficient memory\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
#ifdef CKPT
#ifndef CR_AGGRE
        if (strcmp(name, "MV2_CKPT_FILE") == 0)
            strncpy(ckpt_filename, value, CR_MAX_FILENAME);
#endif
#endif                          /* CKPT */

        setenv(name, value, 1);

        free(name);
        free(value);
    }
    return 0;
}

static inline void setup_local_environment(int index)
{
    char buffer[32];
     
    sprintf(buffer,"%d", local_processes[index].rank);
    setenv("PMI_ID", buffer, 1);
    setenv("MV2_COMM_WORLD_RANK", buffer, 1);

    sprintf(buffer,"%d", index);
    setenv("MV2_COMM_WORLD_LOCAL_RANK", buffer, 1);

    sprintf(buffer,"%d", NCHILD);
    setenv("MV2_COMM_WORLD_LOCAL_SIZE", buffer, 1);

    sprintf(buffer,"%d", N);
    setenv("MV2_COMM_WORLD_SIZE", buffer, 1);

#ifdef CKPT
    setenv("MV2_CKPT_FILE", ckpt_filename, 1);
    setenv("MV2_CKPT_SESSIONID", sessionid, 1);

    /* Setup MV2_CKPT_MPD_BASE_PORT for legacy reasons */
    setenv("MV2_CKPT_MPD_BASE_PORT", "0", 1);
#ifdef CR_AGGRE
    if (use_aggre && use_aggre_mig) {
        setenv("MV2_CKPT_AGGRE_MIG_FILE", crfs_mig_filename, 1);
    }
    dbg("========  ckpt-file=%s, mig-file= %s\n", ckpt_filename, crfs_mig_filename);
#endif
#endif

}

int initialize_checkin_sock (void)
{
    struct sockaddr_in checkin_sockaddr;

    checkin_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (checkin_sock < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    checkin_sockaddr.sin_family = AF_INET;
    checkin_sockaddr.sin_addr.s_addr = INADDR_ANY;
    checkin_sockaddr.sin_port = 0;

    if (bind(checkin_sock, (struct sockaddr *) &checkin_sockaddr, sockaddr_len) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    if (getsockname(checkin_sock, (struct sockaddr *) &checkin_sockaddr, &sockaddr_len) < 0) {
        perror("getsockname");
        exit(EXIT_FAILURE);
    }

    listen(checkin_sock, 64);

    return (int) ntohs(checkin_sockaddr.sin_port);
}

/*Obtain the host_ist from a file. This function is used when the number of
 * processes is beyond the threshold. */
char *obtain_host_list_from_file()
{
    //Obtain id of the host file and number of byte to read
    //Number of bytes sent when it is used the file approach to exachange
    //the host_list
    int num_bytes;
    FILE *fp;
    char *host_list_file = NULL, *host_list = NULL;

    host_list_file = env2str("HOST_LIST_FILE");
    num_bytes = env2int("HOST_LIST_NBYTES");

    fp = fopen(host_list_file, "r");
    if (fp == NULL) {

        fprintf(stderr, "host list temp file could not be read\n");
    }

    host_list = malloc(num_bytes);
    fscanf(fp, "%s", host_list);
    fclose(fp);
    return host_list;
}

void find_children (int id, int degree, int size, int child[degree])
{
    size_t c_start = degree * id + 1;
    size_t i;

    if (degree < 1) {
        return;
    }

    for (i = 0; i < degree; i++) {
        child[i] = -1;
    }

    if (c_start < size) {
        i = (degree > size - c_start) ? size - c_start : degree;

        while (i--) {
            child[i] = c_start + i;
        }
    }
}

void spawn_mpispawn_tree (int argc, char * argv[], int mt_nnodes, int
        mt_degree)
{
    int port = initialize_checkin_sock();
    int nargc = env2int("MPISPAWN_NARGC");
    int i, j, k, *np;
    char *mpmd_on = env2str("MPISPAWN_MPMD");
    char *host_list = env2str("MPISPAWN_HOSTLIST");
    char *command = mkstr("cd %s; %s", env2str("MPISPAWN_WD"), ENV_CMD);
    char *mpispawn_env = NULL;
    char hostname[MAX_HOST_LEN + 1];
    char hostnameip[MAX_HOST_LEN + 1];
    char *nargv[7], buf[20], *args, **host;
    int target[mt_degree];

    for (i = 0; i < nargc; i++) {
        sprintf(buf, "MPISPAWN_NARGV_%d", i);
        nargv[i] = env2str(buf);
    }

    /*
     * If the number of processes is beyond or equal the PROCS_THRES it
     * receives the host list in a file
     */
    if (host_list == NULL) {
        host_list = obtain_host_list_from_file();
    }

    gethostname(hostname, sizeof(hostname));
    gethostip(hostnameip, sizeof(hostnameip));

    mpispawn_env = mkstr("MPISPAWN_MPIRUN_HOST=%s"
            " MPISPAWN_MPIRUN_HOSTIP=%s"
            " MPISPAWN_CHECKIN_PORT=%d"
            " MPISPAWN_MPIRUN_PORT=%d",
            hostname, hostnameip, port, port);

    for (i = 0; environ[i] != NULL; i++) {
        char *var, *val;
        char *dup = strdup(environ[i]);
        var = strtok(dup, "=");
        val = var + strlen(var) + 1;
        if (val &&
                0 != strcmp(var, "MPISPAWN_ID") &&
                0 != strcmp(var, "MPISPAWN_LOCAL_NPROCS") &&
                0 != strcmp(var, "MPISPAWN_MPIRUN_HOST") &&
                0 != strcmp(var, "MPISPAWN_CHECKIN_PORT") &&
                0 != strcmp(var, "MPISPAWN_MPIRUN_PORT")) {

            if (strchr(val, '(') != NULL || strchr(val, ')') != NULL) {
                /* Ignore variables with ( or ), see Trac #841 */
            } else if (strchr(val, ' ') != NULL) {
                /* Add Quotes to variables with space */
                mpispawn_env = mkstr("%s %s='%s'", mpispawn_env, var, val);

            } else {
                /*If mpmd is selected the name and args of the executable are written in the HOST_LIST, not in the
                 * MPISPAWN_ARGV and MPISPAWN_ARGC. So the value of these varibles is not exact and we don't
                 * read this value.*/
                if (mpmd_on) {
                    if (strstr(var, "MPISPAWN_ARGV_") == NULL && strstr(var, "MPISPAWN_ARGC") == NULL) {

                        mpispawn_env = mkstr("%s %s=%s", mpispawn_env, var, val);
                    }
                } else
                    mpispawn_env = mkstr("%s %s=%s", mpispawn_env, var, val);
            }
        }

        free(dup);
    }

    args = mkstr("%s", argv[0]);
    for (i = 1; i < argc - 1; i++) {
        args = mkstr("%s %s", args, argv[i]);
    }
    nargv[nargc + 2] = NULL;

    host = (char **) malloc(mt_nnodes * sizeof(char *));
    np = (int *) malloc(mt_nnodes * sizeof(int));
    ranks = (int **) malloc(mt_nnodes * sizeof(int *));

    /*
     * These three variables are used to collect information on name, args
     * and number of args in case of mpmd
     */
    char **exe = (char **) malloc(mt_nnodes * sizeof(char *));
    char **args_exe = (char **) malloc(mt_nnodes * sizeof(char *));
    int *num_args = (int *) malloc(mt_nnodes * sizeof(int));

    for (i = mt_nnodes, j = 0; i > 0; i--, j++) {
        if (i == mt_nnodes) {
            host[j] = strtok(host_list, ":");
        }

        else {
            host[j] = strtok(NULL, ":");
        }

        np[j] = atoi(strtok(NULL, ":"));
        ranks[j] = (int *) malloc(np[j] * sizeof(int));

        for (k = 0; k < np[j]; k++) {
            ranks[j][k] = atoi(strtok(NULL, ":"));
        }

        /* 
         * If mpmd is selected the executable name and the arguments are
         * written in the hostlist.  So we need to read these information
         * from the hostlist.
         */
        if (mpmd_on) {
            exe[j] = strtok(NULL, ":");
            num_args[j] = atoi(strtok(NULL, ":"));

            if (num_args[j] > 1) {
                char *arg_tmp = NULL;
                args_exe[j] = NULL;
                k = 0;
                while (k < num_args[j] - 1) {
                    if (k == 0) {
                        arg_tmp = strtok(NULL, ":");
                    }

                    else {
                        arg_tmp = mkstr("%s:%s", arg_tmp, strtok(NULL, ":"));
                    }

                    k++;
                }
                if(arg_tmp)
                    args_exe[j] = strdup(arg_tmp);
            }
        }
    }

    find_children(mt_id, mt_degree, mt_nnodes, target);

    /*
     * Launch mpispawns
     */
    for (i = 0; i < mt_degree && target[i] != -1; i++) {
        /*
         * If mpmd is selected we need to add the MPISPAWN_ARGC and
         * MPISPAWN_ARGV to the mpispwan environment using the information
         * we have read in the host_list.
         */
        if (mpmd_on) {
            /*
             * We need to add MPISPAWN_ARGV
             */
            mpispawn_env = mkstr("%s MPISPAWN_ARGC=%d", mpispawn_env,
                    num_args[target[i]]);
            mpispawn_env = mkstr("%s MPISPAWN_ARGV_0=%s", mpispawn_env,
                    exe[target[i]]);
            if(num_args[target[i]] && args_exe[target[i]]) {
                char **tmp_arg = tokenize(args_exe[target[i]], ":");
                for (j = 0; j < num_args[target[i]] - 1; j++) {
                    mpispawn_env = mkstr("%s MPISPAWN_ARGV_%d=%s", mpispawn_env,
                            j + 1, tmp_arg[j]);
                }
            }
        }

        nargv[nargc] = host[target[i]];
        MPISPAWN_NCHILD++;

        if (0 == fork()) {
            mpispawn_env = mkstr("%s MPISPAWN_ID=%d MPISPAWN_LOCAL_NPROCS=%d",
                    mpispawn_env, target[i], np[target[i]]);
            command = mkstr("%s %s %s %d", command, mpispawn_env, args,
                    mt_nnodes);

            nargv[nargc + 1] = command;
            PRINT_DEBUG(DEBUG_Fork_verbose, "FORK mpispawn (pid=%d)\n",
                    getpid());
            PRINT_DEBUG(DEBUG_Fork_verbose > 1,
                    "EXEC mpispawn command line: %s %s\n", argv[0], argv[1]);

            execv(nargv[0], (char *const *) nargv);
            perror("execv");
        }
    }
}

void spawn_processes(int n)
{
    char my_host_name[MAX_HOST_LEN + MAX_PORT_LEN];
    gethostname(my_host_name, MAX_HOST_LEN);
    int i;
    npids = n;
    local_processes = (process_info_t *) malloc(process_info_s * n);

    if (!local_processes) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
#ifdef CKPT
    int cached_cr_mig_tgt = 0;
#ifdef CR_FTB
    cached_cr_mig_tgt = cr_mig_tgt;
    cr_mig_tgt = 0;
#endif
#endif
    for (i = 0; i < n; i++) {
        get_lvalues(i);
    }

    for (i = 0; i < n; i++) {
        local_processes[i].pid = fork();
        if (local_processes[i].pid == 0) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "FORK MPI proc (pid=%d)\n", getpid());

#ifdef CKPT
            restart_mpi_process(cached_cr_mig_tgt, i);
            // This never returns (call exec())
#endif

            int argc, nwritten;
            char **argv, buffer[80];

            setup_local_environment(i);

            argc = env2int("MPISPAWN_ARGC");

            argv = malloc(sizeof(char *) * (argc + 1));
            if (!argv) {
                fprintf(stderr, "%s:%d Insufficient memory\n", __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

            argv[argc] = NULL;

            while (argc--) {
                nwritten = snprintf(buffer, 80, "MPISPAWN_ARGV_%d", argc);
                if (nwritten < 0 || nwritten > 80) {
                    fprintf(stderr, "%s:%d Overflow\n", __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }

                /* if executable is not in working directory */
                if (argc == 0 && getenv("MPISPAWN_BINARY_PATH")) {
                    char *tmp = env2str(buffer);
                    if (tmp[0] != '/') {
                        snprintf(buffer, 80, "%s/%s", getenv("MPISPAWN_BINARY_PATH"), tmp);
                    }
                    free(tmp);
                    argv[argc] = strdup(buffer);
                } else
                    argv[argc] = env2str(buffer);
            }

            /*Check if the executable is in the working directory */
            char *tmp_argv = strdup(argv[0]);
            if (tmp_argv[0] != '.' && tmp_argv[0] != '/') {
                char *tmp = malloc(sizeof(char *) * (strlen(argv[0]) + 2));;
                sprintf(tmp, "%s%s", "./", argv[0]);
                if (access(tmp, F_OK) == 0)
                    argv[0] = strdup(tmp);

                free(tmp);
            }
            free(tmp_argv);

            PRINT_DEBUG(DEBUG_Fork_verbose > 1, "EXEC MPI proc command line: %s %s\n", argv[0], argv[1]);
            execvp(argv[0], argv);

            PRINT_ERROR_ERRNO("Failed to execvp() '%s'", errno, argv[0]);
            PRINT_DEBUG(DEBUG_Fork_verbose, "exit(EXIT_FAILURE)\n");
            exit(EXIT_FAILURE);

        }
    }
}

void process_cleanup(void)
{
    // Run process cleanup only once
    static OPA_int_t process_cleanup_started = {0};
    static int process_cleanup_complete = 0;
    int started = OPA_fetch_and_add_int( &process_cleanup_started, 1 );

    /*
     * Do not return until process cleanup is done running
     */
    if (started) {
	while (!process_cleanup_complete);
	return;
    }

    PRINT_DEBUG(DEBUG_FT_verbose, "Cleanup stray processes\n");
    int i;
    for (i = 0; i < npids; i++) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGINT to pid %ld\n", local_processes[i].pid);
        int rv = kill(local_processes[i].pid, SIGINT);
        if (rv == 0) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %ld with SIGINT returned successfully\n", local_processes[i].pid);
        } else {
//             PRINT_ERROR_ERRNO("kill pid %ld with SIGINT returned %d", errno, local_processes[i].pid, rv);
        }
    }
    if (!USE_LINEAR_SSH)
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGINT to pid %d\n", mpispawn_pids[i]);
            int rv = kill(mpispawn_pids[i], SIGINT);
            if (rv == 0) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %d with SIGINT returned successfully\n", mpispawn_pids[i]);
            } else {
//             PRINT_ERROR_ERRNO("kill pid %d with SIGINT returned %d", errno, mpispawn_pids[i], rv);
            }
        }

    sleep(1);

    for (i = 0; i < npids; i++) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGTERM to pid %ld\n", local_processes[i].pid);
        int rv = kill(local_processes[i].pid, SIGTERM);
        if (rv == 0) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %ld with SIGTERM returned successfully\n", local_processes[i].pid);
        } else {
//             PRINT_ERROR_ERRNO("kill pid %ld with SIGTERM returned %d", errno, local_processes[i].pid, rv);
        }
    }
    if (!USE_LINEAR_SSH)
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGTERM to pid %d\n", mpispawn_pids[i]);
            int rv = kill(mpispawn_pids[i], SIGTERM);
            if (rv == 0) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %d with SIGTERM returned successfully\n", mpispawn_pids[i]);
            } else {
//             PRINT_ERROR_ERRNO("kill pid %d with SIGTERM returned %d", errno, mpispawn_pids[i], rv);
            }
        }

    sleep(1);

    for (i = 0; i < npids; i++) {
        kill(local_processes[i].pid, SIGKILL);
        PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGKILL to pid %ld\n", local_processes[i].pid);
        int rv = kill(local_processes[i].pid, SIGKILL);
        if (rv == 0) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %ld with SIGKILL returned successfully\n", local_processes[i].pid);
        } else {
//             PRINT_ERROR_ERRNO("kill pid %ld with SIGKILL returned %d", errno, local_processes[i].pid, rv);
        }
    }
    if (!USE_LINEAR_SSH)
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            PRINT_DEBUG(DEBUG_Fork_verbose, "send SIGKILL to pid %d\n", mpispawn_pids[i]);
            int rv = kill(mpispawn_pids[i], SIGKILL);
            if (rv == 0) {
                PRINT_DEBUG(DEBUG_Fork_verbose, "kill pid %d with SIGKILL returned successfully\n", mpispawn_pids[i]);
            } else {
//             PRINT_ERROR_ERRNO("kill pid %d with SIGKILL returned %d", errno, mpispawn_pids[i], rv);
            }
        }

    free(local_processes);
    local_processes = NULL;
    free(children);
    children = NULL;
    process_cleanup_complete = 1;
}

void local_signal(int sig, pid_t const pid[], size_t npids)
{
    size_t i;

    for (i = 0; i < npids; i++) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "kill(%d, %d)\n", pid[i], sig);
        kill(pid[i], sig);
    }
}

void send_signal(int sig)
{
    pid_t pids[npids];
    int i;

    for (i = 0; i < npids; i++) {
        pids[i] = local_processes[i].pid;
    }

    local_signal(sig, pids, npids);
}

void sigtstp_handler(int signal)
{
    send_signal(SIGSTOP);
    raise(SIGSTOP);
    send_signal(SIGCONT);
}

void cleanup_handler(int sig)
{
    mpispawn_abort(MPISPAWN_CLEANUP_SIGNAL);
}

void child_handler(int signal)
{
    static int num_exited = 0;
    int status, pid, rank, i;
    char my_host_name[MAX_HOST_LEN];
    gethostname(my_host_name, MAX_HOST_LEN);

    rank = mt_id;
    PRINT_DEBUG(DEBUG_Fork_verbose, "mpispawn child_handler: got signal %d: %s\n", signal, sys_siglist[signal]);
    while (1) {
        do {
            pid = waitpid(-1, &status, WNOHANG);
        } while (pid == -1 && errno == EINTR);

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

        if (pid == 0) {
            // No more exited child -> end handler
            return;
        } else if (pid < 0) {
            if (errno == ECHILD) {
#if defined(CKPT) && defined(CR_FTB)
                if ( cr_mig_src ) {
                    // If I am a migration source:
                    // - change state (so I won't exit)
                    // - do not exit
                    // I will exit when I will receive a cleanup signal
                    mpispawn_state = MPISPAWN_STATE_FINALIZING;
                    return;
                }
#endif
                // No more unwaited-for child
                if ( mpispawn_state == MPISPAWN_STATE_RUNNING ) {
                    // If we are in running state (ie the MPI processes have been started), then exit properly
                    mpispawn_state = MPISPAWN_STATE_FINALIZING;
#ifdef CKPT
                    cr_cleanup();
#endif
                    PRINT_DEBUG(DEBUG_Fork_verbose, "_exit(EXIT_SUCCESS)\n");
                    _exit(EXIT_SUCCESS);
                } else {
                    return;
                }
            } else {
                // Unhandled cases -> error
                PRINT_ERROR_ERRNO("waitpid returned %d", errno, pid);
                abort();
            }
        } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            if (++num_exited == npids) {
                mpispawn_state = MPISPAWN_STATE_FINALIZING;
#ifdef CKPT
#ifdef CR_FTB
                FTB_event_properties_t eprop;
                FTB_event_handle_t ehandle;
                if (cr_mig_src) {
                    PRINT_DEBUG(DEBUG_Fork_verbose, "All migrated children has terminated: migration is over on source node\n");
#ifdef CR_AGGRE
                    if (use_aggre && use_aggre_mig) {
                        // I'm src in aggregation-based migration
                        cr_mig_src_can_exit = 1;
                        return;
                    }
#endif
                    //cr_mig_src = 0;
                    snprintf(my_host_name, MAX_HOST_LEN, "%s %s", cr_mig_src_host, cr_mig_tgt_host);
                    SET_EVENT(eprop, FTB_EVENT_NORMAL, my_host_name);
                    dbg(" at %s: Sending out CR_FTB_MIGRATE_PIC\n", cr_mig_src_host);
                    ///////////
                    status = FTB_Publish(ftb_handle, EVENT(CR_FTB_MIGRATE_PIC), &eprop, &ehandle);
                    cr_mig_src_can_exit = 1;
                    ///////////////////////
                    return;
                }
#endif
                cr_cleanup();
#endif
                PRINT_DEBUG(DEBUG_Fork_verbose, "_exit(EXIT_SUCCESS)\n");
                _exit(EXIT_SUCCESS);
            }
        } else {
            rank = -1;
            gethostname(my_host_name, MAX_HOST_LEN);
            for (i = 0; i < npids; i++) {
                if (pid == local_processes[i].pid) {
                    rank = local_processes[i].rank;
                }
            }
            const unsigned int str_max_size = 256;
            char process_str[str_max_size];
            if (rank != -1) {
                snprintf(process_str, str_max_size, "MPI process (rank: %d, pid: %d)", rank, pid);
            } else {
                snprintf(process_str, str_max_size, "Process with pid %d", pid);
            }
            if (WIFEXITED(status)) {
                PRINT_ERROR("%s exited with status %d\n", process_str, WEXITSTATUS(status));
                if (rank != -1) {
                    // Report non-zero exit code to mpirun_rsh
                    report_error(MPISPAWN_MPIPROCESS_NONZEROEXIT);
                }
            } else if (WIFSIGNALED(status)) {
                PRINT_ERROR("%s terminated with signal %d -> abort job\n", process_str, WTERMSIG(status));
                if (rank != -1) {
                    // Abort only if a child MPI process terminates because of a signal
                    mpispawn_abort(MPISPAWN_MPIPROCESS_ERROR);
                }
            } else if (WIFSTOPPED(status)) {
                PRINT_ERROR("%s stopped with signal %d\n", process_str, WSTOPSIG(status));
            } else if (WIFCONTINUED(status)) {
                PRINT_ERROR("%s continued\n", process_str);
            }
        }
    }
}

void mpispawn_checkin(char * l_port)
{
    int connect_attempt = 0, max_connect_attempts = 5, i, sock;
    struct hostent *mpirun_hostent;
    struct sockaddr_in sockaddr;
    /*struct sockaddr_in c_sockaddr; */
    int offset = 0, id;
    pid_t pid = getpid();
    char port[MAX_PORT_LEN + 1];

    if (!USE_LINEAR_SSH) {
        if (mt_id != 0) {
            offset = 1;
            MPISPAWN_HAS_PARENT = 1;
        }
        mpispawn_fds = (int *) malloc(sizeof(int) * (MPISPAWN_NCHILD + MPISPAWN_HAS_PARENT));
        if (MPISPAWN_NCHILD) {
            mpispawn_pids = (pid_t *) malloc(sizeof(pid_t) * MPISPAWN_NCHILD);
            for (i = 0; i < MPISPAWN_NCHILD; i++) {
                while ((sock = accept(checkin_sock, NULL, 0)) < 0) {
                    if (errno == EINTR || errno == EAGAIN)
                        continue;
                    perror("accept [mt_checkin]");
                }
                mpispawn_fds[i + offset] = sock;

                if (read_socket(sock, &id, sizeof(int))
                        || read_socket(sock, &mpispawn_pids[i], sizeof(pid_t))
                        || read_socket(sock, &port, MAX_PORT_LEN + 1)) {
                    PRINT_ERROR("read_socket() failed\n");
#ifdef CKPT
                    cr_cleanup();
#endif
                    process_cleanup();
                    abort();
                }

            }
        }
    }
    mpirun_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!USE_LINEAR_SSH && mt_id != 0)
        mpispawn_fds[0] = mpirun_socket;
    if (mpirun_socket < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    mpirun_hostent = gethostbyname(getenv("MPISPAWN_MPIRUN_HOST"));
    if (mpirun_hostent == NULL) {
        mpirun_hostent = gethostbyname(getenv("MPISPAWN_MPIRUN_HOSTIP"));
        if (mpirun_hostent == NULL) {
            herror("gethostbyname");
            exit(EXIT_FAILURE);
        }
    }

    sockaddr.sin_family = AF_INET;
    sockaddr.sin_addr = *(struct in_addr *) (*mpirun_hostent->h_addr_list);
    sockaddr.sin_port = htons(env2int("MPISPAWN_CHECKIN_PORT"));

    while (connect(mpirun_socket, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0) {
        if (++connect_attempt > max_connect_attempts) {
            perror("connect [mt_checkin]");
            exit(EXIT_FAILURE);
        }
    }

    if (write_socket(mpirun_socket, &mt_id, sizeof(int))) {
        fprintf(stderr, "Error writing id [%d]!\n", mt_id);
        close(mpirun_socket);
        exit(EXIT_FAILURE);
    }

    if (write_socket(mpirun_socket, &pid, sizeof(pid_t))) {
        fprintf(stderr, "Error writing pid [%d]!\n", pid);
        close(mpirun_socket);
        exit(EXIT_FAILURE);
    }


    if (write_socket(mpirun_socket, l_port, MAX_PORT_LEN + 1)) {
        fprintf(stderr, "Error writing l_port!\n");
        close(mpirun_socket);
        exit(EXIT_FAILURE);
    }

    if (recv_environ(mpirun_socket)) {
        fprintf(stderr, "Error writing receiving environ!\n");
        close(mpirun_socket);
        exit(EXIT_FAILURE);
    }

    if (!USE_LINEAR_SSH) {
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            if (send_environ(mpispawn_fds[i + offset])) {
                close(mpirun_socket);
                exit(EXIT_FAILURE);
            }
        }
    }

    else if (!(mt_id == 0 && env2int("MPISPAWN_USE_TOTALVIEW"))) {
        close(mpirun_socket);
    }
}

char * init_listening_socket(int *mc_socket)
{
    static char port[MAX_PORT_LEN + 1];
    struct sockaddr_in mc_sockaddr;
    socklen_t mc_sockaddr_len = sizeof(mc_sockaddr);
    int s;

    *mc_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (*mc_socket < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    mc_sockaddr.sin_family = AF_INET;
    mc_sockaddr.sin_addr.s_addr = INADDR_ANY;
    mc_sockaddr.sin_port = 0;

    if (bind(*mc_socket, (struct sockaddr *) &mc_sockaddr, mc_sockaddr_len)
        < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    if (getsockname(*mc_socket, (struct sockaddr *) &mc_sockaddr, &mc_sockaddr_len) < 0) {
        perror("getsockname");
        exit(EXIT_FAILURE);
    }

    listen(*mc_socket, MT_MAX_DEGREE);
    s = getnameinfo((struct sockaddr *)&mc_sockaddr, mc_sockaddr_len, NULL, 0,
            port, MAX_PORT_LEN + 1, NI_NUMERICSERV);
    if (0 != s) {
        PRINT_ERROR("%s\n", gai_strerror(s));
        exit(EXIT_FAILURE);
    }

    return port;
}

void wait_for_errors(int s, struct sockaddr *sockaddr, unsigned int sockaddr_len)
{
    int wfe_socket, wfe_abort_code, wfe_abort_rank, wfe_abort_msglen;

    char my_host_name[MAX_HOST_LEN];
    gethostname(my_host_name, MAX_HOST_LEN);
    //fprintf (stderr, "hostname %s wait_for_errors:inside :\n",my_host_name);

  WFE:
    while ((wfe_socket = accept(s, sockaddr, &sockaddr_len)) < 0) {
        if (errno == EINTR || errno == EAGAIN)
            continue;
        PRINT_ERROR_ERRNO("accept() failed", errno);
        mpispawn_abort(MPISPAWN_INTERNAL_ERROR);
    }

    if (read_socket(wfe_socket, &wfe_abort_code, sizeof(int))
        || read_socket(wfe_socket, &wfe_abort_rank, sizeof(int))
        || read_socket(wfe_socket, &wfe_abort_msglen, sizeof(int))) {
        PRINT_ERROR("Termination socket read failed!\n");
    } else {
        char wfe_abort_message[wfe_abort_msglen];
        if (read_socket(wfe_socket, &wfe_abort_message, wfe_abort_msglen))
            wfe_abort_message[0] = '\0';
        PRINT_ERROR("Abort signaled by rank %d: '%s'\n", wfe_abort_rank, wfe_abort_message);
    }
    mpispawn_abort(MPISPAWN_MPIPROCESS_ERROR);
    goto WFE;
}

#define MPISPAWN_PARENT_FD mpispawn_fds[0]
#define MPISPAWN_CHILD_FDS (&mpispawn_fds[MPISPAWN_HAS_PARENT])
static void dump_fds()
{
    int i;
    PRINT_DEBUG(DEBUG_Fork_verbose, "has-parent = %d, MT_CHILD=%d,  NCHILD=%d\n", MPISPAWN_HAS_PARENT, MPISPAWN_NCHILD, NCHILD);
    if (MPISPAWN_HAS_PARENT) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "parent-fd=%d\n", MPISPAWN_PARENT_FD);
    }
    for (i = 0; i < MPISPAWN_NCHILD; i++) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "MT_CHILD_%d: fd=%d\n", i, MPISPAWN_CHILD_FDS[i]);
    }
    for (i = 0; i < NCHILD; i++) {
        PRINT_DEBUG(DEBUG_Fork_verbose, "NCLD_%d:  fd=%d\n", i, children[i].fd);
    }
}

void init_debug(int mt_id)
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
    snprintf(output_prefix, MAX_LENGTH, "%s:mpispawn_%i", hostname, mt_id);
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
            cleanup_handler(signal);
            break;
        case SIGCHLD:
            child_handler(signal);
            break;
	case SIGUSR2:
            PRINT_ERROR("Caught SIGUSR2 (signal %d)\n", signal);
            report_error(MPISPAWN_TRIGGER_MIGRATION);
            break;
        case SIGTSTP:
            sigtstp_handler(signal);
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
    sigaddset(&sigmask, SIGCHLD);
    sigaddset(&sigmask, SIGTSTP);
    sigaddset(&sigmask, SIGUSR2);

    start_sp_thread(sigmask, signal_processor, 1);
}

int main(int argc, char *argv[])
{
    mpispawn_state = MPISPAWN_STATE_INITIALIZING;
    
    int l_socket, i;
    char * l_port = init_listening_socket(&l_socket);
    int mt_degree = 0;
    int mt_nnodes;

    // Global variable
    mt_id = env2int("MPISPAWN_ID");

    init_debug(mt_id);
    setup_signal_handling_thread();

    // Local variable
    mt_nnodes = env2int("MPISPAWN_NNODES");

    // Static variable
    USE_LINEAR_SSH = env2int("USE_LINEAR_SSH");

    // Global variable, used in pmi_tree.c
    NCHILD = env2int("MPISPAWN_LOCAL_NPROCS");
    N = env2int("MPISPAWN_GLOBAL_NPROCS");

    children = (child_t *) malloc(NCHILD * child_s);

    char *portname = getenv("PARENT_ROOT_PORT_NAME");
    if (portname) {
        add_kvc("PARENT_ROOT_PORT_NAME", portname, 1);
    }

#if defined(CKPT) && defined(CR_AGGRE)
    int rv = init_ckpt_aggregation();
    if (rv != 0) {
        PRINT_ERROR("Failed to initialize Write Aggregation for Checkpoint/Restart. Aborting...\n");
        PRINT_ERROR("To disable Write Aggregation, use MV2_CKPT_USE_AGGREGATION=0.\n");
        exit(EXIT_FAILURE);
    }
#endif

    /* Create listening socket for ranks */
    /* Doesn't need to be TCP as we're all on local node */
    c_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (c_socket < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    c_sockaddr.sin_family = AF_INET;
    c_sockaddr.sin_addr.s_addr = INADDR_ANY;
    c_sockaddr.sin_port = 0;

    if (bind(c_socket, (struct sockaddr *) &c_sockaddr, sockaddr_len) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }
    if (getsockname(c_socket, (struct sockaddr *) &c_sockaddr, &sockaddr_len) < 0) {
        perror("getsockname");
        exit(EXIT_FAILURE);
    }
    listen(c_socket, NCHILD);
    c_port = (int) ntohs(c_sockaddr.sin_port);

#ifdef CKPT
    CR_Init(NCHILD);
#endif

    mt_degree = env2int("MV2_MT_DEGREE");
    if (!mt_degree) {
        mt_degree = MT_DEFAULT_DEGREE;
    }

    else if (mt_degree < MT_MIN_DEGREE) {
        mt_degree = MT_MIN_DEGREE;
    }

    else if (mt_degree > MT_MAX_DEGREE) {
        mt_degree = MT_MAX_DEGREE;
    }
 
    mpispawn_state = MPISPAWN_STATE_STARTING;

    if (!USE_LINEAR_SSH) {
        spawn_mpispawn_tree(argc, argv, mt_nnodes, mt_degree);
    }

    int r = setup_global_environment();
    if ( r != 0 ) {
        exit(EXIT_FAILURE);
    }

    if (chdir(getenv("MPISPAWN_WORKING_DIR"))) {
        perror("chdir");
        exit(EXIT_FAILURE);
    }

    mpispawn_checkin(l_port);

    if (USE_LINEAR_SSH) {
        mpispawn_fds = mpispawn_tree_init(mt_id, mt_degree, mt_nnodes, l_socket);
        if (mpispawn_fds == NULL) {
            PRINT_ERROR("Internal error: mpispawn_fds is null\n");
            exit(EXIT_FAILURE);
        }
    }

#ifdef CKPT
    mtpmi_init();
    cr_spawn_degree = mt_degree;
    dbg("mt_degree=%d\n", mt_degree);
    if (!NCHILD) {
        goto skip_spawn_processes;
    }
#if defined(CR_FTB)
  spawn_processes:
#endif
#endif
    spawn_processes(NCHILD);
    mpispawn_state = MPISPAWN_STATE_RUNNING;

    for (i = 0; i < NCHILD; i++) {
        int sock;
      ACCEPT_HID:
        sock = accept(c_socket, (struct sockaddr *) &c_sockaddr, &sockaddr_len);
        if (sock < 0) {
            printf("%d", errno);
            if ((errno == EINTR) || (errno == EAGAIN)) {
                goto ACCEPT_HID;
            } else {
                perror("accept");
                return (EXIT_FAILURE);
            }
        }
        children[i].fd = sock;
        children[i].rank = 0;
        children[i].c_barrier = 0;
        dbg("has accept() child_%d of %d: fd=%d\n", i, NCHILD, sock);
    }

    dump_fds();

#if defined(CKPT)
#if defined(CR_FTB)
    // Begin of CKPT && CR_FTB  ////////////////////////////////////////////////////////////////////

  skip_spawn_processes:
    mtpmi_processops();
    dbg(" ====  after mtpmi_processops...\n");
    // respawn_processes: //come back here for respawing again for subsequent restart process migration

    if (cr_mig_tgt) {
        while (!cr_mig_spare_cond) ;
        // cr_mig_tgt = 0;
        cr_mig_spare_cond = 0;
        NCHILD = eNCHILD;
        restart_context = 1;
        dbg("host %s: mig-tgt: NCHILD=%d, Jump to spawn_processes\n", my_hostname, NCHILD);
        //fflush(stdout);
        goto spawn_processes;
    }
    /*else if( cr_mig_src ) {
       while( !cr_mig_src_can_exit ) usleep(100000); 
       dbg("host %s: on mig-src, will exit...\n", my_hostname );
       // now, child MPI-proc has finished,can exit this mpispawn
       cr_mig_src_can_exit = 0;
       //cr_mig_src = 0;
       cleanup();
       return EXIT_FAILURE;
       } */
    //   Spawn wait_for_error_thread 
    if (pthread_create(&CR_wfe_tid, NULL, CR_wait_for_errors, NULL)) {
        perror("[main:mpispawn] pthread_create()");
        exit(EXIT_FAILURE);
    }
    // Wait for Connect_MPI_Procs() to start listening
    dbg("has created wait_for_err thr, cr_mig_tgt=%d...\n", cr_mig_tgt);
    do {
        sleep(1);
    }
    while (!cr_mig_tgt && num_migrations > 0);
    // At src of migration. Keep idle till mpirun_rsh tells me to stop

    dbg("%s pthread_cancel wfe_thread\n", my_hostname);
    pthread_cancel(CR_wfe_tid);

    pthread_join(CR_wfe_tid, NULL);
    dbg("%s: ******  will exit now... \n", my_hostname);
    //goto respawn_processes;

    // Enf of CKPT && CR_FTB  //////////////////////////////////////////////////////////////////////
#else
    // Begin of CKPT && !CR_FTB  ///////////////////////////////////////////////////////////////////

  skip_spawn_processes:
    mtpmi_processops();
    dbg(" ====  after mtpmi_processops...\n");

    wait_for_errors(c_socket, (struct sockaddr *) &c_sockaddr, sockaddr_len);

    // End of CKPT && !CR_FTB   //////////////////////////////////////////////////////////////////////
#endif
#else
    // Begin of !CKPT  ///////////////////////////////////////////////////////////////////////////////

    mtpmi_init();
    mtpmi_processops();
    dbg(" ====  after mtpmi_processops...\n");

    wait_for_errors(c_socket, (struct sockaddr *) &c_sockaddr, sockaddr_len);

    // End of !CKPT  /////////////////////////////////////////////////////////////////////////////////
#endif

    mpispawn_state = MPISPAWN_STATE_FINALIZING;

    // Cleanup before exiting
#if defined(CKPT)
    cr_cleanup();
#endif
    process_cleanup();
    PRINT_DEBUG(DEBUG_Fork_verbose, "exit(EXIT_FAILURE)\n");
    return EXIT_FAILURE;
}
