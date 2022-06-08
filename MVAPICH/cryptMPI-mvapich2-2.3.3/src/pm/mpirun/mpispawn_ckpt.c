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

#include "mpichconf.h"

#ifdef CKPT

#include <sys/socket.h>
#include <stdlib.h>
#include <netdb.h>
#include <fcntl.h>
#include <sys/wait.h>

#include "mpispawn_ckpt.h"
#include "mpispawn_tree.h"
#include "mpirun_util.h"
#include "debug_utils.h"
#include "crfs.h"
// TODO: remove debug.h
#include "debug.h"

#ifdef CR_FTB
#include <libftb.h>
#endif

#define MAX_CR_MSG_LEN      256
#define DEFAULT_MPIRUN_PORT 14678
#define DEFAULT_MPD_PORT    24678
#define CR_RSRT_PORT_CHANGE 16

#ifdef CR_FTB

#define MAX_HOST_LEN 256
static char my_hostname[MAX_HOST_LEN];

volatile int cr_mig_src = 0;
volatile int cr_mig_tgt = 0;
volatile int num_migrations = 0;

int eNCHILD;

volatile int cr_mig_spare_cond = 0;
volatile int cr_mig_src_can_exit = 0;
char cr_mig_src_host[32];
char cr_mig_tgt_host[32];

int *migrank;

static FTB_client_t ftb_cinfo;
static FTB_subscribe_handle_t shandle;
FTB_event_info_t cr_ftb_events[] = CR_FTB_EVENT_INFO;

static int cr_ftb_init(char *);
static void cr_ftb_finalize();
static int cr_ftb_callback(FTB_receive_event_t *, void *);
static int get_src_tgt(char *, char *, char *);
static int get_tgt_rank_list(char *, int *, int **);
static int ftb_init_done;

pthread_t CR_wfe_tid;

FTB_client_handle_t ftb_handle;

#else                           /* !CR_FTB */

static volatile int cr_cond = 0;
static pthread_t worker_tid;

static int *mpispawn_fd;
static int mpirun_fd;
static int mpispawn_port;
static volatile int cr_worker_can_exit = 0;

static void *CR_Worker(void *arg);

#endif

int cr_spawn_degree;

int checkpoint_count;
int restart_context;
char ckpt_filename[CR_MAX_FILENAME];
char *sessionid;

static char session_file[CR_MAX_FILENAME];

#ifdef CR_AGGRE
int use_aggre_mig = 0;          // whether we enable migration func in CRFS
int use_aggre = 0;
#endif

// Dependencies from mpispawn.c TODO: remove this
extern void wait_for_errors(int s, struct sockaddr *sockaddr, unsigned int sockaddr_len);
extern int c_socket;
extern struct sockaddr_in c_sockaddr;
extern unsigned int sockaddr_len;
extern in_port_t c_port;
extern int MPISPAWN_NCHILD;
extern child_t *children;

void cr_cleanup(void)
{
    PRINT_DEBUG(DEBUG_FT_verbose, "Cleanup Checkpoint/Restart/Migration\n");
    unlink(session_file);
#ifdef CR_AGGRE
    if (use_aggre) {
        stop_crfs_wa();
    }
    if (use_aggre_mig) {
        stop_crfs_mig();
    }
#endif
#ifdef CR_FTB
    cr_ftb_finalize();
#else
    cr_worker_can_exit = 1;
#endif
}

void *CR_wait_for_errors(void *arg)
{
    dbg("CR_wait_for_errors\n");

    wait_for_errors(c_socket, (struct sockaddr *) &c_sockaddr, sockaddr_len);

    dbg("CR_wait_for_errors done\n");
    return NULL;
}

int CR_Init(int nProcs)
{
    char *temp;

#ifndef CR_FTB
    int mpirun_port;
    temp = getenv("MPISPAWN_MPIRUN_CR_PORT");
    if (temp) {
        mpirun_port = atoi(temp);
    } else {
        PRINT_ERROR("MPISPAWN_MPIRUN_CR_PORT unknown\n");
        exit(EXIT_FAILURE);
    }
#endif

    temp = getenv("MPISPAWN_CR_CONTEXT");
    if (temp) {
        restart_context = atoi(temp);
    } else {
        PRINT_ERROR("MPISPAWN_CR_CONTEXT unknown\n");
        exit(EXIT_FAILURE);
    }

    sessionid = getenv("MPISPAWN_CR_SESSIONID");
    if (!sessionid) {
        PRINT_ERROR("MPISPAWN_CR_SESSIONID unknown\n");
        exit(EXIT_FAILURE);
    }

    snprintf(session_file, CR_MAX_FILENAME, "/tmp/cr.session.%s", sessionid);

    temp = getenv("MPISPAWN_CR_CKPT_CNT");
    if (temp) {
        checkpoint_count = atoi(temp);
    } else {
        PRINT_ERROR("MPISPAWN_CR_CKPT_CNT unknown\n");
        exit(EXIT_FAILURE);
    }

#ifndef CR_AGGRE
    strncpy(ckpt_filename, DEFAULT_CHECKPOINT_FILENAME, CR_MAX_FILENAME);
#endif

#ifdef CR_FTB

    char pmi_port[MAX_HOST_LEN + MAX_PORT_LEN];
    char hostname[MAX_HOST_LEN];
    FILE *fp;

    /* Get PMI Port information */
    gethostname(hostname, MAX_HOST_LEN);
    sprintf(pmi_port, "%s:%d", hostname, c_port);

    /* Create the session file with PMI Port information */
    fp = fopen(session_file, "w+");
    if (!fp) {
        PRINT_ERROR_ERRNO("fopen() failed to create session file '%s'\n", errno, session_file);
        exit(EXIT_FAILURE);
    } else {
        PRINT_DEBUG(DEBUG_FT_verbose, "Session file = %s\n", session_file);
    }
    if (fwrite(pmi_port, sizeof(pmi_port), 1, fp) == 0) {
        PRINT_ERROR("fwrite() failed to write PMI Port number\n");
        exit(EXIT_FAILURE);
    }
    dbg("write pmi-port= %s to session-file %s\n", pmi_port, session_file);
    fclose(fp);

    if (cr_ftb_init(sessionid)) {
        PRINT_ERROR("cr_ftb_init() failed\n");
        exit(EXIT_FAILURE);
    }
#else
    struct hostent *hp;
    struct sockaddr_in sa;

    /* Connect to mpirun_rsh */
    mpirun_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (mpirun_fd < 0) {
        perror("[CR_Init] socket()");
        exit(EXIT_FAILURE);
    }

    hp = gethostbyname(getenv("MPISPAWN_MPIRUN_HOST"));
    if (!hp) {
        perror("[CR_Init] gethostbyname()");
        exit(EXIT_FAILURE);
    }

    bzero((void *) &sa, sizeof(sa));
    bcopy((void *) hp->h_addr, (void *) &sa.sin_addr, hp->h_length);
    sa.sin_family = AF_INET;
    sa.sin_port = htons(mpirun_port);

    if (connect(mpirun_fd, (struct sockaddr *) &sa, sizeof(sa)) < 0) {
        perror("[CR_Init] connect()");
        exit(EXIT_FAILURE);
    }

    mpispawn_fd = malloc(nProcs * sizeof(int));
    if (!mpispawn_fd) {
        perror("[CR_Init] malloc()");
        exit(EXIT_FAILURE);
    }

    /* Spawn CR Worker Thread */
    cr_worker_can_exit = 0;
    if (pthread_create(&worker_tid, NULL, CR_Worker, (void *) (uintptr_t) nProcs)) {
        perror("[CR_Init] pthread_create()");
        exit(EXIT_FAILURE);
    }

    /* Wait for Connect_MPI_Procs() to start listening */
    while (!cr_cond) ;

#endif                          /* CR_FTB */

    return (0);
}

#ifdef CR_FTB

static int cr_ftb_init(char *sessionid)
{
    dbg(" ----- now init, cnt=%d\n", ++cnt);
    if (ftb_init_done)
        return 0;
    ftb_init_done = 1;

    int ret;
    char *subscription_str;

    memset(&ftb_cinfo, 0, sizeof(ftb_cinfo));
    strcpy(ftb_cinfo.client_schema_ver, "0.5");
    strcpy(ftb_cinfo.event_space, "FTB.STARTUP.MV2_MPISPAWN");
    strcpy(ftb_cinfo.client_name, "MV2_MPISPAWN");

    gethostname(my_hostname, MAX_HOST_LEN);

    /* sessionid should be <= 16 bytes since client_jobid is 16 bytes. */
    snprintf(ftb_cinfo.client_jobid, FTB_MAX_CLIENT_JOBID, "%s", sessionid);

    strcpy(ftb_cinfo.client_subscription_style, "FTB_SUBSCRIPTION_BOTH");

    ret = FTB_Connect(&ftb_cinfo, &ftb_handle);
    if (ret != FTB_SUCCESS)
        goto err_connect;

    ret = FTB_Declare_publishable_events(ftb_handle, NULL, cr_ftb_events, CR_FTB_EVENTS_MAX);
    if (ret != FTB_SUCCESS)
        goto err_declare_events;

    subscription_str = malloc(sizeof(char) * FTB_MAX_SUBSCRIPTION_STR);
    if (!subscription_str)
        goto err_malloc;

    snprintf(subscription_str, FTB_MAX_SUBSCRIPTION_STR, "event_space=FTB.STARTUP.MV2_MPIRUN , jobid=%s", sessionid);

    ret = FTB_Subscribe(&shandle, ftb_handle, subscription_str, cr_ftb_callback, NULL);
    free(subscription_str);
    if (ret != FTB_SUCCESS)
        goto err_subscribe;

    ftb_init_done = 1;
    return (0);

  err_connect:
    fprintf(stderr, "FTB_Connect() failed with %d\n", ret);
    ret = -1;
    goto exit_connect;

  err_declare_events:
    fprintf(stderr, "FTB_Declare_publishable_events() failed with %d\n", ret);
    ret = -2;
    goto exit_declare_events;

  err_malloc:
    fprintf(stderr, "Failed to malloc() subscription_str\n");
    ret = -3;
    goto exit_malloc;

  err_subscribe:
    fprintf(stderr, "FTB_Subscribe() failed with %d\n", ret);
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
    dbg("finalize cnt=%d...\n", ++cnt);
    if (ftb_init_done) {
        ftb_init_done = 0;
        FTB_Unsubscribe(&shandle);
        usleep(20000);
        FTB_Disconnect(ftb_handle);
    }
    ftb_init_done = 0;
}

#ifdef CR_AGGRE
/// we have started a proc-migration using Aggregation-based strategy
int cr_ftb_aggre_based_mig(char *msg)
{
    int i, j;
    int num_mig_procs = 0;
    char buf[256];

    strncpy(buf, msg, 255);
    buf[255] = 0;
    dbg("enter with buf = \"%s\"\n", buf);

    /// "buf" is in format:  "srcnode  tgtnode  proc_cnt  procid1  procid2 ..."
    // parse this string to extract all infor
    char *tok;

    tok = strtok(buf, " \n\t"); // src
    strcpy(cr_mig_src_host, tok);
    tok = strtok(NULL, " \n\t");    // tgt
    strcpy(cr_mig_tgt_host, tok);
    tok = strtok(NULL, " \n\t");    // proc-count
    num_mig_procs = atoi(tok);

    if (strstr(my_hostname, cr_mig_src_host) == my_hostname)
        cr_mig_src = 1;
    if (strstr(my_hostname, cr_mig_tgt_host) == my_hostname)
        cr_mig_tgt = 1;
    dbg(" src=%s, tgt=%s, mig-src=%d, mig-tgt=%d, num_procs = %d\n", cr_mig_src_host, cr_mig_tgt_host, cr_mig_src, cr_mig_tgt, num_mig_procs);

    if (num_mig_procs <= 0) {
        fprintf(stderr, "[mpispanw_%d]: %s: procs to be migrated wrong:: %s\n", mt_id, __func__, tok);
        return -1;
    }

    if (cr_mig_tgt) {
        migrank = (int *) malloc(num_mig_procs * sizeof(int));
        if (!migrank) {
            fprintf(stderr, "[mpispawn_%d]: %s: malloc migrank failed\n", mt_id, __func__);
            return -1;
        }

        tok = strtok(NULL, " \n\t");    // proc-count
        i = 0;
        while (tok) {
            if (i >= num_mig_procs) {
                fprintf(stderr, "[mpispawn_%d]: %s: too many proc-ranks: %d\n", mt_id, __func__, i);
                free(migrank);
                migrank = NULL;
                return -1;
            }
            migrank[i] = atoi(tok);
            i++;
            tok = strtok(NULL, " \n\t");
        }
    }
    // adjust MPISPAWN_Tree's topology: num of MPISPAWN-child nodes not to
    // participate in a barrier
    for (i = 0; i < MPISPAWN_NCHILD; i++) {
        PRINT_DEBUG(DEBUG_FT_verbose, "spawninfo[i=%d].spawnhost = %s\n", i, spawninfo[i].spawnhost);
        j = mt_id * cr_spawn_degree + i + 1;
        PRINT_DEBUG(DEBUG_FT_verbose, "spawninfo[j=%d].spawnhost = %s\n", j, spawninfo[j].spawnhost);
        if (!strcmp(spawninfo[j].spawnhost, cr_mig_src_host)) {
            --exclude_spare;
            PRINT_DEBUG(DEBUG_FT_verbose, "--exclude_spare = %d\n", exclude_spare);
        }
        if (!strcmp(spawninfo[j].spawnhost, cr_mig_tgt_host)) {
            ++exclude_spare;
            PRINT_DEBUG(DEBUG_FT_verbose, "++exclude_spare = %d\n", exclude_spare);
        }
    }

    if (!cr_mig_tgt)
        return (0);

    //// if I'm target node, start new process now... 
    eNCHILD = num_mig_procs;
    children = (child_t *) malloc(eNCHILD * sizeof(child_t));
    cr_mig_spare_cond = 1;

    return (0);
}
#endif                          /* CR_AGGRE */

static int cr_ftb_callback(FTB_receive_event_t * revent, void *arg)
{
    int i, j;
// char my_hostname[256];
// gethostname(my_hostname, 255);

    dbg("  at %s: Got event %s from %s:payload=\"%s\"\n", my_hostname, revent->event_name, revent->client_name, revent->event_payload);
    //fflush(stdout);

    /* TODO: Do some sanity checking to see if this is the intended target */

    if (!strcmp(revent->event_name, EVENT(CR_FTB_MIGRATE))) {
        num_migrations++;
#ifdef CR_AGGRE
        if (use_aggre && use_aggre_mig) {
            i = cr_ftb_aggre_based_mig(revent->event_payload);
            dbg("Aggre-based Mig: ret %d\n", i);
            return 0;
        }
#endif
        /* Arm source & target for Migration */
        get_src_tgt(revent->event_payload, cr_mig_src_host, cr_mig_tgt_host);
        if (strstr(my_hostname, cr_mig_src_host) == my_hostname)
            cr_mig_src = 1;
        if (strstr(my_hostname, cr_mig_tgt_host) == my_hostname)
            cr_mig_tgt = 1;
        dbg(" src=%s, tgt=%s, mig-src=%d, mig-tgt=%d\n", cr_mig_src_host, cr_mig_tgt_host, cr_mig_src, cr_mig_tgt);
        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_MIGRATE_PIIC))) {
        char my_hostname[256];
        gethostname(my_hostname, 255);
        //int has_src = 0;
        //int has_tgt = 0;
        /* Adjust exclude_spares based on new process distribution */
        for (i = 0; i < MPISPAWN_NCHILD; i++) {
            PRINT_DEBUG(DEBUG_FT_verbose, "spawninfo[i=%d].spawnhost = %s\n", i, spawninfo[i].spawnhost);
            j = mt_id * cr_spawn_degree + i + 1;
            PRINT_DEBUG(DEBUG_FT_verbose, "spawninfo[j=%d].spawnhost = %s\n", j, spawninfo[j].spawnhost);
            if (!strcmp(spawninfo[j].spawnhost, cr_mig_src_host)) {
                //has_src = 1;
                --exclude_spare;
                PRINT_DEBUG(DEBUG_FT_verbose, "--exclude_spare = %d\n", exclude_spare);
            }
            if (!strcmp(spawninfo[j].spawnhost, cr_mig_tgt_host)) {
                ++exclude_spare;
                PRINT_DEBUG(DEBUG_FT_verbose, "++exclude_spare = %d\n", exclude_spare);
                //has_tgt =1;
            }

        }

        dbg(" [mpispawn:%s] cr_mig_tgt=%d\n", my_hostname, cr_mig_tgt);
        //fflush(stdout);
        if (!cr_mig_tgt)
            return (0);

        /* Setup environment for process launch if target */
        get_tgt_rank_list(revent->event_payload, &eNCHILD, &migrank);
        children = (child_t *) malloc(eNCHILD * child_s);
        cr_mig_spare_cond = 1;
        return (0);
    }
    return 0;
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

static int get_tgt_rank_list(char *str, int *n, int **lst)
{
    int i, j, ci, *list;
    char cnum[16];

    if (!str || !n || !lst)
        return (-1);

    /* Find number of ranks */
    ci = i = 0;
    while (str[i]) {
        if (str[i] == ' ')
            ++ci;
        ++i;
    }
    ++ci;
    *n = ci;

    list = (int *) malloc((*n) * sizeof(int));
    if (!list) {
        fprintf(stderr, "[get_tgt_rank_list] malloc failed\n");
        return (-1);
    }

    i = j = ci = 0;
    while (str[i]) {

        if (str[i] == ' ') {
            cnum[ci] = '\0';
            ci = 0;
            list[j++] = atoi(cnum);
            ++i;
            continue;
        }

        cnum[ci++] = str[i++];
    }
    cnum[ci] = '\0';
    list[j++] = atoi(cnum);

    *lst = list;

    return (0);
}

#else                           /* !CR_FTB */

static int Connect_MPI_Procs(int nProcs)
{
    int i;
    FILE *fp;
    int mpispawn_listen_fd;
    struct sockaddr_in sa;

    mpispawn_listen_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (mpispawn_listen_fd < 0) {
        perror("[Connect_MPI_Procs] socket()");
        exit(EXIT_FAILURE);
    }

    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = 0;

    if (bind(mpispawn_listen_fd, (struct sockaddr *) &sa, sizeof(sa)) < 0) {
        perror("[Connect_MPI_Procs] bind()");
        exit(EXIT_FAILURE);
    }

    i = sizeof(sa);
    if (getsockname(mpispawn_listen_fd, (struct sockaddr *) &sa, (socklen_t *) & i) < 0) {
        perror("[Connect_MPI_Procs] getsockname()");
        close(mpispawn_listen_fd);
        exit(EXIT_FAILURE);
    }

    mpispawn_port = ntohs(sa.sin_port);

    fp = fopen(session_file, "w+");
    if (!fp) {
        fprintf(stderr, "[Connect_MPI_Procs] Cannot create Session File\n");
        fflush(stderr);
        close(mpispawn_listen_fd);
        exit(EXIT_FAILURE);
    }
    if (fwrite(&mpispawn_port, sizeof(mpispawn_port), 1, fp) == 0) {
        fprintf(stderr, "[Connect_MPI_Procs] Cannot write Session Id\n");
        fflush(stderr);
        close(mpispawn_listen_fd);
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    if (listen(mpispawn_listen_fd, nProcs) < 0) {
        perror("[Connect_MPI_Procs] listen()");
        exit(EXIT_FAILURE);
    }

    /* Signal CR_Init() that you are listening */
    cr_cond = 1;

    for (i = 0; i < nProcs; i++) {
        if ((mpispawn_fd[i] = accept(mpispawn_listen_fd, 0, 0)) < 0) {
            if (errno == EINTR || errno == EAGAIN) {
                i--;
//                 debug("%s: error::  errno=%d\n", __func__, errno);
                continue;
            }
            perror("[Connect_MPI_Procs] accept()");
            exit(EXIT_FAILURE);
        }
    }

    close(mpispawn_listen_fd);

    return (0);
}

static void *CR_Worker(void *arg)
{
    int ret, i, nProcs;
    char cr_msg_buf[MAX_CR_MSG_LEN];
    fd_set set;
    int max_fd;

    nProcs = (int) (uintptr_t) arg;

    Connect_MPI_Procs(nProcs);
    struct timeval tv;
    int ready = 0;
    dbg("after connect-MPI-procs: \n");

    while (1) {

        FD_ZERO(&set);
        FD_SET(mpirun_fd, &set);
        max_fd = mpirun_fd;

        for (i = 0; i < nProcs; i++) {
            FD_SET(mpispawn_fd[i], &set);
            max_fd = (max_fd >= mpispawn_fd[i]) ? max_fd : mpispawn_fd[i];
        }

        max_fd += 1;

        tv.tv_sec = 0;
        tv.tv_usec = 100000;
        ready = select(max_fd, &set, NULL, NULL, &tv);
        if (ready == 0) {
            // time out, or interrupted
            if (cr_worker_can_exit) {
                dbg("will exit CR_Worker()...\n");
                break;
            }
            continue;

        } else if (ready < 0) {
            dbg("select has returned %d: %s", ready, strerror(errno));
            continue;
        }

        if (FD_ISSET(mpirun_fd, &set)) {

            /* We need to send a message from mpirun_rsh -> MPI Processes */

            ret = CR_MPDU_readline(mpirun_fd, cr_msg_buf, MAX_CR_MSG_LEN);
            if (!ret)
                continue;
            for (i = 0; i < nProcs; i++)
                CR_MPDU_writeline(mpispawn_fd[i], cr_msg_buf);

        } else {

            /* We need to send a message from MPI Processes -> mpirun_rsh */

            for (i = 0; i < nProcs; i++) {
                if (FD_ISSET(mpispawn_fd[i], &set))
                    break;
            }

            ret = CR_MPDU_readline(mpispawn_fd[i], cr_msg_buf, MAX_CR_MSG_LEN);
            if (!ret)
                continue;

            /* Received a PMI Port Query */
            if (strstr(cr_msg_buf, "query_pmi_port")) {
                snprintf(cr_msg_buf, MAX_CR_MSG_LEN, "cmd=reply_pmi_port val=%s\n", getenv("PMI_PORT"));
                CR_MPDU_writeline(mpispawn_fd[i], cr_msg_buf);
                continue;
            }

            CR_MPDU_writeline(mpirun_fd, cr_msg_buf);

            /* Received a Finalize Checkpoint message */
            if (strstr(cr_msg_buf, "finalize_ckpt")) {
                return (0);
            }

        }

    }                           /* while(1) */
    return NULL;
}

#endif                          /* CR_FTB */

int restart_mpi_process(int cached_cr_mig_tgt, int i)
{
    char **cr_argv;
    char str[32];
    int rank;
    if (restart_context) {

        PRINT_DEBUG(DEBUG_FT_verbose, "i = %d, cached_cr_mig_tgt = %d\n", i, cached_cr_mig_tgt);
        restart_context = 0;
        cr_argv = (char **) malloc(sizeof(char *) * 4);
        if (!cr_argv) {
            perror("malloc(cr_argv)");
            exit(EXIT_FAILURE);
        }

        cr_argv[0] = malloc(sizeof(char) * (strlen(CR_RESTART_CMD) + 1));
        if (!cr_argv[0]) {
            perror("malloc(cr_argv[0])");
            exit(EXIT_FAILURE);
        }
        strcpy(cr_argv[0], CR_RESTART_CMD);
        cr_argv[1] = "--restore-pid";   // restore same pid on restart?

        cr_argv[2] = malloc(sizeof(char) * CR_MAX_FILENAME);
        if (!cr_argv[2]) {
            perror("malloc(cr_argv[2])");
            exit(EXIT_FAILURE);
        }

#ifdef CR_FTB
        if (cached_cr_mig_tgt) {
#ifdef CR_AGGRE
            if (use_aggre && use_aggre_mig) {   //use Fuse-mig-fs mnt point as ckpt_filename
                snprintf(cr_argv[2], CR_MAX_FILENAME, "%s.0.%d", crfs_mig_filename, migrank[i]);
                int tmpfd = open(cr_argv[2], O_RDWR | O_CREAT, 0644);
                close(tmpfd);
            } else              //simple strategy for migration
#endif
                snprintf(cr_argv[2], CR_MAX_FILENAME, "%s.0.%d", ckpt_filename, migrank[i]);

            rank = migrank[i];
        } else
#endif 
        {
            snprintf(str, 32, "MPISPAWN_MPIRUN_RANK_%d", i);
            rank = atoi(getenv(str));
            snprintf(cr_argv[2], CR_MAX_FILENAME, "%s.%d.%d", ckpt_filename, checkpoint_count, rank);
        }
        cr_argv[3] = NULL;
        PRINT_DEBUG(DEBUG_Fork_verbose > 1, "EXEC restart MPI proc command line: %s %s %s\n", cr_argv[0], cr_argv[1], cr_argv[2]);

        execvp(CR_RESTART_CMD, cr_argv);

        PRINT_ERROR_ERRNO("execvp() failed", errno);
        PRINT_DEBUG(DEBUG_Fork_verbose, "exit(EXIT_FAILURE)\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}


#ifdef CR_AGGRE
int init_ckpt_aggregation()
{
    char *str = getenv("MV2_CKPT_USE_AGGREGATION");
    if (str == NULL) {
        // Use default value, ie 1 since aggregation has been enabled at configure time
        use_aggre = 1;
    } else {
        // Use the value forwarded by mpirun_rsh in MV2_CKPT_USE_AGGREGATION
        use_aggre = atoi(str);
    }
    use_aggre_mig = env2int("MV2_CKPT_AGGRE_MIG_ROLE");
    PRINT_DEBUG(DEBUG_FT_verbose, "use-aggre=%d, use-aggre-mig=%d\n", use_aggre, use_aggre_mig);

    if (getenv("MV2_CKPT_FILE")) {
        strncpy(ckpt_filename, getenv("MV2_CKPT_FILE"), CR_MAX_FILENAME);
    } else {
        strncpy(ckpt_filename, DEFAULT_CHECKPOINT_FILENAME, CR_MAX_FILENAME);
    }

    if (use_aggre || use_aggre_mig) {
        // Check for fusermount
        int status = system("fusermount -V > /dev/null");
        if (status == -1) {
            if (errno == ECHILD) {
                // It looks like it means that:
                // The new process finished before system() could call waitpid() to wait for the child  process to end. 
                // This error can be ignored because the child process has already completed successfully and returned.
                // Nothing to do.
            } else {
                // The 'system' call failed (because of failed fork, missing sh, ...)
                // This is a serious error, aborting
                PRINT_ERROR_ERRNO("system() failed", errno);
                return -3;
            }
        } else {
            if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
                // Debug information
                if (WIFEXITED(status)) {
                    PRINT_DEBUG(DEBUG_FT_verbose, "'sh -c fusermount -V' exited with status %d\n", WEXITSTATUS(status));
                } else if (WIFSIGNALED(status)) {
                    PRINT_DEBUG(DEBUG_FT_verbose, "'sh -c fusermount -V' terminated with signal %d\n", WTERMSIG(status));
                } else {
                    PRINT_DEBUG(DEBUG_FT_verbose, "fail to execute 'sh -c fusermount -V' process for an unknown reason\n");
                }

                // Failed to run 'fusermount -V', disabling aggregation and RDMA migration
                PRINT_ERROR("Error: 'fusermount' not found.\n");
                return -1;
            }
        }
    }

    if (!use_aggre) {
        //strncpy(ckpt_filename, DEFAULT_CHECKPOINT_FILENAME, CR_MAX_FILENAME ); 
        return 0;
    }

    char *sessionid = getenv("MPISPAWN_CR_SESSIONID");
    if (start_crfs(sessionid, ckpt_filename, use_aggre_mig) != 0) {
        PRINT_ERROR("Error: start_crfs() failed. Please check that the fuse module is loaded on this node.\n");
        return -2;
    }
    PRINT_DEBUG(DEBUG_FT_verbose, "Now, ckptname is: %s\n", ckpt_filename);
    return 0;
}
#endif

#endif                          /* CKPT */
