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

/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "ibv_param.h"
#include "dreg.h"
#include "mpidimpl.h"
#include "mpichconf.h"
#include "mpidi_ch3_impl.h"

#if defined(CKPT)

#include <mpimem.h>
#include <ctype.h>
#include <sys/types.h>
#ifdef CR_FTB
#include <attr/xattr.h>
#endif
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <libcr.h>
#include "rdma_impl.h"
#include "upmi.h"
#include "cm.h"
#include "mem_hooks.h"
#include "mpiutil.h"
#include "error_handling.h"
#include "debug_utils.h"

#include "hwloc_bind.h"
#include "coll_shmem.h"

#ifdef ENABLE_SCR
#include "scr.h"
#endif

#ifdef CR_FTB

#include <libftb.h>

#define FTB_MAX_SUBSCRIPTION_STR 128
#define CR_FTB_EVENT_INFO {               \
        {"CR_FTB_CHECKPOINT",    "info"}, \
        {"CR_FTB_MIGRATE",       "info"}, \
        {"CR_FTB_MIGRATE_PIIC",  "info"}, \
        {"CR_FTB_CKPT_DONE",     "info"}, \
        {"CR_FTB_CKPT_FAIL",     "info"}, \
        {"CR_FTB_RSRT_DONE",     "info"}, \
        {"CR_FTB_RSRT_FAIL",     "info"}, \
        {"CR_FTB_APP_CKPT_REQ",  "info"}, \
        {"CR_FTB_CKPT_FINALIZE", "info"}, \
        {"CR_FTB_MIGRATE_PIC",   "info"}, \
        {"FTB_MIGRATE_TRIGGER",  "info"},  \
        {"MPI_PROCS_CKPTED", "info"},       \
        {"MPI_PROCS_CKPT_FAIL", "info"},    \
        {"MPI_PROCS_RESTARTED", "info"},    \
        {"MPI_PROCS_RESTART_FAIL", "info"}, \
        {"MPI_PROCS_MIGRATED", "info"},     \
        {"MPI_PROCS_MIGRATE_FAIL", "info"} \
}

/* event info table indices */
#define CR_FTB_CHECKPOINT    0
#define CR_FTB_MIGRATE       1
#define CR_FTB_MIGRATE_PIIC  2
#define CR_FTB_CKPT_DONE     3
#define CR_FTB_CKPT_FAIL     4
#define CR_FTB_RSRT_DONE     5
#define CR_FTB_RSRT_FAIL     6
#define CR_FTB_APP_CKPT_REQ  7
#define CR_FTB_CKPT_FINALIZE 8
#define CR_FTB_MIGRATE_PIC   9
#define FTB_MIGRATE_TRIGGER  10

/* standardized FTB-MPI events */
#define MPI_PROCS_CKPTED        11
#define MPI_PROCS_CKPT_FAIL     12
#define MPI_PROCS_RESTARTED     13
#define MPI_PROCS_RESTART_FAIL  14
#define MPI_PROCS_MIGRATED      15
#define MPI_PROCS_MIGRATE_FAIL  16

#define CR_FTB_EVENTS_MAX       17

/* Type of event to publish */
#define FTB_EVENT_NORMAL   1
#define FTB_EVENT_RESPONSE 2

/* Macro to initialize the event property structure */
#define SET_EVENT(_eProp, _etype, _payload...)             \
do {                                                       \
    _eProp.event_type = _etype;                            \
    snprintf(_eProp.event_payload, FTB_MAX_PAYLOAD_DATA,   \
                _payload);                                 \
} while(0)

/* Macro to pick an CR_FTB event */
#define EVENT(n) (cr_ftb_events[n].event_name)

static FTB_client_t ftb_cinfo;
static FTB_client_handle_t ftb_handle;
static FTB_event_info_t cr_ftb_events[] = CR_FTB_EVENT_INFO;
static FTB_subscribe_handle_t shandle;
static int ftb_init_done;

static pthread_cond_t cr_ftb_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t cr_ftb_mutex = PTHREAD_MUTEX_INITIALIZER;
static int cr_ftb_ckpt_req;
static int cr_ftb_mig_req;
static int cr_ftb_ckpt_finalize;

static char cr_mig_src_host[32];
static char cr_mig_tgt_host[32];

static int CR_FTB_Init(int, char *);
static void CR_FTB_Finalize();
static int CR_FTB_Callback(FTB_receive_event_t *, void *);
static int get_src_tgt(char *, char *, char *);

#define CRU_MAX_VAL_LEN 64

#else

/*Connection info to MPD*/
#define CR_RSRT_PORT_CHANGE 16
#define MAX_CR_MSG_LEN 256
#define CRU_MAX_KEY_LEN 64
#define CRU_MAX_VAL_LEN 64

struct CRU_keyval_pairs {
    char key[CRU_MAX_KEY_LEN];
    char value[CRU_MAX_VAL_LEN];
};

static struct CRU_keyval_pairs CRU_keyval_tab[64] = { {{0}} };

static int CRU_keyval_tab_idx = 0;
static int MPICR_MPD_fd = -1;

#endif                          /* CR_FTB */

static int MPICR_MPD_port;

static cr_callback_id_t MPICR_callback_id;
static cr_client_id_t MPICR_cli_id;
static pthread_t MPICR_child_thread;
static MPIDI_PG_t *MPICR_pg = NULL;
static int MPICR_pg_size = -1;
static int MPICR_pg_rank = -1;
static int MPICR_is_initialized = 0;
static int MPICR_is_restarting = 0;
static int checkpoint_count = 0;
static int restart_count = 0;
static int MPICR_max_save_ckpts = 0;
volatile int MPICR_callback_fin = 0;

extern int mv2_init_call_once;
extern int mv2_mmap_coll_once;
extern int mv2_enable_shmem_collectives;
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
extern shmem_info_t *ckpt_free_head;
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

static pthread_mutex_t MPICR_SMC_lock;
static pthread_cond_t MPICR_SMC_cond = PTHREAD_COND_INITIALIZER;
int g_cr_in_progress;

static int lock1_count = 0;
static int unlock1_count = 0;
static int lock2_count = 0;
static int unlock2_count = 0;

static int use_aggre_mig = 0;

#ifdef CR_FTB
static inline int set_event(FTB_event_properties_t * ep, int etype, int rank)
{
    char buf[32];
    snprintf(buf, 32, "%d", rank);
    buf[31] = 0;
    ep->event_type = etype;
    snprintf(ep->event_payload, FTB_MAX_PAYLOAD_DATA, buf);
    return 0;
}
#endif

static inline void MPIDI_CH3I_SMC_lock()
{
    pthread_mutex_lock(&MPICR_SMC_lock);
    g_cr_in_progress = 1;
    pthread_mutex_unlock(&MPICR_SMC_lock);
    lock1_count++;
}

static inline void MPIDI_CH3I_SMC_unlock()
{
    pthread_mutex_lock(&MPICR_SMC_lock);
    g_cr_in_progress = 0;
    pthread_cond_signal(&MPICR_SMC_cond);
    pthread_mutex_unlock(&MPICR_SMC_lock);
    unlock1_count++;
}

void Wait_for_CR_Completion()
{
    pthread_mutex_lock(&MPICR_SMC_lock);
    if (g_cr_in_progress) {
        MPIDI_CH3I_CR_unlock();
        while (g_cr_in_progress)
            pthread_cond_wait(&MPICR_SMC_cond, &MPICR_SMC_lock);
        MPIDI_CH3I_CR_lock();
    }
    pthread_mutex_unlock(&MPICR_SMC_lock);
}

static inline int MPIDI_SMC_CR_Init()
{
    pthread_mutexattr_t attr;
    g_cr_in_progress = 0;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);

    if (pthread_mutex_init(&MPICR_SMC_lock, &attr)) {
        return (1);
    }
    return (0);
}

static inline int MPIDI_SMC_CR_Finalize()
{
    pthread_cond_destroy(&MPICR_SMC_cond);
    pthread_mutex_destroy(&MPICR_SMC_lock);
    g_cr_in_progress = 0;
    return (0);
}

volatile MPICR_cr_state MPICR_state = MPICR_STATE_ERROR;
static pthread_rwlock_t MPICR_cs_lock;

#define CR_ERR_ABORT(args...)  do {                                                             \
    fprintf(stderr, "[Rank %d][%s:%d] ", MPIDI_Process.my_pg_rank ,__FILE__, __LINE__);         \
    fprintf(stderr, args);                                                                      \
    exit(-1);                                                                                   \
}while(0)

#define CR_ERR(args...)  do {                                                                   \
    fprintf(stderr, "[Rank %d][%s:%d] ", MPIDI_Process.my_pg_rank ,__FILE__, __LINE__);         \
    fprintf(stderr, args);                                                                      \
}while(0)

typedef struct MPICR_remote_update_msg {
    void *recv_mem_addr;
    uint32_t recv_buf_rkey[MAX_NUM_HCAS];
} MPICR_remote_update_msg;

int CR_IBU_Reactivate_channels();
int CR_IBU_Suspend_channels();

#ifndef CR_FTB
int CR_MPDU_readline(int fd, char *buf, int maxlen);
int CR_MPDU_writeline(int fd, char *buf);
int CR_MPDU_parse_keyvals(char *st);
char *CR_MPDU_getval(const char *keystr, char *valstr, int vallen);

int CR_MPDU_connect_MPD()
{
    int optval = 1;
    struct sockaddr_in sa;
    int fd, val;
    char session_file[32];

    MPICR_MPD_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (MPICR_MPD_fd < 0) {
        return -1;
    }

    struct hostent *hp = gethostbyname("localhost");
    bzero((void *) &sa, sizeof(sa));
    bcopy((void *) hp->h_addr, (void *) &sa.sin_addr, hp->h_length);
    sa.sin_family = AF_INET;

    if (using_mpirun_rsh) {
        snprintf(session_file, 32, "/tmp/cr.session.%s", getenv("MV2_CKPT_SESSIONID"));
        fd = open(session_file, O_RDWR);
        if (fd < 0) {
            perror("open");
            CR_ERR_ABORT("Could not get the Session Id\n");
        }
        if (read(fd, &val, sizeof(val)) < 0) {
            perror("read");
            close(fd);
            CR_ERR_ABORT("Could not read the Spawn Port\n");
        }
        close(fd);
        sa.sin_port = htons(val);
    } else {
        /* Legacy Mode */
        sa.sin_port = htons(MPICR_MPD_port + MPICR_pg_rank + restart_count * CR_RSRT_PORT_CHANGE);
    }

    if (setsockopt(MPICR_MPD_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &optval, sizeof(optval))) {
        CR_ERR_ABORT("setsockopt failed\n");
    }

    if (connect(MPICR_MPD_fd, (struct sockaddr *) &sa, sizeof(sa)) < 0) {
        CR_ERR_ABORT("connect %d failed\n", MPICR_MPD_port + MPICR_pg_rank + restart_count * CR_RSRT_PORT_CHANGE);
    }

    return 0;
}

#endif                          /* !CR_FTB */

void CR_MPDU_Ckpt_succeed()
{
#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    if (!cr_ftb_ckpt_req)
        return;
    set_event(&eprop, FTB_EVENT_NORMAL, MPICR_pg_rank);
    ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_CKPTED), &eprop, &ehandle);
    if (ret != FTB_SUCCESS)
        fprintf(stdout, "CR_MPDU_Ckpt_succeed() failed\n");
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    sprintf(cr_msg_buf, "cmd=ckpt_rep result=succeed\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
#endif
}

void CR_MPDU_Ckpt_fail()
{
#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    set_event(&eprop, FTB_EVENT_NORMAL, MPICR_pg_rank);
    ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_CKPT_FAIL), &eprop, &ehandle);
    if (ret != FTB_SUCCESS)
        fprintf(stdout, "CR_MPDU_Ckpt_fail() failed\n");
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    sprintf(cr_msg_buf, "cmd=ckpt_rep result=fail\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
#endif
}

void CR_MPDU_Rsrt_succeed()
{
#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    set_event(&eprop, FTB_EVENT_NORMAL, MPICR_pg_rank);
    if (cr_ftb_mig_req)
        ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_MIGRATED), &eprop, &ehandle);
    else
        ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_RESTARTED), &eprop, &ehandle);
    if (ret != FTB_SUCCESS)
        fprintf(stdout, "CR_MPDU_Rsrt_succeed() failed\n");
    else
        PRINT_DEBUG(DEBUG_CR_verbose, "CR_MPDU_Rsrt_succeed() \n");
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    sprintf(cr_msg_buf, "cmd=rsrt_rep result=succeed\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
#endif
}

void CR_MPDU_Rsrt_fail()
{
#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    set_event(&eprop, FTB_EVENT_NORMAL, MPICR_pg_rank);
    if (cr_ftb_mig_req)
        ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_MIGRATE_FAIL), &eprop, &ehandle);
    else
        ret = FTB_Publish(ftb_handle, EVENT(MPI_PROCS_RESTART_FAIL), &eprop, &ehandle);
    if (ret != FTB_SUCCESS)
        fprintf(stdout, "CR_MPDU_Rsrt_fail() failed\n");
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    sprintf(cr_msg_buf, "cmd=rsrt_rep result=fail\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
#endif
}

#ifndef CR_FTB
int CR_MPDU_Reset_PMI_port()
{
    char cr_msg_buf[MAX_CR_MSG_LEN];
    char value_str[CRU_MAX_VAL_LEN];

    /*Get pmi port number */
    sprintf(cr_msg_buf, "cmd=query_pmi_port\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
    CR_MPDU_readline(MPICR_MPD_fd, cr_msg_buf, MAX_CR_MSG_LEN);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"received msg from MPD: %s\n", cr_msg_buf);
    CR_MPDU_parse_keyvals(cr_msg_buf);
    CR_MPDU_getval("cmd", value_str, CRU_MAX_VAL_LEN);

    if (strcmp(value_str, "reply_pmi_port") != 0) {
        PRINT_DEBUG(DEBUG_CR_verbose,"fail to get pmi_port...\n");
        return -1;
    }

    CR_MPDU_getval("val", value_str, CRU_MAX_VAL_LEN);
    setenv("PMI_PORT", value_str, 1);
    return 0;
}

#endif                          /* CR_FTB */

/* the thread holding the WR_CR-lock */
static pthread_t wr_CR_lock_tid = 0;

/* CR lock to protect upper layers from accessing communication channel */
void MPIDI_CH3I_CR_lock()
{
    /*
     * If the current thread has already acquired the wrlock,
     * don't bother acquiring the rdlock.
     */
    if (MPICR_cs_lock.__data.__writer == syscall(SYS_gettid))
        return;

    pthread_rwlock_rdlock(&MPICR_cs_lock);
}

void MPIDI_CH3I_CR_unlock()
{
    /*
     * If the current thread has already acquired the wrlock,
     * you did not acquire the reader lock. So don't bother to
     * release the rdlock.
     */
    if (MPICR_cs_lock.__data.__writer == syscall(SYS_gettid))
        return;

    pthread_rwlock_unlock(&MPICR_cs_lock);
}

MPICR_cr_state MPIDI_CH3I_CR_Get_state()
{
    return MPICR_state;
}

int CR_Set_state(MPICR_cr_state state)
{
    switch (state) {
    case MPICR_STATE_CHECKPOINTING:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_CHECKPOINTING\n");
        break;
    case MPICR_STATE_ERROR:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_ERROR\n");
        break;
    case MPICR_STATE_POST_COORDINATION:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_POST_COORDINATION\n");
        break;
    case MPICR_STATE_PRE_COORDINATION:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_PRE_COORDINATION\n");
        break;
    case MPICR_STATE_REQUESTED:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_REQUESTED\n");
        break;
    case MPICR_STATE_RESTARTING:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_RESTARTING\n");
        MPICR_is_restarting = 1;
        break;
    case MPICR_STATE_RUNNING:
        PRINT_DEBUG(DEBUG_CR_verbose, "MPICR_STATE_RUNNING\n");
        break;
    default:
        PRINT_DEBUG(DEBUG_CR_verbose, "Unknown state\n");
        return -1;
    }

    MPICR_state = state;
    return 0;
}

#if defined(CR_FTB)
static int connect_to_migration_target(char *tgt)
{
    int ret;
    struct stat sbuf;
    char lockfile[128];
    char syncfile[128];
    snprintf(lockfile, 127, "/tmp/cr-%s-mig", getenv("MV2_CKPT_SESSIONID"));
    snprintf(syncfile, 127, "/tmp/cr-%s-mig-begin", getenv("MV2_CKPT_SESSIONID"));

    int fd = open(lockfile, O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd < 0) /* another proc (a leader) has already create the file */
    {
      /* wait for the leader proc to complete connection */
      wait_for_begin:
        ret = stat(syncfile, &sbuf);
        while (ret != 0) {
            if (errno == ENOENT) {  /* wait a while, then try again */
                usleep(20000);      /* wait for the leader proc to finish */
            } else {      
                PRINT_DEBUG(DEBUG_MIG_verbose > 1,"%s: wait for sync-file %s: blocked...\n", __func__, syncfile);
                usleep(20000);      /* wait for leader proc */
            }
            ret = stat(syncfile, &sbuf);
        }
        PRINT_DEBUG(DEBUG_MIG_verbose,"Mig src: can start proc-mig now...\n");
        return 0;
    }
    PRINT_DEBUG(DEBUG_MIG_verbose > 1,"has created lockfile: %s\n", lockfile);
    close(fd);
    unlink(syncfile);

    /* connect to target */
    char migmnt[128];
    int run = 1;
    snprintf(migmnt, 127, "/tmp/cr-%s/mig/", getenv("MV2_CKPT_SESSIONID"));
    PRINT_DEBUG(DEBUG_MIG_verbose > 1,"will set xattr at: %s\n", migmnt);
    ret = lsetxattr(migmnt, "migration.tgt", tgt, strlen(tgt), 0);
    ret = lsetxattr(migmnt, "migration.state", &run, sizeof(int), 0);
    usleep(10000);  /* wait for connection to be established */
    goto wait_for_begin;

    return 0;
}
#endif

int CR_Thread_loop()
{

#ifdef ENABLE_SCR
    if ( MPICR_pg_rank == 0 )
        PRINT_DEBUG(DEBUG_CR_verbose,"SCR has been enabled!\n");
    int use_scr=0;
#endif

#ifdef CR_FTB
    char my_hostname[256];
#else
    char cr_msg_buf[MAX_CR_MSG_LEN];
    fd_set set;
#endif
    char valstr[CRU_MAX_VAL_LEN];
    cr_checkpoint_handle_t cr_handle;
    cr_checkpoint_args_t cr_args;
    char *no_sync_str = getenv("MV2_CKPT_NO_SYNC");
    int no_sync = ((no_sync_str != NULL) ? atoi(no_sync_str) : 0);
    PRINT_DEBUG(DEBUG_CR_verbose, "CR_Thread_loop:\n");
    char *aggre_mig_file = getenv("MV2_CKPT_AGGRE_MIG_FILE");
    if (aggre_mig_file) {
        use_aggre_mig = 1;
    } else
        use_aggre_mig = 0;
    PRINT_DEBUG(DEBUG_CR_verbose > 1, "ckpt-file=%s, aggre-mig-file=%s\n", getenv("MV2_CKPT_FILE"), aggre_mig_file);
    while (1) {
        int ret = 0;
#ifdef CR_FTB
        pthread_mutex_lock(&cr_ftb_mutex);
        cr_ftb_ckpt_req = cr_ftb_mig_req = 0;
        while (!cr_ftb_ckpt_finalize && !cr_ftb_ckpt_req && !cr_ftb_mig_req)
            pthread_cond_wait(&cr_ftb_cond, &cr_ftb_mutex);
        pthread_mutex_unlock(&cr_ftb_mutex);

        if (cr_ftb_ckpt_finalize)
            return 0;

        if (!cr_ftb_ckpt_req && !cr_ftb_mig_req) {
            continue;
        }
        /* The FTB Callback thread will set cr_ftb_ckpt_req */
        if (cr_ftb_ckpt_req || cr_ftb_mig_req)
#else
        FD_ZERO(&set);
        FD_SET(MPICR_MPD_fd, &set);

        if (select(MPICR_MPD_fd + 1, &set, NULL, NULL, NULL) < 0) {
            CR_ERR_ABORT("CR_Thread_loop:select failed\n");
        }

        CR_MPDU_readline(MPICR_MPD_fd, cr_msg_buf, MAX_CR_MSG_LEN);
        PRINT_DEBUG(DEBUG_CR_verbose, "CR_Thread_loop:Got request from mpirun_rsh %s\n", cr_msg_buf);
        CR_MPDU_parse_keyvals(cr_msg_buf);
        CR_MPDU_getval("cmd", valstr, CRU_MAX_VAL_LEN);

        if (strcmp(valstr, "ckpt_req") == 0)
#endif                          /* CR_FTB */

        {
            /* begin phase 1: Suspend */
            char cr_file[CRU_MAX_VAL_LEN];
            int cr_file_fd;
#ifdef CR_FTB
            if (cr_ftb_ckpt_req)
#endif
                ++checkpoint_count;

#ifdef CR_FTB
            if (cr_ftb_mig_req && use_aggre_mig)
                /* use aggregation-based migration */
                strcpy(valstr, aggre_mig_file);
            else
                strcpy(valstr, getenv("MV2_CKPT_FILE"));
#else
            strcpy(valstr, getenv("MV2_CKPT_FILE"));
#endif

#ifdef CR_FTB
            if (cr_ftb_mig_req)
                PRINT_DEBUG(DEBUG_MIG_verbose > 1, "CR_Thread_loop:Got migration request %s\n", valstr);

            if (cr_ftb_ckpt_req)
                PRINT_DEBUG(DEBUG_CR_verbose, "Got checkpoint request %s\n", valstr);

            if (cr_ftb_mig_req)
                sprintf(cr_file, "%s.0.%d", valstr, MPICR_pg_rank);

            if (cr_ftb_ckpt_req)
#endif
                sprintf(cr_file, "%s.%d.%d", valstr, checkpoint_count, MPICR_pg_rank);
            PRINT_DEBUG(DEBUG_CR_verbose > 1,"cr_file= %s\n", cr_file);

#ifdef ENABLE_SCR
            if (0 == MPICR_pg_rank)
                fprintf(stderr,"Starting SCR checkpoint\n");
            SCR_Start_checkpoint();
            int scr_valid = 1;
            char scr_file[SCR_MAX_FILENAME];
            SCR_Route_file(cr_file, scr_file);
#endif


            CR_Set_state(MPICR_STATE_REQUESTED);
            /*
             * Let the shared memory collectives know that a checkpoint
             * has been requested
             */
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Request SMC lock\n");
            if (mv2_enable_shmem_collectives)
                MPIDI_CH3I_SMC_lock();
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Got SMC lock\n");

            pthread_rwlock_wrlock(&MPICR_cs_lock);
            wr_CR_lock_tid = pthread_self();
            lock2_count++;
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"locking CR MPICR_cs_lock: lock2=%d, unlock2=%d \n", lock2_count, unlock2_count);

            CR_Set_state(MPICR_STATE_PRE_COORDINATION);

            PRINT_DEBUG(DEBUG_CR_verbose, "CR_IBU_Suspend_channels()\n");
            if (CR_IBU_Suspend_channels()) {
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("CR_IBU_Suspend_channels failed\n");
            }

            PRINT_DEBUG(DEBUG_CR_verbose > 1,"locking CM\n");
            /* Lock will be finalized in suspension. */
            MPICM_lock();
            PRINT_DEBUG(DEBUG_CR_verbose > 1, "locked CM...\n");

            /* begin phase 2: checkpoint/migration */

#ifdef CR_FTB
            if (MPICR_pg_rank == 0) {
                if (cr_ftb_ckpt_req)
                    PRINT_DEBUG(DEBUG_CR_verbose, "Begin checkpoint...\n");
                else
                    PRINT_DEBUG(DEBUG_MIG_verbose, "Begin migration: source = %s, target = %s\n", cr_mig_src_host, cr_mig_tgt_host);
            }
#endif
            CR_Set_state(MPICR_STATE_CHECKPOINTING);

#ifdef CR_FTB
            gethostname(my_hostname, 255);
            PRINT_DEBUG(DEBUG_CR_verbose, "cr_mig_src_host = %s, cr_mig_tgt_host = %s \n", cr_mig_src_host, cr_mig_tgt_host);

            if (cr_ftb_mig_req && strstr(my_hostname, cr_mig_src_host) != my_hostname) {
                PRINT_DEBUG(DEBUG_MIG_verbose, "Skipping checkpoint\n");
                goto skip_checkpoint;
            }
            if (cr_ftb_mig_req && use_aggre_mig && strstr(my_hostname, cr_mig_src_host) == my_hostname) {
                /*I'm at mig-src, use aggre-based migration.
                   Need to connect to mig-target first */
                PRINT_DEBUG(DEBUG_MIG_verbose, "At %s: before mig: connect to tgt %s\n", cr_mig_src_host, cr_mig_tgt_host);
                connect_to_migration_target(cr_mig_tgt_host);
                PRINT_DEBUG(DEBUG_MIG_verbose,"%s:  Now, start mig data...\n", cr_mig_src_host);
            }
#endif

#ifdef ENABLE_SCR
            /* open the file given to us by SCR instead */
            fprintf(stderr,"[Rank %d] opening file %s..\n",MPICR_pg_rank,scr_file);
            cr_file_fd = open(scr_file, O_CREAT | O_WRONLY | O_TRUNC, 0666);

#else
            fprintf(stderr,"[Rank %d] opening file %s..\n",MPICR_pg_rank,cr_file);
            cr_file_fd = open(cr_file, O_CREAT | O_WRONLY | O_TRUNC, 0666);

#endif

            if (cr_file_fd < 0) {
                PRINT_ERROR_ERRNO("Failed to open file '%s'", errno, cr_file);
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("Checkpoint failed, aborting...\n");
            }

            MPICR_callback_fin = 0;

#ifdef ENABLE_SCR
            /* BLCR would be the component that gets control during a restart
             * and it will look for cr_file instead of scr_file. As a way to
             * work around this, a symbolic link between cr_file directory entry
             * and scr_file directory entry are created.
             */

            /*TODO: a smarter way to work around this*/
            ret = symlink(scr_file,cr_file); 
            if (ret < 0 ) {
                perror("symlink()");
                fprintf(stderr,"Can't link %s to %s\n",scr_file, cr_file);
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("Checkpoint aborting!\n");
                
            }
#endif

            ret = cr_initialize_checkpoint_args_t(&cr_args);
            if (ret < 0) {
                CR_ERR("BLCR call cr_initialize_checkpoint_args_t() failed\n");
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("Checkpoint failed, aborting...\n");
            }
            cr_args.cr_scope = CR_SCOPE_PROC;
            cr_args.cr_target = getpid();
            cr_args.cr_fd = cr_file_fd;
            cr_args.cr_signal = 0;
            cr_args.cr_timeout = 0;
            cr_args.cr_flags &= ~CR_CHKPT_DUMP_ALL; // Save None
#ifdef ENABLE_SCR
            PRINT_DEBUG( DEBUG_CR_verbose, "cr_request_checkpoint() with file '%s'\n", scr_file );
#else
            PRINT_DEBUG( DEBUG_CR_verbose, "cr_request_checkpoint() with file '%s'\n", cr_file );
#endif
            ret = cr_request_checkpoint(&cr_args, &cr_handle);
            PRINT_DEBUG( DEBUG_CR_verbose, "cr_request_checkpoint() returned %d\n", ret );
            if (ret < 0) {
                CR_ERR("BLCR call cr_request_checkpoint() failed with error %d: %s\n", errno, cr_strerror(errno));
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("Checkpoint failed, aborting...\n");
            }

            // Retry while interrupted
            PRINT_DEBUG( DEBUG_CR_verbose, "cr_poll_checkpoint()\n" );
            do {
                ret = cr_poll_checkpoint(&cr_handle, NULL);
            } while (ret == CR_POLL_CHKPT_ERR_PRE && errno == EINTR);
            PRINT_DEBUG( DEBUG_CR_verbose, "cr_poll_checkpoint() returned %d\n", ret );
            if (ret == CR_POLL_CHKPT_ERR_POST && errno == CR_ERESTARTED) { 
                // We are restarting, ignore this error code
                // Wait for the CR_Callback thread to return from cr_checkpoint()
            } else if (ret < 0) {
                CR_ERR("BLCR call cr_poll_checkpoint() failed with error %d: %s\n", errno, cr_strerror(errno));
                CR_MPDU_Ckpt_fail();
                CR_ERR_ABORT("Checkpoint failed, aborting...\n");
            } else if (ret == 0) {
                // 0 means that the checkpoint is in progress
                // It should never happen because we don't specify any timeout when calling cr_poll_checkpoint()
                CR_ERR_ABORT("Bad assertion\n");
            }

            PRINT_DEBUG(DEBUG_CR_verbose,"cr_request_fd\n");

            while (MPICR_callback_fin == 0) ;

            /*
               PRINT_DEBUG(DEBUG_CR_verbose > 1,"checkpointing, wait for callback to finish\n");
               pthread_mutex_lock(&MPICR_cond_callback_lock);
               pthread_cond_wait(&MPICR_cond_callback, &MPICR_cond_callback_lock);
               pthread_mutex_unlock(&MPICR_cond_callback_lock);
             */

            if (!MPICR_is_restarting) {
                if (no_sync == 0) {
                    PRINT_DEBUG(DEBUG_CR_verbose > 1,"Syncing checkpoint file to disk\n");
                    ret = fsync(cr_file_fd);
                    if (ret < 0) {
                        perror("fsync() failed!");
                        CR_MPDU_Ckpt_fail();
                        CR_ERR_ABORT("Checkpoint failed, aborting...\n");
                    }
                    PRINT_DEBUG(DEBUG_CR_verbose > 1,"fsync done\n");
                }

                ret = close(cr_file_fd);
                if (ret < 0) {
                    PRINT_ERROR_ERRNO("Failed to close file '%s'", errno, cr_file);
                    CR_MPDU_Ckpt_fail();
                    CR_ERR_ABORT("Checkpoint failed, aborting...\n");
                }
                PRINT_DEBUG(DEBUG_CR_verbose > 1,"close cr-file %s: ret=%d\n", cr_file, ret);
            }

            /* end of phase 2 */

#ifdef CR_FTB
            gethostname(my_hostname, 255);
            if (cr_ftb_mig_req && strstr(my_hostname, cr_mig_src_host) == my_hostname) {    // procs on src nodes, will exit soon
                PRINT_DEBUG(DEBUG_MIG_verbose > 1,"host=%s:rank %d: Src node exit\n", my_hostname, MPICR_pg_rank);
                PRINT_DEBUG(DEBUG_MIG_verbose, "Rank %d (pid=%d) on the source node is migrated -> exit\n", MPICR_pg_rank, getpid());
                exit(0);
            }
            // Update error prefix with new hostname
            {
                const int MAX_LENGTH = 256;
                char hostname[MAX_LENGTH];
                gethostname(hostname, MAX_LENGTH);
                hostname[MAX_LENGTH - 1] = '\0';
                char output_prefix[MAX_LENGTH];
                snprintf(output_prefix, MAX_LENGTH, "%s:mpi_rank_%i", hostname, MPICR_pg_rank);
                set_output_prefix(output_prefix);
            }

            PRINT_DEBUG(DEBUG_MIG_verbose,"Resuming process %d in node %s\n", MPICR_pg_rank, my_hostname);
            PRINT_DEBUG(DEBUG_MIG_verbose > 1, "Rank %d (pid=%d) is restarted on the target node\n", MPICR_pg_rank, getpid());
          skip_checkpoint:
#endif

            CR_Set_state(MPICR_STATE_POST_COORDINATION);

#ifdef CR_FTB
            if (cr_ftb_mig_req) {
                PRINT_DEBUG(DEBUG_MIG_verbose > 1,"Entering migration barrier\n");
                if (UPMI_BARRIER() != 0)
                    fprintf(stderr, "At Migration: rank %d, UPMI_BARRIER failed\n", MPICR_pg_rank);
            }
            /* end of phase 3 */

            if (cr_ftb_mig_req && strstr(my_hostname, cr_mig_tgt_host) == my_hostname) {    // this proc is migrated from src-node
                gethostname(my_hostname, 255);
                PRINT_DEBUG(DEBUG_MIG_verbose > 1,"\tHave mig [%d] to tgt-host=%s: Left migration-barrier.\n", MPICR_pg_rank, my_hostname);
            }
#endif
            PRINT_DEBUG(DEBUG_MIG_verbose,"Reactivating communication channels\n");

            if (CR_IBU_Reactivate_channels()) {
                if (MPICR_is_restarting) {
                    CR_MPDU_Rsrt_fail();
                } else {
                    CR_MPDU_Ckpt_fail();
                }
#ifdef CR_FTB
                printf("host %s: fail to reactivate channel\n", my_hostname);
                CR_ERR_ABORT("CR_IBU_Reactivate_channels failed\n");
#endif
            }

            if (MPICR_is_restarting) {
                CR_MPDU_Rsrt_succeed();
            } else {
                CR_MPDU_Ckpt_succeed();

                if (MPICR_max_save_ckpts > 0 && MPICR_max_save_ckpts < checkpoint_count) {
                    /*remove the ealier checkpoints */
                    sprintf(cr_file, "%s.%d.%d", valstr, checkpoint_count - MPICR_max_save_ckpts, MPICR_pg_rank);
                    unlink(cr_file);
                }
            }

            PRINT_DEBUG(DEBUG_CR_verbose > 1, "unlocking CR MPICR_cs_lock\n");
            pthread_rwlock_unlock(&MPICR_cs_lock);
            wr_CR_lock_tid = 0;
            unlock2_count++;
            /*
             * Let the shared memory collectives know that the checkpoint
             * request has completed
             */
            if (mv2_enable_shmem_collectives)
                MPIDI_CH3I_SMC_unlock();

            CR_Set_state(MPICR_STATE_RUNNING);

            if (MPIDI_Process.use_sync_ckpt) {
                pthread_mutex_lock(&MVAPICH2_sync_ckpt_lock);
                pthread_cond_signal(&MVAPICH2_sync_ckpt_cond);
                pthread_mutex_unlock(&MVAPICH2_sync_ckpt_lock);
            }

            /* end of phase 4 */

#ifdef ENABLE_SCR
            SCR_Complete_checkpoint(scr_valid);
            if ( 0 == MPICR_pg_rank )
                PRINT_DEBUG(DEBUG_CR_verbose, "Completed SCR checkpoint\n");
#endif

            MPICR_is_restarting = 0;
            if (MPICR_pg_rank == 0)
                PRINT_DEBUG(DEBUG_CR_verbose, "[%d]: CR completed.\n", MPICR_pg_rank);

        }                       // finished a ckpt-req
        else {
#ifdef CR_FTB
            PRINT_ERROR("Internal Error: Unknown command\n");
#else
            PRINT_ERROR("Internal Error: Unknown command: msg = '%s', cmd = '%s'\n", cr_msg_buf, valstr);
#endif
            abort();
        }
    }
}

int CR_Reset_proc_info()
{
    int has_parent;
    int pg_id_sz;
    int kvs_name_sz;
    MPIU_Free(MPIDI_Process.my_pg->id);
    MPIU_Free(MPIDI_Process.my_pg->ch.kvs_name);
    unsetenv("PMI_FD");

    PRINT_DEBUG(DEBUG_CR_verbose > 2,"unset PMI_FD\n");

    if (UPMI_INIT(&has_parent)) {
        CR_ERR_ABORT("PMI_Init failed\n");
    }

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"PMI_Init, has_parent=%d\n", has_parent);

    if (UPMI_KVS_GET_NAME_LENGTH_MAX(&pg_id_sz)) {
        CR_ERR_ABORT("UPMI_KVS_GET_NAME_LENGTH_MAX failed\n");
    }

    PRINT_DEBUG(DEBUG_MIG_verbose > 1,"UPMI_KVS_GET_NAME_LENGTH_MAX = %d\n", pg_id_sz);
    MPIDI_Process.my_pg->id = MPIU_Malloc(pg_id_sz + 1);

    if (NULL == MPIDI_Process.my_pg->id) {
        CR_ERR_ABORT("MPIU_Malloc failed\n");
    }

    if (UPMI_KVS_GET_MY_NAME(MPIDI_Process.my_pg->id, pg_id_sz)) {
        CR_ERR_ABORT("UPMI_KVS_GET_MY_NAME failed\n");
    }
    PRINT_DEBUG(DEBUG_CR_verbose > 2,"get id=%p\n", MPIDI_Process.my_pg->id);
    if (UPMI_KVS_GET_NAME_LENGTH_MAX(&kvs_name_sz)) {
        CR_ERR_ABORT("UPMI_KVS_GET_NAME_LENGTH_MAX failed\n");
    }

    MPIDI_Process.my_pg->ch.kvs_name = MPIU_Malloc(kvs_name_sz + 1);

    if (NULL == MPIDI_Process.my_pg->id) {
        CR_ERR_ABORT("MPIU_Malloc failed\n");
    }

    if (UPMI_KVS_GET_MY_NAME(MPIDI_Process.my_pg->ch.kvs_name, kvs_name_sz)) {
        CR_ERR_ABORT("UPMI_KVS_GET_MY_NAME failed\n");
    }

    MPIDI_PG_InitConnKVS(MPIDI_Process.my_pg);
    return 0;
}

static int norsh_cr_callback(void *arg)
{
    ++checkpoint_count;
    CR_Set_state(MPICR_STATE_REQUESTED);

    /* lock shared-memory collectives */
    if (mv2_enable_shmem_collectives) {
        PRINT_DEBUG(DEBUG_CR_verbose > 2, "Locking the shmem collectives\n");
        MPIDI_CH3I_SMC_lock();
    }
    
    /* lock the critical-section */
    PRINT_DEBUG(DEBUG_CR_verbose > 2, "Locking the critical-section\n");
    pthread_rwlock_wrlock(&MPICR_cs_lock);
    wr_CR_lock_tid = pthread_self();
    lock2_count++;

    /* lock the channel-manager (the lock will be finalized during suspension) */
    PRINT_DEBUG(DEBUG_CR_verbose > 2, "Locking the channel manager\n");
    MPICM_lock();

    CR_Set_state(MPICR_STATE_PRE_COORDINATION);

    /* suspend the communication channels */
    PRINT_DEBUG(DEBUG_CR_verbose, "Suspending communication channels\n");
    if (CR_IBU_Suspend_channels())
        return -1;

    /* allow BLCR to proceed with the checkpoint (the launcher will poll for it) */
    CR_Set_state(MPICR_STATE_CHECKPOINTING);
    PRINT_DEBUG(DEBUG_CR_verbose, "Calling cr_checkpoint()\n");

    int rc = cr_checkpoint(CR_CHECKPOINT_READY);
    
    if (rc < 0) { /* checkpoint failed */
        PRINT_DEBUG(DEBUG_CR_verbose, "Checkpoint failed: (%d) %s\n", rc, cr_strerror(-rc));
        return -1;
    } else { /* process is continuing after checkpoint (or restarting from failure )*/
        PRINT_DEBUG(DEBUG_CR_verbose,"Reactivating the communication channels\n");
        if (CR_IBU_Reactivate_channels()) {
            PRINT_DEBUG(DEBUG_CR_verbose, "CR_IBU_Reactivate_channels failed!\n");
            return -1;
        }

        /* unlock shared-memory collectives */
        if (mv2_enable_shmem_collectives) {
            MPIDI_CH3I_SMC_unlock();
            PRINT_DEBUG(DEBUG_CR_verbose, "Unlocked shmem collectives!\n");
        }

    }

    /*unlock the critical-section*/
    PRINT_DEBUG(DEBUG_CR_verbose > 2, "Unlocking the CH3 critical section lock\n");
    pthread_rwlock_unlock(&MPICR_cs_lock);
    wr_CR_lock_tid = 0;
    unlock2_count++;

    CR_Set_state(MPICR_STATE_RUNNING);

    PRINT_DEBUG(DEBUG_CR_verbose > 2,"Exiting norsh_cr_callback\n");
    MPICR_callback_fin = 1;
    
    return 0;

}

static int CR_Callback(void *arg)
{

#ifdef CR_FTB
    CR_FTB_Finalize();
#endif

    int rc = cr_checkpoint(0);

    if (rc < 0) /* cr_checkpoint() failed */
    {
        CR_ERR("BLCR call cr_checkpoint() failed with error %d: %s\n", rc, cr_strerror(-rc));
        CR_MPDU_Ckpt_fail();
        CR_ERR_ABORT("Checkpoint failed, aborting...\n");
    } else if (rc > 0) /* the procs are restarting */
    {
        /*Build the pipe between mpdman and app procs */
        CR_Set_state(MPICR_STATE_RESTARTING);

#ifdef CR_FTB
        if (CR_FTB_Init(MPICR_pg_rank, getenv("MV2_CKPT_SESSIONID")) != 0) {
            fprintf(stderr, "%s: Rank[%d] CR_FTB_Init() Failed\n", __func__, MPICR_pg_rank);
        } else {
            PRINT_DEBUG(DEBUG_CR_verbose > 1, "%s: Rank[%d] CR_FTB_Init()done\n", __func__, MPICR_pg_rank);
        }
#endif

        ++restart_count;

#ifdef CR_FTB
        //cr_ftb_mig_req = 0;
#else
        if (CR_MPDU_connect_MPD()) {
            CR_ERR_ABORT("CR_MPDU_connect_MPD failed\n");
        }

        PRINT_DEBUG(DEBUG_CR_verbose, "connected to mpirun_rsh\n");

        if (CR_MPDU_Reset_PMI_port()) {
            CR_MPDU_Rsrt_fail();
            CR_ERR_ABORT("CR_MPDU_Reset_PMI_port failed\n");
        }

        PRINT_DEBUG(DEBUG_CR_verbose > 1, "PMI_port reset\n");
#endif                          /* !CR_FTB */

        if (CR_Reset_proc_info()) {
            CR_MPDU_Rsrt_fail();
            CR_ERR_ABORT("CR_Reset_proc_info failed\n");
        }

        PRINT_DEBUG(DEBUG_CR_verbose > 2,"proc info reset\n");
    }
#ifdef CR_FTB
    else if (rc == 0 && !cr_ftb_mig_req)    // normal ckpt
#else
    else if (rc == 0)           // normal ckpt
#endif
    {
#ifdef CR_FTB
        if (CR_FTB_Init(MPICR_pg_rank, getenv("MV2_CKPT_SESSIONID")) != 0)
            fprintf(stdout, "CR_FTB_Init() Failed\n");
#endif

    }

    PRINT_DEBUG(DEBUG_CR_verbose > 2,"Out CR_Callback\n");
    MPICR_callback_fin = 1;

    /* pthread_cond_signal(&MPICR_cond_callback); */
    return 0;
}

void *CR_Thread_entry(void *arg)
{
    checkpoint_count = restart_count = 0;
    MPICR_cli_id = cr_init();

    if (MPICR_cli_id < 0) {
        CR_ERR_ABORT("BLCR call cr_init() failed\n");
    }

    if (using_mpirun_rsh) {
        MPICR_callback_id = cr_register_callback(CR_Callback, NULL, CR_THREAD_CONTEXT);
        if (-1 == MPICR_callback_id) {
            CR_ERR_ABORT("BLCR call cr_register_callback() failed with error %d: %s\n", errno, cr_strerror(errno));
        }

        char *temp = getenv("MV2_CKPT_MPD_BASE_PORT");

        if (temp) {
            MPICR_MPD_port = atoi(temp);
        } else {
            CR_ERR_ABORT("MV2_CKPT_MPD_BASE_PORT is not set\n");
        }

        temp = getenv("MV2_CKPT_MAX_SAVE_CKPTS");

        if (temp) {
            MPICR_max_save_ckpts = atoi(temp);
            PRINT_DEBUG(DEBUG_CR_verbose > 1,"MV2_CKPT_MAX_SAVE_CKPTS  %s\n", temp);
        } else {
            MPICR_max_save_ckpts = 0;
        }

        #ifdef CR_FTB
        if (CR_FTB_Init(MPICR_pg_rank, getenv("MV2_CKPT_SESSIONID")) != 0)
            fprintf(stderr, "CR_FTB_Init() Failed\n");
        #else
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Connecting to MPD\n");

        if (CR_MPDU_connect_MPD()) {
            CR_ERR_ABORT("CR_MPDU_connect_MPD failed\n");
        }
        #endif /* CR_FTB */

        CR_Set_state(MPICR_STATE_RUNNING);
        MPICR_is_initialized = 1;
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Finish initialization, going to CR_Thread_loop\n");

        CR_Thread_loop();
    } else {
        MPICR_callback_id = cr_register_callback(norsh_cr_callback, NULL, CR_THREAD_CONTEXT);
        if (-1 == MPICR_callback_id) {
            CR_ERR_ABORT("BLCR call cr_register_callback() failed with error %d: %s\n", errno, cr_strerror(errno));
        }

        CR_Set_state(MPICR_STATE_RUNNING);
        MPICR_is_initialized = 1;
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Finished initialization for the non-mpirun_rsh case\n");
    }

    return NULL;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CR_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CR_Init(MPIDI_PG_t * pg, int rank, int size)
{
    MPICR_pg = pg;
    MPICR_pg_rank = rank;
    MPICR_pg_size = size;
    int mpi_errno = MPI_SUCCESS;

    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"Creating a new thread for running cr controller: MPICR_cs_lock\n");
    if (pthread_rwlock_init(&MPICR_cs_lock, &attr)) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s", "pthread_mutex_init", strerror(errno)
            );
    }

    pthread_rwlockattr_destroy(&attr);

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"Creating a new thread for running cr controller\n");

    if (pthread_create(&MPICR_child_thread, NULL, CR_Thread_entry, NULL)) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                                                            "%s: %s",
                                                            "pthread_create",
                                                            strerror(errno));
    }

    /* Initialize the shared memory collectives lock */
    if (mv2_enable_shmem_collectives) {
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Initializing SMC locks\n");
        MPIDI_SMC_CR_Init();
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Initialized SMC locks\n");
    }

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPIDI_CH3I_CR_Finalize()
{
    if (!MPICR_is_initialized) {
        return 0;
    }
#ifdef CR_FTB
    pthread_mutex_lock(&cr_ftb_mutex);
    cr_ftb_ckpt_finalize = 1;
    pthread_cond_signal(&cr_ftb_cond);
    pthread_mutex_unlock(&cr_ftb_mutex);
#endif

//    pthread_cancel(MPICR_child_thread);
//    pthread_join(MPICR_child_thread, NULL);
//    pthread_rwlock_destroy(&MPICR_cs_lock);

#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;

    if (ftb_init_done) {
        ftb_init_done = 0;

        SET_EVENT(eprop, FTB_EVENT_NORMAL, " ");
        PRINT_DEBUG(DEBUG_CR_verbose > 1, "\tCR_Finalize: FTB_publish() MPICR_cs_lock\n");
        ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_CKPT_FINALIZE), &eprop, &ehandle);
        if (ret != FTB_SUCCESS)
            fprintf(stderr, "MPIDI_CH3I_CR_Finalize() failed with %d\n", ret);
        FTB_Unsubscribe(&shandle);
        usleep(20000);
        PRINT_DEBUG(DEBUG_CR_verbose > 1, "\tCR_Finalize: done FTB_publish() MPICR_cs_lock\n");
        FTB_Disconnect(ftb_handle);
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"\tCR_Finalize: done FTB_Disconnect() MPICR_cs_lock\n");

    }
    pthread_cancel(MPICR_child_thread);
    pthread_join(MPICR_child_thread, NULL);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t destroy() MPICR_cs_lock\n");
    pthread_rwlock_destroy(&MPICR_cs_lock);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t done destroy() MPICR_cs_lock\n");
#else

    pthread_cancel(MPICR_child_thread);
    pthread_join(MPICR_child_thread, NULL);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t destroy() MPICR_cs_lock\n");
    pthread_rwlock_destroy(&MPICR_cs_lock);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t done destroy() MPICR_cs_lock\n");

    if ( using_mpirun_rsh && !MPICR_pg_rank) {
        char cr_msg_buf[MAX_CR_MSG_LEN];

        sprintf(cr_msg_buf, "cmd=finalize_ckpt\n");
        CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
    }

    if ( using_mpirun_rsh && MPICR_MPD_fd > 0) {
        close(MPICR_MPD_fd);
    }
#endif

    MPICR_is_initialized = 0;

    /* Uninitialize the shared memory collectives lock */
    if (mv2_enable_shmem_collectives) {
        MPIDI_SMC_CR_Finalize();
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Finalized SMC locks\n");
    }

    PRINT_DEBUG(DEBUG_CR_verbose,"\tCR_Finalize: success\n");
    return MPI_SUCCESS;
}

void MPIDI_CH3I_CR_Sync_ckpt_request()
{
    if (MPICR_state != MPICR_STATE_RUNNING) {
        /*Another checkpoint is going on */
        return;
    }
#ifdef CR_FTB
    int ret;
    FTB_event_properties_t eprop;
    FTB_event_handle_t ehandle;
    SET_EVENT(eprop, FTB_EVENT_NORMAL, " ");
    ret = FTB_Publish(ftb_handle, EVENT(CR_FTB_APP_CKPT_REQ), &eprop, &ehandle);
    if (ret != FTB_SUCCESS)
        fprintf(stderr, "MPIDI_CH3I_CR_Sync_ckpt_request() failed\n");
#else
    PRINT_DEBUG(DEBUG_CR_verbose,"Send ckpt request to MPD\n");

    char cr_msg_buf[MAX_CR_MSG_LEN];
    sprintf(cr_msg_buf, "cmd=app_ckpt_req\n");
    CR_MPDU_writeline(MPICR_MPD_fd, cr_msg_buf);
#endif
}

/* CR message handler in progress engine */
void MPIDI_CH3I_CR_Handle_recv(MPIDI_VC_t * vc, MPIDI_CH3_Pkt_type_t msg_type, vbuf * v)
{
    MPICR_remote_update_msg *msg = NULL;
    struct MPID_Request *sreq = NULL;
    int found = 0;
    PRINT_DEBUG(DEBUG_CR_verbose > 2,"%s: [%d <= %d]: got msg: %s(%d)\n", __func__, MPIDI_Process.my_pg_rank, vc->pg_rank, MPIDI_CH3_Pkt_type_to_string[msg_type], msg_type);
    switch (msg_type) {
    case MPIDI_CH3_PKT_CR_REMOTE_UPDATE:
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Received MPIDI_CH3_PKT_CR_REMOTE_UPDATE\n");

        msg = (MPICR_remote_update_msg *) (v->buffer + sizeof(MPIDI_CH3I_MRAILI_Pkt_comm_header));
        sreq = (struct MPID_Request *) vc->mrail.sreq_head;

        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Looking for address match %p\n", msg->recv_mem_addr);
        found = 0;

        while (sreq) {
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Verifying request %p\n", sreq);

            if (sreq->mrail.remote_addr == msg->recv_mem_addr) {
                PRINT_DEBUG(DEBUG_CR_verbose > 2,"Found a address match req: %p\n", sreq);

                /*FIXME: Is address match enough? */
                MPIU_Memcpy(sreq->mrail.rkey, msg->recv_buf_rkey, sizeof(uint32_t) * MAX_NUM_HCAS);
                found++;
                if (vc->mrail.sreq_to_update <= 0) {
                    PRINT_DEBUG(DEBUG_CR_verbose > 2,"[%d <== %d]: %s: Warn: REM_UPDATE: sreq-to-up=%d...\n", MPICR_pg_rank, vc->pg_rank, __func__, vc->mrail.sreq_to_update);
                    /// a spurious REM_UPDATE msg, not a real bug...
                    //MPIU_Assert( vc->mrail.sreq_to_update > 0 );// don't need to crash the prog
                } else
                    vc->mrail.sreq_to_update--; // has updated one sender' rndv(sreq)
                if (vc->mrail.sreq_to_update == 0) {
                    //reset rput_stop at ibv_chann_man
                    //vc->ch.rput_stop = 0; // all pending sender rndv have been updated
                }

                PRINT_DEBUG(DEBUG_CR_verbose > 2,"rkey updated hca0:%x\n", sreq->mrail.rkey[0]);
            }

            sreq = sreq->mrail.next_inflow;
        }
        if (!found) {
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"[%d <== %d]: %s: Warn: REM_UPDATE: no match found...\n", MPICR_pg_rank, vc->pg_rank, __func__);
        }
#if defined(MPIDI_MRAILI_COALESCE_ENABLED)
        v->content_consumed += sizeof(MPICR_remote_update_msg);
#endif                          /* defined(MPIDI_MRAILI_COALESCE_ENABLED) */
        break;
    default:
        CR_ERR_ABORT("unknown message type: %d\n", msg_type);
        break;
    }
}

void MPIDI_CH3I_CR_Handle_send_completion(MPIDI_VC_t * vc, MPIDI_CH3_Pkt_type_t msg_type, vbuf * v)
{
    switch (msg_type) {
    case MPIDI_CH3_PKT_CR_REMOTE_UPDATE:
        break;
    default:
        CR_ERR_ABORT("unknown message type: %d\n", msg_type);
        break;
    }
}

/*===========================*/
/*  IB management functions  */
/*===========================*/

/* The request involving memory registration (e.g. rndv recv). */
struct MPID_Request *MPICR_req_list_head = NULL;
struct MPID_Request *MPICR_req_list_tail = NULL;

int CR_IBU_Release_network()
{
    int i = 0;
    int rail_index;

    MPIDI_PG_t *pg = MPIDI_Process.my_pg;
    MPIDI_VC_t *vc = NULL;

    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_PG_Get_size(pg);
    if (MPICM_Finalize_UD() != 0) {
        CR_ERR_ABORT("MPICM_Finalize_UD failed\n");
    } else {
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"CR_IBU_Release_network: MPICM_Finalize_UD succeed.\n");
    }
    PRINT_DEBUG(DEBUG_CR_verbose > 2,"has fin UD, size=%d, rank=%d\n", pg_size, pg_rank);
    for (; i < pg_size; ++i) {
        if (i == pg_rank) {
            continue;
        }

        MPIDI_PG_Get_vc(pg, i, &vc);

        switch (vc->ch.state) {
        case MPIDI_CH3I_VC_STATE_SUSPENDED:
            break;
        case MPIDI_CH3I_VC_STATE_IDLE:
            CR_ERR_ABORT("Having active vc when releasing networks\n");
        default:
            continue;
        }

        /*
         * Don't try to destroy the QP when SMP is used.
         */
        if (!(SMP_INIT && (vc->smp.local_nodes >= 0))) {
            for (rail_index = 0; rail_index < vc->mrail.num_rails; ++rail_index) {
                ibv_destroy_qp(vc->mrail.rails[rail_index].qp_hndl);
            }
        }
#ifndef MV2_DISABLE_HEADER_CACHING
        if (vc->mrail.rfp.cached_incoming)
            MPIU_Free(vc->mrail.rfp.cached_incoming);
        if (vc->mrail.rfp.cached_outgoing)
            MPIU_Free(vc->mrail.rfp.cached_outgoing);
#endif                          /* !MV2_DISABLE_HEADER_CACHING */
    }
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"CR_IBU_Release_network: CH3I_RDMA.has_srq\n");
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        for (i = 0; i < rdma_num_hcas; ++i) {
            pthread_cond_destroy(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
            pthread_mutex_destroy(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_mutex_lock[i]);
            pthread_cancel(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i]);
            pthread_join(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i], NULL);

            if (ibv_destroy_srq(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i])) {
                ibv_error_abort(IBV_RETURN_ERR, "Couldn't destroy SRQ\n");
            }
        }
    }
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    /* Begin: Free memory registered by zcopy collectives */
    shmem_info_t * curr = ckpt_free_head;
    shmem_info_t * next = NULL;
    while (curr) {
        if(curr->local_rank == 0) {
            PRINT_DEBUG(DEBUG_CR_verbose > 1,"Freeing shmem region %p, reg = %d\n", curr, curr->buffer_registered);
            if(curr->buffer_registered == 1) {
                mv2_shm_coll_dereg_buffer(curr->mem_handle);
            }
            /* Reset pointers for zcopy bcast/reduce */
            curr->buffer_registered         = 0;
            curr->bcast_knomial_factor      = -1;
            curr->bcast_exchange_rdma_keys  = 1;
            curr->reduce_knomial_factor     = -1;
            curr->reduce_exchange_rdma_keys = 1;

        }
        next = curr->next;
        curr->next = NULL;
        curr = next;
    }
    ckpt_free_head = NULL;
    /* End: Free memory registered by zcopy collectives */
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

    /* NOTE: dreg_deregister_all MUST come after de-registering shmem region */
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"CR_IBU_Release_network: deregister_all\n");
    dreg_deregister_all();

    for (i = 0; i < rdma_num_hcas; ++i) {
        if (rdma_iwarp_use_multiple_cq && MV2_IS_CHELSIO_IWARP_CARD(mv2_MPIDI_CH3I_RDMA_Process.hca_type) && (mv2_MPIDI_CH3I_RDMA_Process.cluster_size != VERY_SMALL_CLUSTER)) {
            /* Trac #423 */
            ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i]);
            ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i]);
        } else {
            ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i]);
        }
        deallocate_vbufs(i);
        ibv_dealloc_pd(mv2_MPIDI_CH3I_RDMA_Process.ptag[i]);
        ibv_close_device(mv2_MPIDI_CH3I_RDMA_Process.nic_context[i]);
    }
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"CR_IBU_Release_network: ibv_close\n");
#if !defined(DISABLE_PTMALLOC)
    mvapich2_mfin();
#endif                          /* !defined(DISABLE_PTMALLOC) */

    /* Release the SMP channel */
    if (SMP_INIT) {
        if (MPIDI_CH3I_SMP_finalize()) {
            fprintf(stderr, "[%s:%d] MPIDI_CH3I_SMP_finalize() returned error\n", __FILE__, __LINE__);
            return (-1);
        }
    }
    if(mv2_enable_affinity == 1) { 
       hwloc_topology_destroy(topology);
    } 

    return 0;
}

/*
 * TODO finish adding error handling
 */
#undef FUNCNAME
#define FUNCNAME CR_IBU_Rebuild_network
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int CR_IBU_Rebuild_network()
{
    int mpi_errno = MPI_SUCCESS;
    char tmp_hname[256];
    gethostname(tmp_hname, 255);
    MPIDI_PG_t *pg = MPICR_pg;
    int pg_rank = MPICR_pg_rank;
    int pg_size = MPICR_pg_size;

    uint32_t *ud_qpn_all = (uint32_t *) MPIU_Malloc(pg_size * sizeof(uint32_t));
    uint16_t *lid_all = (uint16_t *) MPIU_Malloc(pg_size * sizeof(uint16_t));
    union ibv_gid *gid_all = (union ibv_gid *)
                                 MPIU_Malloc(pg_size * sizeof(union ibv_gid));
    rdma_num_rails = rdma_num_hcas * rdma_num_ports * rdma_num_qp_per_port;

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"num_qp_per_port %d, num_rails = %d\n", rdma_num_qp_per_port, rdma_num_rails);

    init_vbuf_lock();

    MPIDI_VC_t *vc = NULL;
    int i = 0;

    /*vc structure doesn't need to be initialized */
    for (; i < pg_size; ++i) {
        MPIDI_PG_Get_vc(pg, i, &vc);
        vc->mrail.num_rails = rdma_num_rails;
    }

    /* Open the device and create cq and qp's */
    if (rdma_open_hca(&mv2_MPIDI_CH3I_RDMA_Process)) {
        MPL_error_printf("rdma_open_hca failed\n");
        return -1;
    }

    mpi_errno = rdma_iba_hca_init_noqp(&mv2_MPIDI_CH3I_RDMA_Process, pg_rank, pg_size);
    if (mpi_errno) {
        MPL_error_printf("Failed to Initialize HCA type\n");
        MPIR_ERR_POP(mpi_errno);
    }

    dreg_reregister_all();
    vbuf_reregister_all();
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"Reregistered dreg and vbufs\n");

    /* Post the buffers for the SRQ */
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        pthread_spin_init(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_spin_lock, 0);
        pthread_spin_lock(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_spin_lock);

        struct ibv_srq_attr srq_attr;

        for (i = 0; i < rdma_num_hcas; ++i) {
            pthread_mutex_init(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_mutex_lock[i], 0);
            pthread_cond_init(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i], 0);

            mv2_MPIDI_CH3I_RDMA_Process.srq_zero_post_counter[i] = 0;
            mv2_MPIDI_CH3I_RDMA_Process.posted_bufs[i] = mv2_post_srq_buffers(mv2_srq_fill_size, i);

            srq_attr.max_wr = mv2_srq_fill_size;
            srq_attr.max_sge = 1;
            srq_attr.srq_limit = mv2_srq_limit;

            if (ibv_modify_srq(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i], &srq_attr, IBV_SRQ_LIMIT)) {
                ibv_error_abort(IBV_RETURN_ERR, "Couldn't modify SRQ limit\n");
            }

            /* Start the async thread which watches for SRQ limit events. */
            pthread_create(&mv2_MPIDI_CH3I_RDMA_Process.async_thread[i],
                                        NULL,
                                        (void *) async_thread,
                                        (void *) mv2_MPIDI_CH3I_RDMA_Process.nic_context[i]);
        }

        pthread_spin_unlock(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_spin_lock);
    }

    uint32_t ud_qpn_self;
    mpi_errno = MPICM_Init_UD_CM(&ud_qpn_self);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"init UD ret %d\n", mpi_errno);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (pg_size == 1) {
        ud_qpn_all[0] = ud_qpn_self;
        lid_all[0] = mv2_MPIDI_CH3I_RDMA_Process.lids[0][0];
        gid_all[0] = mv2_MPIDI_CH3I_RDMA_Process.gids[0][0];
    } else if (pg_size > 1) {
        if (mv2_MPIDI_CH3I_RDMA_Process.has_ring_startup) {
            ud_addr_info_t self_info;
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Ring-based exchange\n");

            char hostname[HOSTNAME_LEN + 1];

            /* Recalculate your hostid on restart */
            int rv = gethostname(hostname, HOSTNAME_LEN);

            if (rv != 0) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s", "Could not get hostname");
            }

            struct hostent *hostent = gethostbyname(hostname);
            int hostid = (int) ((struct in_addr *)
                                hostent->h_addr_list[0])->s_addr;
            self_info.hostid = hostid;

            memcpy(&self_info.lid, &mv2_MPIDI_CH3I_RDMA_Process.lids, sizeof(uint16_t) * MAX_NUM_HCAS * MAX_NUM_PORTS);
            memcpy(&self_info.gid, &mv2_MPIDI_CH3I_RDMA_Process.gids,
                   sizeof(union ibv_gid) * MAX_NUM_HCAS * MAX_NUM_PORTS);
            self_info.qpn = ud_qpn_self;

            ud_addr_info_t *all_info = (ud_addr_info_t *) MPIU_Malloc(sizeof(ud_addr_info_t) * pg_size);
            /*will be freed in rdma_iba_bootstrap_cleanup */
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Setting up ring-based exchange\n");
            rdma_setup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process, pg_rank, pg_size);

            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Ring-based all-gather exchange\n");
            mpi_errno = rdma_ring_based_allgather(&self_info, sizeof(self_info), pg_rank, all_info, pg_size, &mv2_MPIDI_CH3I_RDMA_Process);
            if (mpi_errno) {
                PRINT_DEBUG(DEBUG_CR_verbose > 2,"rdma-ring-based-allgather failed, ret=%d...\n", mpi_errno);
                MPIR_ERR_POP(mpi_errno);
            }
            for (i = 0; i < pg_size; ++i) {
                /* Use the recalculated hostids on restart */
                MPIDI_VC_t *vc;
                MPIDI_PG_Get_vc(pg, i, &vc);
                vc->smp.hostid = all_info[i].hostid;

                ud_qpn_all[i] = all_info[i].qpn;
                lid_all[i] = all_info[i].lid[0][0];
            }

            PRINT_DEBUG(DEBUG_CR_verbose > 2,"Cleaning up the startup ring\n");
            mpi_errno = rdma_cleanup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        } else {
            /* Exchange the information about HCA_lid and qp_num. */
            sprintf(mv2_pmi_key, "ud_info-%08d", pg_rank);
            if (!use_iboeth) {
                sprintf(mv2_pmi_val, "%08x:%08x", mv2_MPIDI_CH3I_RDMA_Process.lids[0][0],
                        ud_qpn_self);
            } else {
                sprintf(mv2_pmi_val, "%08x:%08x:%016"PRIx64":%016"PRIx64,
                       mv2_MPIDI_CH3I_RDMA_Process.lids[0][0], ud_qpn_self,
                       mv2_MPIDI_CH3I_RDMA_Process.gids[0][0].global.subnet_prefix,
                       mv2_MPIDI_CH3I_RDMA_Process.gids[0][0].global.interface_id);
            }


            if (UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val) != 0) {
                CR_ERR_ABORT("UPMI_KVS_PUT failed\n");
            }

            /* when running with slurm, srun can get overwhelmed with RPC
             * messages. This small delay (if needed) helps spread out the RPCs */
            if(!using_mpirun_rsh) {
                int slurm_pmi_delay = 0;
                char *envval;
                if ((envval = getenv("MV2_CKPT_SLURM_PMI_DELAY")) != NULL) {
                    slurm_pmi_delay = atoi(envval);
                }
                sleep(slurm_pmi_delay);
            }

            if (UPMI_KVS_COMMIT(pg->ch.kvs_name) != 0) {
                CR_ERR_ABORT("UPMI_KVS_COMMIT failed\n");
            }
            if (UPMI_BARRIER() != 0) {
                CR_ERR_ABORT("UPMI_BARRIER failed\n");
            }

            for (i = 0; i < pg_size; ++i) {
                if (pg_rank == i) {
                    lid_all[i] = mv2_MPIDI_CH3I_RDMA_Process.lids[0][0];
                    gid_all[i] = mv2_MPIDI_CH3I_RDMA_Process.gids[0][0];
                    ud_qpn_all[i] = ud_qpn_self;
                    continue;
                }

                sprintf(mv2_pmi_key, "ud_info-%08d", i);

                if (UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen) != 0) {
                    CR_ERR_ABORT("UPMI_KVS_GET failed\n");
                }

                if (!use_iboeth) {
                    sscanf(mv2_pmi_val, "%08hx:%08x", &(lid_all[i]), &(ud_qpn_all[i]));
                } else {
                    sscanf(mv2_pmi_val, "%08hx:%08x:%016"SCNx64":%016"SCNx64, &(lid_all[i]),
                           &(ud_qpn_all[i]), &(gid_all[i].global.subnet_prefix),
                           &(gid_all[i].global.interface_id));
                }
            }
        }
    }

    for (i = 0; i < pg_size; ++i) {
        pg->ch.mrail->cm_shmem.ud_cm[i].cm_lid = lid_all[i];
        memcpy(&pg->ch.mrail->cm_shmem.ud_cm[i].cm_gid, &gid_all[i], sizeof(union ibv_gid));
        pg->ch.mrail->cm_shmem.ud_cm[i].cm_ud_qpn = ud_qpn_all[i];
    }

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"Exchanging parameters done\n");

    PRINT_DEBUG(DEBUG_CR_verbose > 2,"MPICM_Init_UD_struct\n");
    mpi_errno = MPICM_Init_UD_struct(MPICR_pg);
    PRINT_DEBUG(DEBUG_CR_verbose > 2,"MPICM_Create_UD_threads\n");
    MPICM_Create_UD_threads();

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    PRINT_DEBUG(DEBUG_CR_verbose > 2,"Waiting in a UPMI_BARRIER\n");
    if (UPMI_BARRIER() != 0) {
        CR_ERR_ABORT("UPMI_BARRIER failed\n");
    }

    PRINT_DEBUG(DEBUG_CR_verbose,"CR_IBU_Rebuild_network finish\n");

    MPIU_Free(ud_qpn_all);
    MPIU_Free(lid_all);
    MPIU_Free(gid_all);

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int CR_IBU_Prep_remote_update()
{
    struct MPID_Request *temp = MPICR_req_list_head;

    /* Using this to reduce the number of update messages 
     * since all consecutive request to a same memory address 
     * will be combined to a single update message */
    void *last_mem_addr = NULL;

    MPIDI_VC_t *vc = NULL;
    vbuf *v = NULL;
    MPICR_remote_update_msg msg;
    MPIDI_CH3I_CR_msg_log_queue_entry_t *entry = NULL;
    MPIDI_CH3I_MRAILI_Pkt_comm_header *p = NULL;
    struct MPIDI_CH3_Pkt_rndv_clr_to_send *cts_header = NULL;

    /*each req->mrail.rkey is: */
    uint32_t rkey[MAX_NUM_HCAS];

    int i;

    while (temp != NULL) {
        if (temp->mrail.rndv_buf == last_mem_addr) {
            if (temp->mrail.d_entry->memhandle[0]->rkey != rkey[0]) {
                CR_ERR_ABORT("Same addr %p, different rkey %x, %x\n", temp->mrail.rndv_buf, temp->mrail.d_entry->memhandle[0]->rkey, rkey[0]
                    );
            }

            temp = temp->ch.cr_queue_next;
            continue;
        }

        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Found a new memory address %p registered\n", temp->mrail.rndv_buf);
        last_mem_addr = temp->mrail.rndv_buf;

        memset(rkey, 0, sizeof(uint32_t) * MAX_NUM_HCAS);
        vc = temp->ch.vc;

        for (i = 0; i < rdma_num_hcas; ++i) {
            rkey[i] = temp->mrail.d_entry->memhandle[i]->rkey;
#if !defined(NDEBUG)
            MPIU_Assert(vc->mrail.srp.credits[i].backlog.len == 0);
            MPIU_Assert(vc->mrail.rails[i].ext_sendq_head == NULL);
#endif                          /* !defined(NDEBUG) */
        }

        i = 0;
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Search msg log queue for CTS message with the address %p, in vc %p \n", temp->mrail.rndv_buf, vc);
        entry = vc->mrail.msg_log_queue_head;

        while (NULL != entry) {
            v = entry->buf;
            ++i;
            p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *) v->pheader;
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"In msg_log_Q, type %d\n", p->type);

            if (p->type == MPIDI_CH3_PKT_RNDV_CLR_TO_SEND) {
                cts_header = (struct MPIDI_CH3_Pkt_rndv_clr_to_send *) p;
                PRINT_DEBUG(DEBUG_CR_verbose > 2,"CTS in msg_log_Q addr %p\n", cts_header->rndv.buf_addr);

                if (cts_header->rndv.buf_addr == temp->mrail.rndv_buf) {
                    PRINT_DEBUG(DEBUG_CR_verbose > 2,"Found a match in local pending cts\n");
                    MPIU_Memcpy(cts_header->rndv.rkey, rkey, sizeof(uint32_t) * MAX_NUM_HCAS);
                }
            }

            entry = entry->next;
        }

        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Searched %d packets in msg_log_Q\n", i);
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Checking CM queue for vc %p\n", vc);

        /* Check cm_sendq for all cts packets. */
        if (!MPIDI_CH3I_CM_SendQ_empty(vc)) {
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"cm_send_Q is not empty\n");
            struct MPID_Request *req = MPIDI_CH3I_CM_SendQ_head(vc);
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"req %p\n", req);

            i = 0;

            while (req != NULL) {
                MPIDI_CH3_Pkt_t *upkt = (MPIDI_CH3_Pkt_t *) & (req->dev.pending_pkt);
                ++i;
                PRINT_DEBUG(DEBUG_CR_verbose > 2,"In cm_send_Q, type %d\n", upkt->type);

                if (upkt->type == MPIDI_CH3_PKT_RNDV_CLR_TO_SEND) {
                    cts_header = &(upkt->rndv_clr_to_send);
                    PRINT_DEBUG(DEBUG_CR_verbose > 2,"CTS in cm_send_Q addr %p\n", cts_header->rndv.buf_addr);

                    if (cts_header->rndv.buf_addr == temp->mrail.rndv_buf) {
                        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Found a match in local cm queue\n");
                        MPIU_Memcpy(cts_header->rndv.rkey, rkey, sizeof(uint32_t) * MAX_NUM_HCAS);
                    }
                }

                req = req->dev.next;
            }
        }

        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Searched %d packets in cm_send_Q\n", i);

        /* Prepare remote update packet. */
        msg.recv_mem_addr = temp->mrail.rndv_buf;
        MPIU_Memcpy(msg.recv_buf_rkey, rkey, sizeof(uint32_t) * MAX_NUM_HCAS);

        /*FIXME: use recv_mem_addr as only identifier */
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"recv_mem_addr %p, rkey0 %x\n", msg.recv_mem_addr, msg.recv_buf_rkey[0]);

        if(!SMP_ONLY)
        {
            v = get_vbuf_by_offset(MV2_RECV_VBUF_POOL_OFFSET);
            p = (MPIDI_CH3I_MRAILI_Pkt_comm_header*) v->pheader;
        }
        else
        {
            p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *)MPIU_Malloc(sizeof(MPIDI_CH3I_MRAILI_Pkt_comm_header));
        }

        p->type = MPIDI_CH3_PKT_CR_REMOTE_UPDATE;
        MPIU_Memcpy(v->buffer + sizeof(MPIDI_CH3I_MRAILI_Pkt_comm_header), &msg, sizeof(msg));

        /* Push update packet to message log queue. */
        entry = (MPIDI_CH3I_CR_msg_log_queue_entry_t *) MPIU_Malloc(sizeof(MPIDI_CH3I_CR_msg_log_queue_entry_t));
        entry->len = sizeof(MPIDI_CH3I_MRAILI_Pkt_comm_header) + sizeof(msg);
        entry->buf = v;
        MSG_LOG_ENQUEUE(vc, entry);
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"[%d => %d]: enq REM_UPDATE...\n", MPICR_pg_rank, vc->pg_rank);

        temp = temp->ch.cr_queue_next;
    }

    return 0;
}

/*
 * Before ckpt, record the num of active rndv for each vc(as sender), so that
 * we know how many rndvs need to be updated with their rkey during restart
 */
void CR_record_rndv(MPIDI_VC_t ** vc_vector)
{
    struct MPID_Request *sreq = NULL;
    MPIDI_VC_t *vc = NULL;

    int i, n;

    for (i = 0; i < MPICR_pg_size; i++) {
        if (i == MPICR_pg_rank)
            continue;

        vc = vc_vector[i];
        if (vc == NULL) {
            continue;
        }

        n = 0;
        sreq = (struct MPID_Request *) vc->mrail.sreq_head;

        /* record: num of rndvs as sender */
        while (sreq)
        {
            n++;
            sreq = sreq->mrail.next_inflow;
        }

        /* number of rndvs that need to be updated with a new rkey at restart */
        vc->mrail.sreq_to_update = n;
        if (n > 0 || vc->ch.sendq_head || vc->ch.cm_sendq_head || vc->mrail.msg_log_queue_head) {
            PRINT_DEBUG(DEBUG_CR_verbose > 2,"%s: [%d vc_%d]: has  %d  rndv-send to update, ch.sendq_head = %p, ch.cm_sendq_head=%p, vc.mrail.msg_log_q_head=%p\n",
                   __func__, MPICR_pg_rank, vc->pg_rank, n, vc->ch.sendq_head, vc->ch.cm_sendq_head, vc->mrail.msg_log_queue_head);
        }

    } 
}

void CR_record_flowlist(char *title)
{
    extern MPIDI_VC_t *flowlist;

    MPIDI_VC_t *vc = flowlist;

    int n = 0;
    while (vc) {
        n++;
        MPIU_Assert(vc->mrail.inflow == 1);
        vc = vc->mrail.nextflow;
    }

    /* count pending CTS at recv-side */
    struct MPID_Request *temp = MPICR_req_list_head;
    int cts = 0;
    while (temp) {
        cts++;
        temp = temp->ch.cr_queue_next;
    }
    if (n > 0 || cts > 0)
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"[%d]: %s:  flowlist = %d, CTS = %d\n", MPICR_pg_rank, title, n, cts);

}

int CR_IBU_Suspend_channels()
{
    int i = 0;
    MPIDI_VC_t **vc_vector = (MPIDI_VC_t **) MPIU_Malloc(sizeof(MPIDI_VC_t *) * MPICR_pg_size);
    MPIDI_VC_t *vc = NULL;

    MPIU_Assert(vc_vector);
    for (; i < MPICR_pg_size; ++i) {
        if (i == MPICR_pg_rank) {
            continue;
        }
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"PG_get_vc=%d\n", i);
        MPIDI_PG_Get_vc(MPICR_pg, i, &vc);

        /* init the fields to be used at resume */
        if (vc) {
            pthread_spin_init(&vc->mrail.cr_lock, 0);
            vc->mrail.react_send_ready = 0;
            vc->mrail.react_entry = NULL;
        }

        if (vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE || vc->ch.state == MPIDI_CH3I_VC_STATE_SUSPENDING
            || vc->ch.state == MPIDI_CH3I_VC_STATE_CONNECTING_SRV) {
            vc_vector[i] = vc;
        } else {
            vc_vector[i] = NULL;
        }
    }
    int retval = 0;

    if ((retval = MPIDI_CH3I_CM_Suspend(vc_vector))) {
        return retval;
    }
    PRINT_DEBUG(DEBUG_CR_verbose,"fin:  MPIDI_CH3I_CM_suspend\n");

    if(!SMP_ONLY) {
        if ((retval = CR_IBU_Release_network())) {
            return retval;
        }
        PRINT_DEBUG(DEBUG_CR_verbose,"fin:  IBU_Release_network\n");
    } else {
        /* The cm_conn_state_lock is unlocked inside CR_IBU_Release_network().
         * But for the SMP_ONLY case, we need to unlock it explicitly here
         * as CR_IBU_Release_network() is not called */
        if (MPIDI_CH3I_CR_Get_state() == MPICR_STATE_PRE_COORDINATION)
            MPICM_unlock();
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"fin:  MPICM_unlock\n");
    }

    /* record num of rndvs(sender) to update at restart */
    CR_record_rndv(vc_vector);
    //CR_record_flowlist("after susp-chan");
    MPIU_Free(vc_vector);

    return retval;
}

int CR_IBU_Reactivate_channels()
{
    int retval = 0;

    if(!SMP_ONLY)
    {
        PRINT_DEBUG(DEBUG_CR_verbose,"Calling CR_IBU_Rebuild_network\n");
        if ((retval = CR_IBU_Rebuild_network()))
        {
            return retval;
        }
    }

    int pg_rank;
    UPMI_GET_RANK(&pg_rank);
    retval = MPIDI_CH3I_set_affinity(MPICR_pg, pg_rank);
    if (retval) {
        fprintf(stderr, "[%s:%d] MPIDI_CH3I_set_affinity) returned %d\n", 
                                          __FILE__, __LINE__, retval);
        return (retval);
    }

    mv2_init_call_once = 0;
    mv2_mmap_coll_once = 0;

    /* Reinitialize the SMP channel */
    PRINT_DEBUG(DEBUG_CR_verbose, "MPIDI_CH3I_SMP_init()\n");
    retval = MPIDI_CH3I_SMP_init(MPICR_pg);
    if (retval) {
        fprintf(stderr, "[%s:%d] MPIDI_CH3I_SMP_init() returned %d\n", __FILE__, __LINE__, retval);
        return (retval);
    }

    /* Attach shm pool if it has not been mmap'ed() already */
    if ( mv2_shmem_pool_init ) {
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"Attempting to reattach SMP pool used for large-message exchange before checkpoint.\n");
        if ((retval = MPIDI_CH3I_SMP_attach_shm_pool()) != MPI_SUCCESS) {
            return (retval);
        }
        PRINT_DEBUG(DEBUG_CR_verbose > 2,"SMP pool init and attached.\n");
    }

    PRINT_DEBUG(DEBUG_CR_verbose, "CR_IBU_Prep_remote_update\n");
    if ((retval = CR_IBU_Prep_remote_update())) {
        PRINT_DEBUG(DEBUG_CR_verbose, "CR_IBU_Prep_remote_update() returned %d\n",retval);
        return retval;
    }

    MPIDI_VC_t **vc_vector = (MPIDI_VC_t **) MPIU_Malloc(sizeof(MPIDI_VC_t *) * MPICR_pg_size);
    MPIDI_VC_t *vc = NULL;
    int i = 0;

    for (; i < MPICR_pg_size; ++i) {
        if (i == MPICR_pg_rank) {
            continue;
        }

        MPIDI_PG_Get_vc(MPICR_pg, i, &vc);

        /* Now all calling can only be small rank reactivate to big rank. */
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_CONNECTING_CLI);
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_CONNECTING_SRV);
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_SUSPENDING);
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE);
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_REACTIVATING_CLI_1);
        MPIU_Assert(vc->ch.state != MPIDI_CH3I_VC_STATE_REACTIVATING_CLI_2);

        if (vc->ch.state == MPIDI_CH3I_VC_STATE_SUSPENDED || vc->ch.state == MPIDI_CH3I_VC_STATE_REACTIVATING_SRV) {
            vc_vector[i] = vc;

            /* ensure that REACT_DONE are behind REM_UPDATE(if any) */
            pthread_spin_lock(&vc->mrail.cr_lock);
            vc->mrail.react_send_ready = 1;
            if (vc->mrail.react_entry) { /* cm_thread has gotten REACT_DONE from peer */
                /* enquue the REACT_DONE to be sent to peer */
                PRINT_DEBUG(DEBUG_CR_verbose > 2,"%s: [%d => %d]: REACT_DONE came earlier, enq now...\n", __func__, MPICR_pg_rank, vc->pg_rank);
                MSG_LOG_ENQUEUE(vc, vc->mrail.react_entry);
            }
            pthread_spin_unlock(&vc->mrail.cr_lock);
        } else {
            vc_vector[i] = NULL;
        }
    }

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"MPIDI_CH3I_CM_Reactivate\n");

    if ((retval = MPIDI_CH3I_CM_Reactivate(vc_vector))) {
        return retval;
    }
    MPIU_Free(vc_vector);

    return retval;
}

void MPIDI_CH3I_CR_req_enqueue(struct MPID_Request *req, MPIDI_VC_t * vc)
{
    if (req == NULL) {
        return;
    }

    req->ch.cr_queue_next = NULL;
    req->ch.vc = vc;

    if (MPICR_req_list_head == NULL) {
        MPICR_req_list_tail = MPICR_req_list_head = req;
    } else {
        MPICR_req_list_tail->ch.cr_queue_next = req;
        MPICR_req_list_tail = req;
    }
}

void MPIDI_CH3I_CR_req_dequeue(struct MPID_Request *req)
{
    if (req == NULL) {
        return;
    }

    if (MPICR_req_list_head == NULL) {
        return;
    }

    if (MPICR_req_list_head == req) {
        MPICR_req_list_head = req->ch.cr_queue_next;

        if (MPICR_req_list_tail == req) {
            /* Last element in the list. */
            MPICR_req_list_tail = NULL;
        }

        return;
    }

    struct MPID_Request *temp = MPICR_req_list_head;

    while (temp->ch.cr_queue_next != NULL) {
        if (temp->ch.cr_queue_next == req) {
            temp->ch.cr_queue_next = req->ch.cr_queue_next;

            if (MPICR_req_list_tail == req) {
                /* Dequeue last element. */
                MPICR_req_list_tail = temp;
            }

            return;
        }

        temp = temp->ch.cr_queue_next;
    }
}

#ifdef CR_FTB

static int CR_FTB_Init(int rank, char *sessionid)
{
    int ret, fd;
    char *str;
    char session_file[64];

    static int count = 0;
    count++;


    memset(&ftb_cinfo, 0, sizeof(ftb_cinfo));
    strcpy(ftb_cinfo.client_schema_ver, "0.5");
    strcpy(ftb_cinfo.event_space, "FTB.MPI.MVAPICH2");

    snprintf(ftb_cinfo.client_name, FTB_MAX_CLIENT_NAME, "MVAPICH2.%d", rank);
    snprintf(ftb_cinfo.client_jobid, FTB_MAX_CLIENT_JOBID, "%s", sessionid);

    strcpy(ftb_cinfo.client_subscription_style, "FTB_SUBSCRIPTION_BOTH");
    ftb_cinfo.client_polling_queue_len = 64;

    ret = FTB_Connect(&ftb_cinfo, &ftb_handle);
    if (ret != FTB_SUCCESS)
        goto err_connect;

    ret = FTB_Declare_publishable_events(ftb_handle, NULL, cr_ftb_events, CR_FTB_EVENTS_MAX);
    if (ret != FTB_SUCCESS)
        goto err_declare_events;

    str = MPIU_Malloc(sizeof(char) * FTB_MAX_SUBSCRIPTION_STR);
    if (!str)
        goto err_malloc;

    snprintf(str, FTB_MAX_SUBSCRIPTION_STR, "event_space=FTB.STARTUP.MV2_MPIRUN , jobid=%s", sessionid);
    PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t FTB_SUBSRIBE STR:%s: for rank %d\n", str, rank);

    ret = FTB_Subscribe(&shandle, ftb_handle, str, CR_FTB_Callback, NULL);
    if (ret != FTB_SUCCESS)
        goto err_subscribe;

    /* Set PMI_PORT */
    snprintf(session_file, 64, "/tmp/cr.session.%s", getenv("MV2_CKPT_SESSIONID"));
    fd = open(session_file, O_RDWR);
    if (fd < 0)
        goto err_open;

    ret = read(fd, str, sizeof(char) * FTB_MAX_SUBSCRIPTION_STR);
    str[ret] = '\0';
    close(fd);
    if (ret < 0)
        goto err_read;

    PRINT_DEBUG(DEBUG_CR_verbose > 1,"use PMI_PORT=%s\n", str);
    setenv("PMI_PORT", str, 1);

    MPIU_Free(str);
    ftb_init_done = 1;
    return (0);

  err_connect:
    fprintf(stderr, "FTB_Connect() failed with %d for rank %d\n", ret, rank);
    ret = -1;
    goto exit_connect;

  err_declare_events:
    fprintf(stderr, "FTB_Declare_publishable_events() failed with %d" " for rank %d\n", ret, rank);
    ret = -2;
    goto exit_declare_events;

  err_malloc:
    fprintf(stderr, "Failed to malloc() subscription_str for rank %d\n", rank);
    ret = -3;
    goto exit_malloc;

  err_subscribe:
    fprintf(stderr, "FTB_Subscribe() failed with %d for rank %d\n", ret, rank);
    ret = -4;
    goto exit_subscribe;

  err_open:
    perror("open");
    fprintf(stderr, "Rank %d cannot open %s\n", rank, session_file);
    ret = -5;
    goto exit_open;

  err_read:
    perror("read");
    fprintf(stderr, "Could not read the Spawn Port for rank %d\n", rank);
    ret = -6;
    goto exit_read;

  exit_read:
  exit_open:
  exit_subscribe:
  exit_malloc:
  exit_declare_events:
    FTB_Disconnect(ftb_handle);

  exit_connect:
    return (ret);

}

static void CR_FTB_Finalize()
{
    if (ftb_init_done) {
        ftb_init_done = 0;
        int ret = FTB_Unsubscribe(&shandle);
        usleep(20000);

        if (ret != FTB_SUCCESS) {
            fprintf(stderr, "FTB_Unsubscribe failed with code %d\n", ret);
        } else {
            PRINT_DEBUG(DEBUG_CR_verbose ,"FTB_unsubscribe succeeded %d\n", ret);
        }

        //return ret;
        FTB_Disconnect(ftb_handle);
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"\t %s:%d: [CR]rank =%d : Disconnected handle\n", __FILE__, __LINE__, MPIDI_Process.my_pg_rank);
    }
}

static int parse_aggre_migration_string(char *msg, char *src, char *tgt)
{
    char buf[256];

    strncpy(buf, msg, 255);
    buf[255] = 0;
    PRINT_DEBUG(DEBUG_MIG_verbose > 1, "buf = \"%s\"\n", buf);

    /* Format of "buf":  "srcnode  tgtnode  proc_cnt  procid1  procid2 ..." */
    char *tok;
    /* extract source */
    tok = strtok(buf, " \n\t");
    strcpy(src, tok);
    /* extract target */
    tok = strtok(NULL, " \n\t");
    strcpy(tgt, tok);

    PRINT_DEBUG(DEBUG_MIG_verbose > 1, "msg=%s, src=%s, tgt=%s\n", msg, src, tgt);
    return 0;
}

static int CR_FTB_Callback(FTB_receive_event_t * revent, void *arg)
{
    char my_hostname[256];
    gethostname(my_hostname, 255);

    PRINT_DEBUG( DEBUG_FT_verbose, "Got event %s from %s: payload=%s\n", revent->event_name, revent->client_name, revent->event_payload);

    /* TODO: Do some sanity checking */
    if (!strcmp(revent->event_name, EVENT(CR_FTB_CHECKPOINT))) {
        pthread_mutex_lock(&cr_ftb_mutex);
        cr_ftb_ckpt_req = 1;
        pthread_cond_signal(&cr_ftb_cond);
        pthread_mutex_unlock(&cr_ftb_mutex);
        return (0);
    }

    if (!strcmp(revent->event_name, EVENT(CR_FTB_MIGRATE))) {
        if (use_aggre_mig)
            parse_aggre_migration_string(revent->event_payload, cr_mig_src_host, cr_mig_tgt_host);
        else
            get_src_tgt(revent->event_payload, cr_mig_src_host, cr_mig_tgt_host);
        PRINT_DEBUG( DEBUG_FT_verbose, "Migrate %s to %s\n", cr_mig_src_host, cr_mig_tgt_host);
        pthread_mutex_lock(&cr_ftb_mutex);
        cr_ftb_mig_req = 1;
        pthread_cond_signal(&cr_ftb_cond);
        pthread_mutex_unlock(&cr_ftb_mutex);

        return (0);
    }
    return (0);
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

#else

/*===========================*/
/*  MPD messaging functions  */
/*===========================*/

int CR_MPDU_readline(int fd, char *buf, int maxlen)
{
    int n = 1;
    int rc;
    char c;
    char *ptr = buf;

    for (n = 1; n < maxlen; ++n) {
      again:
        rc = read(fd, &c, 1);

        if (rc == 1) {
            *ptr = c;
            ++ptr;

            /* Note \n is stored, like in fgets. */
            if (c == '\n') {
                break;
            }
        } else if (rc == 0) {
            if (n == 1) {
                /* EOF, no data read. */
                return 0;
            } else {
                /* EOF, some data read. */
                break;
            }
        } else {
            if (errno == EINTR) {
                goto again;
            }

            /* Error, errno set by read. */
            return -1;
        }
    }

    /* Null termination, like fgets. */
    *ptr = 0;
    return n;
}

int CR_MPDU_writeline(int fd, char *buf)
{
    int size = strlen(buf);

    if (size > MAX_CR_MSG_LEN) {
        buf[MAX_CR_MSG_LEN - 1] = '\0';
        fprintf(stderr, "\t write_line: message string too big: :%s:\n", buf);
    } else if (buf[strlen(buf) - 1] != '\n') {
        fprintf(stderr, "write_line: message string doesn't end in newline: :%s:\n", buf);
    } else {
        int n = write(fd, buf, size);

        if (n < 0) {
            fprintf(stderr, "write_line error; fd=%d buf=:%s:\n", fd, buf);
            return -1;
        }

        if (n < size) {
            fprintf(stderr, "write_line failed to write entire message\n");
        }
    }

    return 0;
}

int CR_MPDU_parse_keyvals(char *st)
{
    if (!st) {
        return -1;
    }

    CRU_keyval_tab_idx = 0;
    char *p = st;
    char *keystart = NULL;
    char *valstart = NULL;

    while (1) {
        /* Increment until a numeric digit or a letter of the alphabet is found. */
        while (*p != '\0' && !isalnum(*p)) {
            ++p;
        }

        if (*p == '\0') {
            return 0;
        }

        keystart = p;           /* remember where key started */

        while (*p != ' ' && *p != '=' && *p != '\n' && *p != '\0') {
            ++p;
        }

        if (*p == ' ' || *p == '\n' || *p == '\0') {
            PRINT_ERROR("Internal error: Unexpected key delimiter at character %d in '%s'\n", (int) (p - st), st);
            return -1;
        }

        strncpy(CRU_keyval_tab[CRU_keyval_tab_idx].key, keystart, CRU_MAX_KEY_LEN);
        CRU_keyval_tab[CRU_keyval_tab_idx].key[p - keystart] = '\0';    /* store key */

        valstart = ++p;         /* start of value */

        while (*p != ' ' && *p != '\n' && *p != '\0') {
            ++p;
        }

        strncpy(CRU_keyval_tab[CRU_keyval_tab_idx].value, valstart, CRU_MAX_VAL_LEN);
        CRU_keyval_tab[CRU_keyval_tab_idx].value[p - valstart] = '\0';  /* store value */
        ++CRU_keyval_tab_idx;

        if (*p == ' ') {
            continue;
        }

        if (*p == '\n' || *p == '\0') {
            /* Value has been set to empty. */
            return 0;
        }
    }
}

char *CR_MPDU_getval(const char *keystr, char *valstr, int vallen)
{
    int i = 0;

    for (; i < CRU_keyval_tab_idx; ++i) {
        if (strcmp(keystr, CRU_keyval_tab[i].key) == 0) {
            strncpy(valstr, CRU_keyval_tab[i].value, vallen - 1);
            valstr[vallen - 1] = '\0';
            return valstr;
        }
    }

    valstr[0] = '\0';
    return NULL;
}

#endif                          /* CR_FTB */

#endif                          /* ifdef CKPT */
