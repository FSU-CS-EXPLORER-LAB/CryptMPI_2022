/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
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

#if !defined(MPICH_MPIDI_CH3_PRE_H_INCLUDED)
#define MPICH_MPIDI_CH3_PRE_H_INCLUDED

#include "mpichconf.h"
#include "mpidi_ch3_rdma_pre.h"
#include "smp_smpi.h"
#include "mpiu_os_wrappers_pre.h"

#if defined (_SHARP_SUPPORT_)
#include "api/sharp_coll.h"
#endif

/*#define MPICH_DBG_OUTPUT*/

#ifdef _OSU_MVAPICH_
typedef struct {
    MPI_Comm     leader_comm;
    MPI_Comm     shmem_comm;
    MPI_Comm     allgather_comm;
      /************ Added by Mehran ************/
    MPI_Comm     concurrent_comm;
    /*****************************************/
    int*    leader_map;
    int*    leader_rank;
    int*    node_sizes;		 /* number of processes on each node */
    int*    node_disps;      /* displacements into rank_list for each node */
    int*    allgather_new_ranks;
    int*    rank_list;       /* list of ranks, ordered by node id, then shmem rank on each node */
    int     rank_list_index; /* index of this process in the rank_list array */
    int     is_uniform;
    int     is_blocked;
      /************ Added by Mehran ************/
    int     equal_local_sizes;
    /*****************************************/
    int     shmem_comm_rank;
    int     shmem_coll_ok;
    int     allgather_comm_ok; 
    int     leader_group_size;
    int     is_global_block;
    int     is_pof2; /* Boolean to know if comm size is equal to pof2  */
    int     gpof2; /* Greater pof2 < size of comm */
    int     intra_node_done; /* Used to check if intra node communication has been done 
                                with mcast and bcast */
    int     shmem_coll_count;
    int     allgather_coll_count;
    int     allreduce_coll_count;
    int     bcast_coll_count;
    int     scatter_coll_count;
    void    *shmem_info; /* intra node shmem info */
    MPI_Comm     intra_sock_comm;
    MPI_Comm     intra_sock_leader_comm;
    MPI_Comm     global_sock_leader_comm;
    int*         socket_size;
     /***** Added by Mehran *****/
    int*        local_table;
    /***************************/
    int          is_socket_uniform;
    int          use_intra_sock_comm;
    int          my_sock_id;
    int          tried_to_create_leader_shmem;
#if defined(_MCST_SUPPORT_)
    int     is_mcast_ok;
    void    *bcast_info;
#endif

#if defined(_SHARP_SUPPORT_)
    int     is_sharp_ok;
    void    *sharp_coll_info;
#endif

} MPIDI_CH3I_CH_comm_t;
#else
typedef struct {
    int dummy;  /* dummy variable to ensure we don't have an empty structure */
} MPIDI_CH3I_CH_comm_t;
#endif /* _OSU_MVAPICH_ */

#if defined (_SHARP_SUPPORT_)
#define SHARP_REQ_HANDLE void
#define MPID_DEV_SHARP_REQUEST_DECL         \
            SHARP_REQ_HANDLE * sharp_req;   \

#define MPIDI_CH3_SHARP_REQUEST_INIT(_req)  \
            (_req)->sharp_req = NULL;       \

#define MPID_SHARP_COLL_REQ_WAIT(_req)     sharp_coll_req_wait(_req->sharp_req)
#define MPID_SHARP_COLL_REQ_FREE(_req)     sharp_coll_req_free(_req->sharp_req)
#define MPID_SHARP_COLL_SUCCESS            SHARP_COLL_SUCCESS 
#endif

typedef struct MPIDI_CH3I_Process_group_s
{
    char * kvs_name;
    struct MPIDI_VC * unex_finished_list;
    int nEagerLimit;
    int local_process_id;
    int num_local_processes;
    int nRDMAWaitSpinCount;
    int nRDMAWaitYieldCount;
# if defined(MPIDI_CH3I_RDMA_PG_DECL)
    MPIDI_CH3I_RDMA_PG_DECL
# endif
}
MPIDI_CH3I_Process_group_t;

#define MPIDI_CH3_PG_DECL MPIDI_CH3I_Process_group_t ch;

#define MPIDI_CH3_PKT_DECL 
#define MPIDI_CH3_PKT_DEFS

#define MPIDI_DEV_IMPLEMENTS_KVS

#define MPIDI_CH3_IMPLEMENTS_ABORT

typedef enum MPIDI_CH3I_VC_state
{
    MPIDI_CH3I_VC_STATE_INVALID,
    MPIDI_CH3I_VC_STATE_UNCONNECTED,
    MPIDI_CH3I_VC_STATE_CONNECTING_CLI,
    MPIDI_CH3I_VC_STATE_CONNECTING_SRV,
#ifdef CKPT
    MPIDI_CH3I_VC_STATE_SUSPENDING,
    MPIDI_CH3I_VC_STATE_SUSPENDED,
    MPIDI_CH3I_VC_STATE_REACTIVATING_CLI_1,
    MPIDI_CH3I_VC_STATE_REACTIVATING_CLI_2,
    MPIDI_CH3I_VC_STATE_REACTIVATING_SRV,
#endif
#ifdef RDMA_CM
    MPIDI_CH3I_VC_STATE_IWARP_SRV_WAITING,
    MPIDI_CH3I_VC_STATE_IWARP_CLI_WAITING,
#endif 
    MPIDI_CH3I_VC_STATE_IDLE,
    MPIDI_CH3I_VC_STATE_FAILED
}
MPIDI_CH3I_VC_state_t;

#define IS_RC_CONN_ESTABLISHED(vc) \
    (vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE && \
        vc->mrail.state & MRAILI_RC_CONNECTED)

/* This structure requires the iovec structure macros to be defined */
typedef struct MPIDI_CH3I_Buffer_t
{
    int use_iov;
    unsigned int num_bytes;
    void *buffer;
    unsigned int bufflen;
    MPL_IOV *iov;
    int iovlen;
    int index;
    int total;
} MPIDI_CH3I_Buffer_t;

typedef struct MPIDI_CH3I_RDMA_Unex_read_s
{
    struct MPIDI_CH3I_RDMA_Packet_t *pkt_ptr;
    unsigned char *buf;
    unsigned int length;
    int src;
    struct MPIDI_CH3I_RDMA_Unex_read_s *next;
} MPIDI_CH3I_RDMA_Unex_read_t;
#ifdef _ENABLE_XRC_
struct _xrc_pending_conn;
#endif
typedef struct MPIDI_CH3I_VC
{
    struct MPID_Request * sendq_head;
    struct MPID_Request * sendq_tail;
    struct MPID_Request * send_active;
    struct MPID_Request * recv_active;
    struct MPID_Request * req;
    volatile MPIDI_CH3I_VC_state_t state;
    MPIDI_CH3I_Buffer_t read;
    int read_state;
    int port_name_tag;
    /* Connection management */
    struct MPID_Request * cm_sendq_head;
    struct MPID_Request * cm_sendq_tail;
    struct vbuf         * cm_1sc_sendq_head;
    struct vbuf         * cm_1sc_sendq_tail;
#ifdef CKPT
    volatile int rput_stop; /*Stop rput message and wait for rkey update*/
#endif
#ifdef _ENABLE_XRC_
    uint32_t                    xrc_flags;
    struct MPIDI_VC             *orig_vc;
    struct _xrc_pending_conn    *xrc_conn_queue;
    uint32_t                    xrc_srqn[MAX_NUM_HCAS];
    uint32_t                    xrc_rqpn[MAX_NUM_SUBRAILS];
    uint32_t                    xrc_my_rqpn[MAX_NUM_SUBRAILS];
#endif
    MPIDI_msg_sz_t              pending_r3_data;
    MPIDI_msg_sz_t              received_r3_data;
} MPIDI_CH3I_VC;

#ifdef _ENABLE_XRC_
typedef struct _xrc_pending_conn {
    struct _xrc_pending_conn    *next;
    struct MPIDI_VC             *vc;
} xrc_pending_conn_t;
#define xrc_pending_conn_s (sizeof (xrc_pending_conn_t))

#define VC_XST_ISSET(vc, st)      ((vc)->ch.xrc_flags & (st))
#define VC_XST_ISUNSET(vc, st)    (!((vc)->ch.xrc_flags & (st)))
#define VC_XSTS_ISSET(vc, sts)    (((vc)->ch.xrc_flags & (sts)) == (sts))
#define VC_XSTS_ISUNSET(vc, sts)  (((vc)->ch.xrc_flags & (sts)) == 0)

#define VC_XST_SET(vc, st) vc->ch.xrc_flags |= (st);
#define VC_XST_CLR(vc, st) vc->ch.xrc_flags &= ~(st);

#define     XF_NONE             0x00000000 
#define     XF_START_RDMAFP     0x00000001
#define     XF_DIRECT_CONN      0x00000002
#define     XF_INDIRECT_CONN    0x00000004 
#define     XF_NEW_QP           0x00000008
#define     XF_SEND_IDLE        0x00000010
#define     XF_RECV_IDLE        0x00000020
#define     XF_NEW_RECV         0x00000040
#define     XF_CONN_CLOSING     0x00000080
#define     XF_INIT_DONE        0x00000100
#define     XF_SEND_CONNECTING  0x00000200
#define     XF_REUSE_WAIT       0x00000400
#define     XF_SMP_VC           0x00000800
#define     XF_DPM_INI          0x00001000
#define     XF_TERMINATED       0x00002000
#define     XF_UD_CONNECTED     0x00004000
#endif

#ifdef _ENABLE_XRC_
#define VC_NOT_READY(vc) \
    (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE || \
    	(USE_XRC && VC_XST_ISUNSET (vc, XF_SEND_IDLE)))
#else
#define VC_NOT_READY(vc) (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE)
#endif

/* SMP Channel is added by OSU-MPI2 */
typedef enum SMP_pkt_type
{
    SMP_EAGER_MSG,
    SMP_RNDV_MSG,
    SMP_RNDV_MSG_CONT
} SMP_pkt_type_t;

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
typedef enum CUDA_IPC_STATUS
{ 
    CUDA_IPC_UNINITIALIZED,
    CUDA_IPC_ENABLED,
    CUDA_IPC_DISABLED
} cuda_ipc_status_t;
#endif

typedef struct MPIDI_CH3I_SMP_VC
{
    struct MPID_Request * sendq_head;
    struct MPID_Request * sendq_tail;
    struct MPID_Request * send_active;
    struct MPID_Request * recv_active;
    int local_nodes;
    int local_rank;
    SMP_pkt_type_t send_current_pkt_type;
    SMP_pkt_type_t recv_current_pkt_type;
    int hostid;
    int read_index;
    int read_off;

#if defined(_SMP_LIMIC_)
    struct limic_header current_l_header;
#else
    int current_l_header;
#endif 

#if defined(_SMP_CMA_)
    struct cma_header current_c_header;
#else
    int current_c_header;
#endif

    int current_cnb;
    int current_nb;
    int use_limic;
    int use_cma;

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    cuda_ipc_status_t can_access_peer;
#endif
} MPIDI_CH3I_SMP_VC;

#ifndef MPIDI_CH3I_VC_RDMA_DECL
#define MPIDI_CH3I_VC_RDMA_DECL
#endif

#define MPIDI_CH3_VC_DECL \
MPIDI_CH3I_VC ch; \
MPIDI_CH3I_SMP_VC smp; \
MPIDI_CH3I_VC_RDMA_DECL
/* end of OSU-MPI2 */

/*
 * MPIDI_CH3_CA_ENUM (additions to MPIDI_CA_t)
 *
 * MPIDI_CH3I_CA_HANDLE_PKT - The completion of a packet request (send or
 * receive) needs to be handled.
 */
#define MPIDI_CH3_CA_ENUM			\
MPIDI_CH3I_CA_HANDLE_PKT,			\
MPIDI_CH3I_CA_END_RDMA

enum REQ_TYPE {
    REQUEST_NORMAL,
    REQUEST_RNDV_R3_HEADER,
    REQUEST_RNDV_R3_DATA,
    REQUEST_LIGHT
};

/*
 * MPIDI_CH3_REQUEST_DECL (additions to MPID_Request)
 */
#define MPIDI_CH3_REQUEST_DECL						\
struct MPIDI_CH3I_Request						\
{									\
    /*  pkt is used to temporarily store a packet header associated	\
       with this request */						\
    MPIDI_CH3_Pkt_t pkt;                        \
    enum REQ_TYPE   reqtype;						\
    /* For CKPT, hard to put in ifdef because it's in macro define*/    \
    struct MPID_Request *cr_queue_next;                                 \
    struct MPIDI_VC *vc;                                                \
} ch;

#ifdef _ENABLE_CUDA_
#define  MPIDI_CH3_REQUEST_INIT_CUDA(_rreq)          \
    (_rreq)->mrail.cuda_transfer_mode = 0;           \
    (_rreq)->mrail.pipeline_nm = 0;                  \
    (_rreq)->mrail.cuda_event = NULL;                \
    (_rreq)->dev.pending_pkt = NULL;                 \
    (_rreq)->dev.cuda_srbuf_entry = NULL;            \
    (_rreq)->dev.is_device_tmpbuf = 0;
#else
#define MPIDI_CH3_REQUEST_INIT_CUDA(sreq_)
#endif

#if defined(HAVE_CUDA_IPC)
#define MPIDI_CH3_REQUEST_INIT_CUDA_IPC(_rreq)      \
    (_rreq)->mrail.cudaipc_stage_index = 0;         \
    (_rreq)->mrail.ipc_cuda_event = NULL;           \
    (_rreq)->mrail.ipc_baseptr = NULL;              \
    (_rreq)->mrail.ipc_size = 0;                    \
    (_rreq)->mrail.ipc_event = NULL;                \
    (_rreq)->mrail.cuda_reg = NULL;

#else
#define MPIDI_CH3_REQUEST_INIT_CUDA_IPC(_rreq)
#endif 

#define MPIDI_CH3_REQUEST_INIT(_rreq)    \
    (_rreq)->dev.OnDataAvail = NULL;     \
    (_rreq)->dev.OnFinal = NULL;         \
    (_rreq)->mrail.rndv_buf_alloc = 0;   \
    (_rreq)->mrail.rndv_buf = NULL;      \
    (_rreq)->mrail.rndv_buf_sz = 0;      \
    (_rreq)->mrail.rndv_buf_off = 0;     \
    (_rreq)->mrail.protocol = 0;         \
    (_rreq)->mrail.d_entry = NULL;       \
    (_rreq)->mrail.remote_addr = NULL;   \
    (_rreq)->mrail.nearly_complete = 0;  \
    (_rreq)->mrail.local_complete  = 0;  \
    (_rreq)->mrail.remote_complete = 0;  \
    (_rreq)->mrail.is_rma_last_stream_unit = 1;  \
    MPIDI_CH3_REQUEST_INIT_CUDA(_rreq)   \
    MPIDI_CH3_REQUEST_INIT_CUDA_IPC(_rreq) 

typedef struct MPIDI_CH3I_Progress_state
{
    int completion_count;
}
MPIDI_CH3I_Progress_state;

/* This variable is used in the definitions of the MPID_Progress_xxx macros,
   and must be available to the routines in src/mpi */
extern volatile unsigned int MPIDI_CH3I_progress_completion_count;

#define MPIDI_CH3_PROGRESS_STATE_DECL MPIDI_CH3I_Progress_state ch;

typedef pthread_mutex_t MPIDI_CH3I_SHM_MUTEX;

#if defined(CHANNEL_MRAIL)
                                                                                 
#define MPIDI_CH3_WIN_DECL                                                       \
    int  fall_back;                                                              \
    int  enable_fast_path;                                                       \
    int  use_rdma_path;                                                          \
    int  is_active;                                                              \
    int  using_lock;                                                             \
    long long cc_for_test;                                                       \
    struct dreg_entry* completion_counter_dreg_entry;                            \
    volatile long long * completion_counter;                                     \
    long long ** all_completion_counter;                                         \
    uint32_t  *completion_counter_rkeys; /* rkey for complete couters on         \
                                            remote nodes */                      \
    struct dreg_entry* win_dreg_entry;                                           \
    uint32_t *win_rkeys;          /* exposed buffer addresses on remote          \
                                    windows */                                   \
    struct dreg_entry* post_flag_dreg_entry;                                     \
    volatile int *post_flag;     /* flag from post to complete, one flag for     \
                                    each target, updated by RDMA */              \
    uint32_t *post_flag_rkeys;                                                   \
    int ** remote_post_flags;                                                    \
                                                                                 \
    int using_start;                                                             \
    /*for get/put queue*/                                                        \
    MPIDI_CH3I_RDMA_put_get_list * put_get_list;                                 \
    int put_get_list_size;                                                       \
    int put_get_list_tail;                                                       \
    int * put_get_list_size_per_process;                                         \
    int wait_for_complete;                                                       \
    int rma_issued;                                                              \
    /* Preregistered buffer for small msg */                                     \
    char * pinnedpool_1sc_buf;                                                   \
    int    pinnedpool_1sc_index;                                                 \
    struct dreg_entry * pinnedpool_1sc_dentry;                                   \
                                                                                 \
    int16_t outstanding_rma;                                                     \
    int *shm_l2g_rank;                                                           \
    int node_comm_size;                                                          \
    MPID_Comm *node_comm_ptr;                                                    \
    volatile int poll_flag; /* flag to indicate if polling for one sided completions is needed */ \
    void *shm_base_addr;        /* base address of shared memory region */              \
    int shm_coll_comm_ref;                                                              \
    MPI_Aint shm_segment_len;   /* size of shared memory region */                      \
    MPIU_SHMW_Hnd_t shm_segment_handle; /* handle to shared memory region */            \
    MPIDI_CH3I_SHM_MUTEX *shm_mutex;    /* shared memory windows -- lock for            \
                                           accumulate/atomic operations */              \
    MPIU_SHMW_Hnd_t shm_mutex_segment_handle; /* handle to interprocess mutex memory    \
                                                 region */                              \
    void *info_shm_base_addr; /* base address of shared memory region for window info */          \
    MPI_Aint info_shm_segment_len; /* size of shared memory region for window info */             \
    MPIU_SHMW_Hnd_t info_shm_segment_handle; /* handle to shared memory region for window info */ \

extern int mv2_create_dummy_request();
extern int mv2_free_dummy_request();
#endif /* defined(CHANNEL_MRAIL) */

#endif /* !defined(MPICH_MPIDI_CH3_PRE_H_INCLUDED) */
