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

#ifndef _IBV_PARAM_H
#define _IBV_PARAM_H

#include <infiniband/verbs.h>
#include "debug_utils.h"
#include "mv2_arch_hca_detect.h"

extern int mv2_enable_eager_threshold_reduction;
#if defined(_SHARP_SUPPORT_)
extern int mv2_enable_sharp_coll;
extern int mv2_sharp_port;
extern char * mv2_sharp_hca_name;
#endif
extern int mv2_is_in_finalize;
/* Support multiple QPs/port, multiple ports, multiple HCAs and combinations */
extern int rdma_num_hcas;
extern int rdma_num_req_hcas;
extern int rdma_num_ports;
extern int rdma_num_qp_per_port;
extern int rdma_num_rails;
extern int mv2_cm_wait_time;

extern unsigned long rdma_default_max_cq_size;
extern int rdma_num_cqes_per_poll;
extern int rdma_default_port;
extern int rdma_default_gid_index;
extern int rdma_default_max_send_wqe;
extern int rdma_default_max_recv_wqe;
extern uint32_t rdma_default_max_sg_list;
extern uint16_t rdma_default_pkey_ix;
extern uint16_t rdma_default_pkey;
extern uint8_t rdma_default_qp_ous_rd_atom;
extern uint8_t rdma_supported_max_qp_ous_rd_atom;
extern uint8_t rdma_default_max_rdma_dst_ops;
extern uint8_t rdma_supported_max_rdma_dst_ops;
extern enum ibv_mtu rdma_default_mtu;
extern uint32_t rdma_default_psn;
extern uint8_t rdma_default_min_rnr_timer;
extern uint8_t rdma_default_service_level;
extern uint8_t rdma_default_static_rate;
extern uint8_t rdma_default_src_path_bits;
extern uint8_t rdma_default_time_out;
extern uint8_t rdma_default_retry_count;
extern uint8_t rdma_default_rnr_retry;
extern int rdma_default_put_get_list_size;
extern float rdma_credit_update_threshold;
extern int num_rdma_buffer;
extern int rdma_iba_eager_threshold;
extern unsigned int rdma_ndreg_entries;
extern int rdma_vbuf_max;
extern int rdma_vbuf_pool_size;
extern int rdma_vbuf_secondary_pool_size;
extern int rdma_initial_prepost_depth;
extern int rdma_prepost_depth;
extern int rdma_prepost_threshold;
extern int rdma_prepost_noop_extra;
extern int rdma_initial_credits;
extern int rdma_prepost_rendezvous_extra;
extern int rdma_dynamic_credit_threshold;
extern int rdma_credit_notify_threshold;
extern int rdma_credit_preserve;
extern int rdma_rq_size;
extern unsigned long rdma_dreg_cache_limit;
extern int rdma_rndv_protocol;
extern int smp_rndv_protocol;
extern int rdma_rndv_immediate;
extern int rdma_r3_threshold;
extern int rdma_intra_node_r3_threshold;
extern int rdma_inter_node_r3_threshold;
extern int rdma_r3_threshold_nocache;
extern int rdma_max_r3_pending_data;
extern int rdma_vbuf_total_size;
extern int rdma_max_inline_size;
extern int rdma_local_id;
extern int rdma_num_local_procs;

extern uint32_t mv2_srq_alloc_size;
extern uint32_t mv2_srq_fill_size;
extern uint32_t mv2_srq_limit;
extern uint32_t mv2_max_r3_oust_send;

extern int rdma_polling_set_threshold;
extern int rdma_polling_set_limit;
extern int rdma_fp_buffer_size;
extern int rdma_fp_sendconn_accepted;
extern int rdma_pending_conn_request;
extern int rdma_eager_limit;
extern int rdma_rndv_ext_sendq_size;
extern int rdma_global_ext_sendq_size;
extern int rdma_num_extra_polls;

extern int rdma_pin_pool_size;
extern int rdma_put_fallback_threshold;
extern int rdma_get_fallback_threshold;
extern int rdma_iba_eager_threshold;
extern long rdma_eagersize_1sc;
extern int rdma_qos_num_sls;
extern int rdma_use_qos;
extern int rdma_3dtorus_support;
extern int rdma_path_sl_query;
extern int rdma_num_sa_query_retries;
extern int rdma_multirail_usage_policy;
extern int rdma_small_msg_rail_sharing_policy;
extern int rdma_med_msg_rail_sharing_policy;
extern int rdma_med_msg_rail_sharing_threshold;
extern int rdma_large_msg_rail_sharing_threshold;


extern int mv2_use_pmi_ibarrier;
extern int mv2_use_pmi_iallgather;
extern int mv2_on_demand_ud_info_exchange;
extern int mv2_homogeneous_cluster;
extern int mv2_show_env_info;
extern int mv2_use_pmi_ibarrier;
extern int mv2_shmem_backed_ud_cm;
extern int mv2_show_runlog_level;
/* HSAM Definitions */

extern int striping_threshold;
extern int rdma_rail_sharing_policy;
extern int alpha;
extern int stripe_factor;
extern int apm_tester;

extern int rdma_coalesce_threshold;
extern int rdma_use_coalesce;

extern int rdma_use_blocking;
extern unsigned long rdma_blocking_spin_count_threshold;
extern unsigned long rdma_polling_spin_count_threshold;
extern int mv2_use_thread_yield;
extern int mv2_spins_before_lock;
extern int rdma_use_xrc;
extern int xrc_rdmafp_init;
extern int rdma_use_smp;
extern int use_iboeth;
extern int rdma_iwarp_multiple_cq_threshold;
extern int rdma_iwarp_use_multiple_cq;
extern int using_mpirun_rsh;

extern int use_hwloc_cpu_binding;
extern int max_rdma_connect_attempts;
#ifdef _MULTI_SUBNET_SUPPORT_
extern int mv2_rdma_cm_multi_subnet_support;
#endif /* _MULTI_SUBNET_SUPPORT_ */
extern int rdma_cm_connect_retry_interval;
extern int rdma_num_rails_per_hca;
extern int rdma_process_binding_rail_offset;
extern int g_atomics_support;
extern int g_atomics_support_be;

/* Use of LIMIC of RMA Communication */
extern int limic_put_threshold;
extern int limic_get_threshold;

extern int rdma_enable_hugepage;

#ifdef _ENABLE_CUDA_
extern int rdma_cuda_block_size;
extern int rdma_num_cuda_rndv_blocks;
extern int rdma_cuda_event_count;
extern int rdma_enable_cuda;
extern int rdma_cuda_dynamic_init;
extern int rdma_cuda_vec_thread_blksz;
extern int rdma_cuda_vec_thread_ysz;
extern int rdma_cuda_subarr_thread_blksz;
extern int rdma_cuda_subarr_thread_xdim;
extern int rdma_cuda_subarr_thread_ydim;
extern int rdma_cuda_subarr_thread_zdim;
extern int rdma_cuda_nonblocking_streams;
extern int rdma_eager_cudahost_reg;
extern int rdma_cuda_vector_dt_opt;
extern int rdma_cuda_kernel_dt_opt;
extern int cuda_initialized; 
#if defined(HAVE_CUDA_IPC)
extern int rdma_cuda_ipc;
extern int rdma_enable_ipc_share_gpu;
extern int rdma_cuda_smp_ipc;
extern int rdma_cuda_enable_ipc_cache;
extern int rdma_cuda_ipc_threshold;
extern int cudaipc_cache_max_entries;
#endif /*#if defined(HAVE_CUDA_IPC) */
extern int rdma_cuda_use_naive;
extern int rdma_cuda_register_naive_buf;
extern int rdma_cuda_gather_naive_limit;
extern int rdma_cuda_scatter_naive_limit;
extern int rdma_cuda_gatherv_naive_limit;
extern int rdma_cuda_scatterv_naive_limit;
extern int rdma_cuda_allgather_naive_limit;
extern int rdma_cuda_allgatherv_naive_limit;
extern int rdma_cuda_alltoall_naive_limit;
extern int rdma_cuda_alltoallv_naive_limit;
extern int rdma_cuda_bcast_naive_limit;
extern int rdma_cuda_alltoall_dynamic;
extern int rdma_cuda_allgather_rd_limit;
extern int rdma_cuda_allgather_fgp;
extern int rdma_cuda_init_context;
extern int rdma_check_cuda_attribute;
#endif /*#ifdef _ENABLE_CUDA_ */


#define MV2_DEFAULT_UD_MTU 2048
extern uint16_t rdma_default_ud_mtu;
#if defined(_ENABLE_UD_)
extern uint8_t rdma_enable_hybrid;
extern uint8_t rdma_enable_only_ud;
extern uint8_t rdma_use_ud_zcopy;
extern uint32_t rdma_hybrid_enable_threshold;
extern uint32_t rdma_default_max_ud_send_wqe;
extern uint32_t rdma_default_max_ud_recv_wqe;
extern uint32_t rdma_default_ud_sendwin_size;
extern uint32_t rdma_default_ud_recvwin_size;
extern long rdma_ud_progress_timeout;
extern long rdma_ud_retry_timeout;
extern long rdma_ud_max_retry_timeout;
extern long rdma_ud_last_check;
extern uint16_t rdma_ud_max_retry_count;
extern uint16_t rdma_ud_progress_spin;
extern uint16_t rdma_ud_max_ack_pending;
extern uint16_t rdma_ud_num_rndv_qps;
extern uint32_t rdma_ud_num_msg_limit;
extern uint32_t rdma_ud_vbuf_pool_size;
extern uint32_t rdma_ud_zcopy_threshold;
extern uint32_t rdma_ud_zcopy_rq_size;
extern uint16_t rdma_hybrid_max_rc_conn;
extern uint16_t rdma_hybrid_pending_rc_conn;
#ifdef _MV2_UD_DROP_PACKET_RATE_
extern uint32_t ud_drop_packet_rate;
#endif
#endif
#if defined(_MCST_SUPPORT_)
extern uint32_t mcast_bcast_min_msg;
extern uint32_t mcast_bcast_max_msg;
extern uint8_t rdma_enable_mcast;
#if defined(RDMA_CM)
extern uint8_t rdma_use_rdma_cm_mcast;
#endif /*defined(RDMA_CM)*/
extern uint8_t mcast_enable_rel;
extern uint8_t mcast_use_mcast_nack;
extern uint16_t mcast_window_size;
extern uint16_t mcast_drop_packet_rate;
extern uint32_t mcast_num_nodes_threshold;
extern uint32_t mcast_max_ud_recv_wqe;
extern long mcast_retry_timeout;
extern long mcast_max_retry_timeout;
extern long mcast_comm_init_timeout;
extern int mcast_comm_init_retries;
extern int mcast_nspin_threshold;
extern int mcast_skip_loopback;
#endif
extern int mv2_enable_progress_affinity;
extern int mv2_use_eager_fast_send;
extern int mv2_rdma_fast_path_preallocate_buffers;

extern int rdma_default_async_thread_stack_size;

#define PKEY_MASK 0x7fff        /* the last bit is reserved */
#define RDMA_MAX_CQE_ENTRIES_PER_POLL   (96)
#define RDMA_PIN_POOL_SIZE              (2*1024*1024)
#define RDMA_DEFAULT_MAX_CQ_SIZE        (40000)
#define RDMA_DEFAULT_IWARP_CQ_SIZE      (8192)
#define RDMA_DEFAULT_PORT               (-1)
#define RDMA_DEFAULT_GID_INDEX          (0)
#define RDMA_DEFAULT_MAX_PORTS          (2)
#define RDMA_DEFAULT_MAX_SEND_WQE       (64)
#define RDMA_DEFAULT_MAX_RECV_WQE       (128)
#define RDMA_DEFAULT_MAX_UD_SEND_WQE    (2048)
#define RDMA_DEFAULT_MAX_UD_RECV_WQE    (4096)
#define RDMA_UD_NUM_MSG_LIMIT           (4096)
#define RDMA_DEFAULT_MAX_SG_LIST        (1)
#define RDMA_DEFAULT_PKEY_IX            (0)
#define RDMA_DEFAULT_PKEY               (0x0)
#define RDMA_DEFAULT_MAX_RDMA_DST_OPS   (4)
#define RDMA_DEFAULT_PSN                (0)
#define RDMA_DEFAULT_MIN_RNR_TIMER      (12)
#define RDMA_DEFAULT_SERVICE_LEVEL      (0)
#define RDMA_DEFAULT_STATIC_RATE        (0)
#define RDMA_DEFAULT_SRC_PATH_BITS      (0)
#define RDMA_DEFAULT_TIME_OUT          (20)
#define RDMA_DEFAULT_RETRY_COUNT        (7)
#define RDMA_DEFAULT_RNR_RETRY          (7)
#define RDMA_DEFAULT_PUT_GET_LIST_SIZE  (200)
#define RDMA_IBA_NULL_HCA               "nohca"
#define RDMA_DEFAULT_POLLING_SET_LIMIT  (64)
#define RDMA_FP_DEFAULT_BUF_SIZE        (4096)
#define MAX_NUM_HCAS                    (4)
#ifndef MAX_NUM_PORTS
#define MAX_NUM_PORTS                   (2)
#endif
#ifndef MAX_NUM_QP_PER_PORT
#define MAX_NUM_QP_PER_PORT             (2)
#endif
#define RDMA_QOS_MAX_NUM_SLS	        (15)
#define RDMA_QOS_DEFAULT_NUM_SLS	    (8)
#define RDMA_DEFAULT_NUM_SA_QUERY_RETRIES   (20)
#define RDMA_DEFAULT_MED_MSG_RAIL_SHARING_THRESHOLD (2048)
#define RDMA_DEFAULT_LARGE_MSG_RAIL_SHARING_THRESHOLD (16384)
#define DEF_MV2_CM_WAIT_TIME            (5)
#define RDMA_DEFAULT_QP_OUS_RD_ATOM     (1)
#define DEFAULT_SHMEM_PRIORITY_FACTOR   (1)
#define RDMA_DEFAULT_R3_THRESHOLD       (4096)
#define RDMA_DEFAULT_INTER_NODE_R3_THRESHOLD       (4096)
#define RDMA_DEFAULT_INTRA_NODE_R3_THRESHOLD       (1)

/* This is a overprovision of resource, do not use in critical structures */
#define MAX_NUM_SUBRAILS                (MAX_NUM_HCAS*  \
                                         MAX_NUM_PORTS* \
                                         MAX_NUM_QP_PER_PORT)

#define RDMA_NDREG_ENTRIES              (1100)
#define RDMA_NDREG_ENTRIES_MAX          (4096)
#define RDMA_VBUF_POOL_SIZE             (512)
#define RDMA_OPT_VBUF_POOL_SIZE         (80)
#define RDMA_UD_VBUF_POOL_SIZE          (8192)
#define RDMA_MIN_VBUF_POOL_SIZE         (512)
#define RDMA_OPT_MIN_VBUF_POOL_SIZE     (32)
#define RDMA_VBUF_SECONDARY_POOL_SIZE   (256)
#define RDMA_OPT_VBUF_SECONDARY_POOL_SIZE   (16)
#define RDMA_PREPOST_DEPTH              (64)
#define RDMA_INITIAL_PREPOST_DEPTH      (10)
#define RDMA_LOW_WQE_THRESHOLD          (10)
#define RDMA_MAX_RDMA_SIZE              (4194304)
#define DEFAULT_RDMA_CONNECT_ATTEMPTS   (20)
#define RDMA_DEFAULT_CONNECT_INTERVAL   (100)

#define DEFAULT_SMALL_VBUF_SIZE          (256)
#define DEFAULT_MEDIUM_VBUF_SIZE         (5120)

#ifdef _ENABLE_CUDA_
#define DEFAULT_CUDA_VBUF_SIZES          {DEFAULT_SMALL_VBUF_SIZE, DEFAULT_MEDIUM_VBUF_SIZE, rdma_vbuf_total_size, rdma_cuda_block_size, rdma_cuda_block_size}
#define DEFAULT_CUDA_VBUF_POOL_SIZE      {rdma_vbuf_pool_size, rdma_vbuf_pool_size, rdma_vbuf_pool_size, rdma_vbuf_pool_size, rdma_vbuf_pool_size}
#define DEFAULT_CUDA_VBUF_SECONDARY_POOL_SIZE {rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size}
#define DEFAULT_CUDA_BLOCK_SIZE          (262144)
#endif

#define DEFAULT_VBUF_SIZES               {DEFAULT_SMALL_VBUF_SIZE, DEFAULT_MEDIUM_VBUF_SIZE, rdma_vbuf_total_size, rdma_vbuf_total_size}
#define DEFAULT_VBUF_POOL_SIZE           {rdma_vbuf_pool_size, rdma_vbuf_pool_size, rdma_vbuf_pool_size, rdma_vbuf_pool_size}
#define DEFAULT_VBUF_SECONDARY_POOL_SIZE {rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size, rdma_vbuf_secondary_pool_size}

#define RDMA_IWARP_DEFAULT_MULTIPLE_CQ_THRESHOLD  (32)
#define RDMA_DEFAULT_ASYNC_THREAD_STACK_SIZE  (1<<20)

/* Inline not supported for PPC */
#define HOSTNAME_LEN                    (255)
#define RDMA_MAX_REGISTERED_PAGES       (0)

/* #define MIN(a,b) ((a)<(b)?(a):(b)) */

#define NUM_BOOTSTRAP_BARRIERS  2

/* Statistically sending a stripe below this may not lead
 * to benefit */
#define STRIPING_THRESHOLD              8 * 1024
extern char rdma_iba_hcas[MAX_NUM_HCAS][32];

typedef enum _mv2_vbuf_pool_offsets {
    MV2_SMALL_DATA_VBUF_POOL_OFFSET = 0,
    MV2_MEDIUM_DATA_VBUF_POOL_OFFSET,
    MV2_LARGE_DATA_VBUF_POOL_OFFSET,
    MV2_RECV_VBUF_POOL_OFFSET,
#ifdef _ENABLE_CUDA_
    MV2_CUDA_VBUF_POOL_OFFSET,
#endif /*_ENABLE_CUDA_*/
    MV2_MAX_NUM_VBUF_POOLS
} mv2_vbuf_pool_offsets;

typedef enum _mv2_iba_network_classes {
    MV2_NETWORK_CLASS_UNKNOWN = 0,
    MV2_NETWORK_CLASS_IB = 1,
    MV2_NETWORK_CLASS_IWARP,
} mv2_iba_network_classes;

/* Below ROUND_ROBIN refers to the rails where the rails are alternately
 * given to any process asking for it. Where as FIXED_MAPPING refers
 * to a scheduling policy where processes are bound to rails in a round
 * robin manner. So once a process is bound to a rail it will use only
 * that rail to send out messages */

typedef enum _mv2_multirail_policies {
    MV2_MRAIL_BINDING = 0,
    MV2_MRAIL_SHARING,
} mv2_multirail_policies;

typedef enum _mv2_rail_sharing_policies {
    ROUND_ROBIN = 0,
    USE_FIRST,
    EVEN_STRIPING,
    ADAPTIVE_STRIPING,
    FIXED_MAPPING,
    PARTIAL_ADAPTIVE,
    BEST_ADAPTIVE
} mv2_rail_sharing_policies;

/* This is to allow users to specify rail mapping at run time */
extern int mrail_use_default_mapping;
extern int mrail_user_defined_p2r_mapping;
extern char *mrail_p2r_string;
extern int mrail_p2r_length;

#define DYNAMIC_TOTAL_WEIGHT            (3* 1024)

#define CHELSIO_RNIC                    "cxgb"
#define INTEL_NE020_RNIC                "nes0"

/* MV2_POLLING_LEVEL
Level 1 : Exit on finding a message on any channel
Level 2 : Exit on finding a message on RDMA_FP or SMP channel.
          Continue on ibv_poll_cq success.
Level 3 : Exit on finding a message on RDMA_FP channel.
          Continue polling on SMP and ibv_poll_cq channels
          until no more messages.
Level 4 : Exit only after processing all the messages on
          all the channels
*/
typedef enum mv2_polling_level {
    MV2_POLLING_LEVEL_1 = 1,
    MV2_POLLING_LEVEL_2,
    MV2_POLLING_LEVEL_3,
    MV2_POLLING_LEVEL_4,
} mv2_polling_level;

extern mv2_polling_level rdma_polling_level;

/* enum list of MV2 runtime environment variables */
typedef enum mv2_env_param_id {
    /* mpirun_rsh */
    MV2_COMM_WORLD_LOCAL_RANK,
    PMI_ID,
    MPIRUN_COMM_MULTIPLE,
    MPIRUN_RSH_LAUNCH,
    MPISPAWN_BINARY_PATH,
    MPISPAWN_CR_CKPT_CNT,
    MPISPAWN_CR_CONTEXT,
    MPISPAWN_CR_SESSIONID,
    MPISPAWN_GLOBAL_NPROCS,
    MPISPAWN_MPIRUN_CR_PORT,
    MPISPAWN_MPIRUN_HOST,
    MPISPAWN_MPIRUN_ID,
    MPISPAWN_NNODES,
    MPISPAWN_WORKING_DIR,
    MPIEXEC_TIMEOUT,
    MPISPAWN_USE_TOTALVIEW,
    MV2_FASTSSH_THRESHOLD,
    MV2_MPIRUN_TIMEOUT,
    MV2_MT_DEGREE,
    MV2_NPROCS_THRESHOLD,
    USE_LINEAR_SSH,
    PMI_SUBVERSION,
    PMI_VERSION,
    PMI_PORT,
    PARENT_ROOT_PORT_NAME,
    MV2_SHMEM_BACKED_UD_CM,
    /* QoS */
    MV2_3DTORUS_SUPPORT,
    MV2_NUM_SA_QUERY_RETRIES,
    MV2_NUM_SLS,
    MV2_DEFAULT_SERVICE_LEVEL,
    MV2_PATH_SL_QUERY,
    MV2_USE_QOS,
    /* collectives */
    MV2_USE_MCAST,
    MV2_USE_RDMA_CM_MCAST,
    MV2_MCAST_BCAST_MIN_MSG,
    MV2_MCAST_BCAST_MAX_MSG,
    MV2_ALLGATHER_BRUCK_THRESHOLD,
    MV2_ALLGATHER_RD_THRESHOLD,
    MV2_ALLGATHER_REVERSE_RANKING,
    MV2_ALLGATHERV_RD_THRESHOLD,
    MV2_ALLREDUCE_2LEVEL_MSG,
    MV2_ALLREDUCE_SHORT_MSG,
    MV2_ALLTOALL_MEDIUM_MSG,
    MV2_ALLTOALL_SMALL_MSG,
    MV2_ALLTOALL_THROTTLE_FACTOR,
    MV2_BCAST_TWO_LEVEL_SYSTEM_SIZE,
    MV2_GATHER_SWITCH_PT,
    MV2_INTRA_SHMEM_REDUCE_MSG,
    MV2_KNOMIAL_2LEVEL_BCAST_MESSAGE_SIZE_THRESHOLD,
    MV2_KNOMIAL_2LEVEL_BCAST_SYSTEM_SIZE_THRESHOLD,
    MV2_KNOMIAL_INTER_LEADER_THRESHOLD,
    MV2_KNOMIAL_INTER_NODE_FACTOR,
    MV2_KNOMIAL_INTRA_NODE_FACTOR,
    MV2_KNOMIAL_INTRA_NODE_THRESHOLD,
    MV2_RED_SCAT_LARGE_MSG,
    MV2_RED_SCAT_SHORT_MSG,
    MV2_REDUCE_2LEVEL_MSG,
    MV2_REDUCE_SHORT_MSG,
    MV2_SCATTER_MEDIUM_MSG,
    MV2_SCATTER_SMALL_MSG,
    MV2_SHMEM_ALLREDUCE_MSG,
    MV2_SHMEM_COLL_MAX_MSG_SIZE,
    MV2_SHMEM_COLL_NUM_COMM,
    MV2_SHMEM_COLL_NUM_PROCS,
    MV2_SHMEM_COLL_SPIN_COUNT,
    MV2_SHMEM_DIR,
    MV2_SHMEM_REDUCE_MSG,
    MV2_USE_BCAST_SHORT_MSG,
    MV2_USE_DIRECT_GATHER,
    MV2_USE_DIRECT_GATHER_SYSTEM_SIZE_MEDIUM,
    MV2_USE_DIRECT_GATHER_SYSTEM_SIZE_SMALL,
    MV2_USE_DIRECT_SCATTER,
    MV2_USE_OSU_COLLECTIVES,
    MV2_USE_OSU_NB_COLLECTIVES,
    MV2_USE_KNOMIAL_2LEVEL_BCAST,
    MV2_USE_KNOMIAL_INTER_LEADER_BCAST,
    MV2_USE_SCATTER_RD_INTER_LEADER_BCAST,
    MV2_USE_SCATTER_RING_INTER_LEADER_BCAST,
    MV2_USE_SHMEM_ALLREDUCE,
    MV2_USE_SHMEM_BARRIER,
    MV2_USE_SHMEM_BCAST,
    MV2_USE_SHMEM_COLL,
    MV2_USE_SHMEM_REDUCE,
    MV2_USE_TWO_LEVEL_GATHER,
    MV2_USE_TWO_LEVEL_SCATTER,
    MV2_USE_XOR_ALLTOALL,
    MV2_ENABLE_SOCKET_AWARE_COLLECTIVES,
    MV2_USE_SOCKET_AWARE_ALLREDUCE,
    MV2_USE_SOCKET_AWARE_BARRIER,
    MV2_USE_SOCKET_AWARE_SHARP_ALLREDUCE,
    MV2_SOCKET_AWARE_ALLREDUCE_MAX_MSG,
    MV2_SOCKET_AWARE_ALLREDUCE_MIN_MSG,
    /* ckpt */
    MV2_CKPT_AGGRE_MIG_ROLE,
    MV2_CKPT_AGGREGATION_BUFPOOL_SIZE,
    MV2_CKPT_AGGREGATION_CHUNK_SIZE,
    MV2_CKPT_AGGRE_MIG_FILE,
    MV2_CKPT_FILE,
    MV2_CKPT_INTERVAL,
    MV2_CKPT_MAX_CKPTS,
    MV2_CKPT_MAX_SAVE_CKPTS,
    MV2_CKPT_MPD_BASE_PORT,
    MV2_CKPT_NO_SYNC,
    MV2_CKPT_SESSIONID,
    MV2_CKPT_USE_AGGREGATION,
    /*start up */
    MV2_FORCE_HCA_TYPE,
    MV2_FORCE_ARCH_TYPE,
    MV2_CM_MAX_SPIN_COUNT,
    MV2_CM_RECV_BUFFERS,
    MV2_CM_SEND_DEPTH,
    MV2_CM_TIMEOUT,
    MV2_CM_UD_PSN,
    MV2_DEFAULT_SRC_PATH_BITS,
    MV2_DEFAULT_STATIC_RATE,
    MV2_DEFAULT_TIME_OUT,
    MV2_DEFAULT_MTU,
    MV2_DEFAULT_PKEY,
    MV2_DEFAULT_PORT,
    MV2_DEFAULT_GID_INDEX,
    MV2_DEFAULT_PSN,
    MV2_DEFAULT_MAX_RECV_WQE,
    MV2_DEFAULT_MAX_SEND_WQE,
    MV2_DEFAULT_MAX_SG_LIST,
    MV2_DEFAULT_MIN_RNR_TIMER,
    MV2_DEFAULT_QP_OUS_RD_ATOM,
    MV2_DEFAULT_RETRY_COUNT,
    MV2_DEFAULT_RNR_RETRY,
    MV2_DEFAULT_MAX_CQ_SIZE,
    MV2_DEFAULT_MAX_RDMA_DST_OPS,
    MV2_IGNORE_SYSTEM_CONFIG,
    MV2_IGNORE_USER_CONFIG,
    MV2_INITIAL_PREPOST_DEPTH,
    MV2_IBA_HCA,
    MV2_IWARP_MULTIPLE_CQ_THRESHOLD,
    MV2_NUM_HCAS,
    MV2_NUM_NODES_IN_JOB,
    MV2_NUM_PORTS,
    MV2_NUM_QP_PER_PORT,
    MV2_MAX_RDMA_CONNECT_ATTEMPTS,
    MV2_ON_DEMAND_THRESHOLD,
    MV2_ON_DEMAND_UD_INFO_EXCHANGE,
    MV2_PREPOST_DEPTH,
    MV2_USER_CONFIG,
    MV2_USE_RING_STARTUP,
    MV2_HOMOGENEOUS_CLUSTER,
    MV2_UNIVERSE_SIZE,
    /* pt-pt */
    MV2_NUM_CQES_PER_POLL,
    MV2_COALESCE_THRESHOLD,
    MV2_DREG_CACHE_LIMIT,
    MV2_IBA_EAGER_THRESHOLD,
    MV2_MAX_INLINE_SIZE,
    MV2_MAX_R3_PENDING_DATA,
    MV2_MED_MSG_RAIL_SHARING_POLICY,
    MV2_NDREG_ENTRIES,
    MV2_NUM_RDMA_BUFFER,
    MV2_NUM_SPINS_BEFORE_LOCK,
    MV2_POLLING_LEVEL,
    MV2_POLLING_SET_LIMIT,
    MV2_POLLING_SET_THRESHOLD,
    MV2_PROCESS_TO_RAIL_MAPPING,
    MV2_R3_NOCACHE_THRESHOLD,
    MV2_R3_THRESHOLD,
    MV2_RAIL_SHARING_LARGE_MSG_THRESHOLD,
    MV2_RAIL_SHARING_MED_MSG_THRESHOLD,
    MV2_RAIL_SHARING_POLICY,
    MV2_RDMA_EAGER_LIMIT,
    MV2_RDMA_FAST_PATH_BUF_SIZE,
    MV2_RDMA_NUM_EXTRA_POLLS,
    MV2_RNDV_EXT_SENDQ_SIZE,
    MV2_RNDV_PROTOCOL,
    MV2_SMP_RNDV_PROTOCOL,
    MV2_SMALL_MSG_RAIL_SHARING_POLICY,
    MV2_SM_SCHEDULING,
    MV2_SPIN_COUNT,
    MV2_SRQ_LIMIT,
    MV2_SRQ_MAX_SIZE,
    MV2_SRQ_SIZE,
    MV2_STRIPING_THRESHOLD,
    MV2_USE_BLOCKING,
    MV2_USE_COALESCE,
    MV2_USE_LAZY_MEM_UNREGISTER,
    MV2_USE_RDMA_FAST_PATH,
    MV2_USE_SRQ,
    MV2_USE_XRC,
    MV2_VBUF_MAX,
    MV2_VBUF_POOL_SIZE,
    MV2_VBUF_SECONDARY_POOL_SIZE,
    MV2_VBUF_TOTAL_SIZE,
    MV2_USE_IWARP_MODE,
    MV2_USE_RoCE,
    /* smp */
    MV2_CPU_BINDING_POLICY,
    MV2_CPU_BINDING_LEVEL,
    MV2_USE_HWLOC_CPU_BINDING,
    MV2_CPU_MAPPING,
    MV2_ENABLE_AFFINITY,
    MV2_HCA_AWARE_PROCESS_MAPPING,
    MV2_ENABLE_LEASTLOAD,
    MV2_LIMIC_GET_THRESHOLD,
    MV2_LIMIC_PUT_THRESHOLD,
    MV2_SMP_USE_LIMIC2,
    MV2_USE_LIMIC2_COLL,
    MV2_SMP_USE_CMA,
    MV2_SMP_BATCH_SIZE,
    MV2_SMP_EAGERSIZE,
    MV2_SMP_QUEUE_LENGTH,
    MV2_SMP_NUM_SEND_BUFFER,
    MV2_SMP_SEND_BUF_SIZE,
    MV2_USE_SHARED_MEM,
    MV2_SMP_CMA_MAX_SIZE,
    MV2_SMP_LIMIC2_MAX_SIZE,
    /* cuda */
    MV2_CUDA_BLOCK_SIZE,
    MV2_CUDA_NUM_RNDV_BLOCKS,
    MV2_CUDA_NUM_STREAMS,
    MV2_CUDA_VECTOR_OPT,
    MV2_CUDA_KERNEL_OPT,
    MV2_EAGER_CUDAHOST_REG,
    MV2_USE_CUDA,
    MV2_CUDA_NUM_EVENTS,
    MV2_CUDA_IPC,
    MV2_CUDA_IPC_THRESHOLD,
    MV2_CUDA_ENABLE_IPC_CACHE,
    MV2_CUDA_IPC_MAX_CACHE_ENTRIES,
    MV2_CUDA_IPC_NUM_STAGE_BUFFERS,
    MV2_CUDA_IPC_STAGE_BUF_SIZE,
    MV2_CUDA_IPC_BUFFERED,
    MV2_CUDA_IPC_BUFFERED_LIMIT,
    MV2_CUDA_IPC_SYNC_LIMIT,
    MV2_CUDA_USE_NAIVE,
    MV2_CUDA_REGISTER_NAIVE_BUF,
    MV2_CUDA_GATHER_NAIVE_LIMIT,
    MV2_CUDA_SCATTER_NAIVE_LIMIT,
    MV2_CUDA_ALLGATHER_NAIVE_LIMIT,
    MV2_CUDA_ALLGATHERV_NAIVE_LIMIT,
    MV2_CUDA_ALLTOALL_NAIVE_LIMIT,
    MV2_CUDA_ALLTOALLV_NAIVE_LIMIT,
    MV2_CUDA_BCAST_NAIVE_LIMIT,
    MV2_CUDA_GATHERV_NAIVE_LIMIT,
    MV2_CUDA_SCATTERV_NAIVE_LIMIT,
    MV2_CUDA_ALLTOALL_DYNAMIC,
    MV2_CUDA_ALLGATHER_RD_LIMIT,
    MV2_CUDA_ALLGATHER_FGP,
    MV2_SMP_CUDA_PIPELINE,
    MV2_CUDA_INIT_CONTEXT,
    /* debug */
    MV2_DEBUG_CORESIZE,
    MV2_DEBUG_SHOW_BACKTRACE,
    MV2_ABORT_SLEEP_SECONDS,
    MV2_SHOW_ENV_INFO,
    MV2_SYSREPORT,
    TOTALVIEW,
    MV2_DEBUG_CM_VERBOSE,
    MV2_DEBUG_CUDA_VERBOSE,
    MV2_DEBUG_FORK_VERBOSE,
    MV2_DEBUG_FT_VERBOSE,
    MV2_DEBUG_MEM_USAGE_VERBOSE,
    MV2_DEBUG_MIG_VERBOSE,
    MV2_DEBUG_UDSTAT_VERBOSE,
    MV2_DEBUG_UD_VERBOSE,
    MV2_DEBUG_XRC_VERBOSE,
    MV2_DEBUG_ZCOPY_VERBOSE,
    /* one-sided */
    MV2_DEFAULT_PUT_GET_LIST_SIZE,
    MV2_EAGERSIZE_1SC,
    MV2_GET_FALLBACK_THRESHOLD,
    MV2_PIN_POOL_SIZE,
    MV2_PUT_FALLBACK_THRESHOLD,
    MV2_USE_LIMIC_ONE_SIDED,
    MV2_USE_RDMA_ONE_SIDED,
    MV2_USE_SHM_ONE_SIDED,
    /* rdma cm */
    MV2_RDMA_CM_ARP_TIMEOUT,
    MV2_RDMA_CM_CONNECT_RETRY_INTERVAL,
    MV2_RDMA_CM_MAX_PORT,
    MV2_RDMA_CM_MIN_PORT,
    MV2_RDMA_CM_PORT,
    MV2_USE_RDMA_CM,
    /* hybrid */
    MV2_UD_DROP_PACKET_RATE,
    MV2_UD_MAX_ACK_PENDING,
    MV2_UD_MAX_RECV_WQE,
    MV2_UD_MAX_RETRY_TIMEOUT,
    MV2_UD_MAX_SEND_WQE,
    MV2_UD_MTU,
    MV2_UD_NUM_MSG_LIMIT,
    MV2_UD_NUM_ZCOPY_RNDV_QPS,
    MV2_UD_PROGRESS_SPIN,
    MV2_UD_PROGRESS_TIMEOUT,
    MV2_UD_RECVWINDOW_SIZE,
    MV2_UD_RETRY_COUNT,
    MV2_UD_RETRY_TIMEOUT,
    MV2_UD_SENDWINDOW_SIZE,
    MV2_UD_VBUF_POOL_SIZE,
    MV2_UD_ZCOPY_RQ_SIZE,
    MV2_UD_ZCOPY_THRESHOLD,
    MV2_USE_UD_ZCOPY,
    MV2_USE_UD_HYBRID,
    MV2_USE_ONLY_UD,
    MV2_USE_UD_SRQ,
    MV2_HYBRID_ENABLE_THRESHOLD,
    MV2_HYBRID_MAX_RC_CONN,
    /* threads */
    MV2_ASYNC_THREAD_STACK_SIZE,
    MV2_CM_THREAD_STACKSIZE,
    MV2_THREAD_YIELD_SPIN_THRESHOLD,
    MV2_USE_THREAD_WARNING,
    MV2_USE_THREAD_YIELD,
    /* other */
    MV2_SUPPORT_DPM,
    MV2_USE_APM,
    MV2_USE_APM_TEST,
    MV2_USE_HSAM,
    MV2_USE_HUGEPAGES,
    MV2_MAX_PARAM_ID,
} mv2_env_param_id_t;

typedef enum mv2_env_param_type {
    MV2_PARAM_TYPE_INVALID = 0,
    MV2_PARAM_TYPE_INT8,
    MV2_PARAM_TYPE_INT16,
    MV2_PARAM_TYPE_INT,
    MV2_PARAM_TYPE_LONG,
    MV2_PARAM_TYPE_STRING,
} mv2_env_param_type_t;

/* parameter categories */
typedef enum mv2_env_param_group {
    MV2_PARAM_GROUP_launcher,
    MV2_PARAM_GROUP_QoS,
    MV2_PARAM_GROUP_collective,
    MV2_PARAM_GROUP_ckpt,
    MV2_PARAM_GROUP_startup,
    MV2_PARAM_GROUP_pt2pt,
    MV2_PARAM_GROUP_intranode,
    MV2_PARAM_GROUP_cuda,
    MV2_PARAM_GROUP_debugger,
    MV2_PARAM_GROUP_rma,
    MV2_PARAM_GROUP_rdma_cm,
    MV2_PARAM_GROUP_hybrid,
    MV2_PARAM_GROUP_threads,
    MV2_PARAM_GROUP_other,
    MV2_PARAM_NUM_GROUPS
} mv2_env_param_group_t;

/* runtime environment list structure */
typedef struct mv2_env_param_list {
    mv2_env_param_id_t id;      /* param id */
    mv2_env_param_type_t type;  /* param datatype */
    mv2_env_param_group_t group;        /* param category */
    char *name;                 /* param name */
    void *value;                /* param value store addr */
    int external_visible;       /* 1 or 0 */
    char *descrption;           /* param descrption */
} mv2_env_param_list_t;

extern mv2_env_param_list_t param_list[];
void mv2_show_all_params();
void mv2_show_runlog_info(int level);
void rdma_set_rdma_fast_path_params(int num_proc);
const char *mv2_ibv_mtu_enum_to_string(enum ibv_mtu mtu);
uint16_t mv2_ibv_mtu_enum_to_value(enum ibv_mtu mtu);
extern int rdma_get_rail_sharing_policy(char *value);

mv2_arch_hca_type MV2_get_arch_hca_type();


extern int dreg_max_use_count;
#endif /* _RDMA_PARAM_H */

/* default values of CVARs */
#define USE_MCAST_DEFAULT_FLAG           1
#define DEFAULT_NUM_PORTS                1
#define DEFAULT_NUM_QP_PER_PORT          1
#define DEFAULT_COALESCE_THRESHOLD       6
#define DEFAULT_USE_COALESCE             1
#define DEFAULT_SPIN_COUNT               5000
#define MAX_NUM_CQES_PER_POLL            96
#define MIN_NUM_CQES_PER_POLL            1

