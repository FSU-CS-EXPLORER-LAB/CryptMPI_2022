/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
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
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <infiniband/verbs.h>
#include "rdma_impl.h"
#include "vbuf.h"
#include "ibv_param.h"
#include "sysreport.h"
#include "smp_smpi.h"
#include "mv2_utils.h"
#include "upmi.h"
#include "ibv_mcast.h"
#include <inttypes.h>
#ifdef HAVE_ROMIO
#include "romioconf.h"
#endif
#include "coll_shmem.h"

/* Extra buffer space for header(s); used to adjust the eager-threshold */
#define EAGER_THRESHOLD_ADJUST    0
#define INLINE_THRESHOLD_ADJUST  (40)
extern const char MPIR_Version_string[];
extern unsigned int mv2_enable_affinity;
extern int g_mv2_num_cpus;
extern int rdma_skip_network_card(mv2_iba_network_classes network_type,
                           struct ibv_device *ib_dev);
static inline void rdma_get_vbuf_user_parameters(int num_proc, int me);

/*
 * ==============================================================
 * Initialize global parameter variables to default values
 * ==============================================================
 */

int mv2_enable_eager_threshold_reduction = 1;
#if defined(_SHARP_SUPPORT_)
int mv2_enable_sharp_coll = 0;
int mv2_sharp_port = -1;
char * mv2_sharp_hca_name = 0;
#endif
int mv2_num_extra_polls = 0;
int mv2_is_in_finalize = 0;
int mv2_cm_wait_time = DEF_MV2_CM_WAIT_TIME;
int rdma_num_cqes_per_poll = RDMA_MAX_CQE_ENTRIES_PER_POLL;
int rdma_num_hcas = 1;
int rdma_num_req_hcas = 0;
int rdma_num_ports = DEFAULT_NUM_PORTS;
int rdma_num_qp_per_port = DEFAULT_NUM_QP_PER_PORT;
int rdma_num_rails;
int rdma_memory_optimization = 1;
int rdma_pin_pool_size = RDMA_PIN_POOL_SIZE;
unsigned long rdma_default_max_cq_size = RDMA_DEFAULT_MAX_CQ_SIZE;
int rdma_default_port = RDMA_DEFAULT_PORT;
int rdma_default_gid_index = RDMA_DEFAULT_GID_INDEX;
int rdma_default_max_send_wqe = RDMA_DEFAULT_MAX_SEND_WQE;
int rdma_default_max_recv_wqe = RDMA_DEFAULT_MAX_RECV_WQE;
uint32_t rdma_default_max_sg_list = RDMA_DEFAULT_MAX_SG_LIST;
uint16_t rdma_default_pkey_ix = RDMA_DEFAULT_PKEY_IX;
uint16_t rdma_default_pkey = RDMA_DEFAULT_PKEY;
uint8_t rdma_default_qp_ous_rd_atom = RDMA_DEFAULT_QP_OUS_RD_ATOM;
uint8_t rdma_supported_max_qp_ous_rd_atom = RDMA_DEFAULT_QP_OUS_RD_ATOM;
uint8_t rdma_default_max_rdma_dst_ops = RDMA_DEFAULT_MAX_RDMA_DST_OPS;
uint8_t rdma_supported_max_rdma_dst_ops = RDMA_DEFAULT_MAX_RDMA_DST_OPS;
enum ibv_mtu rdma_default_mtu = IBV_MTU_1024;
uint32_t rdma_default_psn = RDMA_DEFAULT_PSN;
uint8_t rdma_default_min_rnr_timer = RDMA_DEFAULT_MIN_RNR_TIMER;
uint8_t rdma_default_service_level = RDMA_DEFAULT_SERVICE_LEVEL;
uint8_t rdma_default_static_rate = RDMA_DEFAULT_STATIC_RATE;
uint8_t rdma_default_src_path_bits = RDMA_DEFAULT_SRC_PATH_BITS;
uint8_t rdma_default_time_out = RDMA_DEFAULT_TIME_OUT;
uint8_t rdma_default_retry_count = RDMA_DEFAULT_RETRY_COUNT;
uint8_t rdma_default_rnr_retry = RDMA_DEFAULT_RNR_RETRY;
int rdma_default_put_get_list_size = RDMA_DEFAULT_PUT_GET_LIST_SIZE;
long rdma_eagersize_1sc;
int rdma_put_fallback_threshold;
int rdma_get_fallback_threshold;
int rdma_polling_set_limit = -1;
int rdma_polling_set_threshold = 256;
int rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
int rdma_fp_sendconn_accepted = 0;
int rdma_pending_conn_request = 0;
int rdma_eager_limit = 32;
int rdma_iba_eager_threshold;
char rdma_iba_hcas[MAX_NUM_HCAS][32];
int rdma_max_inline_size;
unsigned int rdma_ndreg_entries = 0;
unsigned int rdma_ndreg_entries_max = RDMA_NDREG_ENTRIES_MAX; 
int rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
int smp_rndv_protocol  = MV2_RNDV_PROTOCOL_RGET;
int rdma_rndv_immediate = 1;
int mv2_use_balanced_rndv = 1;
int rdma_r3_threshold = RDMA_DEFAULT_R3_THRESHOLD;
int rdma_intra_node_r3_threshold = RDMA_DEFAULT_INTRA_NODE_R3_THRESHOLD;
int rdma_inter_node_r3_threshold = RDMA_DEFAULT_INTER_NODE_R3_THRESHOLD;
int rdma_r3_threshold_nocache = 8192 * 4;
int rdma_max_r3_pending_data = 512 * 1024;
int num_rdma_buffer = 0;
int rdma_use_xrc = 0;
int xrc_rdmafp_init = 0;
int rdma_use_smp = 1;
int rdma_use_qos = 0;
#ifdef ENABLE_3DTORUS_SUPPORT
int rdma_3dtorus_support = 1;
#else
int rdma_3dtorus_support = 0;
#endif /* ENABLE_3DTORUS_SUPPORT */
int rdma_path_sl_query = 0;
int rdma_num_sa_query_retries = RDMA_DEFAULT_NUM_SA_QUERY_RETRIES;
MPID_Node_id_t rdma_num_nodes_in_job = 0;
int rdma_qos_num_sls = RDMA_QOS_DEFAULT_NUM_SLS;
int max_rdma_connect_attempts = DEFAULT_RDMA_CONNECT_ATTEMPTS;
#ifdef _MULTI_SUBNET_SUPPORT_
int mv2_rdma_cm_multi_subnet_support = 0;
#endif /* _MULTI_SUBNET_SUPPORT_ */
int rdma_cm_connect_retry_interval = RDMA_DEFAULT_CONNECT_INTERVAL;
int rdma_default_async_thread_stack_size = RDMA_DEFAULT_ASYNC_THREAD_STACK_SIZE;
int rdma_num_rails_per_hca = 1;
int rdma_process_binding_rail_offset = 0;
int rdma_multirail_usage_policy = MV2_MRAIL_BINDING;
int rdma_small_msg_rail_sharing_policy = ROUND_ROBIN;
int rdma_med_msg_rail_sharing_policy = ROUND_ROBIN;
int rdma_med_msg_rail_sharing_threshold =
    RDMA_DEFAULT_MED_MSG_RAIL_SHARING_THRESHOLD;
int rdma_large_msg_rail_sharing_threshold =
    RDMA_DEFAULT_LARGE_MSG_RAIL_SHARING_THRESHOLD;

/* This is to allow users to specify rail mapping at run time */
int mrail_user_defined_p2r_mapping = -1;
int mrail_p2r_length;
int mrail_use_default_mapping = 0;
char *mrail_p2r_string = NULL;
/* Threshold of job size beyond which we want to use 2-cq approach */
int rdma_iwarp_multiple_cq_threshold = RDMA_IWARP_DEFAULT_MULTIPLE_CQ_THRESHOLD;
int rdma_iwarp_use_multiple_cq = 0;
/* Force to use rendezvous if extended sendq size exceeds this value */
int rdma_rndv_ext_sendq_size = 5;
/* Global count of extended sendq size across all rails*/
int rdma_global_ext_sendq_size = 0;
/* Number of times to poll while global ext sendq has outstanding requests */
int rdma_num_extra_polls = 1;
int rdma_local_id = -1;
int rdma_num_local_procs = -1;
/* Whether coalescing of messages should be attempted */
int rdma_use_coalesce = DEFAULT_USE_COALESCE;
unsigned long rdma_polling_spin_count_threshold = 5;
int mv2_use_thread_yield = 1;
int mv2_spins_before_lock = 2000;
int mv2_on_demand_ud_info_exchange = 1;
int mv2_homogeneous_cluster = 0;
int mv2_show_env_info = 0;
int mv2_use_pmi_ibarrier = 0;
int mv2_use_pmi_iallgather = 0;
int mv2_shmem_backed_ud_cm = 1;

/* If this number of eager sends are already outstanding
 * the message can be coalesced with other messages (and
 * will not be sent until a previous message completes)
 */
int rdma_coalesce_threshold = DEFAULT_COALESCE_THRESHOLD;

/* max (total) number of vbufs to allocate, after which process
 * terminates with a fatal error.
 * -1 means no limit.
 */
int rdma_vbuf_max = -1;
/* number of vbufs to allocate in a secondary region if we should
 * run out of the initial allocation.  This is re-computed (below)
 * once other parameters are known.
 */
int rdma_vbuf_secondary_pool_size = RDMA_VBUF_SECONDARY_POOL_SIZE;

/* number of vbufs to allocate initially.
 * This will be re-defined after reading the parameters below
 * to scale to the number of VIs and other factors.
 */
int rdma_vbuf_pool_size = RDMA_VBUF_POOL_SIZE;
int rdma_prepost_depth = RDMA_PREPOST_DEPTH;
int rdma_initial_prepost_depth = RDMA_INITIAL_PREPOST_DEPTH;

/* allow some extra buffers for non-credited packets (eg. NOOP) */
int rdma_prepost_noop_extra = 6;
int rdma_credit_preserve;
int rdma_initial_credits = 0;

uint16_t rdma_default_ud_mtu = 0;
#if defined(_ENABLE_UD_)
uint8_t rdma_enable_hybrid = 1;
uint8_t rdma_enable_only_ud = 0;
uint8_t rdma_use_ud_zcopy = 1;
uint32_t rdma_default_max_ud_send_wqe = RDMA_DEFAULT_MAX_UD_SEND_WQE;
uint32_t rdma_default_max_ud_recv_wqe = RDMA_DEFAULT_MAX_UD_RECV_WQE;
uint32_t rdma_ud_num_msg_limit = RDMA_UD_NUM_MSG_LIMIT;
uint32_t rdma_ud_vbuf_pool_size = RDMA_UD_VBUF_POOL_SIZE;
/* Maximum number of outstanding buffers (waiting for ACK)*/
uint32_t rdma_default_ud_sendwin_size = 400;
/* Maximum number of out-of-order messages that will be buffered */
uint32_t rdma_default_ud_recvwin_size = 2501;
/* Time (usec) until ACK status is checked (and ACKs are sent) */
long rdma_ud_progress_timeout = 48000;
/* Time (usec) until a message is resent */
long rdma_ud_retry_timeout = 500000;
long rdma_ud_max_retry_timeout = 20000000;
long rdma_ud_last_check;
uint32_t rdma_ud_zcopy_threshold;
uint32_t rdma_ud_zcopy_rq_size = 4096;
uint32_t rdma_hybrid_enable_threshold = 1024;
uint16_t rdma_ud_progress_spin = 1200;
uint16_t rdma_ud_max_retry_count = 1000;
uint16_t rdma_ud_max_ack_pending;
uint16_t rdma_ud_num_rndv_qps = 64;
uint16_t rdma_hybrid_max_rc_conn = 64;
uint16_t rdma_hybrid_pending_rc_conn = 0;
#ifdef _MV2_UD_DROP_PACKET_RATE_
uint32_t ud_drop_packet_rate = 0;
#endif
#endif
#if defined(_MCST_SUPPORT_)
uint8_t rdma_enable_mcast = USE_MCAST_DEFAULT_FLAG;
#if defined(RDMA_CM)
uint8_t rdma_use_rdma_cm_mcast = USE_MCAST_DEFAULT_FLAG;
#endif /*defined(RDMA_CM)*/
uint8_t mcast_enable_rel = 1;
uint8_t mcast_use_mcast_nack = 1;
uint16_t mcast_window_size = 256;
uint16_t mcast_drop_packet_rate = 0;
uint32_t mcast_num_nodes_threshold = MCAST_NUM_THRESHOLD;
uint32_t mcast_max_ud_recv_wqe = 2096;
long mcast_retry_timeout = 500000;
long mcast_max_retry_timeout = 20000000;
long mcast_comm_init_timeout = 10000;
int mcast_comm_init_retries = 128;
int mcast_nspin_threshold = 1200;
int mcast_skip_loopback = 1;
uint32_t mcast_bcast_min_msg = 1;
uint32_t mcast_bcast_max_msg = 512 * 1024;
#endif
int mv2_use_eager_fast_send= 1;
int mv2_rdma_fast_path_preallocate_buffers = 0;

/* Max number of entries on the RecvQ of QPs per connection.
 * computed to be:
 * prepost_depth + rdma_prepost_rendezvous_extra + rdma_prepost_noop_extra
 * Must be within NIC MaxQpEntries limit.
 */
int rdma_rq_size;
int using_mpirun_rsh = 0;

uint32_t mv2_srq_alloc_size = 32768;
uint32_t mv2_srq_fill_size = 256;
uint32_t mv2_srq_limit = 30;
uint32_t mv2_max_r3_oust_send = 32;


/* The number of "extra" vbufs that will be posted as receives
 * on a connection in anticipation of an R3 rendezvous message.
 * The TOTAL number of VBUFs posted on a receive queue at any
 * time is rdma_prepost_depth + rdma_prepost_rendezvous_extra
 * regardless of the number of outstanding R3 sends active on
 * a connection.
 */
int rdma_prepost_rendezvous_extra = 10;
int rdma_dynamic_credit_threshold = 10;
int rdma_credit_notify_threshold = 10;
int rdma_prepost_threshold = 5;

unsigned long rdma_dreg_cache_limit = 0;

/* Blocking mode progress */
int rdma_use_blocking = 0;
unsigned long rdma_blocking_spin_count_threshold = DEFAULT_SPIN_COUNT;

/* The total size of each vbuf. Used to be the eager threshold, but
 * it can be smaller, so that each eager message will span over few
 * vbufs
 */
int rdma_vbuf_total_size;

/* Small message scheduling policy
 * Was earlier set to USE_FIRST, optimized for minimal QP cache misses
 * Now setting it to FIXED_MAPPING as we get better performance.
 * 10/06/2010
 */
int rdma_rail_sharing_policy = FIXED_MAPPING;

/* This value should increase with the increase in number
 * of rails */
int striping_threshold = STRIPING_THRESHOLD;

/* Used IBoEth mode */
int use_iboeth = 0;

int rdma_enable_hugepage = 1;

/* Linear update factor for HSAM */
int alpha = 0.9;
int stripe_factor = 1;
int apm_tester = 0;

#ifdef _ENABLE_CUDA_
int rdma_cuda_block_size = DEFAULT_CUDA_BLOCK_SIZE;
int rdma_num_cuda_rndv_blocks = 8;
int rdma_cuda_event_count = 64;
int rdma_enable_cuda = 0;
int rdma_cuda_dynamic_init = 1;
int rdma_cuda_nonblocking_streams = 1;
int cuda_initialized = 0;
int rdma_cuda_vec_thread_blksz = 0;
int rdma_cuda_vec_thread_ysz = 0;
int rdma_cuda_subarr_thread_blksz = 0;
int rdma_cuda_subarr_thread_xdim = 0;
int rdma_cuda_subarr_thread_ydim = 0;
int rdma_cuda_subarr_thread_zdim = 0;
int rdma_eager_cudahost_reg = 0;
int rdma_cuda_vector_dt_opt = 1;
int rdma_cuda_kernel_dt_opt = 1;
#if defined(HAVE_CUDA_IPC)
int rdma_cuda_ipc = 1;
int rdma_enable_ipc_share_gpu = 1;
int rdma_cuda_smp_ipc = 0;
int rdma_cuda_enable_ipc_cache = 0;
int rdma_cuda_ipc_threshold = 512 * 1024;
int cudaipc_cache_max_entries = 1;
#endif /*#if defined(HAVE_CUDA_IPC) */
int rdma_cuda_use_naive = 1;
int rdma_cuda_register_naive_buf = 524288;
int rdma_cuda_gather_naive_limit = 32768;
int rdma_cuda_scatter_naive_limit = 2048;
int rdma_cuda_gatherv_naive_limit = 0;
int rdma_cuda_scatterv_naive_limit = 16384;
int rdma_cuda_allgather_naive_limit = 1048576;
int rdma_cuda_allgatherv_naive_limit = 524288;
int rdma_cuda_alltoall_naive_limit = 262144;
int rdma_cuda_alltoallv_naive_limit = 262144;
int rdma_cuda_bcast_naive_limit = 2097152;
int rdma_cuda_alltoall_dynamic = 1;
int rdma_cuda_allgather_rd_limit = 1024;
int rdma_cuda_allgather_fgp = 0;
int rdma_cuda_init_context = 1;
int rdma_check_cuda_attribute = 0;
#endif /*#ifdef _ENABLE_CUDA_ */
int mv2_enable_progress_affinity = 0;

typedef enum _mv2_user_defined_mapping_policies {

    MV2_UDEF_POLICY_BUNCH = 1,
    MV2_UDEF_POLICY_SCATTER,
    MV2_UDEF_POLICY_NONE,

} user_defined_mapping_policies;

/* Optimal CPU Binding parameters */
int use_hwloc_cpu_binding = 1;

/* Use of LIMIC of RMA Communication */
int limic_put_threshold;
int limic_get_threshold;

mv2_polling_level rdma_polling_level = MV2_POLLING_LEVEL_1;

int g_atomics_support = 0;
int g_atomics_support_be = 0;

static int check_hsam_parameters(void);

static inline int log_2(int np)
{
    int lgN, t;
    for (lgN = 0, t = 1; t < np; lgN++, t += t);
    return lgN;
}

#undef FUNCNAME
#define FUNCNAME MV2_get_arch_hca_type
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
mv2_arch_hca_type MV2_get_arch_hca_type()
{
    struct mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    if (!proc->arch_hca_type) {
        proc->arch_type = mv2_get_arch_type();
        proc->hca_type = MV2_HCA_ANY;
        proc->arch_hca_type = proc->arch_type;
        proc->arch_hca_type <<= 16;
        proc->arch_hca_type |= proc->hca_type;
        proc->arch_hca_type <<= 16;
        proc->arch_hca_type |= g_mv2_num_cpus;
    }

    return proc->arch_hca_type;
}

/*
 * Function: rdma_cm_get_hca_type
 *
 * Description:
 *      Finds out the type of the HCA on the system.
 *
 * Input:
 *      use_iwarp_mode  - Command line input specifying whether we need to use
 *                        iWARP mode.
 * Output:
 *      hca_type        - The type of HCA we are going to use.
 *
 * Return:
 *      Success:    MPI_SUCCESS.
 *      Failure:    ERROR (-1).
 */
#if defined(RDMA_CM)
#undef FUNCNAME
#define FUNCNAME rdma_cm_get_hca_type
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_get_hca_type(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    int i = 0;
    int ret = 0;
    int mpi_errno = MPI_SUCCESS;
    int numdevices = 0;
    int network_type = MV2_HCA_UNKWN, hca_type = MV2_HCA_UNKWN;
    struct ibv_device_attr dev_attr;
    char *dev_name = NULL, *value;
    struct ibv_context **ctx = rdma_get_devices(&numdevices);

    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_GET_HCA_TYPE);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_GET_HCA_TYPE);

    for (i = 0; i < numdevices; ++i) {
        hca_type = mv2_get_hca_type(ctx[i]->device);
        PRINT_DEBUG(DEBUG_INIT_verbose>1, "HCA %s type = %s\n", ctx[i]->device->name,
                    mv2_get_hca_name(hca_type));
        if (network_type <= hca_type) {
            network_type=hca_type;
        }
    }

    for (i = 0; i < numdevices; ++i) {
        proc->hca_type = MV2_HCA_UNKWN;
        dev_name = (char *) ibv_get_device_name(ctx[i]->device);

        if (!dev_name) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER,
                                "**ibv_get_device_name");
        }

        if ((value = getenv("MV2_IBA_HCA")) != NULL) {
            if(strstr(value, dev_name) == NULL) {
                continue;
            }
        }

        if (rdma_skip_network_card(network_type, ctx[i]->device)) {
            /* Skip HCA's that don't match with network type */
            PRINT_DEBUG(DEBUG_INIT_verbose, "1. Skipping HCA %s since type does not match."
                        "Selected: %s; Current: %s\n",
                        ctx[i]->device->name, mv2_get_hca_name(network_type),
                        mv2_get_hca_name(mv2_get_hca_type(ctx[i]->device)));
            continue;
        }

        if ((ret = ibv_query_device(ctx[i], &dev_attr))) {
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**ibv_query_device",
                                 "**ibv_query_device %s", dev_name);
        }

        if (ERROR == rdma_find_active_port(ctx[i], ctx[i]->device)) {
            /* Trac #376 The device has no active ports, continue to next device */
            continue;
        }

        proc->hca_type = mv2_get_hca_type(ctx[i]->device);
        proc->arch_hca_type = mv2_get_arch_hca_type(ctx[i]->device);
        if (MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type)) {
            /* Trac #376 recognize chelsio nic even if it's not the first */
            proc->use_rdma_cm = 1;
            proc->use_iwarp_mode = 1;
            MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_RDMA_CM;
            strncpy(rdma_iba_hcas[0], CHELSIO_RNIC, 32);
        } else if (MV2_HCA_INTEL_NE020 == proc->hca_type) {
            proc->use_rdma_cm = 1;
            proc->use_iwarp_mode = 1;
            MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_RDMA_CM;
            strncpy(rdma_iba_hcas[0], INTEL_NE020_RNIC, 32);
        }

        if (MV2_HCA_UNKWN != proc->hca_type) {
            /* We've found the HCA */
            break;
        }
    }

  fn_fail:
    rdma_free_devices(ctx);
    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_GET_HCA_TYPE);
    return mpi_errno;
}

#endif /* defined(RDMA_CM) */

/* The rdma_get_process_to_rail_mapping function is called from
 * ch3_smp_progress.c to set the mapping given by the user at run time
 */
#undef FUNCNAME
#define FUNCNAME rdma_get_process_to_rail_mapping
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_get_process_to_rail_mapping(int mrail_user_defined_p2r_type)
{
    char *p2r_only_numbers =
        (char *) MPIU_Malloc((mrail_p2r_length + 1) * sizeof(char));
    int i, j = 0;
    int num_total_devices = 0;
    int num_devices = 0;
    char *tp = mrail_p2r_string;
    char *cp = NULL;
    char tp_str[mrail_p2r_length + 1];
    struct ibv_device **dev_list = NULL;
    struct ibv_device **usable_dev_list = MPIU_Malloc(sizeof(struct ibv_device *)*MAX_NUM_HCAS);
    int bunch_hca_count;

    dev_list = ibv_get_device_list(&num_total_devices);

    rdma_find_network_type(dev_list, num_total_devices, usable_dev_list, &num_devices);

    if (rdma_num_req_hcas && rdma_num_req_hcas <= num_devices) {
        num_devices = rdma_num_req_hcas;
    }
    switch (mrail_user_defined_p2r_type) {
        case MV2_UDEF_POLICY_NONE:

            if (((mrail_p2r_length + 1) / 2) != rdma_num_local_procs) {
                if (rdma_local_id == 0) {
                    fprintf(stderr,
                            "Mapping should contain %d values. "
                            "Falling back to default scheme.\n",
                            rdma_num_local_procs);
                }
                mrail_use_default_mapping = 1;
                rdma_rail_sharing_policy = FIXED_MAPPING;
            } else {

                while (*tp != '\0') {
                    i = 0;
                    cp = tp;
                    while (*cp != '\0' && *cp != ':' && i < mrail_p2r_length) {
                        ++cp;
                        ++i;
                    }

                    strncpy(tp_str, tp, i);
                    if (atoi(tp) < 0 || atoi(tp) >= num_devices) {
                        if (rdma_local_id == 0) {
                            fprintf(stderr,
                                    "\nWarning! : HCA #%d does not "
                                    "exist on this machine. Falling back to "
                                    "default scheme\n", atoi(tp));
                        }
                        mrail_use_default_mapping = 1;
                        rdma_rail_sharing_policy = FIXED_MAPPING;
                        goto fn_exit;
                    }
                    tp_str[i] = '\0';

                    if (j == rdma_local_id) {
                        mrail_user_defined_p2r_mapping = atoi(tp_str);
                        break;
                    }

                    if (*cp == '\0') {
                        break;
                    }

                    tp = cp;
                    ++tp;
                    ++j;
                }
            }
            break;

        case MV2_UDEF_POLICY_SCATTER:
            mrail_user_defined_p2r_mapping = rdma_local_id % num_devices;
            break;
        case MV2_UDEF_POLICY_BUNCH:
            bunch_hca_count = rdma_num_local_procs / num_devices;
            if ((bunch_hca_count * num_devices) < rdma_num_local_procs) {
                bunch_hca_count++;
            }
            mrail_user_defined_p2r_mapping = rdma_local_id / bunch_hca_count;
            break;
        default:
            if (rdma_local_id == 0) {
                fprintf(stderr,
                        "\nError determining type of user defined"
                        " binding. Falling back to default scheme...!");
            }
            mrail_use_default_mapping = 1;
            rdma_rail_sharing_policy = FIXED_MAPPING;
            break;
    }
  fn_exit:
    /* Housekeeping operations */
    MPIU_Free(usable_dev_list);
    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
    MPIU_Free(p2r_only_numbers);
    MPIU_Free(mrail_p2r_string);
    mrail_p2r_string = NULL;
    p2r_only_numbers = NULL;

    return 0;
}

#undef FUNCNAME
#define FUNCNAME rdma_get_rail_sharing_policy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_get_rail_sharing_policy(char *value)
{
    int policy = FIXED_MAPPING;

    if (!strcmp(value, "USE_FIRST")) {
        policy = USE_FIRST;
    } else if (!strcmp(value, "ROUND_ROBIN")) {
        policy = ROUND_ROBIN;
    } else if (!strcmp(value, "FIXED_MAPPING")) {
        policy = FIXED_MAPPING;
    } else {
        MPL_usage_printf("Invalid small message scheduling\n");
    }
    return policy;
}

/* Set thresholds for Nnum_rail=unknown */
static void set_limic_thresholds(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    switch (proc->arch_type) {
        case MV2_ARCH_AMD_BARCELONA_16:
            limic_put_threshold = 1 * 1024;
            limic_get_threshold = 256;
            break;
        case MV2_ARCH_INTEL_CLOVERTOWN_8:
            limic_put_threshold = 1 * 1024;
            limic_get_threshold = 1 * 1024;
            break;
        case MV2_ARCH_INTEL_NEHALEM_8:
            limic_put_threshold = 8 * 1024;
            limic_get_threshold = 4 * 1024;
            break;
        default:
            limic_put_threshold = 8 * 1024;
            limic_get_threshold = 8 * 1024;
            break;
    }
}

#undef FUNCNAME
#define FUNCNAME rdma_set_smp_parameters
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_set_smp_parameters(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    char *value = NULL;

#if defined(_SMP_CMA_) || defined(_SMP_LIMIC_)
#if defined(_SMP_CMA_)
    if ((value = getenv("MV2_SMP_USE_CMA")) != NULL) {
        g_smp_use_cma = atoi(value);
    }
#endif

#if defined(_SMP_LIMIC_)
    if ((value = getenv("MV2_SMP_USE_LIMIC2")) != NULL) {
        g_smp_use_limic2 = atoi(value);
    }
    if ((value = getenv("MV2_USE_LIMIC2_COLL")) != NULL) {
        g_use_limic2_coll = atoi(value);
    }

        if(!mv2_enable_affinity || !g_smp_use_limic2) {
            g_use_limic2_coll = 0;
        } else
        {
            g_use_limic2_coll = 0;
        }

#if defined(_SMP_CMA_)
    if(g_smp_use_cma && g_smp_use_limic2) {
        g_smp_use_cma = 0;
    }
#endif /* _SMP_CMA_ */
#endif /* _SMP_LIMIC_ */
#endif /* _SMP_CMA_ || _SMP_LIMIC */


    g_smp_max_switch = 
        (value = getenv("MV2_SMP_USE_MAX_SWITCH")) !=
        NULL ? !!atoi(value) : 0;

    if (!proc->arch_type) {
        proc->arch_type = mv2_get_arch_type();
    }

    switch (proc->arch_type) {
        case MV2_ARCH_INTEL_XEON_E5630_8:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 256;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;
        case MV2_ARCH_INTEL_GENERIC:
        case MV2_ARCH_INTEL_CLOVERTOWN_8:
        case MV2_ARCH_INTEL_XEON_DUAL_4:
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 256;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_HARPERTOWN_8:
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 256;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_NEHALEM_8:
        case MV2_ARCH_INTEL_NEHALEM_16:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 256;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_AMD_BARCELONA_16:
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_AMD_EPYC_7551_64:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 16384;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 16384;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 16384;
            }
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_AMD_EPYC_7742_128:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 16384;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 16384;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 16384;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 16384;
            break;

        case MV2_ARCH_AMD_OPTERON_DUAL_4:
        case MV2_ARCH_AMD_GENERIC:
            g_smp_eagersize = 5120;
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_AMD_MAGNY_COURS_24:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 16384;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 5120;
            }
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 64;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_AMD_OPTERON_6136_32:
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 16384;
            }
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_AMD_OPTERON_6276_64:
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 16384;
            }
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 128;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_X5650_12:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 16384;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2670_16:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 16384;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2680_16:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 524288;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 14336;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 64;
            s_smp_batch_size = 8;
            s_smp_block_size = 65536;
            break;

        case MV2_ARCH_ARM_CAVIUM_V8_2S_28:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;
        
	case MV2_ARCH_ARM_CAVIUM_V8_2S_32:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 8192;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 16384;
            }
            s_smp_queue_length = 393216;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 32768;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 16384;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 16384;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_PHI_7250:
#if defined(_SMP_CMA_)
            /* Use CMA from 2 ppn onwards */
            if (MPIDI_Num_local_processes(MPIDI_Process.my_pg) <= 2) {
                g_smp_use_cma = 0;
            }
            if ((value = getenv("MV2_SMP_USE_CMA")) != NULL) {
                g_smp_use_cma = atoi(value);
            }
            if (g_smp_use_cma) {
                g_smp_eagersize = 65536;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 131072;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 64;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 131072;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 524288;
            s_smp_num_send_buffer = 48;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 32768;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 131072;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
#if defined(_SMP_CMA_)
            if (mv2_enable_eager_threshold_reduction && g_smp_use_cma &&
                    10 < MPIDI_Process.my_pg->ch.num_local_processes) {
                /* if there are large number of processes per node, then
                 * reduce the eager threshold and queue length */
                g_smp_eagersize     = 5120;   
                s_smp_queue_length = 32768;
            }
#endif
            break;

        case MV2_ARCH_IBM_POWER8:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 32768;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 524288;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_IBM_POWER9:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 65536;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 32768;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 32768;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 32768;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 524288;
            s_smp_num_send_buffer = 48;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 5120;
                s_smp_cma_max_size = 4194304;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                g_smp_eagersize = 65536;
                s_smp_limic2_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 32768;
#if defined(_SMP_CMA_)
            if (mv2_enable_eager_threshold_reduction && g_smp_use_cma && 
                    14 < MPIDI_Process.my_pg->ch.num_local_processes) {
                /* if there are large number of processes per node, then
                 * reduce the eager threshold and queue length */
                g_smp_eagersize     = 5120;   
                s_smp_queue_length = 32768;
            }
#endif
            break;

        case MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 16384;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 32768;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 131072;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 1048576;
            s_smp_num_send_buffer = 8;
            s_smp_batch_size = 8;
            s_smp_block_size = 65536;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 65536;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 5120;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_PLATINUM_8170_2S_52:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 5120;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 65536;
            }
            s_smp_queue_length = 262144;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_INTEL_PLATINUM_8280_2S_56:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                g_smp_eagersize = 8192;
                s_smp_cma_max_size = 4194304;
            } else
#endif
            {
                g_smp_eagersize = 8192;
            }
            s_smp_queue_length = 524288;
            s_smp_num_send_buffer = 32;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        case MV2_ARCH_AMD_BULLDOZER_4274HE_16:
#if defined(_SMP_CMA_)
            if (g_smp_use_cma) {
                s_smp_cma_max_size = 4194304;
                g_smp_eagersize = 32768;
                s_smp_queue_length = 131072;
            } else
#endif
#if defined(_SMP_LIMIC_)
            if (g_smp_use_limic2) {
                s_smp_limic2_max_size = 4194304;
                g_smp_eagersize = 8192;
                s_smp_queue_length = 65536;
            } else
#endif
            {
                g_smp_eagersize = 16384;
                s_smp_queue_length = 65536;
            }
            s_smp_num_send_buffer = 16;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;

        default:
            g_smp_eagersize = 16384;
            s_smp_queue_length = 65536;
            s_smp_num_send_buffer = 128;
            s_smp_batch_size = 8;
            s_smp_block_size = 8192;
            break;
    }

#if defined(_ENABLE_CUDA_)
    s_smp_h2h_block_size = s_smp_block_size;

    if ((value = getenv("MV2_CUDA_SMP_PIPELINE")) != NULL) {
        s_smp_cuda_pipeline = atoi(value);
    }

    if (rdma_enable_cuda && s_smp_cuda_pipeline) {
        s_smp_h2h_block_size = s_smp_block_size;
        s_smp_block_size = 128*1024;
    }

#if defined(HAVE_CUDA_IPC)
    smp_cuda_region_size = g_smp_eagersize * 4;

    if ((value = getenv("SMPI_CUDA_REGION_SIZE")) != NULL) {
        smp_cuda_region_size = user_val_to_bytes(value,"SMPI_CUDA_REGION_SIZE");
    }

    if (smp_cuda_region_size <= g_smp_eagersize * 4) {
        smp_cuda_region_size = g_smp_eagersize * 4;
    }
#endif
#endif

    /* Set Limic Thresholds */
    set_limic_thresholds(proc);

    /* Reading SMP user parameters */
    if ((value = getenv("MV2_SMP_DELAY_SHMEM_POOL_INIT")) != NULL) {
        g_smp_delay_shmem_pool_init = atoi(value);
    }

    if ((value = getenv("MV2_SMP_POLLING_TH")) != NULL) {
        g_smp_polling_th = user_val_to_bytes(value, "MV2_SMP_POLLING_TH");
    }

    if ((value = getenv("MV2_SMP_PRIORITY_POLLING")) != NULL) {
        g_smp_priority_polling = atoi(value);
    }

    if ((value = getenv("MV2_SMP_CMA_MAX_SIZE")) != NULL) {
        s_smp_cma_max_size = user_val_to_bytes(value, "MV2_SMP_CMA_MAX_SIZE");
    }

    if ((value = getenv("MV2_SMP_LIMIC2_MAX_SIZE")) != NULL) {
        s_smp_limic2_max_size = user_val_to_bytes(value, "MV2_SMP_LIMIC2_MAX_SIZE");
    }

    if ((value = getenv("MV2_SMP_EAGERSIZE")) != NULL) {
        g_smp_eagersize = user_val_to_bytes(value, "MV2_SMP_EAGERSIZE");
    }

    if ((value = getenv("MV2_SMP_QUEUE_LENGTH")) != NULL) {
        s_smp_queue_length = user_val_to_bytes(value, "MV2_SMP_QUEUE_LENGTH");
    }

    if ((value = getenv("MV2_SMP_NUM_SEND_BUFFER")) != NULL) {
        s_smp_num_send_buffer = atoi(value);
    }

    if ((value = getenv("MV2_SMP_BATCH_SIZE")) != NULL) {
        s_smp_batch_size = atoi(value);
    }

    if ((value = getenv("MV2_SMP_SEND_BUF_SIZE")) != NULL) {
        s_smp_block_size = atoi(value);
    }

    if ((value = getenv("MV2_LIMIC_PUT_THRESHOLD")) != NULL) {
        limic_put_threshold =
            user_val_to_bytes(value, "MV2_LIMIC_PUT_THRESHOLD");
    }
    if ((value = getenv("MV2_LIMIC_GET_THRESHOLD")) != NULL) {
        limic_get_threshold =
            user_val_to_bytes(value, "MV2_LIMIC_GET_THRESHOLD");
    }

    int use_cma, use_xpmem, use_limic;
    use_cma = use_xpmem = use_limic = 0;
#if defined(_SMP_CMA_)
    use_cma = g_smp_use_cma;
#endif
#if defined(_SMP_LIMIC_)
    use_limic = g_smp_use_limic2;
#endif
    if (use_limic || (!use_cma && !use_xpmem)) {
        smp_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
    }

    return 0;
}

#undef FUNCNAME
#define FUNCNAME rdma_get_control_parameters
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_get_control_parameters(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    int size = -1;
    int pg_size = ERROR;
    int my_rank = -1;
    char *value = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mrail_user_defined_p2r_type = 0;

    MPIDI_STATE_DECL(MPID_STATE_RDMA_GET_CONTROL_PARAMETERS);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_GET_CONTROL_PARAMETERS);

    proc->global_used_send_cq = 0;
    proc->global_used_recv_cq = 0;

    UPMI_GET_SIZE(&pg_size);
    UPMI_GET_RANK(&my_rank);

    if ((value = getenv("MV2_NUM_NODES_IN_JOB")) != NULL) {
        rdma_num_nodes_in_job = atoi(value);
    } else {
        MPID_Get_max_node_id(NULL, &rdma_num_nodes_in_job);
        /* For some reason, MPID_Get_max_node_id does a '--' before
         * returning the num_nodes, hence incrementing it by 1 */
        rdma_num_nodes_in_job++;
    }

#if defined(_MCST_SUPPORT_)
    if ((value = getenv("MV2_USE_MCAST")) != NULL) {
        rdma_enable_mcast = !!atoi(value);
#if defined(RDMA_CM)
        if (rdma_enable_mcast == 0) {
            rdma_use_rdma_cm_mcast = 0;
        }
#endif /*defined(RDMA_CM)*/
    } else if (rdma_num_nodes_in_job < mcast_num_nodes_threshold) {
        /* Disable mcast by default when number of nodes less than 8 */
        rdma_enable_mcast = 0;
#if defined(RDMA_CM)
        rdma_use_rdma_cm_mcast = 0;
#endif /*defined(RDMA_CM)*/
        PRINT_DEBUG(DEBUG_MCST_verbose,"Disabling mcast by default as the number"
                " of nodes are less than %d. Set MV2_USE_MCAST=1 or "
                " MV2_MCAST_NUM_NODES_THRESHOLD=%d to avoid this"
                " behavior\n", mcast_num_nodes_threshold, mcast_num_nodes_threshold-1);
    }
#if defined(RDMA_CM)
    if ((value = getenv("MV2_USE_RDMA_CM_MCAST")) != NULL) {
        /* Set both values so that user only has to set MV2_USE_RDMA_CM_MCAST */
        rdma_enable_mcast = rdma_use_rdma_cm_mcast = !!atoi(value);
    } else if (rdma_num_nodes_in_job < mcast_num_nodes_threshold) {
        /* Disable mcast by default when number of nodes less than 8 */
        rdma_enable_mcast = 0;
        rdma_use_rdma_cm_mcast = 0;
        PRINT_DEBUG(DEBUG_MCST_verbose,"Disabling mcast by default as the number"
                " of nodes are less than %d. Set MV2_USE_RDMA_CM_MCAST=1 or "
                " MV2_MCAST_NUM_NODES_THRESHOLD=%d to avoid this"
                " behavior\n", mcast_num_nodes_threshold, mcast_num_nodes_threshold-1);
    }
#endif /*defined(RDMA_CM)*/
#endif /*defined(_MCST_SUPPORT_)*/

#ifdef ENABLE_QOS_SUPPORT
    if ((value = getenv("MV2_USE_QOS")) != NULL) {
        rdma_use_qos = !!atoi(value);
    }

    if ((value = getenv("MV2_3DTORUS_SUPPORT")) != NULL) {
        rdma_3dtorus_support = !!atoi(value);
    }

    if ((value = getenv("MV2_PATH_SL_QUERY")) != NULL) {
        rdma_path_sl_query = !!atoi(value);
    }

    if ((value = getenv("MV2_NUM_SLS")) != NULL) {
        rdma_qos_num_sls = atoi(value);
        if (rdma_qos_num_sls <= 0 && rdma_qos_num_sls > RDMA_QOS_MAX_NUM_SLS) {
            rdma_qos_num_sls = RDMA_QOS_DEFAULT_NUM_SLS;
        }
        /* User asked us to use multiple SL's without enabling QoS globally. */
        if (rdma_use_qos == 0) {
            rdma_use_qos = 1;
        }
    }
#endif /* ENABLE_QOS_SUPPORT */

    if ((value = getenv("MV2_NUM_SA_QUERY_RETRIES")) != NULL) {
        rdma_num_sa_query_retries = atoi(value);
        if (rdma_num_sa_query_retries < RDMA_DEFAULT_NUM_SA_QUERY_RETRIES) {
            rdma_num_sa_query_retries = RDMA_DEFAULT_NUM_SA_QUERY_RETRIES;
        }
    }

    if ((value = getenv("MV2_MAX_RDMA_CONNECT_ATTEMPTS")) != NULL) {
        max_rdma_connect_attempts = atoi(value);
        if (max_rdma_connect_attempts <= 0) {
            max_rdma_connect_attempts = DEFAULT_RDMA_CONNECT_ATTEMPTS;
        }
    }

    if ((value = getenv("MV2_RDMA_CM_CONNECT_RETRY_INTERVAL")) != NULL) {
        rdma_cm_connect_retry_interval = atoi(value);
        if (rdma_cm_connect_retry_interval <= 0) {
            rdma_cm_connect_retry_interval = RDMA_DEFAULT_CONNECT_INTERVAL;
        }
    }

    /* Parameters to decide the p2r mapping
     * The check for this parameter should always be done before we check for
     * MV2_SM_SCHEDULING below as we find out if the user has specified a
     * mapping for the user defined scheme to take effect */
    if ((value = getenv("MV2_PROCESS_TO_RAIL_MAPPING")) != NULL) {
        mrail_p2r_length = strlen(value);

        mrail_p2r_string =
            (char *) MPIU_Malloc(mrail_p2r_length * sizeof(char));

        strcpy(mrail_p2r_string, value);
        mrail_p2r_string[mrail_p2r_length] = '\0';
        if (!strcmp(value, "BUNCH")) {
            mrail_user_defined_p2r_type = MV2_UDEF_POLICY_BUNCH;
        } else if (!strcmp(value, "SCATTER")) {
            mrail_user_defined_p2r_type = MV2_UDEF_POLICY_SCATTER;
        } else {
            mrail_user_defined_p2r_type = MV2_UDEF_POLICY_NONE;
        }
        rdma_get_process_to_rail_mapping(mrail_user_defined_p2r_type);
    } else {
        mrail_use_default_mapping = 1;
    }

    /* Start HSAM Parameters */
    if ((value = getenv("MV2_USE_HSAM")) != NULL) {
        proc->has_hsam = atoi(value);
        if (proc->has_hsam) {
            check_hsam_parameters();
        }
    } else {
        /* By default disable the HSAM, due to problem with
         * multi-pathing with current version of opensm and
         * up/down */
        proc->has_hsam = 0;
    }
    /* End : HSAM Parameters */

    proc->has_apm = (value =
                     getenv("MV2_USE_APM")) != NULL ? (int) atoi(value) : 0;
    apm_tester = (value =
                  getenv("MV2_USE_APM_TEST")) != NULL ? (int) atoi(value) : 0;

    /* Scheduling Parameters */
    if ((value = getenv("MV2_SM_SCHEDULING")) != NULL) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
        rdma_rail_sharing_policy = rdma_get_rail_sharing_policy(value);
    }

    if ((value = getenv("MV2_SMALL_MSG_RAIL_SHARING_POLICY")) != NULL) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
        rdma_small_msg_rail_sharing_policy =
            rdma_get_rail_sharing_policy(value);
    }

    if ((value = getenv("MV2_MED_MSG_RAIL_SHARING_POLICY")) != NULL) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
        rdma_med_msg_rail_sharing_policy = rdma_get_rail_sharing_policy(value);
    }

    if ((value = getenv("MV2_RAIL_SHARING_POLICY")) != NULL) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
        rdma_rail_sharing_policy = rdma_med_msg_rail_sharing_policy =
            rdma_small_msg_rail_sharing_policy =
            rdma_get_rail_sharing_policy(value);
    }
    /* If there is only one process per node, or just one node allow it use all HCAs */
    if ((rdma_num_nodes_in_job == pg_size) || (rdma_num_nodes_in_job == 1)) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
    }

#if defined(RDMA_CM)
    if ((value = getenv("MV2_USE_IWARP_MODE")) != NULL) {
        proc->use_rdma_cm = !!atoi(value);
        proc->use_iwarp_mode = !!atoi(value);
    }

    if (!proc->use_rdma_cm) {
        if ((value = getenv("MV2_USE_RDMA_CM")) != NULL) {
#if !defined (ROMIO_IME)
            proc->use_rdma_cm = !!atoi(value);
#else
            if (value && atoi(value) && (my_rank == 0)) {
                MPL_error_printf("Error: IME FS does not work with RDMA CM. "
                                 "Proceeding without RDMA support.\n");
            }
#endif
        } else {
            proc->use_rdma_cm = 0;
            proc->use_iwarp_mode = 0;
        }
    }

#ifdef _MULTI_SUBNET_SUPPORT_
    if ((value = getenv("MV2_RDMA_CM_MULTI_SUBNET_SUPPORT")) != NULL) {
        mv2_rdma_cm_multi_subnet_support = !!atoi(value);
        if (proc->use_rdma_cm == 0) {
            proc->use_rdma_cm = mv2_rdma_cm_multi_subnet_support;
        }
    }
#endif /* _MULTI_SUBNET_SUPPORT_ */

    if ((value = getenv("MV2_SUPPORT_DPM")) && !!atoi(value)) {
        proc->use_rdma_cm = 0;
        proc->use_iwarp_mode = 0;
        mv2_use_eager_fast_send = 0;
        mv2_on_demand_ud_info_exchange = 0;    /* Trac #780 */
    }

    if (proc->use_rdma_cm) {
        int threshold = ERROR;

        if (proc->use_iwarp_mode) {
            /* Trac #423 */
            threshold = MPIDI_CH3I_CM_DEFAULT_IWARP_ON_DEMAND_THRESHOLD;
        } else {
            threshold = MPIDI_CH3I_RDMA_CM_DEFAULT_ON_DEMAND_THRESHOLD;
        }

        if ((value = getenv("MV2_ON_DEMAND_THRESHOLD")) != NULL) {
            threshold = atoi(value);
        }
        if (pg_size > threshold) {
            proc->use_rdma_cm_on_demand = 1;
            mv2_on_demand_ud_info_exchange = 1;
            proc->has_ring_startup = 0;
        }
#ifdef _ENABLE_XRC_
        /* XRC will not work with RDMA_CM */
        USE_XRC = 0;
        value = getenv("MV2_USE_XRC");
        if (value && (my_rank == 0)) {
            if (atoi(value)) {
                MPL_error_printf("Error: XRC does not work with RDMA CM. "
                                  "Proceeding without XRC support.\n");
            }
        }
#endif
    }
#endif

    if ((mpi_errno = rdma_open_hca(proc)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Heterogeniety detection for HCAs */
    int i = 0;
    int heterogeneous = 0;
    uint64_t node_guid[MAX_NUM_HCAS];
    mv2_hca_type    hca_type[MAX_NUM_HCAS];

    for (i = 0; i < rdma_num_hcas; ++i) {
        hca_type[i] = mv2_new_get_hca_type(proc->nic_context[i], proc->ib_dev[i], &node_guid[i]);
        if (i && (hca_type[i] != hca_type[i-1])) {
            heterogeneous = 1;
        }
    }
    proc->node_guid = node_guid[0];

    if (heterogeneous) {
        proc->hca_type = MV2_HCA_UNKWN;
    } else {
        proc->hca_type = hca_type[0];
    }

    proc->arch_hca_type = mv2_new_get_arch_hca_type(proc->hca_type);
    proc->arch_type = MV2_GET_ARCH(proc->arch_hca_type);

    if (rdma_num_nodes_in_job == 0) {
        UPMI_GET_SIZE(&size);
    } else {
        size = rdma_num_nodes_in_job;
    }

    if (size <= 8) {
        proc->cluster_size = VERY_SMALL_CLUSTER;
    } else if (size <= 32) {
        proc->cluster_size = SMALL_CLUSTER;
    } else if (size < 128) {
        proc->cluster_size = MEDIUM_CLUSTER;
    } else {
        proc->cluster_size = LARGE_CLUSTER;
    }

#ifdef _ENABLE_UD_
    if (pg_size < 1024) {
        rdma_ud_progress_timeout = 48000;
        rdma_ud_retry_timeout = 500000;
        rdma_ud_max_retry_count = 1024;
        rdma_ud_num_msg_limit = 512;
        rdma_hybrid_max_rc_conn = 32;
    } else if (pg_size < 4096) {
        rdma_ud_progress_timeout = 96000;
        rdma_ud_retry_timeout = 1000000;
        rdma_ud_max_retry_count = 512;
        rdma_ud_num_msg_limit = 1024;
        rdma_hybrid_max_rc_conn = 64;
    } else if (pg_size < 8192) {
        rdma_ud_progress_timeout = 96000;
        rdma_ud_retry_timeout = 1000000;
        rdma_ud_max_retry_count = 512;
        rdma_ud_num_msg_limit = 2048;
        rdma_hybrid_max_rc_conn = 128;
    } else {
        rdma_ud_progress_timeout = 190000;
        rdma_ud_retry_timeout = 2000000;
        rdma_ud_max_retry_count = 256;
        rdma_ud_num_msg_limit = 4096;
        rdma_hybrid_max_rc_conn = 128;
    }
#endif

    proc->has_srq = (value = getenv("MV2_USE_SRQ")) != NULL ? !!atoi(value) : 1;

    if ((value = getenv("MV2_IWARP_MULTIPLE_CQ_THRESHOLD")) != NULL) {
        rdma_iwarp_multiple_cq_threshold = atoi(value);
        if (rdma_iwarp_multiple_cq_threshold < 0) {
            rdma_iwarp_multiple_cq_threshold =
                RDMA_IWARP_DEFAULT_MULTIPLE_CQ_THRESHOLD;
        }
    }

    if (size > rdma_iwarp_multiple_cq_threshold) {
        rdma_iwarp_use_multiple_cq = 1;
    }
#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        proc->has_srq = 1;
        MPIU_Assert(MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND);
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_ON_DEMAND;
        rdma_use_coalesce = 0;
        rdma_use_blocking = 0;
    }
#endif /* _ENABLE_XRC_ */

    if (proc->has_srq && proc->hca_type != MV2_HCA_QLGIC_PATH_HT &&
        proc->hca_type != MV2_HCA_QLGIC_QIB &&
        proc->hca_type != MV2_HCA_MLX_PCI_X &&
        proc->hca_type != MV2_HCA_IBM_EHCA &&
        proc->hca_type != MV2_HCA_MARVEL_QEDR
#if defined(RDMA_CM)
        && !proc->use_iwarp_mode
#endif /* defined(RDMA_CM) */
    ) {
        proc->post_send = post_srq_send;
    } else {
        proc->has_srq = 0;
        proc->post_send = post_send;
    }
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        proc->post_send = post_hybrid_send;
    }
#endif

#if !defined(DISABLE_PTMALLOC)
    proc->has_lazy_mem_unregister = (value =
                                     getenv("MV2_USE_LAZY_MEM_UNREGISTER")) !=
        NULL ? !!atoi(value) : 1;
#endif /* !defined(DISABLE_PTMALLOC) */

    proc->enable_rma_fast_path = (value = 
                              getenv("MV2_USE_RMA_FAST_PATH")) != 
        NULL ? !! atoi(value) : 1;                    

    proc->force_ib_atomic = (value = 
                              getenv("MV2_FORCE_IB_ATOMIC")) != 
        NULL ? !! atoi(value) : 0;                    
#if defined(CKPT)
    {
        proc->has_one_sided = 0;
    }
#else /* defined(CKPT) */
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        proc->has_one_sided = 0;
    } else
#endif
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        proc->has_one_sided = 0;
    } else
#endif
    {
        proc->has_one_sided = (value =
                               getenv("MV2_USE_RDMA_ONE_SIDED")) !=
            NULL ? !!atoi(value) : 1;
    }
#endif /* defined(CKPT) */

    if ((value = getenv("MV2_RNDV_EXT_SENDQ_SIZE")) != NULL) {
        rdma_rndv_ext_sendq_size = atoi(value);
        if (rdma_rndv_ext_sendq_size <= 1) {
            MPL_usage_printf("Setting MV2_RNDV_EXT_SENDQ_SIZE smaller than 1 "
                              "will severely limit the MPI bandwidth.\n");
        }
    }

    if ((value = getenv("MV2_RDMA_NUM_EXTRA_POLLS")) != NULL) {
        rdma_num_extra_polls = atoi(value);
        if (rdma_num_extra_polls <= 0) {
            rdma_num_extra_polls = 1;
        }
    }

    if ((value = getenv("MV2_COALESCE_THRESHOLD")) != NULL) {
        rdma_coalesce_threshold = atoi(value);
        if (rdma_coalesce_threshold < 1) {
            MPL_usage_printf("MV2_COALESCE_THRESHOLD must be >= 1\n");
            rdma_coalesce_threshold = 1;
        }
    }

    if (proc->hca_type == MV2_HCA_MLX_CX_SDR ||
        proc->hca_type == MV2_HCA_MLX_CX_DDR) {
        rdma_use_coalesce = 0;
    }
#ifdef _ENABLE_XRC_
    if (!USE_XRC) {
#endif
        if ((value = getenv("MV2_USE_COALESCE")) != NULL) {
            rdma_use_coalesce = !!atoi(value);
        }
#ifdef _ENABLE_XRC_
    }
#endif

    if ((value = getenv("MV2_SPIN_COUNT")) != NULL) {
        rdma_blocking_spin_count_threshold = atol(value);
    }

    if ((value = getenv("MV2_RNDV_PROTOCOL")) != NULL) {
        if (strncmp(value, "RPUT", 4) == 0) {
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        } else if (strncmp(value, "RGET", 4) == 0
#ifdef _ENABLE_XRC_
                   && !USE_XRC
#endif
    ) {
#if defined(CKPT)
            MPL_usage_printf("MV2_RNDV_PROTOCOL "
                              "must be either \"RPUT\" or \"R3\" when checkpoint is enabled\n");
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
#else /* defined(CKPT) */
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
#endif /* defined(CKPT) */
        } else if (strncmp(value, "R3", 2) == 0) {
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
        } else {
#ifdef _ENABLE_XRC_
            if (!USE_XRC)
#endif
                MPL_usage_printf("MV2_RNDV_PROTOCOL "
                                  "must be either \"RPUT\", \"RGET\", or \"R3\"\n");
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        }
    }

    if ((value = getenv("MV2_SMP_RNDV_PROTOCOL")) != NULL) {
        if (strncmp(value, "RPUT", 4) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        } else if (strncmp(value, "RGET", 4) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
        } else if (strncmp(value, "R3", 2) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
        } else {
            MPL_usage_printf("MV2_SMP_RNDV_PROTOCOL "
                    "must be one of: RPUT, RGET, R3\n");
            smp_rndv_protocol = rdma_rndv_protocol;
        }
    }

    if ((value = getenv("MV2_R3_THRESHOLD")) != NULL) {
        rdma_r3_threshold = user_val_to_bytes(value, "MV2_R3_THRESHOLD");
        if (rdma_r3_threshold < 0) {
            rdma_r3_threshold = 0;
        }
    }

    if ((value = getenv("MV2_INTRA_NODE_R3_THRESHOLD")) != NULL) {
        rdma_intra_node_r3_threshold = user_val_to_bytes(value, "MV2_INTRA_NODE_R3_THRESHOLD");
        if (rdma_intra_node_r3_threshold <= 0) {
            rdma_intra_node_r3_threshold = RDMA_DEFAULT_R3_THRESHOLD;
        }
    }

    if ((value = getenv("MV2_INTER_NODE_R3_THRESHOLD")) != NULL) {
        rdma_inter_node_r3_threshold = user_val_to_bytes(value, "MV2_INTER_NODE_R3_THRESHOLD");
        if (rdma_inter_node_r3_threshold <= 0) {
            rdma_inter_node_r3_threshold = RDMA_DEFAULT_R3_THRESHOLD;
        }
    }

    if ((value = getenv("MV2_R3_NOCACHE_THRESHOLD")) != NULL) {
        rdma_r3_threshold_nocache =
            user_val_to_bytes(value, "MV2_R3_NOCACHE_THRESHOLD");
        if (rdma_r3_threshold_nocache < 0) {
            rdma_r3_threshold_nocache = 0;
        }
    }

    if ((value = getenv("MV2_MAX_R3_PENDING_DATA")) != NULL) {
        rdma_max_r3_pending_data =
            user_val_to_bytes(value, "MV2_MAX_R3_PENDING_DATA");
        if (rdma_max_r3_pending_data < 0) {
            rdma_max_r3_pending_data = 0;
        }
    }
#if defined(RDMA_CM)
    if (proc->use_rdma_cm_on_demand) {
        proc->use_iwarp_mode = 1;
    }
#endif /* defined(RDMA_CM) */

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_GET_CONTROL_PARAMETERS);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/* Set params based on cluster size */
static void rdma_set_params_based_on_cluster_size(int cluster_size,
                                                  int lc_vbuf_total_size,
                                                  int lc_num_rdma_buff,
                                                  int mc_vbuf_total_size,
                                                  int mc_num_rdma_buff,
                                                  int sc_vbuf_total_size,
                                                  int sc_num_rdma_buff,
                                                  int vsc_vbuf_total_size,
                                                  int vsc_num_rdma_buff,
                                                  int def_vbuf_total_size,
                                                  int def_num_rdma_buff)
{
    switch (cluster_size) {

        case LARGE_CLUSTER:
            rdma_vbuf_total_size = lc_vbuf_total_size + EAGER_THRESHOLD_ADJUST;
            rdma_fp_buffer_size = 4 * 1024;
            num_rdma_buffer = lc_num_rdma_buff;
            rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
            break;

        case MEDIUM_CLUSTER:
            rdma_vbuf_total_size = mc_vbuf_total_size + EAGER_THRESHOLD_ADJUST;
            rdma_fp_buffer_size = 4 * 1024;
            num_rdma_buffer = mc_num_rdma_buff;
            rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
            break;
        case SMALL_CLUSTER:
            rdma_vbuf_total_size = sc_vbuf_total_size + EAGER_THRESHOLD_ADJUST;
            rdma_fp_buffer_size = 4 * 1024;
            num_rdma_buffer = sc_num_rdma_buff;
            rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
            break;
        case VERY_SMALL_CLUSTER:
            rdma_vbuf_total_size = vsc_vbuf_total_size + EAGER_THRESHOLD_ADJUST;
            rdma_fp_buffer_size = 4 * 1024;
            num_rdma_buffer = vsc_num_rdma_buff;
            rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
            break;
        default:
            rdma_vbuf_total_size = def_vbuf_total_size + EAGER_THRESHOLD_ADJUST;
            rdma_fp_buffer_size = 4 * 1024;
            num_rdma_buffer = def_num_rdma_buff;
            rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
            break;
    }
}

/* Set thresholds for Nnum_rail=4 */
static void rdma_set_default_parameters_numrail_4(struct 
                                                  mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    if (MV2_IS_ARCH_HCA_TYPE
        (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5630_8,
         MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 4 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_PLATINUM_8280_2S_56,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 64 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_28,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BULLDOZER_4274HE_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_CLOVERTOWN_8,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_OPTERON_6136_32,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_SDR)) {
        rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7551_64,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    /* same tuning value for rome with EDR HCA */
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_X5650_12,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 19 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_NEHALEM_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_MAGNY_COURS_24,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 128;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_HARPERTOWN_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 9 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_IBM_EHCA) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_MLX_PCI_X)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_CHELSIO_T3) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_CHELSIO_T4)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 64;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_INTEL_NE020)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 16 * 1024, 16,        /* Values for large cluster size */
                                              16 * 1024, 16,    /* Values for medium cluster size */
                                              16 * 1024, 16,    /* Values for small cluster size */
                                              16 * 1024, 16,    /* Values for very small cluster size */
                                              16 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 256 * 1024;
        rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    }
}

/* Set thresholds for Nnum_rail=3 */
static void rdma_set_default_parameters_numrail_3(struct
                                                  mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    if (MV2_IS_ARCH_HCA_TYPE
        (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5630_8,
         MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 4 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_PLATINUM_8280_2S_56,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 64 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_28,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BULLDOZER_4274HE_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_OPTERON_6136_32,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_CLOVERTOWN_8,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_SDR)) {
        rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7551_64,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    /* same tuning value for rome with EDR HCA */
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_X5650_12,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 19 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_NEHALEM_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_MAGNY_COURS_24,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 128;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_HARPERTOWN_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 9 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_IBM_EHCA) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_MLX_PCI_X)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_CHELSIO_T3) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_CHELSIO_T4)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 64;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_INTEL_NE020)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 16 * 1024, 16,        /* Values for large cluster size */
                                              16 * 1024, 16,    /* Values for medium cluster size */
                                              16 * 1024, 16,    /* Values for small cluster size */
                                              16 * 1024, 16,    /* Values for very small cluster size */
                                              16 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 256 * 1024;
        rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    }
}

/* Set thresholds for Nnum_rail=2 */
static void rdma_set_default_parameters_numrail_2(struct
                                                  mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    if (MV2_IS_ARCH_HCA_TYPE
        (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5630_8,
         MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 4 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_PLATINUM_8280_2S_56,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 64 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_28,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BULLDOZER_4274HE_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_OPTERON_6136_32,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_CLOVERTOWN_8,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_SDR)) {
        rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7551_64,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    /* same tuning value for rome with EDR HCA */
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_X5650_12,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 19 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_NEHALEM_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_MAGNY_COURS_24,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 128;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_HARPERTOWN_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 9 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_IBM_EHCA) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_MLX_PCI_X)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_CHELSIO_T3) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_CHELSIO_T4)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 64;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_INTEL_NE020)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 16 * 1024, 16,        /* Values for large cluster size */
                                              16 * 1024, 16,    /* Values for medium cluster size */
                                              16 * 1024, 16,    /* Values for small cluster size */
                                              16 * 1024, 16,    /* Values for very small cluster size */
                                              16 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 256 * 1024;
        rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    }
}

/* Set thresholds for Nnum_rail=1 */
static void rdma_set_default_parameters_numrail_1(struct
                                                  mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    if (MV2_IS_ARCH_HCA_TYPE
        (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5630_8,
         MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 4 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_PLATINUM_8280_2S_56,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 64 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_28,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BULLDOZER_4274HE_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_OPTERON_6136_32,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_CLOVERTOWN_8,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_SDR)) {
        rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7551_64,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    /* same tuning value for rome with EDR HCA */
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_X5650_12,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 19 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_NEHALEM_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_MAGNY_COURS_24,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 128;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_HARPERTOWN_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 9 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 9 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_IBM_EHCA) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_MLX_PCI_X)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_CHELSIO_T3) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_CHELSIO_T4)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 64;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_INTEL_NE020)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 16 * 1024, 16,        /* Values for large cluster size */
                                              16 * 1024, 16,    /* Values for medium cluster size */
                                              16 * 1024, 16,    /* Values for small cluster size */
                                              16 * 1024, 16,    /* Values for very small cluster size */
                                              16 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 256 * 1024;
        rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    }
}

/* Set thresholds for Nnum_rail=unknown */
static void rdma_set_default_parameters_numrail_unknwn(struct
                                                       mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    if (MV2_IS_ARCH_HCA_TYPE
        (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5630_8,
         MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 4 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 20 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 64 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_CONNIB)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER8,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_IBM_POWER9,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 32 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_28,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ARM_CAVIUM_V8_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_PHI_7250,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2680_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BULLDOZER_4274HE_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_OPTERON_6136_32,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_CLOVERTOWN_8,
              MV2_HCA_MLX_CX_DDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7551_64,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_HDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    /* same tuning value for rome with EDR HCA */
    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_EPYC_7742_128,
              MV2_HCA_MLX_CX_EDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_AMD_BARCELONA_16,
              MV2_HCA_MLX_CX_SDR)) {
        rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_X5650_12,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 19 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 17 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_XEON_E5_2670_16,
              MV2_HCA_MLX_CX_FDR)) {
        rdma_vbuf_total_size = 16 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 8 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_INTEL_HARPERTOWN_8,
              MV2_HCA_MLX_CX_QDR)) {
        rdma_vbuf_total_size = 9 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 5 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_FDR)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
        rdma_fp_buffer_size = 4 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_QDR)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
        rdma_fp_buffer_size = 4 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_MLX_CX_DDR)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 16,     /* Values for medium cluster size */
                                              12 * 1024, 32,    /* Values for small cluster size */
                                              12 * 1024, 32,    /* Values for very small cluster size */
                                              12 * 1024, 32);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 0;
        rdma_fp_buffer_size = 9 * 1024;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_CHELSIO_T3) ||
             MV2_IS_ARCH_HCA_TYPE(proc->arch_hca_type, MV2_ARCH_ANY,
                                  MV2_HCA_CHELSIO_T4)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 64;
    }

    else if (MV2_IS_ARCH_HCA_TYPE
             (proc->arch_hca_type, MV2_ARCH_ANY, MV2_HCA_INTEL_NE020)) {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 2 * 1024, 4,  /* Values for large cluster size */
                                              4 * 1024, 8,      /* Values for medium cluster size */
                                              9 * 1024, 16,     /* Values for small cluster size */
                                              32 * 1024, 16,    /* Values for very small cluster size */
                                              32 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 394 * 1024;
    }

    else {
        rdma_set_params_based_on_cluster_size(proc->cluster_size, 16 * 1024, 16,        /* Values for large cluster size */
                                              16 * 1024, 16,    /* Values for medium cluster size */
                                              16 * 1024, 16,    /* Values for small cluster size */
                                              16 * 1024, 16,    /* Values for very small cluster size */
                                              16 * 1024, 16);   /* Values for unknown cluster size */
        rdma_eagersize_1sc = 4 * 1024;
        rdma_put_fallback_threshold = 8 * 1024;
        rdma_get_fallback_threshold = 256 * 1024;
        rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    }
}

const char *mv2_ibv_mtu_enum_to_string(enum ibv_mtu mtu)
{
    switch (mtu) {
        case IBV_MTU_256:
            return "IBV_MTU_256";
        case IBV_MTU_512:
            return "IBV_MTU_512";
        case IBV_MTU_1024:
            return "IBV_MTU_1024";
        case IBV_MTU_2048:
            return "IBV_MTU_2048";
        case IBV_MTU_4096:
            return "IBV_MTU_4096";
        default:
            return "Unknown MTU";
    }
}

uint16_t mv2_ibv_mtu_enum_to_value(enum ibv_mtu mtu)
{
    switch (mtu) {
        case IBV_MTU_256:
            return 256;
        case IBV_MTU_512:
            return 512;
        case IBV_MTU_1024:
            return 1024;
        case IBV_MTU_2048:
            return 2048;
        case IBV_MTU_4096:
            return 4096;
        default:
            return -1;
    }
}

void rdma_set_default_parameters(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    mv2_multirail_info_type multirail_info = mv2_get_multirail_info();

    /* Setting the default values; these values are fine-tuned for specific platforms
     * in the following code */
    rdma_vbuf_total_size = 12 * 1024 + EAGER_THRESHOLD_ADJUST;
    rdma_fp_buffer_size = RDMA_FP_DEFAULT_BUF_SIZE;
    num_rdma_buffer = 16;
    rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;

    rdma_eagersize_1sc = 4 * 1024;
    rdma_put_fallback_threshold = 2 * 1024;
    rdma_get_fallback_threshold = 192 * 1024;

    switch (multirail_info) {

        case mv2_num_rail_4:
            /* Set thresholds for Nnum_rail=4 */
            rdma_set_default_parameters_numrail_4(proc);
            break;

        case mv2_num_rail_3:
            /* Set thresholds for Nnum_rail=3 */
            rdma_set_default_parameters_numrail_3(proc);
            break;
        case mv2_num_rail_2:
            /* Set thresholds for Nnum_rail=2 */
            rdma_set_default_parameters_numrail_2(proc);
            break;

        case mv2_num_rail_1:
            /* Set thresholds for Nnum_rail=1 */
            rdma_set_default_parameters_numrail_1(proc);
            break;

            /* mv2_num_rail_unknwon */
        default:
            rdma_set_default_parameters_numrail_unknwn(proc);
            break;
    }

    if ((MV2_HCA_QLGIC_PATH_HT == proc->hca_type) ||
        (MV2_HCA_QLGIC_QIB == proc->hca_type)) {
        rdma_default_qp_ous_rd_atom = 1;
    } else {
        rdma_default_qp_ous_rd_atom = 4;
    }

    if (MV2_HCA_IBM_EHCA == proc->hca_type) {
        rdma_max_inline_size = -1;
    } else if (MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type)) {
        rdma_max_inline_size = 64;
    } else if (MV2_HCA_INTEL_NE020 == proc->hca_type) {
        rdma_max_inline_size = 64;
    } else {
        rdma_max_inline_size = 128 + INLINE_THRESHOLD_ADJUST;
    }

    if (MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type)) {
        /* Trac #423 */
        struct ibv_device_attr dev_attr;
        int mpi_errno = MPI_SUCCESS;

        /*quering device for cq depth */
        mpi_errno = ibv_query_device(proc->nic_context[0], &dev_attr);

        if (!mpi_errno) {
            if (dev_attr.max_cqe < rdma_default_max_cq_size) {
                rdma_default_max_cq_size = dev_attr.max_cqe;
            }
        } else {
            rdma_default_max_cq_size = RDMA_DEFAULT_IWARP_CQ_SIZE;
        }
        rdma_prepost_noop_extra = 8;
    }

    if (MV2_HCA_INTEL_NE020 == proc->hca_type) {
        rdma_default_max_cq_size = 32766;
        rdma_prepost_noop_extra = 8;
    }

    if (proc->has_srq) {
        rdma_credit_preserve = 100;
    } else {
        rdma_credit_preserve = 3;
    }

    return;
}

/* rdma_param_handle_heterogeneity resets control parameters given the arch_hca_type
 * from all ranks. Parameters may change:
 *      rdma_default_mtu
 *      rdma_iba_eager_threshold
 *      proc->has_srq
 *      rdma_credit_preserve
 *      rdma_max_inline_size
 *      rdma_default_qp_ous_rd_atom
 *      rdma_put_fallback_threshold
 *      rdma_get_fallback_threshold
 *      num_rdma_buffer
 *      rdma_vbuf_total_size
 */
void rdma_param_handle_heterogeneity(mv2_arch_hca_type arch_hca_type[],
                                    int pg_size)
{
    mv2_arch_hca_type type;
    mv2_MPIDI_CH3I_RDMA_Process.heterogeneity = 0;
    int i;

    type = arch_hca_type[0];
    for (i = 0; i < pg_size; ++i) {
        if (MV2_IS_ARCH_HCA_TYPE
            (arch_hca_type[i], MV2_ARCH_ANY, MV2_HCA_QLGIC_PATH_HT) ||
            MV2_IS_ARCH_HCA_TYPE(arch_hca_type[i], MV2_ARCH_ANY,
                                 MV2_HCA_QLGIC_QIB)) {
            mv2_MPIDI_CH3I_RDMA_Process.has_srq = 0;
            mv2_MPIDI_CH3I_RDMA_Process.post_send = post_send;
            rdma_credit_preserve = 3;
            rdma_default_qp_ous_rd_atom = 1;
        }

        else if (MV2_IS_ARCH_HCA_TYPE
                 (arch_hca_type[i], MV2_ARCH_ANY, MV2_HCA_MLX_PCI_X)) {
            mv2_MPIDI_CH3I_RDMA_Process.has_srq = 0;
            mv2_MPIDI_CH3I_RDMA_Process.post_send = post_send;
            rdma_credit_preserve = 3;
        }

        else if (MV2_IS_ARCH_HCA_TYPE
                 (arch_hca_type[i], MV2_ARCH_ANY, MV2_HCA_IBM_EHCA)) {
            mv2_MPIDI_CH3I_RDMA_Process.has_srq = 0;
            mv2_MPIDI_CH3I_RDMA_Process.post_send = post_send;
            rdma_credit_preserve = 3;
            rdma_max_inline_size = -1;
        }

        if (arch_hca_type[i] != type) {
            mv2_MPIDI_CH3I_RDMA_Process.heterogeneity = 1;
        }

        DEBUG_PRINT("rank %d, type %d\n", i, arch_hca_type[i]);
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity) {
        DEBUG_PRINT("heterogenous hcas detected\n");
        rdma_default_mtu = IBV_MTU_1024;
        rdma_vbuf_total_size = 8 * 1024 + EAGER_THRESHOLD_ADJUST;
        rdma_fp_buffer_size = 8 * 1024;
        rdma_iba_eager_threshold = VBUF_BUFFER_SIZE;
        rdma_max_inline_size = (rdma_max_inline_size == -1) ? -1 : 64;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 192 * 1024;
        num_rdma_buffer = 16;
    }
}

void rdma_set_rdma_fast_path_params(int num_proc)
{
    char *value = NULL;

#if defined(CKPT)
    mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path = 0;
    rdma_polling_set_limit = 0;
#else /* defined(CKPT) */
    if ((value = getenv("MV2_USE_RDMA_FAST_PATH")) != NULL) {
        mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path = !!atoi(value);

        if (!mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path) {
            rdma_polling_set_limit = 0;
            mv2_rdma_fast_path_preallocate_buffers = 0;
        }
    } else {
        mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path = 1;
    }
#endif /* defined(CKPT) */

#ifdef _ENABLE_XRC_
    if (!USE_XRC) {
#endif
        /* Automatically turn off RDMA fast path */
        if (rdma_use_blocking) {
            mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path = 0;
        }
#ifdef _ENABLE_XRC_
    }
#endif

    if (mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path) {
        if ((value = getenv("MV2_RDMA_FAST_PATH_BUF_SIZE")) != NULL) {
            rdma_fp_buffer_size = atoi(value);
        }
    
        if ((value = getenv("MV2_POLLING_SET_LIMIT")) != NULL) {
            rdma_polling_set_limit = atoi(value);
            if (rdma_polling_set_limit == -1) {
                rdma_polling_set_limit = log_2(num_proc);
            }
        } else {
            rdma_polling_set_limit = RDMA_DEFAULT_POLLING_SET_LIMIT;
        }
    
        if ((value = getenv("MV2_POLLING_SET_THRESHOLD")) != NULL) {
            rdma_polling_set_threshold = atoi(value);
        }
    
        if ((value = getenv("MV2_RDMA_EAGER_LIMIT")) != NULL) {
            rdma_eager_limit = atoi(value);
            if (rdma_eager_limit < 0) {
                rdma_eager_limit = 0;
            }
        }
        if ((value = getenv("MV2_NUM_RDMA_BUFFER")) != NULL) {
            num_rdma_buffer = atoi(value);
        }
        if ((value = getenv("MV2_RDMA_FAST_PATH_PREALLOCATE_BUFFERS")) != NULL) {
            mv2_rdma_fast_path_preallocate_buffers = !!atoi(value);
        }
    }

    return;
}

void rdma_get_user_parameters(int num_proc, int me)
{
    char *value;

    int dpm_support=0;
    if((value=getenv("MV2_SUPPORT_DPM"))!=NULL) {
	    dpm_support=atoi(value);
    }

    if(!dpm_support) {
	    /* Ensure BW tests have enough RFP buffers*/
	    if (2 == num_proc) {
		    num_rdma_buffer = 64;
	    }
    }

    /* Check for a system report. See sysreport.h and sysreport.c */
    value = getenv("MV2_SYSREPORT");
    if (value != NULL) {
        enable_sysreport = atoi(value);
    }

    if ((value = getenv("MV2_DEFAULT_MTU")) != NULL) {

        if (strncmp(value, "IBV_MTU_256", 11) == 0) {
            rdma_default_mtu = IBV_MTU_256;
        } else if (strncmp(value, "IBV_MTU_512", 11) == 0) {
            rdma_default_mtu = IBV_MTU_512;
        } else if (strncmp(value, "IBV_MTU_1024", 12) == 0) {
            rdma_default_mtu = IBV_MTU_1024;
        } else if (strncmp(value, "IBV_MTU_2048", 12) == 0) {
            rdma_default_mtu = IBV_MTU_2048;
        } else if (strncmp(value, "IBV_MTU_4096", 12) == 0) {
            rdma_default_mtu = IBV_MTU_4096;
        } else {
            rdma_default_mtu = IBV_MTU_1024;
        }
    }

    /* Number of CQE's retrieved per poll */
    if ((value = getenv("MV2_NUM_CQES_PER_POLL")) != NULL) {
        rdma_num_cqes_per_poll = atoi(value);
        if (rdma_num_cqes_per_poll <= 0 ||
            rdma_num_cqes_per_poll >= RDMA_MAX_CQE_ENTRIES_PER_POLL) {
            rdma_num_cqes_per_poll = RDMA_MAX_CQE_ENTRIES_PER_POLL;
        }
    }
 
    /* Get number of ports/HCA used by a process */
    if ((value = getenv("MV2_NUM_PORTS")) != NULL) {
        rdma_num_ports = atoi(value);
        if (rdma_num_ports > MAX_NUM_PORTS) {
            rdma_num_ports = MAX_NUM_PORTS;
            MPL_usage_printf("Warning, max ports per hca is %d, change %s in "
                              "ibv_param.h to overide the option\n",
                              MAX_NUM_PORTS, "MAX_NUM_PORTS");
        }
    }

    /* Get number of qps/port used by a process */
    if ((value = getenv("MV2_NUM_QP_PER_PORT")) != NULL) {

        rdma_num_qp_per_port = atoi(value);

        if (rdma_num_qp_per_port > MAX_NUM_QP_PER_PORT) {
            rdma_num_qp_per_port = MAX_NUM_QP_PER_PORT;
            MPL_usage_printf("Warning, max qps per port is %d, change %s in "
                              "ibv_param.h to overide the option\n",
                              MAX_NUM_QP_PER_PORT, "MAX_NUM_QP_PER_PORT");
        }
#ifdef _ENABLE_UD_
        if (rdma_enable_only_ud || rdma_enable_hybrid) {
            rdma_num_qp_per_port = 1;
            PRINT_INFO((me==0), "Cannot have more than one QP with UD_ONLY / Hybrid mode.\n");
            PRINT_INFO((me==0), "Resetting MV2_NUM_QP_PER_PORT to 1.\n");
        }
#endif /* _ENABLE_UD_ */
    }

    if ((value = getenv("MV2_PIN_POOL_SIZE")) != NULL) {
        rdma_pin_pool_size = atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_MAX_CQ_SIZE")) != NULL) {
        rdma_default_max_cq_size = atoi(value);
    }
    /* We have read the value of the rendezvous threshold, and the number of
     * rails used for communication, increase the striping threshold
     * accordingly */

    /* Messages in between will use the rendezvous protocol, however will
     * not be striped */

    striping_threshold =
        rdma_vbuf_total_size * rdma_num_ports * rdma_num_qp_per_port *
        rdma_num_hcas;

    if ((value = getenv("MV2_IBA_EAGER_THRESHOLD")) != NULL) {
        rdma_iba_eager_threshold =
            user_val_to_bytes(value, "MV2_IBA_EAGER_THRESHOLD");
    }

    if ((value = getenv("MV2_STRIPING_THRESHOLD")) != NULL) {
        striping_threshold = user_val_to_bytes(value, "MV2_STRIPING_THRESHOLD");
        if (striping_threshold <= 0) {
            /* Invalid value - set to computed value */
            striping_threshold =
                rdma_vbuf_total_size * rdma_num_ports * rdma_num_qp_per_port *
                rdma_num_hcas;
        }
        if (striping_threshold < rdma_iba_eager_threshold) {
            /* checking to make sure that the striping threshold is not less
             * than the RNDV threshold since it won't work as expected.
             */
            striping_threshold = rdma_iba_eager_threshold;
        }
    }

    if ((value = getenv("MV2_RAIL_SHARING_MED_MSG_THRESHOLD")) != NULL) {
        rdma_med_msg_rail_sharing_threshold =
            user_val_to_bytes(value, "MV2_RAIL_SHARING_MED_MSG_THRESHOLD");
        if (rdma_med_msg_rail_sharing_threshold <= 0) {
            rdma_med_msg_rail_sharing_threshold =
                RDMA_DEFAULT_MED_MSG_RAIL_SHARING_THRESHOLD;
        }
    }

    rdma_large_msg_rail_sharing_threshold = rdma_vbuf_total_size;

    if ((value = getenv("MV2_RAIL_SHARING_LARGE_MSG_THRESHOLD")) != NULL) {
        rdma_large_msg_rail_sharing_threshold =
            user_val_to_bytes(value, "MV2_RAIL_SHARING_LARGE_MSG_THRESHOLD");
        if (rdma_large_msg_rail_sharing_threshold <= 0) {
            rdma_large_msg_rail_sharing_threshold = rdma_vbuf_total_size;
        }
    }

    if ((value = getenv("MV2_DEFAULT_PUT_GET_LIST_SIZE")) != NULL) {
        rdma_default_put_get_list_size = atoi(value);
    }
    if ((value = getenv("MV2_EAGERSIZE_1SC")) != NULL) {
        rdma_eagersize_1sc = atoi(value);
    }
    if ((value = getenv("MV2_PUT_FALLBACK_THRESHOLD")) != NULL) {
        rdma_put_fallback_threshold = atoi(value);
    }
    if ((value = getenv("MV2_GET_FALLBACK_THRESHOLD")) != NULL) {
        rdma_get_fallback_threshold =
            user_val_to_bytes(value, "MV2_GET_FALLBACK_THRESHOLD");
    }
    if ((value = getenv("MV2_DEFAULT_PORT")) != NULL) {
        rdma_default_port = atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_GID_INDEX")) != NULL) {
        rdma_default_gid_index = atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_QP_OUS_RD_ATOM")) != NULL) {
        rdma_default_qp_ous_rd_atom = (uint8_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_MAX_RDMA_DST_OPS")) != NULL) {
        rdma_default_max_rdma_dst_ops = (uint8_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_PSN")) != NULL) {
        rdma_default_psn = (uint32_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_PKEY")) != NULL) {
        rdma_default_pkey = (uint16_t)strtol(value, (char **) NULL,0) & PKEY_MASK;
    }
    if ((value = getenv("MV2_DEFAULT_MIN_RNR_TIMER")) != NULL) {
        rdma_default_min_rnr_timer = (uint8_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_SERVICE_LEVEL")) != NULL) {
        rdma_default_service_level = (uint8_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_TIME_OUT")) != NULL) {
        rdma_default_time_out = (uint8_t) atol(value);
    }
    if ((value = getenv("MV2_DEFAULT_STATIC_RATE")) != NULL) {
        rdma_default_static_rate = (uint8_t) atol(value);
    }
    if ((value = getenv("MV2_DEFAULT_SRC_PATH_BITS")) != NULL) {
        rdma_default_src_path_bits = (uint8_t) atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_RETRY_COUNT")) != NULL) {
        rdma_default_retry_count = (uint8_t) atol(value);
    }
    if ((value = getenv("MV2_DEFAULT_RNR_RETRY")) != NULL) {
        rdma_default_rnr_retry = (uint8_t) atol(value);
    }
    if ((value = getenv("MV2_DEFAULT_MAX_SG_LIST")) != NULL) {
        rdma_default_max_sg_list = (uint32_t) atol(value);
    }
    if ((value = getenv("MV2_DEFAULT_MAX_SEND_WQE")) != NULL) {
        rdma_default_max_send_wqe = atol(value);
    }
    if ((value = getenv("MV2_CM_WAIT_TIME")) != NULL) {
        mv2_cm_wait_time = atoi(value);
        if (mv2_cm_wait_time <= 0) {
            mv2_cm_wait_time = DEF_MV2_CM_WAIT_TIME;
        }

    }

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    if ((value = getenv("MV2_UD_MTU")) != NULL) {
        rdma_default_ud_mtu = atol(value);
        if (rdma_default_ud_mtu < 256 || rdma_default_ud_mtu > 4096) {
            MPL_usage_printf("Invalid value used for UD MTU (Min: 256; Max: 4K). Resetting to default value (2K)\n");
            rdma_default_ud_mtu = 2048;
        }
    }
#endif /* #if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)*/
#ifdef _ENABLE_UD_
    if ((value = getenv("MV2_HYBRID_MAX_RC_CONN")) != NULL) {
        rdma_hybrid_max_rc_conn = atoi(value);
        if (rdma_enable_only_ud && rdma_hybrid_max_rc_conn) {
            PRINT_INFO((me == 0),
                       "User requested only UD. Resetting MV2_HYBRID_MAX_RC_CONN to 0.\n");
            rdma_hybrid_max_rc_conn = 0;
        }
    } else {
        if (rdma_enable_only_ud) {
            rdma_hybrid_max_rc_conn = 0;
        }
    }
    if ((value = getenv("MV2_UD_NUM_MSG_LIMIT")) != NULL) {
        rdma_ud_num_msg_limit = atoi(value);
    }
    if ((value = getenv("MV2_UD_SENDWINDOW_SIZE")) != NULL) {
        rdma_default_ud_sendwin_size = atoi(value);
    }
    if ((value = getenv("MV2_UD_RECVWINDOW_SIZE")) != NULL) {
        rdma_default_ud_recvwin_size = atoi(value);
    }
    if ((value = getenv("MV2_UD_RETRY_TIMEOUT")) != NULL) {
        rdma_ud_retry_timeout = atoi(value);
    }
    if ((value = getenv("MV2_UD_MAX_RETRY_TIMEOUT")) != NULL) {
        rdma_ud_max_retry_timeout = atoi(value);
    }
    if ((value = getenv("MV2_UD_PROGRESS_SPIN")) != NULL) {
        rdma_ud_progress_spin = atoi(value);
    }
    if ((value = getenv("MV2_UD_RETRY_COUNT")) != NULL) {
        rdma_ud_max_retry_count = atoi(value);
    }
    if ((value = getenv("MV2_UD_PROGRESS_TIMEOUT")) != NULL) {
        rdma_ud_progress_timeout = atoi(value);
    }
    if ((value = getenv("MV2_UD_MAX_SEND_WQE")) != NULL) {
        rdma_default_max_ud_send_wqe = atol(value);
    }
    if ((value = getenv("MV2_UD_MAX_RECV_WQE")) != NULL) {
        rdma_default_max_ud_recv_wqe = atol(value);
    }
    if ((value = getenv("MV2_UD_VBUF_POOL_SIZE")) != NULL) {
        rdma_ud_vbuf_pool_size = atol(value);
    }

    if ((value = getenv("MV2_UD_MAX_ACK_PENDING")) != NULL) {
        rdma_ud_max_ack_pending = atoi(value);
    } else {
        rdma_ud_max_ack_pending = (rdma_default_ud_sendwin_size / 4);
    }
    if ((value = getenv("MV2_USE_UD_ZCOPY")) != NULL) {
        rdma_use_ud_zcopy = atoi(value);
    }
    rdma_ud_zcopy_threshold = rdma_iba_eager_threshold;
    if ((value = getenv("MV2_UD_ZCOPY_THRESHOLD")) != NULL) {
        rdma_ud_zcopy_threshold = atoi(value);
    }
    if ((value = getenv("MV2_UD_NUM_ZCOPY_RNDV_QPS")) != NULL) {
        rdma_ud_num_rndv_qps = atoi(value);
    }
    if ((value = getenv("MV2_UD_ZCOPY_RQ_SIZE")) != NULL) {
        rdma_ud_zcopy_rq_size = atoi(value);
    }
#ifdef _MV2_UD_DROP_PACKET_RATE_
    if ((value = getenv("MV2_UD_DROP_PACKET_RATE")) != NULL) {
        ud_drop_packet_rate = atoi(value);
    }
#endif

#endif

#if defined(_MCST_SUPPORT_)
    if ((value = getenv("MV2_MCAST_ENABLE_REL")) != NULL) {
        mcast_enable_rel = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_USE_MCAST_NACK")) != NULL) {
        mcast_use_mcast_nack = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_NUM_NODES_THRESHOLD")) != NULL) {
        int env_num_threshold = atoi(value);
        if (env_num_threshold < MCAST_MIN_THRESHOLD) {
            mcast_num_nodes_threshold = MCAST_MIN_THRESHOLD;
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]:"
                    " MV2_MCAST_NUM_NODES_THRESHOLD cannot be less than %d, setting it to %d\n",MCAST_MIN_THRESHOLD,MCAST_MIN_THRESHOLD);
        }
        else{
             mcast_num_nodes_threshold = env_num_threshold;
        }
    }
    if ((value = getenv("MV2_MCAST_MAX_RECV_WQE")) != NULL) {
        mcast_max_ud_recv_wqe = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_WINDOW_SIZE")) != NULL) {
        mcast_window_size = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_DROP_PACKET_RATE")) != NULL) {
        mcast_drop_packet_rate = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_RETRY_TIMEOUT")) != NULL) {
        mcast_retry_timeout = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_MAX_RETRY_TIMEOUT")) != NULL) {
        mcast_max_retry_timeout = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_NSPIN_THRESHOLD")) != NULL) {
        mcast_nspin_threshold = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_COMM_INIT_TIMEOUT")) != NULL) {
        mcast_comm_init_timeout = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_COMM_INIT_RETRIES")) != NULL) {
        mcast_comm_init_retries = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_SKIP_LOOPBACK")) != NULL) {
        mcast_skip_loopback = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_BCAST_MIN_MSG")) != NULL) {
        mcast_bcast_min_msg = atoi(value);
    }
    if ((value = getenv("MV2_MCAST_BCAST_MAX_MSG")) != NULL) {
        mcast_bcast_max_msg = atoi(value);
    }
#endif

    if ((value = getenv("MV2_DEFAULT_MAX_RECV_WQE")) != NULL) {
        rdma_default_max_recv_wqe = atol(value);
    }
    if ((value = getenv("MV2_NDREG_ENTRIES_MAX")) != NULL) {
        rdma_ndreg_entries_max = atol(value);
    }
    if ((value = getenv("MV2_NDREG_ENTRIES")) != NULL) {
        rdma_ndreg_entries =
           (((unsigned int)atoi(value) < rdma_ndreg_entries_max) ?
            (unsigned int)atoi(value) : rdma_ndreg_entries_max);
    } else {
        rdma_ndreg_entries =
           ((RDMA_NDREG_ENTRIES + 2*num_proc < rdma_ndreg_entries_max) ?
             RDMA_NDREG_ENTRIES + 2*num_proc : rdma_ndreg_entries_max);
    }
    if ((value = getenv("MV2_DREG_CACHE_LIMIT")) != NULL) {
        rdma_dreg_cache_limit = atol(value);
    }
#if defined(_ENABLE_UD_)
    if (rdma_enable_only_ud) {
        rdma_vbuf_pool_size = 0;
        mv2_MPIDI_CH3I_RDMA_Process.has_srq = 0;
    }
#endif
    if (rdma_initial_prepost_depth <= rdma_prepost_noop_extra) {
        rdma_initial_credits = rdma_initial_prepost_depth;
    } else {
        rdma_initial_credits =
            rdma_initial_prepost_depth - rdma_prepost_noop_extra;
    }

    rdma_rq_size =
        rdma_prepost_depth + rdma_prepost_rendezvous_extra +
        rdma_prepost_noop_extra;

    if ((value = getenv("MV2_USE_HWLOC_CPU_BINDING")) != NULL) {
        use_hwloc_cpu_binding = atoi(value);
    }
    if ((value = getenv("MV2_THREAD_YIELD_SPIN_THRESHOLD")) != NULL) {
        rdma_polling_spin_count_threshold = atol(value);
    }
    if ((value = getenv("MV2_USE_THREAD_YIELD")) != NULL) {
        mv2_use_thread_yield = atoi(value);
    }
    if ((value = getenv("MV2_NUM_SPINS_BEFORE_LOCK")) != NULL) {
        mv2_spins_before_lock = atoi(value);
    }
    if ((value = getenv("MV2_ASYNC_THREAD_STACK_SIZE")) != NULL) {
        rdma_default_async_thread_stack_size = atoi(value);
        if (rdma_default_async_thread_stack_size < 1 << 10) {
            MPL_usage_printf
                ("Warning! Too small stack size for async thread (%d).  "
                 "Reset to %d\n", rdma_vbuf_secondary_pool_size,
                 RDMA_DEFAULT_ASYNC_THREAD_STACK_SIZE);
            rdma_default_async_thread_stack_size =
                RDMA_DEFAULT_ASYNC_THREAD_STACK_SIZE;
        }
    }
    if ((value = getenv("MV2_USE_HUGEPAGES")) != NULL) {
        rdma_enable_hugepage = atoi(value);
    }

    /* Read VBUF related user parameters */
    rdma_get_vbuf_user_parameters(num_proc, me);

    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        rdma_credit_preserve =
            (mv2_srq_fill_size >
             200) ? (mv2_srq_fill_size - 100) : (mv2_srq_fill_size / 4);
    }

    return;

}

static inline void rdma_get_vbuf_user_parameters(int num_proc, int me)
{
    char *value = NULL;

    if ((value = getenv("MV2_MEMORY_OPTIMIZATION")) != NULL) {
        rdma_memory_optimization = !!atoi(value);
    }
#ifdef _ENABLE_UD_
    if (!rdma_enable_only_ud && rdma_memory_optimization)
#else
    if (rdma_memory_optimization)
#endif
    {
        mv2_srq_limit           = RDMA_OPT_VBUF_POOL_SIZE/8;
        mv2_srq_fill_size       = RDMA_OPT_VBUF_POOL_SIZE;
        rdma_vbuf_pool_size     = RDMA_OPT_VBUF_POOL_SIZE;
        rdma_credit_preserve    = RDMA_OPT_VBUF_POOL_SIZE/6;
        rdma_vbuf_secondary_pool_size = RDMA_OPT_VBUF_SECONDARY_POOL_SIZE;
    }

    if ((value = getenv("MV2_SRQ_MAX_SIZE")) != NULL) {
        mv2_srq_alloc_size = (uint32_t) atoi(value);
#if defined(RDMA_CM)
    } else if (MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_RDMA_CM) {
        /* When using RDMA_CM, we cannot support very large SRQ. So, unless user
         * set it, reduce the max_srq_size to 4K */
        mv2_srq_alloc_size = 4096;
#endif /*defined(RDMA_CM)*/
    }

    if ((value = getenv("MV2_SRQ_SIZE")) != NULL) {
        mv2_srq_fill_size = (uint32_t) atoi(value);
    }
 
    if ((value = getenv("MV2_SRQ_LIMIT")) != NULL) {
        mv2_srq_limit = (uint32_t) atoi(value);
 
        if (mv2_srq_limit > mv2_srq_fill_size) {
            MPL_usage_printf("SRQ limit shouldn't be greater than SRQ size\n");
        }
    }
 
    if ((value = getenv("MV2_MAX_INLINE_SIZE")) != NULL) {
        rdma_max_inline_size = atoi(value);
    }
    if ((value = getenv("MV2_VBUF_TOTAL_SIZE")) != NULL) {
        if (rdma_memory_optimization) {
            if (RDMA_OPT_MIN_VBUF_POOL_SIZE <
                user_val_to_bytes(value, "MV2_VBUF_TOTAL_SIZE")) {
                rdma_vbuf_total_size =
                    user_val_to_bytes(value,
                                      "MV2_VBUF_TOTAL_SIZE") +
                    EAGER_THRESHOLD_ADJUST;
            } else {
                /* We do not accept vbuf size < RDMA_MIN_VBUF_POOL_SIZE */
                MPL_usage_printf("Warning, it is inefficient to use a value for"
                                  "VBUF which is less than %d. Retaining the"
                                  " system default value of %d\n",
                                  RDMA_OPT_MIN_VBUF_POOL_SIZE,
                                  rdma_vbuf_total_size);
            }
        } else {
            if (RDMA_MIN_VBUF_POOL_SIZE <
                user_val_to_bytes(value, "MV2_VBUF_TOTAL_SIZE")) {
                rdma_vbuf_total_size =
                    user_val_to_bytes(value,
                                      "MV2_VBUF_TOTAL_SIZE") +
                    EAGER_THRESHOLD_ADJUST;
            } else {
                /* We do not accept vbuf size < RDMA_MIN_VBUF_POOL_SIZE */
                MPL_usage_printf("Warning, it is inefficient to use a value for"
                                  "VBUF which is less than %d. Retaining the"
                                  " system default value of %d\n",
                                  RDMA_MIN_VBUF_POOL_SIZE,
                                  rdma_vbuf_total_size);
            }
        }
    }
    if ((value = getenv("MV2_VBUF_MAX")) != NULL) {
        rdma_vbuf_max = atoi(value);
    }
    if ((value = getenv("MV2_INITIAL_PREPOST_DEPTH")) != NULL) {
        rdma_initial_prepost_depth = atoi(value);
    }
    if ((value = getenv("MV2_PREPOST_DEPTH")) != NULL) {
        rdma_prepost_depth = atoi(value);
    }
    if ((value = getenv("MV2_VBUF_POOL_SIZE")) != NULL) {
        rdma_vbuf_pool_size = atoi(value);
    }
#ifdef _ENABLE_UD_
    if (!rdma_enable_only_ud && rdma_vbuf_pool_size <= 10)
#else
    if (rdma_vbuf_pool_size <= 10)
#endif
    {
        if (rdma_memory_optimization) {
            rdma_vbuf_pool_size = RDMA_OPT_VBUF_POOL_SIZE;
            MPL_usage_printf("Warning! Too small vbuf pool size (%d).  "
                              "Reset to %d\n", rdma_vbuf_pool_size,
                              RDMA_OPT_VBUF_POOL_SIZE);
        } else {
            rdma_vbuf_pool_size = RDMA_VBUF_POOL_SIZE;
            MPL_usage_printf("Warning! Too small vbuf pool size (%d).  "
                              "Reset to %d\n", rdma_vbuf_pool_size,
                              RDMA_VBUF_POOL_SIZE);
        }
    }
    if ((value = getenv("MV2_VBUF_SECONDARY_POOL_SIZE")) != NULL) {
        rdma_vbuf_secondary_pool_size = atoi(value);
    }
    if (rdma_vbuf_secondary_pool_size <= 0) {
        if (rdma_memory_optimization) {
            rdma_vbuf_secondary_pool_size = RDMA_OPT_VBUF_SECONDARY_POOL_SIZE;
            MPL_usage_printf("Warning! Too small secondary vbuf pool size (%d)"
                              ". Reset to %d\n", rdma_vbuf_secondary_pool_size,
                              RDMA_OPT_VBUF_SECONDARY_POOL_SIZE);
        } else {
            rdma_vbuf_secondary_pool_size = RDMA_VBUF_SECONDARY_POOL_SIZE;
            MPL_usage_printf("Warning! Too small secondary vbuf pool size (%d)"
                              ". Reset to %d\n", rdma_vbuf_secondary_pool_size,
                              RDMA_VBUF_SECONDARY_POOL_SIZE);
        }
    }

    int i = 0;
    int result = 0;
    int alignment_dma = getpagesize();
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        rdma_num_vbuf_pools = MV2_MAX_NUM_VBUF_POOLS;

        if ((value = getenv("MV2_CUDA_BLOCK_SIZE")) != NULL) {
            rdma_cuda_block_size = atoi(value);
        }
        if ((value = getenv("MV2_CUDA_NUM_RNDV_BLOCKS")) != NULL) {
            rdma_num_cuda_rndv_blocks = atoi(value);
        }

        int default_vbuf_sizes[] = DEFAULT_CUDA_VBUF_SIZES;
        int default_vbuf_init_count[] = DEFAULT_CUDA_VBUF_POOL_SIZE;
        int default_vbuf_secondary_count[] =
            DEFAULT_CUDA_VBUF_SECONDARY_POOL_SIZE;

        result = MPIU_Memalign((void**) &rdma_vbuf_pools, alignment_dma,
                                (sizeof(vbuf_pool_t) * rdma_num_vbuf_pools));
        if ((result != 0) || (NULL == rdma_vbuf_pools)) {
            ibv_error_abort(GEN_EXIT_ERR, "Unable to malloc vbuf_pool");
        }
        MPIU_Memset(rdma_vbuf_pools, 0,
                    sizeof(vbuf_pool_t) * rdma_num_vbuf_pools);

        for (i = 0; i < rdma_num_vbuf_pools; i++) {
            RDMA_VBUF_POOL_INIT(rdma_vbuf_pools[i]);
            rdma_vbuf_pools[i].buf_size = default_vbuf_sizes[i];
            rdma_vbuf_pools[i].incr_count = default_vbuf_secondary_count[i];
            rdma_vbuf_pools[i].initial_count = default_vbuf_init_count[i];
            rdma_vbuf_pools[i].index = i;
        }
    } else
#endif
    {
#ifdef _ENABLE_CUDA_
        /* If built with CUDA support, the last VBUF pool is for CUDA VBUF.
         * This is not needed if CUDA support is not enabled at runtime */
        rdma_num_vbuf_pools = MV2_MAX_NUM_VBUF_POOLS-1;
#else
        rdma_num_vbuf_pools = MV2_MAX_NUM_VBUF_POOLS;
#endif

        int default_vbuf_sizes[] = DEFAULT_VBUF_SIZES;
        int default_vbuf_init_count[] = DEFAULT_VBUF_POOL_SIZE;
        int default_vbuf_secondary_count[] = DEFAULT_VBUF_SECONDARY_POOL_SIZE;

        result = MPIU_Memalign((void**) &rdma_vbuf_pools, alignment_dma,
                                (sizeof(vbuf_pool_t) * rdma_num_vbuf_pools));
        if ((result != 0) || (NULL == rdma_vbuf_pools)) {
            ibv_error_abort(GEN_EXIT_ERR, "Unable to malloc vbuf_pool");
        }
        MPIU_Memset(rdma_vbuf_pools, 0,
                    sizeof(vbuf_pool_t) * rdma_num_vbuf_pools);
        MPIU_Memset(&mv2_srq_repost_pool, 0, sizeof(vbuf_pool_t));

        for (i = 0; i < rdma_num_vbuf_pools; i++) {
            RDMA_VBUF_POOL_INIT(rdma_vbuf_pools[i]);
            rdma_vbuf_pools[i].buf_size = default_vbuf_sizes[i];
            rdma_vbuf_pools[i].incr_count = default_vbuf_secondary_count[i];
            rdma_vbuf_pools[i].initial_count = default_vbuf_init_count[i];
            rdma_vbuf_pools[i].index = i;
        }
    }
    return;
}

/* This function is specifically written to make sure that HSAM
 * parameters are configured correctly */

static int check_hsam_parameters(void)
{
    char *value;
    int size;

    /* Get the number of processes */
    UPMI_GET_SIZE(&size);

    /* If the number of processes is less than 64, we can afford * to
     * have more RC QPs and hence a value of 4 is chosen, for * other
     * cases, a value of 2 is chosen */

    /* (rdma_num_qp_per_port/ stripe factor) represents the number
     * of QPs which will be chosen for data transfer at a given point */

    /* If the user has not specified any value, then perform
     * this tuning */

    if ((value = getenv("MV2_NUM_QP_PER_PORT")) != NULL) {
        rdma_num_qp_per_port = atoi(value);
        if (rdma_num_qp_per_port <= 2) {
            stripe_factor = 1;
        } else {
            stripe_factor = (rdma_num_qp_per_port / 2);
        }
    } else {
        /* Speculated value */

        /* The congestion is actually never seen for less
         * than 8 nodes */
        if ((size > 8) && (size < 64)) {
            rdma_num_qp_per_port = 4;
            stripe_factor = (rdma_num_qp_per_port / 2);
        } else {
            rdma_num_qp_per_port = 2;
            stripe_factor = 1;
        }
    }

    return MPI_SUCCESS;
}

void rdma_get_pm_parameters(mv2_MPIDI_CH3I_RDMA_Process_t * proc)
{
    int ring_setup, i;
    char *value;
    int my_rank = -1;

    UPMI_GET_RANK(&my_rank);

    value = getenv("MPIRUN_RSH_LAUNCH");
    if (value != NULL && (atoi(value) == 1)) {
        using_mpirun_rsh = 1;
    }
#if defined(RDMA_CM)
    if ((value = getenv("MV2_USE_RDMA_CM")) != NULL) {
#if !defined(ROMIO_IME)
        proc->use_rdma_cm = !!atoi(value);
#else
        if (value && atoi(value) && (my_rank == 0)) {
            MPL_error_printf("Error: IME FS does not work with RDMA CM. "
                             "Proceeding without RDMA support.\n");
        }
#endif
#ifdef _ENABLE_XRC_
        /* XRC will not work with RDMA_CM */
        if (proc->use_rdma_cm) {
            USE_XRC = 0;
            value = getenv("MV2_USE_XRC");
            if (value && (my_rank == 0)) {
                if (atoi(value)) {
                    MPL_error_printf("Error: XRC does not work with RDMA CM. "
                                      "Proceeding without XRC support.\n");
                }
            }
        }
#endif
    }
#endif

    if ((value = getenv("MV2_USE_RoCE")) != NULL) {
        use_iboeth = !!atoi(value);
    }

    switch (MPIDI_CH3I_Process.cm_type) {
        case MPIDI_CH3I_CM_ON_DEMAND:
#if defined(RDMA_CM)
        case MPIDI_CH3I_CM_RDMA_CM:
#endif
            ring_setup = 1;
            if (using_mpirun_rsh) {
                ring_setup = 0;
#ifdef _ENABLE_XRC_
                if (USE_XRC) {
                    ring_setup = 1;
                }
#endif
            }
            proc->has_ring_startup = (value =
                                      getenv("MV2_USE_RING_STARTUP")) !=
                NULL ? !!atoi(value) : ring_setup;
#ifdef CKPT
            if (!using_mpirun_rsh)
                proc->has_ring_startup = 0;

#endif /* CKPT */
            break;
        default:
            proc->has_ring_startup = (value =
                                      getenv("MV2_USE_RING_STARTUP")) !=
                NULL ? !!atoi(value) : 0;
#ifdef CKPT
            if (!using_mpirun_rsh)
                proc->has_ring_startup = 0;
#endif /* CKPT */

            break;
    }

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        proc->has_ring_startup = 0;
    }
#endif
    if ((value = getenv("MV2_ON_DEMAND_UD_INFO_EXCHANGE")) != NULL) {
        mv2_on_demand_ud_info_exchange = !!atoi(value);
        if (mv2_on_demand_ud_info_exchange) {
            proc->has_ring_startup = 0;
        }
    }

#if (defined(HAVE_PMI2_KVS_IFENCE) && defined(HAVE_PMI2_KVS_WAIT)) \
    || (defined(HAVE_PMI_IBARRIER) && defined(HAVE_PMI_WAIT))
    mv2_use_pmi_ibarrier = 1; /* enable by default if available */

    if ((value = getenv("MV2_USE_PMI_IBARRIER")) != NULL) {
        mv2_use_pmi_ibarrier = !!atoi(value);
    }

    if (mv2_use_pmi_ibarrier) {
        mv2_on_demand_ud_info_exchange = 1;
        proc->has_ring_startup = 0;
    }

#endif

#if defined(HAVE_PMI2_IALLGATHER) && defined(HAVE_PMI2_IALLGATHER_WAIT)
    if ((value = getenv("MV2_USE_PMI_IALLGATHER")) != NULL) {
        mv2_use_pmi_iallgather = !!atoi(value);
    }

    if (mv2_use_pmi_iallgather) {
        mv2_use_pmi_ibarrier = 0;
        mv2_on_demand_ud_info_exchange = 1;
        proc->has_ring_startup = 0;
    }
#endif

    if (proc->has_ring_startup ) {
        /* The on demand ud info is temporarily disabled
         * Revisit the issue later */
        mv2_on_demand_ud_info_exchange = 0;
    }

    if ((value = getenv("MV2_NUM_HCAS")) != NULL) {
        rdma_num_req_hcas = atoi(value);

        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;

        if (rdma_num_req_hcas > MAX_NUM_HCAS) {
            rdma_num_req_hcas = MAX_NUM_HCAS;

            MPL_msg_printf("Warning, max hca is %d, change %s in ibv_param.h "
                            "to overide the option\n", MAX_NUM_HCAS,
                            "MAX_NUM_HCAS");
        }
    }

    for (i = 0; i < MAX_NUM_HCAS; ++i) {
        strncpy(rdma_iba_hcas[i], RDMA_IBA_NULL_HCA, 32);
    }

    if ((value = getenv("MV2_IBA_HCA")) != NULL) {
        rdma_multirail_usage_policy = MV2_MRAIL_SHARING;
        rdma_num_req_hcas = 0;
        {
            char *tok = NULL;
            char *inp = value;

            tok = strtok(inp, ":");
            inp = NULL;
            while (tok != NULL) {
                strncpy(rdma_iba_hcas[rdma_num_req_hcas], tok, 32);
                tok = strtok(inp, ":");
                DEBUG_PRINT("tok = %s, hca name = %s, hca num = %d\n", tok,
                            rdma_iba_hcas[rdma_num_req_hcas],
                            rdma_num_req_hcas);
                rdma_num_req_hcas++;
            }
        }
    }

    if ((value = getenv("MV2_HOMOGENEOUS_CLUSTER")) != NULL) {
        mv2_homogeneous_cluster = atoi(value);
    }
}

void mv2_print_env_info(mv2_MPIDI_CH3I_RDMA_Process_t * proc, struct coll_info *colls_arch_hca)
{
    int i = 0;
    mv2_arch_type arch_type = MV2_GET_ARCH(proc->arch_hca_type);
    mv2_hca_type hca_type = MV2_GET_HCA(proc->arch_hca_type);
    mv2_cpu_family_type family_type = mv2_get_cpu_family();
    fprintf(stderr, "\n MVAPICH2-%s Parameters\n", MPIR_Version_string);
    fprintf(stderr,
            "---------------------------------------------------------------------\n");
    fprintf(stderr, "\tPROCESSOR ARCH NAME            : %s\n",
            mv2_get_arch_name(arch_type));
    fprintf(stderr, "\tPROCESSOR FAMILY NAME          : %s\n",
            mv2_get_cpu_family_name(family_type));
    fprintf(stderr, "\tPROCESSOR MODEL NUMBER         : %d\n",
            mv2_get_cpu_model());
    fprintf(stderr, "\tHCA NAME                       : %s\n",
            mv2_get_hca_name(hca_type));
    fprintf(stderr, "\tHETEROGENEOUS HCA              : %s\n",
            (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity) ? "YES" : "NO");
    if (!SMP_ONLY) {
        fprintf(stderr, "\tMV2_VBUF_TOTAL_SIZE            : %d\n",
                rdma_vbuf_total_size);
        fprintf(stderr, "\tMV2_IBA_EAGER_THRESHOLD        : %d\n",
                rdma_iba_eager_threshold);
        fprintf(stderr, "\tMV2_RDMA_FAST_PATH_BUF_SIZE    : %d\n",
                rdma_fp_buffer_size);
        fprintf(stderr, "\tMV2_PUT_FALLBACK_THRESHOLD     : %d\n",
                rdma_put_fallback_threshold);
        fprintf(stderr, "\tMV2_GET_FALLBACK_THRESHOLD     : %d\n",
                rdma_get_fallback_threshold);
    }
    fprintf(stderr, "\tMV2_EAGERSIZE_1SC              : %lu\n",
            rdma_eagersize_1sc);
    fprintf(stderr, "\tMV2_SMP_EAGERSIZE              : %d\n", g_smp_eagersize);
    fprintf(stderr, "\tMV2_SMP_QUEUE_LENGTH           : %zu\n",
            s_smp_queue_length);
    fprintf(stderr, "\tMV2_SMP_NUM_SEND_BUFFER        : %d\n",
            s_smp_num_send_buffer);
    fprintf(stderr, "\tMV2_SMP_BATCH_SIZE             : %d\n",
            s_smp_batch_size);
    fprintf(stderr, "\t%s                  : %s %s\n", "Tuning Table:", mv2_get_arch_name(colls_arch_hca[0].arch_type), mv2_get_hca_name(colls_arch_hca[0].hca_type));
    fprintf(stderr, "---------------------------------------------------------------------\n");
    if (mv2_show_env_info >= 2) {
        mv2_show_all_params();
        fprintf(stderr, "---------------------------------------------------------------------\n");
    }

    if (mv2_show_env_info >= 3) {
        fprintf(stderr, "\nCollective Tuning Tables\n");
        fprintf(stderr, "\t%-20s %-40s %-40s\n", "Collective", "Architecture", "Interconnect");
        for(i = 0; i < colls_max; i++) {
            fprintf(stderr, "\t%-20s %-40s %-40s\n", collective_names[i], mv2_get_arch_name(colls_arch_hca[i].arch_type), mv2_get_hca_name(colls_arch_hca[i].hca_type));
        }
        fprintf(stderr, "\n---------------------------------------------------------------------\n");
    }
}
/* vi:set sw=4 */
