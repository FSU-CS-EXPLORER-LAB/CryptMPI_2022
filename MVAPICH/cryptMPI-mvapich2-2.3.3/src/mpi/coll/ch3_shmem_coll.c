/*
 * This source file was derived from code in the MPICH-GM implementation
 * of MPI, which was developed by Myricom, Inc.
 * Myricom MPICH-GM ch_gm backend
 * Copyright (c) 2001 by Myricom, Inc.
 * All rights reserved.
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
#define _GNU_SOURCE 1

#include "mpichconf.h"
#include <mpimem.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include "upmi.h"

#include "mv2_utils.h"

#include <sched.h>

#ifdef MAC_OSX
#include <netinet/in.h>
#endif

#if defined(CHANNEL_MRAIL)
#include "rdma_impl.h"
#endif
#include "common_tuning.h"
#include "shmem_bar.h"
#include "coll_shmem.h"
#include "coll_shmem_internal.h"
#include "gather_tuning.h"
#include "bcast_tuning.h"
#include "alltoall_tuning.h"
#include "alltoallv_tuning.h"
#include "scatter_tuning.h"
#include "allreduce_tuning.h"
#include "reduce_tuning.h"
#include "allgather_tuning.h"
#include "red_scat_tuning.h"
#include "red_scat_block_tuning.h"
#include "allgatherv_tuning.h"
#include "igather_tuning.h"
#include "ibcast_tuning.h"
#include "ialltoall_tuning.h"
#include "ialltoallv_tuning.h"
#include "iscatter_tuning.h"
#include "iallreduce_tuning.h"
#include "ireduce_tuning.h"
#include "iallgather_tuning.h"
#include "ired_scat_tuning.h"
#include "iallgatherv_tuning.h"
#include "ibarrier_tuning.h"
#if defined(CKPT)
#include <cr.h>
#endif
#if defined(_ENABLE_CUDA_)
#include "datatype.h"
#endif
#include "debug_utils.h"
#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#endif

/* 
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars: 
    - name        : GATHER_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for gather operation.

    - name        : ALLGATHER_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for allgather operation.

    - name        : ALLREDUCE_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for allreduce operation.

    - name        : ALLTOALL_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for alltoall operation.

    - name        : ALLTOALLV_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for alltoallv operation.

    - name        : BCAST_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for broadcast operation.

    - name        : REDUCE_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for reduce operation.

    - name        : SCATTER_COLLECTIVE_ALGORITHM
      category    : CH3
      type        : int
      default     : -1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This CVAR selects proper collective algorithm for scatter operation.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

MPI_T_cvar_handle mv2_gather_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_allgather_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_allreduce_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_alltoall_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_alltoallv_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_bcast_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_reduce_coll_algo_handle = NULL;
MPI_T_cvar_handle mv2_scatter_coll_algo_handle = NULL;

#ifdef CKPT
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
shmem_info_t *ckpt_free_head = NULL;
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */
#endif /* CKPT */

typedef unsigned long addrint_t;

/* Shared memory collectives mgmt*/
struct shmem_coll_mgmt {
    void *mmap_ptr;
    int fd;
};
struct shmem_coll_mgmt mv2_shmem_coll_obj = { NULL, -1 };

int mv2_enable_knomial_2level_bcast = 1;
int mv2_inter_node_knomial_factor = 4;
int mv2_knomial_2level_bcast_message_size_threshold = 2048;
int mv2_knomial_2level_bcast_system_size_threshold = 64;
int mv2_enable_zcpy_bcast=1; 
int mv2_enable_zcpy_reduce=1; 
int mv2_gatherv_ssend_threshold = 8192;

int mv2_init_call_once = 0;
int mv2_mmap_coll_once = 0;
int mv2_unlink_call_once = 0;
int finalize_coll_comm = 0;

size_t  mv2_shmem_coll_size = 0;
char *mv2_shmem_coll_file = NULL;

static char mv2_hostname[SHMEM_COLL_HOSTNAME_LEN];
static int mv2_my_rank;

int mv2_g_shmem_coll_blocks = MV2_SHMEM_COLL_BLOCKS;

#if defined(_SMP_LIMIC_)
int mv2_max_limic_comms = LIMIC_COLL_NUM_COMM;
#endif                          /*#if defined(_SMP_LIMIC_) */

int mv2_g_shmem_coll_max_msg_size = MV2_SHMEM_MAX_MSG_SIZE;

int mv2_tuning_table[COLL_COUNT][COLL_SIZE] = { {2048, 1024, 512},
{-1, -1, -1},
{-1, -1, -1}
};

/* array used to tune scatter*/
int mv2_size_mv2_scatter_mv2_tuning_table = 4;
struct scatter_tuning mv2_scatter_mv2_tuning_table[] = {
    {64, 4096, 8192},
    {128, 8192, 16384},
    {256, 4096, 8192},
    {512, 4096, 8192}
};

/* array used to tune allgatherv */
int mv2_size_mv2_allgatherv_mv2_tuning_table = 4;
struct allgatherv_tuning mv2_allgatherv_mv2_tuning_table[] = {
    {64, 32768},
    {128, 65536},
    {256, 131072},
    {512, 262144}
};

int mv2_enable_shmem_collectives = 1;
int mv2_allgather_ranking = 1;
int mv2_enable_shmem_allreduce = 1;
int shmem_coll_count_threshold=16; 
int mv2_two_level_comm_early_init_threshold=2048; /* max process count for MPI_COMM_WORLD to create 2lvl comm in MPI_INIT */ 
#if defined(_MCST_SUPPORT_)
int mv2_use_mcast_allreduce = 1;
int mv2_mcast_allreduce_small_msg_size = 1024;
int mv2_mcast_allreduce_large_msg_size = 128 * 1024;
#endif                          /*   #if defined(_MCST_SUPPORT_)  */
#if defined (_SHARP_SUPPORT_)
/* max collective message that uses SHArP */
int mv2_sharp_tuned_msg_size = MV2_DEFAULT_SHARP_MAX_MSG_SIZE;
#endif
int mv2_enable_shmem_reduce = 1;
int mv2_use_knomial_reduce = 1;
int mv2_reduce_inter_knomial_factor = -1;
int mv2_reduce_zcopy_inter_knomial_factor = 4;
int mv2_reduce_intra_knomial_factor = -1;
int mv2_reduce_zcopy_max_inter_knomial_factor = 4; 
int mv2_enable_shmem_barrier = 1;
int mv2_use_two_level_gather = 1;
int mv2_use_direct_gather = 1;
int mv2_use_two_level_scatter = 1;
int mv2_use_direct_scatter = 1;
int mv2_gather_direct_system_size_small = MV2_GATHER_DIRECT_SYSTEM_SIZE_SMALL;
int mv2_gather_direct_system_size_medium = MV2_GATHER_DIRECT_SYSTEM_SIZE_MEDIUM;
int mv2_use_xor_alltoall = 1;
int mv2_enable_shmem_bcast = 1;
int mv2_use_old_bcast = 0;
int mv2_use_old_allgather = 0;
int mv2_use_old_alltoall = 0;
int mv2_alltoall_inplace_old = 0;
int mv2_use_scatter_dest_alltoallv = 0;
int mv2_use_old_scatter = 0;
int mv2_use_old_allreduce = 0;
int mv2_use_old_reduce = 0;
int mv2_scatter_rd_inter_leader_bcast = 1;
int mv2_scatter_ring_inter_leader_bcast = 1;
int mv2_knomial_inter_leader_bcast = 1;
int mv2_knomial_intra_node_threshold = 128 * 1024;
int mv2_knomial_inter_leader_threshold = 64 * 1024;
int mv2_bcast_two_level_system_size = 64;
int mv2_pipelined_knomial_factor = 2;
int mv2_pipelined_zcpy_knomial_factor = -1;
int zcpy_knomial_factor = 2;
int mv2_intra_node_knomial_factor = 4;
int mv2_shmem_coll_spin_count = 5;

int mv2_enable_socket_aware_collectives = 1;
int mv2_use_socket_aware_allreduce = 1;
int mv2_use_socket_aware_sharp_allreduce = 0;
int mv2_use_socket_aware_barrier = 1;
int mv2_socket_aware_allreduce_max_msg = 2048;
int mv2_socket_aware_allreduce_min_msg = 1;
int mv2_tune_parameter = 0;
/* Runtime threshold for scatter */
int mv2_user_scatter_small_msg = 0;
int mv2_user_scatter_medium_msg = 0;
#if defined(_MCST_SUPPORT_)
int mv2_mcast_scatter_msg_size = 32;
int mv2_mcast_scatter_small_sys_size = 32;
int mv2_mcast_scatter_large_sys_size = 2048;
int mv2_use_mcast_scatter = 1;
#endif                          /*  #if defined(_MCST_SUPPORT_) */

/*Index tuning knobs*/
int mv2_use_indexed_bcast_tuning = 1;
int mv2_use_indexed_scatter_tuning = 1;
int mv2_use_indexed_gather_tuning = 1;
int mv2_use_indexed_reduce_tuning = 1;
int mv2_use_indexed_allreduce_tuning = 1;
int mv2_use_indexed_allgather_tuning = 1;
int mv2_use_indexed_alltoall_tuning = 1;
int mv2_use_indexed_alltoallv_tuning = 1;

/* Runtime threshold for gather */
int mv2_user_gather_switch_point = 0;
int igather_segment_size = 8192;
int mv2_enable_igather = 1;
char *mv2_user_gather_intra = NULL;
char *mv2_user_igather_intra = NULL;
char *mv2_user_gather_inter = NULL;
char *mv2_user_igather_inter = NULL;
char *mv2_user_gather_intra_multi_lvl = NULL;

/* runtime flag for alltoall tuning  */
char *mv2_user_alltoall = NULL;
char *mv2_user_ialltoall_intra = NULL;
char *mv2_user_ialltoall_inter = NULL;
int ialltoall_segment_size = 8192;
int mv2_enable_ialltoall = 1;

/* runtime flag for alltoallv tuning  */
char *mv2_user_alltoallv = NULL;

/* runtime flag for alltoallv tuning  */
char *mv2_user_ialltoallv_intra = NULL;
char *mv2_user_ialltoallv_inter = NULL;
int ialltoallv_segment_size = 8192;
int mv2_enable_ialltoallv = 1;

/* Runtime threshold for bcast */
char *mv2_user_bcast_intra = NULL;
char *mv2_user_ibcast_intra = NULL;
char *mv2_user_bcast_inter = NULL;
char *mv2_user_ibcast_inter = NULL;
int mv2_enable_ibcast = 1;

/* Runtime threshold for scatter */
char *mv2_user_scatter_intra = NULL;
char *mv2_user_scatter_inter = NULL;
char *mv2_user_iscatter_intra = NULL;
char *mv2_user_iscatter_inter = NULL;
int mv2_enable_iscatter = 1;
int iscatter_segment_size = 8192;

/* Runtime threshold for allreduce */
char *mv2_user_allreduce_intra = NULL;
char *mv2_user_allreduce_inter = NULL;
int mv2_user_allreduce_two_level = 0;
char *mv2_user_iallreduce_intra = NULL;
char *mv2_user_iallreduce_inter = NULL;
int iallreduce_segment_size = 8192;
int mv2_enable_iallreduce = 1;

/* Runtime threshold for reduce */
char *mv2_user_reduce_intra = NULL;
char *mv2_user_reduce_inter = NULL;
char *mv2_user_ireduce_intra = NULL;
char *mv2_user_ireduce_inter = NULL;
int mv2_user_reduce_two_level = 0;
int ireduce_segment_size = 8192;
int mv2_enable_ireduce = 1;

/* Runtime threshold for reduce */
char *mv2_user_ireduce_scatter_intra = NULL;
char *mv2_user_ireduce_scatter_inter = NULL;
int ireduce_scatter_segment_size = 8192;
int mv2_enable_ireduce_scatter = 1;

/* Runtime threshold for allgather */
char *mv2_user_allgather_intra = NULL;
char *mv2_user_allgather_inter = NULL;
char *mv2_user_iallgather_intra = NULL;
char *mv2_user_iallgather_inter = NULL;
int mv2_user_allgather_two_level = 0;
int iallgather_segment_size = 8192;
int mv2_enable_iallgather = 1;

/* Runtime threshold for allgatherv */
char *mv2_user_iallgatherv_intra = NULL;
char *mv2_user_iallgatherv_inter = NULL;
int iallgatherv_segment_size = 8192;
int mv2_enable_iallgatherv = 1;

/* Runtime threshold for barrier */
char *mv2_user_ibarrier_intra = NULL;
char *mv2_user_ibarrier_inter = NULL;
int ibarrier_segment_size = 8192;
int mv2_enable_ibarrier = 1;
int mv2_gather_status_alignment = 16;
int mv2_bcast_status_alignment = 16;

int mv2_enable_allreduce_all_compute = 1;

/* Runtime threshold for red_scat */
char *mv2_user_red_scat_inter = NULL;

/* Runtime threshold for red_scat_block */
char *mv2_user_red_scat_block_inter = NULL;

/* Runtime threshold for allgatherv */
char *mv2_user_allgatherv_inter = NULL;

/* Runtime threshold for allgatherv */
int mv2_user_allgatherv_switch_point = 0;

int mv2_bcast_short_msg = MPIR_BCAST_SHORT_MSG;
int mv2_bcast_large_msg = MPIR_BCAST_LARGE_MSG;

/* after these threshold, force ring algorithm */
// int mv2_allreduce_red_scat_allgather_algo_threshold = 2*1024*1024;
int mv2_allreduce_red_scat_allgather_algo_ppn_threshold = 8; 
// int mv2_allreduce_ring_algo_threshold = 2*1024*1024;
// int mv2_allreduce_ring_algo_ppn_threshold = 8; 
int mv2_allgather_ring_algo_threshold = 131072;
int mv2_allgather_cyclic_algo_threshold = 1024;
int mv2_allreduce_cyclic_algo_threshold = 1024*1024;
int mv2_redscat_cyclic_algo_threshold = 1024;
// int mv2_red_scat_ring_algo_threshold = 131072;

/************ Mohsen *************/

#if 0 // Default Values
int mv2_allreduce_red_scat_allgather_algo_threshold =  2*1024*1024;  //8*1024*1024; //524288;
int mv2_allreduce_ring_algo_threshold = 2*1024*1024; //1024*1024;
int mv2_allreduce_ring_algo_ppn_threshold = 8;
int mv2_red_scat_ring_algo_threshold = 131072;
#else // Recommended setting for Concurrent Ring
int mv2_allreduce_red_scat_allgather_algo_threshold =  8*1024*1024; // 2048; //524288;
int mv2_allreduce_ring_algo_threshold = 512; // 128; //2*1024*1024; //128; //128;//2048; //1024*1024;	
int mv2_allreduce_ring_algo_ppn_threshold = 64;
int mv2_red_scat_ring_algo_threshold = 2048; //2048;//131072;
#endif	

/************ End *************/

int mv2_alltoallv_intermediate_wait_threshold = 256*1024;

int mv2_red_scat_short_msg = MPIR_RED_SCAT_SHORT_MSG;
int mv2_red_scat_long_msg = MPIR_RED_SCAT_LONG_MSG;

int mv2_bcast_scatter_ring_overlap = 1;

int mv2_bcast_scatter_ring_overlap_msg_upperbound = 1048576;
int mv2_bcast_scatter_ring_overlap_cores_lowerbound = 32;
int mv2_use_pipelined_bcast = 1;
int bcast_segment_size = 8192;
int ibcast_segment_size = 8192;

int mv2_allred_use_ring = 1;

static char *mv2_kvs_name;

int mv2_use_osu_collectives = 1;
int mv2_use_indexed_tuning = 1;
int mv2_use_osu_nb_collectives = 1;
int mv2_use_anl_collectives = 0;
int mv2_shmem_coll_num_procs = 64;

int mv2_use_bitonic_comm_split = 0;
int mv2_bitonic_comm_split_threshold = MV2_DEFAULT_BITONIC_COMM_SPLIT_THRESHOLD;

int mv2_shm_window_size = 128;
int mv2_shm_reduce_tree_degree = 4; 
int mv2_shm_slot_len = 8192;
int mv2_use_slot_shmem_coll = 1;
int mv2_use_slot_shmem_bcast = 1;
int mv2_use_mcast_pipeline_shm = 0;

#if defined(_SMP_LIMIC_)
int use_limic_gather = 0;
#endif                          /*#if defined(_SMP_LIMIC_) */
int use_2lvl_allgather = 0;

int mv2_enable_skip_tuning_table_search = 1;
int mv2_coll_skip_table_threshold = MV2_DEFAULT_COLL_SKIP_TABLE_THRESHOLD; /* msg sizes larger than this will pick an algorithm from tuning table */

struct coll_runtime mv2_coll_param = { MPIR_ALLGATHER_SHORT_MSG,
    MPIR_ALLGATHER_LONG_MSG,
    MPIR_ALLREDUCE_SHORT_MSG,
    MPIR_ALLREDUCE_2LEVEL_THRESHOLD,
    MPIR_REDUCE_SHORT_MSG,
    MPIR_REDUCE_2LEVEL_THRESHOLD,
    SHMEM_ALLREDUCE_THRESHOLD,
    SHMEM_REDUCE_THRESHOLD,
    SHMEM_INTRA_REDUCE_THRESHOLD,
    MPIR_ALLTOALL_SHORT_MSG,
    MPIR_ALLTOALL_MEDIUM_MSG,
    MPIR_ALLTOALL_THROTTLE,
    MPIR_ALLTOALL_INTRA_THROTTLE,
    MPIR_ALLTOALL_LARGE_MSG_THROTTLE,
};

#if defined(CKPT)
void *smc_store;
int smc_store_set;
#endif

#ifdef _ENABLE_CUDA_
static void *mv2_cuda_host_send_buf = NULL;
static void *mv2_cuda_host_recv_buf = NULL;
static void *mv2_cuda_dev_sr_buf = NULL;
static int mv2_cuda_host_sendbuf_size = 0;
static int mv2_cuda_host_recvbuf_size = 0;
static int mv2_cuda_dev_srbuf_size = 0;
static int *mv2_cuda_host_send_displs = NULL;
static int *mv2_cuda_host_recv_displs = NULL;
static int mv2_cuda_host_send_peers = 0;
static int mv2_cuda_host_recv_peers = 0;
static int *mv2_cuda_original_send_displs = NULL;
static int *mv2_cuda_original_recv_displs = NULL;
static void *mv2_cuda_original_send_buf = NULL;
static void *mv2_cuda_original_recv_buf = NULL;
void *mv2_cuda_allgather_store_buf = NULL;
int mv2_cuda_allgather_store_buf_size = 256 * 1024;
static void *mv2_cuda_coll_pack_buf = 0;
static int mv2_cuda_coll_pack_buf_size = 0;
static void *mv2_cuda_coll_unpack_buf = 0;
static int mv2_cuda_coll_unpack_buf_size = 0;
static void *mv2_cuda_orig_recvbuf = NULL;
static int mv2_cuda_orig_recvcount = 0;
static MPI_Datatype mv2_cuda_orig_recvtype;
#endif

/*
 * MPIT variables used by collectives
 */
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_2level_comm_requests);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_2level_comm_success);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_shmem_coll_calls);

/* Runtime parameter to enable/disable MPI_T timers */
int mv2_enable_pvar_timer = 0;

#undef FUNCNAME
#define FUNCNAME MV2_collectives_arch_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MV2_collectives_arch_init(int heterogeneity, struct coll_info *colls_arch_hca)
{
    int mpi_errno = MPI_SUCCESS;

#if defined(CHANNEL_PSM)
    /* tune the shmem size based on different architecture */
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
             MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
        /* TACC KNL */
        if (MPIDI_Process.my_pg->ch.num_local_processes <= 64) {
            mv2_g_shmem_coll_max_msg_size = 2*1024*1024;
            mv2_g_shmem_coll_blocks       = 7;
        }
    }
#endif

#if defined(_MCST_SUPPORT_)
    char *value = NULL;
    /* Disable mcast for any system other than Frontera by default */
    if (rdma_enable_mcast &&
        ((value = getenv("MV2_USE_MCAST")) == NULL) &&
#if defined(RDMA_CM)
        ((value = getenv("MV2_USE_RDMA_CM_MCAST")) == NULL) &&
#endif /*defined(RDMA_CM)*/
        !(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
          MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR))) {
        rdma_enable_mcast = 0;
#if defined(RDMA_CM)
        rdma_use_rdma_cm_mcast = 0;
#endif /*defined(RDMA_CM)*/
        MPIU_Free(mcast_ctx);
        PRINT_DEBUG(DEBUG_MCST_verbose,"Disable mcast by default on this"
                " architecture. Set MV2_USE_MCAST=1 to avoid this behavior\n");
    }
#endif

    MV2_Read_env_vars();
    
    if (mv2_use_osu_collectives) {
      MV2_set_gather_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_bcast_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_alltoall_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_alltoallv_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_scatter_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_allreduce_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_reduce_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_allgather_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_red_scat_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_red_scat_block_tuning_table(heterogeneity, colls_arch_hca);
      MV2_set_allgatherv_tuning_table(heterogeneity, colls_arch_hca);
    }
    if (mv2_use_osu_nb_collectives) {
      MV2_set_igather_tuning_table(heterogeneity);
      MV2_set_ibcast_tuning_table(heterogeneity);
      MV2_set_ialltoall_tuning_table(heterogeneity);
      MV2_set_ialltoallv_tuning_table(heterogeneity);
      MV2_set_iscatter_tuning_table(heterogeneity);
      MV2_set_iallreduce_tuning_table(heterogeneity);
      MV2_set_ireduce_tuning_table(heterogeneity);
      MV2_set_ireduce_scatter_tuning_table(heterogeneity);
      MV2_set_iallgather_tuning_table(heterogeneity);
      MV2_set_iallgatherv_tuning_table(heterogeneity);
      MV2_set_ibarrier_tuning_table(heterogeneity);
    }

    /* Functions to set collective algorithm based on MPI_T CVAR */
    if (mv2_use_osu_collectives) {
        mpi_errno = mv2_set_gather_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS ){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_allgather_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_bcast_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_reduce_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_allreduce_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_alltoall_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_alltoallv_collective_algorithm();
        if(mpi_errno != MPI_SUCCESS){
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = mv2_set_scatter_collective_algorithm();
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;

}

/* Change the values set inside the array by the one define by the user */
static int tuning_runtime_init()
{

    int i;

    /* If MV2_SCATTER_SMALL_MSG is define */
    if (mv2_user_scatter_small_msg > 0) {
        for (i = 0; i < mv2_size_mv2_scatter_mv2_tuning_table; i++) {
            mv2_scatter_mv2_tuning_table[i].small = mv2_user_scatter_small_msg;
        }
    }

    /* If MV2_SCATTER_MEDIUM_MSG is define */
    if (mv2_user_scatter_medium_msg > 0) {
        for (i = 0; i < mv2_size_mv2_scatter_mv2_tuning_table; i++) {
            if (mv2_scatter_mv2_tuning_table[i].small < mv2_user_scatter_medium_msg) {
                mv2_scatter_mv2_tuning_table[i].medium = mv2_user_scatter_medium_msg;
            }
        }
    }

    /* If MV2_GATHER_SWITCH_POINT is define and if MV2_INTRA_GATHER_TUNING && MV2_INTER_GATHER_TUNING are not define */
    if (mv2_user_gather_switch_point > 0 && mv2_user_gather_inter == NULL) {
        MV2_user_gather_switch_point_is_define(mv2_user_gather_switch_point);
    }

    /* If MV2_INTRA_GATHER_TUNING is define && MV2_INTER_GATHER_TUNING is not define */
    if (mv2_user_gather_intra != NULL && mv2_user_gather_inter == NULL) {
        MV2_intranode_Gather_is_define(mv2_user_gather_intra);
    }
#if defined(_SMP_LIMIC_)
    /* MV2_INTER_GATHER_TUNING is define with/without MV2_INTRA_GATHER_TUNING */
    if (mv2_user_gather_inter != NULL) {
        if (mv2_user_gather_intra_multi_lvl != NULL) {

            int multi_lvl_scheme = atoi(mv2_user_gather_intra_multi_lvl);
            if ((multi_lvl_scheme >= 1) && (multi_lvl_scheme <= 8)
                && (g_use_limic2_coll) && (use_limic_gather)) {
                MV2_intranode_multi_lvl_Gather_is_define(mv2_user_gather_inter,
                                                         mv2_user_gather_intra,
                                                         mv2_user_gather_intra_multi_lvl);
            } else {
                /*Since we are not using any limic schemes, we use the default
                 * scheme*/
                MV2_internode_Gather_is_define(mv2_user_gather_inter,
                                               mv2_user_gather_intra);
            }

        } else {

            MV2_internode_Gather_is_define(mv2_user_gather_inter, mv2_user_gather_intra);
        }
    }
#else                           /*#if defined(_SMP_LIMIC_) */
    /* if MV2_INTER_GATHER_TUNING is define with/without MV2_INTRA_GATHER_TUNING */
    if (mv2_user_gather_inter != NULL) {
        MV2_internode_Gather_is_define(mv2_user_gather_inter, mv2_user_gather_intra);
    }
#endif

    /* If MV2_INTRA_BCAST_TUNING is define && MV2_INTER_BCAST_TUNING is not
       define */
    if (mv2_user_bcast_intra != NULL && mv2_user_bcast_inter == NULL) {
        MV2_intranode_Bcast_is_define(mv2_user_bcast_intra);
    }

    /* if MV2_INTER_BCAST_TUNING is define with/without MV2_INTRA_BCAST_TUNING */
    if (mv2_user_bcast_inter != NULL) {
        MV2_internode_Bcast_is_define(mv2_user_bcast_inter, mv2_user_bcast_intra);
    }
    
    /* If MV2_INTRA_IBCAST_TUNING is define && MV2_INTER_BCAST_TUNING is not
       define */
    if (mv2_user_ibcast_intra != NULL && mv2_user_ibcast_inter == NULL) {
        MV2_intranode_Ibcast_is_define(mv2_user_ibcast_intra);
    }
    
    /* if MV2_INTER_IBCAST_TUNING is define with/without MV2_INTRA_IBCAST_TUNING */
    if (mv2_user_ibcast_inter != NULL) {
        MV2_internode_Ibcast_is_define(mv2_user_ibcast_inter, mv2_user_ibcast_intra);
    }
    
    /* if MV2_INTRA_IGATHER_TUNING is define && MV2_INTER_IGATHER_TUNING is not
       define */
    if (mv2_user_igather_intra != NULL && mv2_user_igather_inter == NULL) {
        MV2_intranode_Igather_is_define(mv2_user_igather_intra);
    }
    
    /* if MV2_INTER_IGATHER_TUNING is define with/without MV2_INTRA_IGATHER_TUNING */
    if (mv2_user_igather_inter != NULL) {
        MV2_internode_Igather_is_define(mv2_user_igather_inter, mv2_user_igather_intra);
    }
    
    /* if MV2_INTRA_ISCATTER_TUNING is define && MV2_INTER_ISCATTER_TUNING is not
       define */
    if (mv2_user_iscatter_intra != NULL && mv2_user_iscatter_inter == NULL) {
        MV2_intranode_Iscatter_is_define(mv2_user_iscatter_intra);
    }
    
    /* if MV2_INTER_ISCATTER_TUNING is define with/without MV2_INTRA_ISCATTER_TUNING */
    if (mv2_user_iscatter_inter != NULL) {
        MV2_internode_Iscatter_is_define(mv2_user_iscatter_inter, mv2_user_iscatter_intra);
    }
    
    /* if MV2_INTRA_IREDUCE_TUNING is define && MV2_INTER_IREDUCE_TUNING is not
       define */
    if (mv2_user_ireduce_intra != NULL && mv2_user_ireduce_inter == NULL) {
        MV2_intranode_Ireduce_is_define(mv2_user_ireduce_intra);
    }
    
    /* if MV2_INTER_IREDUCE_TUNING is define with/without MV2_INTRA_IREDUCE_TUNING */
    if (mv2_user_ireduce_inter != NULL) {
        MV2_internode_Ireduce_is_define(mv2_user_ireduce_inter, mv2_user_ireduce_intra);
    }
    
    /* if MV2_INTRA_IBARRIER_TUNING is define && MV2_INTER_IBARRIER_TUNING is not
       define */
    if (mv2_user_ibarrier_intra != NULL && mv2_user_ibarrier_inter == NULL) {
        MV2_intranode_Ibarrier_is_define(mv2_user_ibarrier_intra);
    }
    
    /* if MV2_INTER_IBARRIER_TUNING is define with/without MV2_INTRA_IBARRIER_TUNING */
    if (mv2_user_ibarrier_inter != NULL) {
        MV2_internode_Ibarrier_is_define(mv2_user_ibarrier_inter, mv2_user_ibarrier_intra);
    }
    
    /* if MV2_INTRA_IALLREDUCE_TUNING is define && MV2_INTER_IALLREDUCE_TUNING is not
       define */
    if (mv2_user_iallreduce_intra != NULL && mv2_user_iallreduce_inter == NULL) {
        MV2_intranode_Iallreduce_is_define(mv2_user_iallreduce_intra);
    }
    
    /* if MV2_INTER_IALLREDUCE_TUNING is define with/without MV2_INTRA_IALLREDUCE_TUNING */
    if (mv2_user_iallreduce_inter != NULL) {
        MV2_internode_Iallreduce_is_define(mv2_user_iallreduce_inter, mv2_user_iallreduce_intra);
    }
    
    /* if MV2_INTRA_IREDUCE_SCATTER_TUNING is define && MV2_INTER_IREDUCE_SCATTER_TUNING is not
       define */
    if (mv2_user_ireduce_scatter_intra != NULL && mv2_user_ireduce_scatter_inter == NULL) {
        MV2_intranode_Ireduce_scatter_is_define(mv2_user_ireduce_scatter_intra);
    }
    
    /* if MV2_INTER_IREDUCE_SCATTER_TUNING is define with/without MV2_INTRA_IREDUCE_SCATTER_TUNING */
    if (mv2_user_ireduce_scatter_inter != NULL) {
        MV2_internode_Ireduce_scatter_is_define(mv2_user_ireduce_scatter_inter,
						mv2_user_ireduce_scatter_intra);
    }
    
    /* if MV2_INTRA_IALLTOALL_TUNING is define && MV2_INTER_IALLTOALL_TUNING is not
       define */
    if (mv2_user_ialltoall_intra != NULL && mv2_user_ialltoall_inter == NULL) {
        MV2_intranode_Ialltoall_is_define(mv2_user_ialltoall_intra);
    }
    
    /* if MV2_INTER_IALLTOALL_TUNING is define with/without MV2_INTRA_IALLTOALL_TUNING */
    if (mv2_user_ialltoall_inter != NULL) {
        MV2_internode_Ialltoall_is_define(mv2_user_ialltoall_inter,
					  mv2_user_ialltoall_intra);
    }
    
    /* if MV2_INTRA_IALLTOALLV_TUNING is define && MV2_INTER_IALLTOALLV_TUNING is not
       define */
    if (mv2_user_ialltoallv_intra != NULL && mv2_user_ialltoallv_inter == NULL) {
        MV2_intranode_Ialltoallv_is_define(mv2_user_ialltoallv_intra);
    }
    
    /* if MV2_INTER_IALLTOALLV_TUNING is define with/without MV2_INTRA_IALLTOALLV_TUNING */
    if (mv2_user_ialltoallv_inter != NULL) {
        MV2_internode_Ialltoallv_is_define(mv2_user_ialltoallv_inter,
					  mv2_user_ialltoallv_intra);
    }
    
    /* if MV2_INTRA_IALLGATHER_TUNING is define && MV2_INTER_IALLGATHER_TUNING is not
       define */
    if (mv2_user_iallgather_intra != NULL && mv2_user_iallgather_inter == NULL) {
        MV2_intranode_Iallgather_is_define(mv2_user_iallgather_intra);
    }
    
    /* if MV2_INTER_IALLGATHER_TUNING is define with/without MV2_INTRA_IALLGATHER_TUNING */
    if (mv2_user_iallgather_inter != NULL) {
        MV2_internode_Iallgather_is_define(mv2_user_iallgather_inter,
					  mv2_user_iallgather_intra);
    }
    
    /* if MV2_INTRA_IALLGATHERV_TUNING is define && MV2_INTER_IALLGATHERV_TUNING is not
       define */
    if (mv2_user_iallgatherv_intra != NULL && mv2_user_iallgatherv_inter == NULL) {
        MV2_intranode_Iallgatherv_is_define(mv2_user_iallgatherv_intra);
    }
    
    /* if MV2_INTER_IALLGATHERV_TUNING is define with/without MV2_INTRA_IALLGATHERV_TUNING */
    if (mv2_user_iallgatherv_inter != NULL) {
        MV2_internode_Iallgatherv_is_define(mv2_user_iallgatherv_inter,
					  mv2_user_iallgatherv_intra);
    }
    
    /* If MV2_INTRA_SCATTER_TUNING is define && MV2_INTER_SCATTER_TUNING is not
       define */
    if (mv2_user_scatter_intra != NULL && mv2_user_scatter_inter == NULL) {
        MV2_intranode_Scatter_is_define(mv2_user_scatter_intra);
    }

    /* if MV2_INTER_SCATTER_TUNING is define with/without MV2_INTRA_SCATTER_TUNING */
    if (mv2_user_scatter_inter != NULL) {
        MV2_internode_Scatter_is_define(mv2_user_scatter_inter, mv2_user_scatter_intra);
    }
    
    /* If MV2_INTRA_ALLREDUCE_TUNING is define && MV2_INTER_ALLREDUCE_TUNING is not define */
    if (mv2_user_allreduce_intra != NULL && mv2_user_allreduce_inter == NULL) {
        MV2_intranode_Allreduce_is_define(mv2_user_allreduce_intra);
    }

    /* if MV2_INTER_ALLREDUCE_TUNING is define with/without MV2_INTRA_ALLREDUCE_TUNING */
    if (mv2_user_allreduce_inter != NULL) {
        MV2_internode_Allreduce_is_define(mv2_user_allreduce_inter, mv2_user_allreduce_intra);
    }

    /* If MV2_INTRA_REDUCE_TUNING is define && MV2_INTER_REDUCE_TUNING is not define */
    if (mv2_user_reduce_intra != NULL && mv2_user_reduce_inter == NULL) {
        MV2_intranode_Reduce_is_define(mv2_user_reduce_intra);
    }

    /* if MV2_INTER_REDUCE_TUNING is define with/without MV2_INTRA_REDUCE_TUNING */
    if (mv2_user_reduce_inter != NULL) {
        MV2_internode_Reduce_is_define(mv2_user_reduce_inter, mv2_user_reduce_intra);
    }

    /* if MV2_INTER_ALLGATHER_TUNING is define with/without MV2_INTRA_ALLGATHER_TUNING */
    if (mv2_user_allgather_inter != NULL) {
        MV2_internode_Allgather_is_define(mv2_user_allgather_inter);
    }

    /* if MV2_INTER_RED_SCAT_TUNING is define */
    if (mv2_user_red_scat_inter != NULL) {
        MV2_internode_Red_scat_is_define(mv2_user_red_scat_inter);
    }

    /* if MV2_INTER_RED_SCAT_BLOCK_TUNING is define */
    if (mv2_user_red_scat_block_inter != NULL) {
        MV2_internode_Red_scat_block_is_define(mv2_user_red_scat_block_inter);
    }

    /* if MV2_INTER_ALLGATHERV_TUNING is define */
    if (mv2_user_allgatherv_inter != NULL) {
        MV2_internode_Allgatherv_is_define(mv2_user_allgatherv_inter);
    }

    /* If MV2_ALLGATHERV_RD_THRESHOLD is define */
    if (mv2_user_allgatherv_switch_point > 0) {
        for (i = 0; i < mv2_size_mv2_allgatherv_mv2_tuning_table; i++) {
            mv2_allgatherv_mv2_tuning_table[i].switchp = mv2_user_allgatherv_switch_point;
        }
    }

    /* If MV2_ALLTOALL_TUNING is define  */
    if (mv2_user_alltoall != NULL ) {
        MV2_Alltoall_is_define(mv2_user_alltoall);
    }

    /* If MV2_ALLTOALLV_TUNING is define  */
    if (mv2_user_alltoallv != NULL ) {
        MV2_Alltoallv_is_define(mv2_user_alltoallv);
    }
    return 0;
}

void MV2_collectives_arch_finalize()
{
    if (mv2_use_osu_collectives) {
        MV2_cleanup_gather_tuning_table();
        MV2_cleanup_bcast_tuning_table();
        MV2_cleanup_alltoall_tuning_table();
        MV2_cleanup_alltoallv_tuning_table();
        MV2_cleanup_scatter_tuning_table();
        MV2_cleanup_allreduce_tuning_table();
        MV2_cleanup_reduce_tuning_table();
        MV2_cleanup_allgather_tuning_table();
        MV2_cleanup_red_scat_tuning_table();
        MV2_cleanup_red_scat_block_tuning_table();
        MV2_cleanup_allgatherv_tuning_table();
    }
    if (mv2_use_osu_nb_collectives) {
        MV2_cleanup_igather_tuning_table();
        MV2_cleanup_ibcast_tuning_table();
        MV2_cleanup_ialltoall_tuning_table();
        MV2_cleanup_ialltoallv_tuning_table();
        MV2_cleanup_iscatter_tuning_table();
        MV2_cleanup_iallreduce_tuning_table();
        MV2_cleanup_ireduce_tuning_table();
        MV2_cleanup_ireduce_scatter_tuning_table();
        MV2_cleanup_iallgather_tuning_table();
        MV2_cleanup_iallgatherv_tuning_table();
        MV2_cleanup_ibarrier_tuning_table();
    }
    /* Cleanup MPI_T handles if allocated */
    if (mv2_use_osu_collectives) {
        if (mv2_bcast_coll_algo_handle) {
            MPIU_Free(mv2_bcast_coll_algo_handle);
            mv2_bcast_coll_algo_handle = NULL;
        }
        if (mv2_gather_coll_algo_handle) {
            MPIU_Free(mv2_gather_coll_algo_handle);
            mv2_gather_coll_algo_handle = NULL;
        }
        if (mv2_allgather_coll_algo_handle) {
            MPIU_Free(mv2_allgather_coll_algo_handle);
            mv2_allgather_coll_algo_handle = NULL;
        }
        if (mv2_reduce_coll_algo_handle) {
            MPIU_Free(mv2_reduce_coll_algo_handle);
            mv2_reduce_coll_algo_handle = NULL;
        }
        if (mv2_allreduce_coll_algo_handle) {
            MPIU_Free(mv2_allreduce_coll_algo_handle);
            mv2_allreduce_coll_algo_handle = NULL;
        }
        if (mv2_scatter_coll_algo_handle) {
            MPIU_Free(mv2_scatter_coll_algo_handle);
            mv2_scatter_coll_algo_handle = NULL;
        }
        if (mv2_alltoall_coll_algo_handle) {
            MPIU_Free(mv2_alltoall_coll_algo_handle);
            mv2_alltoall_coll_algo_handle = NULL;
        }
        if (mv2_alltoallv_coll_algo_handle) {
            MPIU_Free(mv2_alltoallv_coll_algo_handle);
            mv2_alltoallv_coll_algo_handle = NULL;
        }
    }
}

void MPIDI_CH3I_SHMEM_COLL_Cleanup()
{
    /*unmap */
    if (mv2_shmem_coll_obj.mmap_ptr != NULL) {
        munmap(mv2_shmem_coll_obj.mmap_ptr, mv2_shmem_coll_size);
    }
    /*unlink and close */
    if (mv2_shmem_coll_obj.fd != -1) {
        close(mv2_shmem_coll_obj.fd);
        unlink(mv2_shmem_coll_file);
    }
    /*free filename variable */
    if (mv2_shmem_coll_file != NULL) {
        MPIU_Free(mv2_shmem_coll_file);
    }
    mv2_shmem_coll_obj.mmap_ptr = NULL;
    mv2_shmem_coll_obj.fd = -1;
    mv2_shmem_coll_file = NULL;
}

void MPIDI_CH3I_SHMEM_COLL_Unlink()
{
    if (mv2_unlink_call_once == 1)
        return;

    mv2_unlink_call_once = 1;
    if (mv2_init_call_once == 1 && mv2_mmap_coll_once == 1)
        finalize_coll_comm =1;

    if (mv2_shmem_coll_obj.fd != -1) {
        unlink(mv2_shmem_coll_file);
        mv2_shmem_coll_obj.mmap_ptr = NULL;
        mv2_shmem_coll_obj.fd = -1;
    }

    if (mv2_shmem_coll_file != NULL) {
        MPIU_Free(mv2_shmem_coll_file);
        mv2_shmem_coll_file = NULL;
    }
}
   
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
#undef FUNCNAME
#define FUNCNAME mv2_post_zcpy_mid_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int mv2_post_zcpy_mid_request(MPID_Comm *leader_commptr, shmem_info_t * shmem)
{
    int mpi_errno = MPI_SUCCESS;
    /* Post Ibarrier with mid-request */ 
    mpi_errno = MPIR_Ibarrier_impl(leader_commptr, &(shmem->mid_request));
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    shmem->mid_request_active = 1;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME mv2_flush_zcpy_mid_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int mv2_flush_zcpy_mid_request(shmem_info_t * shmem)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status status;

    /* Wait for previous ibarrier with mid-request to complete */
    mpi_errno = MPIR_Wait_impl(&(shmem->mid_request), &status);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    shmem->mid_request = MPI_REQUEST_NULL;
    shmem->mid_request_active = 0;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME mv2_post_zcpy_end_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int mv2_post_zcpy_end_request(MPID_Comm *leader_commptr, shmem_info_t * shmem)
{
    int mpi_errno = MPI_SUCCESS;
    /* Post Ibarrier with mid-request */ 
    mpi_errno = MPIR_Ibarrier_impl(leader_commptr, &(shmem->end_request));
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    shmem->end_request_active = 1;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME mv2_flush_zcpy_end_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int mv2_flush_zcpy_end_request(shmem_info_t * shmem)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status status; 

    /* Wait for previous ibarrier with end-request to complete */ 
    mpi_errno = MPIR_Wait_impl(&(shmem->end_request), &status);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    shmem->end_request = MPI_REQUEST_NULL;
    shmem->end_request_active = 0;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
#endif /*defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)*/

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_Helper_fn
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SHMEM_Helper_fn(MPIDI_PG_t * pg, int local_id, char **filename,
                                char *prefix, int *fd, size_t file_size)
{
    int mpi_errno = MPI_SUCCESS;
    size_t pathlen = 0;
    char *shmem_dir = NULL, *shmdir = NULL;
    struct stat file_status;
#if defined(SOLARIS)
    char *setdir = "/tmp";
#else
    char *setdir = "/dev/shm";
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_HELPER_FN);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_HELPER_FN);

    PRINT_DEBUG(DEBUG_SHM_verbose>0, "Set up shmem for %s\n", prefix);

    /* Obtain prefix to create shared file name */
    if ((shmdir = getenv("MV2_SHMEM_DIR")) != NULL) {
        shmem_dir = shmdir;
    } else {
        shmem_dir = setdir;
    }
    /*Find length of prefix */
    pathlen = strlen(shmem_dir);

    /* Get hostname to create unique shared file name */
    if (gethostname(mv2_hostname, sizeof (char) * HOSTNAME_LEN) < 0) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
                                  "gethostname", strerror(errno));
    }
 
    /* Get KVS name to create unique shared file name */
    MPIDI_PG_GetConnKVSname(&mv2_kvs_name);

    /* Allocate memory for unique file name */
    *filename = (char *) MPIU_Malloc(sizeof (char) *
                            (pathlen + HOSTNAME_LEN + 26 + PID_CHAR_LEN));
    if (!*filename) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s", "mv2_shmem_coll_file");
    }

    /* Create unique shared file name */
    sprintf(*filename, "%s/%s-%s-%s-%d.tmp",
            shmem_dir, prefix, mv2_kvs_name, mv2_hostname, getuid());

    /* Open the shared memory file */
    *fd = open(*filename, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
    if (*fd < 0) {
        /* Fallback */
        sprintf(*filename, "/tmp/%s-%s-%s-%d.tmp",
                prefix, mv2_kvs_name, mv2_hostname, getuid());

        *fd = open(*filename, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
        if (*fd < 0) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER,
                                      "**fail", "%s: %s", "open", strerror(errno));
        }
    }
    PRINT_DEBUG(DEBUG_SHM_verbose>0, "Shmem file opened for %s\n", prefix);

    if (local_id == 0) {
        PRINT_DEBUG(DEBUG_SHM_verbose>0, "Set file size for %s\n", prefix);
        if (ftruncate(*fd, 0)) {
            /* to clean up tmp shared file */
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "ftruncate", strerror(errno));
            goto fn_fail;
        }

        /* set file size, without touching pages */
        if (ftruncate(*fd, file_size)) {
            /* to clean up tmp shared file */
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "ftruncate", strerror(errno));
            goto fn_fail;
        }

        PRINT_DEBUG(DEBUG_SHM_verbose>0, "Call MPIU_Calloc for %s\n", prefix);
/* Ignoring optimal memory allocation for now */
#if !defined(__x86_64__)

#define FSIZE_LIMIT 2147483640 /* 2G - c */
        {
            char *buf = (char *) MPIU_Calloc(file_size + 1, sizeof (char));
            size_t remaining_bytes, transfer_size;
            size_t transferred = 0, offset = 0;
 
            /* if filesize exceeds 2G limit, then split it up in multiple chunks
            * and then do the write iteratively. 
            */
            remaining_bytes = file_size;
            while (remaining_bytes > 0) {
                transfer_size = (remaining_bytes > FSIZE_LIMIT) ? FSIZE_LIMIT : remaining_bytes;
                write(*fd, buf+offset, transfer_size);
                remaining_bytes -= transfer_size;
                transferred += transfer_size;
 
                offset += transfer_size;
            }

            /* if total bytes transferred doesn't match filesize, throw error */
            if (transferred != file_size) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**fail", "%s: %s", "write",
                                                 strerror(errno));
                MPIU_Free(buf);
                goto fn_fail;
            }
            MPIU_Free(buf);
        }
#endif  /* !defined(__x86_64__) */

        PRINT_DEBUG(DEBUG_SHM_verbose>0, "Call lseek for %s\n", prefix);
        if (lseek(*fd, 0, SEEK_SET) != 0) {
            /* to clean up tmp shared file */
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "lseek", strerror(errno));
            goto fn_fail;
        }
    }

    PRINT_DEBUG(DEBUG_SHM_verbose>0, "Wait for local sync for %s\n", prefix);
    /* Synchronization between local processes */
    do {
       if (fstat(*fd, &file_status) != 0) {
           /* to clean up tmp shared file */
           mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                   "%s: %s", "fstat",
                   strerror(errno));
           goto fn_fail;
       }
       usleep(1);
    } while (file_status.st_size != file_size);

    PRINT_DEBUG(DEBUG_SHM_verbose>0, "Finished set up shmem for %s\n", prefix);

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_HELPER_FN);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SHMEM_COLL_init(MPIDI_PG_t * pg, int local_id)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_INIT);
    int mpi_errno = MPI_SUCCESS;
    char *value;

    if (mv2_init_call_once == 1)
        return mpi_errno;

    mv2_init_call_once = 1;

    mv2_shmem_coll_num_procs = pg->ch.num_local_processes;
    if ((value = getenv("MV2_SHMEM_COLL_NUM_PROCS")) != NULL) {
        mv2_shmem_coll_num_procs = (int) atoi(value);
    }

    if ((value = getenv("MV2_ENABLE_ALLREDUCE_ALL_COMPUTE")) != NULL) {
        mv2_enable_allreduce_all_compute = atoi(value);
    }

    if ((value = getenv("MV2_GATHER_STATUS_ALIGNMENT")) != NULL) {
        mv2_gather_status_alignment = atoi(value);
    }

    if ((value = getenv("MV2_BCAST_STATUS_ALIGNMENT")) != NULL) {
        mv2_bcast_status_alignment = atoi(value);
    }

    /* Find out the size of the region to create */
    mv2_shmem_coll_size = SHMEM_ALIGN(SHMEM_COLL_BUF_SIZE + getpagesize())
                                + SHMEM_CACHE_LINE_SIZE;

    /* Call helper function to create shmem region */
    mpi_errno = MPIDI_CH3I_SHMEM_Helper_fn(pg, local_id, &mv2_shmem_coll_file,
                                "ib_shmem_coll", &mv2_shmem_coll_obj.fd,
                                mv2_shmem_coll_size);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (mv2_tune_parameter == 1) {
        tuning_runtime_init();
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_INIT);
    return mpi_errno;

  fn_fail:
    MPIDI_CH3I_SHMEM_COLL_Cleanup();
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_Mmap
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SHMEM_COLL_Mmap(MPIDI_PG_t * pg, int local_id)
{
    int i = 0;
    int j = 0;
    char *buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int num_cntrl_bufs=5; 
    MPIDI_VC_t *vc = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLLMMAP);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLLMMAP);

    if (mv2_mmap_coll_once)
        return mpi_errno;

    mv2_mmap_coll_once = 1;

    MPIDI_PG_Get_vc(pg, mv2_my_rank, &vc);

    mv2_shmem_coll_obj.mmap_ptr = mmap(0, mv2_shmem_coll_size,
                                       (PROT_READ | PROT_WRITE), (MAP_SHARED),
                                       mv2_shmem_coll_obj.fd, 0);
    if (mv2_shmem_coll_obj.mmap_ptr == (void *) -1) {
        /* to clean up tmp shared file */
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                         FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                         "%s: %s", "mmap", strerror(errno));
        goto cleanup_files;
    }
#if defined(CKPT)
    if (smc_store_set) {
        MPIU_Memcpy(mv2_shmem_coll_obj.mmap_ptr, smc_store, mv2_shmem_coll_size);
        MPIU_Free(smc_store);
        smc_store_set = 0;
    }
#endif

    shmem_coll = (shmem_coll_region *) mv2_shmem_coll_obj.mmap_ptr;

    /* layout the shared memory for variable length vars */
    buf =
        &shmem_coll->shmem_coll_buf +
        (mv2_g_shmem_coll_blocks * 2 * SHMEM_COLL_BLOCK_SIZE);

    if (local_id == 0) {
#if defined(_SMP_LIMIC_)
        num_cntrl_bufs = 6;
#endif
        MPIU_Memset(buf, 0, (num_cntrl_bufs-1)*SHMEM_COLL_SYNC_ARRAY_SIZE + 
                SHMEM_BCAST_SYNC_ARRAY_SIZE);
    }  

    shmem_coll_block_status = (volatile int *) buf;
    buf += SHMEM_COLL_STATUS_ARRAY_SIZE;
    child_complete_bcast = (volatile int *) buf;
    buf += SHMEM_BCAST_SYNC_ARRAY_SIZE;
    child_complete_gather = (volatile int *) buf;
    buf += SHMEM_COLL_SYNC_ARRAY_SIZE;
    root_complete_gather = (volatile int *) buf;
    buf += SHMEM_COLL_SYNC_ARRAY_SIZE;
    barrier_gather = (volatile int *) buf;
    buf += SHMEM_COLL_SYNC_ARRAY_SIZE;
    barrier_bcast = (volatile int *) buf;
#if defined(_SMP_LIMIC_)
    buf += SHMEM_COLL_SYNC_ARRAY_SIZE;
    limic_progress = (volatile int *) buf;
#endif

    if (local_id == 0) {
        for (j = 0; j < mv2_g_shmem_coll_blocks; ++j) {
            for (i = 0; i < mv2_shmem_coll_num_procs; ++i) {
                SHMEM_BCAST_SYNC_CLR(child_complete_bcast, j, i);
                WRITEBAR();
            }

            for (i = 0; i < mv2_shmem_coll_num_procs; ++i) {
                SHMEM_COLL_SYNC_SET(root_complete_gather, j, i);
                WRITEBAR();
            }

        }
        for (j = 0; j < mv2_g_shmem_coll_blocks; j++) {
            SHMEM_COLL_BLOCK_STATUS_CLR(shmem_coll_block_status, j);
            WRITEBAR();
        }

#if defined(_SMP_LIMIC_)
        for (j = 0; j < mv2_max_limic_comms; ++j) {
            for (i = 0; i < mv2_shmem_coll_num_procs; ++i) {
                SHMEM_COLL_SYNC_CLR(limic_progress, j, i);
                WRITEBAR();
            }
        }
        memset(shmem_coll->limic_hndl, 0, (sizeof (limic_user) * LIMIC_COLL_NUM_COMM));

#endif                          /*#if defined(_SMP_LIMIC_) */
        pthread_spin_init(&shmem_coll->shmem_coll_lock, PTHREAD_PROCESS_SHARED);
        shmem_coll->mv2_shmem_comm_count = 0;
#if defined(CKPT)
        /*
         * FIXME: The second argument to pthread_spin_init() indicates whether the
         * Lock can be accessed by a process other than the one that initialized
         * it. So, it should actually be PTHREAD_PROCESS_SHARED. However, the
         * "shmem_coll_lock" above sets this to 0. Hence, I am doing the same.
         */
        pthread_spin_init(&shmem_coll->cr_smc_spinlock, PTHREAD_PROCESS_SHARED);
        shmem_coll->cr_smc_cnt = 0;
#endif
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLLMMAP);
    return mpi_errno;

  cleanup_files:
    MPIDI_CH3I_SHMEM_COLL_Cleanup();
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SHMEM_COLL_finalize(int local_id, int num_local_nodes)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_FINALIZE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_FINALIZE);

#if defined(CKPT)

    extern int g_cr_in_progress;

    if (g_cr_in_progress) {
        /* Wait for other local processes to check-in */
        pthread_spin_lock(&shmem_coll->cr_smc_spinlock);
        ++(shmem_coll->cr_smc_cnt);
        pthread_spin_unlock(&shmem_coll->cr_smc_spinlock);
        while (shmem_coll->cr_smc_cnt < num_local_nodes) ;

        if ((mv2_shmem_coll_obj.mmap_ptr != NULL) && (local_id == 0)) {
            smc_store = MPIU_Malloc(mv2_shmem_coll_size);
            PRINT_DEBUG(DEBUG_CR_verbose > 1, "mv2_shmem_coll_obj.mmap_ptr = %p\n", mv2_shmem_coll_obj.mmap_ptr);
            MPIU_Memcpy(smc_store, mv2_shmem_coll_obj.mmap_ptr, mv2_shmem_coll_size);
            smc_store_set = 1;
        }
    }
#endif

    MPIDI_CH3I_SHMEM_COLL_Cleanup();

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_FINALIZE);
    return MPI_SUCCESS;
}

int MPIDI_CH3I_SHMEM_Coll_get_free_block()
{
    int i = 0;
    while (i < mv2_g_shmem_coll_blocks) {
        READBAR();
        if (SHMEM_COLL_BLOCK_STATUS_INUSE(shmem_coll_block_status, i)) {
            i++;
        } else {
            break;
        }
    }

    if (i < mv2_g_shmem_coll_blocks) {
        SHMEM_COLL_BLOCK_STATUS_SET(shmem_coll_block_status, i);
        WRITEBAR();
        return i;
    } else {
        return -1;
    }

}

void MPIDI_CH3I_SHMEM_Coll_Block_Clear_Status(int block_id)
{
    SHMEM_COLL_BLOCK_STATUS_CLR(shmem_coll_block_status, block_id);
    WRITEBAR();
}

/* Shared memory gather: rank zero is the root always*/
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_GetShmemBuf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(int size, int rank, int shmem_comm_rank,
                                       void **output_buf)
{
    int i = 1, cnt = 0, err = 0;
    char *shmem_coll_buf = (char *) (&(shmem_coll->shmem_coll_buf));
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_GETSHMEMBUF);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_GETSHMEMBUF);

    READBAR();
    if (rank == 0) {
        for (i = 1; i < size; ++i) {
            READBAR();
            while (SHMEM_COLL_SYNC_ISCLR(child_complete_gather, shmem_comm_rank, i)) {
#if defined(CKPT)
                Wait_for_CR_Completion();
#endif
                MV2_INC_NUM_UNEXP_RECV();
                MPID_Progress_test();
                MV2_DEC_NUM_UNEXP_RECV();
                /* Yield once in a while */
                MPIU_THREAD_CHECK_BEGIN++ cnt;
                if (cnt >= mv2_shmem_coll_spin_count) {
                    cnt = 0;
#if defined(CKPT)
                    MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
                        do {
                    } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                        MPIDI_CH3I_CR_lock();
#endif
                }
                MPIU_THREAD_CHECK_END
                READBAR();
            }
        }
        /* Set the completion flags back to zero */
        for (i = 1; i < size; ++i) {
            SHMEM_COLL_SYNC_CLR(child_complete_gather, shmem_comm_rank, i);
            WRITEBAR();
        }

        *output_buf = (char *) shmem_coll_buf + shmem_comm_rank * SHMEM_COLL_BLOCK_SIZE;
    } else {
        READBAR();
        while (SHMEM_COLL_SYNC_ISCLR(root_complete_gather, shmem_comm_rank, rank)) {
#if defined(CKPT)
            Wait_for_CR_Completion();
#endif
            MV2_INC_NUM_UNEXP_RECV();
            MPID_Progress_test();
            MV2_DEC_NUM_UNEXP_RECV();
            /* Yield once in a while */
            MPIU_THREAD_CHECK_BEGIN++ cnt;
            if (cnt >= mv2_shmem_coll_spin_count) {
                cnt = 0;
#if defined(CKPT)
                MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
                    do {
                } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                    MPIDI_CH3I_CR_lock();
#endif
            }
            MPIU_THREAD_CHECK_END
            READBAR();
        }

        SHMEM_COLL_SYNC_CLR(root_complete_gather, shmem_comm_rank, rank);
        WRITEBAR();
        *output_buf = (char *) shmem_coll_buf + shmem_comm_rank * SHMEM_COLL_BLOCK_SIZE;
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_GETSHMEMBUF);
}

/* Shared memory bcast: rank zero is the root always*/
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_Bcast_GetBuf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_Bcast_GetBuf(int size, int rank,
                                   int shmem_comm_rank, void **output_buf)
{
    int i = 1, cnt = 0, err = 0;
    char *shmem_coll_buf = (char *) (&(shmem_coll->shmem_coll_buf) +
                                     mv2_g_shmem_coll_blocks * SHMEM_COLL_BLOCK_SIZE);
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_BCAST_GETBUF);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_BCAST_GETBUF);

    READBAR();
    if (rank == 0) {
        for (i = 1; i < size; ++i) {
            READBAR();
            while (SHMEM_BCAST_SYNC_ISSET(child_complete_bcast, shmem_comm_rank, i)) {
#if defined(CKPT)
                Wait_for_CR_Completion();
#endif
                MV2_INC_NUM_UNEXP_RECV();
                MPID_Progress_test();
                MV2_DEC_NUM_UNEXP_RECV();
                /* Yield once in a while */
                MPIU_THREAD_CHECK_BEGIN++ cnt;
                if (cnt >= mv2_shmem_coll_spin_count) {
                    cnt = 0;
#if defined(CKPT)
                    MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
                        do {
                    } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                        MPIDI_CH3I_CR_lock();
#endif
                }
                MPIU_THREAD_CHECK_END
                READBAR();
            }
        }
        *output_buf = (char *) shmem_coll_buf + shmem_comm_rank * SHMEM_COLL_BLOCK_SIZE;
    } else {
        READBAR();
        while (SHMEM_BCAST_SYNC_ISCLR(child_complete_bcast, shmem_comm_rank, rank)) {
#if defined(CKPT)
            Wait_for_CR_Completion();
#endif
            MV2_INC_NUM_UNEXP_RECV();
            MPID_Progress_test();
            MV2_DEC_NUM_UNEXP_RECV();
            /* Yield once in a while */
            MPIU_THREAD_CHECK_BEGIN++ cnt;
            if (cnt >= mv2_shmem_coll_spin_count) {
                cnt = 0;
#if defined(CKPT)
                MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
                    do {
                } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                    MPIDI_CH3I_CR_lock();
#endif
            }
            MPIU_THREAD_CHECK_END
            READBAR();
        }
        *output_buf = (char *) shmem_coll_buf + shmem_comm_rank * SHMEM_COLL_BLOCK_SIZE;
    }
    READBAR();
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_BCAST_GETBUF);
}

/* Shared memory bcast: rank zero is the root always*/
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_Bcast_Complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_Bcast_Complete(int size, int rank, int shmem_comm_rank)
{
    int i = 1;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETBCASTCOMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETBCASTCOMPLETE);

    READBAR();
    if (rank == 0) {
        for (i = 1; i < size; ++i) {
            SHMEM_BCAST_SYNC_SET(child_complete_bcast, shmem_comm_rank, i);
            WRITEBAR();
        }
    } else {
        SHMEM_BCAST_SYNC_CLR(child_complete_bcast, shmem_comm_rank, rank);
        WRITEBAR();
    }
    READBAR();
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETBCASTCOMPLETE);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_SetGatherComplete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(int size, int rank, int shmem_comm_rank)
{
    int i = 1;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETGATHERCOMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETGATHERCOMPLETE);

    READBAR();
    if (rank == 0) {
        for (; i < size; ++i) {
            SHMEM_COLL_SYNC_SET(root_complete_gather, shmem_comm_rank, i);
            WRITEBAR();
        }
    } else {
        SHMEM_COLL_SYNC_SET(child_complete_gather, shmem_comm_rank, rank);
        WRITEBAR();
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_SETGATHERCOMPLETE);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_Barrier_gather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_COLL_Barrier_gather(int size, int rank, int shmem_comm_rank)
{
    int i = 1, cnt = 0, err = 0;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_GATHER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_GATHER);

    READBAR();
    if (rank == 0) {
        for (i = 1; i < size; ++i) {
            READBAR();
            while (SHMEM_COLL_SYNC_ISCLR(barrier_gather, shmem_comm_rank, i)) {
#if defined(CKPT)
                Wait_for_CR_Completion();
#endif
                MV2_INC_NUM_UNEXP_RECV();
                MPID_Progress_test();
                MV2_DEC_NUM_UNEXP_RECV();
                /* Yield once in a while */
                MPIU_THREAD_CHECK_BEGIN++ cnt;
                if (cnt >= mv2_shmem_coll_spin_count) {
                    cnt = 0;
#if defined(CKPT)
                    MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
                        do {
                    } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                        MPIDI_CH3I_CR_lock();
#endif
                }
                MPIU_THREAD_CHECK_END
                READBAR();
            }
        }
        for (i = 1; i < size; ++i) {
            SHMEM_COLL_SYNC_CLR(barrier_gather, shmem_comm_rank, i);
            WRITEBAR();
        }
    } else {
        SHMEM_COLL_SYNC_SET(barrier_gather, shmem_comm_rank, rank);
        WRITEBAR();
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_GATHER);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHMEM_COLL_Barrier_bcast
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SHMEM_COLL_Barrier_bcast(int size, int rank, int shmem_comm_rank)
{
    int i = 1, cnt = 0, err = 0;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_BCAST);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_BCAST);

    READBAR();
    if (rank == 0) {
        for (i = 1; i < size; ++i) {
            SHMEM_COLL_SYNC_SET(barrier_bcast, shmem_comm_rank, i);
            WRITEBAR();
        }
    } else {
        READBAR();
        while (SHMEM_COLL_SYNC_ISCLR(barrier_bcast, shmem_comm_rank, rank)) {
#if defined(CKPT)
            Wait_for_CR_Completion();
#endif
            MV2_INC_NUM_UNEXP_RECV();
            MPID_Progress_test();
            MV2_DEC_NUM_UNEXP_RECV();
            /* Yield once in a while */
            MPIU_THREAD_CHECK_BEGIN++ cnt;
            if (cnt >= mv2_shmem_coll_spin_count) {
                cnt = 0;
#if defined(CKPT)
                MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
                    do {
                } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                    MPIDI_CH3I_CR_lock();
#endif
            }
            MPIU_THREAD_CHECK_END
            READBAR();
        }
        SHMEM_COLL_SYNC_CLR(barrier_bcast, shmem_comm_rank, rank);
        WRITEBAR();
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHMEM_COLL_BARRIER_BCAST);
}

void lock_shmem_region()
{
    pthread_spin_lock(&shmem_coll->shmem_coll_lock);
}

void unlock_shmem_region()
{
    pthread_spin_unlock(&shmem_coll->shmem_coll_lock);
}

void increment_mv2_shmem_comm_count()
{
    ++shmem_coll->mv2_shmem_comm_count;
}

int get_mv2_shmem_comm_count()
{
    return shmem_coll->mv2_shmem_comm_count;
}

int is_shmem_collectives_enabled()
{
    return mv2_enable_shmem_collectives;
}

int mv2_increment_shmem_coll_counter(MPID_Comm *comm_ptr)
{
   int mpi_errno = MPI_SUCCESS, flag=0; 
   PMPI_Comm_test_inter(comm_ptr->handle, &flag);

   if(flag == 0 && mv2_enable_shmem_collectives
      && comm_ptr->dev.ch.shmem_coll_ok == 0
      && check_split_comm(pthread_self())) { 
        comm_ptr->dev.ch.shmem_coll_count++; 

        if(comm_ptr->dev.ch.shmem_coll_count >= shmem_coll_count_threshold) { 
            disable_split_comm(pthread_self());
            mpi_errno = create_2level_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
            enable_split_comm(pthread_self());
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
        } 
   } 

fn_exit:
  return mpi_errno;    

fn_fail:
  goto fn_exit; 
} 

int mv2_increment_allgather_coll_counter(MPID_Comm *comm_ptr)
{   
   MPIR_Errflag_t errflag = MPIR_ERR_NONE;
   int mpi_errno = MPI_SUCCESS, flag=0;
   PMPI_Comm_test_inter(comm_ptr->handle, &flag);

   if(flag == 0 
      && mv2_allgather_ranking 
      && mv2_enable_shmem_collectives
      && check_split_comm(pthread_self())) {
        comm_ptr->dev.ch.allgather_coll_count++;

        if(comm_ptr->dev.ch.allgather_coll_count >= shmem_coll_count_threshold) {
            disable_split_comm(pthread_self());
            if(comm_ptr->dev.ch.shmem_coll_ok == 0) { 
                /* If comm_ptr does not have leader/shmem sub-communicators,
                 * create them now */ 
                mpi_errno = create_2level_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
                if(mpi_errno) {
                   MPIR_ERR_POP(mpi_errno);
                }
            } 

            if(comm_ptr->dev.ch.shmem_coll_ok == 1) { 
                /* Before we go ahead with allgather-comm creation, be sure that 
                 * the sub-communicators are actually ready */ 
                mpi_errno = create_allgather_comm(comm_ptr, &errflag);
                if(mpi_errno) {
                   MPIR_ERR_POP(mpi_errno);
                }
            } 
            enable_split_comm(pthread_self());
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
        }
   }

fn_exit:
  return mpi_errno;

fn_fail:
  goto fn_exit;
}



#if defined(_SMP_LIMIC_)
void increment_mv2_limic_comm_count()
{
    ++shmem_coll->mv2_limic_comm_count;
}

int get_mv2_limic_comm_count()
{
    return shmem_coll->mv2_limic_comm_count;
}

void UpdateNumCoresPerSock(int numcores)
{
    shmem_coll->limic_coll_numCores_perSock = numcores;
}

void UpdateNumSocketsPerNode(int numSocketsNode)
{
    shmem_coll->limic_coll_num_sockets = numSocketsNode;
}
#endif                          /*#if defined(_SMP_LIMIC_) */
void MV2_Read_env_vars(void)
{
    char *value;
    int flag;

    /************ Mohsen *************/
    
    char *tc = getenv("MV2_ALLGATHER_PRINT_FUN_NAME");
	if (tc && (strncmp(tc, "1",1) == 0))	ALLGATHER_PRINT_FUN_NAME = 1;

    tc = getenv("MV2_ALLREDUCE_PRINT_FUN_NAME");
	if (tc && (strncmp(tc, "1",1) == 0))	PRINT_FUN_NAME = 1;
	
	tc = getenv("MV2_ENABLE_SECURE_DEBUG");
	if (tc && (strncmp(tc, "1",1) == 0))	ENABLE_SECURE_DEBUG = 1;
	
	tc = getenv("MV2_SHOW_FINAL_RESULT");
	if (tc && (strncmp(tc, "1",1) == 0))	SHOW_FINAL_RESULT = 1;	
		
	tc = getenv("MV2_PRINT_Ring_FUN_NAME");
	if (tc && (strncmp(tc, "1",1) == 0))	PRINT_Ring_FUN_NAME = 1;
	
	tc = getenv("MV2_PRINT_RSA_FUN_NAME");
	if (tc && (strncmp(tc, "1",1) == 0))	PRINT_RSA_FUN_NAME = 1;
	
	tc = getenv("MV2_ALLREDUCE_PRINT_SUPER");
	if (tc && (strncmp(tc, "1",1) == 0))	PRINT_SUPER = 1;
	
	
	/* *********** End *********** */

    if ((value = getenv("MV2_USE_BITONIC_COMM_SPLIT")) != NULL) {
        mv2_use_bitonic_comm_split = !!atoi(value);
    }

    if (mv2_use_bitonic_comm_split) {
        if ((value = getenv("MV2_BITONIC_COMM_SPLIT_THRESHOLD")) != NULL) {
            mv2_bitonic_comm_split_threshold = atoi(value);
            if (mv2_bitonic_comm_split_threshold < 0) {
                mv2_bitonic_comm_split_threshold = MV2_DEFAULT_BITONIC_COMM_SPLIT_THRESHOLD;
            }
        }
    }

    if ((value = getenv("MV2_ENABLE_SKIP_TUNING_TABLE_SEARCH")) != NULL) {
        mv2_enable_skip_tuning_table_search = !!atoi(value);
    }
    if ((value = getenv("MV2_COLL_SKIP_TABLE_THRESHOLD")) != NULL) {
        mv2_coll_skip_table_threshold = atoi(value);
        if (mv2_coll_skip_table_threshold < 0) {
            mv2_enable_skip_tuning_table_search = MV2_DEFAULT_COLL_SKIP_TABLE_THRESHOLD;
        }
    }

    if ((value = getenv("MV2_USE_OSU_COLLECTIVES")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_osu_collectives = 1;
        } else {
            mv2_use_osu_collectives = 0;
            mv2_use_anl_collectives = 1;
        }
    }
    
    if ((value = getenv("MV2_USE_OSU_NB_COLLECTIVES")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_osu_nb_collectives = 1;
        } else {
            mv2_use_osu_nb_collectives = 0;
            mv2_use_anl_collectives = 1;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_tuning = 1;
        } else {
            mv2_use_indexed_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_BCAST_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_bcast_tuning = 1;
        } else {
            mv2_use_indexed_bcast_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_SCATTER_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_scatter_tuning = 1;
        } else {
            mv2_use_indexed_scatter_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_GATHER_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_gather_tuning = 1;
        } else {
            mv2_use_indexed_gather_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_REDUCE_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_reduce_tuning = 1;
        } else {
            mv2_use_indexed_reduce_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_ALLREDUCE_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_allreduce_tuning = 1;
        } else {
            mv2_use_indexed_allreduce_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_ALLGATHER_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_allgather_tuning = 1;
        } else {
            mv2_use_indexed_allgather_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_USE_INDEXED_ALLTOALL_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_alltoall_tuning = 1;
        } else {
            mv2_use_indexed_alltoall_tuning = 0;
        }
    }

    if ((value = getenv("MV2_USE_INDEXED_ALLTOALLV_TUNING")) != NULL) {
        if (atoi(value) == 1) {
            mv2_use_indexed_alltoallv_tuning = 1;
        } else {
            mv2_use_indexed_alltoallv_tuning = 0;
        }
    }
    
    if ((value = getenv("MV2_IBCAST_ENABLE")) != NULL) {
        mv2_enable_ibcast = atoi(value);
    }
    if ((value = getenv("MV2_IGATHER_ENABLE")) != NULL) {
        mv2_enable_igather = atoi(value);
    }
    if ((value = getenv("MV2_ISCATTER_ENABLE")) != NULL) {
        mv2_enable_iscatter = atoi(value);
    }
    if ((value = getenv("MV2_IALLGATHER_ENABLE")) != NULL) {
        mv2_enable_iallgather = atoi(value);
    }
    if ((value = getenv("MV2_IREDUCE_ENABLE")) != NULL) {
        mv2_enable_ireduce = atoi(value);
    }
    if ((value = getenv("MV2_IALLREDUCE_ENABLE")) != NULL) {
        mv2_enable_iallreduce = atoi(value);
    }
    if ((value = getenv("MV2_IREDUCE_SCATTER_ENABLE")) != NULL) {
        mv2_enable_ireduce_scatter = atoi(value);
    }
    if ((value = getenv("MV2_IALLTOALL_ENABLE")) != NULL) {
        mv2_enable_ialltoall = atoi(value);
    }
    if ((value = getenv("MV2_IALLTOALLV_ENABLE")) != NULL) {
        mv2_enable_ialltoallv = atoi(value);
    }
    if ((value = getenv("MV2_IALLGATHERV_ENABLE")) != NULL) {
        mv2_enable_iallgatherv = atoi(value);
    }
    if ((value = getenv("MV2_IBARRIER_ENABLE")) != NULL) {
        mv2_enable_ibarrier = atoi(value);
    }
    
    if ((value = getenv("MV2_USE_SHMEM_COLL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_enable_shmem_collectives = 1;
        else
            mv2_enable_shmem_collectives = 0;
    }
    if ((value = getenv("MV2_USE_SHMEM_ALLREDUCE")) != NULL) {
        mv2_enable_shmem_allreduce = !!atoi(value);
    }
#if defined(_MCST_SUPPORT_)
    if ((value = getenv("MV2_USE_MCAST_ALLREDUCE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_mcast_allreduce = 1;
        else
            mv2_use_mcast_allreduce = 0;
    }
    if ((value = getenv("MV2_USE_MCAST_ALLREDUCE_SMALL_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_mcast_allreduce_small_msg_size = flag;
    }
    if ((value = getenv("MV2_USE_MCAST_ALLREDUCE_LARGE_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_mcast_allreduce_large_msg_size = flag;
    }
#endif                          /* #if defined(_MCST_SUPPORT_) */
#if defined (_SHARP_SUPPORT_)
    if ((value = getenv("MV2_SHARP_MAX_MSG_SIZE")) != NULL) {
        /* Force SHArP for the message range that user specified */
        mv2_sharp_tuned_msg_size = atoi(value);
        if (mv2_sharp_tuned_msg_size < 0) {
            mv2_sharp_tuned_msg_size = MV2_DEFAULT_SHARP_MAX_MSG_SIZE;
        }
    } 
#endif 
    if ((value = getenv("MV2_USE_SHMEM_REDUCE")) != NULL) {
        mv2_enable_shmem_reduce = !!atoi(value);
    }
    if ((value = getenv("MV2_USE_KNOMIAL_REDUCE")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_use_knomial_reduce = flag;
    }
    if ((value = getenv("MV2_USE_INTER_KNOMIAL_REDUCE_FACTOR")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_reduce_inter_knomial_factor = flag;
    }
    if ((value = getenv("MV2_USE_ZCOPY_INTER_KNOMIAL_REDUCE_FACTOR")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0) { 
            mv2_reduce_zcopy_inter_knomial_factor = flag;
            mv2_reduce_zcopy_max_inter_knomial_factor = flag;
        } 
    }
    if ((value = getenv("MV2_USE_INTRA_KNOMIAL_REDUCE_FACTOR")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_reduce_intra_knomial_factor = flag;
    }
    if ((value = getenv("MV2_USE_SHMEM_BARRIER")) != NULL) {
        mv2_enable_shmem_barrier = !!atoi(value);
    }
    if ((value = getenv("MV2_SHMEM_COLL_NUM_COMM")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_g_shmem_coll_blocks = flag;
    }
    if ((value = getenv("MV2_SHMEM_COLL_MAX_MSG_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_g_shmem_coll_max_msg_size = flag;
    }
    if ((value = getenv("MV2_USE_SHARED_MEM")) != NULL) {
        flag = (int) atoi(value);
        if (flag <= 0)
            mv2_enable_shmem_collectives = 0;
    }
    if ((value = getenv("MV2_USE_BLOCKING")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_enable_shmem_collectives = 0;
    }
    if ((value = getenv("MV2_ALLREDUCE_SHORT_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.allreduce_short_msg = flag;
    }
    if ((value = getenv("MV2_ALLGATHER_REVERSE_RANKING")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_allgather_ranking = 1;
        else
            mv2_allgather_ranking = 0;
    }
    if ((value = getenv("MV2_ALLGATHER_RD_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.allgather_rd_threshold = flag;
    }
    if ((value = getenv("MV2_ALLGATHERV_RD_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        mv2_user_allgatherv_switch_point = atoi(value);
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_ALLGATHER_BRUCK_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.allgather_bruck_threshold = flag;
    }
    if ((value = getenv("MV2_ALLREDUCE_2LEVEL_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0) {
            mv2_coll_param.allreduce_2level_threshold = flag;
        }
    }
    if ((value = getenv("MV2_REDUCE_SHORT_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.reduce_short_msg = flag;
    }
    if ((value = getenv("MV2_SHMEM_ALLREDUCE_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.shmem_allreduce_msg = flag;
    }
    if ((value = getenv("MV2_REDUCE_2LEVEL_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0) {
            mv2_coll_param.reduce_2level_threshold = flag;
        }
    }
    if ((value = getenv("MV2_SCATTER_SMALL_MSG")) != NULL) {
        mv2_user_scatter_small_msg = atoi(value);
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_SCATTER_MEDIUM_MSG")) != NULL) {
        mv2_user_scatter_medium_msg = atoi(value);
        mv2_tune_parameter = 1;
    }
#if defined(_MCST_SUPPORT_)
    if ((value = getenv("MV2_USE_MCAST_SCATTER")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1)
            mv2_use_mcast_scatter = 1;
        else
            mv2_use_mcast_scatter = 0;
    }
    if ((value = getenv("MV2_MCAST_SCATTER_MSG_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1)
            mv2_mcast_scatter_msg_size = flag;
    }
    if ((value = getenv("MV2_MCAST_SCATTER_SMALL_SYS_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1)
            mv2_mcast_scatter_small_sys_size = flag;
    }
    if ((value = getenv("MV2_MCAST_SCATTER_LARGE_SYS_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1)
            mv2_mcast_scatter_large_sys_size = flag;
    }
#endif                          /* #if defined(_MCST_SUPPORT_) */
    if ((value = getenv("MV2_GATHER_SWITCH_PT")) != NULL) {
        mv2_user_gather_switch_point = atoi(value);
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_GATHER_TUNING")) != NULL) {
        mv2_user_gather_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_GATHER_TUNING")) != NULL) {
        mv2_user_gather_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_MULTI_LEVEL_GATHER_TUNING")) != NULL) {
        mv2_user_gather_intra_multi_lvl = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_SHMEM_REDUCE_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.shmem_reduce_msg = flag;
    }
    if ((value = getenv("MV2_INTRA_SHMEM_REDUCE_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_coll_param.shmem_intra_reduce_msg = flag;
    }
    if ((value = getenv("MV2_USE_OLD_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_bcast = 1;
        else
            mv2_use_old_bcast = 0;
    }
    if ((value = getenv("MV2_ALLRED_USE_RING")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_allred_use_ring = 1;
        else
            mv2_allred_use_ring = 0;
    }
    if ((value = getenv("MV2_USE_OLD_ALLGATHER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_allgather = 1;
        else
            mv2_use_old_allgather = 0;
    }
    if ((value = getenv("MV2_USE_OLD_SCATTER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_scatter = 1;
        else
            mv2_use_old_scatter = 0;
    }
    if ((value = getenv("MV2_USE_OLD_ALLREDUCE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_allreduce = 1;
        else
            mv2_use_old_allreduce = 0;
    }
    if ((value = getenv("MV2_USE_OLD_REDUCE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_reduce = 1;
        else
            mv2_use_old_reduce = 0;
    }

    if ((value = getenv("MV2_USE_SHMEM_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_enable_shmem_bcast = 1;
        else
            mv2_enable_shmem_bcast = 0;
    }
    if ((value = getenv("MV2_USE_OLD_ALLTOALL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_old_alltoall = 1;
        else
            mv2_use_old_alltoall = 0;
    }
    if ((value = getenv("MV2_ALLTOALL_INPLACE_OLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_alltoall_inplace_old = 1;
        else
            mv2_alltoall_inplace_old = 0;
    }
    if ((value = getenv("MV2_USE_SCATTER_DEST_ALLTOALLV")) != NULL) {
        mv2_use_scatter_dest_alltoallv = !!atoi(value);
    }
    if ((value = getenv("MV2_USE_TWO_LEVEL_GATHER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_two_level_gather = 1;
        else
            mv2_use_two_level_gather = 0;
    }
    if ((value = getenv("MV2_USE_DIRECT_GATHER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_direct_gather = 1;
        else
            mv2_use_direct_gather = 0;
    }
    if ((value = getenv("MV2_USE_TWO_LEVEL_SCATTER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_two_level_scatter = 1;
        else
            mv2_use_two_level_scatter = 0;
    }
    if ((value = getenv("MV2_USE_DIRECT_SCATTER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_use_direct_scatter = 1;
        else
            mv2_use_direct_scatter = 0;
    }
    if ((value = getenv("MV2_USE_DIRECT_GATHER_SYSTEM_SIZE_SMALL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_gather_direct_system_size_small = flag;
        else
            mv2_gather_direct_system_size_small = MV2_GATHER_DIRECT_SYSTEM_SIZE_SMALL;
    }
    if ((value = getenv("MV2_USE_DIRECT_GATHER_SYSTEM_SIZE_MEDIUM")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_gather_direct_system_size_medium = flag;
        else
            mv2_gather_direct_system_size_medium = MV2_GATHER_DIRECT_SYSTEM_SIZE_MEDIUM;
    }
    if ((value = getenv("MV2_ALLTOALL_SMALL_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_coll_param.alltoall_small_msg = flag;
    }
    if ((value = getenv("MV2_ALLTOALL_THROTTLE_FACTOR")) != NULL) {
        flag = (int) atoi(value);
        if (flag <= 1) {
            mv2_coll_param.alltoall_throttle_factor = 1;
            mv2_coll_param.alltoall_intra_throttle_factor = 1;
            mv2_coll_param.alltoall_large_msg_throttle_factor = 1;
        } else {
            mv2_coll_param.alltoall_throttle_factor = flag;
            mv2_coll_param.alltoall_intra_throttle_factor = flag;
            mv2_coll_param.alltoall_large_msg_throttle_factor = flag;
        }
    }
    if ((value = getenv("MV2_ALLTOALL_MEDIUM_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_coll_param.alltoall_medium_msg = flag;
    }
    if ((value = getenv("MV2_USE_XOR_ALLTOALL")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_use_xor_alltoall = flag;
    }
    if ((value = getenv("MV2_KNOMIAL_INTER_LEADER_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_knomial_inter_leader_threshold = flag;
    }
    if ((value = getenv("MV2_KNOMIAL_INTRA_NODE_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0) {
            if (flag < mv2_g_shmem_coll_max_msg_size) {
                mv2_knomial_intra_node_threshold = flag;
            } else {
                mv2_knomial_intra_node_threshold = mv2_g_shmem_coll_max_msg_size;
            }
        }
    }
    if ((value = getenv("MV2_USE_SCATTER_RING_INTER_LEADER_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_scatter_ring_inter_leader_bcast = flag;
    }
    if ((value = getenv("MV2_USE_SCATTER_RD_INTER_LEADER_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_scatter_rd_inter_leader_bcast = flag;
    }
    if ((value = getenv("MV2_USE_KNOMIAL_INTER_LEADER_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_knomial_inter_leader_bcast = flag;
    }
    if ((value = getenv("MV2_BCAST_TWO_LEVEL_SYSTEM_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_bcast_two_level_system_size = flag;
    }
    if ((value = getenv("MV2_USE_BCAST_SHORT_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_bcast_short_msg = flag;
    }
    if ((value = getenv("MV2_INTRA_BCAST_TUNING")) != NULL) {
        mv2_user_bcast_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_IBCAST_TUNING")) != NULL) {
        mv2_user_ibcast_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_IGATHER_TUNING")) != NULL) {
        mv2_user_igather_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_BCAST_TUNING")) != NULL) {
        mv2_user_bcast_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IBCAST_TUNING")) != NULL) {
        mv2_user_ibcast_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IGATHER_TUNING")) != NULL) {
        mv2_user_igather_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_ALLTOALL_TUNING")) != NULL) {
        mv2_user_alltoall = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_ALLTOALLV_TUNING")) != NULL) {
        mv2_user_alltoallv = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_SCATTER_TUNING")) != NULL) {
        mv2_user_scatter_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_SCATTER_TUNING")) != NULL) {
        mv2_user_scatter_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_ISCATTER_TUNING")) != NULL) {
        mv2_user_iscatter_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_ISCATTER_TUNING")) != NULL) {
        mv2_user_iscatter_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTRA_ALLREDUCE_TUNING")) != NULL) {
        mv2_user_allreduce_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_ALLREDUCE_TUNING")) != NULL) {
        mv2_user_allreduce_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IALLREDUCE_TUNING")) != NULL) {
        mv2_user_iallreduce_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IALLREDUCE_TUNING")) != NULL) {
        mv2_user_iallreduce_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTRA_REDUCE_TUNING")) != NULL) {
        mv2_user_reduce_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_REDUCE_TUNING")) != NULL) {
        mv2_user_reduce_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IREDUCE_TUNING")) != NULL) {
        mv2_user_ireduce_intra = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTER_IREDUCE_TUNING")) != NULL) {
        mv2_user_ireduce_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IREDUCE_SCATTER_TUNING")) != NULL) {
        mv2_user_ireduce_scatter_intra = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTER_IREDUCE_SCATTER_TUNING")) != NULL) {
        mv2_user_ireduce_scatter_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTER_ALLGATHER_TUNING")) != NULL) {
        mv2_user_allgather_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IALLGATHER_TUNING")) != NULL) {
        mv2_user_iallgather_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_IALLGATHER_TUNING")) != NULL) {
        mv2_user_iallgather_intra = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IALLTOALL_TUNING")) != NULL) {
        mv2_user_ialltoall_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IALLTOALL_TUNING")) != NULL) {
        mv2_user_ialltoall_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IALLTOALLV_TUNING")) != NULL) {
        mv2_user_ialltoallv_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IALLTOALLV_TUNING")) != NULL) {
        mv2_user_ialltoallv_inter = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTER_IALLGATHERV_TUNING")) != NULL) {
        mv2_user_iallgatherv_inter = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTRA_IALLGATHERV_TUNING")) != NULL) {
        mv2_user_iallgatherv_intra = value;
        mv2_tune_parameter = 1;
    }
    
    if ((value = getenv("MV2_INTRA_IBARRIER_TUNING")) != NULL) {
        mv2_user_ibarrier_intra = value;
        mv2_tune_parameter = 1;
    }
    if ((value = getenv("MV2_INTER_IBARRIER_TUNING")) != NULL) {
        mv2_user_ibarrier_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTER_RED_SCAT_TUNING")) != NULL) {
        mv2_user_red_scat_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTER_RED_SCAT_BLOCK_TUNING")) != NULL) {
        mv2_user_red_scat_block_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTER_ALLGATHERV_TUNING")) != NULL) {
        mv2_user_allgatherv_inter = value;
        mv2_tune_parameter = 1;
    }

    if ((value = getenv("MV2_INTER_ALLREDUCE_TUNING_TWO_LEVEL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_user_allreduce_two_level = flag;
    }

    if ((value = getenv("MV2_INTER_REDUCE_TUNING_TWO_LEVEL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_user_reduce_two_level = flag;
    }
    if ((value = getenv("MV2_INTER_ALLGATHER_TUNING_TWO_LEVEL")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_user_allgather_two_level = flag;
    }

    if ((value = getenv("MV2_SHMEM_COLL_SPIN_COUNT")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_shmem_coll_spin_count = flag;
    }

    if ((value = getenv("MV2_USE_KNOMIAL_2LEVEL_BCAST")) != NULL) {
        mv2_enable_knomial_2level_bcast = !!atoi(value);
        if (mv2_enable_knomial_2level_bcast <= 0) {
            mv2_enable_knomial_2level_bcast = 0;
        }
    }

    if ((value = getenv("MV2_KNOMIAL_2LEVEL_BCAST_MESSAGE_SIZE_THRESHOLD"))
        != NULL) {
        mv2_knomial_2level_bcast_message_size_threshold = atoi(value);
    }

    if ((value = getenv("MV2_KNOMIAL_2LEVEL_BCAST_SYSTEM_SIZE_THRESHOLD"))
        != NULL) {
        mv2_knomial_2level_bcast_system_size_threshold = atoi(value);
    }

    if ((value = getenv("MV2_KNOMIAL_INTRA_NODE_FACTOR")) != NULL) {
        mv2_intra_node_knomial_factor = atoi(value);
        if (mv2_intra_node_knomial_factor < MV2_INTRA_NODE_KNOMIAL_FACTOR_MIN) {
            mv2_intra_node_knomial_factor = MV2_INTRA_NODE_KNOMIAL_FACTOR_MIN;
        }
        if (mv2_intra_node_knomial_factor > MV2_INTRA_NODE_KNOMIAL_FACTOR_MAX) {
            mv2_intra_node_knomial_factor = MV2_INTRA_NODE_KNOMIAL_FACTOR_MAX;
        }
    }

    if ((value = getenv("MV2_KNOMIAL_INTER_NODE_FACTOR")) != NULL) {
        mv2_inter_node_knomial_factor = atoi(value);
        if (mv2_inter_node_knomial_factor < MV2_INTER_NODE_KNOMIAL_FACTOR_MIN) {
            mv2_inter_node_knomial_factor = MV2_INTER_NODE_KNOMIAL_FACTOR_MIN;
        }
        if (mv2_inter_node_knomial_factor > MV2_INTER_NODE_KNOMIAL_FACTOR_MAX) {
            mv2_inter_node_knomial_factor = MV2_INTER_NODE_KNOMIAL_FACTOR_MAX;
        }
    }
    if ((value = getenv("MV2_USE_PIPELINED_ZCPY_BCAST_KNOMIAL_FACTOR")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
           mv2_pipelined_zcpy_knomial_factor = flag;
    }
    if ((value = getenv("MV2_USE_ZCOPY_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
           mv2_enable_zcpy_bcast = flag;
    }
    if ((value = getenv("MV2_USE_ZCOPY_REDUCE")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
           mv2_enable_zcpy_reduce = flag;
    }
    
    if ((value = getenv("MV2_GATHERV_SSEND_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0) {
            mv2_gatherv_ssend_threshold = flag;
	}
    }

    if ((value = getenv("MV2_ALLREDUCE_RING_ALGO_PPN_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1) {
            mv2_allreduce_ring_algo_ppn_threshold = flag;
        }
    }
    if ((value = getenv("MV2_ALLREDUCE_RING_ALGO_THRESHOLD")) != NULL) {
        mv2_allreduce_ring_algo_threshold =
            user_val_to_bytes(value, "MV2_ALLREDUCE_RING_ALGO_THRESHOLD");

        if (mv2_allreduce_ring_algo_threshold < 0)
            mv2_allreduce_ring_algo_threshold = 0;
    }

    if ((value = getenv("MV2_ALLREDUCE_RED_SCAT_ALLGATHER_ALGO_PPN_THRESHOLD")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 1) {
            mv2_allreduce_red_scat_allgather_algo_ppn_threshold = flag;
        }
    }
    if ((value = getenv("MV2_ALLREDUCE_RED_SCAT_ALLGATHER_ALGO_THRESHOLD")) != NULL) {
        mv2_allreduce_red_scat_allgather_algo_threshold =
            user_val_to_bytes(value, "MV2_ALLREDUCE_RED_SCAT_ALLGATHER_ALGO_THRESHOLD");

        if (mv2_allreduce_red_scat_allgather_algo_threshold < 0)
            mv2_allreduce_red_scat_allgather_algo_threshold = 0;
    }

    if ((value = getenv("MV2_ALLGATHER_RING_ALGO_THRESHOLD")) != NULL) {

        mv2_allgather_ring_algo_threshold = user_val_to_bytes(value, "MV2_ALLGATHER_RING_ALGO_THRESHOLD");

        if (mv2_allgather_ring_algo_threshold < 0)
            mv2_allgather_ring_algo_threshold = 0;
    }

    if ((value = getenv("MV2_ALLTOALLV_INTERMEDIATE_WAIT_THRESHOLD")) != NULL) {

        mv2_alltoallv_intermediate_wait_threshold = user_val_to_bytes(value, "MV2_ALLTOALLV_INTERMEDIATE_WAIT_THRESHOLD");

        if (mv2_alltoallv_intermediate_wait_threshold < 0)
            mv2_alltoallv_intermediate_wait_threshold = 1024*1024;
    }

    if ((value = getenv("MV2_ALLGATHER_CYCLIC_ALGO_THRESHOLD")) != NULL) {

        mv2_allgather_cyclic_algo_threshold = user_val_to_bytes(value, "MV2_ALLGATHER_CYCLIC_ALGO_THRESHOLD");

        if (mv2_allgather_cyclic_algo_threshold < 0)
            mv2_allgather_cyclic_algo_threshold = 0;
    }

    if ((value = getenv("MV2_ALLREDUCE_CYCLIC_ALGO_THRESHOLD")) != NULL) {

        mv2_allreduce_cyclic_algo_threshold = user_val_to_bytes(value, "MV2_ALLREDUCE_CYCLIC_ALGO_THRESHOLD");

        if (mv2_allreduce_cyclic_algo_threshold < 0)
            mv2_allreduce_cyclic_algo_threshold = 0;
    }

    if ((value = getenv("MV2_REDSCAT_CYCLIC_ALGO_THRESHOLD")) != NULL) {

        mv2_redscat_cyclic_algo_threshold = user_val_to_bytes(value, "MV2_REDSCAT_CYCLIC_ALGO_THRESHOLD");
        
        if (mv2_redscat_cyclic_algo_threshold < 0)
            mv2_redscat_cyclic_algo_threshold = 0;
    }

    if ((value = getenv("MV2_RED_SCAT_RING_ALGO_THRESHOLD")) != NULL) {

        mv2_red_scat_ring_algo_threshold = user_val_to_bytes(value, "MV2_RED_SCAT_RING_ALGO_THRESHOLD");

        if (mv2_red_scat_ring_algo_threshold < 0)
            mv2_red_scat_ring_algo_threshold = 0;
    }

    if ((value = getenv("MV2_RED_SCAT_SHORT_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_red_scat_short_msg = flag;
    }

    if ((value = getenv("MV2_RED_SCAT_LARGE_MSG")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_red_scat_long_msg = flag;
    }

    if ((value = getenv("MV2_BCAST_SCATTER_RING_OVERLAP")) != NULL) {
        flag = (int) atoi(value);
        if (flag == 0)
            mv2_bcast_scatter_ring_overlap = 0;
    }

    if ((value = getenv("MV2_BCAST_SCATTER_RING_OVERLAP_MSG_UPPERBOUND")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_bcast_scatter_ring_overlap_msg_upperbound = flag;
    }

    if ((value = getenv("MV2_BCAST_SCATTER_RING_OVERLAP_CORES_LOWERBOUND")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            mv2_bcast_scatter_ring_overlap_cores_lowerbound = flag;
    }

    if ((value = getenv("MV2_USE_PIPELINE_BCAST")) != NULL) {
        flag = (int) atoi(value);
        if (flag >= 0)
            mv2_use_pipelined_bcast = flag;
    }

    if ((value = getenv("BCAST_SEGMENT_SIZE")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            bcast_segment_size = flag;
    }

#if defined(_SMP_LIMIC_)
    if ((value = getenv("MV2_USE_LIMIC_GATHER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            use_limic_gather = flag;
    }
#endif                          /*#if defined(_SMP_LIMIC_) */

    if ((value = getenv("MV2_USE_2LEVEL_ALLGATHER")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            use_2lvl_allgather = flag;
    }

    if ((value = getenv("MV2_SHMEM_COLL_WINDOW_SIZE")) != NULL) {
        mv2_shm_window_size = atoi(value);
    }
    if ((value = getenv("MV2_SHMEM_REDUCE_TREE_DEGREE")) != NULL) {
        mv2_shm_reduce_tree_degree = atoi(value);
    }
    if ((value = getenv("MV2_SHMEM_COLL_SLOT_LEN")) != NULL) {
        mv2_shm_slot_len = atoi(value);
    }
    if ((value = getenv("MV2_USE_SLOT_SHMEM_COLL")) != NULL) {
        mv2_use_slot_shmem_coll = atoi(value);
    }
    if ((value = getenv("MV2_USE_SLOT_SHMEM_BCAST")) != NULL) {
        mv2_use_slot_shmem_bcast = atoi(value);
    }
    if ((value = getenv("MV2_USE_MCAST_PIPELINE_SHM")) != NULL) {
        mv2_use_mcast_pipeline_shm = atoi(value);
    }
    if ((value = getenv("MV2_TWO_LEVEL_COMM_THRESHOLD")) != NULL) {
        shmem_coll_count_threshold = atoi(value);
    }
    if ((value = getenv("MV2_TWO_LEVEL_COMM_EARLY_INIT_THRESHOLD")) != NULL) {
        mv2_two_level_comm_early_init_threshold = atoi(value);
    }

    if(mv2_use_slot_shmem_coll == 0 || mv2_use_slot_shmem_bcast  == 0) { 
       /* Disable zero-copy bcast if slot-shmem, or slot-shmem-bcast params
        * are off  */ 
       mv2_enable_zcpy_bcast = 0; 
    }

#if ENABLE_PVAR_MV2
    if ((value = getenv("MV2_ENABLE_PVAR_TIMER")) !=NULL) {
      mv2_enable_pvar_timer = atoi(value);
    }
#endif 

    if ((value = getenv("MV2_ENABLE_SOCKET_AWARE_COLLECTIVES")) !=NULL) {
        mv2_enable_socket_aware_collectives = !!atoi(value);
    }

    if ((value = getenv("MV2_USE_SOCKET_AWARE_ALLREDUCE")) !=NULL) {
        mv2_use_socket_aware_allreduce = !!atoi(value);
    }

    if ((value = getenv("MV2_USE_SOCKET_AWARE_SHARP_ALLREDUCE")) !=NULL) {
        mv2_use_socket_aware_sharp_allreduce = !!atoi(value);                   
    }

    if ((value = getenv("MV2_USE_SOCKET_AWARE_BARRIER")) !=NULL) {
        mv2_use_socket_aware_barrier = !!atoi(value);
    }

    if ((value = getenv("MV2_SOCKET_AWARE_ALLREDUCE_MAX_MSG")) !=NULL) {
        mv2_socket_aware_allreduce_max_msg = atoi(value);
    }

    if ((value = getenv("MV2_SOCKET_AWARE_ALLREDUCE_MIN_MSG")) !=NULL) {
        mv2_socket_aware_allreduce_min_msg = atoi(value);
    }

    /* Override MPICH2 default env values for Gatherv */
    MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS = 1024;
    if ((value = getenv("MV2_GATHERV_SSEND_MIN_PROCS")) != NULL) {
        flag = (int) atoi(value);
        if (flag > 0)
            MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS = flag;
    }
    init_thread_reg();
}

#ifdef _ENABLE_CUDA_
/******************************************************************/
//Checks if cuda stage buffer size is sufficient, if not allocates//
//more memory.                                                    //
/******************************************************************/
int cuda_stage_alloc(void **send_buf, int sendsize,
                     void **recv_buf, int recvsize,
                     int send_on_device, int recv_on_device, int disp)
{
    int page_size = getpagesize();
    int result, mpi_errno = MPI_SUCCESS;

    if (send_on_device && *send_buf != MPI_IN_PLACE && mv2_cuda_host_sendbuf_size < sendsize) {
        if (mv2_cuda_host_send_buf) {
            if (mv2_cuda_host_sendbuf_size >= rdma_cuda_register_naive_buf) {
                ibv_cuda_unregister(mv2_cuda_host_send_buf);
            }
            MPIU_Memalign_Free(mv2_cuda_host_send_buf);
        }
        mv2_cuda_host_sendbuf_size =
            sendsize < rdma_cuda_block_size ? rdma_cuda_block_size : sendsize;
        result = MPIU_Memalign(&mv2_cuda_host_send_buf, page_size, mv2_cuda_host_sendbuf_size);
        if ((result != 0) || (NULL == mv2_cuda_host_send_buf)) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "posix_memalign", strerror(errno));
            return (mpi_errno);
        }
        if (mv2_cuda_host_sendbuf_size >= rdma_cuda_register_naive_buf) {
            ibv_cuda_register(mv2_cuda_host_send_buf, mv2_cuda_host_sendbuf_size);
        }
    }
    if (recv_on_device && mv2_cuda_host_recvbuf_size < recvsize) {
        if (mv2_cuda_host_recv_buf) {
            if (mv2_cuda_host_recvbuf_size >= rdma_cuda_register_naive_buf) {
                ibv_cuda_unregister(mv2_cuda_host_recv_buf);
            }
            MPIU_Memalign_Free(mv2_cuda_host_recv_buf);
        }
        mv2_cuda_host_recvbuf_size =
            recvsize < rdma_cuda_block_size ? rdma_cuda_block_size : recvsize;
        result = MPIU_Memalign(&mv2_cuda_host_recv_buf, page_size, mv2_cuda_host_recvbuf_size);
        if ((result != 0) || (NULL == mv2_cuda_host_recv_buf)) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "posix_memalign", strerror(errno));
            return (mpi_errno);
        }
        if (mv2_cuda_host_recvbuf_size >= rdma_cuda_register_naive_buf) {
            ibv_cuda_register(mv2_cuda_host_recv_buf, mv2_cuda_host_recvbuf_size);
        }
    }

    if (send_on_device && *send_buf != MPI_IN_PLACE) {
        if (send_on_device) {
            MPIU_Memcpy_CUDA(mv2_cuda_host_send_buf, *send_buf, sendsize, cudaMemcpyDeviceToHost);
        }
    } else {
        if (recv_on_device) {
            MPIU_Memcpy_CUDA(((char *) (mv2_cuda_host_recv_buf) + disp),
                             ((char *) (*recv_buf) + disp),
                             sendsize, cudaMemcpyDeviceToHost);
        }
    }

    if (send_on_device && send_buf != MPI_IN_PLACE) {
        mv2_cuda_original_send_buf = *send_buf;
        *send_buf = mv2_cuda_host_send_buf;
    } else {
        mv2_cuda_original_send_buf = NULL;
    }
    if (recv_on_device) {
        mv2_cuda_original_recv_buf = *recv_buf;
        *recv_buf = mv2_cuda_host_recv_buf;
    } else {
        mv2_cuda_original_recv_buf = NULL;
    }
    return mpi_errno;
}

/******************************************************************/
//After performing the cuda collective operation, sendbuf and recv//
//-buf are made to point back to device buf.                      //
/******************************************************************/
void cuda_stage_free(void **send_buf,
                     void **recv_buf, int recvsize,
                     int send_on_device, int recv_on_device)
{

    if (send_on_device && mv2_cuda_original_send_buf && send_buf != MPI_IN_PLACE) {
        if (!recv_on_device && !mv2_cuda_original_recv_buf) {
            MPIU_Memcpy_CUDA(mv2_cuda_original_send_buf, *send_buf,
                             recvsize, cudaMemcpyHostToDevice);
        }
        *send_buf = mv2_cuda_original_send_buf;
        mv2_cuda_original_send_buf = NULL;
    }
    if (recv_on_device && mv2_cuda_original_recv_buf) {
        MPIU_Memcpy_CUDA(mv2_cuda_original_recv_buf, *recv_buf, recvsize, cudaMemcpyHostToDevice);
        *recv_buf = mv2_cuda_original_recv_buf;
        mv2_cuda_original_recv_buf = NULL;
    }
}

int cuda_stage_alloc_v(void **send_buf, int *send_counts, MPI_Datatype send_type,
                       int **send_displs, int send_peers,
                       void **recv_buf, int *recv_counts, MPI_Datatype recv_type,
                       int **recv_displs, int recv_peers,
                       int send_buf_on_device, int recv_buf_on_device, int rank)
{
    int mpi_errno = MPI_SUCCESS;
    int i, page_size, result;
    int total_send_size = 0, total_recv_size = 0, total_buf_size = 0, offset = 0;
    int recv_type_contig = 0;
    MPI_Aint send_type_extent, recv_type_extent;
    MPID_Datatype *dtp;
    cudaError_t cuda_err = cudaSuccess;

    page_size = getpagesize();
    MPID_Datatype_get_extent_macro(send_type, send_type_extent);
    MPID_Datatype_get_extent_macro(recv_type, recv_type_extent);

    if (HANDLE_GET_KIND(recv_type) == HANDLE_KIND_BUILTIN)
        recv_type_contig = 1;
    else {
        MPID_Datatype_get_ptr(recv_type, dtp);
        recv_type_contig = dtp->is_contig;
    }

    if (send_buf_on_device && *send_buf != MPI_IN_PLACE) {
        for (i = 0; i < send_peers; i++) {
            total_send_size += send_counts[i] * send_type_extent;
        }
    }
    if (recv_buf_on_device && *recv_buf != MPI_IN_PLACE) {
        for (i = 0; i < recv_peers; i++) {
            total_recv_size += recv_counts[i] * recv_type_extent;
        }
    }
    total_buf_size =
        (total_send_size > total_recv_size) ? total_send_size : total_recv_size;

    /* Allocate the packing buffer on device if one does not exist 
     * or is not large enough. Free the older one */
    if (mv2_cuda_dev_srbuf_size < total_buf_size) {
        if (mv2_cuda_dev_sr_buf) {
            cudaFree(mv2_cuda_dev_sr_buf);
        }
        mv2_cuda_dev_srbuf_size = total_buf_size;
        cuda_err = cudaMalloc(&mv2_cuda_dev_sr_buf, mv2_cuda_dev_srbuf_size);
        if (cuda_err != cudaSuccess) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                             FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                             "%s: %s", "cudaMalloc failed",
                                             cudaGetErrorString(cuda_err));
            return (mpi_errno);
        }
    }

    /* Allocate the stage out (send) host buffers if they do not exist 
     * or are not large enough. Free the older one */
    if (send_buf_on_device && *send_buf != MPI_IN_PLACE) {
        /*allocate buffer to stage displacements */
        if (mv2_cuda_host_send_peers < send_peers) {
            if (mv2_cuda_host_send_displs) {
                MPIU_Free(mv2_cuda_host_send_displs);
            }
            mv2_cuda_host_send_peers = send_peers;
            mv2_cuda_host_send_displs = MPIU_Malloc(sizeof (int) * mv2_cuda_host_send_peers);
            MPIU_Memset((void *) mv2_cuda_host_send_displs, 0, sizeof (int) * mv2_cuda_host_send_peers);
        }
        /*allocate buffer to stage the data */
        if (mv2_cuda_host_sendbuf_size < total_send_size) {
            if (mv2_cuda_host_send_buf) {
                if (mv2_cuda_host_sendbuf_size >= rdma_cuda_register_naive_buf) {
                    ibv_cuda_unregister(mv2_cuda_host_send_buf);
                }
                MPIU_Memalign_Free(mv2_cuda_host_send_buf);
            }
            mv2_cuda_host_sendbuf_size = total_send_size < rdma_cuda_block_size ?
                rdma_cuda_block_size : total_send_size;
            result = MPIU_Memalign(&mv2_cuda_host_send_buf, page_size, mv2_cuda_host_sendbuf_size);
            if ((result != 0) || (NULL == mv2_cuda_host_send_buf)) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**fail", "%s: %s", "posix_memalign",
                                                 strerror(errno));
                return (mpi_errno);
            }
            if (mv2_cuda_host_sendbuf_size >= rdma_cuda_register_naive_buf) {
                ibv_cuda_register(mv2_cuda_host_send_buf, mv2_cuda_host_sendbuf_size);
            }
        }
    }

    /* allocate the stage in (recv) host buffers if they do not exist 
     * or are not large enough */
    if (recv_buf_on_device && *recv_buf != MPI_IN_PLACE) {
        /*allocate buffer to stage displacements */
        if (mv2_cuda_host_recv_peers < recv_peers) {
            if (mv2_cuda_host_recv_displs) {
                MPIU_Free(mv2_cuda_host_recv_displs);
            }
            mv2_cuda_host_recv_peers = recv_peers;
            mv2_cuda_host_recv_displs = MPIU_Malloc(sizeof (int) * mv2_cuda_host_recv_peers);
            MPIU_Memset(mv2_cuda_host_recv_displs, 0, sizeof (int) * mv2_cuda_host_recv_peers);
        }
        /*allocate buffer to stage the data */
        if (mv2_cuda_host_recvbuf_size < total_recv_size) {
            if (mv2_cuda_host_recv_buf) {
                if (mv2_cuda_host_recvbuf_size >= rdma_cuda_register_naive_buf) {
                    ibv_cuda_unregister(mv2_cuda_host_recv_buf);
                }
                MPIU_Memalign_Free(mv2_cuda_host_recv_buf);
            }
            mv2_cuda_host_recvbuf_size = total_recv_size < rdma_cuda_block_size ?
                rdma_cuda_block_size : total_recv_size;
            result = MPIU_Memalign(&mv2_cuda_host_recv_buf, page_size, mv2_cuda_host_recvbuf_size);
            if ((result != 0) || (NULL == mv2_cuda_host_recv_buf)) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**fail", "%s: %s", "posix_memalign",
                                                 strerror(errno));
                return (mpi_errno);
            }
            if (mv2_cuda_host_recvbuf_size >= rdma_cuda_register_naive_buf) {
                ibv_cuda_register(mv2_cuda_host_recv_buf, mv2_cuda_host_recvbuf_size);
            }
        }
    }

    /*Stage out the data to be sent, set the send buffer and displaceemnts */
    offset = 0;
    if (send_buf_on_device && *send_buf != MPI_IN_PLACE) {
        for (i = 0; i < send_peers; i++) {
            MPIU_Memcpy_CUDA((void *) ((char *) mv2_cuda_dev_sr_buf
                                       + offset * send_type_extent),
                             (void *) ((char *) (*send_buf)
                                       + (*send_displs)[i] * send_type_extent),
                             send_counts[i] * send_type_extent, cudaMemcpyDeviceToDevice);
            mv2_cuda_host_send_displs[i] = offset;
            offset += send_counts[i];
        }
        MPIU_Memcpy_CUDA(mv2_cuda_host_send_buf, mv2_cuda_dev_sr_buf,
                         total_send_size, cudaMemcpyDeviceToHost);

        mv2_cuda_original_send_buf = (void *) *send_buf;
        mv2_cuda_original_send_displs = (int *)*send_displs;
        *send_buf = mv2_cuda_host_send_buf;
        *send_displs = mv2_cuda_host_send_displs;
    }

    /*Stage out buffer into which data is to be received and set the stage in 
       (recv) displacements */
    offset = 0;
    if (recv_buf_on_device && *recv_buf != MPI_IN_PLACE) {
        for (i = 0; i < recv_peers; i++) {
            mv2_cuda_host_recv_displs[i] = offset;
            offset += recv_counts[i];
        }
        /*If data type is not contig, copy the device receive buffer out onto host receive buffer 
           to maintain the original data in un-touched parts while copying back */
        if (!recv_type_contig) {
            for (i = 0; i < recv_peers; i++) {
                MPIU_Memcpy_CUDA((void *) ((char *) mv2_cuda_dev_sr_buf
                                           + mv2_cuda_host_recv_displs[i] * recv_type_extent),
                                 (void *) ((char *) (*recv_buf)
                                           + (*recv_displs)[i] * recv_type_extent),
                                 recv_counts[i] * recv_type_extent,
                                 cudaMemcpyDeviceToDevice);
            }
            MPIU_Memcpy_CUDA(mv2_cuda_host_recv_buf, mv2_cuda_dev_sr_buf,
                             total_recv_size, cudaMemcpyDeviceToHost);
        }
        mv2_cuda_original_recv_buf = *recv_buf;
        mv2_cuda_original_recv_displs = *recv_displs;
        *recv_buf = mv2_cuda_host_recv_buf;
        *recv_displs = mv2_cuda_host_recv_displs;
    }

    return mpi_errno;
}

void cuda_stage_free_v(void **send_buf, int *send_counts, MPI_Datatype send_type,
                       int **send_displs, int send_peers,
                       void **recv_buf, int *recv_counts, MPI_Datatype recv_type,
                       int **recv_displs, int recv_peers,
                       int send_buf_on_device, int recv_buf_on_device, int rank)
{
    int i, total_recv_size = 0;
    MPI_Aint recv_type_extent = 0;

    if (recv_buf_on_device && *recv_buf != MPI_IN_PLACE) {
        MPID_Datatype_get_extent_macro(recv_type, recv_type_extent);
        for (i = 0; i < recv_peers; i++) {
            total_recv_size += recv_counts[i] * recv_type_extent;
        }
    }

    if (send_buf_on_device && *send_buf != MPI_IN_PLACE) {
        MPIU_Assert(mv2_cuda_original_send_buf != NULL);
        *send_buf = mv2_cuda_original_send_buf;
        *send_displs = mv2_cuda_original_send_displs;
        mv2_cuda_original_send_buf = NULL;
        mv2_cuda_original_send_displs = NULL;
    }

    if (recv_buf_on_device && *recv_buf != MPI_IN_PLACE) {
        MPIU_Memcpy_CUDA(mv2_cuda_dev_sr_buf, *recv_buf, total_recv_size, cudaMemcpyHostToDevice);
        for (i = 0; i < recv_peers; i++) {
            if (send_buf_on_device && *send_buf == MPI_IN_PLACE && i == rank)
                continue;
            MPIU_Memcpy_CUDA((void *) ((char *) mv2_cuda_original_recv_buf
                                       + mv2_cuda_original_recv_displs[i] * recv_type_extent),
                             (void *) ((char *) mv2_cuda_dev_sr_buf
                                       + (*recv_displs)[i] * recv_type_extent),
                             recv_counts[i] * recv_type_extent, cudaMemcpyDeviceToDevice);
        }
        *recv_buf = mv2_cuda_original_recv_buf;
        *recv_displs = mv2_cuda_original_recv_displs;
        mv2_cuda_original_recv_buf = NULL;
        mv2_cuda_original_recv_displs = NULL;
    }
}

/******************************************************************/
//Freeing the stage buffers during finalize                       //
/******************************************************************/
void CUDA_COLL_Finalize()
{
    if (mv2_cuda_host_recv_buf) {
        if (mv2_cuda_host_recvbuf_size >= rdma_cuda_register_naive_buf) {
            ibv_cuda_unregister(mv2_cuda_host_recv_buf);
        }
        MPIU_Memalign_Free(mv2_cuda_host_recv_buf);
        mv2_cuda_host_recv_buf = NULL;
    }

    if (mv2_cuda_host_send_displs) {
        MPIU_Free(mv2_cuda_host_send_displs);
        mv2_cuda_host_send_displs = NULL;
    }

    if (mv2_cuda_host_send_buf) {
        if (mv2_cuda_host_sendbuf_size >= rdma_cuda_register_naive_buf) {
            ibv_cuda_unregister(mv2_cuda_host_send_buf);
        }
        MPIU_Memalign_Free(mv2_cuda_host_send_buf);
        mv2_cuda_host_send_buf = NULL;
    }

    if (mv2_cuda_host_recv_displs) {
        MPIU_Free(mv2_cuda_host_recv_displs);
        mv2_cuda_host_recv_displs = NULL;
    }

    if (mv2_cuda_allgather_store_buf) {
        ibv_cuda_unregister(mv2_cuda_allgather_store_buf);
        MPIU_Memalign_Free(mv2_cuda_allgather_store_buf);
        mv2_cuda_allgather_store_buf = NULL;
    }

    if (mv2_cuda_coll_pack_buf_size) {
        MPIU_Free_CUDA(mv2_cuda_coll_pack_buf);
        mv2_cuda_coll_pack_buf_size = 0;
    }

    if (mv2_cuda_coll_unpack_buf_size) {
        MPIU_Free_CUDA(mv2_cuda_coll_unpack_buf);
        mv2_cuda_coll_unpack_buf_size = 0;
    }
}

/******************************************************************/
//Packing non-contig sendbuf                                      //
/******************************************************************/
void cuda_coll_pack(void **sendbuf, int *sendcount, MPI_Datatype * sendtype,
                    void **recvbuf, int *recvcount, MPI_Datatype * recvtype,
                    int disp, int procs_in_sendbuf, int comm_size)
{

    int sendtype_size = 0, recvtype_size = 0, sendsize = 0, recvsize = 0,
        send_copy_size = 0;
    int sendtype_iscontig = 0, recvtype_iscontig = 0;

    if (*sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(*sendtype, &sendtype_iscontig);
    }
    if (*recvtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(*recvtype, &recvtype_iscontig);
    }

    MPID_Datatype_get_size_macro(*sendtype, sendtype_size);
    MPID_Datatype_get_size_macro(*recvtype, recvtype_size);

    /*Calulating size of data in recv and send buffers */
    if (*sendbuf != MPI_IN_PLACE) {
        sendsize = *sendcount * sendtype_size;
        send_copy_size = *sendcount * sendtype_size * procs_in_sendbuf;
    } else {
        sendsize = *recvcount * recvtype_size;
        send_copy_size = *recvcount * recvtype_size * procs_in_sendbuf;
    }
    recvsize = *recvcount * recvtype_size * comm_size;

    /*Creating packing and unpacking buffers */
    if (!sendtype_iscontig && send_copy_size > mv2_cuda_coll_pack_buf_size) {
        MPIU_Free_CUDA(mv2_cuda_coll_pack_buf);
        MPIU_Malloc_CUDA(mv2_cuda_coll_pack_buf, send_copy_size);
        mv2_cuda_coll_pack_buf_size = send_copy_size;
    }
    if (!recvtype_iscontig && recvsize > mv2_cuda_coll_unpack_buf_size) {
        MPIU_Free_CUDA(mv2_cuda_coll_unpack_buf);
        MPIU_Malloc_CUDA(mv2_cuda_coll_unpack_buf, recvsize);
        mv2_cuda_coll_unpack_buf_size = recvsize;
    }

    /*Packing of data to sendbuf */
    if (*sendbuf != MPI_IN_PLACE && !sendtype_iscontig) {
        MPIR_Localcopy(*sendbuf, *sendcount * procs_in_sendbuf, *sendtype,
                       mv2_cuda_coll_pack_buf, send_copy_size, MPI_BYTE);
        *sendbuf = mv2_cuda_coll_pack_buf;
        *sendcount = sendsize;
        *sendtype = MPI_BYTE;
    } else if (*sendbuf == MPI_IN_PLACE && !recvtype_iscontig) {
        MPIR_Localcopy((void *) ((char *) (*recvbuf) + disp),
                       (*recvcount) * procs_in_sendbuf, *recvtype,
                       mv2_cuda_coll_pack_buf, send_copy_size, MPI_BYTE);
        *sendbuf = mv2_cuda_coll_pack_buf;
        *sendcount = sendsize;
        *sendtype = MPI_BYTE;
    }

    /*Changing recvbuf to contig temp recvbuf */
    if (!recvtype_iscontig) {
        mv2_cuda_orig_recvbuf = *recvbuf;
        mv2_cuda_orig_recvcount = *recvcount;
        mv2_cuda_orig_recvtype = *recvtype;
        *recvbuf = mv2_cuda_coll_unpack_buf;
        *recvcount = *recvcount * recvtype_size;
        *recvtype = MPI_BYTE;
    }
}

/******************************************************************/
//Unpacking data to non-contig recvbuf                            //
/******************************************************************/
void cuda_coll_unpack(int *recvcount, int comm_size)
{

    int recvtype_iscontig = 0;

    if (mv2_cuda_orig_recvbuf && mv2_cuda_orig_recvtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(mv2_cuda_orig_recvtype, &recvtype_iscontig);
    }

    /*Unpacking of data to recvbuf */
    if (mv2_cuda_orig_recvbuf && !recvtype_iscontig) {
        MPIR_Localcopy(mv2_cuda_coll_unpack_buf, *recvcount * comm_size, MPI_BYTE,
                       mv2_cuda_orig_recvbuf, mv2_cuda_orig_recvcount * comm_size, mv2_cuda_orig_recvtype);
    }

    mv2_cuda_orig_recvbuf = NULL;
}
#endif                          /* #ifdef _ENABLE_CUDA_ */

#if defined(_SMP_LIMIC_)
int MPIDI_CH3I_LIMIC_Gather_Start(void *buf, MPI_Aint size, int sendbuf_offset,
                                  int comm_size, int rank, int shmem_comm_rank,
                                  MPID_Comm * comm_ptr)
{
    int i = 1, cnt = 0, err = 0;
    int mpi_errno = MPI_SUCCESS;
    int tx_init_size;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_LIMIC_GATHER_START);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_LIMIC_GATHER_START);

    if (rank == 0) {

        for (; i < comm_size; ++i) {
            READBAR();
            while (SHMEM_COLL_SYNC_ISSET(limic_progress, shmem_comm_rank, i)) {
#if defined(CKPT)
                Wait_for_CR_Completion();
#endif
                MPIDI_CH3I_Progress_test();
                /* Yield once in a while */
                MPIU_THREAD_CHECK_BEGIN++ cnt;
                if (cnt >= 20) {
                    cnt = 0;
#if defined(CKPT)
                    MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
                        do {
                    } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                        MPIDI_CH3I_CR_lock();
#endif
                }
                MPIU_THREAD_CHECK_END
                READBAR();
            }
        }

        if (buf != NULL && size > 0) {

            tx_init_size = limic_tx_init(limic_fd, buf, size,
                                         &(shmem_coll->limic_hndl[shmem_comm_rank]));
            if (tx_init_size != size) {
                printf("Error limic_tx_init:ibv_error_abort, ch3_shmem_coll\n");
            }
            /*store offset after offsetting the sender buf */
            shmem_coll->leader_buf_offset[shmem_comm_rank] =
                shmem_coll->limic_hndl[shmem_comm_rank].offset + sendbuf_offset;
        }
    } else {
        READBAR();
        while (SHMEM_COLL_SYNC_ISCLR(limic_progress, shmem_comm_rank, rank)) {
#if defined(CKPT)
            Wait_for_CR_Completion();
#endif
            MPIDI_CH3I_Progress_test();
            /* Yield once in a while */
            MPIU_THREAD_CHECK_BEGIN++ cnt;
            if (cnt >= 20) {
                cnt = 0;
#if defined(CKPT)
                MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
                    do {
                } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                MPIU_THREAD_CHECK_BEGIN
                    MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                    MPIDI_CH3I_CR_lock();
#endif
            }
            MPIU_THREAD_CHECK_END
            READBAR();
        }
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_LIMIC_GATHER_START);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_LIMIC_Bcast_Complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_LIMIC_Gather_Complete(int rank, int comm_size, int shmem_comm_rank)
{
    int i = 1, cnt = 0, err = 0;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_LIMIC_COLL_GATHER_COMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_LIMIC_COLL_GATHER_COMPLETE);

    if (rank == 0) {
        for (; i < comm_size; ++i) {
            SHMEM_COLL_SYNC_SET(limic_progress, shmem_comm_rank, i);
            WRITEBAR();
        }

        /*Wait for all the non-leaders to complete */
        for (i = 1; i < comm_size; ++i) {
            READBAR();
            while (SHMEM_COLL_SYNC_ISSET(limic_progress, shmem_comm_rank, i)) {
#if defined(CKPT)
                Wait_for_CR_Completion();
#endif
                MPIDI_CH3I_Progress_test();
                /* Yield once in a while */
                MPIU_THREAD_CHECK_BEGIN++ cnt;
                if (cnt >= 20) {
                    cnt = 0;
#if defined(CKPT)
                    MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
                        do {
                    } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                    MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                    MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
                        MPIDI_CH3I_CR_lock();
#endif
                }
                MPIU_THREAD_CHECK_END
                READBAR();
            }
        }

    } else {
        SHMEM_COLL_SYNC_CLR(limic_progress, shmem_comm_rank, rank);
        WRITEBAR();
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_LIMIC_COLL_GATHER_COMPLETE);
    return (mpi_errno);
}

int MPIR_Limic_Gather_OSU(void *recvbuf,
                          int recvbytes,
                          void *sendbuf, int sendbytes, MPID_Comm * shmem_comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Comm shmem_comm;
    int shmem_comm_rank;
    int local_rank, local_size;
    int offset;
    limic_user local_limic_hndl;

    shmem_comm = shmem_comm_ptr->handle;
    shmem_comm_rank = shmem_comm_ptr->dev.ch.shmem_comm_rank;   /*shmem_comm_rank; */

    mpi_errno = PMPI_Comm_rank(shmem_comm, &local_rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (local_size == 1 || sendbytes == 0 || recvbytes == 0) {
        return MPI_SUCCESS;
    }

    mpi_errno = MPIDI_CH3I_LIMIC_Gather_Start(recvbuf, recvbytes, sendbytes,
                                              local_size, local_rank,
                                              shmem_comm_rank, shmem_comm_ptr);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (local_rank == 0) {
        mpi_errno = MPIDI_CH3I_LIMIC_Gather_Complete(local_rank, local_size,
                                                     shmem_comm_rank);
    } else {
        int err = 0;
        local_limic_hndl = shmem_coll->limic_hndl[shmem_comm_rank];

        offset = shmem_coll->leader_buf_offset[shmem_comm_rank] +
            (sendbytes * (local_rank - 1));

        local_limic_hndl.offset = offset;

        err = limic_tx_comp(limic_fd, (void *) sendbuf, sendbytes, &(local_limic_hndl));
        if (!err) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                      "**fail", "**fail %s",
                                      "LiMIC: (MPIR_Limic_Gather_OSU) limic_tx_comp fail");
        }
        if (err != sendbytes) {
            printf("MPIR_Gather, expecting %d bytes, got %d bytes \n", sendbytes, err);
        }
        mpi_errno = MPIDI_CH3I_LIMIC_Gather_Complete(local_rank, local_size,
                                                     shmem_comm_rank);

        local_limic_hndl.offset = shmem_coll->limic_hndl[shmem_comm_rank].offset;
        local_limic_hndl.length = shmem_coll->limic_hndl[shmem_comm_rank].length;
    }
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
  fn_fail:
    return mpi_errno;

}
#endif                          /*#if defined(_SMP_LIMIC_) */

static inline void mv2_shm_progress(int *nspin)
{
    int err = 0;
#if defined(CKPT)
    Wait_for_CR_Completion();
#endif
    MV2_INC_NUM_UNEXP_RECV();
    MPID_Progress_test();
    MV2_DEC_NUM_UNEXP_RECV();
    /* Yield once in a while */
    MPIU_THREAD_CHECK_BEGIN if (*nspin % 20 == 0) {
#if defined(CKPT)
        MPIDI_CH3I_CR_unlock();
#endif
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
        MPIU_THREAD_CHECK_BEGIN MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
        MPIU_THREAD_CHECK_END
#endif
            do {
        } while (0);
#if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
        MPIU_THREAD_CHECK_BEGIN MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
        MPIU_THREAD_CHECK_END
#endif
#if defined(CKPT)
            MPIDI_CH3I_CR_lock();
#endif
    }
MPIU_THREAD_CHECK_END}

void mv2_shm_barrier(shmem_info_t * shmem)
{
    int i, nspin = 0;
    MPIU_Assert(shmem->write == shmem->read);
    int idx = shmem->read % mv2_shm_window_size;
    if (shmem->local_rank == 0) {
        for (i = 1; i < shmem->local_size; i++) {
            while (shmem->queue[i].shm_slots[idx]->psn != shmem->read) {
                nspin++;
                if (nspin % mv2_shmem_coll_spin_count == 0) {
                    mv2_shm_progress(&nspin);
                }
            }
            READBAR();
        }
        WRITEBAR();
        shmem->queue[0].shm_slots[idx]->psn = shmem->write;
    } else {
        shmem->queue[shmem->local_rank].shm_slots[idx]->psn = shmem->write;
        while (shmem->queue[0].shm_slots[idx]->psn != shmem->read) {
            nspin++;
            if (nspin % mv2_shmem_coll_spin_count == 0) {
                mv2_shm_progress(&nspin);
            }
        }
        READBAR();
    }
    shmem->write++;
    shmem->read++;
}



void mv2_shm_reduce(shmem_info_t * shmem, char *in_buf, int len,
                    int count, int root, MPI_User_function * uop, MPI_Datatype datatype, 
                    int is_cxx_uop) 
{
    char *buf=NULL; 
    int i, nspin = 0;
    int windex = shmem->write % mv2_shm_window_size;
    int rindex = shmem->read % mv2_shm_window_size;

    /* Copy the data from the input buffer to the shm slot. This is to
     * ensure that we don't mess with the sendbuf supplied 
     * by the application */
#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        MPIR_Localcopy(in_buf, len, MPI_BYTE,
            shmem->queue[shmem->local_rank].shm_slots[windex]->buf, len, MPI_BYTE);
    } else
#endif
    {
        MPIU_Memcpy(shmem->queue[shmem->local_rank].shm_slots[windex]->buf, in_buf, len);
    }

    buf = shmem->queue[shmem->local_rank].shm_slots[windex]->buf;

    if (shmem->local_rank == root) {
        for (i = 1; i < shmem->local_size; i++) {
            while (shmem->queue[i].shm_slots[rindex]->psn != shmem->read) {
                nspin++;
                if (nspin % mv2_shmem_coll_spin_count == 0) {
                    mv2_shm_progress(&nspin);
                }
            }
            READBAR();
#ifdef HAVE_CXX_BINDING
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (
                    shmem->queue[i].shm_slots[rindex]->buf, buf,
                    count, datatype, uop);
            } else
#endif
            (*uop) (shmem->queue[i].shm_slots[rindex]->buf, buf, &count, &datatype);
        }
    } else {
        WRITEBAR();
        shmem->queue[shmem->local_rank].shm_slots[windex]->psn = shmem->write;
    }
}

void mv2_shm_tree_reduce(shmem_info_t * shmem, char *in_buf, int len,
                    int count, int root, MPI_User_function * uop, MPI_Datatype datatype, 
                    int is_cxx_uop)
{
    void *buf=NULL; 
    int i, nspin = 0;
    int start=0, end=0;
    int windex = shmem->write % mv2_shm_window_size;
    int rindex = shmem->read % mv2_shm_window_size;

    if (shmem->local_rank == root || shmem->local_rank % mv2_shm_reduce_tree_degree == 0) {
        start = shmem->local_rank;
        end   = shmem->local_rank +  mv2_shm_reduce_tree_degree;
        if(end > shmem->local_size) end = shmem->local_size;

        /* Copy the data from the input buffer to the shm slot. This is to ensure
        * that we don't mess with the sendbuf supplied by the application */ 
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda) {
            MPIR_Localcopy(in_buf, len, MPI_BYTE,
                shmem->queue[shmem->local_rank].shm_slots[windex]->buf, len, MPI_BYTE);
        } else
#endif
        {
            MPIU_Memcpy(shmem->queue[shmem->local_rank].shm_slots[windex]->buf, in_buf, len);
        }

        buf = shmem->queue[shmem->local_rank].shm_slots[windex]->buf;


        for(i=start + 1; i<end; i++) {
            while (shmem->queue[i].shm_slots[rindex]->psn != shmem->read) {
                nspin++;
                if (nspin % mv2_shmem_coll_spin_count == 0) {
                    mv2_shm_progress(&nspin);
                }
            }
            READBAR();
#ifdef HAVE_CXX_BINDING
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (
                    shmem->queue[i].shm_slots[rindex]->buf, buf, 
                    count, datatype, uop);
            } else
#endif
            (*uop) (shmem->queue[i].shm_slots[rindex]->buf, buf, &count, &datatype);
        }

        if(shmem->local_rank == root) {
            for (i = mv2_shm_reduce_tree_degree; i < shmem->local_size; 
                 i = i + mv2_shm_reduce_tree_degree) {
                while (shmem->queue[i].shm_slots[rindex]->psn != shmem->read) {
                    nspin++;
                    if (nspin % mv2_shmem_coll_spin_count == 0) {
                        mv2_shm_progress(&nspin);
                    }
                }
                READBAR();
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (
                        shmem->queue[i].shm_slots[rindex]->buf, buf,
                        count, datatype, uop);
                } else
#endif

                (*uop) (shmem->queue[i].shm_slots[rindex]->buf, buf, &count, &datatype);
            }
        } else {
            WRITEBAR();
            shmem->queue[shmem->local_rank].shm_slots[windex]->psn = shmem->write;
        }

    } else {
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda) {
            MPIR_Localcopy(in_buf, len, MPI_BYTE,
                shmem->queue[shmem->local_rank].shm_slots[windex]->buf, len, MPI_BYTE);
        } else
#endif
        {
            MPIU_Memcpy(shmem->queue[shmem->local_rank].shm_slots[windex]->buf, in_buf, len);
        }

        WRITEBAR();
        shmem->queue[shmem->local_rank].shm_slots[windex]->psn = shmem->write;
    }
} 

int mv2_shm_bcast(shmem_info_t * shmem, char *buf, int len, int root)
{
    int mpi_errno = MPI_SUCCESS; 
    int nspin = 0;
    int windex = -1, rindex=-1; 
    MPID_Comm *shmem_commptr = NULL, *comm_ptr=NULL;

    MPID_Comm_get_ptr(shmem->comm, comm_ptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_commptr);
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    int intra_node_root = 0;
    MPID_Comm *leader_commptr = NULL;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_commptr);
#endif
    shmem  = shmem_commptr->dev.ch.shmem_info; 
    windex = shmem->write % mv2_shm_window_size;
    rindex = shmem->read % mv2_shm_window_size;

    if(shmem->local_size > 0) { 
        if (shmem->local_rank == root) {
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda) { 
                MPIR_Localcopy(buf, len, MPI_BYTE, 
                    shmem->queue[root].shm_slots[windex]->buf, len, MPI_BYTE);
            } else
#endif
            {
                MPIU_Memcpy(shmem->queue[root].shm_slots[windex]->buf, buf, len);
            }
            WRITEBAR();
            shmem->queue[root].shm_slots[windex]->psn = shmem->write;
        } else {
            while (shmem->queue[root].shm_slots[rindex]->psn != shmem->read) {
                nspin++;
                if (nspin % mv2_shmem_coll_spin_count == 0) {
                    mv2_shm_progress(&nspin);
                }
            }
            READBAR();
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda) { 
                MPIR_Localcopy(shmem->queue[root].shm_slots[rindex]->buf, len, MPI_BYTE, 
                                buf, len, MPI_BYTE);
            } else 
#endif
            {
                MPIU_Memcpy(buf, shmem->queue[root].shm_slots[rindex]->buf, len);
            }
        }
    } 
    shmem->write++;
    shmem->read++;
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    if (shmem->half_full_complete == 0 &&
        IS_SHMEM_WINDOW_HALF_FULL(shmem->write, shmem->tail)) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window half full: %llu \n", shmem->write);
        mv2_shm_barrier(shmem);
        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {

            if(shmem->end_request_active == 1){ 
                mpi_errno = mv2_flush_zcpy_end_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 

            if(shmem->mid_request_active == 0){ 
                mpi_errno = mv2_post_zcpy_mid_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 
        } 
        shmem->half_full_complete = 1;
    } 

    if (IS_SHMEM_WINDOW_FULL(shmem->write, shmem->tail)) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window full: %llu \n", shmem->write);
        mv2_shm_barrier(shmem);
        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {

            if(shmem->mid_request_active == 1){ 
                mpi_errno = mv2_flush_zcpy_mid_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }

            if(shmem->end_request_active == 0){ 
                mpi_errno = mv2_post_zcpy_end_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
        }
        shmem->tail = shmem->read;
        shmem->half_full_complete = 0;
    }
#else /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */
    if (IS_SHMEM_WINDOW_FULL(shmem->write, shmem->tail)) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window full: %llu \n", shmem->write);
        mv2_shm_barrier(shmem);
        shmem->tail = shmem->read;
    }
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

fn_exit:
    return mpi_errno;
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
fn_fail:
#endif
    goto fn_exit;
}


#ifdef CHANNEL_MRAIL_GEN2
int mv2_shm_zcpy_bcast(shmem_info_t * shmem, char *buf, int len, int root,
                       int src, int expected_recv_count,
                       int *dst_array, int expected_send_count,
                       int knomial_degree,
                       MPID_Comm *comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int nspin = 0, i=0, intra_node_root=0;
    int windex = shmem->write % mv2_shm_window_size;
    int rindex = shmem->read % mv2_shm_window_size;
    MPIDI_VC_t *vc=NULL;
    MPID_Comm *shmem_commptr=NULL, *leader_commptr=NULL;
    shm_coll_pkt pkt;
    shm_coll_pkt *remote_handle_info_parent = shmem->bcast_remote_handle_info_parent;
    shm_coll_pkt *remote_handle_info_children = shmem->bcast_remote_handle_info_children;

    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_commptr);

    if(shmem->buffer_registered == 0) {
       if(shmem_commptr->rank == 0 && leader_commptr->local_size > 1) {
#ifdef CKPT
           shmem->prev = NULL;
           shmem->next = NULL;
           if (ckpt_free_head) {
               ckpt_free_head->prev = shmem;
               shmem->next = ckpt_free_head;
           }
           ckpt_free_head = shmem;
           PRINT_DEBUG(DEBUG_CR_verbose > 1,
                   "Adding shmem region=%p, prev=%p, next=%p\n",
                   shmem, shmem->prev, shmem->next);
#endif /* CKPT */
           mpi_errno = mv2_shm_coll_reg_buffer(shmem->buffer, shmem->size,
                                               shmem->mem_handle, &(shmem->buffer_registered));
           if(mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
           }
       }
    }

    if (shmem->local_rank == intra_node_root) {
        MPID_Comm *leader_commptr=NULL;
        MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_commptr);

        if(leader_commptr->local_size > 1) {
            /*Exchange keys, (if needed) */
            if(shmem->bcast_exchange_rdma_keys == 1) {
                if(remote_handle_info_parent != NULL) {
                    MPIU_Free(remote_handle_info_parent);
                    remote_handle_info_parent = NULL;
                }
                if(remote_handle_info_children != NULL) {
                    MPIU_Free(remote_handle_info_children);
                    remote_handle_info_children = NULL;
                }
                if(expected_recv_count > 0) {
                    remote_handle_info_parent = MPIU_Malloc(sizeof(shm_coll_pkt)*1);
                    i=0;
                    do {
                       pkt.key[i]  =  shmem->mem_handle[i]->rkey;
                       pkt.addr[i] =  (uintptr_t) (shmem->buffer);
                       i++;
                    } while( i < rdma_num_hcas);

                    mpi_errno = MPIC_Send(&pkt, sizeof(shm_coll_pkt), MPI_BYTE, src,
                                             MPIR_BCAST_TAG, leader_commptr, &errflag);
                    if (mpi_errno) {
                                MPIR_ERR_POP(mpi_errno);
                    }
                    remote_handle_info_parent->peer_rank = src;
                }

                if(expected_send_count > 0) {
                    remote_handle_info_children = MPIU_Malloc(sizeof(shm_coll_pkt)*expected_send_count);
                }

                for(i=0 ; i< expected_send_count; i++) {
                    /* Sending to dst. Receive its key info */
                    int j=0;
                    mpi_errno = MPIC_Recv(&pkt, sizeof(shm_coll_pkt), MPI_BYTE, dst_array[i],
                                              MPIR_BCAST_TAG, leader_commptr, MPI_STATUS_IGNORE, &errflag);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }

                    do {
                        remote_handle_info_children[i].key[j] = pkt.key[j];
                        remote_handle_info_children[i].addr[j] = pkt.addr[j];
                        j++;
                    } while( j < rdma_num_hcas);
                    remote_handle_info_children[i].peer_rank = dst_array[i];
                }

                /* Trac #657: Wait till VC is ready to send */
                for(i = 0; i < expected_send_count; i++) {
                    MPIDI_Comm_get_vc(leader_commptr, remote_handle_info_children[i].peer_rank, &vc);
                    while (VC_NOT_READY(vc)) {
                        /* Yield CPU and wait for CM thread to complete connection */
                        MPIU_PW_Sched_yield();
                    }
                }

                shmem->bcast_exchange_rdma_keys          = 0;
                shmem->bcast_knomial_factor              = knomial_degree;
                shmem->bcast_remote_handle_info_parent   = remote_handle_info_parent;
                shmem->bcast_remote_handle_info_children = remote_handle_info_children;
                shmem->bcast_expected_send_count         = expected_send_count;
            }

            /********************************************
             * the RDMA keys for the shmem buffer has been exchanged
             * We are ready to move the data around
              **************************************/
            if(leader_commptr->rank != root) {
                /* node-level leader, but not the root of the bcast */
                shmem->queue[intra_node_root].shm_slots[windex]->tail_psn =  (volatile uint32_t *)
                                 ((char *) (shmem->queue[intra_node_root].shm_slots[windex]->buf) + len);
                /* Wait until the peer is yet to update the psn flag and 
                 * until the psn and the tail flags have the same values*/
                while(shmem->write !=
                        (volatile uint32_t ) *(shmem->queue[intra_node_root].shm_slots[windex]->tail_psn)) {
                    mv2_shm_progress(&nspin);
                }
                WRITEBAR();
                shmem->queue[intra_node_root].shm_slots[windex]->psn = shmem->write;
            } else {
                /* node-level leader, and the root of the bcast */
#if defined(_ENABLE_CUDA_)
                if (rdma_enable_cuda) {
                    MPIR_Localcopy(buf, len, MPI_BYTE,
                            shmem->queue[intra_node_root].shm_slots[windex]->buf, len, MPI_BYTE);
                } else
#endif
                {

                    MPIU_Memcpy(shmem->queue[intra_node_root].shm_slots[windex]->buf, buf, len);
                }
                WRITEBAR();
                shmem->queue[intra_node_root].shm_slots[windex]->psn = shmem->write;
                WRITEBAR();
                shmem->queue[intra_node_root].shm_slots[windex]->tail_psn = (volatile uint32_t *)
                            ((char *) (shmem->queue[intra_node_root].shm_slots[windex]->buf) + len);
                *((volatile uint32_t *) shmem->queue[intra_node_root].shm_slots[windex]->tail_psn) =
                                  shmem->write;
            }

            /* Post the rdma-writes to all the children in the tree */
            for(i=0; i< shmem->bcast_expected_send_count; i++) {
                uint32_t local_rdma_key, remote_rdma_key;
                uint64_t local_rdma_addr, remote_rdma_addr, offset;
                int rail=0, hca_num;

                MPIDI_Comm_get_vc(leader_commptr, remote_handle_info_children[i].peer_rank, &vc);
                offset= ((uintptr_t ) (shmem->queue[intra_node_root].shm_slots[windex]->buf) -
                          (uintptr_t ) (shmem->buffer));
#ifdef CHANNEL_NEMESIS_IB
                rail = MPIDI_nem_ib_send_select_rail(vc);
#else
                rail = MRAILI_Send_select_rail(vc);
#endif
                hca_num = rail / (rdma_num_rails / rdma_num_hcas);

                local_rdma_addr  =  (uint64_t) (shmem->queue[intra_node_root].shm_slots[windex]->buf);
                local_rdma_key   = (shmem->mem_handle[hca_num])->lkey;
                remote_rdma_addr =  (uint64_t) remote_handle_info_children[i].addr[hca_num] + offset;
                remote_rdma_key  = remote_handle_info_children[i].key[hca_num];

                mv2_shm_coll_prepare_post_send(local_rdma_addr, remote_rdma_addr, 
                                  local_rdma_key, remote_rdma_key,
                                  len + sizeof(volatile uint32_t), rail,  vc);
            }

            /* If a node-leader is not root, we finally copy the data back into 
             * the user-level buffer */
            if(leader_commptr->rank != root) {
#if defined(_ENABLE_CUDA_)
                if (rdma_enable_cuda) {
                    mpi_errno = MPIR_Localcopy(shmem->queue[intra_node_root].shm_slots[windex]->buf,
                                   len, MPI_BYTE,
                                   buf, len, MPI_BYTE);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                } else
#endif
                {
                    MPIU_Memcpy(buf, shmem->queue[intra_node_root].shm_slots[windex]->buf, len);
                }
            }
        } else {
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda) {
                mpi_errno = MPIR_Localcopy(buf, len, MPI_BYTE,
                    shmem->queue[intra_node_root].shm_slots[windex]->buf, len, MPI_BYTE);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            } else
#endif
            {
                MPIU_Memcpy(shmem->queue[intra_node_root].shm_slots[windex]->buf, buf, len);
            }
            WRITEBAR();
            shmem->queue[intra_node_root].shm_slots[windex]->psn = shmem->write;
        }
    } else {
        while (shmem->queue[intra_node_root].shm_slots[rindex]->psn != shmem->read) {
            nspin++;
            if (nspin % mv2_shmem_coll_spin_count == 0) {
                mv2_shm_progress(&nspin);
            }
        }
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda) {
            mpi_errno = MPIR_Localcopy(shmem->queue[intra_node_root].shm_slots[rindex]->buf, len, MPI_BYTE,
                            buf, len, MPI_BYTE);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        } else
#endif
        {
            MPIU_Memcpy(buf, shmem->queue[intra_node_root].shm_slots[rindex]->buf, len);
        }
    }
    shmem->write++;
    shmem->read++;

    if (shmem->half_full_complete == 0 &&
        IS_SHMEM_WINDOW_HALF_FULL(shmem->write, shmem->tail)) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window half full: %llu \n", shmem->write);
        mv2_shm_barrier(shmem);
        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {
            
            if(shmem->end_request_active == 1){ 
                mpi_errno = mv2_flush_zcpy_end_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 

            if(shmem->mid_request_active == 0){ 
                mpi_errno = mv2_post_zcpy_mid_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 
        } 
        shmem->half_full_complete = 1; 
    } 

    if (IS_SHMEM_WINDOW_FULL(shmem->write, shmem->tail)) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window full: %llu \n", shmem->write);
        mv2_shm_barrier(shmem);
        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {

            if(shmem->mid_request_active == 1){ 
                mpi_errno = mv2_flush_zcpy_mid_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 

            if(shmem->end_request_active == 0){ 
                mpi_errno = mv2_post_zcpy_end_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 
        }
        shmem->tail = shmem->read;
        shmem->half_full_complete = 0; 
    }

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int mv2_shm_zcpy_reduce(shmem_info_t * shmem, 
                         void *in_buf, void **out_buf, 
                         int count, int len, 
                         MPI_Datatype datatype, 
                         MPI_Op op, int root, 
                         int expected_recv_count, int *src_array,
                         int expected_send_count, int dst,
                         int knomial_degree,
                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    void *buf=NULL; 
    int mpi_errno = MPI_SUCCESS;
    int nspin = 0, i=0, j=0, intra_node_root=0;
    int windex = 0, rindex=0;
    MPIDI_VC_t *vc=NULL;
    MPID_Comm *shmem_commptr=NULL, *leader_commptr=NULL;
    shm_coll_pkt pkt;
    shm_coll_pkt *remote_handle_info_parent = shmem->reduce_remote_handle_info_parent;
    shm_coll_pkt *remote_handle_info_children = shmem->reduce_remote_handle_info_children;
    int *inter_node_reduce_status_array =  shmem->inter_node_reduce_status_array;  
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    int slot_len; 
    slot_len = mv2_shm_slot_len + sizeof (shm_slot_t) + sizeof(volatile uint32_t);
    MV2_SHM_ALIGN_LEN(slot_len, MV2_SHM_ALIGN); 
    int is_cxx_uop = 0;
    int shmem_comm_rank, local_size, local_rank; 

    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_commptr);

    shmem_comm_rank = shmem_commptr->dev.ch.shmem_comm_rank; 
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    windex = shmem->write % mv2_shm_window_size;
    rindex = shmem->read % mv2_shm_window_size;

    if (shmem->half_full_complete == 0 && 
        (IS_SHMEM_WINDOW_HALF_FULL(shmem->write, shmem->tail))) { 
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window half full: %llu \n", shmem->write);

        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {
            
            if(shmem->mid_request_active == 0){
                mpi_errno = mv2_post_zcpy_mid_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            } 

            if(shmem->end_request_active == 1){
                mpi_errno = mv2_flush_zcpy_end_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
     
        }
        shmem->write++;
        shmem->read++;
        shmem->half_full_complete = 1; 
        windex = shmem->write % mv2_shm_window_size;
        rindex = shmem->read % mv2_shm_window_size;
    }

    if (IS_SHMEM_WINDOW_FULL(shmem->write, shmem->tail)   ||
       ((mv2_shm_window_size - windex < 2) || (mv2_shm_window_size - rindex < 2))) {
        PRINT_DEBUG(DEBUG_SHM_verbose > 1, "shmem window full: %llu \n", shmem->write);
        if (shmem->local_size > 1) {
             MPIDI_CH3I_SHMEM_COLL_Barrier_gather(local_size, local_rank,
                                             shmem_comm_rank);
        } 

        if (shmem->local_rank == intra_node_root &&
            leader_commptr->local_size > 1) {
            if(shmem->end_request_active == 0){
                mpi_errno = mv2_post_zcpy_end_request(leader_commptr, shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
            
            
            if(shmem->mid_request_active == 1){
                mpi_errno = mv2_flush_zcpy_mid_request(shmem);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }

        } 
  
        if (shmem->local_size > 1) {
             MPIDI_CH3I_SHMEM_COLL_Barrier_bcast(local_size, local_rank,
                                            shmem_comm_rank);
        }
        shmem->half_full_complete = 0; 
        shmem->write++;
        shmem->read++;
        shmem->tail = shmem->read;
        windex = shmem->write % mv2_shm_window_size;
        rindex = shmem->read % mv2_shm_window_size;
        
    }
    
    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            } else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
            }
#if defined(HAVE_CXX_BINDING)
        }
#endif                          /* defined(HAVE_CXX_BINDING) */
    }


   if (shmem->local_rank == intra_node_root) {
        MPID_Comm *leader_commptr=NULL;
        int inter_node_reduce_completions=0; 
        MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_commptr);
    
        if(leader_commptr->local_size > 1) {
            if(shmem->buffer_registered == 0) {
#ifdef CKPT
               shmem->prev = NULL;
               shmem->next = NULL;
               if (ckpt_free_head) {
                   ckpt_free_head->prev = shmem;
                   shmem->next = ckpt_free_head;
               }
               ckpt_free_head = shmem;
               PRINT_DEBUG(DEBUG_CR_verbose > 1,
                       "Adding shmem region=%p, prev=%p, next=%p\n",
                       shmem, shmem->prev, shmem->next);
#endif /* CKPT */

                mpi_errno = mv2_shm_coll_reg_buffer(shmem->buffer, shmem->size,
                                                   shmem->mem_handle, &(shmem->buffer_registered));
                if(mpi_errno) {   
                    MPIR_ERR_POP(mpi_errno);
                }                 
            }

            /*Exchange keys, (if needed) */
            if(shmem->reduce_exchange_rdma_keys == 1) {
                if(remote_handle_info_parent != NULL) {
                    MPIU_Free(remote_handle_info_parent);
                    remote_handle_info_parent = NULL;
                }
                if(remote_handle_info_children != NULL) {
                    MPIU_Free(remote_handle_info_children);
                    remote_handle_info_children = NULL;
                }
               if( shmem->inter_node_reduce_status_array != NULL ) { 
                    MPIU_Free(shmem->inter_node_reduce_status_array); 
                    shmem->inter_node_reduce_status_array = NULL; 
                } 
 
               if(expected_send_count > 0) { 
                    MPI_Status status; 
                    remote_handle_info_parent = MPIU_Malloc(sizeof(shm_coll_pkt)*
                                                  expected_send_count);
                    if (remote_handle_info_parent == NULL) {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                                  "**nomem %s", "mv2_shmem_file");
                    }
                    j=0;

                    /* I am sending data to "dst". I will be needing dst's 
                     * RDMA info */ 
                    mpi_errno = MPIC_Recv(&pkt, sizeof(shm_coll_pkt), MPI_BYTE, dst,
                                          MPIR_REDUCE_TAG, leader_commptr,
                                          &status, errflag);
                    if (mpi_errno) {
                         MPIR_ERR_POP(mpi_errno);
                    }

                   do {
                        remote_handle_info_parent->key[j] = pkt.key[j];
                        remote_handle_info_parent->addr[j] = pkt.addr[j];
                        j++;
                    } while( j < rdma_num_hcas);
                    remote_handle_info_parent->peer_rank = dst;
                    remote_handle_info_parent->recv_id   = pkt.recv_id; 

                    /* Trac #657: Wait till VC is ready to send */
                    MPIDI_Comm_get_vc(leader_commptr, remote_handle_info_parent->peer_rank, &vc);
                    while (VC_NOT_READY(vc)) {
                        /* Yield CPU and wait for CM thread to complete connection */
                        MPIU_PW_Sched_yield();
                    }
               } 
 
               if(expected_recv_count > 0) { 
                    int j=0; 
                    MPID_Request **request_array = NULL; 
                    MPI_Status *status_array = NULL; 
                    shm_coll_pkt *pkt_array = NULL; 
                    remote_handle_info_children = MPIU_Malloc(sizeof(shm_coll_pkt)*
                                                  expected_recv_count);

                    pkt_array = MPIU_Malloc(sizeof(shm_coll_pkt)*expected_recv_count); 
                    request_array = (MPID_Request **) MPIU_Malloc(sizeof(MPID_Request*)*expected_recv_count); 
                    status_array = MPIU_Malloc(sizeof(MPI_Status)*expected_recv_count); 

                    if (pkt_array == NULL || request_array == NULL || 
                        status_array == NULL || remote_handle_info_children == NULL) {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                                  "**nomem %s", "mv2_shmem_file");
                    }

                    for(i=0 ; i< expected_recv_count; i++) {
                        j=0; 
                        do {
                            pkt_array[i].key[j]  =  shmem->mem_handle[j]->rkey;
                            pkt_array[i].addr[j] =  (uintptr_t) (shmem->buffer);
                            j++;
                        } while( j < rdma_num_hcas);
                    } 

                    for(i=0 ; i< expected_recv_count; i++) {
                        int src=-1; 
                        /* Receiving data from src. Send my key info to src*/
                        src = src_array[i]; 
                        pkt_array[i].recv_id =  i;
                        mpi_errno = MPIC_Isend(&pkt_array[i], sizeof(shm_coll_pkt), MPI_BYTE, src,
                                             MPIR_REDUCE_TAG, leader_commptr,
                                             &request_array[i], errflag);
                        if (mpi_errno) {
                                MPIR_ERR_POP(mpi_errno);
                        }
                        remote_handle_info_children[i].peer_rank = src_array[i];
                    }
                    mpi_errno = MPIC_Waitall(expected_recv_count, request_array, 
                                                status_array, errflag); 
                    if (mpi_errno) {
                          MPIR_ERR_POP(mpi_errno);
                    }
                    MPIU_Free(request_array); 
                    MPIU_Free(status_array); 
                    MPIU_Free(pkt_array); 
               } 
               shmem->reduce_exchange_rdma_keys          = 0;
               shmem->reduce_knomial_factor              = knomial_degree;
               shmem->reduce_remote_handle_info_parent   = remote_handle_info_parent;
               shmem->reduce_remote_handle_info_children = remote_handle_info_children;
               shmem->reduce_expected_recv_count         = expected_recv_count;
               shmem->reduce_expected_send_count         = expected_send_count;
               if(shmem->reduce_expected_recv_count > 0) { 
                   inter_node_reduce_status_array = MPIU_Malloc(sizeof(int)*
                                                    shmem->reduce_expected_recv_count); 
                    if (inter_node_reduce_status_array == NULL) {  
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                                  "**nomem %s", "mv2_shmem_file");
                    }
                   MPIU_Memset(inter_node_reduce_status_array, '0', 
                               sizeof(int)*shmem->reduce_expected_recv_count); 
                   shmem->inter_node_reduce_status_array = inter_node_reduce_status_array; 
               } 
               if(src_array != NULL) { 
                  MPIU_Free(src_array); 
               }  
           }
       } 

       /********************************************
       * the RDMA keys for the shmem buffer has been exchanged
       * We are ready to move the data around
        **************************************/

      
       /* Lets start with intra-node, shmem, slot-based reduce*/ 
     
       buf = shmem->queue[shmem->local_rank].shm_slots[windex]->buf; 

       /* Now, do the intra-node shm-slot reduction */ 
       mv2_shm_tree_reduce(shmem, in_buf, len, count, intra_node_root, uop, 
                      datatype, is_cxx_uop); 

       WRITEBAR();
       shmem->queue[shmem->local_rank].shm_slots[windex]->psn = shmem->write;
       WRITEBAR();
       shmem->queue[shmem->local_rank].shm_slots[windex]->tail_psn = (volatile uint32_t *)
                             ((char *) (shmem->queue[shmem->local_rank].shm_slots[windex]->buf) + len);
        *((volatile uint32_t *) shmem->queue[shmem->local_rank].shm_slots[windex]->tail_psn) =
                                  shmem->write;


       /* I am my node's leader, if I am an intermediate process in the  tree, 
        * I need to wait until my inter-node peers have written their data
        */ 
   
       while(inter_node_reduce_completions <  shmem->reduce_expected_recv_count) { 
           for(i=0; i< shmem->reduce_expected_recv_count ; i++) { 
               /* Wait until the peer is yet to update the psn flag and 
                * until the psn and the tail flags have the same values*/
               if(inter_node_reduce_status_array[i] != 1) { 
                   shmem->queue[i].shm_slots[windex + 1]->tail_psn =  (volatile uint32_t *)
                                  ((char *) (shmem->queue[i].shm_slots[windex + 1]->buf) + len);
                   if(shmem->write ==
                           (volatile uint32_t ) *(shmem->queue[i].shm_slots[windex + 1]->tail_psn)) {
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (
                            shmem->queue[i].shm_slots[rindex+1]->buf, buf,
                            count, datatype, uop);
                    } else
#endif
                    {
                       (*uop) (shmem->queue[i].shm_slots[(rindex+1)]->buf, buf, &count, &datatype);
                    }
                    inter_node_reduce_completions++;
                    inter_node_reduce_status_array[i] = 1;
                   }
               } 
           } 
           if (nspin % mv2_shmem_coll_spin_count == 0) {
               mv2_shm_progress(&nspin);
           } 
           nspin++; 
       } 

       /* Post the rdma-write to the parent in the tree */
       if(shmem->reduce_expected_send_count > 0) {
            uint32_t local_rdma_key, remote_rdma_key;
            uint64_t local_rdma_addr, remote_rdma_addr, offset;
            int rail=0, row_id=0, hca_num;

            MPIDI_Comm_get_vc(leader_commptr, remote_handle_info_parent->peer_rank, &vc);

            row_id = remote_handle_info_parent->recv_id; 
            offset = (slot_len*mv2_shm_window_size)*(row_id) 
                             + slot_len*(windex + 1) 
                             + sizeof(shm_slot_cntrl_t); 
#ifdef CHANNEL_NEMESIS_IB
            rail = MPIDI_nem_ib_send_select_rail(vc);
#else
            rail = MRAILI_Send_select_rail(vc);
#endif
            hca_num = rail / (rdma_num_rails / rdma_num_hcas);

            local_rdma_addr  =  (uint64_t) (shmem->queue[intra_node_root].shm_slots[windex]->buf);
            local_rdma_key   = (shmem->mem_handle[hca_num])->lkey;
            remote_rdma_addr =  (uint64_t) remote_handle_info_parent->addr[hca_num] + offset;
            remote_rdma_key  = remote_handle_info_parent->key[hca_num];

            mv2_shm_coll_prepare_post_send(local_rdma_addr, remote_rdma_addr,
                              local_rdma_key, remote_rdma_key,
                              len + sizeof(volatile uint32_t), rail,  vc);
       }
       if(shmem->reduce_expected_recv_count > 0) { 
           MPIU_Memset(inter_node_reduce_status_array, '0', 
                   sizeof(int)*shmem->reduce_expected_recv_count); 
       } 
    } else { 
       mv2_shm_tree_reduce(shmem, in_buf, len, count, intra_node_root, uop, 
                      datatype, is_cxx_uop); 
       WRITEBAR();
       shmem->queue[shmem->local_rank].shm_slots[windex]->psn = shmem->write;
    } 

    *out_buf = buf; 
    shmem->write = shmem->write + 2;
    shmem->read = shmem->read + 2;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;

}
#endif /* CHANNEL_MRAIL_GEN2 */


int mv2_reduce_knomial_trace(int root, int mv2_reduce_knomial_factor,
        MPID_Comm *comm_ptr, int *expected_send_count, 
        int *expected_recv_count)
{ 
    int mask=0x1, k, comm_size, rank, relative_rank, lroot=0;
    int recv_iter=0, send_iter=0, dst;

    rank      = comm_ptr->rank;
    comm_size = comm_ptr->local_size;

    lroot = root;
    relative_rank = (rank - lroot + comm_size) % comm_size;

    /* First compute to whom we need to send data */
    while (mask < comm_size) {
        if (relative_rank % (mv2_reduce_knomial_factor*mask)) {
            dst = relative_rank/(mv2_reduce_knomial_factor*mask)*
                (mv2_reduce_knomial_factor*mask)+root;
            if (dst >= comm_size) {
                dst -= comm_size;
            }
            send_iter++;
            break;
        }
        mask *= mv2_reduce_knomial_factor;
    }
    mask /= mv2_reduce_knomial_factor;

    /* Now compute how many children we have in the knomial-tree */
    while (mask > 0) {
        for(k=1;k<mv2_reduce_knomial_factor;k++) {
            if (relative_rank + mask*k < comm_size) {
                recv_iter++;
            }
        }
        mask /= mv2_reduce_knomial_factor;
    }

    *expected_recv_count = recv_iter; 
    *expected_send_count = send_iter; 
    return 0;
}
   
shmem_info_t *mv2_shm_coll_init(int id, int local_rank, int local_size, 
                                MPID_Comm *comm_ptr)
{
    int slot_len, i, k;
    int mpi_errno = MPI_SUCCESS;
    int root=0, expected_max_send_count=0, expected_max_recv_count=0; 
    size_t size;
    char *shmem_dir, *ptr;
    char s_hostname[SHMEM_COLL_HOSTNAME_LEN];
    struct stat file_status;
    shmem_info_t *shmem = NULL;
    int max_local_size = 0;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPID_Comm *shmem_ptr = NULL;
 
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr);

    shmem = MPIU_Malloc(sizeof (shmem_info_t));
    MPIU_Assert(shmem != NULL);

    shmem->count = mv2_shm_window_size;
    shmem->write = 1;
    shmem->read = 1;
    shmem->tail = 0;
    shmem->file_fd = -1;
    shmem->file_name = NULL;
    shmem->size = 0;
    shmem->buffer = NULL;
    shmem->local_rank = local_rank;
    shmem->local_size = local_size; 
    shmem->comm   = comm_ptr->handle; 
    shmem->max_local_size = 0;

    slot_len = mv2_shm_slot_len + sizeof (shm_slot_t) + sizeof(volatile uint32_t);
                                
    MV2_SHM_ALIGN_LEN(slot_len, MV2_SHM_ALIGN)

    mpi_errno = mv2_reduce_knomial_trace(root, mv2_reduce_zcopy_max_inter_knomial_factor,
                             comm_ptr, &expected_max_send_count,
                             &expected_max_recv_count); 
    if (local_size < expected_max_recv_count) { 
        local_size = expected_max_recv_count;
    } 

    /* Initialize max_local_size to local_size so that non-leader processes
     * get right value */
    max_local_size = local_size;
    mpi_errno = MPIR_Reduce_impl(&local_size, &max_local_size, 1, MPI_INT,
				 MPI_MAX, 0, shmem_ptr, &errflag);
    if (mpi_errno) {
      MPIR_ERR_POP(mpi_errno);
    }

    size = (shmem->count) * slot_len * max_local_size;

    MV2_SHM_ALIGN_LEN(size, MV2_SHM_ALIGN);

    if ((shmem_dir = getenv("MV2_SHMEM_DIR")) == NULL) {
        shmem_dir = "/dev/shm";
    }

    if (gethostname(s_hostname, sizeof (char) * SHMEM_COLL_HOSTNAME_LEN) < 0) {
        PRINT_ERROR("gethostname filed\n");
    }
    shmem->file_name =
        (char *) MPIU_Malloc(sizeof (char) * (strlen(shmem_dir) +
                                              SHMEM_COLL_HOSTNAME_LEN + 26 +
                                              PID_CHAR_LEN));
    if (!shmem->file_name) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s", "mv2_shmem_file");
    }

    MPIDI_PG_GetConnKVSname(&mv2_kvs_name);

    sprintf(shmem->file_name, "%s/slot_shmem-coll-%s-%s-%d-%d.tmp",
            shmem_dir, mv2_kvs_name, s_hostname, id, getuid());
    shmem->file_fd =
        open(shmem->file_name, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
    if (shmem->file_fd < 0) {
        PRINT_ERROR("shmem open failed for file:%s\n", shmem->file_name);
        goto cleanup_slot_shmem;
    }

    if (local_rank == 0) {
        if (ftruncate(shmem->file_fd, 0)) {
            PRINT_ERROR("ftruncate failed file:%s\n", shmem->file_name);
            goto cleanup_slot_shmem;
        }
        if (ftruncate(shmem->file_fd, size)) {
            PRINT_ERROR("ftruncate failed file:%s\n", shmem->file_name);
            goto cleanup_slot_shmem;
        }
    }

    do {
        if (fstat(shmem->file_fd, &file_status) != 0) {
            PRINT_ERROR("fstat failed. file:%s\n", shmem->file_name);
            goto cleanup_slot_shmem;
        }
        usleep(1);
    } while (file_status.st_size < size);

    shmem->buffer =
        mmap(0, size, (PROT_READ | PROT_WRITE), (MAP_SHARED), shmem->file_fd, 0);
    if (shmem->buffer == (void *) -1) {
        PRINT_ERROR("mmap failed. file:%s\n", shmem->file_name);
        goto cleanup_slot_shmem;
    }
    shmem->size = size;


    ptr = shmem->buffer;
    shmem->queue = (shm_queue_t *) MPIU_Malloc(sizeof (shm_queue_t *)
            * max_local_size);
    shmem->max_local_size = max_local_size;
    for (k = 0; k < max_local_size; k++) {
        shmem->queue[k].shm_slots = (shm_slot_t **)
            MPIU_Malloc(mv2_shm_window_size * sizeof (shm_slot_t *));
        for (i = 0; i < mv2_shm_window_size; i++) {
            shmem->queue[k].shm_slots[i] = (shm_slot_t *) ptr;
            ptr += slot_len;
        }
    }

#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    shmem->bcast_remote_handle_info_parent = NULL; 
    shmem->bcast_remote_handle_info_children = NULL; 
    shmem->bcast_knomial_factor = -1; 
    shmem->bcast_exchange_rdma_keys   = 1; 
    shmem->buffer_registered = 0;  
    shmem->bcast_expected_send_count = 0;  

    shmem->reduce_remote_handle_info_parent = NULL; 
    shmem->reduce_remote_handle_info_children = NULL; 
    shmem->inter_node_reduce_status_array = NULL; 

    shmem->reduce_knomial_factor = -1; 
    shmem->reduce_exchange_rdma_keys   = 1; 
    shmem->reduce_expected_send_count = 0;  
    shmem->reduce_expected_recv_count = 0;  
    shmem->mid_request_active=0;
    shmem->end_request_active=0;
    shmem->end_request = MPI_REQUEST_NULL;
    shmem->mid_request = MPI_REQUEST_NULL;
    shmem->half_full_complete = 0; 

    for ( k = 0 ; k < rdma_num_hcas; k++ ) {
        shmem->mem_handle[k] = NULL; 
    } 
#ifdef CKPT
    shmem->next = NULL;
#endif /* CKPT */
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

    mv2_shm_barrier(shmem);

    PRINT_DEBUG(DEBUG_SHM_verbose > 0 && local_rank == 0,
                "Created shm file:%s size:%d \n", shmem->file_name, shmem->size);
    /* unlink the shmem file */
    if (shmem->file_name != NULL) {
        unlink(shmem->file_name);
    }

  fn_exit:
    return shmem;
  cleanup_slot_shmem:
    mv2_shm_coll_cleanup(shmem);
  fn_fail:
    MPIU_Free(shmem);
    shmem = NULL;
    goto fn_exit;

}

void mv2_shm_coll_cleanup(shmem_info_t * shmem)
{
    int k;
    PRINT_DEBUG(DEBUG_SHM_verbose > 0, " Cleanup shmem file:%s fd:%d size:%d\n",
                shmem->file_name, shmem->file_fd, shmem->size);
    for (k = 0; k < shmem->max_local_size; k++) {
        MPIU_Free(shmem->queue[k].shm_slots);
    }
    
    MPIU_Free(shmem->queue);
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    int mpi_errno = MPI_SUCCESS;

    if (shmem->end_request_active) {
        mpi_errno = mv2_flush_zcpy_end_request(shmem);
        if (mpi_errno) PRINT_ERROR("MPIR_Wait_impl failure \n");
    }
    if (shmem->mid_request_active) {
        mpi_errno = mv2_flush_zcpy_mid_request(shmem);
        if (mpi_errno) PRINT_ERROR("MPIR_Wait_impl failure \n");
    }

    if(shmem->local_rank == 0) { 
        if(shmem->buffer_registered == 1) { 
#ifdef CKPT
            PRINT_DEBUG(DEBUG_CR_verbose > 1,
                    "Cleaning up shmem=%p, prev=%p, next=%p\n",
                    shmem, shmem->prev, shmem->next);
            if (shmem->prev) {
                shmem->prev->next = shmem->next;
            }
            if (shmem->next) {
                shmem->next->prev = shmem->prev;
            }
            if (shmem == ckpt_free_head) {
                ckpt_free_head = shmem->next;
            }
#endif /* CKPT */
            mv2_shm_coll_dereg_buffer(shmem->mem_handle); 
        } 
        shmem->buffer_registered = 0; 
    } 
    if(shmem->bcast_remote_handle_info_parent != NULL) { 
        MPIU_Free(shmem->bcast_remote_handle_info_parent); 
    } 
    if(shmem->bcast_remote_handle_info_children != NULL) { 
        MPIU_Free(shmem->bcast_remote_handle_info_children); 
    } 
    if(shmem->reduce_remote_handle_info_parent != NULL) { 
        MPIU_Free(shmem->reduce_remote_handle_info_parent); 
    } 
    if(shmem->reduce_remote_handle_info_children != NULL) { 
        MPIU_Free(shmem->reduce_remote_handle_info_children); 
    } 
    if( shmem->inter_node_reduce_status_array != NULL ) {
        MPIU_Free(shmem->inter_node_reduce_status_array);
    }
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */
    if (shmem->buffer != NULL) {
        munmap(shmem->buffer, shmem->size);
    }
    if (shmem->file_fd != -1) {
        close(shmem->file_fd);
    }
    if (shmem->file_name != NULL) {
        MPIU_Free(shmem->file_name);
        shmem->file_name = NULL;
    }
}

int mv2_set_bcast_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;
    
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_BCAST_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL, 
                          &mv2_bcast_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_bcast_coll_algo_handle,
                                      (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */ 
    if (read_value != -1 &&
       (getenv("MV2_INTRA_BCAST_TUNING") != NULL ||
        getenv("MV2_INTRA_BCAST_TUNING") != NULL ||
        getenv("MV2_USE_ZCOPY_BCAST") != NULL ||
        getenv("MV2_USE_PIPELINED_ZCPY_BCAST_KNOMIAL_FACTOR") != NULL)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for MPI_Bcast."
            " This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value > MV2_MAX_NUM_BCAST_FUNCS || read_value < 0 ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_BCAST_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d).\n",
            MV2_MAX_NUM_BCAST_FUNCS,
            read_value); 
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_FLAT_BINOMIAL:
            mv2_user_bcast_inter = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DOUBLING_ALLGATHER_FLAT:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_DOUBLING_ALLGATHER_FLAT;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_RING_ALLGATHER_FLAT:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_RING_ALLGATEHR_FLAT;
            mv2_tune_parameter = 1;
            break;
        case MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_2:
            mv2_enable_zcpy_bcast = KNOMIAL_BCAST_INTRA;
            mv2_user_bcast_inter = MV2_PIPELINED_BCAST_ZCPY_MV2;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            mv2_pipelined_zcpy_knomial_factor = 2;
            break;
        case MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_4:
            mv2_enable_zcpy_bcast = KNOMIAL_BCAST_INTRA;
            mv2_user_bcast_inter = MV2_PIPELINED_BCAST_ZCPY_MV2;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            mv2_pipelined_zcpy_knomial_factor = 4;
            break;
        case MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_8:
            mv2_enable_zcpy_bcast = KNOMIAL_BCAST_INTRA;
            mv2_user_bcast_inter = MV2_PIPELINED_BCAST_ZCPY_MV2;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            mv2_pipelined_zcpy_knomial_factor = 8;
            break;
        case MV2_PIPELINED_BCAST_KNOMIAL_INTRA:
            mv2_user_bcast_inter = MV2_PIPELINED_BCAST_MV2;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_BINOMIAL_KNOMIAL_INTRA:
            mv2_user_bcast_inter = MV2_BCAST_BIONOMIAL_INTRA;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DOUBLING_ALLGATHER_KNOMIAL_INTRA_1:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_DOUBLING_ALLGATHER;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_RING_ALLGATHER_KNOMIAL_INTRA_1:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_RING_ALLGATHER;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DOUBLING_ALLGATHER_KNOMIAL_INTRA_2:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_RING_ALLGATHER_SHM;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_RING_ALLGATHER_KNOMIAL_INTRA_2:
            mv2_user_bcast_inter = MV2_KNOMIAL_BCAST_INTER_NODE_WRAPPER;
            mv2_user_bcast_intra = MV2_KNOMIAL_BCAST_INTRA;
            mv2_tune_parameter = 1;
            break;
        case MV2_PIPELINED_BCAST_SHMEM_INTRA:
            mv2_user_bcast_inter = MV2_PIPELINED_BCAST_MV2;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        case MV2_BINOMIAL_SHMEM_INTRA:
            mv2_user_bcast_inter = MV2_BCAST_BIONOMIAL_INTRA;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        case MV2_SCATTER_DOUBLING_ALLGATHER_SHMEM_INTRA_1:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_DOUBLING_ALLGATHER;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        case MV2_SCATTER_RING_ALLGATHER_SHMEM_INTRA_1:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_RING_ALLGATHER;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        case MV2_SCATTER_DOUBLING_ALLGATHER_SHMEM_INTRA_2:
            mv2_user_bcast_inter = MV2_BCAST_SCATTER_RING_ALLGATHER_SHM;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        case MV2_SCATTER_RING_ALLGATHER_SHMEM_INTRA_2:
            mv2_user_bcast_inter = MV2_KNOMIAL_BCAST_INTER_NODE_WRAPPER;
            mv2_user_bcast_intra = MV2_SHMEM_BCAST_INTRA;
            mv2_tune_parameter = 1;    
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of CVAR"
                " MPIR_CVAR_BCAST_COLLECTIVE_ALGORITHM is not valid. Valid "
                " values are natural numbers less than %d.\n",
                 MV2_MAX_NUM_BCAST_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_scatter_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_SCATTER_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL, 
                    &mv2_scatter_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_scatter_coll_algo_handle,
                                         (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (read_value != -1 &&
       (getenv("MV2_INTER_SCATTER_TUNING") != NULL || 
        getenv("MV2_INTRA_SCATTER_TUNING") != NULL)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for"
            " MPI_Scatter. This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value >= MV2_MAX_NUM_SCATTER_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_SCATTER_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_SCATTER_FUNCS,
            read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_SCATTER_BINOMIAL: 
            mv2_user_scatter_inter = MV2_SCATTER_BINOMIAL_RHS;
            mv2_tune_parameter = 1;			
            break;
        case MV2_SCATTER_DIRECT: 
            mv2_user_scatter_inter = MV2_SCATTER_DIRECT_RHS;
            mv2_tune_parameter = 1;			
            break;
        case MV2_SCATTER_BINOMIAL_AND_BINOMIAL_INTRA: 
            mv2_user_scatter_inter = MV2_SCATTER_TWO_LEVEL_BINOMIAL;
            mv2_user_scatter_intra = MV2_SCATTER_BINOMIAL_RHS;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_BINOMIAL_AND_DIRECT_INTRA: 
            mv2_user_scatter_inter = MV2_SCATTER_TWO_LEVEL_BINOMIAL;
            mv2_user_scatter_intra = MV2_SCATTER_DIRECT_RHS;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DIRECT_AND_BINOMIAL_INTRA: 
            mv2_user_scatter_inter = MV2_SCATTER_TWO_LEVEL_DIRECT;
            mv2_user_scatter_intra = MV2_SCATTER_BINOMIAL_RHS;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DIRECT_AND_DIRECT_INTRA:
            mv2_user_scatter_inter = MV2_SCATTER_TWO_LEVEL_DIRECT;
            mv2_user_scatter_intra = MV2_SCATTER_DIRECT_RHS;
            mv2_tune_parameter = 1;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of CVAR:"
                " MPIR_CVAR_SCATTER_COLLECTIVE_ALGORITHM is not valid. Valid values"
                " are natural numbers less than %d\n",
                 MV2_MAX_NUM_SCATTER_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);    
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_gather_collective_algorithm()
{
    int count = 0;
    int read_value = 0;
    int cvar_index = 0;
    int mpi_errno = MPI_SUCCESS;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_GATHER_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL, 
                                        &mv2_gather_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_gather_coll_algo_handle, 
                                        (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (read_value != -1 && 
        (getenv("MV2_INTRA_GATHER_TUNING") != NULL ||
         getenv("MV2_INTER_GATHER_TUNING") != NULL)){
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for MPI_Gather."
            " This is a conflict. Please use one of them\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value > MV2_MAX_NUM_GATHER_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_GATHER_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_GATHER_FUNCS,
            read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_GATHER_BINOMIAL: 
            mv2_user_gather_inter =  MV2_GATHER_INTER;
            mv2_tune_parameter = 1; 	      	
            break;
        case MV2_GATHER_DIRECT: 
            mv2_user_gather_inter =  MV2_GATHER_MV2_DIRECT;
            mv2_tune_parameter = 1;		
            break;
        case MV2_GATHER_TWO_LEVELT_DIRECT_AND_BINOMIAL_INTRA: 
            mv2_user_gather_inter =  MV2_GATHER_MV2_TWO_LEVEL_DIRECT;
            mv2_user_gather_intra =  MV2_GATHER_INTER;
            mv2_tune_parameter = 1;
            break;
        case MV2_GATHER_TWO_LEVELT_DIRECT_AND_DIRECT_INTRA: 
            mv2_user_gather_inter =  MV2_GATHER_MV2_TWO_LEVEL_DIRECT;
            mv2_user_gather_intra =  MV2_GATHER_INTRA;
            mv2_tune_parameter = 1;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of "
                "CVAR MPIR_CVAR_GATHER_COLLECTIVE_ALGORITHM is not valid. "
                " Valid values are natural numbers less than %d\n",
                 MV2_MAX_NUM_GATHER_FUNCS);
            mpi_errno = MPI_ERR_INTERN;	
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_reduce_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_REDUCE_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL, 
                                        &mv2_reduce_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_reduce_coll_algo_handle, 
                                        (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (read_value != -1 && 
       (getenv("MV2_INTER_REDUCE_TUNING") != NULL || 
        getenv("MV2_INTRA_REDUCE_TUNING") != NULL || 
        getenv("MV2_USE_ZCOPY_REDUCE") != NULL || 
        getenv("MV2_INTER_REDUCE_TUNING_TWO_LEVEL") != NULL )) {   
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for "
            " MPI_Reduce This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value > MV2_MAX_NUM_REDUCE_FUNCS) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_REDUCE_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_REDUCE_FUNCS,
            read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_BINOMIAL_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_BINOMIAL;
            mv2_tune_parameter = 1;			
            break;
        case MV2_KNOMIAL_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_INTER_KNOMIAL;
            mv2_tune_parameter = 1;		
            break;
        case MV2_REDUCE_SCATTER_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_RDSC_GATHER;
            mv2_tune_parameter = 1;			
            break;
        case MV2_ZCPY_REDUCE:
            mv2_enable_zcpy_reduce = REDUCE_BINOMIAL;
            mv2_user_reduce_inter = MV2_REDUCE_ZCPY;		
            mv2_tune_parameter = 1;
            break;
        case MV2_BINOMIAL_AND_KNOMIAL_INTRA_REDUCE:       		
            mv2_user_reduce_inter = MV2_REDUCE_BINOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_INTRA_KNOMIAL;
            mv2_tune_parameter = 1;
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_BINOMIAL_AND_SHMEM_INTRA_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_BINOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_SHMEM;
            mv2_tune_parameter = 1;
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_BINOMIAL_AND_BINOMIAL_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_BINOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_BINOMIAL;
            mv2_tune_parameter = 1;	
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_REDUCE_SCATTER_REDUCE_AND_KNOMIAL_INTRA_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_RDSC_GATHER;
            mv2_user_reduce_intra = MV2_REDUCE_INTRA_KNOMIAL;
            mv2_tune_parameter = 1;	
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_REDUCE_SCATTER_REDUCE_AND_SHMEM_INTRA_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_RDSC_GATHER;
            mv2_user_reduce_intra = MV2_REDUCE_SHMEM;
            mv2_tune_parameter = 1;		
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_REDUCE_SCATTER_REDUCE_AND_BINOMIAL_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_RDSC_GATHER;
            mv2_user_reduce_intra = MV2_REDUCE_BINOMIAL;
            mv2_tune_parameter = 1;
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_KNOMIAL_AND_KNOMIAL_INTRA_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_INTER_KNOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_INTRA_KNOMIAL;
            mv2_tune_parameter = 1;		
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_KNOMIAL_AND_SHMEM_INTRA_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_INTER_KNOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_SHMEM;
            mv2_tune_parameter = 1;
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        case MV2_KNOMIAL_AND_BINOMIAL_REDUCE:
            mv2_user_reduce_inter = MV2_REDUCE_INTER_KNOMIAL;
            mv2_user_reduce_intra = MV2_REDUCE_BINOMIAL;
            mv2_tune_parameter = 1;
            mv2_user_reduce_two_level = REDUCE_BINOMIAL;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of "
                "CVAR MPIR_CVAR_REDUCE_COLLECTIVE_ALGORITHM is not valid. "
                " Valid values are natural numbers less than %d\n",
                 MV2_MAX_NUM_REDUCE_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_allgather_collective_algorithm()
{	
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_ALLGATHER_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
                                 &mv2_allgather_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_allgather_coll_algo_handle, 
                                      (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
     if (read_value != -1 &&
        (getenv("MV2_INTER_ALLGATHER_TUNING") != NULL)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for " 
            "MPI_Allgather. This is a conflict. Please use one of them.\n");        
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value > MV2_MAX_NUM_ALLGATHER_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            "MPIR_CVAR_ALLGATHER_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_ALLGATHER_FUNCS,
            read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    switch(read_value){
        case MV2_RD_ALLGATHER_COMM:
            mv2_user_allgather_inter = MV2_ALLGATHER_RD_ALLGATHER_COMM;
            mv2_tune_parameter = 1;
            break;
        case MV2_RD_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_RD;
            mv2_tune_parameter = 1;		
            break;
        case MV2_BRUCK_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_BRUCK;
            mv2_tune_parameter = 1;			     
            break;
        case MV2_RING_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_RING;
            mv2_tune_parameter = 1;
            break;
        case MV2_DIRECT_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_DIRECT;
            mv2_tune_parameter = 1;
            break;
        case MV2_DIRECTSPREAD_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_DIRECTSPREAD;
            mv2_tune_parameter = 1;
            break;
        case MV2_GATHER_BCAST_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_GATHER_BCAST;
            mv2_tune_parameter = 1;
            break;
        case MV2_2LVL_NONBLOCKED_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_2LVL_NONBLOCKED;
            mv2_tune_parameter = 1;
            break;
        case MV2_2LVL_RING_NONBLOCKED_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_2LVL_RING_NONBLOCKED;
            mv2_tune_parameter = 1;
            break;
        case MV2_2LVL_DIRECT_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_2LVL_DIRECT;
            mv2_tune_parameter = 1;
            break;
        case MV2_2LVL_RING_ALLGATHER:
            mv2_user_allgather_inter = MV2_ALLGATHER_2LVL_RING;
            mv2_tune_parameter = 1;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of "
                "CVAR MPIR_CVAR_ALLGATHER_COLLECTIVE_ALGORITHM is not valid. "
                " Valid values are natural numbers less than %d\n",
                 MV2_MAX_NUM_ALLGATHER_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_allreduce_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;
    
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_ALLREDUCE_COLLECTIVE_ALGORITHM, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL, 
                            &mv2_allreduce_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_allreduce_coll_algo_handle,
                                       (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (read_value != -1 && 
       (getenv("MV2_INTER_ALLREDUCE_TUNING") != NULL || 
        getenv("MV2_INTRA_ALLREDUCE_TUNING") != NULL || 
        getenv("MV2_INTER_ALLREDUCE_TUNING_TWO_LEVEL") != NULL )) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for "
            "MPI_Allreduce. This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value > MV2_MAX_NUM_ALLREDUCE_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_ALLREDUCE_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_ALLREDUCE_FUNCS,
            read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_RD_ALLREDUCE: 
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RD;
            mv2_tune_parameter = 1;
            break;
        case MV2_REDUCE_SCATTER_ALLGATHER_ALLREDUCE:
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RS;
            mv2_tune_parameter = 1;
            break;
        case MV2_RD_ALLREDUCE_AND_RD_ALLREDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RD;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_RD;
            mv2_tune_parameter = 1;
            break;
        case MV2_RD_ALLREDUCE_AND_REDUCE_SCATTER_ALLGATHER:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RD;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_RS;
            mv2_tune_parameter = 1;      	
            break;
        case MV2_RD_ALLREDUCE_AND_SHMEM_REDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RD;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_SHMEM_REDUCE;
            mv2_tune_parameter = 1;
            break;
        case MV2_RD_ALLREDUCE_AND_P2P_REDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RD;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_REDUCE;
            mv2_tune_parameter = 1;    	
            break;
        case MV2_REDUCE_SCATTER_ALLGATHER_AND_RD_ALLREDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RS;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_RD;
            mv2_tune_parameter = 1;   	
            break;
        case MV2_REDUCE_SCATTER_ALLGATHER_AND_REDUCE_SCATTER_ALLGATHER:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RS;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_RS;
            mv2_tune_parameter = 1;      	
            break;
        case MV2_REDUCE_SCATTER_ALLGATHER_AND_SHMEM_REDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RS;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_SHMEM_REDUCE;
            mv2_tune_parameter = 1;      	
            break;
        case MV2_REDUCE_SCATTER_ALLGATHER_AND_P2P_REDUCE:
            mv2_user_allreduce_two_level = ALLREDUCE_P2P_RD;
            mv2_user_allreduce_inter = MV2_ALLREDUCE_P2P_RS;
            mv2_user_allreduce_intra = MV2_ALLREDUCE_P2P_REDUCE;
            mv2_tune_parameter = 1;   	
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of CVAR "
                "MPIR_CVAR_ALLREDUCE_COLLECTIVE_ALGORITHM is not valid."
                " Valid values are natural numbers less than %d\n",
                 MV2_MAX_NUM_ALLREDUCE_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

int mv2_set_alltoall_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;
    
    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_ALLTOALL_COLLECTIVE_ALGORITHM,
                             cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    /* Allocate CVAR handle */
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
                     &mv2_alltoall_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Read value of CVAR */
    mpi_errno = MPIR_T_cvar_read_impl(mv2_alltoall_coll_algo_handle, 
                                       (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* The user did not set any value for the CVAR. Exit. */
    if (read_value < 0) {
        goto fn_exit;
    }
    /* Check if environment variable and CVAR has been set at the same time */
    if (read_value != -1 &&
       (getenv("MV2_ALLTOALL_TUNING") != NULL)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for "
            "MPI_Alltoall. This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Check if value for CVAR is valid */
    if (read_value < 0 || read_value > MV2_MAX_NUM_ALLTOALL_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_ALLTOALL_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n", 
            MV2_MAX_NUM_ALLTOALL_FUNCS, read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    /* Choose algorithm based on CVAR */
    switch(read_value){
        case MV2_BRUCK_ALLTOALL: 
            mv2_user_alltoall = MV2_ALLTOALL_BRUCK_MV2;
            mv2_tune_parameter = 1;
            break;
        case MV2_RD_ALLTOALL: 
            mv2_user_alltoall = MV2_ALLTOALL_RD_MV2;
            mv2_tune_parameter = 1;
            break;
        case MV2_SCATTER_DESTINATION_ALLTOALL:
            mv2_user_alltoall = MV2_ALLTOALL_SCATTER_DEST_MV2;
            mv2_tune_parameter = 1;
            break;
        case MV2_PAIRWISE_ALLTOALL:
            mv2_user_alltoall = MV2_ALLTOALL_PAIRWISE_MV2;
            mv2_tune_parameter = 1;
            break;
        case MV2_INPLACE_ALLTOALL:
            mv2_user_alltoall = MV2_ALLTOALL_INPLACE_MV2;
            mv2_tune_parameter = 1;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of "
                "CVAR MPIR_CVAR_ALLTOALL_COLLECTIVE_ALGORITHM is not valid. "
                " Valid values are natural numbers less than %d\n", 
                MV2_MAX_NUM_ALLTOALL_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}


int mv2_set_alltoallv_collective_algorithm()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int count = 0;
    int read_value = 0;

    //Get CVAR index by name
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_ALLTOALLV_COLLECTIVE_ALGORITHM,
                             cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    // Allocate CVAR handle
    mpi_errno = MPIR_T_cvar_handle_alloc_impl(cvar_index, NULL,
                     &mv2_alltoallv_coll_algo_handle, &count);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    // Read value of CVAR
    mpi_errno = MPIR_T_cvar_read_impl(mv2_alltoallv_coll_algo_handle,
                                       (void*)&read_value);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    // The user did not set any value for the CVAR. Exit.
    if (read_value < 0) {
        goto fn_exit;
    }
    // Check if environment variable and CVAR has been set at the same time
    if (read_value != -1 &&
       (getenv("MV2_ALLTOALLV_TUNING") != NULL)) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "User has set environment "
            "variables and CVAR for choosing collective algorithm for "
            "MPI_Alltoallv. This is a conflict. Please use one of them.\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    // Check if value for CVAR is valid
    if (read_value < 0 || read_value > MV2_MAX_NUM_ALLTOALLV_FUNCS ) {
        PRINT_INFO(MPIDI_Process.my_pg_rank == 0 , "\nSelected value of CVAR:"
            " MPIR_CVAR_ALLTOALLV_COLLECTIVE_ALGORITHM is out of range; valid"
            "values are natural numbers less than %d (entered value is %d)\n",
            MV2_MAX_NUM_ALLTOALLV_FUNCS, read_value);
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    // Choose algorithm based on CVAR
    switch(read_value){
        case MV2_INTRA_SCATTER_ALLTOALLV:
            mv2_user_alltoallv = MV2_ALLTOALLV_INTRA_SCATTER_MV2;
            mv2_tune_parameter = 1;
            break;
        case MV2_INTRA_ALLTOALLV:
            mv2_user_alltoallv = MV2_ALLTOALLV_INTRA_MV2;
            mv2_tune_parameter = 1;
            break;
        default:
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "\nSelected value of "
                "CVAR MPIR_CVAR_ALLTOALLV_COLLECTIVE_ALGORITHM is not valid. "
                " Valid values are natural numbers less than %d\n",
                MV2_MAX_NUM_ALLTOALLV_FUNCS);
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
            break;
    }

fn_fail:
fn_exit:
    return mpi_errno;
}

