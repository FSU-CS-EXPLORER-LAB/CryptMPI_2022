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


#ifndef _COLL_SHMEM_
#define _COLL_SHMEM_
#include <pthread.h>
#include <math.h>
#include "mpidimpl.h"

#include "mv2_arch_hca_detect.h"

#if defined(_SMP_LIMIC_)
#define LIMIC_COLL_NUM_COMM  128
#endif /* #if defined(_SMP_LIMIC_) */ 

#define MV2_SHMEM_MAX_MSG_SIZE 128*1024
#define MV2_SHMEM_COLL_BLOCKS 8

#define PID_CHAR_LEN 22

#define SHMEM_COLL_HOSTNAME_LEN  (255)

#define SHMEM_SMALLEST_SIZE (64)

#define SHMEM_MAX_INT ((unsigned int)(-1))

#define MV2_DEFAULT_SHMEM_BCAST_LEADERS    4096
#define MV2_GATHER_DIRECT_SYSTEM_SIZE_SMALL      384
#define MV2_GATHER_DIRECT_SYSTEM_SIZE_MEDIUM     1024

#define MV2_INTER_NODE_KNOMIAL_FACTOR_MAX 8
#define MV2_INTER_NODE_KNOMIAL_FACTOR_MIN 2
#define MV2_INTRA_NODE_KNOMIAL_FACTOR_MAX 8
#define MV2_INTRA_NODE_KNOMIAL_FACTOR_MIN 2 

#define MV2_DEFAULT_SHARP_MAX_MSG_SIZE 1024 

#if defined(_IA32_)

#define SHMEM_CACHE_LINE_SIZE 64
#define SHMEM_ALIGN(a)                                    \
((a + SHMEM_CACHE_LINE_SIZE + 7) & 0xFFFFFFF8)
#define SHMEM_AVAIL(a)	                                  \
 ((a & 0xFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#elif defined(_IA64_)

#define SHMEM_CACHE_LINE_SIZE 128
#define SHMEM_ALIGN(a)                                    \
((a + SHMEM_CACHE_LINE_SIZE + 7) & 0xFFFFFFFFFFFFFFF8)
#define SHMEM_AVAIL(a)                                    \
 ((a & 0xFFFFFFFFFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#elif defined(__x86_64__)

#define SHMEM_CACHE_LINE_SIZE 64
#define SHMEM_ALIGN(a)                                    \
((a + SHMEM_CACHE_LINE_SIZE + 7) & 0xFFFFFFFFFFFFFFF8)
#define SHMEM_AVAIL(a)                                    \
 ((a & 0xFFFFFFFFFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#elif defined(_EM64T_)

#define SHMEM_CACHE_LINE_SIZE 64
#define SHMEM_ALIGN(a) (a + SHMEM_CACHE_LINE_SIZE)
#define SHMEM_AVAIL(a)                                   \
((a & 0xFFFFFFFFFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#elif defined(MAC_OSX)

#define SHMEM_CACHE_LINE_SIZE 16
#define SHMEM_ALIGN(a)                                   \
(((a + SHMEM_CACHE_LINE_SIZE + 7) & 0xFFFFFFF8))
#define SHMEM_AVAIL(a)                                   \
((a & 0xFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#else

#define SHMEM_CACHE_LINE_SIZE 64
#define SHMEM_ALIGN(a) (a + SHMEM_CACHE_LINE_SIZE)
#define SHMEM_AVAIL(a)                                   \
((a & 0xFFFFFFFFFFFFFFF8) - SHMEM_CACHE_LINE_SIZE)

#endif


int MPIDI_CH3I_SHMEM_COLL_init(MPIDI_PG_t *pg, int local_id);

int MPIDI_CH3I_SHMEM_COLL_Mmap(MPIDI_PG_t *pg, int local_id); 

int MPIDI_CH3I_SHMEM_COLL_finalize(int local_id, int num_local_nodes);

void MPIDI_CH3I_SHMEM_COLL_Unlink(void);

void MV2_Read_env_vars(void);

#define SHMEM_COLL_BLOCK_SIZE (MPIDI_Process.my_pg->ch.num_local_processes * mv2_g_shmem_coll_max_msg_size)
#define MPICH_LARGE_MSG_COLLECTIVE_SIZE (1*1024*1024*1024)

#define COLL_COUNT              7
#define COLL_SIZE               3
#define ALLGATHER_ID            0
#define ALLREDUCE_SHMEM_ID      1
#define ALLREDUCE_2LEVEL_ID     2
#define BCAST_KNOMIAL_ID        3
#define BCAST_SHMEM_ID          4
#define REDUCE_SHMEM_ID         5
#define REDUCE_2LEVEL_ID        6

#define SMALL                   0
#define MEDIUM                  1
#define LARGE                   2

#define MV2_MAX_NB_THRESHOLDS  32

#define MV2_PARA_PACKET_SIZE    5

extern int mv2_tuning_table[COLL_COUNT][COLL_SIZE]; 

struct scatter_tuning{
    int numproc;
    int small;
    int medium;
};

struct gather_tuning{
    int numproc;
    int switchp;
};

struct allgatherv_tuning{
    int numproc;
    int switchp;
};


#define BCAST_LEN 20
#define SHMEM_BCAST_FLAGS	1024
/*
 * We're converting this into a environment variable
 * #define SHMEM_BCAST_LEADERS     1024
 */
#define SHMEM_BCAST_METADATA	(sizeof(addrint_t) + 2*sizeof(int))       
  /* METADATA: buffer address, offset, num_bytes */ 

extern int shmem_coll_count_threshold;
extern int mv2_g_shmem_coll_max_msg_size;
extern int mv2_g_shmem_coll_blocks;
extern int mv2_shmem_coll_num_procs;
extern int mv2_shmem_coll_spin_count;
extern int mv2_enable_shmem_collectives;
int is_shmem_collectives_enabled();
extern int mv2_two_level_comm_early_init_threshold;

extern struct coll_runtime mv2_coll_param;
void MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(int, int, int, void**);
void MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(int, int, int);
int create_allgather_comm(MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

#define MV2_DEFAULT_COLL_SKIP_TABLE_THRESHOLD 1024

extern int mv2_allred_use_ring;

extern int mv2_coll_skip_table_threshold;
extern int mv2_enable_skip_tuning_table_search;
extern int mv2_tune_parameter;
extern int mv2_use_indexed_bcast_tuning;
extern int mv2_use_indexed_scatter_tuning;
extern int mv2_use_indexed_gather_tuning;
extern int mv2_use_indexed_reduce_tuning;
extern int mv2_use_indexed_allreduce_tuning;
extern int mv2_use_indexed_allgather_tuning;
extern int mv2_use_indexed_alltoall_tuning;
extern int mv2_use_indexed_alltoallv_tuning;
extern int mv2_enable_ibcast;
extern int mv2_enable_ibarrier;
extern int mv2_enable_iscatter;
extern int mv2_enable_igather;
extern int mv2_enable_iallgather;
extern int mv2_enable_iallgatherv;
extern int mv2_enable_ialltoall;
extern int mv2_enable_ialltoallv;
extern int mv2_enable_ireduce;
extern int mv2_enable_ireduce_scatter;
extern int mv2_enable_iallreduce;

/* Enable/Disable MPI_T pvar timers */
extern int mv2_enable_pvar_timer;

/* Use for collective tuning based on arch detection*/
int MV2_collectives_arch_init(int heterogeneity, struct coll_info *collective_tuning_info);
void MV2_collectives_arch_finalize();

/* Use for allgather_osu.c */
#define MV2_ALLGATHER_SMALL_SYSTEM_SIZE       128
#define MV2_ALLGATHER_MEDIUM_SYSTEM_SIZE      256
#define MV2_ALLGATHER_LARGE_SYSTEM_SIZE       512 
extern int mv2_allgather_ranking;

/* Use for allgatherv_osu.c */
extern int mv2_size_mv2_allgatherv_mv2_tuning_table;
extern struct allgatherv_tuning mv2_allgatherv_mv2_tuning_table[4];
extern int mv2_user_allgatherv_switch_point;

/* Use for scatter_osu.c*/
extern int mv2_user_scatter_small_msg;
extern int mv2_user_scatter_medium_msg;
extern int mv2_size_mv2_scatter_mv2_tuning_table;
extern struct scatter_tuning mv2_scatter_mv2_tuning_table[4];
extern int mv2_use_two_level_scatter; 
extern int mv2_use_direct_scatter;
#if defined(_MCST_SUPPORT_)
extern int mv2_use_mcast_scatter;
extern int mv2_mcast_scatter_msg_size; 
extern int mv2_mcast_scatter_small_sys_size;
extern int mv2_mcast_scatter_large_sys_size;
#endif  /* #if defined(_MCST_SUPPORT_) */ 

/* Use inside reduce_osu.c*/
extern int mv2_user_reduce_two_level;
extern int mv2_user_allgather_two_level;
extern int mv2_reduce_zcopy_inter_knomial_factor;

/* Use inside allreduce_osu.c*/
extern int mv2_enable_shmem_allreduce;
extern int mv2_user_allreduce_two_level;
#if defined(_MCST_SUPPORT_)
extern int mv2_use_mcast_allreduce; 
extern int mv2_mcast_allreduce_small_msg_size; 
extern int mv2_mcast_allreduce_large_msg_size; 
#endif  /* #if defined(_MCST_SUPPORT_) */ 

/* Use inside alltoall_osu.h */
extern int mv2_use_xor_alltoall; 

extern char *mv2_user_bcast_intra;
extern char *mv2_user_bcast_inter;

/* Use inside barrier_osu.c*/
extern int mv2_enable_shmem_barrier;
extern void MPIDI_CH3I_SHMEM_COLL_Barrier_gather(int, int, int);
extern void MPIDI_CH3I_SHMEM_COLL_Barrier_bcast(int, int, int);


extern int mv2_enable_socket_aware_collectives;
extern int mv2_use_socket_aware_barrier;
extern int mv2_use_socket_aware_allreduce;
extern int mv2_use_socket_aware_sharp_allreduce;
extern int mv2_socket_aware_allreduce_max_msg;
extern int mv2_socket_aware_allreduce_min_msg;
/* Use inside bcast_osu.c */
typedef struct bcast_ring_allgather_shm_packet
{
   int j;
   int jnext;
   int root;
   MPIDI_msg_sz_t nbytes;
   MPIDI_msg_sz_t scatter_size;
} bcast_ring_allgather_shm_packet;

extern int  mv2_bcast_short_msg; 
extern int  mv2_bcast_large_msg; 
extern int  mv2_knomial_2level_bcast_system_size_threshold;
extern int  mv2_knomial_2level_bcast_message_size_threshold;
extern int  mv2_enable_knomial_2level_bcast;
extern int  mv2_inter_node_knomial_factor;
extern int  mv2_intra_node_knomial_factor;
extern int  mv2_scatter_rd_inter_leader_bcast; 
extern int  mv2_scatter_ring_inter_leader_bcast;
extern int  mv2_knomial_intra_node_threshold; 
extern int  mv2_knomial_inter_leader_threshold; 
extern int  mv2_knomial_inter_leader_bcast;
extern int  mv2_enable_shmem_bcast;
extern int  mv2_bcast_two_level_system_size; 
extern int  mv2_alltoall_inplace_old;
extern int  mv2_use_scatter_dest_alltoallv;

extern int mv2_allreduce_ring_algo_ppn_threshold;
extern int  mv2_allreduce_red_scat_allgather_algo_ppn_threshold;
extern int  mv2_allreduce_red_scat_allgather_algo_threshold;
extern int  mv2_allreduce_ring_algo_threshold;
extern int  mv2_allgather_ring_algo_threshold;
extern int  mv2_allgather_cyclic_algo_threshold;
extern int  mv2_redscat_cyclic_algo_threshold;
extern int  mv2_allreduce_cyclic_algo_threshold;
extern int  mv2_red_scat_ring_algo_threshold;

extern int mv2_alltoallv_intermediate_wait_threshold; 

extern int mv2_bcast_scatter_ring_overlap;
extern int mv2_bcast_scatter_ring_overlap_msg_upperbound;
extern int mv2_bcast_scatter_ring_overlap_cores_lowerbound;
extern int mv2_enable_zcpy_bcast; 
extern int mv2_enable_zcpy_reduce;
extern int mv2_gatherv_ssend_threshold;

/* Used inside reduce_osu.c */
extern int mv2_enable_shmem_reduce;
extern int mv2_use_knomial_reduce;
extern int mv2_reduce_inter_knomial_factor;
extern int mv2_reduce_intra_knomial_factor;
extern int MPIR_Reduce_two_level_helper_MV2(const void *sendbuf,
                                     void *recvbuf,
                                     int count,
                                     MPI_Datatype datatype,
                                     MPI_Op op,
                                     int root,
                                     MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag); 
extern int MPIR_Reduce_redscat_gather_MV2(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          int root,
                                          MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag); 
extern int MPIR_Reduce_binomial_MV2(const void *sendbuf,
                                    void *recvbuf,
                                    int count,
                                    MPI_Datatype datatype,
                                    MPI_Op op,
                                    int root,
                                    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag); 





/* Use inside red_scat_osu.c */
#define MPIR_RED_SCAT_SHORT_MSG 64
#define MPIR_RED_SCAT_LONG_MSG  512*1024
extern int mv2_red_scat_short_msg;
extern int mv2_red_scat_long_msg;

/* Lock/unlock shmem region */
void lock_shmem_region(void);
void unlock_shmem_region(void);

/* utils */
int mv2_increment_shmem_coll_counter(MPID_Comm *comm_ptr); 
int mv2_increment_allgather_coll_counter(MPID_Comm *comm_ptr); 
void increment_mv2_shmem_comm_count(void);
int get_mv2_shmem_comm_count(void);
int MPIDI_CH3I_SHMEM_Coll_get_free_block(); 
void MPIDI_CH3I_SHMEM_Coll_Block_Clear_Status(int block_id); 
#if defined(_SMP_LIMIC_)
void UpdateNumCoresPerSock(int numcores);
void UpdateNumSocketsPerNode(int numSocketsNode);
void increment_mv2_limic_comm_count();
int get_mv2_limic_comm_count();
extern int mv2_max_limic_comms;
extern int limic_fd;
#endif
void MPIDI_CH3I_SHMEM_Bcast_GetBuf(int, int, int, void**);
void MPIDI_CH3I_SHMEM_Bcast_Complete(int ,int , int);
int init_thread_reg(void);

extern int mv2_use_osu_collectives;
extern int mv2_use_indexed_tuning;
extern int mv2_use_osu_nb_collectives;
extern int mv2_use_anl_collectives;

#define MV2_DEFAULT_BITONIC_COMM_SPLIT_THRESHOLD    (12000)

extern int mv2_use_bitonic_comm_split;
extern int mv2_bitonic_comm_split_threshold;

/* Comm functions*/
extern int split_comm;
int check_split_comm(pthread_t);
int disable_split_comm(pthread_t);
int create_2level_comm (MPI_Comm, int, int);
int free_2level_comm (MPID_Comm *);
int enable_split_comm(pthread_t);
void MPIR_pof2_comm(MPID_Comm *, int, int);
#if defined(_MCST_SUPPORT_)
int create_mcast_comm (MPI_Comm, int, int);
#endif /*defined(_MCST_SUPPORT_)*/
#if defined (_SHARP_SUPPORT_)
int create_sharp_comm(MPI_Comm, int, int);
#endif /*defined (_SHARP_SUPPORT_)*/


/*Fn pointers for collectives */
int (*reduce_fn)(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

#ifdef _ENABLE_CUDA_
int cuda_stage_alloc(void **, int, void **, int,
                      int, int, int);
void cuda_stage_free (void **, void **, int, int, 
                        int);
void CUDA_COLL_Finalize ();                        
void cuda_coll_pack (void **, int *, MPI_Datatype *,
                     void **, int *, MPI_Datatype *,
                     int, int, int);
void cuda_coll_unpack (int *, int);
#endif /*_ENABLE_CUDA_*/

extern int mv2_shm_window_size;
extern int mv2_shm_slot_len;
extern int mv2_use_slot_shmem_coll;
extern int mv2_use_slot_shmem_bcast;
extern int mv2_use_mcast_pipeline_shm;

#define MV2_SHM_ALIGN (128)

#define MV2_SHM_ALIGN_LEN(len, align_unit)          \
{                                                   \
    len = ((int)(((len)+align_unit-1) /             \
                align_unit)) * align_unit;          \
}
#define IS_SHMEM_WINDOW_FULL(start, end) \
    ((((int)(start) - (end)) >= mv2_shm_window_size -1) ? 1 : 0)

#define IS_SHMEM_WINDOW_HALF_FULL(start, end) \
    ((((int)(start) - (end)) >= mv2_shm_window_size/2) ? 1 : 0)


int IS_SHMEM_WINDOW_REDUCE_HALF_FULL(int start, int end); 

#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
#define MAX_NUM_HCAS                    (4)
typedef struct shm_coll_pkt{
     int  peer_rank;
     int  recv_id;
     uint32_t key[MAX_NUM_HCAS]; 
     uint64_t addr[MAX_NUM_HCAS]; 
} shm_coll_pkt; 
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

typedef struct shm_slot_t {
    volatile uint32_t psn __attribute__((aligned(MV2_SHM_ALIGN)));
    volatile uint32_t *tail_psn __attribute__((aligned(MV2_SHM_ALIGN)));
    char buf[] __attribute__((aligned(MV2_SHM_ALIGN)));
} shm_slot_t;

typedef struct shm_slot_cntrl_t {
    volatile uint32_t psn __attribute__((aligned(MV2_SHM_ALIGN)));
    volatile uint32_t *tail_psn __attribute__((aligned(MV2_SHM_ALIGN)));
} shm_slot_cntrl_t;

typedef struct shm_queue_t {
    shm_slot_t **shm_slots;
}shm_queue_t;

typedef struct shm_info_t {
    char *buffer;
    char *file_name;
    int local_rank;
    int local_size;
    int file_fd;
    int size;
    int count;
    unsigned long long int write;
    unsigned long long int read;
    int tail;
    shm_queue_t *queue;
    MPI_Comm comm; 
    int max_local_size;
#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
    int buffer_registered; 
    /* zcpy bcast */
    int bcast_exchange_rdma_keys; 
    int bcast_knomial_factor; 
    int bcast_expected_send_count; 
    shm_coll_pkt  *bcast_remote_handle_info_parent; 
    shm_coll_pkt  *bcast_remote_handle_info_children; 
    /* zcpy reduce */
    int reduce_exchange_rdma_keys; 
    int reduce_knomial_factor; 
    int reduce_expected_recv_count; 
    int reduce_expected_send_count; 
    shm_coll_pkt  *reduce_remote_handle_info_parent; 
    shm_coll_pkt  *reduce_remote_handle_info_children; 
    int *inter_node_reduce_status_array; 
    /* request info */
    int mid_request_active; 
    int end_request_active; 
    MPI_Request   mid_request; 
    MPI_Request   end_request; 
    int half_full_complete; 
    struct ibv_mr *mem_handle[MAX_NUM_HCAS]; /* mem hndl for entire region */
#ifdef CKPT
    struct shm_info_t *next;
    struct shm_info_t *prev;
#endif /* CKPT */
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */
} shmem_info_t;

shmem_info_t * mv2_shm_coll_init(int id, int local_rank, int local_size, 
                                 MPID_Comm *comm_ptr);
void mv2_shm_coll_cleanup(shmem_info_t * shmem);
void mv2_shm_barrier(shmem_info_t * shmem);
int mv2_shm_bcast(shmem_info_t * shmem, char *buf, int len, int root);
void mv2_shm_reduce(shmem_info_t *shmem, char *buf, int len, 
                        int count, int root, MPI_User_function *uop, MPI_Datatype datatype, int is_cxx_uop);  
void mv2_shm_tree_reduce(shmem_info_t * shmem, char *in_buf, int len,
                    int count, int root, MPI_User_function * uop, MPI_Datatype datatype, 
                    int is_cxx_uop); 

#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB)
int mv2_shm_coll_reg_buffer(void *buffer, int size, struct ibv_mr *mem_handle[], 
                           int *buffer_registered); 
int mv2_shm_coll_dereg_buffer(struct ibv_mr *mem_handle[]);
void mv2_shm_coll_prepare_post_send(uint64_t local_rdma_addr, uint64_t remote_rdma_addr,
                      uint32_t local_rdma_key, uint32_t remote_rdma_key,
                      int len, int rail, MPIDI_VC_t * vc); 
int mv2_shm_zcpy_bcast(shmem_info_t * shmem, char *buf, int len, int root,
                       int src, int expected_recv_count,
                       int *dst_array, int expected_send_count,
                       int knomial_degree,
                       MPID_Comm *comm_ptr);
int mv2_shm_zcpy_reduce(shmem_info_t * shmem,
                         void *in_buf, void **out_buf,
                         int count, int len,
                         MPI_Datatype datatype,
                         MPI_Op op, int root,
                         int expected_recv_count, int *src_array,
                         int expected_send_count, int dst,
                         int knomial_degree,
                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
extern int MPIDI_CH3I_SHMEM_Helper_fn(MPIDI_PG_t * pg, int local_id, char **filename,
                                char *prefix, int *fd, size_t file_size);
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_NEMESIS_IB) */

#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_PSM)
static inline int Cyclic_Rank_list_mapper(MPID_Comm * comm_ptr, int idx)
{
    return comm_ptr->dev.ch.rank_list[idx];
};

static inline int Bunch_Rank_list_mapper(MPID_Comm * comm_ptr, int idx)
{
    return idx;
};
#endif /* defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_PSM) */

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_shmem_coll_calls);

int mv2_set_bcast_collective_algorithm();
int mv2_set_scatter_collective_algorithm();
int mv2_set_gather_collective_algorithm();
int mv2_set_reduce_collective_algorithm();
int mv2_set_allgather_collective_algorithm();
int mv2_set_allreduce_collective_algorithm();
int mv2_set_alltoall_collective_algorithm();
int mv2_set_alltoallv_collective_algorithm();

/* Collective values for gather */
#define MV2_GATHER_INTRA                    "0"
#define MV2_GATHER_INTER                    "1"
#define MV2_GATHER_MV2_DIRECT               "2"
#define MV2_GATHER_MV2_TWO_LEVEL_DIRECT     "3"

/* Collective values for reduce */
#define MV2_REDUCE_BINOMIAL                 "1"
#define MV2_REDUCE_INTER_KNOMIAL            "2"
#define MV2_REDUCE_INTRA_KNOMIAL            "3"
#define MV2_REDUCE_SHMEM                    "4"
#define MV2_REDUCE_RDSC_GATHER              "5"
#define MV2_REDUCE_ZCPY                     "6"

/* Collective values for scatter */
#define MV2_SCATTER_BINOMIAL_RHS            "1"
#define MV2_SCATTER_DIRECT_RHS              "2"
#define MV2_SCATTER_TWO_LEVEL_BINOMIAL      "3"
#define MV2_SCATTER_TWO_LEVEL_DIRECT        "4"
#define MV2_SCATTER_MCAST                   "5"

/* Collective values for bcast  */
#define MV2_SHMEM_BCAST_INTRA                       "0"
#define MV2_KNOMIAL_BCAST_INTRA                     "1"
#define MV2_BCAST_BIONOMIAL_INTRA                   "2"
#define MV2_BCAST_SCATTER_DOUBLING_ALLGATHER_FLAT   "3"
#define MV2_BCAST_SCATTER_DOUBLING_ALLGATHER        "4"
#define MV2_BCAST_SCATTER_RING_ALLGATEHR_FLAT       "5"
#define MV2_BCAST_SCATTER_RING_ALLGATHER            "6"
#define MV2_BCAST_SCATTER_RING_ALLGATHER_SHM        "7"
#define MV2_KNOMIAL_BCAST_INTER_NODE_WRAPPER        "8"
#define MV2_PIPELINED_BCAST_MV2                     "9"
#define MV2_PIPELINED_BCAST_ZCPY_MV2                "10"

/* Collective values for allreduce  */
#define MV2_ALLREDUCE_P2P_RD                "1"
#define MV2_ALLREDUCE_P2P_RS                "2"
#define MV2_ALLREDUCE_MCAST_2LEVEL          "3"
#define MV2_ALLREDUCE_MCAST_RSA             "4"
#define MV2_ALLREDUCE_SHMEM_REDUCE          "5"
#define MV2_ALLREDUCE_P2P_REDUCE            "6"

/* Collective values for allgather  */
#define MV2_ALLGATHER_RD_ALLGATHER_COMM     "1"
#define MV2_ALLGATHER_RD                    "2"
#define MV2_ALLGATHER_BRUCK                 "3"
#define MV2_ALLGATHER_RING                  "4"
#define MV2_ALLGATHER_DIRECT                "5"
#define MV2_ALLGATHER_DIRECTSPREAD          "6"
#define MV2_ALLGATHER_GATHER_BCAST          "7"
#define MV2_ALLGATHER_2LVL_NONBLOCKED       "8"
#define MV2_ALLGATHER_2LVL_RING_NONBLOCKED  "9"
#define MV2_ALLGATHER_2LVL_DIRECT           "10"
#define MV2_ALLGATHER_2LVL_RING             "11"

/* Collective values for alltoall  */
#define MV2_ALLTOALL_BRUCK_MV2              "0"
#define MV2_ALLTOALL_RD_MV2                 "1"
#define MV2_ALLTOALL_SCATTER_DEST_MV2       "2"
#define MV2_ALLTOALL_PAIRWISE_MV2           "3"
#define MV2_ALLTOALL_INPLACE_MV2            "4"

/* Collective values for alltoallv  */
#define MV2_ALLTOALLV_INTRA_SCATTER_MV2     "0"
#define MV2_ALLTOALLV_INTRA_MV2             "1"

/* Function set for collective algorithm selection */
enum mv2_gather_coll_funcs
{
    MV2_GATHER_BINOMIAL,
    MV2_GATHER_DIRECT,
    MV2_GATHER_TWO_LEVELT_DIRECT_AND_BINOMIAL_INTRA,
    MV2_GATHER_TWO_LEVELT_DIRECT_AND_DIRECT_INTRA,
    MV2_MAX_NUM_GATHER_FUNCS
};

enum mv2_scatter_coll_funcs
{
    MV2_SCATTER_BINOMIAL,
    MV2_SCATTER_DIRECT,
    MV2_SCATTER_BINOMIAL_AND_BINOMIAL_INTRA,
    MV2_SCATTER_BINOMIAL_AND_DIRECT_INTRA,
    MV2_SCATTER_DIRECT_AND_BINOMIAL_INTRA,
    MV2_SCATTER_DIRECT_AND_DIRECT_INTRA,
    MV2_MAX_NUM_SCATTER_FUNCS
};

enum mv2_bcast_coll_funcs
{
    MV2_FLAT_BINOMIAL,
    MV2_SCATTER_DOUBLING_ALLGATHER_FLAT,
    MV2_SCATTER_RING_ALLGATHER_FLAT,
    MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_2,
    MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_4,
    MV2_ZERO_COPY_BCAST_KNOMIAL_INTRA_8,
    MV2_PIPELINED_BCAST_KNOMIAL_INTRA,
    MV2_BINOMIAL_KNOMIAL_INTRA,
    MV2_SCATTER_DOUBLING_ALLGATHER_KNOMIAL_INTRA_1,
    MV2_SCATTER_RING_ALLGATHER_KNOMIAL_INTRA_1,
    MV2_SCATTER_DOUBLING_ALLGATHER_KNOMIAL_INTRA_2,
    MV2_SCATTER_RING_ALLGATHER_KNOMIAL_INTRA_2,
    MV2_PIPELINED_BCAST_SHMEM_INTRA,
    MV2_BINOMIAL_SHMEM_INTRA,
    MV2_SCATTER_DOUBLING_ALLGATHER_SHMEM_INTRA_1,
    MV2_SCATTER_RING_ALLGATHER_SHMEM_INTRA_1,
    MV2_SCATTER_DOUBLING_ALLGATHER_SHMEM_INTRA_2,
    MV2_SCATTER_RING_ALLGATHER_SHMEM_INTRA_2,
    MV2_MAX_NUM_BCAST_FUNCS
};

enum mv2_reduce_coll_funcs
{
    MV2_BINOMIAL_REDUCE,
    MV2_KNOMIAL_REDUCE,
    MV2_REDUCE_SCATTER_REDUCE,
    MV2_ZCPY_REDUCE,    
    MV2_BINOMIAL_AND_KNOMIAL_INTRA_REDUCE,
    MV2_BINOMIAL_AND_SHMEM_INTRA_REDUCE,
    MV2_BINOMIAL_AND_BINOMIAL_REDUCE,
    MV2_REDUCE_SCATTER_REDUCE_AND_KNOMIAL_INTRA_REDUCE,
    MV2_REDUCE_SCATTER_REDUCE_AND_SHMEM_INTRA_REDUCE,
    MV2_REDUCE_SCATTER_REDUCE_AND_BINOMIAL_REDUCE,
    MV2_KNOMIAL_AND_KNOMIAL_INTRA_REDUCE,
    MV2_KNOMIAL_AND_SHMEM_INTRA_REDUCE,
    MV2_KNOMIAL_AND_BINOMIAL_REDUCE,
    MV2_MAX_NUM_REDUCE_FUNCS
};

enum mv2_allreduce_coll_funcs
{
    MV2_RD_ALLREDUCE,
    MV2_REDUCE_SCATTER_ALLGATHER_ALLREDUCE,
    MV2_RD_ALLREDUCE_AND_RD_ALLREDUCE,
    MV2_RD_ALLREDUCE_AND_REDUCE_SCATTER_ALLGATHER,
    MV2_RD_ALLREDUCE_AND_SHMEM_REDUCE,
    MV2_RD_ALLREDUCE_AND_P2P_REDUCE,
    MV2_REDUCE_SCATTER_ALLGATHER_AND_RD_ALLREDUCE,
    MV2_REDUCE_SCATTER_ALLGATHER_AND_REDUCE_SCATTER_ALLGATHER,
    MV2_REDUCE_SCATTER_ALLGATHER_AND_SHMEM_REDUCE,
    MV2_REDUCE_SCATTER_ALLGATHER_AND_P2P_REDUCE,
    MV2_MAX_NUM_ALLREDUCE_FUNCS
};

enum mv2_allgather_coll_funcs
{
    MV2_RD_ALLGATHER_COMM,
    MV2_RD_ALLGATHER,
    MV2_BRUCK_ALLGATHER,
    MV2_RING_ALLGATHER,
    MV2_DIRECT_ALLGATHER,
    MV2_DIRECTSPREAD_ALLGATHER,
    MV2_GATHER_BCAST_ALLGATHER,
    MV2_2LVL_NONBLOCKED_ALLGATHER,
    MV2_2LVL_RING_NONBLOCKED_ALLGATHER,
    MV2_2LVL_DIRECT_ALLGATHER,
    MV2_2LVL_RING_ALLGATHER,
    MV2_MAX_NUM_ALLGATHER_FUNCS
};

enum mv2_alltoall_coll_funcs
{
    MV2_BRUCK_ALLTOALL,MV2_RD_ALLTOALL,
    MV2_SCATTER_DESTINATION_ALLTOALL,
    MV2_PAIRWISE_ALLTOALL,
    MV2_INPLACE_ALLTOALL,
    MV2_MAX_NUM_ALLTOALL_FUNCS
};

enum mv2_alltoallv_coll_funcs
{
    MV2_INTRA_SCATTER_ALLTOALLV,
    MV2_INTRA_ALLTOALLV,
    MV2_MAX_NUM_ALLTOALLV_FUNCS
};
#endif  /* _COLL_SHMEM_ */
