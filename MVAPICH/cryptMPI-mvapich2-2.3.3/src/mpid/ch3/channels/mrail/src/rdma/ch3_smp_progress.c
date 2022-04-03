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

#include "mpichconf.h"
#include "mpidi_ch3_impl.h"
#include <mpimem.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include "upmi.h"
#include "smp_smpi.h"
#include "mpiutil.h"
#include "mv2_arch_hca_detect.h"
#include "coll_shmem.h"

#if defined(_ENABLE_MPIT_TOOL_) || defined (_SMP_LIMIC_)
#include "rdma_impl.h"
#endif

#include <sys/syscall.h>
#include <sys/uio.h>
#include <assert.h>
#if defined(MAC_OSX)
#include <netinet/in.h>
#endif /* defined(MAC_OSX) */
#if defined _SMP_CMA_ 
#include <sys/types.h>
#include <stdint.h>
#endif 

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_eager_sent);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_rndv_sent);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_eager_received);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_rndv_received);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_eager_total_buffer);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_rndv_total_buffer);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_eager_avail_buffer);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_rndv_avail_buffer);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_eager_buffer_max_use);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_smp_rndv_buffer_max_use);

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_read_progress_poll);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_write_progress_poll);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_read_progress_poll_success);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_smp_write_progress_poll_success);

int mv2_shmem_pool_init = 0;
int g_smp_delay_shmem_pool_init = 1;
int polling_set_p_head = 0;
int polling_set_p_tail = 0;
int polling_set_c_head = 0;
int polling_set_c_tail = 0;
static POLLING_ELEMENT_T* polling_set_p = NULL;
static POLLING_ELEMENT_T* polling_set_c = NULL;
int *polling_counters;
static int is_fair_polling = 0;
int g_smp_max_switch;

int g_smp_polling_th = 200;
int g_smp_priority_polling = 1;
int g_smp_priority_factor = DEFAULT_SHMEM_PRIORITY_FACTOR;
static size_t cm_shmem_file_size = 0;

extern int finalize_coll_comm;
extern MPID_Request * create_eagercontig_request(MPIDI_VC_t * vc,
                         MPIDI_CH3_Pkt_type_t reqtype,
                         const void * buf, MPIDI_msg_sz_t data_sz, int rank,
                         int tag, MPID_Comm * comm, int context_offset);

#define ENQUEUE_p(x) \
    if(!polling_set_p_head) { \
        polling_set_p_head = x; \
        polling_set_p_tail = x; \
        polling_set_p[x].prev = 0; \
        polling_set_p[x].next = 0; \
    } else { \
        polling_set_p[polling_set_p_tail].next = x; \
        polling_set_p[x].prev = polling_set_p_tail; \
        polling_set_p[x].next = 0; \
        polling_set_p_tail = x; \
    } \
    if(polling_set_c[x].prev) { \
        polling_set_c[polling_set_c[x].prev].next = \
            polling_set_c[x].next; \
    } else { \
        polling_set_c_head = polling_set_c[x].next; \
    } \
    if(polling_set_c[x].next) { \
        polling_set_c[polling_set_c[x].next].prev = \
            polling_set_c[x].prev; \
    } else { \
        polling_set_c_tail = polling_set_c[x].prev; \
    }

#define SMP_EXIT_ERR -1
#define smp_error_abort(code, message) do {                     \
    if (errno) {                                                \
        PRINT_ERROR_ERRNO( "%s:%d: " message, errno, __FILE__, __LINE__);     \
    } else {                                                    \
        PRINT_ERROR( "%s:%d: " message "\n", __FILE__, __LINE__);     \
    }                                                           \
    fflush (stderr);                                            \
    exit(code);                                                 \
} while (0)

/* Macros for flow control and rqueues management */
#define SMPI_FIRST_S(sender,receiver)                                 \
    g_smpi_shmem->rqueues_limits_s[receiver].first

#define SMPI_LAST_S(sender,receiver)                                  \
    g_smpi_shmem->rqueues_limits_s[receiver].last

#define SMPI_FIRST_R(sender,receiver)                                 \
    g_smpi_shmem->rqueues_limits_r[sender].first

#define SMPI_LAST_R(sender,receiver)                                  \
    g_smpi_shmem->rqueues_limits_r[sender].last

/* Shared Tail Pointer: updated by receiver after every receive;
 * read by sender when local header meets local tail. */
#define SMPI_SHARED_TAIL(sender,receiver)                             \
    g_smpi_shmem->shared_tails[receiver][sender].ptr

#define SMPI_BUF_POOL_PTR(destination,index) \
    ((SEND_BUF_T *) ((unsigned long) s_buffer_head[destination] \
     + (sizeof(SEND_BUF_T) + s_smp_block_size)*index))    

#define SMPI_MY_BUF_POOL_PTR(index) \
    ((SEND_BUF_T *) ((unsigned long) s_my_buffer_head \
     + (sizeof(SEND_BUF_T) + s_smp_block_size)*index)) 

#define    SMP_CBUF_FREE 0
#define    SMP_CBUF_BUSY 1
#define    SMP_CBUF_PENDING 2
#define    SMP_CBUF_END 3

struct smpi_var g_smpi;
struct shared_mem *g_smpi_shmem;
static struct shared_buffer_pool s_sh_buf_pool;
static SEND_BUF_T** s_buffer_head = NULL;
static SEND_BUF_T* s_my_buffer_head = NULL;
int SMP_INIT = 0;
int SMP_ONLY = 0;
static void** s_current_ptr = NULL;
static MPIDI_msg_sz_t* s_current_bytes = NULL;
static MPIDI_msg_sz_t* s_total_bytes = NULL;
static char *shmem_file = NULL;
static char *pool_file = NULL;

/* local header/tail for send and receive pointing to cyclic buffer */
static size_t* s_header_ptr_s = NULL;
static size_t* s_tail_ptr_s = NULL;
static size_t* avail = NULL;
static size_t* s_header_ptr_r = NULL;

size_t g_size_shmem = 0;
size_t g_size_pool = 0; 

/* SMP user parameters */
 
int g_smp_eagersize;
size_t s_smp_queue_length;
int s_smp_num_send_buffer;
int s_smp_batch_size;
int s_smp_block_size;
int s_smp_cma_max_size;
int s_smp_limic2_max_size;
unsigned long eager_buffer_max_usage = 0;
unsigned long rndv_buffer_max_usage = 0;

#if defined(_ENABLE_CUDA_)
int s_smp_cuda_pipeline = 1;
int s_smp_h2h_block_size;
#endif

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
void **smp_cuda_region_base;
void **smp_cuda_region_send;
void **smp_cuda_region_recv;
size_t *smp_cuda_region_head;
int smp_cuda_region_size;
int smp_cuda_eagersize;
int s_smp_cuda_handle_cache = 0;

CUevent *sr_event;
CUevent *sr_event_local;
CUevent *loop_event;
CUevent *loop_event_local;
#endif

#if defined(_SMP_CMA_)
int g_smp_use_cma = 1;
size_t MV2_CMA_MSG_LIMIT = 1<<30;
#ifndef HAVE_PROCESS_VM_READV
#  if defined(i386)
#    define __NR_process_vm_readv 347
#    define __NR_process_vm_writev 348
#  elif defined(_ARCH_PPC)
#    define __NR_process_vm_readv 351
#    define __NR_process_vm_writev 352
#  else
#    define __NR_process_vm_readv 310
#    define __NR_process_vm_writev 311
#  endif

static inline ssize_t process_vm_readv(pid_t pid,
                         const struct iovec *local_iov,
                         unsigned long liovcnt,
                         const struct iovec *remote_iov,
                         unsigned long riovcnt,
                         unsigned long flags) {
    return syscall(__NR_process_vm_readv, pid, local_iov, liovcnt, remote_iov, riovcnt, flags);
}

static inline ssize_t process_vm_writev(pid_t pid,
                         const struct iovec *local_iov,
                         unsigned long liovcnt,
                         const struct iovec *remote_iov,
                         unsigned long riovcnt,
                         unsigned long flags) {
    return syscall(__NR_process_vm_writev, pid, local_iov, liovcnt, remote_iov, riovcnt, flags);
}

#endif /*#ifndef HAVE_PROCESS_VM_READV */

extern MPID_Request * create_request(void * hdr, MPIDI_msg_sz_t hdr_sz,
                                            MPIU_Size_t nb);
#endif /*#if defined(_SMP_CMA_)*/

#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_)
void MPIDI_CH3I_SMP_send_comp(void *c_header,
        MPIDI_VC_t* vc, MPIDI_msg_sz_t nb, 
        smp_dma_flag_t dma_flag, smp_fallback_flag_t fallback);
#endif 

#if defined(_SMP_LIMIC_)
int limic_fd;
int g_smp_use_limic2 = 1;
int g_use_limic2_coll = 1;
static inline void adjust_lu_info(struct limic_user *lu, int old_len);
extern MPID_Request * create_request(void * hdr, MPIDI_msg_sz_t hdr_sz,
                                            MPIU_Size_t nb);
#endif /*#if defined(_SMP_LIMIC_)*/

extern int mv2_enable_shmem_collectives;
extern struct mv2_MPIDI_CH3I_RDMA_Process_t mv2_MPIDI_CH3I_RDMA_Process;

extern int MPIDI_Get_num_nodes();
extern int rdma_set_smp_parameters(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc);
extern void MPIDI_CH3I_SHMEM_COLL_Cleanup();

static int smpi_exchange_info(MPIDI_PG_t *pg);
static inline SEND_BUF_T *get_buf_from_pool (void);
static inline void send_buf_reclaim (void);
static inline void put_buf_to_pool (int, int);
static inline void link_buf_to_send_queue (int, int);

/*
 * called by sender after every successful write to cyclic buffer, in order to
 * set current flag, set data size, clear next flag and update local send  header
 * pointer.
 */
static inline void smpi_complete_send(unsigned int destination,
    unsigned int length, int data_sz,
    volatile void *ptr, volatile void *ptr_head, volatile void *ptr_flag)
{
    s_header_ptr_s[destination] += length + sizeof(int)*2;
    /* set next flag to free */
    *((volatile int *) ptr_head) = data_sz;
    *((volatile int *) ptr) = SMP_CBUF_FREE;
    WRITEBAR();
    /* set current flag to busy */
    *((volatile int *) ptr_flag) = SMP_CBUF_BUSY;
    WRITEBAR();
    avail[destination] -= length + sizeof(int)*2;
}

/*
 * called by receiver after every successful read from cyclic buffer, in order
 * to clear flag, update shared tail, update local receive header pointer.
 */
static inline void smpi_complete_recv(unsigned int from_grank,
    unsigned int my_id,
    unsigned int length)
{
    /* update shared tail */
    SMPI_SHARED_TAIL(from_grank, my_id) =
        s_header_ptr_r[from_grank] + length
        + sizeof(int)*2 - 1;
    volatile void *ptr_flag = (volatile void *)((unsigned long)g_smpi_shmem->pool +
            s_header_ptr_r[from_grank]);
    unsigned long header;
    header = s_header_ptr_r[from_grank];
    s_header_ptr_r[from_grank] =
        s_header_ptr_r[from_grank] +
        length + sizeof(int)*2;
    READBAR();
    if(header == SMPI_FIRST_R(from_grank, my_id)) {
        *(volatile int *)ptr_flag = SMP_CBUF_FREE;
        WRITEBAR();
    }
}

/*
 * called by sender before every send in order to check if enough room left in
 * cyclic buffer.
 */
static inline int smpi_check_avail(int rank, int len,
    volatile void **pptr_flag, smp_ctrl_avail_flag_t num)
{
    /* check if avail is less than data size */
    if(avail[rank] < num * (len + sizeof(int)*3)) {
        /* update local tail according to shared tail */
        if (s_header_ptr_s[rank] + len + sizeof(int)*3 >=
                SMPI_LAST_S(g_smpi.my_local_id, rank)) {
            /* check if the beginning of the cyclic buffer is already free */
            if(*(int *)((unsigned long)g_smpi_shmem->pool +
                SMPI_FIRST_S(g_smpi.my_local_id, rank))) {
                return 0;
            }
            s_tail_ptr_s[rank] =
                SMPI_SHARED_TAIL(g_smpi.my_local_id, rank);
            if (s_tail_ptr_s[rank] == 
                    SMPI_FIRST_S(g_smpi.my_local_id, rank)) {
                avail[rank] = SMPI_LAST_S(g_smpi.my_local_id, rank) -
                        s_header_ptr_s[rank];
                return 0;
            }
            s_header_ptr_s[rank] =
                SMPI_FIRST_S(g_smpi.my_local_id, rank);
            volatile void *ptr_flag;
            ptr_flag = *pptr_flag;
            *(volatile int *)ptr_flag = SMP_CBUF_END;
            WRITEBAR();
            ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
                    s_header_ptr_s[rank]);
            *pptr_flag = ptr_flag;
        } else {
            s_tail_ptr_s[rank] =
                SMPI_SHARED_TAIL(g_smpi.my_local_id, rank);
        }

        /* update avail */
        READBAR();
        avail[rank] = (s_tail_ptr_s[rank] >=
                s_header_ptr_s[rank] ?
                (s_tail_ptr_s[rank] - s_header_ptr_s[rank]) :
                (SMPI_LAST_S(g_smpi.my_local_id, rank) -
                        s_header_ptr_s[rank]));
        if(avail[rank] < len + sizeof(int)*3)
            return 0;
    }
    return 1;
}

static inline int MPIDI_CH3I_SMP_attach_shm_pool_inline();

void MPIDI_CH3I_SMP_cleanup() 
{ 
    /*clean up pool file*/
    if (g_smpi.send_buf_pool_ptr != NULL) {
        munmap(g_smpi.send_buf_pool_ptr, g_size_pool); 
    }
    if (g_smpi.fd_pool != -1) { 
        close(g_smpi.fd_pool);
        unlink(pool_file);
    } 
    if (pool_file != NULL) {
        MPIU_Free(pool_file);
    }
    g_smpi.send_buf_pool_ptr = NULL;
    g_smpi.fd_pool = -1;
    pool_file = NULL;

    /*clean up shmem file*/
    if (g_smpi.mmap_ptr != NULL) { 
        munmap((void *)g_smpi.mmap_ptr, g_size_shmem);        
    }
    if (g_smpi.fd != -1) { 
        close(g_smpi.fd);
        unlink(shmem_file);
    }
    if (shmem_file != NULL) { 
        MPIU_Free(shmem_file);
    }
    g_smpi.mmap_ptr = NULL;
    g_smpi.fd = -1;
    shmem_file = NULL;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_attach_shm_pool
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_attach_shm_pool()
{
    return MPIDI_CH3I_SMP_attach_shm_pool_inline();
}

static inline int MPIDI_CH3I_SMP_attach_shm_pool_inline()
{
    int mpi_errno = MPI_SUCCESS;
    int i, j;
    int pagesize = getpagesize();
    SEND_BUF_T *send_buf = NULL;
    volatile char tmpchar ATTRIBUTE((unused));

    g_smpi.send_buf_pool_ptr = mmap (0, g_size_pool,
        (PROT_READ | PROT_WRITE), (MAP_SHARED), g_smpi.fd_pool, 0);
    if (g_smpi.send_buf_pool_ptr == (void *) -1) {
       /* to clean up tmp shared file */
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", "%s: %s",
                "mmap", strerror(errno));
       goto cleanup_files;
    }

    s_buffer_head = (SEND_BUF_T **) MPIU_Malloc(sizeof(SEND_BUF_T *) * g_smpi.num_local_nodes);
    if(!s_buffer_head) {
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
       goto cleanup_files;
    }

    for(i=0; i < g_smpi.num_local_nodes; ++i) {
       s_buffer_head[i] = (SEND_BUF_T *)((unsigned long)g_smpi.send_buf_pool_ptr +
           SMPI_ALIGN((sizeof(SEND_BUF_T) + s_smp_block_size) * s_smp_num_send_buffer +
               pagesize) * i);

       if (((long) s_buffer_head[i] & (SMPI_CACHE_LINE_SIZE - 1)) != 0) {
          /* to clean up tmp shared file */
          mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s",
                   "error in shifting mmaped pool");
          goto cleanup_files;
       }
    }
    s_my_buffer_head = s_buffer_head[g_smpi.my_local_id];

    s_sh_buf_pool.free_head = 0;

    s_sh_buf_pool.send_queue = (int *) MPIU_Malloc(sizeof(int) * g_smpi.num_local_nodes);
    if(!s_sh_buf_pool.send_queue) {
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
       goto cleanup_files;
    }

    s_sh_buf_pool.tail = (int *) MPIU_Malloc(sizeof(int) * g_smpi.num_local_nodes);
    if(!s_sh_buf_pool.tail) {
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
       goto cleanup_files;
    }

    for (i = 0; i < g_smpi.num_local_nodes; ++i) {
       s_sh_buf_pool.send_queue[i] = s_sh_buf_pool.tail[i] = -1;
    }

    for (i = 0; i < s_smp_num_send_buffer; ++i) {
        send_buf = SMPI_MY_BUF_POOL_PTR(i); 
        send_buf->myindex = i;
        send_buf->next = i+1;
        send_buf->busy = 0;
        send_buf->len = 0;
        send_buf->has_next = 0;
        send_buf->msg_complete = 0;

        for (j = 0; j < s_smp_block_size; j += pagesize) {
            tmpchar = *((char *) &send_buf->buf + j);
        }
    }
    send_buf->next = -1;

    mv2_shmem_pool_init = 1;

#if defined(_ENABLE_CUDA_)
    /*registering the regions with the GPU device if CUDA support is enabled*/
    if (rdma_enable_cuda && cuda_initialized) {
        ibv_cuda_register (g_smpi.send_buf_pool_ptr, g_size_pool);
    }
#endif
 
fn_exit:
    return mpi_errno;

cleanup_files:
    MPIDI_CH3I_SMP_cleanup();
    if (mv2_enable_shmem_collectives){
        MPIDI_CH3I_SHMEM_COLL_Cleanup();
    }
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_process_header
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_CH3I_SMP_Process_header(MPIDI_VC_t* vc, MPIDI_CH3_Pkt_t* pkt, int* index, 
                  void *limic_header, void *cma_header, 
                  int *use_limic, int *use_cma)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_PROGRESS_HEADER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_PROGRESS_HEADER);
    int mpi_errno = MPI_SUCCESS;

    if (pkt->type == MPIDI_CH3_PKT_EAGER_SEND_CONTIG) {
        MPIDI_msg_sz_t buflen = s_total_bytes[vc->smp.local_nodes] -
            sizeof(MPIDI_CH3_Pkt_eager_send_t);
        if ((mpi_errno = MPIDI_CH3_PktHandler_EagerSend_Contig(
                        vc,
                        pkt, ((char *)pkt + MPIDI_CH3U_PKT_SIZE(pkt)),
                        &buflen,
                        &vc->smp.recv_active)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        if (!vc->smp.recv_active) {
            s_current_ptr[vc->smp.local_nodes] = NULL;
            s_current_bytes[vc->smp.local_nodes] = 0;        
            smpi_complete_recv(vc->smp.local_nodes,
                    g_smpi.my_local_id,
#if defined(_ENABLE_CUDA) && defined(HAVE_CUDA_IPC)
                    sizeof(MPIDI_CH3_Pkt_eager_send_t));
#else
                    s_total_bytes[vc->smp.local_nodes]);
#endif
            s_total_bytes[vc->smp.local_nodes] = 0;
        }
        goto fn_exit;
    }
#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_) 
    if (MPIDI_CH3_PKT_SMP_DMA_COMP == pkt->type)
    {
        /* convert to MPIDI_CH3_Pkt_limic_comp_t */
        MPIDI_CH3_Pkt_comp_t *lc_pkt = (MPIDI_CH3_Pkt_comp_t *)pkt;
        MPID_Request *sreq = (MPID_Request *)(lc_pkt->send_req_id);
        
        MPIDI_msg_sz_t nb = lc_pkt->nb;
        int complete = 0;

#if defined(_SMP_LIMIC_)
        if (lc_pkt->fallback) {
            g_smp_use_limic2 = 0;
            MPIDI_CH3_Rendezvous_push(vc, sreq);
        } else 
#endif
        {
            if (MPIDI_CH3I_Request_adjust_iov(sreq, nb)) {
                MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
                /* We have recived the expected fin message as part of the
                 * CMA/LiMIC based message transfer */
                MV2_DEC_NUM_POSTED_RECV();
            }
            MPIU_Assert(complete);
        }
   
        return mpi_errno;
    }
#endif 

    if (MPIDI_CH3_PKT_RNDV_R3_DATA == pkt->type)
    {
        MPIDI_CH3_Pkt_rndv_r3_data_t * pkt_header = (MPIDI_CH3_Pkt_rndv_r3_data_t*) pkt;

#if defined(_SMP_CMA_) 
        struct cma_header *c_header = (struct cma_header *)cma_header;
        /* This is transferred through cma, retrieve related info */
        if (pkt_header->csend_req_id) {
            *use_cma = 1;
            MPIU_Memcpy(c_header->remote, s_current_ptr[vc->smp.local_nodes], sizeof(struct iovec));
         
            s_current_ptr[vc->smp.local_nodes] = (void*)(
                (unsigned long) s_current_ptr[vc->smp.local_nodes]
                + sizeof(struct iovec));

            c_header->pid = *((pid_t *)s_current_ptr[vc->smp.local_nodes]);
            s_current_ptr[vc->smp.local_nodes] = (void*)(
                (unsigned long) s_current_ptr[vc->smp.local_nodes]
                + sizeof(pid_t));
            c_header->total_bytes = *((MPIDI_msg_sz_t *)s_current_ptr[vc->smp.local_nodes]);
            c_header->csend_req_id = (MPID_Request *)(pkt_header->csend_req_id);

            s_current_ptr[vc->smp.local_nodes] = (void*)(
                (unsigned long) s_current_ptr[vc->smp.local_nodes]
                + sizeof(MPIDI_msg_sz_t));
            s_current_bytes[vc->smp.local_nodes] = s_current_bytes[vc->smp.local_nodes] -
                sizeof(struct iovec) - sizeof(MPIDI_msg_sz_t) - sizeof(pid_t);
        } else {
#endif /*#if defined(_SMP_CMA_)*/

#if defined(_SMP_LIMIC_) 
        /* This is transferred through limic2, retrieve related info */
        struct limic_header *l_header = (struct limic_header *)limic_header; 
        if (pkt_header->send_req_id) {
            *use_limic = 1;
            MPIU_Memcpy(&(l_header->lu), s_current_ptr[vc->smp.local_nodes], sizeof(limic_user));
         
            s_current_ptr[vc->smp.local_nodes] = (void*)(
                (unsigned long) s_current_ptr[vc->smp.local_nodes]
                + sizeof(limic_user));

            l_header->total_bytes = *((MPIDI_msg_sz_t *)s_current_ptr[vc->smp.local_nodes]);
            l_header->send_req_id = (MPID_Request *)(pkt_header->send_req_id);

            s_current_ptr[vc->smp.local_nodes] = (void*)(
                (unsigned long) s_current_ptr[vc->smp.local_nodes]
                + sizeof(MPIDI_msg_sz_t));

            s_current_bytes[vc->smp.local_nodes] = s_current_bytes[vc->smp.local_nodes] -
                sizeof(struct limic_user) - sizeof(MPIDI_msg_sz_t);
        } else {
#endif /*#if defined(_SMP_LIMIC_)*/
            if ((*index = pkt_header->src.smp_index) == -1)
            {
                MPIR_ERR_SETFATALANDJUMP1(
                    mpi_errno,
                    MPI_ERR_OTHER,
                    "**fail",
                    "**fail %s",
                    "*index == -1"
                );
            }
#if defined(_SMP_LIMIC_) 
        }
#endif

#if defined(_SMP_CMA_)
        }
#endif
        vc->smp.recv_current_pkt_type = SMP_RNDV_MSG;

        MPID_Request* rreq = NULL;
        MPID_Request_get_ptr(((MPIDI_CH3_Pkt_rndv_r3_data_t*) pkt)->receiver_req_id, rreq);
        PRINT_DEBUG(DEBUG_RNDV_verbose>1, "R3 data received from: %d, rreq: %p\n", vc->pg_rank, rreq);
        vc->smp.recv_active = rreq;
        goto fn_exit;
    } else if (pkt->type == MPIDI_CH3_PKT_RPUT_FINISH) {
        PRINT_DEBUG(DEBUG_RNDV_verbose>1, "RPUT FINISH received from: %d\n", vc->pg_rank);
        MPIDI_CH3_Rendezvous_rput_finish(vc, (MPIDI_CH3_Pkt_rput_finish_t *) pkt);
        goto fn_exit;
    } else if (pkt->type == MPIDI_CH3_PKT_RGET_FINISH) {
        PRINT_DEBUG(DEBUG_RNDV_verbose>1, "RGET FINISH received from: %d\n", vc->pg_rank);
        MPIDI_CH3_Rendezvous_rget_send_finish(vc, (MPIDI_CH3_Pkt_rget_finish_t *) pkt);
        goto fn_exit;
    }

#if defined(CKPT)
    /*
     * Handle the MPIDI_CH3_PKT_CM_SUSPEND packet
     * for the shared memory channel
     */
    else if (pkt->type == MPIDI_CH3_PKT_CM_SUSPEND) {
		vc->mrail.suspended_rails_recv++;
		PRINT_DEBUG(DEBUG_CR_verbose>1, "%s (pid %ld):[%d <= %d]: get CM_SUSPEND vcstate=%d, send=%d,recv=%d\n",
            __func__, pthread_self(), MPIDI_Process.my_pg_rank, vc->pg_rank, vc->ch.state,
			vc->mrail.suspended_rails_send, vc->mrail.suspended_rails_recv );

		if( vc->mrail.suspended_rails_send > 0 && 
			vc->mrail.suspended_rails_recv > 0 ) {
			vc->ch.state = MPIDI_CH3I_VC_STATE_SUSPENDED;
			vc->mrail.suspended_rails_send = 0;
			vc->mrail.suspended_rails_recv = 0;
			PRINT_DEBUG(DEBUG_CR_verbose>1, "Suspend channel from %d to %d\n",
				MPIDI_Process.my_pg_rank, vc->pg_rank );
		}
	    goto fn_exit;
    }

    /*
     * Handle the MPIDI_CH3_PKT_CM_REACTIVATION_DONE packet
     * for the shared memory channel
     */
    else if (pkt->type == MPIDI_CH3_PKT_CM_REACTIVATION_DONE) {
		PRINT_DEBUG(DEBUG_CR_verbose>1, "(pid %ld):CM_REACT channel from %d to %d. vcstate=%d\n",
			        pthread_self(), MPIDI_Process.my_pg_rank, vc->pg_rank, vc->ch.state);
        if (vc->ch.state == MPIDI_CH3I_VC_STATE_SUSPENDED) {
            vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
            if (MPIDI_CH3I_SMP_SendQ_empty(vc)) {
                if (mv2_use_eager_fast_send) {
                    vc->eager_fast_fn = mv2_smp_fast_write_contig;
                }
            }
        }
        vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
        goto fn_exit;
    }
#endif /* defined(CKPT) */

    if (pkt->type != MPIDI_CH3_PKT_EAGER_SEND_CONTIG) {
        MPIDI_msg_sz_t buflen = sizeof(MPIDI_CH3_Pkt_t);

        if ((mpi_errno = MPIDI_CH3U_Handle_recv_pkt(
                        vc,
                        pkt, ((char *)pkt + MPIDI_CH3U_PKT_SIZE(pkt)),
                        &buflen,
                        &vc->smp.recv_active)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        vc->smp.recv_current_pkt_type = SMP_EAGER_MSG;
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_PROGRESS_HEADER);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_write_progress
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_write_progress(MPIDI_PG_t *pg)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_WRITE_PROGRESS);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_WRITE_PROGRESS);
    int mpi_errno = MPI_SUCCESS;
    int nb;
    int i = 0;
    MPIDI_VC_t *vc;
    int complete;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    /* track smp write progress polling for MPIT*/
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_write_progress_poll, 1);

    for (i=0; i < g_smpi.num_local_nodes; ++i)
    {
        MPIDI_PG_Get_vc(pg, g_smpi.l2g_rank[i], &vc);

#if defined(CKPT)
        /* Don't touch a suspended channel */
        if (vc->ch.state == MPIDI_CH3I_VC_STATE_SUSPENDED ||
            vc->ch.state == MPIDI_CH3I_VC_STATE_SUSPENDING)
            continue;
#endif /* defined(CKPT) */

/*
        if (vc->smp.send_active)
*/
        while (vc->smp.send_active != NULL) {
                MPID_Request *req = vc->smp.send_active;
                PRINT_DEBUG(DEBUG_SHM_verbose>1,
                        "smp send active, vc->rank: %d, req: %p, type: %d, ch.reqtype: %d\n",
                        vc->pg_rank, req, MPIDI_Request_get_type(req), req->ch.reqtype);

                if(req->dev.iov_offset >= req->dev.iov_count) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "**fail %s",
                    "req->dev.iov_offset >= req->dev.iov_count");
                }

#if defined(_ENABLE_CUDA_)
                if (rdma_enable_cuda && s_smp_cuda_pipeline) {
                    /*as it is all data, we check the first iov to determine if the buffer is on device*/
                    iov_isdev = is_device_buffer((void *) req->dev.iov[req->dev.iov_offset].MPL_IOV_BUF);
                }
#endif

                switch (req->ch.reqtype)
                {
                case REQUEST_RNDV_R3_HEADER:
                    vc->smp.send_current_pkt_type = SMP_RNDV_MSG;
                            MPIDI_CH3I_SMP_writev_rndv_header(
                                vc,
                                req->dev.iov + req->dev.iov_offset,
                                req->dev.iov_count - req->dev.iov_offset,
                                &nb
                            );
                        break;
                    case REQUEST_RNDV_R3_DATA:
#if defined(_ENABLE_CUDA_)
                            if (iov_isdev) { 
                               int is_cont = (vc->smp.send_current_pkt_type 
                                               == SMP_RNDV_MSG) ? 0 : 1;
                               mpi_errno = 
                                   MPIDI_CH3I_SMP_writev_rndv_data_cuda(vc, 
                                     req, 
                                     req->dev.iov + req->dev.iov_offset,
                                     req->dev.iov_count - req->dev.iov_offset,
                                     &nb,
                                     is_cont);
                            } else 
#endif
                            {
                               if (vc->smp.send_current_pkt_type == SMP_RNDV_MSG)
                               {
                                   mpi_errno = MPIDI_CH3I_SMP_writev_rndv_data(
                                       vc,
                                       req,
                                       req->dev.iov + req->dev.iov_offset,
                                       req->dev.iov_count - req->dev.iov_offset,
                                       &nb
                                  );
                               }
                               else
                               {
                                   MPIDI_CH3I_SMP_writev_rndv_data_cont(
                                       vc,
                                       req,
                                       req->dev.iov + req->dev.iov_offset,
                                       req->dev.iov_count - req->dev.iov_offset,
                                       &nb
                                   );
                               }
                            }
                        break;
                    default:
                            MPIDI_CH3I_SMP_writev(
                                vc,
                                req->dev.iov + req->dev.iov_offset,
                                req->dev.iov_count - req->dev.iov_offset,
                                &nb
                            );

                        break;
            }

            if (mpi_errno != MPI_SUCCESS)
            {
                MPIR_ERR_POP(mpi_errno);
            }

            PRINT_DEBUG(DEBUG_SHM_verbose>1, "req: %p, shm_writev returned %d\n", req, nb);

            if (nb > 0)
            {
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "Wrote R3 data, dest: %d, req: %p, bytes: %d\n",
                        vc->pg_rank, req, nb);
                    if (MPIDI_CH3I_Request_adjust_iov(req, nb))
                    {
                        /* Write operation complete */
#if defined(_ENABLE_CUDA_)
                        if (iov_isdev && (req->ch.reqtype == REQUEST_RNDV_R3_DATA)) {
                            /* dequeue the request so that the next one is processed 
                             * the actual request is completed when the copy completes */
                            req->ch.reqtype = REQUEST_NORMAL;
                            if(!MPIDI_CH3I_SMP_SendQ_empty(vc) ){
                                MPIDI_CH3I_SMP_SendQ_dequeue(vc);
                            }
                        } else 
#endif
                        {
                           if ((mpi_errno = MPIDI_CH3U_Handle_send_req(vc, req, &complete)) != MPI_SUCCESS)
                           {
                               MPIR_ERR_POP(mpi_errno);
                           }

                           if (complete) {
                              req->ch.reqtype = REQUEST_NORMAL;
							  if( !MPIDI_CH3I_SMP_SendQ_empty(vc) ){
                            	  MPIDI_CH3I_SMP_SendQ_dequeue(vc);
							  }
                              PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                                      "Dequeue request from sendq: %p, now head %p\n",
                                      req, vc->smp.sendq_head);
#ifdef CKPT
						      MPIDI_CH3I_MRAILI_Pkt_comm_header* p = 
								 (MPIDI_CH3I_MRAILI_Pkt_comm_header*)(&(req->dev.pending_pkt));
						      if( p->type >= MPIDI_CH3_PKT_CM_SUSPEND && 
							     p->type <= MPIDI_CH3_PKT_CR_REMOTE_UPDATE ){
							     PRINT_DEBUG(DEBUG_SHM_verbose>1, "%s [%d vc_%d]: imm-write msg %s(%d)\n", __func__,
							     MPIDI_Process.my_pg_rank, vc->pg_rank, 
							     MPIDI_CH3_Pkt_type_to_string[p->type],p->type );
						      }
						      if( p->type == MPIDI_CH3_PKT_CM_SUSPEND ){
							     vc->mrail.suspended_rails_send++;
						         //	printf("%s: [%d vc_%d]: imm-write SUSP_MSG, send=%d, recv=%d\n", 
						         //	__func__, MPIDI_Process.my_pg_rank, vc->pg_rank, 
						         //	vc->mrail.suspended_rails_send, vc->mrail.suspended_rails_recv);
							     if( vc->mrail.suspended_rails_send > 0 &&
								    vc->mrail.suspended_rails_recv > 0 )
							     {
								    vc->ch.state = MPIDI_CH3I_VC_STATE_SUSPENDED;
								    vc->mrail.suspended_rails_send = 0;
								    vc->mrail.suspended_rails_recv =0;
                    				PRINT_DEBUG(DEBUG_SHM_verbose>1, "%s[%d <= %d]:turn to SUSPENDED, \
                                       send-act=%p, sendq-head=%p\n", __func__, 
									   MPIDI_Process.my_pg_rank, vc->pg_rank, 
									   vc->smp.send_active,vc->smp.sendq_head);
							     }      
						      }
#endif
                           } else {
                             if (vc->smp.send_current_pkt_type == SMP_RNDV_MSG)
                                vc->smp.send_current_pkt_type = SMP_RNDV_MSG_CONT;
                           }
                        }
                        vc->smp.send_active = MPIDI_CH3I_SMP_SendQ_head(vc);
                    } else {
                        if (vc->smp.send_current_pkt_type == SMP_RNDV_MSG)
                            vc->smp.send_current_pkt_type = SMP_RNDV_MSG_CONT;

                        MPIDI_DBG_PRINTF((65, FCNAME,
                            "iovec updated by %d bytes but not complete",
                            nb));

                        if(req->dev.iov_offset >= req->dev.iov_count) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                    "**fail", "**fail %s",
                                    "req->dev.iov_offset >= req->dev.iov_count");
                        }

                        break;
                    }
            } else {
                MPIDI_DBG_PRINTF((65, FCNAME,
                        "shm_post_writev returned %d bytes",
                        nb));
                break;
            }
            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_write_progress_poll_success, 1);
        } /* while (vc->smp.send_active != NULL) */
    } /* for (i=0; i < g_smpi.num_local_nodes; ++i) */

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_WRITE_PROGRESS);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_read_progress
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_read_progress (MPIDI_PG_t* pg)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_READ_PROGRESS);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_READ_PROGRESS);
    int mpi_errno = MPI_SUCCESS;
    MPIDI_VC_t* vc = NULL;
    MPIDI_CH3_Pkt_t* pkt_head = NULL;
    size_t nb = 0;
    int complete = 0;
    int i = 0;
    int index = -1;
    static int skip = 0;

    /* track smp read progress polling for MPIT*/
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_read_progress_poll, 1);

#if defined(_SMP_LIMIC_)
    struct limic_header l_header;
#else
    int l_header;
#endif

#if defined(_SMP_CMA_)
    struct cma_header c_header;
#else
    int c_header;
#endif

    int use_limic = 0;
    int use_cma = 0;

#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    if(g_smp_priority_polling) {
    int from;
    int ptr = 0;
    int poll_flag = 0;  

    if(polling_set_p_head)
    {
        ptr = polling_set_p_head;
        is_fair_polling = 0;
    } else if (++skip >= g_smp_priority_factor) {
        ptr = polling_set_c_head;
        is_fair_polling = 1;
        skip = 0;
    }

    while(ptr) {
        from = (g_smpi.my_local_id + ptr) % g_smpi.num_local_nodes;
        MPIDI_PG_Get_vc(pg, g_smpi.l2g_rank[from], &vc);

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
        if (rdma_enable_cuda && vc->smp.local_rank == -1) {
            continue;
        }
#endif
        if (!vc->smp.recv_active)
        {
            MPIDI_CH3I_SMP_pull_header(vc, &pkt_head);

            if (pkt_head)
            {
                poll_flag = 1;
                use_cma = 0;
                use_limic = 0;

                PRINT_DEBUG(DEBUG_RNDV_verbose>1, "vc rank: %d, pkt_head: %p, limic: %d, cma: %d\n",
                                                   vc->pg_rank, pkt_head, use_limic, use_cma);
                MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_read_progress_poll_success, 1);
                mpi_errno = MPIDI_CH3I_SMP_Process_header(vc, pkt_head, &index, &l_header, &c_header,
                        &use_limic, &use_cma);

                if (mpi_errno!=MPI_SUCCESS)
                {
                    MPIR_ERR_POP(mpi_errno);
                }
                if(is_fair_polling) {
                    polling_counters[ptr]++;
                    if (polling_counters[ptr] == g_smp_polling_th) {
                        ENQUEUE_p(ptr);
                    }
                }
            }
        }

        if (vc->smp.recv_active)
        {
            struct MPID_Request * req ATTRIBUTE((unused)) = vc->smp.recv_active;
            PRINT_DEBUG(DEBUG_SHM_verbose>1,
                    "smp recv active, rank: %d, req: %p, type: %d, ch.reqtype: %d, pkt_type: %d\n",
                    vc->pg_rank, req, MPIDI_Request_get_type(req), req->ch.reqtype, vc->smp.recv_current_pkt_type);
            poll_flag = 1;
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda && s_smp_cuda_pipeline) {
                /*as it is all data, we check the first iov to determine if the buffer is on device*/
                iov_isdev = is_device_buffer((void *) vc->smp.recv_active->dev.iov
                        [vc->smp.recv_active->dev.iov_offset].MPL_IOV_BUF);
            }
#endif
            switch(vc->smp.recv_current_pkt_type)
            {
                case SMP_RNDV_MSG:
#if defined(_ENABLE_CUDA_)
                    if (iov_isdev) {
                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cuda(vc,
                                vc->smp.recv_active,
                                &vc->smp.recv_active->dev.iov
                                [vc->smp.recv_active->dev.iov_offset],
                                vc->smp.recv_active->dev.iov_count
                                - vc->smp.recv_active->dev.iov_offset,
                                index,
                                &nb,
                                0);
                    } else
#endif
                    {
                        smp_dma_flag_t dma_flag = SMP_DMA_NONE;
                        if (use_limic == 1)
                            dma_flag = SMP_DMA_LIMIC;
                        else if (use_cma == 1)
                            dma_flag = SMP_DMA_CMA;

                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv(
                                vc,
                                &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                                vc->smp.recv_active->dev.iov_count - vc->smp.recv_active->dev.iov_offset,
                                index, &l_header, &c_header, &nb, dma_flag);
                    }
                    break;
                case SMP_RNDV_MSG_CONT:
#if defined(_ENABLE_CUDA_)
                    if (iov_isdev) {
                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cuda(vc,
                                vc->smp.recv_active,
                                &vc->smp.recv_active->dev.iov
                                [vc->smp.recv_active->dev.iov_offset],
                                vc->smp.recv_active->dev.iov_count
                                - vc->smp.recv_active->dev.iov_offset,
                                index,
                                &nb,
                                1);
                    } else
#endif
                    {
                        smp_dma_flag_t dma_flag = SMP_DMA_NONE;
                        if (vc->smp.use_limic == 1)
                            dma_flag = SMP_DMA_LIMIC;
                        else if (vc->smp.use_cma == 1)
                            dma_flag = SMP_DMA_CMA;
                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cont(vc,
                                &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                                vc->smp.recv_active->dev.iov_count -
                                vc->smp.recv_active->dev.iov_offset, index, &vc->smp.current_l_header,
                                &vc->smp.current_c_header,
                                &nb, dma_flag);
                    }
                    break;
                default:
                    mpi_errno = MPIDI_CH3I_SMP_readv(vc,
                            &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                            vc->smp.recv_active->dev.iov_count -
                            vc->smp.recv_active->dev.iov_offset, &nb);
                    break;
            }

            if (mpi_errno)
            {
                MPIR_ERR_POP(mpi_errno);
            }

            PRINT_DEBUG(DEBUG_SHM_verbose>1, "request to fill: iovlen %ld, iov[0].len %ld, [1] %ld, nb %ld\n",
                    vc->smp.recv_active->dev.iov_count - vc->smp.recv_active->dev.iov_offset,
                    vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset].MPL_IOV_LEN,
                    vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset + 1].MPL_IOV_LEN, nb);

            if (nb > 0) {
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "Read %lu bytes from rank: %d, req: %p, cma: %d, limic: %d\n",
                        nb, vc->pg_rank, vc->smp.recv_active, use_cma, use_limic);
                if (MPIDI_CH3I_Request_adjust_iov(vc->smp.recv_active, nb)) {
#if defined(_ENABLE_CUDA_)
                    if (iov_isdev
                            && (vc->smp.recv_current_pkt_type == SMP_RNDV_MSG
                                || vc->smp.recv_current_pkt_type == SMP_RNDV_MSG_CONT)) {
                        /*set recv_active to NULL so that next request is picked up*/
                        vc->smp.recv_active = NULL;
                    } else
#endif
                    {
                        PRINT_DEBUG(DEBUG_SHM_verbose>1, "adjust iov finished, handle req\n");
                        mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, vc->smp.recv_active, &complete);
                        PRINT_DEBUG(DEBUG_SHM_verbose>1, "finished handle req, complete %d\n", complete);

                        if(mpi_errno) MPIR_ERR_POP(mpi_errno);

                        while(vc->smp.recv_current_pkt_type==SMP_EAGER_MSG && !complete) {
                            /* continue to fill request */
                            mpi_errno = MPIDI_CH3I_SMP_readv(vc,
                                    &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                                    vc->smp.recv_active->dev.iov_count -
                                    vc->smp.recv_active->dev.iov_offset, &nb);

                            if(!MPIDI_CH3I_Request_adjust_iov(vc->smp.recv_active, nb)) {
                                goto fn_exit;
                            }
                            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, vc->smp.recv_active, &complete);
                            if(mpi_errno) MPIR_ERR_POP(mpi_errno);
                        }

                        if (complete) {
#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_)
                            /* send completion message with sender's send request
                             * and number of bytes received.
                             * header type is MPIDI_CH3_PKT_LIMIC_COMP
                             */
                            if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG) {
                                if(use_limic)
                                    MPIDI_CH3I_SMP_send_comp(&l_header, vc, nb, SMP_DMA_LIMIC, NO_FALLBACK);
                                else if(use_cma)
                                    MPIDI_CH3I_SMP_send_comp(&c_header, vc, nb, SMP_DMA_CMA, NO_FALLBACK);
                            } else if (vc->smp.recv_current_pkt_type == SMP_RNDV_MSG_CONT){
                                if(vc->smp.use_limic){
                                    vc->smp.current_nb += nb;
                                    MPIDI_CH3I_SMP_send_comp(&vc->smp.current_l_header, vc, vc->smp.current_nb, SMP_DMA_LIMIC, NO_FALLBACK);
                                }
                                else if(vc->smp.use_cma){
                                    vc->smp.current_cnb += nb;
                                    MPIDI_CH3I_SMP_send_comp(&vc->smp.current_c_header, vc, vc->smp.current_cnb, SMP_DMA_CMA, NO_FALLBACK);
                                }
                            }
#endif
                            vc->smp.recv_active = NULL;
                        } else {
                            if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG) {
                                vc->smp.recv_current_pkt_type = SMP_RNDV_MSG_CONT;

                                vc->smp.use_limic = 0;
                                vc->smp.use_cma = 0;
#if defined(_SMP_LIMIC_)
                                if (use_limic) {
                                    vc->smp.current_l_header = l_header;
                                    vc->smp.current_nb = nb;
                                    vc->smp.use_limic = 1;
                                }
#endif

#if defined(_SMP_CMA_)
                                if (use_cma) {
                                    vc->smp.current_c_header = c_header;
                                    vc->smp.current_cnb = nb;
                                    vc->smp.use_cma = 1;
                                }
#endif
                            }
#if defined(_SMP_CMA_) || defined(_SMP_LIMIC_)
                            else {
#if defined(_SMP_LIMIC_)
                                if (vc->smp.use_limic) {
                                    vc->smp.current_nb += nb;
                                }
#endif
#if defined(_SMP_CMA_)
                                if (vc->smp.use_cma) {
                                    vc->smp.current_cnb += nb;
                                }
#endif
                            }
#endif
                        }
                    }
                } else {
#if defined(_SMP_CMA_)
                    MPIU_Assert(vc->smp.recv_current_pkt_type != SMP_RNDV_MSG ||
                            !use_cma);
#endif
#if defined(_SMP_LIMIC_)
                    MPIU_Assert(vc->smp.recv_current_pkt_type != SMP_RNDV_MSG ||
                            !use_limic);
#endif
                    if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG) {
                        vc->smp.recv_current_pkt_type = SMP_RNDV_MSG_CONT;

                        vc->smp.use_limic = 0;
                        vc->smp.use_cma   = 0;
#if defined(_SMP_CMA_)
                        if (use_cma) {
                            vc->smp.use_cma = 1;
                            vc->smp.current_c_header = c_header;
                            vc->smp.current_cnb = nb;
                        }
#endif

#if defined(_SMP_LIMIC_)
                        if (use_limic) {
                            vc->smp.use_limic = 1;
                            vc->smp.current_l_header = l_header;
                            vc->smp.current_nb = nb;
                        }
#endif
                    }
                }
            }
            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_read_progress_poll_success, 1);
        }

        if(is_fair_polling) {
            ptr = polling_set_c[ptr].next;
        } else {
            ptr = polling_set_p[ptr].next;
            if(!ptr) {
                if(poll_flag)
                {
                    ptr = 0;
                } else if (++skip >= g_smp_priority_factor) {
                    ptr = polling_set_c_head;
                    is_fair_polling = 1;
                    skip = 0;
                }
            } 
        }
    }
    } else {
    int from;

    for (i=1; i < g_smpi.num_local_nodes; ++i)
    {
        from = (g_smpi.my_local_id + i) % g_smpi.num_local_nodes;

        MPIDI_PG_Get_vc(pg, g_smpi.l2g_rank[from], &vc);

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
        if (rdma_enable_cuda && vc->smp.local_rank == -1) {
            continue;
        }
#endif

        if (!vc->smp.recv_active)
        {
            MPIDI_CH3I_SMP_pull_header(vc, &pkt_head);

            if (pkt_head)
            {
                use_cma = 0;
                use_limic = 0;

                mpi_errno = MPIDI_CH3I_SMP_Process_header(vc, pkt_head, &index, &l_header, &c_header,
                        &use_limic, &use_cma);

                if (mpi_errno!=MPI_SUCCESS)
                {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
        }

        if (vc->smp.recv_active)
        {
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda && s_smp_cuda_pipeline) {
                /*as it is all data, we check the first iov to determine if the buffer is on device*/
                iov_isdev = is_device_buffer((void *) vc->smp.recv_active->dev.iov
                           [vc->smp.recv_active->dev.iov_offset].MPL_IOV_BUF);
            }
#endif

            switch(vc->smp.recv_current_pkt_type)
            {
            case SMP_RNDV_MSG:
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cuda(vc,
                            vc->smp.recv_active,
                            &vc->smp.recv_active->dev.iov
                                [vc->smp.recv_active->dev.iov_offset],
                            vc->smp.recv_active->dev.iov_count
                                - vc->smp.recv_active->dev.iov_offset,
                            index,
                            &nb,
                            0);
                } else
#endif
                {
                    smp_dma_flag_t dma_flag = SMP_DMA_NONE;
                    if (use_limic == 1)
                        dma_flag = SMP_DMA_LIMIC;
                    else if (use_cma == 1)
                        dma_flag = SMP_DMA_CMA;

                    mpi_errno = MPIDI_CH3I_SMP_readv_rndv(
                            vc,
                            &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                            vc->smp.recv_active->dev.iov_count - vc->smp.recv_active->dev.iov_offset,
                            index, &l_header, &c_header, &nb, dma_flag);
                }
                break;
            case SMP_RNDV_MSG_CONT:
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                        mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cuda(vc,
                            vc->smp.recv_active,
                            &vc->smp.recv_active->dev.iov
                                [vc->smp.recv_active->dev.iov_offset],
                            vc->smp.recv_active->dev.iov_count
                                - vc->smp.recv_active->dev.iov_offset,
                            index,
                            &nb,
                            1);
                } else
#endif
                {
                    smp_dma_flag_t dma_flag = SMP_DMA_NONE;
                    if (vc->smp.use_limic == 1)
                        dma_flag = SMP_DMA_LIMIC;
                    else if (vc->smp.use_cma == 1)
                        dma_flag = SMP_DMA_CMA;
                    mpi_errno = MPIDI_CH3I_SMP_readv_rndv_cont(vc,
                            &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                            vc->smp.recv_active->dev.iov_count -
                            vc->smp.recv_active->dev.iov_offset, index, &vc->smp.current_l_header,
                            &vc->smp.current_c_header,
                            &nb, dma_flag);
                }
                break;
            default:
                        mpi_errno = MPIDI_CH3I_SMP_readv(vc,
                        &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                        vc->smp.recv_active->dev.iov_count -
                        vc->smp.recv_active->dev.iov_offset, &nb);
                        break;
            }

            if (mpi_errno)
            {
                MPIR_ERR_POP(mpi_errno);
            }
        PRINT_DEBUG(DEBUG_SHM_verbose>1, "request to fill: iovlen %ld, iov[0].len %ld, [1] %ld, nb %ld\n",
            vc->smp.recv_active->dev.iov_count - vc->smp.recv_active->dev.iov_offset,
            vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset].MPL_IOV_LEN,
            vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset + 1].MPL_IOV_LEN, nb);
        if (nb > 0) {
        if (MPIDI_CH3I_Request_adjust_iov(vc->smp.recv_active, nb)) {
#if defined(_ENABLE_CUDA_)
           if (iov_isdev
               && (vc->smp.recv_current_pkt_type == SMP_RNDV_MSG
                   || vc->smp.recv_current_pkt_type == SMP_RNDV_MSG_CONT)) {
               /*set recv_active to NULL so that next request is picked up*/
               vc->smp.recv_active = NULL;
           } else
#endif
           {
               PRINT_DEBUG(DEBUG_SHM_verbose>1, "adjust iov finished, handle req\n");
               mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, vc->smp.recv_active, &complete);
               PRINT_DEBUG(DEBUG_SHM_verbose>1, "finished handle req, complete %d\n", complete);

               if(mpi_errno) MPIR_ERR_POP(mpi_errno);

               while(vc->smp.recv_current_pkt_type==SMP_EAGER_MSG && !complete) {
                   /* fill request */
                   mpi_errno = MPIDI_CH3I_SMP_readv(vc,
                           &vc->smp.recv_active->dev.iov[vc->smp.recv_active->dev.iov_offset],
                           vc->smp.recv_active->dev.iov_count -
                           vc->smp.recv_active->dev.iov_offset, &nb);

                   if(!MPIDI_CH3I_Request_adjust_iov(vc->smp.recv_active, nb)) {
                       goto fn_exit;
                   }
                   mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, vc->smp.recv_active, &complete);
                   if(mpi_errno) MPIR_ERR_POP(mpi_errno);
               }

               if (complete) {
#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_)
                   /* send completion message with sender's send request
                    * and number of bytes received.
                    * header type is MPIDI_CH3_PKT_LIMIC_COMP
                    */
                   if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG)  {
                       if(use_limic)
                           MPIDI_CH3I_SMP_send_comp(&l_header, vc, nb, SMP_DMA_LIMIC, NO_FALLBACK);
                       else if(use_cma)
                           MPIDI_CH3I_SMP_send_comp(&c_header, vc, nb, SMP_DMA_CMA, NO_FALLBACK);
                   } else if (vc->smp.recv_current_pkt_type == SMP_RNDV_MSG_CONT){
                       if(vc->smp.use_limic){
                           vc->smp.current_nb += nb;
                           MPIDI_CH3I_SMP_send_comp(&vc->smp.current_l_header, vc, vc->smp.current_nb, SMP_DMA_LIMIC, NO_FALLBACK);
                       }
                       else if(vc->smp.use_cma){
                           vc->smp.current_cnb += nb;
                           MPIDI_CH3I_SMP_send_comp(&vc->smp.current_c_header, vc, vc->smp.current_cnb, SMP_DMA_CMA, NO_FALLBACK);
                       }
                   }

#endif

                   vc->smp.recv_active = NULL;
               } else {
                   if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG) {
                       vc->smp.recv_current_pkt_type = SMP_RNDV_MSG_CONT;

                       vc->smp.use_limic = 0;
                       vc->smp.use_cma = 0;

#if defined(_SMP_LIMIC_)
                       if (use_limic) {
                           vc->smp.current_l_header = l_header;
                           vc->smp.current_nb = nb;
                           vc->smp.use_limic = 1;
                       } 
#endif
#if defined(_SMP_CMA_)
                       if (use_cma) {
                           vc->smp.current_c_header = c_header;
                           vc->smp.current_cnb = nb;
                           vc->smp.use_cma = 1;
                       } 
#endif

                   }

#if defined(_SMP_CMA_) || defined(_SMP_LIMIC_)
                   else {
#if defined(_SMP_LIMIC_)
                       if (vc->smp.use_limic) {
                           vc->smp.current_nb += nb;
                       }
#endif
#if defined(_SMP_CMA_)
                       if (vc->smp.use_cma) {
                           vc->smp.current_cnb += nb;
                       }
#endif
                   }
#endif
               }
           }
        } else {
#if defined(_SMP_CMA_)
            MPIU_Assert(vc->smp.recv_current_pkt_type != SMP_RNDV_MSG ||
                    !use_cma);
#endif

#if defined(_SMP_LIMIC_)
            MPIU_Assert(vc->smp.recv_current_pkt_type != SMP_RNDV_MSG ||
                    !use_limic);
#endif

            if(vc->smp.recv_current_pkt_type == SMP_RNDV_MSG) {
                vc->smp.recv_current_pkt_type = SMP_RNDV_MSG_CONT;

                vc->smp.use_limic = 0;
                vc->smp.use_cma = 0;

#if defined(_SMP_CMA_) 
                if (use_cma) {
                    vc->smp.use_cma = 1;
                    vc->smp.current_c_header = c_header;
                    vc->smp.current_cnb = nb;
                }
#endif

#if defined(_SMP_LIMIC_)
                if (use_limic) {
                    vc->smp.use_limic = 1;
                    vc->smp.current_l_header = l_header;
                    vc->smp.current_nb = nb;
                } 
#endif
            }
        }
        }
        }
    }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_READ_PROGRESS);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

void MPIDI_CH3I_SMP_unlink() 
{ 
    /*clean up pool file*/
    if (g_smpi.fd_pool != -1) { 
        unlink(pool_file);
    } 
    if (pool_file != NULL) {
        MPIU_Free(pool_file);
    }
    pool_file = NULL;

    /*clean up shmem file*/
    if (g_smpi.fd != -1) { 
        unlink(shmem_file);
    }
    if (shmem_file != NULL) { 
        MPIU_Free(shmem_file);
    }
    shmem_file = NULL;
}

void MPIDI_CH3I_set_smp_only()
{
    char *value;

    g_smpi.only_one_device = 0;
    SMP_ONLY = 0;
    if (MPIDI_CH3I_Process.has_dpm) {
        return;
    }

    if ((value = getenv("MV2_USE_SHARED_MEM")) != NULL) {
        rdma_use_smp = !!atoi(value);
    }

    if ((value = getenv("MV2_USE_BLOCKING")) != NULL) {
        rdma_use_blocking = !!atoi(value);
    }
    /* Disable SMP if blocking is enabled */
    if (rdma_use_blocking) {
        rdma_use_smp = 0;
    }

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    if (!rdma_use_smp) {
        rdma_cuda_smp_ipc = 0;
    }
#endif

    if (MPIDI_Get_num_nodes() == 1) {
        if(!rdma_use_smp || rdma_use_blocking) {
            return;
        }
        g_smpi.only_one_device = 1;
        SMP_ONLY = 1;
    }
}

void MPIDI_CH3I_SMP_Init_VC(MPIDI_VC_t *vc)
{
    /*initialize RNDV parameter*/
    vc->mrail.sreq_head = NULL;
    vc->mrail.sreq_tail = NULL;
    vc->mrail.nextflow  = NULL;
    vc->mrail.inflow    = 0;
#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    vc->mrail.cudaipc_sreq_head = NULL;
    vc->mrail.cudaipc_sreq_tail = NULL;
#endif
}

#if defined(_SMP_CMA_)
ssize_t check_cma_usability (pid_t remote_pid, char * rbuffer)
{
    char buffer = 1;
    struct iovec local[1];
    struct iovec remote[1];

    local[0].iov_base = &buffer;
    local[0].iov_len = 1;
    remote[0].iov_base = rbuffer;
    remote[0].iov_len = 1;

    return process_vm_writev(remote_pid, local, 1, remote, 1, 0);
}
#endif /* defined(_SMP_CMA_) */

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_SHMEM_Sync
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_SHMEM_Sync(volatile int * volatile bar_array, int my_local_id, int num_local_procs)
{
    int mpi_errno = MPI_SUCCESS;
    int i = 0, wait = 0, pid = 0;

    /* Sanity check */
    while (bar_array == NULL);

    if (0 == my_local_id) {
        PRINT_DEBUG(DEBUG_CM_verbose>0, "Before barrier\n");
        wait = 1;
        while (wait) {
            wait = 0;
            for (i = 1; i < num_local_procs; ++i) {
                if (bar_array[i] == 0) {
                    wait = 1;
                }
            }
        }

        PRINT_DEBUG(DEBUG_CM_verbose>1, "Non-root processes have arrived\n");
        pid = getpid();
        if (0 == pid) {
            mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                    FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                    "getpid", strerror(errno));
            return mpi_errno;
        }

        bar_array[my_local_id] = pid;
        WRITEBAR();
    } else {
        PRINT_DEBUG(DEBUG_CM_verbose>1, "Before barrier\n");
        while (bar_array[0] != 0);
        while (bar_array[0] == 0) {
            bar_array[my_local_id] = getpid();
            WRITEBAR();
        }
        PRINT_DEBUG(DEBUG_CM_verbose>1, "Root has arrived\n");
        for (i = 0; i < num_local_procs; ++i) {
            if (bar_array[i] <= 0) {
                mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "getpid", strerror(errno));
                return mpi_errno;
            }
        }
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Barrier done\n");

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_Unlink_region
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_Unlink_region(MPIDI_PG_t *pg)
{
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPIDI_CH3I_CM_UNLINK_REGION);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_CM_UNLINK_REGION);

    /* Unlink */
    if (pg->ch.mrail->cm_shmem_fd != -1) {
        unlink(pg->ch.mrail->cm_shmem_file);
        pg->ch.mrail->cm_shmem_fd         = -1;
        pg->ch.mrail->cm_shmem_mmap_ptr   = NULL;
    }
    /* Free filename variable */
    if (pg->ch.mrail->cm_shmem_file != NULL) {
        MPIU_Free(pg->ch.mrail->cm_shmem_file);
        pg->ch.mrail->cm_shmem_file       = NULL;
    }

    MPIDI_FUNC_EXIT(MPIDI_CH3I_CM_UNLINK_REGION);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_Destroy_region
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_Destroy_region(MPIDI_PG_t *pg)
{
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPIDI_CH3I_CM_DESTROY_REGION);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_CM_DESTROY_REGION);

    /* Unmap UD_CM SHMEM region */
    if (pg->ch.mrail->cm_shmem_mmap_ptr != NULL) {
        munmap((void *)pg->ch.mrail->cm_shmem_mmap_ptr, cm_shmem_file_size);
    }
    /* Unlink and close */
    if (pg->ch.mrail->cm_shmem_fd != -1) {
        close(pg->ch.mrail->cm_shmem_fd);
        unlink(pg->ch.mrail->cm_shmem_file);
    }
    /* Free filename variable */
    if (pg->ch.mrail->cm_shmem_file != NULL) {
        MPIU_Free(pg->ch.mrail->cm_shmem_file);
    }
    pg->ch.mrail->cm_shmem_fd         = -1;
    pg->ch.mrail->cm_shmem_file       = NULL;
    pg->ch.mrail->cm_shmem_mmap_ptr   = NULL;

    MPIDI_FUNC_EXIT(MPIDI_CH3I_CM_DESTROY_REGION);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_Create_region
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_Create_region(MPIDI_PG_t *pg)
{
    int pid_len = 0, ud_cm_len = 0, ud_len = 0;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPIDI_CH3I_CM_CREATE_REGION);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_CM_CREATE_REGION);

    /* Identify size of shared mem file size for storing UD CM info */
    pid_len = sizeof(int) * pg->ch.num_local_processes;
    pid_len = (pid_len + SMPI_CACHE_LINE_SIZE) -
                (pid_len % SMPI_CACHE_LINE_SIZE);
    ud_cm_len = sizeof(MPIDI_CH3I_MRAIL_UD_CM_t) * pg->size;
    ud_cm_len = (ud_cm_len + SMPI_CACHE_LINE_SIZE) -
                (ud_cm_len % SMPI_CACHE_LINE_SIZE);
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        ud_len = (sizeof(mv2_ud_exch_info_t) * pg->size * rdma_num_hcas);
        ud_len = (ud_len + SMPI_CACHE_LINE_SIZE) -
                    (ud_len % SMPI_CACHE_LINE_SIZE);
    }
#endif /* _ENABLE_UD_ */
    cm_shmem_file_size = pid_len + ud_cm_len + ud_len;

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Setting up shmem segment of size %ld\n",
                cm_shmem_file_size);
    /* Call helper function to create shmem region */
    mpi_errno = MPIDI_CH3I_SHMEM_Helper_fn(pg, pg->ch.local_process_id,
                                            &pg->ch.mrail->cm_shmem_file,
                                            "cm_shmem",
                                            &pg->ch.mrail->cm_shmem_fd,
                                            cm_shmem_file_size);
    if (mpi_errno != MPI_SUCCESS) {
       MPIR_ERR_POP(mpi_errno);
    }

    /* Mmap UD_CM SHMEM region */
    pg->ch.mrail->cm_shmem_mmap_ptr = mmap(0, cm_shmem_file_size,
                                            (PROT_READ | PROT_WRITE),
                                            (MAP_SHARED),
                                            pg->ch.mrail->cm_shmem_fd, 0);
    if (pg->ch.mrail->cm_shmem_mmap_ptr == (void *) -1) {
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                                         FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                                         "%s: %s", "mmap", strerror(errno));
        goto fn_fail;
    }

    /* Assign SHMEM region to accessible variable */
    pg->ch.mrail->cm_shmem.pid = (int*)
                ((char*)pg->ch.mrail->cm_shmem_mmap_ptr);
    pg->ch.mrail->cm_shmem.ud_cm = (MPIDI_CH3I_MRAIL_UD_CM_t*)
                ((char*)pg->ch.mrail->cm_shmem_mmap_ptr + pid_len);
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        int i = 0, j = 0;
        MPIU_Assert(rdma_num_hcas>0);
        char *base_addr = (char*)pg->ch.mrail->cm_shmem_mmap_ptr +
                            pid_len + ud_cm_len;
        for (i = 0; i < pg->size; i++) {
            pg->ch.mrail->cm_shmem.remote_ud_info[i] = (mv2_ud_exch_info_t *)
                ((char*)base_addr + i*rdma_num_hcas*sizeof(mv2_ud_exch_info_t));
            /* Initialize UD Data SHMEM region */
            for (j = 0; j < rdma_num_hcas; ++j) {
                pg->ch.mrail->cm_shmem.remote_ud_info[i][j].lid = UINT16_MAX;
                pg->ch.mrail->cm_shmem.remote_ud_info[i][j].qpn = UINT32_MAX;
            }
        }
        PRINT_DEBUG(DEBUG_CM_verbose>1, "cm_shmem.pid = %p, cm_shmem.ud_cm = %p,"
                " cm_shmem.remote_ud_info = %p\n", pg->ch.mrail->cm_shmem.pid,
                pg->ch.mrail->cm_shmem.ud_cm,
                pg->ch.mrail->cm_shmem.remote_ud_info);
    }
#else
    PRINT_DEBUG(DEBUG_CM_verbose>1, "cm_shmem.pid = %p, cm_shmem.ud_cm = %p\n",
                pg->ch.mrail->cm_shmem.pid, pg->ch.mrail->cm_shmem.ud_cm);
#endif /* _ENABLE_UD_ */

    /* Initialize UD_CM SHMEM region */
    if (pg->ch.local_process_id == 0) {
        MPIU_Memset(pg->ch.mrail->cm_shmem.ud_cm, 0, ud_cm_len);
    }

    /* Synchronization barrier */
    mpi_errno = MPIDI_CH3I_CM_SHMEM_Sync(pg->ch.mrail->cm_shmem.pid,
                                            pg->ch.local_process_id,
                                            pg->ch.num_local_processes);
    if (mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }
    /* Unlinking shared memory files*/
    MPIDI_CH3I_CM_Unlink_region(pg);

fn_exit:
    MPIDI_FUNC_EXIT(MPIDI_CH3I_CM_CREATE_REGION);
    return mpi_errno;

fn_fail:
    mpi_errno = MPIDI_CH3I_CM_Destroy_region(pg);
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_init(MPIDI_PG_t *pg)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_INIT);
    int mpi_errno = MPI_SUCCESS;
    unsigned int i;
    size_t sh_size, pid_len, st_len;
    int pagesize = getpagesize();
    volatile struct shared_mem * shmem;

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    int cu_ipc_offset, cu_ipc_len;
#endif
#if defined(_SMP_CMA_)
    size_t cma_test_buffer_offset;
#endif /* defined(_SMP_CMA_) */
#if defined(__x86_64__)
    volatile char tmpchar ATTRIBUTE((unused));
#endif /* defined(__x86_64__) */

    /* Set SMP params based on architecture */
    rdma_set_smp_parameters(&mv2_MPIDI_CH3I_RDMA_Process);
    
    if(rdma_use_blocking) {
        /* blocking is enabled, so
         * automatically disable
         * shared memory */
        return MPI_SUCCESS;
    }

    if (!rdma_use_smp) {
        return MPI_SUCCESS;
    }

    /*
     * Do the initializations here. These will be needed on restart
     * after a checkpoint has been taken.
     */
    if ((mpi_errno = smpi_exchange_info(pg)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "finished exchange info\n");

    /* Convert to bytes */
    g_smp_eagersize = g_smp_eagersize + 1;

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "smp eager size %d\n, smp queue length %zu\n",
                g_smp_eagersize, s_smp_queue_length);

    if (g_smp_eagersize > s_smp_queue_length / 2) {
       MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
           "**fail %s", "MV2_SMP_EAGERSIZE should not exceed half of "
           "MV2_SMP_QUEUE_LENGTH. Note that MV2_SMP_EAGERSIZE "
           "and MV2_SMP_QUEUE_LENGTH are set in KBytes.");
    }

    /* Initialize variables before setting up shmem regions */
    g_smpi.fd = -1;
    g_smpi.fd_pool = -1; 
    g_smpi.mmap_ptr = NULL; 
    g_smpi.send_buf_pool_ptr = NULL;
    g_smpi.available_queue_length =
          (s_smp_queue_length - g_smp_eagersize - sizeof(size_t));

    /* Compute the size of shmem files */
    pid_len = g_smpi.num_local_nodes * sizeof(size_t);
    /* pid_len need to be padded to cache aligned, in order to make sure the
     * following flow control structures cache aligned.
     */
    pid_len = pid_len + SMPI_CACHE_LINE_SIZE - (pid_len % SMPI_CACHE_LINE_SIZE);
    st_len = sizeof(smpi_shared_tails) * g_smpi.num_local_nodes * 
                (g_smpi.num_local_nodes - 1);
    sh_size = sizeof(struct shared_mem) + pid_len 
                + SMPI_ALIGN(st_len) + SMPI_CACHE_LINE_SIZE * 3;

    g_size_shmem = (SMPI_CACHE_LINE_SIZE + sh_size + pagesize 
          + (g_smpi.num_local_nodes * (g_smpi.num_local_nodes - 1) 
          * (SMPI_ALIGN(s_smp_queue_length + pagesize))));

    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_total_buffer, 
            ((g_smpi.num_local_nodes - 1) 
             * (SMPI_ALIGN(s_smp_queue_length + pagesize))));
           
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_avail_buffer, 
            ((g_smpi.num_local_nodes - 1) 
             * (SMPI_ALIGN(s_smp_queue_length + pagesize))));

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    cu_ipc_len = sizeof(smpi_cu_ipc_attr) * g_smpi.num_local_nodes *
        (g_smpi.num_local_nodes - 1);
    cu_ipc_offset = g_size_shmem;
    g_size_shmem += SMPI_ALIGN(cu_ipc_len);

#endif
#if defined(_SMP_CMA_)
    cma_test_buffer_offset = g_size_shmem;
    g_size_shmem += SMPI_ALIGN(1);
#endif /* defined(_SMP_CMA_) */

    g_size_pool =
    SMPI_ALIGN ((sizeof (SEND_BUF_T) + s_smp_block_size) 
                * s_smp_num_send_buffer + pagesize) 
                * g_smpi.num_local_nodes + SMPI_CACHE_LINE_SIZE;
	
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_total_buffer, 
            SMPI_ALIGN ((sizeof (SEND_BUF_T) + s_smp_block_size)
                * s_smp_num_send_buffer + pagesize));
                
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, 
            SMPI_ALIGN ((sizeof (SEND_BUF_T) + s_smp_block_size)
                * s_smp_num_send_buffer + pagesize));

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "size_shmem=%zu, size_pool = %zu\n", 
                g_size_shmem, g_size_pool);

    /* Call helper function to create shmem region */
    mpi_errno = MPIDI_CH3I_SHMEM_Helper_fn(pg, g_smpi.my_local_id, &shmem_file,
                                        "ib_shmem", &g_smpi.fd, g_size_shmem);
    if (mpi_errno != MPI_SUCCESS) {
       goto cleanup_files;
    }

    /* Call helper function to create shmem region */
    mpi_errno = MPIDI_CH3I_SHMEM_Helper_fn(pg, g_smpi.my_local_id, &pool_file,
                                        "ib_pool", &g_smpi.fd_pool, g_size_pool);
    if (mpi_errno != MPI_SUCCESS) {
       goto cleanup_files;
    }

    if (mv2_enable_shmem_collectives) {
        /* Shared memory for collectives */
        if ((mpi_errno = MPIDI_CH3I_SHMEM_COLL_init(pg, g_smpi.my_local_id)) != MPI_SUCCESS)
        {
            mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", 
                   "%s", "SHMEM_COLL_init failed");
            goto cleanup_files;
        }
    }

    g_smpi_shmem = (struct shared_mem *) MPIU_Malloc(sizeof(struct shared_mem));
    if(!g_smpi_shmem) {
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
       goto cleanup_files;
    }

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "before mmap\n");

    /* mmap of the shared memory file */
    g_smpi.mmap_ptr = mmap(0, g_size_shmem,
        (PROT_READ | PROT_WRITE), (MAP_SHARED), g_smpi.fd, 0);
    if (g_smpi.mmap_ptr == (void *) -1) {
       /* to clean up tmp shared file */
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", "%s: %s", 
                "mmap", strerror(errno));
       goto cleanup_files;
    }

    shmem = (struct shared_mem *) g_smpi.mmap_ptr;
    if (((long) shmem & (SMPI_CACHE_LINE_SIZE - 1)) != 0) {
       /* to clean up tmp shared file */
       mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", "%s", 
                "error in shifting mmaped shmem");
       goto cleanup_files;
    }

    if(!g_smp_delay_shmem_pool_init) {
        mpi_errno = MPIDI_CH3I_SMP_attach_shm_pool_inline();
        if (mpi_errno != MPI_SUCCESS) {
            /* to clean up tmp shared file */
            mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                    FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", "%s: %s", 
                    "mmap", strerror(errno));
            goto cleanup_files;
        }
    }

    /* Initialize shared_mem pointers */
    g_smpi_shmem->pid = (int *) shmem;

    g_smpi_shmem->rqueues_limits_s =
        (smpi_rq_limit *) MPIU_Malloc(sizeof(smpi_rq_limit)*g_smpi.num_local_nodes);
    g_smpi_shmem->rqueues_limits_r =
        (smpi_rq_limit *) MPIU_Malloc(sizeof(smpi_rq_limit)*g_smpi.num_local_nodes);
    g_smpi_shmem->shared_tails = (smpi_shared_tails **)
        MPIU_Malloc(sizeof(smpi_shared_tails *)*g_smpi.num_local_nodes);

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)

    g_smpi_shmem->cu_attrbs = (smpi_cu_ipc_attr**)
        MPIU_Malloc(sizeof(smpi_cu_ipc_attr*)*g_smpi.num_local_nodes);

#endif

    if (g_smpi_shmem->rqueues_limits_s == NULL ||
        g_smpi_shmem->rqueues_limits_r == NULL ||
        g_smpi_shmem->shared_tails == NULL ) {
         mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                  FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
         goto cleanup_files;
    }

    if (g_smpi.num_local_nodes > 1) {
        g_smpi_shmem->shared_tails[0] = 
                (smpi_shared_tails *)((char *)shmem +  pid_len + SMPI_CACHE_LINE_SIZE);

        for (i = 1; i < g_smpi.num_local_nodes; ++i) {
            g_smpi_shmem->shared_tails[i] = (smpi_shared_tails *)
                (g_smpi_shmem->shared_tails[i-1]+ g_smpi.num_local_nodes);
        }

        g_smpi_shmem->pool =
            (char *)((char *)g_smpi_shmem->shared_tails[1] + SMPI_ALIGN(st_len) +
                    SMPI_CACHE_LINE_SIZE);

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)

        g_smpi_shmem->cu_attrbs[0] =
                (smpi_cu_ipc_attr*)((char *)shmem +  cu_ipc_offset);

        for (i = 1; i < g_smpi.num_local_nodes; ++i) {
            g_smpi_shmem->cu_attrbs[i] = (smpi_cu_ipc_attr*)
                (g_smpi_shmem->cu_attrbs[i-1]+ g_smpi.num_local_nodes);
        }

#endif

    } else {
        g_smpi_shmem->shared_tails[0] = NULL;
        g_smpi_shmem->pool =
            (char *)((char *)shmem + pid_len + SMPI_CACHE_LINE_SIZE);
    }
 
    for (i=0; i < g_smpi.num_local_nodes; ++i) {
        if ( i == g_smpi.my_local_id)
            continue;
        g_smpi_shmem->rqueues_limits_s[i].first = 
            SMPI_ALIGN(pagesize + (pagesize + s_smp_queue_length) *
                (i * (g_smpi.num_local_nodes - 1) + 
                (g_smpi.my_local_id > i ? (g_smpi.my_local_id - 1) : g_smpi.my_local_id)));
        g_smpi_shmem->rqueues_limits_r[i].first = 
            SMPI_ALIGN(pagesize + (pagesize + s_smp_queue_length) * 
                (g_smpi.my_local_id * (g_smpi.num_local_nodes - 1) + 
                (i > g_smpi.my_local_id ? (i - 1): i)));
        g_smpi_shmem->rqueues_limits_s[i].last =
            SMPI_ALIGN(pagesize + (pagesize + s_smp_queue_length) * 
                (i * (g_smpi.num_local_nodes - 1) + 
                (g_smpi.my_local_id > i ? (g_smpi.my_local_id - 1) : g_smpi.my_local_id)) +
                g_smpi.available_queue_length);
        g_smpi_shmem->rqueues_limits_r[i].last =
            SMPI_ALIGN(pagesize + (pagesize + s_smp_queue_length) * 
                (g_smpi.my_local_id * (g_smpi.num_local_nodes - 1) + 
                (i > g_smpi.my_local_id ? (i - 1): i)) + 
                g_smpi.available_queue_length);
        g_smpi_shmem->shared_tails[g_smpi.my_local_id][i].ptr =
            g_smpi_shmem->rqueues_limits_r[i].first;
        *(int *)((unsigned long)(g_smpi_shmem->pool) + 
            g_smpi_shmem->rqueues_limits_r[i].first) = 0;
    }

    if (mv2_enable_shmem_collectives) {
        /* Memory Mapping shared files for collectives*/
        if ((mpi_errno = MPIDI_CH3I_SHMEM_COLL_Mmap(pg, g_smpi.my_local_id)) != MPI_SUCCESS)
        {
           mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                 FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s", 
                 "SHMEM_COLL_Mmap failed");
           goto cleanup_files;
        }
    }

    /* Another synchronization barrier */
    mpi_errno = MPIDI_CH3I_CM_SHMEM_Sync(g_smpi_shmem->pid, g_smpi.my_local_id,
                                            g_smpi.num_local_nodes);
    if (mpi_errno != MPI_SUCCESS) {
        goto cleanup_files;
    }

#if defined(_SMP_CMA_)
    if (g_smp_use_cma && g_smpi.num_local_nodes > 1) {
        g_smpi_shmem->cma_test_buffer = (volatile char **)
            ((char *)shmem + cma_test_buffer_offset);

        switch (g_smpi.my_local_id) {
            case 0:
                while (!*g_smpi_shmem->cma_test_buffer);
                if (-1 != (intptr_t)*g_smpi_shmem->cma_test_buffer) {
                    if (-1 == check_cma_usability(g_smpi_shmem->pid[1],
                                (char *)*g_smpi_shmem->cma_test_buffer)) {
                        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS,
                                MPI_ERR_OTHER, FCNAME, __LINE__, MPI_ERR_OTHER,
                                "**fail", "%s: %s", "process_vm_readv",
                                strerror(errno));
                        PRINT_ERROR("CMA is not available. Set "
                                "MV2_SMP_USE_CMA=0 to disable CMA.\n");
                        goto cleanup_files;
                    }
                } else {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPI_ERR_OTHER, FCNAME, __LINE__, MPI_ERR_OTHER,
                            "**nomem", "**nomem %s",
                            "*g_smpi_shmem->cma_test_buffer");
                    PRINT_ERROR("CMA is not available. Set MV2_SMP_USE_CMA=0 "
                            "to disable CMA.\n");
                    goto cleanup_files;
                }
                break;
            case 1:
                *g_smpi_shmem->cma_test_buffer = MPIU_Calloc(1, 1);

                if (!*g_smpi_shmem->cma_test_buffer) {
                    *g_smpi_shmem->cma_test_buffer = (char *)-1;
                } else {
                    while (!**g_smpi_shmem->cma_test_buffer);
                    MPIU_Free(*g_smpi_shmem->cma_test_buffer);
                }

                break;
        }
    }
#endif /* defined(_SMP_CMA_) */
    /* Unlinking shared memory files*/
    MPIDI_CH3I_SMP_unlink();
    if (mv2_enable_shmem_collectives){
        MPIDI_CH3I_SHMEM_COLL_Unlink();
    }

#if defined(__x86_64__)
    /*
     * Okay, here we touch every page in the shared memory region.
     * We do this to get the pages allocated so that they are local
     * to the receiver on a numa machine (instead of all being located
     * near the first process).
     */
    {
       int receiver, sender;
 
       for (receiver = 0; receiver < g_smpi.num_local_nodes; ++receiver) {
           volatile char *ptr = g_smpi_shmem->pool;
           volatile char tmp ATTRIBUTE((unused));
 
           sender = g_smpi.my_local_id;
           if (sender != receiver) {
              int k;
           
              for (k = SMPI_FIRST_S(sender, receiver);
                  k < SMPI_LAST_S(sender, receiver); k += pagesize) {
                  tmp = ptr[k];
              }
           }
       }
    }
#endif /* defined(__x86_64__) */

    s_current_ptr = (void **) MPIU_Malloc(sizeof(void *) * g_smpi.num_local_nodes);
    if (!s_current_ptr) {
      MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
          "**nomem %s", "s_current_ptr");
    }

    s_current_bytes = (MPIDI_msg_sz_t *) MPIU_Malloc(sizeof(MPIDI_msg_sz_t) * g_smpi.num_local_nodes);
    if (!s_current_bytes) {
      MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
          "**nomem %s", "s_current_bytes");
    }

    s_total_bytes = (MPIDI_msg_sz_t *) MPIU_Malloc(sizeof(MPIDI_msg_sz_t) * g_smpi.num_local_nodes);
    if (!s_total_bytes) {
       MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
           "**nomem %s", "s_total_bytes");
    }

    s_header_ptr_s = (size_t *) MPIU_Malloc(sizeof(size_t) * g_smpi.num_local_nodes);

    if(!s_header_ptr_s) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
        "**nomem %s", "s_header_ptr");
    }

    s_header_ptr_r = (size_t *) MPIU_Malloc(sizeof(size_t) * g_smpi.num_local_nodes);

    if(!s_header_ptr_r) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
        "**nomem %s", "s_header_ptr");
    }

    s_tail_ptr_s = (size_t *) MPIU_Malloc(sizeof(size_t) * g_smpi.num_local_nodes);

    if(!s_tail_ptr_s) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
        "**nomem %s", "s_tail_ptr");
    }

    avail = (size_t *) MPIU_Malloc(sizeof(size_t) * g_smpi.num_local_nodes);

    if(!avail) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
        "**nomem %s", "avail");
    }

    for (i = 0; i < g_smpi.num_local_nodes; ++i) {
       s_current_ptr[i] = NULL;
       s_current_bytes[i] = 0;
       s_total_bytes[i] = 0;

       if ( i == g_smpi.my_local_id)
           continue;
       s_header_ptr_r[i] = SMPI_FIRST_R(i, g_smpi.my_local_id);
       s_header_ptr_s[i] = SMPI_FIRST_S(g_smpi.my_local_id, i);
       s_tail_ptr_s[i] = SMPI_LAST_S(g_smpi.my_local_id, i);
       avail[i] = s_tail_ptr_s[i] - s_header_ptr_s[i];
    }

#if defined(_SMP_LIMIC_)
    if (g_smp_use_limic2) {
        limic_fd = limic_open();
    
        if (limic_fd == -1) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "%s: %s", "LiMIC2 device does not exist",
                        strerror(errno));
        }
        //get num of sockets within a node
        if (g_use_limic2_coll)
            hwlocSocketDetection(0);
        mv2_MPIDI_CH3I_RDMA_Process.g_smp_can_fallback = 1;
    }
#endif /*#if defined(_SMP_LIMIC_)*/



    polling_set_p = (POLLING_ELEMENT_T *) MPIU_Malloc(sizeof(POLLING_ELEMENT_T) * g_smpi.num_local_nodes);
    polling_set_c = (POLLING_ELEMENT_T *) MPIU_Malloc(sizeof(POLLING_ELEMENT_T) * g_smpi.num_local_nodes);

    for(i=0; i < g_smpi.num_local_nodes; ++i){
        polling_set_p[i].rank = i;
        polling_set_p[i].next = 0;
        polling_set_p[i].prev = 0;

        polling_set_c[i].rank = i;
        polling_set_c[i].prev = i-1;
        polling_set_c[i].next = i+1;
    }

    polling_set_c[g_smpi.num_local_nodes-1].next = 0;
    polling_set_c_tail = g_smpi.num_local_nodes-1;
    if(g_smpi.num_local_nodes > 1)
        polling_set_c_head = 1;

    polling_counters = (int *) MPIU_Malloc(sizeof(int)* g_smpi.num_local_nodes);
    for(i=0; i < g_smpi.num_local_nodes; ++i){
        polling_counters[i] = 0;
    }

    SMP_INIT = 1;

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_INIT);
    return mpi_errno;

cleanup_files:
    MPIDI_CH3I_SMP_cleanup();
    if (mv2_enable_shmem_collectives){
        MPIDI_CH3I_SHMEM_COLL_Cleanup();
    }
fn_fail:
    goto fn_exit;
}

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
void MPIDI_CH3I_SMP_cuda_ipc_init(MPID_Comm *comm_ptr)
{
    int err = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int i, total_region_size;
    cudaIpcMemHandle_t *handle_exchange;
    cudaIpcMemHandle_t *smp_cuda_memhandle;
    CUipcEventHandle *sr_eventhandle;
    CUipcEventHandle *sr_eventhandle_local;
    CUipcEventHandle *loop_eventhandle;
    CUipcEventHandle *loop_eventhandle_local;
    MPIDI_VC_t* vc = NULL;    

    MPIR_Barrier_impl(comm_ptr, &errflag);

    total_region_size = smp_cuda_region_size * g_smpi.num_local_nodes;

    smp_cuda_region_base = (void **)
        MPIU_Malloc(sizeof(void *)*g_smpi.num_local_nodes);
    smp_cuda_region_send = (void **)
        MPIU_Malloc(sizeof(void *)*g_smpi.num_local_nodes);
    smp_cuda_region_recv = (void **)
        MPIU_Malloc(sizeof(void *)*g_smpi.num_local_nodes);

    CUDA_CHECK(cudaMalloc(&smp_cuda_region_base[g_smpi.my_local_id], total_region_size));

    smp_cuda_memhandle = (cudaIpcMemHandle_t *)
        MPIU_Malloc(sizeof(cudaIpcMemHandle_t)*g_smpi.num_local_nodes);
    CUDA_CHECK(cudaIpcGetMemHandle(&smp_cuda_memhandle[g_smpi.my_local_id],
            smp_cuda_region_base[g_smpi.my_local_id]));

    handle_exchange = (cudaIpcMemHandle_t *)
        MPIU_Malloc(sizeof(cudaIpcMemHandle_t)*comm_ptr->local_size);
    MPIU_Memcpy(&handle_exchange[comm_ptr->rank],
            &smp_cuda_memhandle[g_smpi.my_local_id], sizeof(cudaIpcMemHandle_t));

    /*Optimization needed - Try to combine all allgathers and alltoalls*/

    err = MPIR_Allgather_impl(MPI_IN_PLACE, 0,
            MPI_BYTE, (void *)handle_exchange, sizeof(cudaIpcMemHandle_t),
            MPI_BYTE, comm_ptr, &errflag);

    if (err != MPI_SUCCESS) {
        smp_error_abort(SMP_EXIT_ERR,"handle exchange using Allgather failed \n");
    }

    int g_rank;
    smpi_cu_ipc_attr *ref;

    for (i=0; i<g_smpi.num_local_nodes; i++) {
        MPIDI_Comm_get_vc(comm_ptr, i, &vc);
        if (i == g_smpi.my_local_id || vc->smp.can_access_peer != CUDA_IPC_ENABLED) {
            continue;
        }

        ref = (g_smpi_shmem->cu_attrbs[g_smpi.my_local_id] + i);
        ref->cuda_head = 0;
        ref->cuda_tail = 0;
        g_rank = g_smpi.l2g_rank[i];

        MPIU_Memcpy(&smp_cuda_memhandle[i], &handle_exchange[g_rank],
                sizeof(cudaIpcMemHandle_t));
        CUDA_CHECK(cudaIpcOpenMemHandle(&smp_cuda_region_base[i], smp_cuda_memhandle[i],
                cudaIpcMemLazyEnablePeerAccess));

        smp_cuda_region_send[i] = (void *)
            ((unsigned long) smp_cuda_region_base[g_smpi.my_local_id]
             + i*smp_cuda_region_size);
        smp_cuda_region_recv[i] = (void *)
            ((unsigned long) smp_cuda_region_base[i]
             + g_smpi.my_local_id*smp_cuda_region_size);

    }

    sr_event_local = (CUevent *) MPIU_Malloc(sizeof(CUevent)*comm_ptr->local_size);
    loop_event_local = (CUevent *) MPIU_Malloc(sizeof(CUevent)*comm_ptr->local_size);
    sr_eventhandle_local = (CUipcEventHandle *)
        MPIU_Malloc(sizeof(CUipcEventHandle)*comm_ptr->local_size);
    loop_eventhandle_local = (CUipcEventHandle *)
        MPIU_Malloc(sizeof(CUipcEventHandle)*comm_ptr->local_size);

    for (i=0; i<g_smpi.num_local_nodes; i++) {
        MPIDI_Comm_get_vc(comm_ptr, i, &vc);
        if (i == g_smpi.my_local_id || vc->smp.can_access_peer != CUDA_IPC_ENABLED) {
            continue;
        }

        g_rank = g_smpi.l2g_rank[i];

        CU_CHECK(cuEventCreate(&sr_event_local[g_rank], CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING));
        CU_CHECK(cuEventRecord(sr_event_local[g_rank], 0));
        CU_CHECK(cuIpcGetEventHandle(&sr_eventhandle_local[g_rank], sr_event_local[g_rank]));
        CU_CHECK(cuEventCreate(&loop_event_local[g_rank], CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING));
        CU_CHECK(cuEventRecord(loop_event_local[g_rank], 0));
        CU_CHECK(cuIpcGetEventHandle(&loop_eventhandle_local[g_rank], loop_event_local[g_rank]));
    }

    sr_event = (CUevent *) MPIU_Malloc(sizeof(CUevent)*comm_ptr->local_size);
    loop_event = (CUevent *) MPIU_Malloc(sizeof(CUevent)*comm_ptr->local_size);

    sr_eventhandle = (CUipcEventHandle *)
        MPIU_Malloc(sizeof(CUipcEventHandle)*comm_ptr->local_size);
    loop_eventhandle = (CUipcEventHandle *)
        MPIU_Malloc(sizeof(CUipcEventHandle)*comm_ptr->local_size);

    err = MPIR_Alltoall_impl((void *)sr_eventhandle_local, sizeof(CUipcEventHandle), MPI_BYTE,
            (void *)sr_eventhandle, sizeof(CUipcEventHandle), MPI_BYTE,
            comm_ptr, &errflag);

    if (err != MPI_SUCCESS) {
        smp_error_abort(SMP_EXIT_ERR,"sr eventhandle exchange using Allgather failed \n");
    }

    unsigned int version = 0;
    cuCtxGetApiVersion(NULL, &version);

    for (i=0; i<g_smpi.num_local_nodes; i++) {
        MPIDI_Comm_get_vc(comm_ptr, i, &vc);
        if (i == g_smpi.my_local_id || vc->smp.can_access_peer != CUDA_IPC_ENABLED) {
            continue;
        }

        g_rank = g_smpi.l2g_rank[i];

        CU_CHECK(cuIpcOpenEventHandle(&sr_event[g_rank], sr_eventhandle[g_rank]));
        CU_CHECK(cuEventRecord(sr_event[g_rank], 0));
    }

    err =
        MPIR_Alltoall_impl((void *)loop_eventhandle_local, sizeof(CUipcEventHandle),
                MPI_BYTE, (void *)loop_eventhandle, sizeof(CUipcEventHandle), MPI_BYTE,
                comm_ptr, &errflag);

    if (err != MPI_SUCCESS) {
        smp_error_abort(SMP_EXIT_ERR,"loop eventhandle exchange using Allgather failed \n");
    }

    for (i=0; i<g_smpi.num_local_nodes; i++) {
        MPIDI_Comm_get_vc(comm_ptr, i, &vc);
        if (i == g_smpi.my_local_id || vc->smp.can_access_peer != CUDA_IPC_ENABLED) {
            continue;
        }

        g_rank = g_smpi.l2g_rank[i];

        CU_CHECK(cuIpcOpenEventHandle(&loop_event[g_rank], loop_eventhandle[g_rank]));
        CU_CHECK(cuEventRecord(loop_event[g_rank], 0));
    }

    if (handle_exchange != NULL) {
        MPIU_Free(handle_exchange);
    }
    if (smp_cuda_memhandle) {
        MPIU_Free(smp_cuda_memhandle);
    }
    if (sr_eventhandle_local) {
        MPIU_Free(sr_eventhandle_local);
    }
    if (sr_eventhandle) {
        MPIU_Free(sr_eventhandle);
    }
    if (loop_eventhandle_local) {
        MPIU_Free(loop_eventhandle_local);
    }
    if (loop_eventhandle) {
        MPIU_Free(loop_eventhandle);
    }
}
#endif


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_finalize()
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_FINALIZE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_FINALIZE);

    /* reset global variables */
    mv2_shmem_pool_init = 0;
    g_smp_delay_shmem_pool_init = 1;
    polling_set_p_head = 0;
    polling_set_p_tail = 0;
    polling_set_c_head = 0;
    polling_set_c_tail = 0;

    /* free polling set structures */
    if (polling_counters) {
        MPIU_Free(polling_counters);
    }
    if (polling_set_p) {
        MPIU_Free(polling_set_p);
    }
    if (polling_set_c) {
        MPIU_Free(polling_set_c);
    }

    /* unmap the shared memory file */
    munmap((void *)g_smpi.mmap_ptr, g_size_shmem);
    close(g_smpi.fd);

    if (g_smpi.send_buf_pool_ptr != NULL) {
        munmap(g_smpi.send_buf_pool_ptr, g_size_pool);
        close(g_smpi.fd_pool);
    }

    if(s_buffer_head) {
        MPIU_Free(s_buffer_head);
    }

    if (g_smpi.l2g_rank) {
        MPIU_Free(g_smpi.l2g_rank);
    } 

    if (g_smpi_shmem) {
        if (g_smpi_shmem->rqueues_limits_s != NULL) { 
            MPIU_Free(g_smpi_shmem->rqueues_limits_s);
        }
        if (g_smpi_shmem->rqueues_limits_r != NULL) { 
            MPIU_Free(g_smpi_shmem->rqueues_limits_r);
        }
        if (g_smpi_shmem->shared_tails != NULL) {
            MPIU_Free(g_smpi_shmem->shared_tails);
        }
#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
        if (g_smpi_shmem->cu_attrbs) {
            MPIU_Free(g_smpi_shmem->cu_attrbs);
        }
#endif
        if(g_smpi_shmem != NULL) { 
            MPIU_Free(g_smpi_shmem);
        }
    }

    if (s_current_ptr) {
        MPIU_Free(s_current_ptr);
    }

    if (s_current_bytes) {
        MPIU_Free(s_current_bytes);
    }

    if (s_total_bytes) {
        MPIU_Free(s_total_bytes);
    }

    if (s_header_ptr_s) {
        MPIU_Free(s_header_ptr_s);
    }

    if (s_header_ptr_r) {
        MPIU_Free(s_header_ptr_r);
    }

    if (s_tail_ptr_s) {
        MPIU_Free(s_tail_ptr_s);
    }

    if (avail) {
        MPIU_Free(avail);
    }    

    if (s_sh_buf_pool.send_queue) {
        MPIU_Free(s_sh_buf_pool.send_queue);
    }

    if (s_sh_buf_pool.tail) {
        MPIU_Free(s_sh_buf_pool.tail);
    }

    if (mv2_enable_shmem_collectives || finalize_coll_comm == 1){
        /* Freeing up shared memory collective resources*/
        MPIDI_CH3I_SHMEM_COLL_finalize( g_smpi.my_local_id, g_smpi.num_local_nodes);
    }

#ifdef _SMP_LIMIC_
    limic_close(limic_fd);
#endif

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_FINALIZE);
    return MPI_SUCCESS;
}

#if defined (_ENABLE_CUDA_)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CUDA_SMP_cuda_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_CUDA_SMP_cuda_init(MPIDI_PG_t *pg)
{
    if (!cuda_initialized) { 
        return;
    }

    if (mv2_shmem_pool_init) {
        ibv_cuda_register (g_smpi.send_buf_pool_ptr, g_size_pool);
    }

#if defined(HAVE_CUDA_IPC)
    if (rdma_cuda_smp_ipc) {
        MPIDI_CH3I_SMP_cuda_ipc_init(MPIR_Process.comm_world);
    }
#endif
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_cuda_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_CUDA_SMP_cuda_finalize(MPIDI_PG_t *pg)
{
    int i;
    MPIDI_VC_t *vc = NULL;

    if (!cuda_initialized) {
        return;
    }

    if (mv2_shmem_pool_init) { 
        ibv_cuda_unregister (g_smpi.send_buf_pool_ptr);
    }

#if defined(HAVE_CUDA_IPC)
    if (SMP_INIT && rdma_cuda_smp_ipc) {
        for (i = 0; i < g_smpi.num_local_nodes; i++) {
            MPIDI_PG_Get_vc(pg, i, &vc);
            if (i == g_smpi.my_local_id || vc->smp.can_access_peer != CUDA_IPC_ENABLED) {
                continue;
            }

            CU_CHECK(cuEventDestroy(sr_event_local[i]));
            CU_CHECK(cuEventDestroy(loop_event_local[i]));
            CU_CHECK(cuEventDestroy(sr_event[i]));
            CU_CHECK(cuEventDestroy(loop_event[i]));
            CUDA_CHECK(cudaIpcCloseMemHandle(smp_cuda_region_base[i]));
        }

        if (smp_cuda_region_base) {
            MPIU_Free(smp_cuda_region_base);
        }
        if (smp_cuda_region_send) {
            MPIU_Free(smp_cuda_region_send);
        }
        if (smp_cuda_region_recv) {
            MPIU_Free(smp_cuda_region_recv);
        }
        if (sr_event_local) {
            MPIU_Free(sr_event_local);
        }
        if (sr_event) {
            MPIU_Free(sr_event);
        }
        if (loop_event) {
            MPIU_Free(loop_event);
        }
        if (loop_event_local) {
            MPIU_Free(loop_event_local);
        }
    }
#endif
}
#endif

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_writev_rndv_header
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SMP_writev_rndv_header(MPIDI_VC_t * vc, const MPL_IOV * iov,
    const int n, int *num_bytes_ptr)
{
    int pkt_len = 0;
    volatile void *ptr_head, *ptr, *ptr_flag;
    int i;
    MPIDI_CH3_Pkt_rndv_r3_data_t *pkt_header;
#if defined(_SMP_LIMIC_)
    size_t err;
    size_t  total_bytes = 0;
    MPID_Request *sreq = NULL;
    limic_user lu;
#endif


#if defined(_SMP_CMA_)
    size_t cma_total_bytes = 0;
    MPID_Request *creq = NULL; 
    pid_t pid;
#endif

    int mpi_errno = MPI_SUCCESS;
    if ( !mv2_shmem_pool_init )
    {
        if ((mpi_errno = MPIDI_CH3I_SMP_attach_shm_pool_inline()) !=
                MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }


    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_HEADER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_HEADER);

    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);
    int len = iov[0].MPL_IOV_LEN;

    /* iov[0] is the header pkt */
    pkt_header = (MPIDI_CH3_Pkt_rndv_r3_data_t *)(iov[0].MPL_IOV_BUF);
    *num_bytes_ptr = 0;
#if defined(_SMP_CMA_)
    creq = pkt_header->csend_req_id;
    /* creq_req_id is set to NULL for non-contig data, then fall back to shared memory
     * instead of using limic; or else, continue data transfer by limic */
    if (g_smp_use_cma && creq) {
        assert(creq->dev.iov_count == 1);

        /* The last sizeof(MPIDI_msg_sz_t) is the total num of data bytes */
        pkt_len = iov[0].MPL_IOV_LEN + sizeof(struct iovec) * creq->dev.iov_count + sizeof(pid_t) + sizeof(MPIDI_msg_sz_t);

        /* check if avail is less than data size */
        if(!smpi_check_avail(vc->smp.local_nodes, pkt_len, (volatile void **)&ptr_flag, ONE_FREE)) {
            goto fn_exit;
        }

        /* number of bytes */
        ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
        ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

        /* header format:
         * flag | pkt_len | normal header | iov | pid | total_num_size
         */
        MPIU_Memcpy((void *)ptr, iov[0].MPL_IOV_BUF, iov[0].MPL_IOV_LEN);
        ptr = (volatile void *) ((unsigned long) ptr +
                  iov[0].MPL_IOV_LEN);

        for(i = 0; i < creq->dev.iov_count; ++i) {
            cma_total_bytes += creq->dev.iov[i].MPL_IOV_LEN;

            /* copy the limic_user information to the shared memory
               and move the shared memory pointer properly
             */
            MPIU_Memcpy((void *)ptr, &(creq->dev.iov[i]), sizeof(struct iovec));
            ptr = (volatile void *) ((unsigned long) ptr + sizeof(struct iovec));
        }
        pid = g_smpi_shmem->pid[g_smpi.my_local_id];
        *((volatile pid_t *) ptr) = pid;
        ptr = (volatile void *) ((unsigned long) ptr + sizeof(pid_t));
        *((volatile MPIDI_msg_sz_t *) ptr) = cma_total_bytes;
        ptr = (volatile void *) ((unsigned long) ptr + sizeof(MPIDI_msg_sz_t));

        *num_bytes_ptr = iov[0].MPL_IOV_LEN;
        smpi_complete_send(vc->smp.local_nodes, pkt_len, pkt_len, ptr, ptr_head, ptr_flag);
        /* For a CMA based trasnfer, we expect a FIN message */
        MV2_INC_NUM_POSTED_RECV();
    } else {
#endif /* _SMP_CMA */

#if defined(_SMP_LIMIC_)
    /* sreq is the send request handle for the data */
    sreq = pkt_header->send_req_id;

    /* sreq_req_id is set to NULL for non-contig data, then fall back to shared memory
     * instead of using limic; or else, continue data transfer by limic */
    if (g_smp_use_limic2 && sreq) {

        assert(sreq->dev.iov_count == 1);
        /* The last sizeof(int) is the total num of data bytes */
        pkt_len = iov[0].MPL_IOV_LEN + sizeof(limic_user) * sreq->dev.iov_count + sizeof(MPIDI_msg_sz_t);

        /* check if avail is less than data size */
        if(!smpi_check_avail(vc->smp.local_nodes, pkt_len, (volatile void **)&ptr_flag, ONE_FREE)) {
            goto fn_exit;
        }

        /* number of bytes */
        ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
        ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

        /* header format:
         * flag | pkt_len | normal header | lu | total_num_size
         */
        MPIU_Memcpy((void *)ptr, iov[0].MPL_IOV_BUF, iov[0].MPL_IOV_LEN);
        ptr = (volatile void *) ((unsigned long) ptr +
                  iov[0].MPL_IOV_LEN);

        for(i = 0; i < sreq->dev.iov_count; ++i) {
            err = limic_tx_init( limic_fd, sreq->dev.iov[i].MPL_IOV_BUF,
                      sreq->dev.iov[i].MPL_IOV_LEN, &lu);
            if (!err) {
                MPIR_ERR_SETFATALANDJUMP1(err, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "LiMIC: (MPIDI_CH3I_SMP_writev_rndv_header) limic_tx_init fail");
            }
            total_bytes += sreq->dev.iov[i].MPL_IOV_LEN;

            /* copy the limic_user information to the shared memory
               and move the shared memory pointer properly
             */
            MPIU_Memcpy((void *)ptr, &lu, sizeof(limic_user));
            ptr = (volatile void *) ((unsigned long) ptr + sizeof(limic_user));
        }

        *((volatile MPIDI_msg_sz_t *) ptr) = total_bytes;
        ptr = (volatile void *) ((unsigned long) ptr + sizeof(MPIDI_msg_sz_t));

        *num_bytes_ptr = iov[0].MPL_IOV_LEN;

        smpi_complete_send(vc->smp.local_nodes, pkt_len, pkt_len, ptr, ptr_head, ptr_flag);
        /* For a LiMIC based trasnfer, we expect a FIN message */
        MV2_INC_NUM_POSTED_RECV();
    } else {
#endif /* _SMP_LIMIC */

    /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, len, (volatile void **)&ptr_flag, TWO_FREE))
        return;

    send_buf_reclaim();

    if (s_sh_buf_pool.free_head == -1) {
        goto fn_exit;
    }

    pkt_header->src.smp_index = s_sh_buf_pool.free_head;

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

    i = 0;
    for (i = 0; i < n; i++) {
        MPIU_Memcpy((void *)ptr,
                (void *) ((unsigned long) iov[i].MPL_IOV_BUF),
                iov[i].MPL_IOV_LEN);
        ptr =
            (volatile void *) ((unsigned long) ptr + iov[i].MPL_IOV_LEN);
        pkt_len += iov[i].MPL_IOV_LEN;
    }

    MPIU_Assert(len == pkt_len);
    smpi_complete_send(vc->smp.local_nodes, len, len, ptr, ptr_head, ptr_flag);

    *num_bytes_ptr += pkt_len;
#if defined(_SMP_LIMIC_)
    }
#endif /* _SMP_LIMIC_ */
#if defined(_SMP_CMA_)
    }
#endif /* _SMP_CMA_ */
fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "writev_rndv_header returns bytes %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_HEADER);
    return;

fn_fail:
    goto fn_exit;
}

#if defined(_ENABLE_CUDA_)
void smp_cuda_send_copy_complete(MPIDI_VC_t * vc, MPID_Request *req, void *ptr_flag) 
{
    int mpi_errno = MPI_SUCCESS;
    int complete = 0;

    READBAR();
    MPIU_Assert(*((volatile int *) ptr_flag) == SMP_CBUF_PENDING);
    *((volatile int *) ptr_flag) = SMP_CBUF_BUSY;   
    WRITEBAR();

    /*This is the last chunk if request is present, complete it*/
    if (req != NULL) { 
        mpi_errno = MPIDI_CH3U_Handle_send_req(vc, req, &complete);
        if (mpi_errno != MPI_SUCCESS) { 
            smp_error_abort(SMP_EXIT_ERR,"handle_send_req returned error\n");
        }
        MPIU_Assert(complete == 1);
    }
}

void smp_cuda_send_copy_enqueue(MPIDI_VC_t * vc, MPID_Request *req, SEND_BUF_T *send_buf, 
        void *user_buf, size_t size, int is_last, volatile void *ptr_flag)
{
    cuda_event_t *cuda_event;
    cudaError_t cuda_err = cudaSuccess; 

    *((volatile int *) ptr_flag) = SMP_CBUF_PENDING;
    WRITEBAR();

    send_buf->len = size;
    MPIU_Memcpy_CUDA_Async(send_buf->buf, user_buf, size, cudaMemcpyDeviceToHost, stream_d2h);

    cuda_event = get_cuda_event();
    if (cuda_event == NULL) {
        allocate_cuda_event(&cuda_event);
        /* add to the busy list */
        cuda_event->is_query_done = 0;
        CUDA_LIST_ADD(cuda_event,
            busy_cuda_event_list_head, busy_cuda_event_list_tail);
    }
    cuda_event->op_type = SMP_SEND;
    cuda_event->vc = vc;
    cuda_event->smp_ptr = (void *) ptr_flag;
    cuda_event->size = size;
    if (is_last) { 
        cuda_event->req = req;
    } else {
        cuda_event->req = NULL;
    }

    cuda_err = cudaEventRecord(cuda_event->event, stream_d2h);
    if (cuda_err != cudaSuccess) {
        smp_error_abort(SMP_EXIT_ERR,"cudaEventRecord failed\n"); 
    }
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_writev_rndv_data_cuda
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_writev_rndv_data_cuda(MPIDI_VC_t * vc, MPID_Request *req, 
    const MPL_IOV * iov, const int n, int *num_bytes_ptr, int is_cont)
{
    int mpi_errno = MPI_SUCCESS;
    int i, offset, length, pkt_avail, pkt_len;
    volatile void *ptr_flag, *ptr;
    SEND_BUF_T *send_buf = NULL;
    int is_last;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA_CUDA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA_CUDA);

    *num_bytes_ptr = 0;

    if (is_cont) { 
        send_buf_reclaim();
        if(s_sh_buf_pool.free_head == -1) {
            return mpi_errno;
        }
    } else { 
        if(s_sh_buf_pool.free_head == -1) {
             MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "s_sh_buf_pool.free_head == -1");
        }
    }

    i = 0;
    is_last = 0;
    offset = 0;
    length = 0;
    do {
        /* check if a space is available for a signal send */
        ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
                s_header_ptr_s[vc->smp.local_nodes]);

        if (is_cont) {
            length = sizeof(int);
        } else {
            length = 0;
        }
        
        if(!smpi_check_avail(vc->smp.local_nodes, length, (volatile void **)&ptr_flag, ONE_FREE)) {
            return 0;
        }

        send_buf = get_buf_from_pool();
        if(send_buf == NULL) break;

        send_buf->has_next = 0;
        send_buf->busy = 1;
        link_buf_to_send_queue (vc->smp.local_nodes, send_buf->myindex);

        pkt_avail = s_smp_block_size;
        pkt_len = 0; 

        for (; i < n;) {
            if (pkt_avail >= (iov[i].MPL_IOV_LEN - offset)) {
                if (i == n-1) is_last = 1;
                smp_cuda_send_copy_enqueue(vc, req, send_buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF + offset),
                    iov[i].MPL_IOV_LEN - offset, 
                    is_last, ptr_flag);
                pkt_len += (iov[i].MPL_IOV_LEN - offset);
                pkt_avail -=  (iov[i].MPL_IOV_LEN - offset);
                offset = 0;
                ++i;
            } else if (pkt_avail > 0) {
                smp_cuda_send_copy_enqueue(vc, req, send_buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF + offset),
                    pkt_avail, is_last, ptr_flag);
                offset += pkt_avail;
                pkt_len += pkt_avail;
                break;
            }
        }

        /* set data size*/
        ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
        *((volatile int *) ptr) = pkt_len;
        if (is_cont) {
            /*set pointer to send_buf*/
            ptr = (volatile void *) ((unsigned long) ptr + sizeof(int));
            *((volatile int *) ptr) = send_buf->myindex;
        }
        /* set next flag to free */
        ptr = (volatile void *) ((unsigned long) ptr + sizeof(int));
        *((volatile int *) ptr) = SMP_CBUF_FREE;
        WRITEBAR();

        /* reduce avail count and header_ptr, the ptr_flag will be set to 
         * BUSY when the copy completes */
        s_header_ptr_s[vc->smp.local_nodes] += (length + sizeof(int)*2);
        avail[vc->smp.local_nodes] -= (length + sizeof(int)*2);

        *num_bytes_ptr += pkt_len;
        is_cont = 1;

        if (i == n) break;
 
        send_buf_reclaim();
    } while (s_sh_buf_pool.free_head != -1);

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA_CUDA);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#endif 

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_writev_rndv_data_cont
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SMP_writev_rndv_data_cont(MPIDI_VC_t * vc, MPID_Request *req,
    const MPL_IOV * iov, const int n, int *num_bytes_ptr)
{
    volatile void *ptr_head, *ptr_flag, *ptr;
    int pkt_avail;
    int pkt_len = 0;
    int i, offset = 0;
    int destination = vc->smp.local_nodes;
    int first_index;
    SEND_BUF_T *send_buf = NULL;
    SEND_BUF_T *tmp_buf = NULL;
    int has_sent = 0;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_DATA_CONT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_DATA_CONT);
    *num_bytes_ptr = 0;

    int len;
    len = sizeof(int);

    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);

    /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, len, (volatile void **)&ptr_flag, ONE_FREE))
        return;

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && s_smp_cuda_pipeline 
        && req->mrail.cuda_transfer_mode == NONE) {
        pkt_avail = s_smp_h2h_block_size;      
    } else 
#endif
    {
        pkt_avail = s_smp_block_size;
    }

    send_buf_reclaim();

    if (s_sh_buf_pool.free_head == -1) {
    *num_bytes_ptr = 0;
    goto fn_exit;
    }

    first_index = s_sh_buf_pool.free_head;

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        /*as it is all data, we check the first iov to determine if the buffer is on device*/
        iov_isdev = is_device_buffer((void *) iov[0].MPL_IOV_BUF);
    } 
#endif

    i = 0;
    *num_bytes_ptr = 0;
    do {
    pkt_len = 0;
    for (; i < n;) {
        PRINT_DEBUG(DEBUG_SHM_verbose>1, "i %d, iov[i].len %ld, (len-offset) %ld, pkt_avail %d\n",
                    i, iov[i].MPL_IOV_LEN, (iov[i].MPL_IOV_LEN - offset), pkt_avail);

        if(has_sent >= s_smp_batch_size)
        break;
        ++has_sent;

        send_buf = get_buf_from_pool();
        if(send_buf == NULL)
        break;

        if (pkt_avail >= (iov[i].MPL_IOV_LEN - offset)) {
        if (offset != 0) {
#if defined(_ENABLE_CUDA_) 
            if (iov_isdev) {
                MPIU_Memcpy_CUDA(&send_buf->buf,
                            (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                            offset),
                            iov[i].MPL_IOV_LEN - offset, 
                            cudaMemcpyDeviceToHost); 
            } else  
#endif
            { 
                MPIU_Memcpy(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                    offset),
                    iov[i].MPL_IOV_LEN - offset);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, iov[i].MPL_IOV_LEN - offset);

            send_buf->busy = 1;
            send_buf->len = iov[i].MPL_IOV_LEN - offset;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += (iov[i].MPL_IOV_LEN - offset);
            offset = 0;
        } else {
#if defined(_ENABLE_CUDA_) 
            if (iov_isdev) {
                MPIU_Memcpy_CUDA(&send_buf->buf, iov[i].MPL_IOV_BUF,
                        iov[i].MPL_IOV_LEN,
                        cudaMemcpyDeviceToHost);
            } else  
#endif
            {
                MPIU_Memcpy(&send_buf->buf, iov[i].MPL_IOV_BUF, 
                        iov[i].MPL_IOV_LEN);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, iov[i].MPL_IOV_LEN);

            send_buf->busy = 1;
            send_buf->len = iov[i].MPL_IOV_LEN;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += iov[i].MPL_IOV_LEN;
        }
        ++i;
        } else if (pkt_avail > 0) {
#if defined(_ENABLE_CUDA_) 
            if (iov_isdev) {
                MPIU_Memcpy_CUDA(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF + offset),
                    pkt_avail,
                    cudaMemcpyDeviceToHost);
            } else  
#endif
            { 
                MPIU_Memcpy(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF + offset),
                    pkt_avail);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, pkt_avail);

            send_buf->busy = 1;
            send_buf->len = pkt_avail;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += pkt_avail;
            offset += pkt_avail;
        }
    }

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current pkt consumed, pkt_len %d\n", pkt_len);
    *num_bytes_ptr += pkt_len;

    if (i == n || has_sent >= s_smp_batch_size) {
        break;
    }

    send_buf_reclaim();

    } while (s_sh_buf_pool.free_head != -1);

    if(tmp_buf != NULL){
    tmp_buf->has_next = 0;
    }

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

    *((volatile int *) ptr) = first_index;
    ptr = (volatile void *) ((unsigned long) ptr + sizeof(int));
    /* update(header) */
    smpi_complete_send(vc->smp.local_nodes, len, *num_bytes_ptr, ptr, ptr_head, ptr_flag);

fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "writev_rndv_data_cont returns bytes %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WRITEV_RNDV_DATA_CONT);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_writev_rndv_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_writev_rndv_data(MPIDI_VC_t * vc, MPID_Request *req,
    const MPL_IOV * iov, const int n, int *num_bytes_ptr)
{
    volatile void *ptr_head, *ptr_flag, *ptr;
    int pkt_avail;
    int pkt_len = 0;
    int i, offset = 0;
    int destination = vc->smp.local_nodes;
    SEND_BUF_T *send_buf = NULL;
    SEND_BUF_T *tmp_buf = NULL;
    int has_sent=0;
    int mpi_errno = MPI_SUCCESS;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA);

    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);
    int len = 0;
    *num_bytes_ptr = 0;
   
    /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, len, (volatile void **)&ptr_flag, ONE_FREE))
        return mpi_errno;

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && s_smp_cuda_pipeline 
        && req->mrail.cuda_transfer_mode == NONE) {
        pkt_avail = s_smp_h2h_block_size;     
    } else 
#endif
    {
        pkt_avail = s_smp_block_size;
    }

    if(s_sh_buf_pool.free_head == -1) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
        "**fail %s", "s_sh_buf_pool.free_head == -1");
    }

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        /*as it is all data, we check the first iov to determine if the buffer is on device*/
        iov_isdev = is_device_buffer((void *) iov[0].MPL_IOV_BUF);
    } 
#endif

    i = 0;
    do {
    pkt_len = 0;
    for (; i < n;) {
        PRINT_DEBUG(DEBUG_SHM_verbose>1, "i %d, iov[i].len %ld, (len-offset) %ld, pkt_avail %d\n",
                    i, iov[i].MPL_IOV_LEN, (iov[i].MPL_IOV_LEN - offset), pkt_avail);
        if(has_sent >= s_smp_batch_size)
        break;
        ++has_sent;

        send_buf = get_buf_from_pool();
        if(send_buf == NULL)
        break;

        if (pkt_avail >= (iov[i].MPL_IOV_LEN - offset)) {
        if (offset != 0) {
#if defined(_ENABLE_CUDA_)
            if (iov_isdev) { 
                MPIU_Memcpy_CUDA(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                    offset),
                    iov[i].MPL_IOV_LEN - offset,
                    cudaMemcpyDeviceToHost);
            } else 
#endif
            {
                MPIU_Memcpy(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                    offset),
                    iov[i].MPL_IOV_LEN - offset);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, iov[i].MPL_IOV_LEN - offset);

            send_buf->busy = 1;
            send_buf->len = iov[i].MPL_IOV_LEN - offset;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += (iov[i].MPL_IOV_LEN - offset);
            offset = 0;
        } else {
#if defined(_ENABLE_CUDA_)
            if (iov_isdev) {
                MPIU_Memcpy_CUDA(&send_buf->buf, iov[i].MPL_IOV_BUF, 
                        iov[i].MPL_IOV_LEN, cudaMemcpyDeviceToHost);
            } else 
#endif
            {
                MPIU_Memcpy(&send_buf->buf, iov[i].MPL_IOV_BUF, 
                        iov[i].MPL_IOV_LEN);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, iov[i].MPL_IOV_LEN);

            send_buf->busy = 1;
            send_buf->len = iov[i].MPL_IOV_LEN;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += iov[i].MPL_IOV_LEN;
        }
        ++i;
        } else if (pkt_avail > 0) {
#if defined(_ENABLE_CUDA_)
            if (iov_isdev) { 
                MPIU_Memcpy_CUDA(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                    offset),
                    pkt_avail,
                    cudaMemcpyDeviceToHost);
            } else 
#endif
            {    
                MPIU_Memcpy(&send_buf->buf,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF +
                    offset),
                    pkt_avail);
            }

            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_sent, pkt_avail);

            send_buf->busy = 1;
            send_buf->len = pkt_avail;
            send_buf->has_next = 1;

            link_buf_to_send_queue (destination, send_buf->myindex);
            tmp_buf = send_buf;

            pkt_len += pkt_avail;
            offset += pkt_avail;
        }
    }

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current pkt consumed, pkt_len %d\n", pkt_len);
    *num_bytes_ptr += pkt_len;

    if (i == n || has_sent >= s_smp_batch_size) {
        break;
    }

    send_buf_reclaim();

    } while (s_sh_buf_pool.free_head != -1);

    if(tmp_buf != NULL){
    tmp_buf->has_next = 0;
    }

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

    /* update(header) */
    smpi_complete_send(vc->smp.local_nodes, len, *num_bytes_ptr, ptr, ptr_head, ptr_flag);

fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "writev_rndv_data returns bytes %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WRITE_RNDV_DATA);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_writev
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SMP_writev(MPIDI_VC_t * vc, const MPL_IOV * iov,
    const int n, int *num_bytes_ptr)
{
    int pkt_len = 0;
    volatile void *ptr_head, *ptr, *ptr_flag;
    int i;
    int len;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0; 
#if defined(HAVE_CUDA_IPC)
    int rem_rank = 0, data_sz = 0;
    int buf_isdev = 0;
    int contig_avail = 0;
    void *cuda_region_ptr = NULL;
    smpi_cu_ipc_attr *my_base;
    int cur_t,cur_h;
#endif
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_WRITEV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_WRITEV);

    Calculate_IOV_len(iov, n, len);

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    /* iov coming from the cuda path is split into 2 iovs one for the 
    ** SMP channel and the other for CUDA transfer. iov[0] refers to the
    ** the former while iov[1] refers to latter*/

    buf_isdev = is_device_buffer((void *) iov[1].MPL_IOV_BUF);

    if (rdma_enable_cuda && rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
        if (buf_isdev && n == 2) {
            /*first iov is the header*/ 
            data_sz = len - iov[0].MPL_IOV_LEN;
            len = iov[0].MPL_IOV_LEN;

            rem_rank = vc->smp.local_nodes;
            my_base = g_smpi_shmem->cu_attrbs[g_smpi.my_local_id];
            cur_h = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head;
            cur_t = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_tail;

            if(cur_t > cur_h) {
                contig_avail =  cur_t - cur_h;
            } else {
                contig_avail =  smp_cuda_region_size - cur_h;
                if (contig_avail < data_sz) {
                    cur_h = 0;
                    contig_avail = cur_t;
                    if (cuStreamWaitEvent(0, loop_event_local[vc->pg_rank], 0) != CUDA_SUCCESS) {
                        PRINT_ERROR("Error in cuStreamWaitEvent\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }

            if(contig_avail < data_sz) {
                *num_bytes_ptr = 0;
                return;
                /* Not advancing header. Probably can be used for another
                   request in the queue?*/
            }
        }
    }
#endif


    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);

    *num_bytes_ptr = 0;

    /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, len, (volatile void **)&ptr_flag, ONE_FREE))
        return;

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

    for (i = 0; i < n; i++) {
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda) {
            iov_isdev = is_device_buffer((void *) iov[i].MPL_IOV_BUF);
        }

        if (iov_isdev) {

#if defined(HAVE_CUDA_IPC)
            if (rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
                cuda_region_ptr = (void *) ((unsigned long) smp_cuda_region_send[vc->smp.local_nodes]  + cur_h);
                MPIU_Memcpy_CUDA(cuda_region_ptr,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF),
                    iov[i].MPL_IOV_LEN,
                    cudaMemcpyDefault);
                if (cuEventRecord(sr_event[vc->pg_rank], 0) != CUDA_SUCCESS) {
                    PRINT_ERROR("[%d] event record failed for rank : %d \n",MPIDI_Process.my_pg_rank, vc->pg_rank);
                    smp_error_abort(SMP_EXIT_ERR,"Error in recording the event\n");
                }
                if(0 == cur_h) {
                    ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head = iov[i].MPL_IOV_LEN;
                } else {
                    ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head += iov[i].MPL_IOV_LEN;
                }       
            } else      
#endif
            {
                MPIU_Memcpy_CUDA((void *)ptr,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF),
                    iov[i].MPL_IOV_LEN, cudaMemcpyDeviceToHost);
            } 
        } else 
#endif
        {
            MPIU_Memcpy((void *)ptr,
                    (void *) ((unsigned long) iov[i].MPL_IOV_BUF),
                    iov[i].MPL_IOV_LEN);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_eager_sent, iov[i].MPL_IOV_LEN);

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
        /*if iov is copied to device eager buffers then there is no
          need of advancing ptr by iov_len as there is nothing being
          copied in the shared channel*/
        
        if (rdma_cuda_smp_ipc && iov_isdev && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
           /*do nothing*/ 
        } else
#endif
        {
            ptr =
                (volatile void *) ((unsigned long) ptr + iov[i].MPL_IOV_LEN);
        }
        pkt_len += iov[i].MPL_IOV_LEN;
    }

    /* update(header) */
    smpi_complete_send(vc->smp.local_nodes, len, len, ptr, ptr_head, ptr_flag);
    /* For a CMA based trasnfer, we expect a FIN message */
    MV2_INC_NUM_POSTED_RECV();

    *num_bytes_ptr += pkt_len;

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "writev returns bytes %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_WRITEV);
}

#undef FUNCNAME
#define FUNCNAME mv2_smp_fast_write_contig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_smp_fast_write_contig(MPIDI_VC_t* vc, const void *buf,
                                MPIDI_msg_sz_t data_sz, int rank, int tag,
                                MPID_Comm *comm, int context_offset, MPID_Request **sreq_p)
{
    int len = 0;
    MPID_Seqnum_t seqnum;
    int mpi_errno = MPI_SUCCESS;
    volatile void *ptr_head = NULL, *ptr = NULL, *ptr_flag = NULL;
#if defined(_ENABLE_CUDA_)
    int buf_isdev = 0;
#if defined(HAVE_CUDA_IPC)
    int rem_rank = 0;
    int contig_avail = 0;
    void *cuda_region_ptr = NULL;
    smpi_cu_ipc_attr *my_base;
    int cur_t,cur_h;
#endif
#endif

    len = data_sz + sizeof(MPIDI_CH3_Pkt_eager_send_t);

#if defined(_ENABLE_CUDA_)
    buf_isdev = is_device_buffer((void *) buf);
#if defined(HAVE_CUDA_IPC)
    if (rdma_enable_cuda && rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
        if (buf_isdev) {
            len = sizeof(MPIDI_CH3_Pkt_eager_send_t);
            rem_rank = vc->smp.local_nodes;
            my_base = g_smpi_shmem->cu_attrbs[g_smpi.my_local_id];
            cur_h = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head;
            cur_t = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_tail;

            if(cur_t > cur_h) {
                contig_avail =  cur_t - cur_h; 
            } else {
                contig_avail =  smp_cuda_region_size - cur_h;
                if (contig_avail < data_sz) {
                    cur_h = 0;
                    contig_avail = cur_t;
                    if (cuStreamWaitEvent(0, loop_event_local[vc->pg_rank], 0) != CUDA_SUCCESS) {
                        smp_error_abort(SMP_EXIT_ERR,"Error in cuStreamWaitEvent\n");
                    }
                }
            }

            if(contig_avail < data_sz) {
                /* Not advancing header. Probably can be used for another
                   request in the queue?*/
                goto fn_fail;
            }
        }
    }
#endif
#endif

    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);

    /* check if avail is less than data size */
    if (likely(smpi_check_avail(vc->smp.local_nodes, len,
                            (volatile void **)&ptr_flag, ONE_FREE))) {
        ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
        ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

        MPIDI_CH3_Pkt_t *upkt;
        MPIDI_CH3_Pkt_eager_send_t * eager_pkt;
    
        /* Write header information directly to SHMEM */
        upkt = (MPIDI_CH3_Pkt_t *) ptr;
        eager_pkt = &((*upkt).eager_send);
        MPIDI_Pkt_init(eager_pkt, MPIDI_CH3_PKT_EAGER_SEND_CONTIG);
        eager_pkt->match.parts.rank         = comm->rank;
        eager_pkt->match.parts.tag          = tag;
        eager_pkt->match.parts.context_id   = comm->context_id + context_offset;
        eager_pkt->sender_req_id            = MPI_REQUEST_NULL;
        eager_pkt->data_sz                  = data_sz;
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
        eager_pkt->in_cuda_region = 0;
#endif

        /* Set sequence number */
        MPIDI_VC_FAI_send_seqnum(vc, seqnum);
        MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);

        /* Increment pointer */
        ptr = (void *)((unsigned long) ptr + sizeof(MPIDI_CH3_Pkt_eager_send_t));

#if defined(_ENABLE_CUDA_)
        if (buf_isdev) {
#if defined(HAVE_CUDA_IPC)
            if (rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
                
                eager_pkt->in_cuda_region = 1;
                cuda_region_ptr = (void *) ((unsigned long) smp_cuda_region_send[vc->smp.local_nodes]  + cur_h);
                MPIU_Memcpy_CUDA(cuda_region_ptr,
                        buf, data_sz,
                        cudaMemcpyDefault);
    
                if (cuEventRecord(sr_event[vc->pg_rank], 0) != CUDA_SUCCESS) {
                    PRINT_ERROR("[%d] event record failed for rank : %d \n",MPIDI_Process.my_pg_rank, vc->pg_rank);
                    smp_error_abort(SMP_EXIT_ERR,"Error in recording the event\n");
                }
    
                cur_h += data_sz;
                ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head = cur_h;
    
            } else
#endif
            { 
                MPIU_Memcpy_CUDA((void *) ptr, buf, data_sz,
                        cudaMemcpyDeviceToHost);
                ptr = (volatile void *)((unsigned long) ptr + data_sz);
            }
        } else 
#endif
        {
            /* Copy data */
            memcpy((void *)ptr, buf, data_sz);
    
            /* Increment pointer */
            ptr = (volatile void *)((unsigned long) ptr + data_sz);
        }

        /* update(header) */
        smpi_complete_send(vc->smp.local_nodes, len, len, ptr, ptr_head, ptr_flag);
    } else {
        MPID_Request *sreq = NULL;
        sreq = create_eagercontig_request(vc, MPIDI_CH3_PKT_EAGER_SEND, buf,
                                            data_sz, rank, tag, comm, context_offset);
        if (sreq == NULL) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ch3|contigsend");
        }
        MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
        vc->smp.send_active = sreq;
        *sreq_p = sreq;
        PRINT_DEBUG(DEBUG_SHM_verbose>1,
                "smp buffer not available, dst: %d, request enqueued: %p, type: %d, ch.reqtype: %d\n",
                vc->pg_rank, sreq, MPIDI_Request_get_type(sreq), sreq->ch.reqtype);
    }

fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_write_contig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SMP_write_contig(MPIDI_VC_t * vc, MPIDI_CH3_Pkt_type_t reqtype,
                          const void * buf, MPIDI_msg_sz_t data_sz, int rank,
                          int tag, MPID_Comm * comm, int context_offset,
                          int *num_bytes_ptr)
{
#if defined(MPID_USE_SEQUENCE_NUMBERS)
    MPID_Seqnum_t seqnum;
#endif
    volatile void *ptr_head, *ptr, *ptr_flag;
    int len;
#if defined(_ENABLE_CUDA_)
    int buf_isdev = 0;
#if defined(HAVE_CUDA_IPC)
    int rem_rank = 0;
    int contig_avail = 0;
    void *cuda_region_ptr = NULL;
    smpi_cu_ipc_attr *my_base;
    int cur_t,cur_h;
#endif
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_WRITE_CONTIG);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_WRITE_CONTIG);
    *num_bytes_ptr = 0;
    len = data_sz + sizeof(MPIDI_CH3_Pkt_eager_send_t);
 
#if defined(_ENABLE_CUDA_)
    buf_isdev = is_device_buffer((void *) buf);
#if defined(HAVE_CUDA_IPC)
    if (rdma_enable_cuda && rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
        if (buf_isdev) {
            len = sizeof(MPIDI_CH3_Pkt_eager_send_t);
            rem_rank = vc->smp.local_nodes;
            my_base = g_smpi_shmem->cu_attrbs[g_smpi.my_local_id];
            cur_h = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head;
            cur_t = ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_tail;

            if(cur_t > cur_h) {
                contig_avail =  cur_t - cur_h; 
            } else {
                contig_avail =  smp_cuda_region_size - cur_h;
                if (contig_avail < data_sz) {
                    cur_h = 0;
                    contig_avail = cur_t;
                    if (cuStreamWaitEvent(0, loop_event_local[vc->pg_rank], 0) != CUDA_SUCCESS) {
                        PRINT_ERROR("Error in cuStreamWaitEvent\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }

            if(contig_avail < data_sz) {
                *num_bytes_ptr = 0;
                return;
                /* Not advancing header. Probably can be used for another
                   request in the queue?*/
            }
        }
    }
#endif
#endif


   ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
            s_header_ptr_s[vc->smp.local_nodes]);

    /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, len, (volatile void **)&ptr_flag, ONE_FREE))
        return;

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);

    MPIDI_CH3_Pkt_t *upkt;
    MPIDI_CH3_Pkt_eager_send_t * eager_pkt;
    *num_bytes_ptr = 0;

    upkt = (MPIDI_CH3_Pkt_t *) ptr;
    eager_pkt = &((*upkt).eager_send);
    MPIDI_Pkt_init(eager_pkt, reqtype);
    eager_pkt->match.parts.rank = comm->rank;
    eager_pkt->match.parts.tag  = tag;
    eager_pkt->match.parts.context_id   = comm->context_id + context_offset;
    eager_pkt->sender_req_id    = MPI_REQUEST_NULL;
    eager_pkt->data_sz      = data_sz;
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    eager_pkt->in_cuda_region = 0;
#endif

    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);

    *num_bytes_ptr += sizeof(MPIDI_CH3_Pkt_eager_send_t);
    ptr = (void *)((unsigned long) ptr + sizeof(MPIDI_CH3_Pkt_eager_send_t));

#if defined(_ENABLE_CUDA_)
    if (buf_isdev) {
#if defined(HAVE_CUDA_IPC)
        if (rdma_cuda_smp_ipc && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
            
            eager_pkt->in_cuda_region = 1;
            cuda_region_ptr = (void *) ((unsigned long) smp_cuda_region_send[vc->smp.local_nodes]  + cur_h);
            MPIU_Memcpy_CUDA(cuda_region_ptr,
                    buf, data_sz,
                    cudaMemcpyDefault);

            if (cuEventRecord(sr_event[vc->pg_rank], 0) != CUDA_SUCCESS) {
                PRINT_ERROR("[%d] event record failed for rank : %d \n",MPIDI_Process.my_pg_rank, vc->pg_rank);
                smp_error_abort(SMP_EXIT_ERR,"Error in recording the event\n");
            }

            cur_h += data_sz;
            ((smpi_cu_ipc_attr *)my_base + rem_rank)->cuda_head = cur_h;

            *num_bytes_ptr += data_sz;
        } else
#endif
        { 
            MPIU_Memcpy_CUDA((void *) ptr, buf, data_sz,
                    cudaMemcpyDeviceToHost);
            *num_bytes_ptr += data_sz;
            ptr = (volatile void *)((unsigned long) ptr + data_sz);
        }
    } else 
#endif
    {
        memcpy((void *)ptr, buf, data_sz);

        *num_bytes_ptr += data_sz;
        ptr = (volatile void *)((unsigned long) ptr + data_sz);
    }

    /* update(header) */
    smpi_complete_send(vc->smp.local_nodes, len, len, ptr, ptr_head, ptr_flag);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_WRITE_CONTIG);
}

#if defined (_ENABLE_CUDA_)
void smp_cuda_recv_copy_complete(MPIDI_VC_t * vc, MPID_Request *req, void *recv_buf_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int complete = 0;
    SEND_BUF_T *recv_buf; 

    recv_buf = (SEND_BUF_T *) recv_buf_ptr;
    *((volatile int *) &recv_buf->busy) = 0;

    /*This is the last chunk if request is present, complete it*/
    if (req != NULL) {
        mpi_errno =  MPIDI_CH3U_Handle_recv_req(vc, req, &complete);
        if (mpi_errno != MPI_SUCCESS) {
            smp_error_abort(SMP_EXIT_ERR,"handle_recv_req returned error\n");
        }
        MPIU_Assert(complete == 1);
    }
}

void smp_cuda_recv_copy_enqueue(MPIDI_VC_t * vc, MPID_Request *req, 
        SEND_BUF_T *recv_buf, int recv_offset, void *user_buf, size_t size, 
        int is_last)
{
    cudaError_t cuda_err = cudaSuccess; 
    cuda_event_t *cuda_event;

    MPIU_Memcpy_CUDA_Async(user_buf, 
            (void *) ((char *)recv_buf->buf + recv_offset), 
            size, 
            cudaMemcpyHostToDevice, 
            stream_h2d);

    cuda_event = get_cuda_event();
    if (cuda_event == NULL) {
        allocate_cuda_event(&cuda_event);
        /* add to the busy list */
        cuda_event->is_query_done = 0;
        CUDA_LIST_ADD(cuda_event,
            busy_cuda_event_list_head, busy_cuda_event_list_tail);
    }
    cuda_event->op_type = SMP_RECV;
    cuda_event->vc = vc;
    cuda_event->smp_ptr = (void *) recv_buf;
    cuda_event->size = size;
    if (is_last) { 
        cuda_event->req = req;
    } else {
        cuda_event->req = NULL;
    }

    cuda_err = cudaEventRecord(cuda_event->event, stream_h2d);
    if (cuda_err != cudaSuccess) {
        smp_error_abort(SMP_EXIT_ERR,"cudaEventRecord failed\n"); 
    }
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_readv_rndv_cuda
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_readv_rndv_cuda(MPIDI_VC_t *recv_vc_ptr, MPID_Request *req, 
        const MPL_IOV * iov, const int iov_count, int index, 
        size_t *num_bytes_ptr, int is_cont)
{
    int mpi_errno = MPI_SUCCESS;
    int iov_off = 0, buf_off = 0;
    int received_bytes = 0;
    int current_index = index;
    int recv_offset = 0;
    int is_last = 0;
    size_t msglen, iov_len;
    void *current_buf;
    SEND_BUF_T *recv_buf;
    void *ptr;
    volatile int *ptr_flag;

    /* all variable must be declared before the state declarations */
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_READV_RNDV_CUDA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_READV_RNDV_CUDA);

    if ( !mv2_shmem_pool_init )
    {
        if ((mpi_errno = MPIDI_CH3I_SMP_attach_shm_pool_inline()) !=
                MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    *num_bytes_ptr = 0;

    if (!is_cont) {
        /*if this is the first flit, complete the header receive*/
        MPIU_Assert(0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]);

        READBAR();
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;

        smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
            g_smpi.my_local_id,
            s_total_bytes[recv_vc_ptr->smp.local_nodes]);
  
        s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
    } 

    if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
        MPIU_Assert(s_total_bytes[recv_vc_ptr->smp.local_nodes] == 0);
 
        ptr = (void*)(g_smpi_shmem->pool +
                s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
        ptr_flag = (volatile int *) ptr;

        if (!is_cont) { 
            READBAR();
            while(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) {
               MV2_CUDA_PROGRESS();     
               READBAR();
            }
        } else {
            READBAR();
            if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) goto fn_exit;
        }
        READBAR();
        if(*ptr_flag == SMP_CBUF_END) {
            s_header_ptr_r[recv_vc_ptr->smp.local_nodes] =
                SMPI_FIRST_R(recv_vc_ptr->smp.local_nodes, g_smpi.my_local_id);
            ptr = (void*)(g_smpi_shmem->pool +
                    s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
            ptr_flag = (volatile int *) ptr;
            if (!is_cont) {
                READBAR();
                while(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) { 
                   MV2_CUDA_PROGRESS();
                   READBAR();
                }
            } else {
                READBAR();
                if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) goto fn_exit;
            }
        }

        READBAR();
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = ptr;
        s_total_bytes[recv_vc_ptr->smp.local_nodes] = *(int*)((unsigned long)ptr +
                sizeof(int));
        s_current_bytes[recv_vc_ptr->smp.local_nodes] =
            s_total_bytes[recv_vc_ptr->smp.local_nodes];

        WRITEBAR();

        if (!is_cont) { 
            smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
                g_smpi.my_local_id, 0);
        } else {
            s_current_ptr[recv_vc_ptr->smp.local_nodes] =
                (void *)((unsigned long) s_current_ptr 
                    [recv_vc_ptr->smp.local_nodes] + sizeof(int)*2);

            current_index = *((int *) s_current_ptr 
                    [recv_vc_ptr->smp.local_nodes]);
            if (current_index < 0 || current_index > 1024) { 
                exit (-1);
            }

            smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
                g_smpi.my_local_id, sizeof(int));
        }
    } else {
        s_total_bytes[recv_vc_ptr->smp.local_nodes] =
            s_current_bytes[recv_vc_ptr->smp.local_nodes];
        current_index = recv_vc_ptr->smp.read_index;
        recv_offset = recv_vc_ptr->smp.read_off;
    }

    if (current_index != -1) {
        recv_buf = SMPI_BUF_POOL_PTR(recv_vc_ptr->smp.local_nodes, current_index);

        MPIU_Assert(*((volatile int *) &recv_buf->busy) == 1);
 
        msglen = recv_buf->len - recv_offset;
        current_buf = (void *)((unsigned long) &recv_buf->buf + recv_offset);
        iov_len = iov[0].MPL_IOV_LEN;
 
        for (;
            iov_off < iov_count
            && s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0
            && current_index != -1;) {

            if (msglen > iov_len) {
                smp_cuda_recv_copy_enqueue(recv_vc_ptr, req, recv_buf, recv_offset,
                    (void *) ((unsigned long) iov[iov_off].MPL_IOV_BUF + buf_off),
                    iov_len, is_last);

                current_buf = (void *) ((unsigned long) current_buf +
                    iov_len);
                msglen -= iov_len;
                s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
                    iov_len;
                received_bytes += iov_len;
                buf_off = 0;
                ++iov_off;
  
                if (iov_off >= iov_count) {
                    recv_vc_ptr->smp.read_index = current_index;
                    recv_vc_ptr->smp.read_off = (unsigned long) current_buf -
                        (unsigned long) &recv_buf->buf;
                    break;
                }
  
                if (s_current_bytes[recv_vc_ptr->smp.local_nodes] <= 0) {
                    MPIU_Assert(s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0);
                    break;
                } 
  
                iov_len = iov[iov_off].MPL_IOV_LEN;
            } else if (msglen == iov_len) {
                if (iov_off == iov_count-1) {            
                   is_last = 1; 
                }

                smp_cuda_recv_copy_enqueue(recv_vc_ptr, req, recv_buf, recv_offset,
                    (void *) ((unsigned long) iov[iov_off].MPL_IOV_BUF + buf_off),
                    iov_len, is_last);

                s_current_bytes[recv_vc_ptr->smp.local_nodes] -= iov_len;
                received_bytes += iov_len;
                buf_off = 0;
                ++iov_off;
  
                if (iov_off >= iov_count) {
                    recv_vc_ptr->smp.read_index = recv_buf->next;
                    recv_vc_ptr->smp.read_off = 0;
                    break;
                }

                if (recv_buf->has_next == 0) { 
                    break;
                }

                current_index = recv_buf->next;
                recv_buf = SMPI_BUF_POOL_PTR(recv_vc_ptr->smp.local_nodes, current_index);
                MPIU_Assert(recv_buf->busy == 1);
                msglen = recv_buf->len;
                current_buf = (void *) &recv_buf->buf;
            } else if (msglen > 0) {
                smp_cuda_recv_copy_enqueue(recv_vc_ptr, req, recv_buf, recv_offset,
                    (void *) ((unsigned long) iov[iov_off].MPL_IOV_BUF + buf_off),
                    msglen, is_last);
                iov_len -= msglen;
                received_bytes += msglen;
                buf_off += msglen;
                s_current_bytes[recv_vc_ptr->smp.local_nodes] -= msglen;

                if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
                    break;
                }

                if (recv_buf->has_next == 0) {
                    break;
                }

                current_index = recv_buf->next;
                recv_buf = SMPI_BUF_POOL_PTR(recv_vc_ptr->smp.local_nodes, current_index);
                MPIU_Assert(recv_buf->busy == 1);
                msglen = recv_buf->len;
                current_buf = (void *) &recv_buf->buf;               
            }
        }
        *num_bytes_ptr += received_bytes;
        if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
            READBAR();
            s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;
            s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
        }
        received_bytes = 0;
        if (iov_off == iov_count) {
            MPIU_Assert(s_current_ptr[recv_vc_ptr->smp.local_nodes] == 0);
            goto fn_exit;
        }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_READV_RNDV_CUDA);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#endif

#if defined(_SMP_CMA_) 
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_do_cma_read
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_do_cma_read(const MPL_IOV * iov,
        const int iovlen, void *cma_header,
        size_t *num_bytes_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    size_t cerr;
    pid_t pid;
    struct cma_header *c_header = (struct cma_header *) cma_header;
    size_t cma_total_bytes = c_header->total_bytes;
    struct iovec *local_iovec;
    size_t msglen, iov_len;
    int iov_off = 0, buf_off = 0;
    size_t received_bytes = 0;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_DO_CMA_READ);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_DO_CMA_READ);

    local_iovec = (struct iovec *)iov;
    pid = c_header->pid;
    msglen = cma_total_bytes;
    iov_len = iov[0].MPL_IOV_LEN;
    for (; cma_total_bytes > 0 && iov_off < iovlen; ) {
        if (unlikely(msglen > MV2_CMA_MSG_LIMIT)) {
            local_iovec[iov_off].iov_len = MV2_CMA_MSG_LIMIT;
            cerr = process_vm_readv(pid, &local_iovec[iov_off], 1, c_header->remote, 1, 0);
            if( cerr == -1 ) 
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "**fail %s",
                        "CMA: (MPIDI_CH3I_SMP_do_cma_read) process_vm_readv fail");

            MPIU_Assert(cerr == MV2_CMA_MSG_LIMIT);
            local_iovec[iov_off].iov_base += MV2_CMA_MSG_LIMIT;
            local_iovec[iov_off].iov_len = iov_len - MV2_CMA_MSG_LIMIT;
            received_bytes += MV2_CMA_MSG_LIMIT;
            cma_total_bytes -= MV2_CMA_MSG_LIMIT;
            msglen -= MV2_CMA_MSG_LIMIT;
            iov_len -= MV2_CMA_MSG_LIMIT;

            c_header->remote[0].iov_len -= MV2_CMA_MSG_LIMIT;
            c_header->remote[0].iov_base += MV2_CMA_MSG_LIMIT;


        } else if (msglen == iov_len) {
            local_iovec[iov_off].iov_base += buf_off;
            cerr = process_vm_readv(pid, &local_iovec[iov_off], 1, c_header->remote, 1, 0);
            if( cerr == -1 ) 
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "**fail %s",
                        "CMA: (MPIDI_CH3I_SMP_do_cma_read) process_vm_readv fail");

            MPIU_Assert(cerr == msglen);
            received_bytes += msglen;
            cma_total_bytes -= msglen;

            MPIU_Assert(cma_total_bytes == 0 && ++iov_off >= iovlen);

        } else if (msglen > iov_len) {
            local_iovec[iov_off].iov_base += buf_off;
            cerr = process_vm_readv(pid, &local_iovec[iov_off], 1, c_header->remote, 1, 0);
            if( cerr == -1 ) 
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "**fail %s",
                        "CMA: (MPIDI_CH3I_SMP_do_cma_read) process_vm_readv fail");

            MPIU_Assert(cerr == iov_len);
            received_bytes += iov_len;
            cma_total_bytes -= iov_len;
            msglen -= iov_len;
           
            c_header->remote[0].iov_len -= iov_len;
            c_header->remote[0].iov_base += iov_len;

            if (++iov_off >= iovlen)
                break;
            buf_off = 0;
            iov_len = iov[iov_off].MPL_IOV_LEN;

        }  else if (msglen > 0) {
            local_iovec[iov_off].iov_base += buf_off;
            cerr = process_vm_readv(pid, &local_iovec[iov_off], 1, c_header->remote, 1, 0);
            if( cerr == -1 ) 
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "**fail %s",
                        "CMA: (MPIDI_CH3I_SMP_do_cma_read) process_vm_readv fail");

            MPIU_Assert(cerr == msglen);
            received_bytes += msglen;
            cma_total_bytes -= msglen;
        }
    }
    *num_bytes_ptr = received_bytes;
    c_header->total_bytes -= received_bytes;

fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "return with nb %ld\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_DO_CMA_READ);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#endif

#if defined(_SMP_LIMIC_)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_do_limic_read
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_do_limic_read (
        MPIDI_VC_t * recv_vc_ptr,
        const MPL_IOV * iov,
        const int iovlen, void *limic_header,
        size_t *num_bytes_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int iov_off = 0, buf_off = 0;
    size_t received_bytes = 0;
    size_t err, old_len;
    size_t msglen, iov_len;
    struct limic_header *l_header = (struct limic_header *) limic_header;
    size_t total_bytes = l_header->total_bytes;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_DO_LIMIC_READ);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_DO_LIMIC_READ);

    /* copy the message from the send buffer to the receive buffer */
    msglen = total_bytes;
    iov_len = iov[0].MPL_IOV_LEN;

    for (; total_bytes > 0 && iov_off < iovlen; ) {
        if (msglen == iov_len) {
            err = limic_rx_comp(limic_fd,
                    (void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF + buf_off),
                    msglen, &(l_header->lu));

            if (mv2_MPIDI_CH3I_RDMA_Process.g_smp_can_fallback && !err) {
                MPIDI_CH3I_SMP_send_comp(l_header, recv_vc_ptr, received_bytes, SMP_DMA_LIMIC, FALLBACK);
                *num_bytes_ptr = 0;
                l_header->total_bytes = 0;    
                recv_vc_ptr->smp.recv_active = NULL;
                goto fn_exit;
            }

            received_bytes += msglen;
            total_bytes -= msglen;

            assert(total_bytes == 0 && ++iov_off >= iovlen);

        } else if (msglen > iov_len) {
            old_len = l_header->lu.length;
            err = limic_rx_comp(limic_fd,
                    (void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF + buf_off),
                    iov_len, &(l_header->lu));

            if (mv2_MPIDI_CH3I_RDMA_Process.g_smp_can_fallback && !err) {
                MPIDI_CH3I_SMP_send_comp(l_header, recv_vc_ptr, received_bytes, SMP_DMA_LIMIC, FALLBACK);
                *num_bytes_ptr = 0;
                l_header->total_bytes = 0;    
                recv_vc_ptr->smp.recv_active = NULL;
                goto fn_exit;
            }

            received_bytes += iov_len;
            total_bytes -= iov_len;
            msglen -= iov_len;

            adjust_lu_info(&(l_header->lu), old_len);

            if (++iov_off >= iovlen)
                break;
            buf_off = 0;
            iov_len = iov[iov_off].MPL_IOV_LEN;

        }  else if (msglen > 0) {
            err = limic_rx_comp(limic_fd,
                    (void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF + buf_off),
                    msglen, &(l_header->lu));
            if (mv2_MPIDI_CH3I_RDMA_Process.g_smp_can_fallback && !err) {
                MPIDI_CH3I_SMP_send_comp(l_header, recv_vc_ptr, received_bytes, SMP_DMA_LIMIC, FALLBACK);
                *num_bytes_ptr = 0;
                l_header->total_bytes = 0;    
                recv_vc_ptr->smp.recv_active = NULL;
                goto fn_exit;
            }

            received_bytes += msglen;
            total_bytes -= msglen;
        }
    }

    *num_bytes_ptr = received_bytes;
    l_header->total_bytes -= received_bytes;

fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "return with nb %ld\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_DO_LIMIC_READ);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#endif

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_readv_rndv_cont
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_readv_rndv_cont(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
        const int iovlen, int index, void *limic_header,
        void *cma_header, size_t *num_bytes_ptr, smp_dma_flag_t dma_flag)
{
    int mpi_errno = MPI_SUCCESS;
    int iov_off = 0, buf_off = 0;
    int received_bytes = 0;
    int destination = recv_vc_ptr->smp.local_nodes;
    int current_index = index;
    int recv_offset = 0;
    size_t msglen, iov_len;
    void *current_buf;
    SEND_BUF_T *recv_buf;

#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    /* all variable must be declared before the state declarations */
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_READV_RNDV_CONT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_READV_RNDV_CONT);

    *num_bytes_ptr = 0;

#if defined(_SMP_CMA_) 
    if (dma_flag == SMP_DMA_CMA) {
        mpi_errno = MPIDI_CH3I_SMP_do_cma_read(
                iov, iovlen, cma_header, num_bytes_ptr);
    } else {
#endif /* _SMP_CMA_ */

#if defined(_SMP_LIMIC_)
    if (dma_flag == SMP_DMA_LIMIC) {
        mpi_errno = MPIDI_CH3I_SMP_do_limic_read(
                recv_vc_ptr, iov, iovlen, limic_header, num_bytes_ptr);
    } else {
#endif /* _SMP_LIMIC_ */

    if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
    if(s_total_bytes[recv_vc_ptr->smp.local_nodes] != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
            "**fail %s", "s_total_bytes[recv_vc_ptr->smp.local_nodes] "
            "!= 0");
    }

    void *ptr;
    ptr = (void*)(g_smpi_shmem->pool +
            s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
    volatile int *ptr_flag;
    ptr_flag = (volatile int *) ptr;

    READBAR();
    if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) goto fn_exit;
    READBAR();
    if(*ptr_flag == SMP_CBUF_END) {
        s_header_ptr_r[recv_vc_ptr->smp.local_nodes] =
            SMPI_FIRST_R(recv_vc_ptr->smp.local_nodes, g_smpi.my_local_id);
        ptr = (void*)(g_smpi_shmem->pool +
                s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
        ptr_flag = (volatile int *) ptr;
        READBAR();
        if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) goto fn_exit;
    }

    READBAR();
    s_current_ptr[recv_vc_ptr->smp.local_nodes] = ptr;
    s_total_bytes[recv_vc_ptr->smp.local_nodes] = *(int*)((unsigned long)ptr +
            sizeof(int));
    s_current_bytes[recv_vc_ptr->smp.local_nodes] =
        s_total_bytes[recv_vc_ptr->smp.local_nodes];

    MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_smp_rndv_avail_buffer, s_current_bytes[recv_vc_ptr->smp.local_nodes]); 

    rndv_buffer_max_usage = (rndv_buffer_max_usage > s_current_bytes[recv_vc_ptr->smp.local_nodes]) ?
        rndv_buffer_max_usage : s_current_bytes[recv_vc_ptr->smp.local_nodes]; 
    MPIR_T_PVAR_LEVEL_SET(MV2, mv2_smp_rndv_buffer_max_use, rndv_buffer_max_usage);

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current byte %ld, total bytes %ld, iovlen %d, iov[0].len %ld\n",
         s_current_bytes[recv_vc_ptr->smp.local_nodes],
         s_total_bytes[recv_vc_ptr->smp.local_nodes], iovlen,
         iov[0].MPL_IOV_LEN);
    WRITEBAR();

    s_current_ptr[recv_vc_ptr->smp.local_nodes] =
        (void *)((unsigned long) s_current_ptr[recv_vc_ptr->smp.local_nodes] +
            sizeof(int)*2);
    current_index = *((int *) s_current_ptr[recv_vc_ptr->smp.local_nodes]);
    smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
        g_smpi.my_local_id, sizeof(int));
    } else {
    s_total_bytes[recv_vc_ptr->smp.local_nodes] =
        s_current_bytes[recv_vc_ptr->smp.local_nodes];
    current_index = recv_vc_ptr->smp.read_index;
    recv_offset = recv_vc_ptr->smp.read_off;
    }

    if (current_index != -1) {
    /** last smp packet has not been drained up yet **/
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "iov_off %d, current bytes %ld, iov len %ld\n",
        iov_off, s_current_bytes[recv_vc_ptr->smp.local_nodes],
        iov[iov_off].MPL_IOV_LEN);

    recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);

    if(recv_buf->busy != 1) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
            "**fail %s", "recv_buf->busy != 1");
    }

    msglen = recv_buf->len - recv_offset;
    current_buf = (void *)((unsigned long) &recv_buf->buf + recv_offset);
    iov_len = iov[0].MPL_IOV_LEN;

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        /*as it is all data, we check the first iov to determine if the buffer is on device*/
        iov_isdev = is_device_buffer((void *) iov[0].MPL_IOV_BUF);
    }
#endif

    for (;
        iov_off < iovlen
        && s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0
        && current_index != -1;) {

        if (msglen > iov_len) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {  
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        iov_len,
                        cudaMemcpyHostToDevice);
        } else
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        iov_len);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, iov_len);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, iov_len);

        READBAR();
        current_buf = (void *) ((unsigned long) current_buf +
            iov_len);
        msglen -= iov_len;
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
            iov_len;
        received_bytes += iov_len;
        buf_off = 0;
        ++iov_off;

        if (iov_off >= iovlen) {
            recv_vc_ptr->smp.read_index = current_index;
            recv_vc_ptr->smp.read_off = (unsigned long) current_buf -
            (unsigned long) &recv_buf->buf;
            break;
        }

        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
            recv_buf->busy = 0;
            break;
        }

        else if (s_current_bytes[recv_vc_ptr->smp.local_nodes] < 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**fail", "**fail %s",
                "s_current_bytes[recv_vc_ptr->smp.local_nodes] < 0");
        }

        iov_len = iov[iov_off].MPL_IOV_LEN;
        } else if (msglen == iov_len) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        iov_len, 
                        cudaMemcpyHostToDevice);
        } else 
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        iov_len);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, iov_len);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, iov_len);

        READBAR();
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
            iov_len;
        received_bytes += iov_len;
        buf_off = 0;
        ++iov_off;

        if (iov_off >= iovlen) {
            recv_vc_ptr->smp.read_index = recv_buf->next;
            recv_vc_ptr->smp.read_off = 0;
            recv_buf->busy = 0;
            break;
        }
        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] <= 0) {
            MPIU_Assert(s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0);
            recv_buf->busy = 0;
            break;
        }

        iov_len = iov[iov_off].MPL_IOV_LEN;

        if(recv_buf->has_next == 0){
            recv_buf->busy = 0;
            break;
        }

        current_index = recv_buf->next;
        recv_buf->busy = 0;
        recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);
        MPIU_Assert(recv_buf->busy == 1);
        msglen = recv_buf->len;
        current_buf = (void *) &recv_buf->buf;

        } else if (msglen > 0) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        msglen,
                        cudaMemcpyHostToDevice);
        } else 
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                        + buf_off),
                        (void *) current_buf, 
                        msglen);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, msglen);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, msglen);

        READBAR();
        iov_len -= msglen;
        received_bytes += msglen;
        buf_off += msglen;
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -= msglen;

        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
            recv_buf->busy = 0;
            break;
        }
        if(recv_buf->has_next == 0){
            recv_buf->busy = 0;
            break;
        }

        current_index = recv_buf->next;
        recv_buf->busy = 0;
        recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);
        MPIU_Assert(recv_buf->busy == 1);
        msglen = recv_buf->len;
        current_buf = (void *) &recv_buf->buf;
        }
    }
    *num_bytes_ptr += received_bytes;
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current bytes %ld, num_bytes %ld, iov_off %d, iovlen %d\n",
         s_current_bytes[recv_vc_ptr->smp.local_nodes], *num_bytes_ptr,
         iov_off, iovlen);

    if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
        READBAR();
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;
        s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
    }
    received_bytes = 0;
    if (iov_off == iovlen) {
        /* assert: s_current_ptr[recv_vc_ptr->smp.local_nodes] == 0 */
        goto fn_exit;
    }
    }
#if defined(_SMP_LIMIC_)
    }
#endif
#if defined(_SMP_CMA_) 
    }
#endif

fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "return with nb %ld\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_READV_RNDV_CONT);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_readv_rndv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_readv_rndv(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
        const int iovlen, int index, void *limic_header,
        void *cma_header, size_t *num_bytes_ptr, smp_dma_flag_t dma_flag)
{
    int mpi_errno = MPI_SUCCESS;
    size_t iov_off = 0, buf_off = 0;
    size_t received_bytes = 0;
    size_t msglen, iov_len;
    /* all variable must be declared before the state declarations */

    int destination = recv_vc_ptr->smp.local_nodes;
    int current_index = index;
    void *current_buf;
    SEND_BUF_T *recv_buf;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_READ_RNDV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_READ_RNDV);

    if ( !mv2_shmem_pool_init )
    {
        if ((mpi_errno = MPIDI_CH3I_SMP_attach_shm_pool_inline()) !=
                MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    *num_bytes_ptr = 0;

    if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
    READBAR();
    s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;

    smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
        g_smpi.my_local_id,
        s_total_bytes[recv_vc_ptr->smp.local_nodes]);

    s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
    }
#if defined(_SMP_CMA_) 
    if (dma_flag == SMP_DMA_CMA) {
        mpi_errno = MPIDI_CH3I_SMP_do_cma_read(
                iov, iovlen, cma_header, num_bytes_ptr);
    } else {
#endif /* _SMP_CMA_ */

#if defined(_SMP_LIMIC_)
    if (dma_flag == SMP_DMA_LIMIC) {
        mpi_errno = MPIDI_CH3I_SMP_do_limic_read(
                recv_vc_ptr, iov, iovlen, limic_header, num_bytes_ptr);
    } else {
#endif /* _SMP_LIMIC_ */

    volatile void *ptr;
    volatile int *ptr_flag;
    ptr = (void*)((unsigned long)g_smpi_shmem->pool + 
            s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
    ptr_flag = (volatile int *) ptr;
    READBAR();
    while(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) {
#if defined (_ENABLE_CUDA_)
        MV2_CUDA_PROGRESS();
#endif 
        READBAR();
    }

    READBAR();
    if(*ptr_flag == SMP_CBUF_END) {
    s_header_ptr_r[recv_vc_ptr->smp.local_nodes] =
            SMPI_FIRST_R(recv_vc_ptr->smp.local_nodes, g_smpi.my_local_id);
    ptr = (volatile void*)((unsigned long)g_smpi_shmem->pool +
            s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
    ptr_flag = (volatile int *)ptr;
    READBAR();
    while(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) {
#if defined (_ENABLE_CUDA_)
        MV2_CUDA_PROGRESS();
#endif         
        READBAR();
    }
    }

    READBAR();
    s_current_ptr[recv_vc_ptr->smp.local_nodes] = (void *)ptr;
    s_total_bytes[recv_vc_ptr->smp.local_nodes] = *(int*)((unsigned long)ptr +
            sizeof(int));
    s_current_bytes[recv_vc_ptr->smp.local_nodes] =
    s_total_bytes[recv_vc_ptr->smp.local_nodes];
    smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
        g_smpi.my_local_id,0);
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current byte %ld, total bytes %ld, iovlen %d, iov[0].len %ld\n",
     s_current_bytes[recv_vc_ptr->smp.local_nodes],
     s_total_bytes[recv_vc_ptr->smp.local_nodes], iovlen,
     iov[0].MPL_IOV_LEN);

    MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_smp_rndv_avail_buffer, s_current_bytes[recv_vc_ptr->smp.local_nodes]); 
    rndv_buffer_max_usage = (rndv_buffer_max_usage > s_current_bytes[recv_vc_ptr->smp.local_nodes]) ?
        rndv_buffer_max_usage : s_current_bytes[recv_vc_ptr->smp.local_nodes]; 
    MPIR_T_PVAR_LEVEL_SET(MV2, mv2_smp_rndv_buffer_max_use, rndv_buffer_max_usage);

    WRITEBAR();

    if (current_index != -1) {
    /** last smp packet has not been drained up yet **/
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "iov_off %ld, current bytes %ld, iov len %ld\n",
        iov_off, s_current_bytes[recv_vc_ptr->smp.local_nodes],
        iov[iov_off].MPL_IOV_LEN);

    recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);

    if(recv_buf->busy != 1) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
            "**fail %s", "recv_buf->busy == 1");
    }

    msglen = recv_buf->len;
    current_buf = (void *) &recv_buf->buf;
    iov_len = iov[0].MPL_IOV_LEN;

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        /*as it is all data, we check the first iov to determine if the buffer is on device*/
        iov_isdev = is_device_buffer((void *) iov[0].MPL_IOV_BUF);
    }
#endif


    for (;
        iov_off < iovlen
        && s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0
        && current_index != -1;) {

        if (msglen > iov_len) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf,   
                        iov_len, 
                        cudaMemcpyHostToDevice); 
        } else 
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf,   
                        iov_len);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, iov_len);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, iov_len);

        READBAR();
        current_buf = (void *) ((unsigned long) current_buf +
            iov_len);
        msglen -= iov_len;
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
            iov_len;
        received_bytes += iov_len;
        buf_off = 0;
        ++iov_off;

        if (iov_off >= iovlen) {
            recv_vc_ptr->smp.read_index = current_index;
            recv_vc_ptr->smp.read_off = (unsigned long) current_buf -
            (unsigned long) &recv_buf->buf;
            break;
        }
        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] <= 0) {
            MPIU_Assert(s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0);
            recv_buf->busy = 0;
            break;
        }

        iov_len = iov[iov_off].MPL_IOV_LEN;
        } else if (msglen == iov_len) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf, 
                        iov_len,
                        cudaMemcpyHostToDevice);
        } else   
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf, 
                        iov_len);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, iov_len);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, iov_len);

        READBAR();
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
            iov_len;
        received_bytes += iov_len;
        buf_off = 0;
        ++iov_off;

        if (iov_off >= iovlen) {
            recv_vc_ptr->smp.read_index = recv_buf->next;
            recv_vc_ptr->smp.read_off = 0;
            recv_buf->busy = 0;
            break;
        }

        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
            recv_buf->busy = 0;
            break;
        }

        else if(s_current_bytes[recv_vc_ptr->smp.local_nodes] < 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**fail", "**fail %s",
                "s_current_bytes[recv_vc_ptr->smp.local_nodes] < 0");
        }

        iov_len = iov[iov_off].MPL_IOV_LEN;

        if(recv_buf->has_next == 0){
            recv_buf->busy = 0;
            break;
        }

        current_index = recv_buf->next;
        recv_buf->busy = 0;
        recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);

        if(recv_buf->busy != 1) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**fail", "**fail %s", "recv_buf->busy != 1");
        }

        msglen = recv_buf->len;
        current_buf = (void *) &recv_buf->buf;

        } else if (msglen > 0) {
        READBAR();
#if defined(_ENABLE_CUDA_)
        if (iov_isdev) {
            MPIU_Memcpy_CUDA((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf, 
                        msglen,
                        cudaMemcpyHostToDevice);
        } else 
#endif
        {
            MPIU_Memcpy((void *) ((unsigned long)iov[iov_off].MPL_IOV_BUF 
                            + buf_off),
                        (void *) current_buf, 
                        msglen);
        }

        MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_rndv_received, msglen);
        MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_rndv_avail_buffer, msglen);

        READBAR();
        iov_len -= msglen;
        received_bytes += msglen;
        buf_off += msglen;
        s_current_bytes[recv_vc_ptr->smp.local_nodes] -= msglen;

        if (s_current_bytes[recv_vc_ptr->smp.local_nodes] == 0) {
            recv_buf->busy = 0;
            break;
        }
        if(recv_buf->has_next == 0){
            recv_buf->busy = 0;
            break;
        }

        current_index = recv_buf->next;
        recv_buf->busy = 0;
        recv_buf = SMPI_BUF_POOL_PTR(destination, current_index);

        if(recv_buf->busy != 1) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**fail", "**fail %s", "recv_buf->busy != 1");
        }

        msglen = recv_buf->len;
        current_buf = (void *) &recv_buf->buf;
        }
    }
    *num_bytes_ptr += received_bytes;
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "current bytes %ld, num_bytes %ld, iov_off %ld, iovlen %d\n",
         s_current_bytes[recv_vc_ptr->smp.local_nodes], *num_bytes_ptr,
         iov_off, iovlen);

    if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
        READBAR();
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;

        s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
    }
    received_bytes = 0;
    if (iov_off == iovlen) {
        /* assert: s_current_ptr[recv_vc_ptr->smp.local_nodes] == 0 */
        goto fn_exit;
    }
    }
#if defined(_SMP_LIMIC_) 
    }
#endif
#if defined(_SMP_CMA_) 
    }
#endif
fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "return with nb %ld\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_READ_RNDV);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_readv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_readv(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
    const int iovlen, size_t  *num_bytes_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    int iov_off = 0, buf_off=0;
    int received_bytes = 0;
#if defined(_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif
    /* all variable must be declared before the state declarations */

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_READV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_READV);

    *num_bytes_ptr = 0;

    if (s_current_ptr[recv_vc_ptr->smp.local_nodes] != NULL) {

        MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_smp_eager_avail_buffer, s_current_bytes[recv_vc_ptr->smp.local_nodes]); 

        eager_buffer_max_usage = (eager_buffer_max_usage > s_current_bytes[recv_vc_ptr->smp.local_nodes]) ?
                                  eager_buffer_max_usage : s_current_bytes[recv_vc_ptr->smp.local_nodes]; 
        MPIR_T_PVAR_LEVEL_SET(MV2, mv2_smp_eager_buffer_max_use, eager_buffer_max_usage);

        for (;
                iov_off < iovlen
                && s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0;) {
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda) {
                iov_isdev = is_device_buffer((void *) iov[iov_off].MPL_IOV_BUF);
            }
#endif
            if (s_current_bytes[recv_vc_ptr->smp.local_nodes] >=
                    iov[iov_off].MPL_IOV_LEN) {

                READBAR();
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                    MPIU_Memcpy_CUDA((void *) iov[iov_off].MPL_IOV_BUF,
                            (void *) s_current_ptr[recv_vc_ptr->smp.local_nodes],
                            iov[iov_off].MPL_IOV_LEN,
                            cudaMemcpyHostToDevice);
                } else 
#endif
                {
                    MPIU_Memcpy((void *) iov[iov_off].MPL_IOV_BUF,
                            s_current_ptr[recv_vc_ptr->smp.local_nodes],
                            iov[iov_off].MPL_IOV_LEN);
                }

                MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_eager_received, iov[iov_off].MPL_IOV_LEN);
                MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_avail_buffer, iov[iov_off].MPL_IOV_LEN); 

                READBAR();
                s_current_ptr[recv_vc_ptr->smp.local_nodes] =
                    (void *) ((unsigned long)
                            s_current_ptr[recv_vc_ptr->smp.local_nodes] +
                            iov[iov_off].MPL_IOV_LEN);
                s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
                    iov[iov_off].MPL_IOV_LEN;
                received_bytes += iov[iov_off].MPL_IOV_LEN;
                ++iov_off;
            } else if (s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0) {
                READBAR();
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                    MPIU_Memcpy_CUDA((void *) iov[iov_off].MPL_IOV_BUF,
                            (void *) s_current_ptr[recv_vc_ptr->smp.local_nodes],
                            s_current_bytes[recv_vc_ptr->smp.local_nodes],
                            cudaMemcpyHostToDevice);
                } else 
#endif
                {
                    MPIU_Memcpy((void *) iov[iov_off].MPL_IOV_BUF,
                            (void *) s_current_ptr[recv_vc_ptr->smp.local_nodes],
                            s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                }

                MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_eager_received, s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_avail_buffer, s_current_bytes[recv_vc_ptr->smp.local_nodes]); 

                READBAR();
                s_current_ptr[recv_vc_ptr->smp.local_nodes] =
                    (void *) ((unsigned long)
                            s_current_ptr[recv_vc_ptr->smp.local_nodes] +
                            s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                received_bytes +=
                    s_current_bytes[recv_vc_ptr->smp.local_nodes];
                buf_off = s_current_bytes[recv_vc_ptr->smp.local_nodes];
                s_current_bytes[recv_vc_ptr->smp.local_nodes] = 0;
            }
        }
        *num_bytes_ptr += received_bytes;
        if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
            s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;

            smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
                    g_smpi.my_local_id,
                    s_total_bytes[recv_vc_ptr->smp.local_nodes]);

            s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
        }

        received_bytes = 0;
        if (iov_off == iovlen) {
            /* assert: s_current_ptr[recv_vc_ptr->smp.local_nodes] == 0 */
            goto fn_exit;
        }

    }
    WRITEBAR();

    void *ptr;
    volatile int *ptr_flag;
    ptr = (void*)(g_smpi_shmem->pool + s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
    ptr_flag = (volatile int*)ptr;

    READBAR();
    while(*ptr_flag != SMP_CBUF_FREE && *ptr_flag != SMP_CBUF_PENDING) {
        READBAR();
        if(*ptr_flag == SMP_CBUF_END) {
            s_header_ptr_r[recv_vc_ptr->smp.local_nodes] =
                SMPI_FIRST_R(recv_vc_ptr->smp.local_nodes, g_smpi.my_local_id);

            ptr = (void*)(g_smpi_shmem->pool +
                    s_header_ptr_r[recv_vc_ptr->smp.local_nodes]);
            ptr_flag = (volatile int*)ptr;

            READBAR();
            if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING)
                goto fn_exit;
        }

        READBAR();
        s_total_bytes[recv_vc_ptr->smp.local_nodes] = *(int*)((unsigned long)ptr +
                sizeof(int));
        ptr = (void *)((unsigned long)ptr + sizeof(int)*2);
        WRITEBAR();
        s_current_bytes[recv_vc_ptr->smp.local_nodes] =
            s_total_bytes[recv_vc_ptr->smp.local_nodes];

        MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_smp_eager_avail_buffer, s_current_bytes[recv_vc_ptr->smp.local_nodes]); 
        eager_buffer_max_usage = (eager_buffer_max_usage > s_current_bytes[recv_vc_ptr->smp.local_nodes]) ?
                                  eager_buffer_max_usage : s_current_bytes[recv_vc_ptr->smp.local_nodes]; 
        MPIR_T_PVAR_LEVEL_SET(MV2, mv2_smp_eager_buffer_max_use, eager_buffer_max_usage);

        READBAR();
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = ptr;

        /****** starting to fill the iov buffers *********/
        for (;
                iov_off < iovlen
                && s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0;) {
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda) {
                iov_isdev = is_device_buffer((void *) iov[iov_off].MPL_IOV_BUF);
            }
#endif
            if (s_current_bytes[recv_vc_ptr->smp.local_nodes] >=
                    iov[iov_off].MPL_IOV_LEN - buf_off) {

                WRITEBAR();
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                    MPIU_Memcpy_CUDA((void *) ((unsigned long) iov[iov_off].
                                MPL_IOV_BUF + buf_off),
                            ptr, iov[iov_off].MPL_IOV_LEN - buf_off,
                            cudaMemcpyHostToDevice);
                } else 
#endif
                {
                    MPIU_Memcpy((void *) ((unsigned long) iov[iov_off].
                                MPL_IOV_BUF + buf_off),
                            ptr, iov[iov_off].MPL_IOV_LEN - buf_off);
                }

                MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_eager_received, iov[iov_off].MPL_IOV_LEN - buf_off);
                MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_avail_buffer, iov[iov_off].MPL_IOV_LEN - buf_off);

                READBAR();
                s_current_bytes[recv_vc_ptr->smp.local_nodes] -=
                    (iov[iov_off].MPL_IOV_LEN - buf_off);
                received_bytes += ( iov[iov_off].MPL_IOV_LEN - buf_off);
                ptr = (void*)((unsigned long)ptr + 
                        (iov[iov_off].MPL_IOV_LEN - buf_off));
                ++iov_off;
                buf_off = 0;
            } else if (s_current_bytes[recv_vc_ptr->smp.local_nodes] > 0) {
                WRITEBAR();
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                    MPIU_Memcpy_CUDA((void *) ((unsigned long) iov[iov_off].
                                MPL_IOV_BUF + buf_off),
                            ptr, s_current_bytes[recv_vc_ptr->smp.local_nodes],
                            cudaMemcpyHostToDevice);
                } else  
#endif
                {
                    MPIU_Memcpy((void *) ((unsigned long) iov[iov_off].
                                MPL_IOV_BUF + buf_off),
                            ptr, s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                }

                MPIR_T_PVAR_COUNTER_INC(MV2, mv2_smp_eager_received, 
                        s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                MPIR_T_PVAR_LEVEL_INC(MV2, mv2_smp_eager_avail_buffer, 
                        s_current_bytes[recv_vc_ptr->smp.local_nodes]);

                READBAR();
                ptr = (void*)((unsigned long)ptr + s_current_bytes[recv_vc_ptr->smp.local_nodes]);
                received_bytes +=
                    s_current_bytes[recv_vc_ptr->smp.local_nodes];
                buf_off += s_current_bytes[recv_vc_ptr->smp.local_nodes];
                s_current_bytes[recv_vc_ptr->smp.local_nodes] = 0;
            }
        }
        *num_bytes_ptr += received_bytes;
        s_current_ptr[recv_vc_ptr->smp.local_nodes] = ptr;

        /* update header */
        if (0 == s_current_bytes[recv_vc_ptr->smp.local_nodes]) {
            READBAR();
            s_current_ptr[recv_vc_ptr->smp.local_nodes] = NULL;
            smpi_complete_recv(recv_vc_ptr->smp.local_nodes,
                    g_smpi.my_local_id,
                    s_total_bytes[recv_vc_ptr->smp.local_nodes]);
            s_total_bytes[recv_vc_ptr->smp.local_nodes] = 0;
        }
        received_bytes = 0;
        if (iov_off == iovlen) {
            goto fn_exit;
        }
        WRITEBAR();
    }
fn_exit:
    PRINT_DEBUG(DEBUG_SHM_verbose>1, "return with nb %ld\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_READV);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_pull_header
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_pull_header(MPIDI_VC_t* vc, MPIDI_CH3_Pkt_t** pkt_head)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_PULL_HEADER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_PULL_HEADER);

    if(s_total_bytes[vc->smp.local_nodes] != 0) {
        s_current_ptr[vc->smp.local_nodes] = NULL;
        smpi_complete_recv(vc->smp.local_nodes,
                    g_smpi.my_local_id,
                    s_total_bytes[vc->smp.local_nodes]);
        s_total_bytes[vc->smp.local_nodes] = 0;
    }

    void *ptr;
    volatile int *ptr_flag;
    ptr = (void*)(g_smpi_shmem->pool + s_header_ptr_r[vc->smp.local_nodes]);
    ptr_flag = (volatile int*)ptr;

    READBAR();
    if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) {
        *pkt_head = NULL;
    } else {
        READBAR();
        if(*ptr_flag == SMP_CBUF_END) {
            *pkt_head = NULL;
            s_header_ptr_r[vc->smp.local_nodes] =
                SMPI_FIRST_R(vc->smp.local_nodes, g_smpi.my_local_id);
            ptr = (void*)(g_smpi_shmem->pool +
                    s_header_ptr_r[vc->smp.local_nodes]);
            ptr_flag = (volatile int*)ptr;

            READBAR();
            if(*ptr_flag == SMP_CBUF_FREE || *ptr_flag == SMP_CBUF_PENDING) {
                goto fn_exit;
            }
        }
        READBAR();
        s_total_bytes[vc->smp.local_nodes] = *(int*)((unsigned long)ptr + sizeof(int));
        *pkt_head = (void *)((unsigned long)ptr + sizeof(int)*2);
        WRITEBAR();
        s_current_bytes[vc->smp.local_nodes] =
            s_total_bytes[vc->smp.local_nodes] - MPIDI_CH3U_PKT_SIZE(*pkt_head);
        s_current_ptr[vc->smp.local_nodes] = (void*)((unsigned long) *pkt_head
                + MPIDI_CH3U_PKT_SIZE(*pkt_head));
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_PULL_HEADER);
    return MPI_SUCCESS;
}


static int smpi_exchange_info(MPIDI_PG_t *pg)
{
    int mpi_errno = MPI_SUCCESS;
    int pg_rank, pg_size;

    int i = 0;
    int j;
    char *value = NULL;

    MPIDI_VC_t* vc = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMPI_EXCHANGE_INFO);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMPI_EXCHANGE_INFO);

    UPMI_GET_RANK(&pg_rank);
    UPMI_GET_SIZE(&pg_size);

    g_smpi.num_local_nodes = MPIDI_Num_local_processes(pg);
    if (g_smpi.num_local_nodes >= MV2_SHMEM_PRIORTY_THRESHOLD)
        g_smp_priority_factor = MV2_SHMEM_PRIORTY_FACTOR;

    if ((value = getenv("MV2_SMP_PRIORITY_FACTOR")) != NULL) {
        g_smp_priority_factor = atoi(value);
        if (g_smp_priority_factor < 1)
            g_smp_priority_factor = 1;
    }

    for (i = 0; i < pg->size; ++i) {
        MPIDI_PG_Get_vc(pg, i, &vc);
        vc->smp.local_nodes = vc->smp.local_rank;
    }

    /* Get my VC */
    MPIDI_PG_Get_vc(pg, pg_rank, &vc);
    g_smpi.my_local_id = vc->smp.local_nodes;

    PRINT_DEBUG(DEBUG_SHM_verbose>1, "num local nodes %d, my local id %d\n",
        g_smpi.num_local_nodes, g_smpi.my_local_id);

    g_smpi.l2g_rank = (unsigned int *) MPIU_Malloc(g_smpi.num_local_nodes * sizeof(int));
    if(g_smpi.l2g_rank == NULL) {
    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
        "**nomem %s", "g_smpi.12g_rank");
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#pragma error_messages(off, E_STATEMENT_NOT_REACHED)
#endif /* defined(__SUNPRO_C) || defined(__SUNPRO_CC) */
    MPL_error_printf(
        "malloc: in ib_rank_lid_table for SMP");
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#pragma error_messages(default, E_STATEMENT_NOT_REACHED)
#endif /* defined(__SUNPRO_C) || defined(__SUNPRO_CC) */
    }

    for (i = 0, j = 0; j < pg_size; ++j) {
        MPIDI_PG_Get_vc(pg, j, &vc);

        if (vc->smp.local_nodes != -1) {
            g_smpi.l2g_rank[i] = j;
            i++;
    	}
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMPI_EXCHANGE_INFO);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

/*----------------------------------------------------------*/
    static inline SEND_BUF_T *
get_buf_from_pool ()
{
    SEND_BUF_T *ptr;

    if (s_sh_buf_pool.free_head == -1) 
    return NULL;

    ptr = SMPI_MY_BUF_POOL_PTR(s_sh_buf_pool.free_head); 
    s_sh_buf_pool.free_head = ptr->next;
    ptr->next = -1;

    MPIU_Assert (ptr->busy == 0);

    return ptr;
}

static inline void send_buf_reclaim ()
{
    int i, index, last_index;
    SEND_BUF_T *ptr;

    for (i = 0; i < g_smpi.num_local_nodes; ++i) {
    if (i != g_smpi.my_local_id) {
        index = s_sh_buf_pool.send_queue[i];
        last_index = -1;
        ptr = NULL;
        while (index != -1) {
        ptr = SMPI_MY_BUF_POOL_PTR(index);
        if(ptr->busy == 1)
            break;
        last_index = index;
        index = ptr->next;
        }
        if (last_index != -1)
        put_buf_to_pool (s_sh_buf_pool.send_queue[i], last_index);
        s_sh_buf_pool.send_queue[i] = index;
        if (s_sh_buf_pool.send_queue[i] == -1)
        s_sh_buf_pool.tail[i] = -1;
    }
    }
}

    static inline void
put_buf_to_pool (int head, int tail)
{
    SEND_BUF_T *ptr;

    MPIU_Assert (head != -1);
    MPIU_Assert (tail != -1);

    ptr = SMPI_MY_BUF_POOL_PTR(tail);

    ptr->next = s_sh_buf_pool.free_head;
    s_sh_buf_pool.free_head = head;
}

static inline void link_buf_to_send_queue (int dest, int index)
{
    if (s_sh_buf_pool.send_queue[dest] == -1) {
        s_sh_buf_pool.send_queue[dest] = index;
    } else {
        SMPI_MY_BUF_POOL_PTR(s_sh_buf_pool.tail[dest])->next = index;
    }
    s_sh_buf_pool.tail[dest] = index;
}

#if defined(_SMP_LIMIC_)
#undef FUNCNAME
#define FUNCNAME adjust_lu_info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline void adjust_lu_info(struct limic_user *lu, int old_len)
{
    unsigned long va = lu->va + lu->length;
    int pgcount, len = old_len - lu->length;
    int pagesize = getpagesize();

    MPIDI_STATE_DECL(MPID_STATE_ADJUST_LU_INFO);
    MPIDI_FUNC_ENTER(MPID_STATE_ADJUST_LU_INFO);

    pgcount = (va + len + pagesize - 1)/pagesize - va/pagesize;
    MPIU_Assert(pgcount);

    lu->va = va;
    lu->nr_pages = pgcount;
    lu->offset = va & (pagesize-1);
    lu->length = len;

    MPIDI_FUNC_EXIT(MPID_STATE_ADJUST_LU_INFO);
}

#endif /* _SMP_LIMIC_ */

#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_send_comp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_SMP_send_comp(void *header,
                                    MPIDI_VC_t* vc, MPIDI_msg_sz_t nb, smp_dma_flag_t dma_flag,
                                    smp_fallback_flag_t fallback)
{
    MPIDI_CH3_Pkt_comp_t pkt;
    int pkt_sz = sizeof(MPIDI_CH3_Pkt_comp_t);
    volatile void *ptr_head, *ptr, *ptr_flag;
    MPID_Request *creq = NULL;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_SEND_COMP);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_SEND_COMP);

#if defined(_SMP_LIMIC_)
    if (dma_flag == SMP_DMA_LIMIC){
        pkt.type = MPIDI_CH3_PKT_SMP_DMA_COMP;
        struct limic_header *l_header = (struct limic_header *)header; 
        pkt.send_req_id = (MPI_Request *)l_header->send_req_id;
        pkt.fallback = fallback;
    }
#endif 

#if defined(_SMP_CMA_)
    if (dma_flag == SMP_DMA_CMA){
        pkt.type = MPIDI_CH3_PKT_SMP_DMA_COMP;
        struct cma_header *c_header = (struct cma_header *)header;
        pkt.send_req_id = (MPI_Request *)c_header->csend_req_id;
        pkt.fallback = fallback;
    }
#endif 

    pkt.nb = nb;
    
    /*make sure the complete message not sent between other unfinished message */
    if (MPIDI_CH3I_SMP_SendQ_head(vc)) {
        creq = create_request(&pkt, pkt_sz, 0);
        int inuse = 0;
        MPIU_Object_release_ref_always(creq, &inuse);
        MPIDI_CH3I_SMP_SendQ_enqueue(vc, creq);
        return;
    }

    ptr_flag = (volatile void *) ((g_smpi_shmem->pool) +
        s_header_ptr_s[vc->smp.local_nodes]);

        /* check if avail is less than data size */
    if(!smpi_check_avail(vc->smp.local_nodes, pkt_sz, (volatile void **)&ptr_flag, ONE_FREE))
    {
        /* queue the message */
        creq = create_request(&pkt, pkt_sz, 0);
        int inuse = 0;
        MPIU_Object_release_ref_always(creq, &inuse);
        MPIDI_CH3I_SMP_SendQ_enqueue(vc, creq);
        return;
    }

    ptr_head = (volatile void *) ((unsigned long) ptr_flag + sizeof(int));
    ptr = (volatile void *) ((unsigned long) ptr_flag + sizeof(int)*2);
    MPIU_Memcpy((void *)ptr, (const void *)&pkt, pkt_sz);

    ptr = (volatile void *) ((unsigned long) ptr + pkt_sz);
    smpi_complete_send(vc->smp.local_nodes, pkt_sz, pkt_sz, ptr, ptr_head, ptr_flag);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_SEND_COMP);
    return;
}

#endif /* _SMP_LIMIC_ */

#if defined(_SMP_CMA_)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_do_cma_get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static pid_t MPIDI_CH3I_SMP_get_pid(MPIDI_VC_t *vc)
{
    pid_t pid;
    MPIU_Assert(IS_VC_SMP(vc));
    pid = g_smpi_shmem->pid[vc->smp.local_nodes];
    return pid;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_do_cma_get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_do_cma_get(MPIDI_VC_t *vc, const void *src, void *dst, ssize_t len)
{
    pid_t pid;
    ssize_t nbytes, total = 0;
    struct iovec local, remote;
    int mpi_errno = MPI_SUCCESS;

    local.iov_base = dst;
    local.iov_len = len;
    remote.iov_base = (void *)src;
    remote.iov_len = len;

    MPIDI_STATE_DECL(MPIDI_CH3I_SMP_DO_CMA_GET);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_SMP_DO_CMA_GET);

    pid = MPIDI_CH3I_SMP_get_pid(vc);
    PRINT_DEBUG(DEBUG_RNDV_verbose > 0,
            "CMA read from rank: %d, pid: %d, src: %p, dst: %p, len: %ld\n",
            vc->pg_rank, pid, src, dst, len);

    do {
        total += nbytes = process_vm_readv(pid, &local, 1, &remote, 1, 0);
        PRINT_DEBUG(DEBUG_RNDV_verbose > 2,
                "CMA read from rank: %d, nbytes: %ld, len: %ld, remaining: %ld\n",
                vc->pg_rank, nbytes, len, len - total);

        if (nbytes < 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "process_vm_readv fail");
        }

        local.iov_base  += nbytes;
        local.iov_len   -= nbytes;
        remote.iov_base += nbytes;
        remote.iov_len  -= nbytes;
    } while (total < len);

    MPIDI_FUNC_EXIT(MPIDI_CH3I_SMP_DO_CMA_GET);
fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SMP_do_cma_get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_SMP_do_cma_put(MPIDI_VC_t *vc, const void *src, void *dst, ssize_t len)
{
    pid_t pid;
    ssize_t nbytes, total = 0;
    struct iovec local, remote;
    int mpi_errno = MPI_SUCCESS;

    local.iov_base = (void *)src;
    local.iov_len = len;
    remote.iov_base = dst;
    remote.iov_len = len;

    MPIDI_STATE_DECL(MPIDI_CH3I_SMP_DO_CMA_PUT);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_SMP_DO_CMA_PUT);

    pid = MPIDI_CH3I_SMP_get_pid(vc);
    PRINT_DEBUG(DEBUG_RNDV_verbose > 0,
            "CMA write to rank: %d, pid: %d, src: %p, dst: %p, len: %ld\n",
            vc->pg_rank, pid, src, dst, len);

    do {
        total += nbytes = process_vm_writev(pid, &local, 1, &remote, 1, 0);
        PRINT_DEBUG(DEBUG_RNDV_verbose > 2,
                "CMA write to rank: %d, nbytes: %ld, len: %ld, remaining: %ld\n",
                vc->pg_rank, nbytes, len, len - total);

        if (nbytes < 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "process_vm_writev fail");
        }

        local.iov_base  += nbytes;
        local.iov_len   -= nbytes;
        remote.iov_base += nbytes;
        remote.iov_len  -= nbytes;
    } while (total < len);

    MPIDI_FUNC_EXIT(MPIDI_CH3I_SMP_DO_CMA_PUT);
fn_fail:
    return mpi_errno;
}

#endif /* _SMP_CMA_ */

/* vi:set sw=4 */
