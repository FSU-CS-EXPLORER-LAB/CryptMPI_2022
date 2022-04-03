/*
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

#include "mpichconf.h"
#include <mpimem.h>

#include "mem_hooks.h"
#include <infiniband/verbs.h>
#include "upmi.h"
#include "rdma_impl.h"
#include "vbuf.h"
#include "dreg.h"
#include "mpiutil.h"
#include <errno.h>
#include <string.h>
#include <debug_utils.h>

/* vbuf pool info */
vbuf_pool_t *rdma_vbuf_pools = NULL;
vbuf_pool_t mv2_srq_repost_pool;
volatile int rdma_num_vbuf_pools = 0;


/* head of list of allocated vbuf regions */
static vbuf_region *vbuf_region_head = NULL;

/*
 * cache the nic handle, and ptag the first time a region is
 * allocated (at init time) for later additional vbuf allocations
 */
static struct ibv_pd *ptag_save[MAX_NUM_HCAS];

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
vbuf *ud_free_vbuf_head = NULL;
int ud_vbuf_n_allocated = 0;
long ud_num_free_vbuf = 0;
long ud_num_vbuf_get = 0;
long ud_num_vbuf_freed = 0;
#endif

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

static pthread_spinlock_t vbuf_lock;

extern int g_atomics_support;

#if defined(DEBUG)
void dump_vbuf(char* msg, vbuf* v)
{
    int i = 0;
    int len = 100;
    MPIDI_CH3I_MRAILI_Pkt_comm_header* header = v->pheader;
    DEBUG_PRINT("%s: dump of vbuf %p, type = %d\n", msg, v, header->type);
    len = 100;

    for (; i < len; ++i)
    {
        if (0 == i % 16)
        {
            DEBUG_PRINT("\n  ");
        }

        DEBUG_PRINT("%2x  ", (unsigned int) v->buffer[i]);
    }

    DEBUG_PRINT("\n");
    DEBUG_PRINT("  END OF VBUF DUMP\n");
}
#endif /* defined(DEBUG) */

void mv2_print_vbuf_usage_usage()
{
    int i = 0;
    unsigned long int tot_mem = 0;
    unsigned long int size = 0;
    int vbuf_n_allocated = 0;

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    tot_mem += (ud_vbuf_n_allocated * (rdma_default_ud_mtu + sizeof(struct vbuf))); 
#endif
    for(i = 0; i < rdma_num_vbuf_pools; i++) {
        size = rdma_vbuf_pools[i].num_allocated *
                  (rdma_vbuf_pools[i].buf_size + sizeof(struct vbuf));
        vbuf_n_allocated += rdma_vbuf_pools[i].num_allocated;
        tot_mem += size;

        PRINT_INFO(DEBUG_MEM_verbose > 0, "[Pool: %d, Size:%lu] num_bufs:%u, tot_mem:%ld kB,"
                    " num_get = %ld, num_freed = %ld\n", i,
                    (long unsigned int)rdma_vbuf_pools[i].buf_size, rdma_vbuf_pools[i].num_allocated,
                    (size/1024),
                    rdma_vbuf_pools[i].num_get, rdma_vbuf_pools[i].num_freed); 
    }

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    PRINT_INFO(DEBUG_MEM_verbose, "RC VBUFs:%d  UD VBUFs:%d TOT MEM:%lu kB\n",
                    vbuf_n_allocated, ud_vbuf_n_allocated, (tot_mem / 1024));
#else
    PRINT_INFO(DEBUG_MEM_verbose, "RC VBUFs:%d, TOT MEM:%lu kB\n",
                    vbuf_n_allocated, (tot_mem / 1024));
#endif
    PRINT_INFO(DEBUG_MEM_verbose, "Reposted VBUF stats: num_freed = %ld, num_get = %ld\n",
                mv2_srq_repost_pool.num_freed, mv2_srq_repost_pool.num_get);
}

int init_vbuf_lock(void)
{
    int mpi_errno = MPI_SUCCESS;

    if (pthread_spin_init(&vbuf_lock, 0))
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno,
            MPIR_ERR_FATAL,
            "init_vbuf_lock",
            __LINE__,
            MPI_ERR_OTHER,
            "**fail",
            "%s: %s",
            "pthread_spin_init",
            strerror(errno));
    }

    return mpi_errno;
}

void deallocate_vbufs(int hca_num)
{
    vbuf_region *r = vbuf_region_head;

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
        || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_lock(&vbuf_lock);
    }

    while (r)
    {
        if (r->mem_handle[hca_num] != NULL
            && ibv_dereg_mr(r->mem_handle[hca_num]))
        {
            ibv_error_abort(IBV_RETURN_ERR, "could not deregister MR");
        }

        DEBUG_PRINT("deregister vbufs\n");
        r = r->next;
    }

    int i;
    for(i = 0; i < rdma_num_vbuf_pools; i++) {
        r = rdma_vbuf_pools[i].region_head;
        while (r) {
            if (r->mem_handle[hca_num] != NULL
                    && ibv_dereg_mr(r->mem_handle[hca_num])) {
                ibv_error_abort(IBV_RETURN_ERR, "could not deregister MR");
            }

            DEBUG_PRINT("deregister vbufs\n");
            r = r->next;
        } 
    }

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
        || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
         pthread_spin_unlock(&vbuf_lock);
    }
}

void deallocate_vbuf_region(void)
{
    vbuf_region *curr = vbuf_region_head;
    vbuf_region *next = NULL;

    while (curr) {
        next = curr->next;
        MPIU_Memalign_Free(curr->malloc_start);

        if (rdma_enable_hugepage && curr->shmid >= 0) {
            MPIU_shmdt(curr->malloc_buf_start);
        } else {
            MPIU_Memalign_Free(curr->malloc_buf_start);
        }

        MPIU_Memalign_Free(curr);
        curr = next;
    }

    int i;
#ifdef _ENABLE_CUDA_
    static CUcontext active_context = NULL;

    /*check if three is an active context, or else skip cuda_unregister,
     *the application might have called destroyed the context before finalize
     */
    if (rdma_enable_cuda) {
        CU_CHECK(cuCtxGetCurrent(&active_context));
    }
#endif

    for(i = 0; i < rdma_num_vbuf_pools; i++) {
        curr = rdma_vbuf_pools[i].region_head;
        while (curr) {
            next = curr->next;
            MPIU_Memalign_Free(curr->malloc_start);
  
#ifdef _ENABLE_CUDA_
            if (active_context != NULL && 
                (rdma_vbuf_pools[i].index == MV2_CUDA_VBUF_POOL_OFFSET ||
                (rdma_eager_cudahost_reg && 
                         rdma_vbuf_pools[i].index <= MV2_RECV_VBUF_POOL_OFFSET))) {
                ibv_cuda_unregister(curr->malloc_buf_start);
            }
#endif
                
            if (rdma_enable_hugepage && curr->shmid >= 0) {
                MPIU_shmdt(curr->malloc_buf_start);
            } else {
                MPIU_Memalign_Free(curr->malloc_buf_start);
            }
            MPIU_Memalign_Free(curr);
            curr = next;
        }
    }
    MPIU_Memalign_Free(rdma_vbuf_pools);
    rdma_num_vbuf_pools = 0;

    return;
}

int alloc_hugepage_region (int *shmid, void **buffer, int *nvbufs, int buf_size)
{
    int ret = 0;
    size_t size = *nvbufs * buf_size;
    MRAILI_ALIGN_LEN(size, HUGEPAGE_ALIGN);

    /* create hugepage shared region */
    *shmid = shmget(IPC_PRIVATE, size, 
                        SHM_HUGETLB | IPC_CREAT | SHM_R | SHM_W);
    if (*shmid < 0) {
        goto fn_fail;
    }

    /* attach shared memory */
    *buffer = (void *) shmat(*shmid, SHMAT_ADDR, SHMAT_FLAGS);
    if (*buffer == (void *) -1) {
        goto fn_fail;
    }
    
    /* Mark shmem for removal */
    if (shmctl(*shmid, IPC_RMID, 0) != 0) {
        fprintf(stderr, "Failed to mark shm for removal\n");
    }
    
    /* Find max no.of vbufs can fit in allocated buffer */
    *nvbufs = size / buf_size;
     
fn_exit:
    return ret;
fn_fail:
    ret = -1;
    if (rdma_enable_hugepage >= 2) {
        fprintf(stderr,"[%d] Failed to allocate buffer from huge pages. "
                       "fallback to regular pages. requested buf size:%lu\n",
                        MPIDI_Process.my_pg_rank, size);
    }
    goto fn_exit;
}    

static inline int reregister_vbuf_pool(vbuf_pool_t *rdma_vbuf_pool)
{
    int i = 0;
    int nvbufs = 0;
    int buf_size = 0;
    void *vbuf_buffer = NULL;
    struct vbuf_region *region = NULL;

    PRINT_DEBUG(DEBUG_CR_verbose > 0,"index = %u; initial_count = %u;"
                "incr_count = %u; buf_size = %u; num_allocated = %u;"
                "num_free = %u; max_num_buf = %u; num_get = %ld;"
                "num_freed = %ld; free_head = %p\n",
                rdma_vbuf_pool->index, rdma_vbuf_pool->initial_count,
                rdma_vbuf_pool->incr_count, rdma_vbuf_pool->buf_size,
                rdma_vbuf_pool->num_allocated, rdma_vbuf_pool->num_free,
                rdma_vbuf_pool->max_num_buf, rdma_vbuf_pool->num_get,
                rdma_vbuf_pool->num_freed, rdma_vbuf_pool->free_head);

    region      = rdma_vbuf_pool->region_head;
    buf_size    = rdma_vbuf_pool->buf_size;
    while (region) {
        nvbufs = region->count;
        vbuf_buffer = region->malloc_buf_start;
        PRINT_DEBUG(DEBUG_CR_verbose > 0,"region_head = %p\n", region);
    
#ifdef _ENABLE_CUDA_
        if (rdma_enable_cuda && (rdma_vbuf_pool->index == MV2_CUDA_VBUF_POOL_OFFSET ||
            (rdma_eager_cudahost_reg && rdma_vbuf_pool->index <= MV2_RECV_VBUF_POOL_OFFSET))) {
            ibv_cuda_register(vbuf_buffer, nvbufs * buf_size);
        }
#endif

        // for posix_memalign
        for (i = 0; i < rdma_num_hcas; ++i) {
            if (g_atomics_support) {
                region->mem_handle[i] = ibv_reg_mr(ptag_save[i], vbuf_buffer,
                        nvbufs * buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_ATOMIC );
            } else {
                region->mem_handle[i] = ibv_reg_mr(ptag_save[i], vbuf_buffer,
                        nvbufs * buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            }
            if (!region->mem_handle[i]) {
                fprintf(stderr, "[%s %d] Cannot register vbuf region\n", __FILE__, __LINE__);
                return -1;	
            }
        }
        region = region->next;
    }

    return 0;
}

int allocate_vbuf_pool(vbuf_pool_t *rdma_vbuf_pool, int nvbufs)
{

    int mpi_errno = MPI_SUCCESS;
    int i = 0;
    struct vbuf_region *region = NULL;
    void *vbuf_struct = NULL;
    void *vbuf_buffer = NULL;
    vbuf *cur = NULL;
    int alignment_vbuf = 64;
    int result = 0;
    int alignment_dma = getpagesize();
    int buf_size;

    if (nvbufs <= 0) {
        return 0;
    }

    DEBUG_PRINT("Allocating a new vbuf region.\n");

    if (rdma_vbuf_pool->free_head != NULL) {
        ibv_error_abort(GEN_ASSERT_ERR, "vbuf_head = NULL");
    }

    buf_size = rdma_vbuf_pool->buf_size;

    /* are we limiting vbuf allocation?  If so, make sure
     * we dont alloc more than allowed
     */
    if (rdma_vbuf_pool->max_num_buf > 0) {
        nvbufs = MIN(nvbufs, rdma_vbuf_pool->max_num_buf  
                - rdma_vbuf_pool->num_allocated);
        if (nvbufs <= 0) {
            DEBUG_PRINT("max vbuf pool size reached for size : %d \n", buf_size); 
            mpi_errno=-1; 
            return mpi_errno;
        }
    }

    /* Allocate vbuf region structure */
    result = MPIU_Memalign((void*)&region, alignment_dma, sizeof(struct vbuf_region));
    if ((result != 0) || (NULL == region)) {
        ibv_error_abort(GEN_EXIT_ERR, "Unable to malloc a new struct vbuf_region");
    }

    result = MPIU_Memalign((void*)&vbuf_buffer, alignment_dma, nvbufs * buf_size);
    if ((result != 0) || (NULL == vbuf_buffer)) {
        ibv_error_abort(GEN_EXIT_ERR, "unable to malloc vbufs DMA buffer");
    }

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && (rdma_vbuf_pool->index == MV2_CUDA_VBUF_POOL_OFFSET ||
        (rdma_eager_cudahost_reg && rdma_vbuf_pool->index <= MV2_RECV_VBUF_POOL_OFFSET))) {
        ibv_cuda_register(vbuf_buffer, nvbufs * buf_size);
    }
#endif

    if (MPIU_Memalign((void**) &vbuf_struct, alignment_vbuf, 
                nvbufs * sizeof(struct vbuf))) {
        ibv_error_abort(GEN_EXIT_ERR, "Unable to allocate vbuf structs");
    }

    MPIU_Memset(vbuf_struct, 0, nvbufs * sizeof(struct vbuf));
    MPIU_Memset(vbuf_buffer, 0, nvbufs * buf_size);

    rdma_vbuf_pool->num_free += nvbufs;
    rdma_vbuf_pool->num_allocated += nvbufs;
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_vbuf_allocated, nvbufs);
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_vbuf_available, nvbufs);

    region->malloc_start = vbuf_struct;
    region->malloc_buf_start = vbuf_buffer;
    region->malloc_end = (void *) ((char *) vbuf_struct 
            + nvbufs * sizeof(struct vbuf));
    region->malloc_buf_end = (void *) ((char *) vbuf_buffer 
            + nvbufs * buf_size);

    region->count = nvbufs;
    rdma_vbuf_pool->free_head = vbuf_struct;
    region->vbuf_head = vbuf_struct;
    region->shmid = -1;

    DEBUG_PRINT(
            "VBUF REGION ALLOCATION SZ %d TOT %d FREE %ld NF %ld NG %ld\n",
            nvbufs,
            rdma_vbuf_pool->num_allocated,
            rdma_vbuf_pool->num_free,
            rdma_vbuf_pool->num_freed,
            rdma_vbuf_pool->num_get);

    // for posix_memalign
    for (i = 0; i < rdma_num_hcas; ++i) {
        if (g_atomics_support) {
            region->mem_handle[i] = ibv_reg_mr(ptag_save[i], vbuf_buffer,
                    nvbufs * buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_ATOMIC );
        } else {
            region->mem_handle[i] = ibv_reg_mr(ptag_save[i], vbuf_buffer,
                    nvbufs * buf_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        }
        if (!region->mem_handle[i]) {
            fprintf(stderr, "[%s %d] Cannot register vbuf region\n", __FILE__, __LINE__);
            return -1;	
        }
    }

    /* init the free list */
    for (i = 0; i < nvbufs; ++i)
    {
        cur = rdma_vbuf_pool->free_head + i;
        cur->desc.next = rdma_vbuf_pool->free_head + i + 1;
        if (i == (nvbufs -1)) cur->desc.next = NULL;
        cur->region = region;
        cur->pool_index = rdma_vbuf_pool ;  
        cur->head_flag = (VBUF_FLAG_TYPE *) ((char *)vbuf_buffer
                + (i + 1) * buf_size - sizeof(*cur->head_flag));
        cur->buffer = (unsigned char *) ((char *)vbuf_buffer
                + i * buf_size);
        cur->eager = 0;
        cur->content_size = 0;
        cur->coalesce = 0;
    }


    /* thread region list */
    region->next = rdma_vbuf_pool->region_head;
    rdma_vbuf_pool->region_head = region;
    region->pool_index = rdma_vbuf_pool;

    return 0;
}

int allocate_vbuf_pools(struct ibv_pd* ptag[])
{
    int mpi_errno = MPI_SUCCESS;
    int i;

    for(i = 0; i < rdma_num_vbuf_pools; i++) {
        mpi_errno = allocate_vbuf_pool(&rdma_vbuf_pools[i],
                                        rdma_vbuf_pools[i].initial_count);
        if(mpi_errno) {
            return mpi_errno;
        }
    }
    return mpi_errno;
}

#ifdef _ENABLE_CUDA_
void register_cuda_vbuf_regions()
{
    int i; 
    vbuf_pool_t *rdma_vbuf_pool; 
    struct vbuf_region *region = NULL;

    MPIU_Assert (rdma_cuda_dynamic_init == 1);
    
    for (i=0; i<rdma_num_vbuf_pools; i++) {
        rdma_vbuf_pool = &rdma_vbuf_pools[i];        
        if (rdma_vbuf_pool->index == MV2_CUDA_VBUF_POOL_OFFSET ||
           (rdma_eager_cudahost_reg && rdma_vbuf_pool->index <= MV2_RECV_VBUF_POOL_OFFSET)) {
            region = rdma_vbuf_pool->region_head;
            while (region != NULL) { 
                ibv_cuda_register(region->malloc_buf_start, 
                        (uint64_t)region->malloc_buf_end - (uint64_t)region->malloc_buf_start);
                region = region->next;
            }
        }
    }
}
#endif /* #ifdef _ENABLE_CUDA_*/

static inline int size_to_offset(size_t message_size)
{
    int i = 0;

    for (i = MV2_SMALL_DATA_VBUF_POOL_OFFSET; i < rdma_num_vbuf_pools; i++) {
        if (likely(message_size <= rdma_vbuf_pools[i].buf_size)) {
            return i;
        }
    }
    /* No hits, return the largest size */
    return rdma_num_vbuf_pools-1;
}

vbuf* get_vbuf(size_t message_size)
{
    vbuf* v = NULL;
    int offset = size_to_offset(message_size);

    vbuf_pool_t *rdma_vbuf_pool = &rdma_vbuf_pools[offset];

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_lock(&vbuf_lock);
    }

    if (unlikely(NULL == rdma_vbuf_pool->free_head)) {
        if (allocate_vbuf_pool(rdma_vbuf_pool, rdma_vbuf_pool->incr_count) != 0) {
            ibv_va_error_abort(GEN_EXIT_ERR,"vbuf pool allocation failed");
        }
    }

    MV2_GET_AND_INIT_RC_VBUF(v, rdma_vbuf_pool);
    
#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
        
    {
        pthread_spin_unlock(&vbuf_lock);
    }

    return(v);
}

vbuf* get_vbuf_by_offset(int offset)
{
    vbuf* v = NULL;

    MPIU_Assert(offset < rdma_num_vbuf_pools);

    vbuf_pool_t *rdma_vbuf_pool = &rdma_vbuf_pools[offset];

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_lock(&vbuf_lock);
    }

    if (unlikely(NULL == rdma_vbuf_pool->free_head)) {
        if (allocate_vbuf_pool(rdma_vbuf_pool, rdma_vbuf_pool->incr_count) != 0) {
            ibv_va_error_abort(GEN_EXIT_ERR,"vbuf pool allocation failed");
        }
    }

    MV2_GET_AND_INIT_RC_VBUF(v, rdma_vbuf_pool);

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
        
    {
        pthread_spin_unlock(&vbuf_lock);
    }

    return(v);
}

void release_vbuf(vbuf* v)
{
    vbuf_pool_t *rdma_vbuf_pool = v->pool_index;

    /* note this correctly handles appending to empty free list */
#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_lock(&vbuf_lock);
    }

    MPIU_Assert(v->padding == NORMAL_VBUF_FLAG ||
                v->padding == RPUT_VBUF_FLAG ||
                v->padding == RGET_VBUF_FLAG ||
                v->padding == RDMA_ONE_SIDED ||
                v->padding == COLL_VBUF_FLAG);

#if defined( _ENABLE_UD_) || defined(_MCST_SUPPORT_)
    if(v->transport == IB_TRANSPORT_UD) {
        MV2_RELEASE_UD_VBUF(v);
    } else 
#endif /* _ENABLE_UD_ */
    {
        DEBUG_PRINT("release_vbuf: releasing %p padding %d\n", v, rdma_vbuf_pool->free_head, v->padding);
        MV2_RELEASE_RC_VBUF(v, rdma_vbuf_pool);
    }

    /* Reset VBUF parameters */
    MV2_RESET_VBUF(v);    

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_unlock(&vbuf_lock);
    }
}

/* this function is only called by the init routines.
 * Cache the nic handle and ptag for later vbuf_region allocations.
 */
int allocate_vbufs(struct ibv_pd* ptag[])
{
    int i = 0;

    for (; i < rdma_num_hcas; ++i)
    {
        ptag_save[i] = ptag[i];
    }

    if (allocate_vbuf_pools(ptag) !=0 ) {
        ibv_va_error_abort(GEN_EXIT_ERR,
                "VBUF CUDA region allocation failed.\n")
    }
    
    return 0;
}

void MRAILI_Release_vbuf(vbuf* v)
{
#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    /* This message might be in progress. Wait for ib send completion 
     * to release this buffer to avoid to reusing buffer
     */
    if (v->flags & UD_VBUF_MCAST_MSG) {
        MPIU_Assert(v->pending_send_polls > 0);
        v->pending_send_polls--;
        if(v->transport== IB_TRANSPORT_UD 
           && (v->flags & UD_VBUF_SEND_INPROGRESS || v->pending_send_polls > 0)) {
            if (v->pending_send_polls == 0) {
                v->flags |= UD_VBUF_FREE_PENIDING;
            }
            return;
        }
    } else {
        if(v->transport== IB_TRANSPORT_UD 
                && (v->flags & UD_VBUF_SEND_INPROGRESS)) {
            v->flags |= UD_VBUF_FREE_PENIDING;
            return;
        }
    }
#endif
 
#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    /* We don't use pools for UD */
    if (likely(v->pool_index && ((vbuf_pool_t*)v->pool_index)->index != MV2_RECV_VBUF_POOL_OFFSET))
#else
    if (likely(((vbuf_pool_t*)v->pool_index)->index != MV2_RECV_VBUF_POOL_OFFSET))
#endif
    {
        MV2_RELEASE_VBUF_NO_LOCK(v);
    } else {
#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
        /* We don't use pools for UD */
        /* These vbufs will be the ones ear marked for SRQ */
        if (likely(v->pool_index && mv2_MPIDI_CH3I_RDMA_Process.has_srq))
#else
        if (likely(mv2_MPIDI_CH3I_RDMA_Process.has_srq))
#endif
        {
            if (likely(mv2_srq_repost_pool.num_free < rdma_credit_preserve)) {
                MV2_RELEASE_RC_VBUF(v, &mv2_srq_repost_pool);
            } else {
                MV2_REPOST_VBUF_TO_SRQ(v, &mv2_srq_repost_pool);
            }
        } else {
            release_vbuf(v);
        }
    }
}

#if defined( _ENABLE_UD_) || defined(_MCST_SUPPORT_)
int allocate_ud_vbuf_region(int nvbufs)
{

    struct vbuf_region *reg = NULL;
    void *mem = NULL;
    int i = 0;
    vbuf *cur = NULL;
    void *vbuf_dma_buffer = NULL;
    int alignment_vbuf = 64;
    int alignment_dma = getpagesize();
    int result = 0;

    PRINT_DEBUG(DEBUG_UD_verbose>0,"Allocating a UD buf region.\n");

    if (ud_free_vbuf_head != NULL)
    {
        ibv_error_abort(GEN_ASSERT_ERR, "ud_free_vbuf_head = NULL");
    }

    result = MPIU_Memalign((void*)&reg, alignment_dma, sizeof(struct vbuf_region));
    if ((result != 0) || (NULL == reg)) {
        ibv_error_abort(GEN_EXIT_ERR, "Unable to malloc a new struct vbuf_region");
    }
    
    if (rdma_enable_hugepage) {
        result = alloc_hugepage_region(&reg->shmid, &vbuf_dma_buffer, &nvbufs,
                                        rdma_default_ud_mtu);
    }

    /* do posix_memalign if enable hugepage disabled or failed */
    if (rdma_enable_hugepage == 0 || result != 0 )  
    {
        reg->shmid = -1;
        result = MPIU_Memalign(&vbuf_dma_buffer, 
            alignment_dma, nvbufs * rdma_default_ud_mtu);
    }

    if ((result!=0) || (NULL == vbuf_dma_buffer))
    {
        ibv_error_abort(GEN_EXIT_ERR, "unable to malloc vbufs DMA buffer");
    }
    
    if (MPIU_Memalign(
                (void**) &mem,
                alignment_vbuf,
                nvbufs * sizeof(vbuf)))
    {
        fprintf(stderr, "[%s %d] Cannot allocate vbuf region\n", 
                __FILE__, __LINE__);
        return -1;
    }

    MPIU_Memset(mem, 0, nvbufs * sizeof(vbuf));
    MPIU_Memset(vbuf_dma_buffer, 0, nvbufs * rdma_default_ud_mtu);

    ud_vbuf_n_allocated += nvbufs;
    ud_num_free_vbuf += nvbufs;
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_ud_vbuf_allocated, nvbufs);
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_ud_vbuf_available, nvbufs);
    reg->malloc_start = mem;
    reg->malloc_buf_start = vbuf_dma_buffer;
    reg->malloc_end = (void *) ((char *) mem + nvbufs * sizeof(vbuf));
    reg->malloc_buf_end = (void *) ((char *) vbuf_dma_buffer + 
            nvbufs * rdma_default_ud_mtu);

    reg->count = nvbufs;
    ud_free_vbuf_head = mem;
    reg->vbuf_head = ud_free_vbuf_head;

    PRINT_DEBUG(DEBUG_UD_verbose>0,
            "VBUF REGION ALLOCATION SZ %d TOT %d FREE %ld NF %ld NG %ld\n",
            rdma_default_ud_mtu,
            ud_vbuf_n_allocated,
            ud_num_free_vbuf,
            ud_num_vbuf_freed,
            ud_num_vbuf_get);

    /* region should be registered for both of the hca */
    for (; i < rdma_num_hcas; ++i)
    {
        reg->mem_handle[i] = ibv_reg_mr(
                ptag_save[i],
                vbuf_dma_buffer,
                nvbufs * rdma_default_ud_mtu,
                IBV_ACCESS_LOCAL_WRITE );

        if (!reg->mem_handle[i])
        {
            fprintf(stderr, "[%s %d] Cannot register vbuf region\n", 
                    __FILE__, __LINE__);
            return -1;
        }
    }

    /* init the free list */
    for (i = 0; i < nvbufs; ++i)
    {
        cur = ud_free_vbuf_head + i;
        cur->desc.next = ud_free_vbuf_head + i + 1;
        if (i == (nvbufs -1)) cur->desc.next = NULL;
        cur->region = reg;
        cur->head_flag = (VBUF_FLAG_TYPE *) ((char *)vbuf_dma_buffer
                + (i + 1) * rdma_default_ud_mtu - sizeof * cur->head_flag);
        cur->buffer = (unsigned char *) ((char *)vbuf_dma_buffer
                + i * rdma_default_ud_mtu);

        cur->eager = 0;
        cur->content_size = 0;
        cur->coalesce = 0;
    }

    /* thread region list */
    reg->next = vbuf_region_head;
    vbuf_region_head = reg;

    return 0;
}

int allocate_ud_vbufs(int nvbufs)
{
    return allocate_ud_vbuf_region(nvbufs);
}

vbuf* get_ud_vbuf(void)
{
    vbuf* v = NULL;

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_lock(&vbuf_lock);
    }

    if (NULL == ud_free_vbuf_head)
    {
        if(allocate_ud_vbuf_region(rdma_vbuf_secondary_pool_size) != 0) {
            ibv_va_error_abort(GEN_EXIT_ERR,
                    "UD VBUF region allocation failed.\n");
        }
    }


    v = ud_free_vbuf_head;
    --ud_num_free_vbuf;
    MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_ud_vbuf_available, 1);
    ++ud_num_vbuf_get;

    /* this correctly handles removing from single entry free list */
    ud_free_vbuf_head = ud_free_vbuf_head->desc.next;

    /* need to change this to RPUT_VBUF_FLAG later
     * if we are doing rput */
    v->padding = NORMAL_VBUF_FLAG;
    v->pheader = (void *)v->buffer;
    v->transport = IB_TRANSPORT_UD;
    v->retry_count = 0;
    v->flags = 0;
    v->pending_send_polls = 0;

    /* this is probably not the right place to initialize shandle to NULL.
     * Do it here for now because it will make sure it is always initialized.
     * Otherwise we would need to very carefully add the initialization in
     * a dozen other places, and probably miss one.
     */
    v->sreq = NULL;
    v->coalesce = 0;
    v->content_size = 0;
    v->eager = 0;

    v->finish_count = 0;
    v->orig_vbuf = NULL;
    v->displacement = 0;
    v->pool_index = NULL;

    /*This is used for RMA put/get*/
    v->target_rank = -1;
    /* Decide which transport need to assign here */

#if !defined(CKPT)
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
#if defined(RDMA_CM)
            || mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
            || MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)
#endif /* !defined(CKPT) */
    {
        pthread_spin_unlock(&vbuf_lock);
    }

    return(v);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_ud_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_ud_recv(vbuf* v, unsigned long len, int hca_num)
{
    MPIU_Assert(v != NULL);

    v->desc.u.rr.next = NULL;
    v->desc.u.rr.wr_id = (uintptr_t) v;
    v->desc.u.rr.num_sge = 1;
    v->desc.u.rr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = v->region->mem_handle[hca_num]->lkey;
    v->desc.sg_entry.addr = (uintptr_t)(v->buffer);
    v->padding = NORMAL_VBUF_FLAG;
    v->rail = hca_num;

}

#endif /* _ENABLE_UD_ || _MCST_SUPPORT_*/

void MRAILI_Release_recv_rdma(vbuf* v)
{
    vbuf *next_free = NULL;
    MPIDI_VC_t * c = (MPIDI_VC_t *)v->vc;
    int i;

    int next = c->mrail.rfp.p_RDMA_recv_tail + 1;

    if (next >= num_rdma_buffer)
    {
        next = 0;
    }

    next_free = &(c->mrail.rfp.RDMA_recv_buf[next]);
    v->padding = FREE_FLAG;
    *v->head_flag = 0;
    v->sreq = NULL;
    v->content_size = 0;

    if (v != next_free)
    {
        return;
    }

    /* search all free buffers */
    for (i = next; i != c->mrail.rfp.p_RDMA_recv;)
    {
        if (c->mrail.rfp.RDMA_recv_buf[i].padding == FREE_FLAG)
        {
            ++c->mrail.rfp.rdma_credit;

            if (++c->mrail.rfp.p_RDMA_recv_tail >= num_rdma_buffer)
            {
                c->mrail.rfp.p_RDMA_recv_tail = 0;
            }

            c->mrail.rfp.RDMA_recv_buf[i].padding = BUSY_FLAG;
            *c->mrail.rfp.RDMA_recv_buf[i].head_flag = 0;
        }
        else
        {
            break;
        }

        if (++i >= num_rdma_buffer)
        {
            i = 0;
        }
    }
}

#if defined(CKPT)

#undef FUNCNAME
#define FUNCNAME vbuf_reregister_all
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_reregister_all()
{
    int i = 0;
    vbuf_region *vr = vbuf_region_head;

    MPIDI_STATE_DECL(MPID_STATE_VBUF_REREGISTER_ALL);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_REREGISTER_ALL);

    for (i = 0; i < rdma_num_hcas; ++i)
    {
        ptag_save[i] = mv2_MPIDI_CH3I_RDMA_Process.ptag[i];
    }

    while (vr)
    {
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Reregistering vbuf_region_head\n");
        for (i = 0; i < rdma_num_hcas; ++i)
        {
            if (g_atomics_support) {
                vr->mem_handle[i] = ibv_reg_mr(
                    ptag_save[i],
                    vr->malloc_buf_start,
                    vr->count * rdma_vbuf_total_size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_ATOMIC);
            } else {
                vr->mem_handle[i] = ibv_reg_mr(
                    ptag_save[i],
                    vr->malloc_buf_start,
                    vr->count * rdma_vbuf_total_size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            }

            if (!vr->mem_handle[i])
            {
                ibv_error_abort(IBV_RETURN_ERR,"Cannot reregister vbuf region\n");
            }
        }

        vr = vr->next;
    }

    for(i = 0; i < rdma_num_vbuf_pools; i++) {
        PRINT_DEBUG(DEBUG_CR_verbose > 1,"Reregistering vbuf pool %d, count = %d\n", i, rdma_vbuf_pools[i].initial_count);
        int err = reregister_vbuf_pool(&rdma_vbuf_pools[i]);
                                        
        if (err) {
            ibv_error_abort(IBV_RETURN_ERR,"Cannot reregister vbuf region\n");
        }
    }

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_REREGISTER_ALL);
}
#endif /* defined(CKPT) */

/* vi:set sw=4 tw=80: */
