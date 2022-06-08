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

#define _GNU_SOURCE
#include <errno.h>
#include <string.h>

#include <infiniband/verbs.h>

#include "mpidimpl.h"
#include <mpimem.h>
#include "upmi.h"
#include "ib_vbuf.h"
#include "mpiutil.h"
#include "ib_process.h"
#include "ib_send.h"
/* add ib_vc.h for MRAILI_Release_recv_rdma */
#include "ib_vc.h"

/* head of list of allocated vbuf regions */
static vbuf_region *vbuf_region_head = NULL;
/*
 * free_vbuf_head is the head of the free list
 */
static vbuf *free_vbuf_head = NULL;

/*
 * cache the nic handle, and ptag the first time a region is
 * allocated (at init time) for later additional vbur allocations
 */
static struct ibv_pd *ptag_save[MAX_NUM_HCAS];

static int vbuf_n_allocated = 0;
static long num_free_vbuf = 0;
static long num_vbuf_get = 0;
static long num_vbuf_freed = 0;

static pthread_spinlock_t vbuf_lock;

#if defined(DEBUG)
void dump_vbuf(char* msg, vbuf* v)
{
    int i = 0;
    int len = 100;
    MPIDI_nem_ib_pkt_comm_header* header = v->iheader;
    printf("%s: dump of vbuf %p, type = %d\n", msg, v, header->type);
    len = 100;

    for (i=0; i < len; ++i)
    {
        if (0 == i % 16)
        {
        	printf("\n  ");
        }

        printf("%2x  ", (unsigned int) v->buffer[i]);
    }

    printf("\n");
    printf("  END OF VBUF DUMP\n");
}
#endif /* defined(DEBUG) */

int init_vbuf_lock()
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
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND*/
#endif /* !defined(CKPT) */
        )
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

#if !defined(CKPT)
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)*/
#endif /* !defined(CKPT) */
        )
    {
         pthread_spin_unlock(&vbuf_lock);
    }
}

void deallocate_vbuf_region()
{
    vbuf_region *curr = vbuf_region_head;
    vbuf_region *next = NULL;

    while (curr) {
        next = curr->next;
        MPIU_Memalign_Free(curr->malloc_start);
        MPIU_Memalign_Free(curr->malloc_buf_start);
        MPIU_Free(curr);
        curr = next;
    }
}

/**
 * Allocating a new vbuf region.
 */
static int allocate_vbuf_region(int nvbufs)
{
	DEBUG_PRINT("Allocating a new vbuf region.\n");

    struct vbuf_region *reg = NULL;
    void *vbuf_dma_buffer   = NULL;
    void *mem               = NULL;
    int res, i = 0;
    int alignment_vbuf = 64;
    int alignment_dma = getpagesize();
    vbuf *cur = NULL;

    if (free_vbuf_head != NULL)
    {
        ibv_error_abort(GEN_ASSERT_ERR, "free_vbuf_head = NULL");
    }

    /* are we limiting vbuf allocation?  If so, make sure
     * we dont alloc more than allowed
     */
    if (rdma_vbuf_max > 0)
    {
        nvbufs = MIN(nvbufs, rdma_vbuf_max - vbuf_n_allocated);

        if (nvbufs <= 0)
        {
            ibv_error_abort(GEN_EXIT_ERR, "VBUF alloc failure, limit exceeded");
        }
    }

    reg = (struct vbuf_region *) MPIU_Malloc (sizeof(struct vbuf_region));

    if (NULL == reg)
    {
        ibv_error_abort(GEN_EXIT_ERR, "Unable to malloc a new struct vbuf_region");
   }

    res = posix_memalign( (void**) &mem, alignment_vbuf, nvbufs * sizeof(vbuf) );
    if (res!=0)
    {
        fprintf(stderr, "[%s %d] Cannot allocate vbuf region, error: %d\n", __FILE__, __LINE__, res);
        return -1;
    }

/*
    vbuf_dma_buffer = (void *) memalign(alignment_dma,  nvbufs * rdma_vbuf_total_size);
    if (NULL == vbuf_dma_buffer)
    {
       ibv_error_abort(GEN_EXIT_ERR, "unable to malloc vbufs DMA buffer");
    }
*/
    res =  posix_memalign((void **)&vbuf_dma_buffer, alignment_dma,  nvbufs * rdma_vbuf_total_size);
    if (res!=0)
    {
       ibv_error_abort(GEN_EXIT_ERR, "unable to malloc vbufs DMA buffer");
    }

    memset(mem, 0, nvbufs * sizeof(vbuf));
    memset(vbuf_dma_buffer, 0, nvbufs * rdma_vbuf_total_size);

    vbuf_n_allocated += nvbufs;
    num_free_vbuf += nvbufs;
    reg->malloc_start = mem;
    reg->malloc_buf_start = vbuf_dma_buffer;
    reg->malloc_end = (void *) ((char *) mem + nvbufs * sizeof(vbuf));
    reg->malloc_buf_end = (void *) ((char *) vbuf_dma_buffer + nvbufs * rdma_vbuf_total_size);

    reg->count = nvbufs;
    free_vbuf_head = mem;
    reg->vbuf_head = free_vbuf_head;

    DEBUG_PRINT(
        "VBUF REGION ALLOCATION SZ %d TOT %d FREE %ld NF %ld NG %ld\n",
        nvbufs,
        vbuf_n_allocated,
        num_free_vbuf,
        num_vbuf_freed,
        num_vbuf_get);

    /* region should be registered for both of the hca */
    for (i=0; i < ib_hca_num_hcas; ++i)
    {
        reg->mem_handle[i] = ibv_reg_mr(
            ptag_save[i],
            vbuf_dma_buffer,
            nvbufs * rdma_vbuf_total_size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

        if (!reg->mem_handle[i])
        {
            fprintf(stderr, "[%s %d] Cannot register vbuf region\n", __FILE__, __LINE__);
            return -1;
        }
    }

    /* init the free list */
    for (i = 0; i < nvbufs - 1; ++i)
    {
        cur = free_vbuf_head + i;
        cur->desc.next = free_vbuf_head + i + 1;
        cur->region = reg;
        cur->head_flag = (VBUF_FLAG_TYPE *) ((char *)vbuf_dma_buffer
            + (i + 1) * rdma_vbuf_total_size
            - sizeof * cur->head_flag);
        cur->buffer = (unsigned char *) ((char *)vbuf_dma_buffer
            + i * rdma_vbuf_total_size);

        cur->eager = 0;
        cur->content_size = 0;
        cur->coalesce = 0;
    }

    /* last one needs to be set to NULL */
    cur = free_vbuf_head + nvbufs - 1;
    cur->desc.next = NULL;
    cur->region = reg;
    cur->head_flag = (VBUF_FLAG_TYPE *) ((char *)vbuf_dma_buffer
        + nvbufs * rdma_vbuf_total_size
        - sizeof * cur->head_flag);
    cur->buffer = (unsigned char *) ((char *)vbuf_dma_buffer
        + (nvbufs - 1) * rdma_vbuf_total_size);
    cur->eager = 0;
    cur->content_size = 0;
    cur->coalesce = 0;

    /* thread region list */
    reg->next = vbuf_region_head;
    vbuf_region_head = reg;

    return 0;
}

/**
 * this function is only called by the init routines.
 * Cache the nic handle and ptag for later vbuf_region allocations.
 */
int allocate_vbufs(int nvbufs)
{
    int i = 0;

    for (i=0; i < ib_hca_num_hcas; ++i)
    {
        ptag_save[i] = hca_list[i].ptag;
    }

    return allocate_vbuf_region(nvbufs);
}

vbuf* get_vbuf()
{
    vbuf* v = NULL;

#if !defined(CKPT)
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)*/
#endif /* !defined(CKPT) */
        )
    {
    	pthread_spin_lock(&vbuf_lock);
    }

    /*
     * It will often be possible for higher layers to recover
     * when no vbuf is available, but waiting for more descriptors
     * to complete. For now, just abort.
     */
    if (NULL == free_vbuf_head)
    {
        allocate_vbuf_region(rdma_vbuf_secondary_pool_size);

        if (NULL == free_vbuf_head)
        {
            ibv_va_error_abort(GEN_EXIT_ERR,
                    "No free vbufs. Pool size %d",
                       vbuf_n_allocated);
        }
    }

    v = free_vbuf_head;
    --num_free_vbuf;
    ++num_vbuf_get;

    /* this correctly handles removing from single entry free list */
    free_vbuf_head = free_vbuf_head->desc.next;

    /* need to change this to RPUT_VBUF_FLAG later
     * if we are doing rput */
    v->padding = NORMAL_VBUF_FLAG;
    v->iheader = (void *)v->buffer;
    v->pheader = (void *)v->buffer + IB_PKT_HEADER_LENGTH;

    /* this is probably not the right place to initialize shandle to NULL.
     * Do it here for now because it will make sure it is always initialized.
     * Otherwise we would need to very carefully add the initialization in
     * a dozen other places, and probably miss one.
     */
    v->sreq = NULL;
    v->coalesce = 0;
    v->content_size = IB_PKT_HEADER_LENGTH;
    v->eager = 0;

#if !defined(CKPT)
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND) */
#endif /* !defined(CKPT) */
        )
    {
        pthread_spin_unlock(&vbuf_lock);
    }

    return(v);
}

void MRAILI_Release_vbuf(vbuf* v)
{
    /* note this correctly handles appending to empty free list */
#if !defined(CKPT)
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)*/
#endif /* !defined(CKPT) */
        )
    {
        pthread_spin_lock(&vbuf_lock);
    }

    DEBUG_PRINT("release_vbuf: releasing %p previous head = %p, padding %d\n", v, free_vbuf_head, v->padding);

    MPIU_Assert(v != free_vbuf_head);
    v->desc.next = free_vbuf_head;

    if (v->padding != NORMAL_VBUF_FLAG
        && v->padding != RPUT_VBUF_FLAG
        && v->padding != RGET_VBUF_FLAG
        && v->padding != COLL_VBUF_FLAG
        && v->padding != RDMA_ONE_SIDED)
    {
        ibv_error_abort(GEN_EXIT_ERR, "vbuf not correct.\n");
    }

    *v->head_flag = 0;
    free_vbuf_head = v;
    v->iheader = NULL;
    v->pheader = NULL;
    v->content_size = 0;
    v->sreq = NULL;
    v->vc = NULL;
    ++num_free_vbuf;
    ++num_vbuf_freed;

#if !defined(CKPT)
    if (process_info.has_srq
#if defined(RDMA_CM)
        || process_info.use_rdma_cm_on_demand
#endif /* defined(RDMA_CM) */
        /*|| MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND)*/
#endif /* !defined(CKPT) */
        )
    {
        pthread_spin_unlock(&vbuf_lock);
    }
}

/**
 *
 */
void MRAILI_Release_recv_rdma(vbuf* v)
{
    vbuf *next_free = NULL;
    MPIDI_VC_t * c = (MPIDI_VC_t *)v->vc;
    int i;

    int next = VC_FIELD(c, connection)->rfp.p_RDMA_recv_tail + 1;

    if (next >= num_rdma_buffer)
    {
        next = 0;
    }

    next_free = &(VC_FIELD(c, connection)->rfp.RDMA_recv_buf[next]);
    v->padding = FREE_FLAG;
    *v->head_flag = 0;
    v->sreq = NULL;
    v->content_size = 0;

    if (v != next_free)
    {
        return;
    }
    /* search all free buffers */
    for (i = next; i != VC_FIELD(c, connection)->rfp.p_RDMA_recv;)
    {
        if (VC_FIELD(c, connection)->rfp.RDMA_recv_buf[i].padding == FREE_FLAG)
        {
            ++VC_FIELD(c, connection)->rfp.rdma_credit;

            if (++VC_FIELD(c, connection)->rfp.p_RDMA_recv_tail >= num_rdma_buffer)
            {
                VC_FIELD(c, connection)->rfp.p_RDMA_recv_tail = 0;
            }

            VC_FIELD(c, connection)->rfp.RDMA_recv_buf[i].padding = BUSY_FLAG;
            *VC_FIELD(c, connection)->rfp.RDMA_recv_buf[i].head_flag = 0;
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



#undef FUNCNAME
#define FUNCNAME vbuf_init_rdma_write
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rdma_write(vbuf* v)
{
    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_RDMA_WRITE);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_RDMA_WRITE);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->padding = FREE_FLAG;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_RDMA_WRITE);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_send(vbuf* v, unsigned long len, int rail)
{
    int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);

    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_SEND);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_SEND;
    v->desc.u.sr.wr_id = (uintptr_t) v;
    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = v->region->mem_handle[hca_num]->lkey;
    v->desc.sg_entry.addr = (uintptr_t)(v->buffer);
    v->padding = NORMAL_VBUF_FLAG;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_SEND);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_recv(vbuf* v, unsigned long len, int rail)
{
    MPIU_Assert(v != NULL);
    int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);

    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_RECV);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_RECV);

    v->desc.u.rr.next = NULL;
    v->desc.u.rr.wr_id = (uintptr_t) v;
    v->desc.u.rr.num_sge = 1;
    v->desc.u.rr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = v->region->mem_handle[hca_num]->lkey;
    v->desc.sg_entry.addr = (uintptr_t)(v->buffer);
    v->padding = NORMAL_VBUF_FLAG;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_RECV);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_rget
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rget(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int len,
    int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_RGET);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_RGET);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_RDMA_READ;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.rdma.remote_addr = (uintptr_t)(remote_address);
    v->desc.u.sr.wr.rdma.rkey = rkey;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(local_address);
    v->padding = RGET_VBUF_FLAG;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_RGET);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_rput
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rput(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int len,
    int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_NIB_VBUF_INIT_RPUT);
    MPIDI_FUNC_ENTER(MPID_STATE_NIB_VBUF_INIT_RPUT);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.rdma.remote_addr = (uintptr_t)(remote_address);
    v->desc.u.sr.wr.rdma.rkey = rkey;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(local_address);
    v->padding = RPUT_VBUF_FLAG;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_NIB_VBUF_INIT_RPUT);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_rma_get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rma_get(vbuf *v, void *l_addr, uint32_t lkey,
                       void *r_addr, uint32_t rkey, int len, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_RMA_GET);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_RMA_GET);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_RDMA_READ;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.rdma.remote_addr = (uintptr_t)(r_addr);
    v->desc.u.sr.wr.rdma.rkey = rkey;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(l_addr);
    v->padding = RDMA_ONE_SIDED;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_RMA_GET);
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_rma_put
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rma_put(vbuf *v, void *l_addr, uint32_t lkey,
                       void *r_addr, uint32_t rkey, int len, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_VBUF_INIT_RMA_PUT);
    MPIDI_FUNC_ENTER(MPID_STATE_VBUF_INIT_RMA_PUT);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.rdma.remote_addr = (uintptr_t)(r_addr);
    v->desc.u.sr.wr.rdma.rkey = rkey;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(l_addr);
    v->padding = RDMA_ONE_SIDED;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_INIT_RMA_PUT);
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

    for (i=0; i < ib_hca_num_hcas; ++i)
    {
        ptag_save[i] = process_info.ptag[i];
    }

    while (vr)
    {
        for (i = 0; i < ib_hca_num_hcas; ++i)
        {
            vr->mem_handle[i] = ibv_reg_mr(
                ptag_save[i],
                vr->malloc_buf_start,
                vr->count * rdma_vbuf_total_size,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

            if (!vr->mem_handle[i])
            {
                ibv_error_abort(IBV_RETURN_ERR,"Cannot reregister vbuf region\n");
            }
        }

        vr = vr->next;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_VBUF_REREGISTER_ALL);
}
#endif /* defined(CKPT) */

/* vi:set sw=4 tw=80: */
