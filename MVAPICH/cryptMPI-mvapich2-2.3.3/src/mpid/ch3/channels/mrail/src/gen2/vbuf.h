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

#ifndef _VBUF_H_
#define _VBUF_H_

#include "mpichconf.h"
#include "infiniband/verbs.h"
#include "ibv_param.h"
#include "mv2_clock.h"
#if defined(_ENABLE_CUDA_)
#include "ibv_cuda_util.h"
#endif
#include <mpiutil.h>

#define CREDIT_VBUF_FLAG (111)
#define NORMAL_VBUF_FLAG (222)
#define RPUT_VBUF_FLAG (333)
#define RGET_VBUF_FLAG (444)
#define RDMA_ONE_SIDED (555)
#define COLL_VBUF_FLAG (666)
/*
** FIXME: Change the size of VBUF_FLAG_TYPE to 4 bytes when size of
** MPIDI_CH3_Pkt_send is changed to mutliple of 4. This will fix the 
** issue of recv memcpy alignment.
*/
#define VBUF_FLAG_TYPE uint64_t

#define FREE_FLAG (0)
#define BUSY_FLAG (1)
#define PKT_NO_SEQ_NUM -2
#define PKT_IS_NULL -1

#define MRAILI_ALIGN_LEN(len, align_unit)           \
{                                                   \
    len = ((int)(((len)+align_unit-1) /             \
                align_unit)) * align_unit;          \
}

#define ROUNDUP(len, unit) ((len + unit - 1) / unit) 

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#ifdef __ia64__
/* Only ia64 requires this */
#define SHMAT_ADDR (void *)(0x8000000000000000UL)
#define SHMAT_FLAGS (SHM_RND)
#else
#define SHMAT_ADDR (void *)(0x0UL)
#define SHMAT_FLAGS (0)
#endif /* __ia64__*/
#define HUGEPAGE_ALIGN  (2*1024*1024)

/*
 * brief justification for vbuf format:
 * descriptor must be aligned (64 bytes).
 * vbuf size must be multiple of this alignment to allow contiguous allocation
 * descriptor and buffer should be contiguous to allow via implementations that
 * optimize contiguous descriptor/data (? how likely ?)
 * need to be able to store send handle in vbuf so that we can mark sends
 * complete when communication completes. don't want to store
 * it in packet header because we don't always need to send over the network.
 * don't want to store at beginning or between desc and buffer (see above) so
 * store at end.
 */

struct ibv_wr_descriptor
{
    union
    {
        struct ibv_recv_wr rr;
        struct ibv_send_wr sr;
    } u;
    union
    {
        struct ibv_send_wr* bad_sr;
        struct ibv_recv_wr* bad_rr;
    } y;
    struct ibv_sge sg_entry;
    void* next;
};

#define VBUF_BUFFER_SIZE (rdma_vbuf_total_size)

#define MRAIL_MAX_EAGER_SIZE VBUF_BUFFER_SIZE
typedef enum {
    IB_TRANSPORT_UD = 1,
    IB_TRANSPORT_RC = 2,
} ib_transport;

#define UD_VBUF_FREE_PENIDING       (0x01)
#define UD_VBUF_SEND_INPROGRESS     (0x02)
#define UD_VBUF_RETRY_ALWAYS        (0x04)
#define UD_VBUF_MCAST_MSG           (0x08)


#define MV2_UD_GRH_LEN (40)

#define PKT_TRANSPORT_OFFSET(_v) ((_v->transport == IB_TRANSPORT_UD) ? MV2_UD_GRH_LEN : 0)

/* extend this macro if there is more control messages */
#define IS_CNTL_MSG(p) \
    (p->type ==  MPIDI_CH3_PKT_FLOW_CNTL_UPDATE || \
        p->type ==  MPIDI_CH3_PKT_NOOP)

#ifdef _MCST_SUPPORT_
#define IS_MCAST_MSG(p) \
    (p->type == MPIDI_CH3_PKT_MCST || \
        p->type == MPIDI_CH3_PKT_MCST_INIT || \
            (mcast_use_mcast_nack && p->type == MPIDI_CH3_PKT_MCST_NACK))
#endif /*_MCST_SUPPORT_*/

#define SET_PKT_LEN_HEADER(_v, _wc) {                                       \
    if(IB_TRANSPORT_UD == (_v)->transport) {                                \
        (_v)->content_size = (_wc).byte_len - MV2_UD_GRH_LEN ;              \
    } else {                                                                \
        (_v)->content_size= _wc.byte_len;                                   \
    }                                                                       \
}

#define SET_PKT_HEADER_OFFSET(_v) {                                         \
    (_v)->pheader = (_v)->buffer + PKT_TRANSPORT_OFFSET(_v);                \
}

#define MRAIL_MAX_RDMA_FP_SIZE (rdma_fp_buffer_size - VBUF_FAST_RDMA_EXTRA_BYTES)

#define MRAIL_MAX_UD_SIZE (rdma_default_ud_mtu - MV2_UD_GRH_LEN)

typedef struct link
{
    void *next;
    void *prev;
} LINK;

typedef struct vbuf
{
    struct ibv_wr_descriptor desc;
    void* pheader;
    void* sreq;
    struct vbuf_region* region;
    void* vc;
    int rail;
    int padding;
    VBUF_FLAG_TYPE* head_flag;
    unsigned char* buffer;

    size_t content_size;
    size_t content_consumed;

    /* used to keep track of eager sends */
    uint8_t eager;
    uint8_t coalesce;
  
    /* used to keep one sided put get list */
    void * list;

    /* used for rma fast path */
    void * tmp_dreg;

    /* target_rank for rma fast path */

    int32_t target_rank;

    /*For fetch_and_op*/
    void * result_addr;

    /* NULL shandle means not send or not complete. Non-null
     * means pointer to send handle that is now complete. Used
     * by MRAILI_Process_send
     */
    ib_transport transport;
    uint16_t seqnum;
#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
    uint16_t retry_count;
    uint16_t pending_send_polls;
    uint8_t flags;
    double timestamp;
#if defined(_ENABLE_UD_)
    uint8_t in_sendwin;
    LINK sendwin_msg;
    LINK recvwin_msg;
    LINK extwin_msg;
    LINK unack_msg;
#endif
#if defined(_MCST_SUPPORT_)
    LINK mcast_sendwin_msg;
    LINK mcast_recvwin_msg;
#endif
#endif
    void *pool_index;
    void *next;
    void *orig_vbuf;
    uint8_t finish_count;
    uint16_t displacement;
} vbuf;

/* one for head and one for tail */
#define VBUF_FAST_RDMA_EXTRA_BYTES (2 * sizeof(VBUF_FLAG_TYPE))

#define FAST_RDMA_ALT_TAG 0x8000
#define FAST_RDMA_SIZE_MASK 0x7fff

#if defined(DEBUG)
void dump_vbuf(char* msg, vbuf* v);
#else /* defined(DEBUG) */
#define dump_vbuf(msg, v)
#endif /* defined(DEBUG) */

void mv2_print_vbuf_usage_usage();
int mv2_preallocate_rdma_fp_bufs();
int mv2_free_prealloc_rdma_fp_bufs();
int init_vbuf_lock(void);
void release_vbuf(vbuf* v);
/*
 * Vbufs are allocated in blocks and threaded on a single free list.
 *
 * These data structures record information on all the vbuf
 * regions that have been allocated.  They can be used for
 * error checking and to un-register and deallocate the regions
 * at program termination.
 *
 */
typedef struct vbuf_region
{
    struct ibv_mr* mem_handle[MAX_NUM_HCAS]; /* mem hndl for entire region */
    void* malloc_start;         /* used to free region later  */
    void* malloc_end;           /* to bracket mem region      */
    void* malloc_buf_start;     /* used to free DMA region later */
    void* malloc_buf_end;       /* bracket DMA region */
    int count;                  /* number of vbufs in region  */
    struct vbuf* vbuf_head;     /* first vbuf in region       */
    struct vbuf_region* next;   /* thread vbuf regions        */
    int shmid;
    void *pool_index;   /* region allocated for a pool */
} vbuf_region;

/* The data structure to hold vbuf pool info */
typedef struct vbuf_pool
{
    uint8_t index;
    uint16_t initial_count;
    uint16_t incr_count;
    uint32_t buf_size;
    uint32_t num_allocated;
    uint32_t num_free;
    uint32_t max_num_buf;
    long num_get;
    long num_freed;
    vbuf *free_head;
    vbuf_region *region_head;
}vbuf_pool_t;

#define RDMA_VBUF_POOL_INIT(rdma_vbuf_pool)     \
do{                                             \
    rdma_vbuf_pool.free_head = NULL;            \
    rdma_vbuf_pool.region_head = NULL;          \
    rdma_vbuf_pool.num_allocated = 0;           \
    rdma_vbuf_pool.num_free = 0;                \
    rdma_vbuf_pool.max_num_buf = -1;            \
    rdma_vbuf_pool.num_get = 0;                 \
    rdma_vbuf_pool.num_freed = 0;               \
} while(0)
    
#define MV2_INIT_VBUF(_v)                               \
do {                                                    \
    /* Need to change this to RPUT_VBUF_FLAG later      \
     * if we are doing rput */                          \
    (_v)->padding = NORMAL_VBUF_FLAG;                   \
    (_v)->pheader = (void *)(_v)->buffer;               \
    /* This is probably not the right place to          \
     * initialize shandle to NULL. Do it here for now   \
     * because it will make sure it is always           \
     * initialized. Otherwise we would need to very     \
     * carefully add the initialization in              \
     * a dozen other places, and probably miss one. */  \
    (_v)->sreq = NULL;                                  \
    (_v)->coalesce = 0;                                 \
    (_v)->content_size = 0;                             \
    (_v)->eager = 0;                                    \
    (_v)->finish_count = 0;                             \
    (_v)->orig_vbuf = NULL;                             \
    (_v)->displacement = 0;                             \
    /*This is used for RMA put/get*/                    \
    (_v)->target_rank = -1;                             \
} while (0)

#define MV2_INIT_RC_VBUF(_v)                            \
do {                                                    \
    /* Init VBUF elements */                            \
    MV2_INIT_VBUF(_v);                                  \
    /* Decide which transport need to assign here */    \
    (_v)->transport = IB_TRANSPORT_RC;                  \
} while (0)

#define MV2_GET_AND_INIT_RC_VBUF(_v, _pool)             \
do {                                                    \
    (_v) = (_pool)->free_head;                          \
    (_pool)->free_head = (_pool)->free_head->desc.next; \
    --(_pool)->num_free;                                \
    ++(_pool)->num_get;                                 \
                                                        \
    MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_vbuf_available, 1);  \
    MV2_INIT_RC_VBUF((_v));                             \
} while (0)

/* Reset VBUF parameters */
#define MV2_RESET_VBUF(_v)      \
do {                            \
    (_v)->vc           = NULL;  \
    (_v)->sreq         = NULL;  \
    (_v)->pheader      = NULL;  \
    (_v)->orig_vbuf    = NULL;  \
    *(_v)->head_flag   = 0;     \
    (_v)->content_size = 0;     \
    (_v)->finish_count = 0;     \
    (_v)->displacement = 0;     \
} while (0)

#define MV2_RELEASE_RC_VBUF(_v, _pool)                  \
do {                                                    \
    (_v)->desc.next = (_pool)->free_head;               \
    (_pool)->free_head = (_v);                          \
    ++(_pool)->num_free;                                \
    ++(_pool)->num_freed;                               \
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_vbuf_freed, 1);    \
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_vbuf_available, 1);  \
} while (0)

#define MV2_RELEASE_VBUF_NO_LOCK(_v)                        \
do {                                                        \
    vbuf_pool_t *__pool = (_v)->pool_index;                 \
                                                            \
    MPIU_Assert((_v)->padding == NORMAL_VBUF_FLAG ||        \
                (_v)->padding == RPUT_VBUF_FLAG ||          \
                (_v)->padding == RGET_VBUF_FLAG ||          \
                (_v)->padding == RDMA_ONE_SIDED ||          \
                (_v)->padding == COLL_VBUF_FLAG);           \
                                                            \
    MV2_RELEASE_RC_VBUF((_v), (__pool));                    \
    MV2_RESET_VBUF((_v));                                   \
} while (0)

#define MV2_PREPARE_VBUF_FOR_REPOST(_v)                     \
do {                                                        \
    vbuf_pool_t *__pool = (_v)->pool_index;                 \
    ++(__pool)->num_get;                                    \
    ++(__pool)->num_freed;                                  \
    MV2_RESET_VBUF((_v));                                   \
    MV2_INIT_RC_VBUF((_v));                                 \
} while (0)

#define VBUF_INIT_RECV(_v, _len, _rail)                                     \
do {                                                                        \
    int _hca_num = (_rail) / (rdma_num_rails / rdma_num_hcas);              \
                                                                            \
    MPIU_Assert((_v) != NULL);                                              \
                                                                            \
    (_v)->desc.u.rr.next = NULL;                                            \
    (_v)->desc.u.rr.wr_id = (uintptr_t) (_v);                               \
    (_v)->desc.u.rr.num_sge = 1;                                            \
    (_v)->desc.u.rr.sg_list = &((_v)->desc.sg_entry);                       \
    (_v)->desc.sg_entry.length = (_len);                                    \
    (_v)->desc.sg_entry.lkey = (_v)->region->mem_handle[_hca_num]->lkey;    \
    (_v)->desc.sg_entry.addr = (uintptr_t)((_v)->buffer);                   \
    (_v)->padding = NORMAL_VBUF_FLAG;                                       \
    (_v)->rail = (_rail);                                                   \
} while (0)

#define MV2_REPOST_VBUF_FROM_POOL_TO_SRQ(_pool)                                     \
do {                                                                                \
    vbuf *_v = NULL;                                                                \
    int _hca_num = -1;                                                              \
    struct ibv_recv_wr* _bad_wr = NULL;                                             \
                                                                                    \
    MV2_GET_AND_INIT_RC_VBUF((_v), (_pool));                                        \
                                                                                    \
    MPIU_Assert(((vbuf_pool_t*)(_v)->pool_index)->index ==                          \
                                MV2_RECV_VBUF_POOL_OFFSET);                         \
                                                                                    \
    _hca_num = (_v)->rail / (rdma_num_rails / rdma_num_hcas);                       \
                                                                                    \
    MV2_PREPARE_VBUF_FOR_REPOST((_v));                                              \
    VBUF_INIT_RECV((_v), VBUF_BUFFER_SIZE, (_v)->rail);                             \
    if (unlikely(ibv_post_srq_recv(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[_hca_num],  \
                            &(_v)->desc.u.rr, &_bad_wr))) {                         \
        MV2_RELEASE_RC_VBUF((_v), (_pool));                                         \
    }                                                                               \
} while (0)

#define MV2_REPOST_VBUF_TO_SRQ(_v, _pool)                                           \
do {                                                                                \
    int _hca_num = -1;                                                              \
    struct ibv_recv_wr* _bad_wr = NULL;                                             \
                                                                                    \
    MPIU_Assert(((vbuf_pool_t*)(_v)->pool_index)->index ==                          \
                                MV2_RECV_VBUF_POOL_OFFSET);                         \
                                                                                    \
    _hca_num = (_v)->rail / (rdma_num_rails / rdma_num_hcas);                       \
                                                                                    \
    MV2_PREPARE_VBUF_FOR_REPOST((_v));                                              \
    VBUF_INIT_RECV((_v), VBUF_BUFFER_SIZE, (_v)->rail);                             \
    if (unlikely(ibv_post_srq_recv(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[_hca_num],  \
                            &(_v)->desc.u.rr, &_bad_wr))) {                         \
        MV2_RELEASE_RC_VBUF((_v), (_pool));                                         \
    }                                                                               \
} while (0)

int allocate_vbuf_pool(vbuf_pool_t *rdma_vbuf_pool, int nvbufs);

#define GET_VBUF_BY_OFFSET_WITHOUT_LOCK(_v, _offset)                                \
do {                                                                                \
    MPIU_Assert((_offset) != MV2_RECV_VBUF_POOL_OFFSET);                            \
                                                                                    \
    vbuf_pool_t *__pool = &rdma_vbuf_pools[(_offset)];                              \
                                                                                    \
    if (likely(__pool->free_head)) {                                                \
        MV2_GET_AND_INIT_RC_VBUF((_v), __pool);                                     \
    } else {                                                                        \
        if (allocate_vbuf_pool(__pool, __pool->incr_count) != 0) {                  \
            ibv_va_error_abort(GEN_EXIT_ERR,"vbuf pool allocation failed");         \
        }                                                                           \
        MV2_GET_AND_INIT_RC_VBUF((_v), __pool);                                     \
    }                                                                               \
} while (0)

#ifdef _ENABLE_CUDA_
#define SIZE_TO_OFFSET(_msg_sz, _offset)                                            \
do {                                                                                \
    (_offset) = (((_msg_sz) <= DEFAULT_SMALL_VBUF_SIZE)?                            \
                    MV2_SMALL_DATA_VBUF_POOL_OFFSET:                                \
                    (((_msg_sz) <= DEFAULT_MEDIUM_VBUF_SIZE)?                       \
                        MV2_MEDIUM_DATA_VBUF_POOL_OFFSET:                           \
                        (((_msg_sz) <= rdma_vbuf_total_size)?                       \
                            MV2_LARGE_DATA_VBUF_POOL_OFFSET:                        \
                            MV2_CUDA_VBUF_POOL_OFFSET)));                           \
} while (0)
#else
#define SIZE_TO_OFFSET(_msg_sz, _offset)                                            \
do {                                                                                \
    (_offset) = (((_msg_sz) <= DEFAULT_SMALL_VBUF_SIZE)?                            \
                    MV2_SMALL_DATA_VBUF_POOL_OFFSET:                                \
                    (((_msg_sz) <= DEFAULT_MEDIUM_VBUF_SIZE)?                       \
                        MV2_MEDIUM_DATA_VBUF_POOL_OFFSET:                           \
                        MV2_LARGE_DATA_VBUF_POOL_OFFSET));                          \
} while (0)
#endif /* _ENABLE_CUDA_ */

#define MV2_GET_RC_VBUF(_v, _msg_sz)                                                \
do {                                                                                \
    int __offset = 0;                                                               \
    SIZE_TO_OFFSET((_msg_sz), (__offset));                                          \
    GET_VBUF_BY_OFFSET_WITHOUT_LOCK((_v), (__offset));                              \
} while (0)

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
#define MV2_RELEASE_UD_VBUF(_v)                             \
do {                                                        \
    MPIU_Assert((_v) != ud_free_vbuf_head);                 \
    (_v)->desc.next = ud_free_vbuf_head;                    \
    ud_free_vbuf_head = (_v);                               \
    ++ud_num_free_vbuf;                                     \
    ++ud_num_vbuf_freed;                                    \
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_ud_vbuf_freed, 1);     \
    MPIR_T_PVAR_LEVEL_INC(MV2, mv2_ud_vbuf_available, 1);   \
} while (0)

#define MV2_INIT_UD_VBUF(_v)                            \
do {                                                    \
    /* Init VBUF elements */                            \
    MV2_INIT_VBUF(_v);                                  \
    /* Init UD specific VBUF elements */                \
    (_v)->flags = 0;                                    \
    (_v)->retry_count = 0;                              \
    (_v)->pool_index = NULL;                            \
    (_v)->pending_send_polls = 0;                       \
    /* Decide which transport need to assign here */    \
    (_v)->transport = IB_TRANSPORT_UD;                  \
} while (0)

#define MV2_GET_UD_VBUF(_v)                                                         \
do {                                                                                \
	(_v) = ud_free_vbuf_head;                                                       \
	--ud_num_free_vbuf;                                                             \
	MPIR_T_PVAR_LEVEL_DEC(MV2, mv2_ud_vbuf_available, 1);                           \
	++ud_num_vbuf_get;                                                              \
	/* this correctly handles removing from single entry free list */               \
	ud_free_vbuf_head = ud_free_vbuf_head->desc.next;                               \
} while (0)

#define MV2_GET_AND_INIT_UD_VBUF(_v)                                                \
do {                                                                                \
    if (likely(ud_free_vbuf_head)) {                                                \
        MV2_GET_UD_VBUF((_v));                                                      \
        MV2_INIT_UD_VBUF((_v));                                                     \
    } else {                                                                        \
        if(allocate_ud_vbuf_region(rdma_vbuf_secondary_pool_size) != 0) {           \
            ibv_va_error_abort(GEN_EXIT_ERR, "UD VBUF allocation failed.\n");       \
        }                                                                           \
        MV2_GET_UD_VBUF((_v));                                                      \
        MV2_INIT_UD_VBUF((_v));                                                     \
    }                                                                               \
} while (0)

#define MRAILI_Get_buffer(_vc, _v, _msg_sz)                                         \
do {                                                                                \
    if ((_vc)->mrail.state & MRAILI_RC_CONNECTED) {                                 \
        MV2_GET_RC_VBUF((_v), (_msg_sz));                                           \
    } else  {                                                                       \
        MV2_GET_AND_INIT_UD_VBUF((_v));                                             \
    }                                                                               \
} while (0)
#else
#define MRAILI_Get_buffer(_vc, _v, _msg_sz)                                         \
do {                                                                                \
    MV2_GET_RC_VBUF((_v), (_msg_sz));                                               \
} while (0)
#endif /*defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)*/

static inline void VBUF_SET_RDMA_ADDR_KEY(
    vbuf* v, 
    int len,
    void* local_addr,
    uint32_t lkey,
    void* remote_addr,
    uint32_t rkey)
{
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    if (likely(len <= rdma_max_inline_size)) {
        v->desc.u.sr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    } else {
        v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    }
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);

    (v)->desc.u.sr.wr.rdma.remote_addr = (uintptr_t) (remote_addr);
    (v)->desc.u.sr.wr.rdma.rkey = (rkey);
    (v)->desc.sg_entry.length = (len);
    (v)->desc.sg_entry.lkey = (lkey);
    (v)->desc.sg_entry.addr = (uintptr_t)(local_addr);
}

static inline void vbuf_init_rdma_write(vbuf* v)
{
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->padding = FREE_FLAG;
}

static inline void vbuf_init_send(vbuf* v, unsigned long len, int rail)
{
    int hca_num = rail / (rdma_num_rails / rdma_num_hcas);

    v->desc.u.sr.next = NULL;
    if (likely(len <= rdma_max_inline_size)) {
        v->desc.u.sr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    } else {
        v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    }
    v->desc.u.sr.opcode = IBV_WR_SEND;
    v->desc.u.sr.wr_id = (uintptr_t) v;
    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = v->region->mem_handle[hca_num]->lkey;
    v->desc.sg_entry.addr = (uintptr_t)(v->buffer);
    v->padding = NORMAL_VBUF_FLAG;
    v->rail = rail;
}

static inline void vbuf_init_rget(
    vbuf* v,
    void* local_address,
    uint32_t lkey, 
    void* remote_address,
    uint32_t rkey,
    int len,
    int rail)
{
    v->desc.u.sr.next = NULL;
    /* IBV_WR_RDMA_READ cannot support INLINE */
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
}

static inline void vbuf_init_rput(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int len,
    int rail)
{
    v->desc.u.sr.next = NULL;
    if (likely(len <= rdma_max_inline_size)) {
        v->desc.u.sr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    } else {
        v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    }
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
}

static inline void vbuf_init_rma_get(vbuf *v, void *l_addr, uint32_t lkey,
                       void *r_addr, uint32_t rkey, uint32_t len, int rail)
{
    v->desc.u.sr.next = NULL;
    /* IBV_WR_RDMA_READ cannot support INLINE */
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
}

static inline void vbuf_init_rma_put(vbuf *v, void *l_addr, uint32_t lkey,
                       void *r_addr, uint32_t rkey, uint32_t len, int rail)
{
    v->desc.u.sr.next = NULL;
    if (likely(len <= rdma_max_inline_size)) {
        v->desc.u.sr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    } else {
        v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    }
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
}

static inline void vbuf_init_rma_fetch_and_add(vbuf *v, void *l_addr, uint32_t lkey,
        void *r_addr, uint32_t rkey, uint64_t add,
        int rail)
{   
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    /* IBV_WR_ATOMIC_FETCH_AND_ADD cannot support INLINE */
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.atomic.remote_addr = (uintptr_t)(r_addr);
    v->desc.u.sr.wr.atomic.rkey = rkey;
    v->desc.u.sr.wr.atomic.compare_add = add;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = sizeof(uint64_t);
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(l_addr);
    v->padding = RDMA_ONE_SIDED;
    v->rail = rail;
}

static inline void vbuf_init_rma_compare_and_swap(vbuf *v, void *l_addr, uint32_t lkey,
                        void *r_addr, uint32_t rkey, uint64_t compare, 
                        uint64_t swap, int rail)
{
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    /* IBV_WR_ATOMIC_CMP_AND_SWP cannot support INLINE */
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.wr_id  = (uintptr_t) v;
    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.atomic.remote_addr = (uintptr_t)(r_addr);
    v->desc.u.sr.wr.atomic.rkey = rkey;
    v->desc.u.sr.wr.atomic.compare_add = compare;
    v->desc.u.sr.wr.atomic.swap = swap;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);

    v->desc.sg_entry.length = sizeof(uint64_t);
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(v->buffer);

    v->padding = RDMA_ONE_SIDED;
    v->rail = rail;
}

int allocate_vbufs(struct ibv_pd* ptag[]);

void deallocate_vbufs(int);
void deallocate_vbuf_region(void);

vbuf* get_vbuf(size_t message_size);
vbuf* get_vbuf_by_offset(int offset);

#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
vbuf* get_ud_vbuf(void);
int allocate_ud_vbufs(int nvbufs);
int allocate_ud_vbuf_region(int nvbufs);
void vbuf_init_ud_recv(vbuf* v, unsigned long len, int rail);
extern vbuf *ud_free_vbuf_head;
extern int ud_vbuf_n_allocated;
extern long ud_num_free_vbuf;
extern long ud_num_vbuf_get;
extern long ud_num_vbuf_freed;
#endif

void MRAILI_Release_vbuf(vbuf* v);

#if defined(CKPT)
void vbuf_reregister_all();
#endif /* defined(CKPT) */

#ifdef _ENABLE_CUDA_
void register_cuda_vbuf_regions();
#endif

extern vbuf_pool_t *rdma_vbuf_pools;
extern vbuf_pool_t mv2_srq_repost_pool;
extern volatile int rdma_num_vbuf_pools;

#endif
