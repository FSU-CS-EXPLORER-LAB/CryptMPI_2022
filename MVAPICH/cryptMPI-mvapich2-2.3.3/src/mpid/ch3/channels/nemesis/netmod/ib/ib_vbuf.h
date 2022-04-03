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

/*#include "mpichconf.h"*/
#include "infiniband/verbs.h"
#include "ib_param.h"
#include "ib_hca.h"
#include "ib_errors.h"

#define IB_PKT_HEADER_LENGTH (sizeof(MPIDI_nem_ib_pkt_comm_header))

#define CREDIT_VBUF_FLAG (111)
#define NORMAL_VBUF_FLAG (222)
#define RPUT_VBUF_FLAG (333)
#define RGET_VBUF_FLAG (444)
#define RDMA_ONE_SIDED (555)
#define COLL_VBUF_FLAG (666)
#define VBUF_FLAG_TYPE uint64_t

#define FREE_FLAG (0)
#define BUSY_FLAG (1)

#define PKT_NO_SEQ_NUM -2
#define PKT_IS_NULL -1

#define VBUF_BUFFER_SIZE (rdma_vbuf_total_size)

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

#define MRAIL_MAX_EAGER_SIZE (VBUF_BUFFER_SIZE - IB_PKT_HEADER_LENGTH)

#define MRAIL_MAX_RDMA_FP_SIZE (rdma_fp_buffer_size - IB_PKT_HEADER_LENGTH - VBUF_FAST_RDMA_EXTRA_BYTES)

typedef struct vbuf
{
    struct ibv_wr_descriptor desc;

    /** Pointer to MPI packet header. */
    void* pheader;

    /** Pointer to netmod/ib header. */
    void* iheader;

    void* sreq;
    struct vbuf_region* region;
    void* vc;

    /**
     * The rail number used to send the buffer.
     */
    int rail;

    int padding;

    VBUF_FLAG_TYPE* head_flag;

    unsigned char* buffer;

    int content_size;
    int content_consumed;

    /** used to keep track of eager sends */
    uint8_t eager;
    uint8_t coalesce;

    /** used to keep one sided put get list */
    void * list;

    /* NULL shandle means not send or not complete. Non-null
     * means pointer to send handle that is now complete. Used
     * by MRAILI_process_send
     */
} vbuf;

/* one for head and one for tail */
#define VBUF_FAST_RDMA_EXTRA_BYTES (2 * sizeof(VBUF_FLAG_TYPE))
#define VBUF_IB_NETMOD_OFFSET (sizeof(VBUF_FLAG_TYPE) + sizeof(MPIDI_CH3_Pkt_t))

#define FAST_RDMA_ALT_TAG 0x8000
#define FAST_RDMA_SIZE_MASK 0x7fff

#if defined(DEBUG)
void dump_vbuf(char* msg, vbuf* v);
#else /* defined(DEBUG) */
#define dump_vbuf(msg, v)
#endif /* defined(DEBUG) */

int init_vbuf_lock();

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
} vbuf_region;

static void inline VBUF_SET_RDMA_ADDR_KEY(
    vbuf* v,
    int len,
    void* local_addr,
    uint32_t lkey,
    void* remote_addr,
    uint32_t rkey)
{
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);

    (v)->desc.u.sr.wr.rdma.remote_addr = (uintptr_t) (remote_addr);
    (v)->desc.u.sr.wr.rdma.rkey = (rkey);
    (v)->desc.sg_entry.length = (len);
    (v)->desc.sg_entry.lkey = (lkey);
    (v)->desc.sg_entry.addr = (uintptr_t)(local_addr);
}

int allocate_vbufs(int nvbufs);

void deallocate_vbufs(int);
void deallocate_vbuf_region();

vbuf* get_vbuf();

void MRAILI_Release_vbuf(vbuf* v);

void vbuf_init_rdma_write(vbuf* v);

void vbuf_init_send(vbuf* v, unsigned long len, int rail);

void vbuf_init_recv(vbuf* v, unsigned long len, int rail);

void vbuf_init_rput(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int nbytes,
    int rail);

void vbuf_init_rget(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int nbytes,
    int rail);

void vbuf_init_rma_get(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int nbytes,
    int rail);

void vbuf_init_rma_put(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_address,
    uint32_t rkey,
    int nbytes,
    int rail);

#if defined(CKPT)
void vbuf_reregister_all();
#endif /* defined(CKPT) */

void MRAILI_Release_recv_rdma(vbuf* v);

#endif
