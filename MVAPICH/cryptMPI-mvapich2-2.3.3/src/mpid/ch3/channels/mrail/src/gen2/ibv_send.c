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

#include "mpichconf.h"
#include "mpiimpl.h"
#include <mpimem.h>
#include "rdma_impl.h"
#include "ibv_impl.h"
#include "vbuf.h"
#include "upmi.h"
#include "mpiutil.h"
#include "dreg.h"
#include "debug_utils.h"
#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#endif 

#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);                                    \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

#define INCR_EXT_SENDQ_SIZE(_c,_rail) \
    ++ rdma_global_ext_sendq_size;      \
    ++ (_c)->mrail.rails[(_rail)].ext_sendq_size;

#define DECR_EXT_SENDQ_SIZE(_c,_rail)  \
        -- rdma_global_ext_sendq_size;      \
        -- (_c)->mrail.rails[(_rail)].ext_sendq_size; 

static inline vbuf * MRAILI_Get_Vbuf(MPIDI_VC_t * vc, size_t pkt_len);
static inline int MPIDI_CH3I_MRAILI_Fast_rdma_ok(MPIDI_VC_t * vc, MPIDI_msg_sz_t len);

static inline int MRAILI_Coalesce_ok(MPIDI_VC_t * vc, int rail)
{
    if(unlikely(rdma_use_coalesce && 
            (vc->mrail.outstanding_eager_vbufs >= rdma_coalesce_threshold || 
               vc->mrail.rails[rail].send_wqes_avail == 0) &&
         (mv2_MPIDI_CH3I_RDMA_Process.has_srq || 
          (vc->mrail.srp.credits[rail].remote_credit > 0 && 
           NULL == &(vc->mrail.srp.credits[rail].backlog))))) {
        return 1;
    }

    return 0;
}

/* to handle Send Q overflow, we maintain an extended send queue
 * above the HCA.  This permits use to have a virtually unlimited send Q depth
 * (limited by number of vbufs available for send)
 */
#undef FUNCNAME
#define FUNCNAME MRAILI_Ext_sendq_enqueue
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline void MRAILI_Ext_sendq_enqueue(MPIDI_VC_t *c,
                                            int rail, 
                                            vbuf * v)          
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);

    v->desc.next = NULL;
    
    if (c->mrail.rails[rail].ext_sendq_head == NULL) {
        c->mrail.rails[rail].ext_sendq_head = v;
    } else {                                     
        c->mrail.rails[rail].ext_sendq_tail->desc.next = v;
    }
    c->mrail.rails[rail].ext_sendq_tail = v;  
    DEBUG_PRINT("[ibv_send] enqueue, head %p, tail %p\n", 
            c->mrail.rails[rail].ext_sendq_head, 
            c->mrail.rails[rail].ext_sendq_tail); 

    INCR_EXT_SENDQ_SIZE(c, rail)

    if (c->mrail.rails[rail].ext_sendq_size > rdma_rndv_ext_sendq_size) {
#ifdef _ENABLE_CUDA_
        if (!rdma_enable_cuda)
#endif
        {
            c->force_rndv = 1;
        }
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);
}

int check_cq_overflow_for_ib(MPIDI_VC_t *c, int rail)
{
    return 0;
}

int check_cq_overflow_for_iwarp(MPIDI_VC_t *c, int rail)
{
    char cq_overflow = 0;

    if(rdma_iwarp_use_multiple_cq) {
      if ((NULL != c->mrail.rails[rail].send_cq_hndl) &&
          (mv2_MPIDI_CH3I_RDMA_Process.global_used_send_cq >= 
           rdma_default_max_cq_size)) {
          /* We are monitoring CQ's and there is CQ overflow */
          cq_overflow = 1;
      }
    } else {
      if ((NULL != c->mrail.rails[rail].send_cq_hndl) &&
          ((mv2_MPIDI_CH3I_RDMA_Process.global_used_send_cq +
            mv2_MPIDI_CH3I_RDMA_Process.global_used_recv_cq) >= 
            rdma_default_max_cq_size)) {
          /* We are monitoring CQ's and there is CQ overflow */       
          cq_overflow = 1; 
      }
    }

    return cq_overflow;
}

/* dequeue and send as many as we can from the extended send queue
 * this is called in each function which may post send prior to it attempting
 * its send, hence ordering of sends is maintained
 */
#undef FUNCNAME
#define FUNCNAME MRAILI_Ext_sendq_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline void MRAILI_Ext_sendq_send(MPIDI_VC_t *c, int rail)    
{
    vbuf *v;
    char cq_overflow = 0;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_EXT_SENDQ_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_EXT_SENDQ_SEND);

#ifdef _ENABLE_XRC_
    MPIU_Assert (!USE_XRC || VC_XST_ISUNSET (c, XF_INDIRECT_CONN));
#endif

    cq_overflow = check_cq_overflow(c, rail);

    while (c->mrail.rails[rail].send_wqes_avail
            && !cq_overflow
            && c->mrail.rails[rail].ext_sendq_head) {
        v = c->mrail.rails[rail].ext_sendq_head;
        c->mrail.rails[rail].ext_sendq_head = v->desc.next;
        if (v == c->mrail.rails[rail].ext_sendq_tail) {
            c->mrail.rails[rail].ext_sendq_tail = NULL;
        }
        v->desc.next = NULL;
        -- c->mrail.rails[rail].send_wqes_avail;                

        DECR_EXT_SENDQ_SIZE(c, rail)

        if (unlikely(1 == v->coalesce)) {
            DEBUG_PRINT("Sending coalesce vbuf %p\n", v);
            MPIDI_CH3I_MRAILI_Pkt_comm_header *p = v->pheader;
            vbuf_init_send(v, v->content_size, v->rail);

            p->seqnum = v->seqnum;

            if(c->mrail.coalesce_vbuf == v) {
                c->mrail.coalesce_vbuf = NULL;
            }
        } 

        IBV_POST_SR(v, c, rail, "Mrail_post_sr (MRAILI_Ext_sendq_send)");
    }

    DEBUG_PRINT( "[ibv_send] dequeue, head %p, tail %p\n",
        c->mrail.rails[rail].ext_sendq_head,
        c->mrail.rails[rail].ext_sendq_tail);

    if (c->mrail.rails[rail].ext_sendq_size <= rdma_rndv_ext_sendq_size) {
        c->force_rndv = 0;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_EXT_SENDQ_SEND);
}

#define FLUSH_SQUEUE(_vc) {                                           \
    if(NULL != (_vc)->mrail.coalesce_vbuf) {                          \
        MRAILI_Ext_sendq_send(_vc, (_vc)->mrail.coalesce_vbuf->rail); \
    }                                                                 \
}

#define FLUSH_RAIL(_vc,_rail) {                                       \
    if(unlikely(NULL != (_vc)->mrail.coalesce_vbuf &&                 \
            (_vc)->mrail.coalesce_vbuf->rail == _rail)) {             \
        MRAILI_Ext_sendq_send(_vc, (_vc)->mrail.coalesce_vbuf->rail); \
        (_vc)->mrail.coalesce_vbuf = NULL;                            \
    }                                                                 \
}


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_put_datav
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_put_datav(MPIDI_VC_t * vc, MPL_IOV * iov, int n,
                              int *num_bytes_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    /* all variable must be declared before the state declarations */
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PUT_DATAV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PUT_DATAV);

    /* Insert implementation here */
    PRINT_ERROR("MPIDI_CH3I_RDMA_put_datav is not implemented\n" );
    exit(EXIT_FAILURE);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PUT_DATAV);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_read_datav
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_read_datav(MPIDI_VC_t * recv_vc_ptr, MPL_IOV * iov,
                               int iovlen, int
                               *num_bytes_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    /* all variable must be declared before the state declarations */
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_READ_DATAV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_READ_DATAV);

    /* Insert implementation here */
    PRINT_ERROR("MPIDI_CH3I_RDMA_read_datav Function not implemented\n");
    exit(EXIT_FAILURE);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_READ_DATAV);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Fast_rdma_fill_start_buf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MRAILI_Fast_rdma_fill_start_buf(MPIDI_VC_t * vc,
                                    MPL_IOV * iov, int n_iov,
                                    int *num_bytes_ptr)
{
    /* FIXME: Here we assume that iov holds a packet header */
#ifndef MV2_DISABLE_HEADER_CACHING 
    MPIDI_CH3_Pkt_send_t *cached =  vc->mrail.rfp.cached_outgoing;
#endif
    MPIDI_CH3_Pkt_send_t *header;
    vbuf *v = &(vc->mrail.rfp.RDMA_send_buf[vc->mrail.rfp.phead_RDMA_send]);
    void *vstart;
    void *data_buf;

    int len = *num_bytes_ptr, avail = 0; 
    int seq_num;
    int i;

    header = iov[0].MPL_IOV_BUF;
    
    seq_num =  header->seqnum = vc->mrail.seqnum_next_tosend;
    vc->mrail.seqnum_next_tosend++;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_FILL_START_BUF);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_FILL_START_BUF);

    /* Calculate_IOV_len(iov, n_iov, len); */

    avail   = len;
    PACKET_SET_RDMA_CREDIT(header, vc);
    *num_bytes_ptr = 0;

    DEBUG_PRINT("Header info, tag %d, rank %d, context_id %d\n", 
            header->match.parts.tag, header->match.parts.rank, header->match.parts.context_id);
#ifndef MV2_DISABLE_HEADER_CACHING 

    if ((header->type == MPIDI_CH3_PKT_EAGER_SEND) &&
        (len - sizeof(MPIDI_CH3_Pkt_eager_send_t) <= MAX_SIZE_WITH_HEADER_CACHING) &&
        (header->match.parts.tag == cached->match.parts.tag) &&
        (header->match.parts.rank == cached->match.parts.rank) &&
        (header->match.parts.context_id == cached->match.parts.context_id) &&
        (header->vbuf_credit == cached->vbuf_credit) &&
        (header->remote_credit == cached->remote_credit) &&
        (header->rdma_credit == cached->rdma_credit)) {
        /* change the header contents */
        ++vc->mrail.rfp.cached_hit;

        if (header->sender_req_id == cached->sender_req_id) {
            MPIDI_CH3I_MRAILI_Pkt_fast_eager *fast_header;
            vstart = v->buffer;

            /*
            DEBUG_PRINT 
                ("[send: fill buf], head cached, head_flag %p, vstart %p, length %d",
                 &v->head_flag, vstart,
                 len - sizeof(MPIDI_CH3_Pkt_eager_send_t) + 
		 sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager));
                 */
    
            fast_header = vstart;
            fast_header->type = MPIDI_CH3_PKT_FAST_EAGER_SEND;
            fast_header->bytes_in_pkt = len - sizeof(MPIDI_CH3_Pkt_eager_send_t);
            fast_header->seqnum = seq_num;
            v->pheader = fast_header;
            data_buf = (void *) ((unsigned long) vstart +
                                 sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager));
   
	    if (iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_eager_send_t)) 
	      MPIU_Memcpy(data_buf, (void *)((uintptr_t)iov[0].MPL_IOV_BUF +
			   sizeof(MPIDI_CH3_Pkt_eager_send_t)), 
			   iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_eager_send_t));

	    data_buf = (void *)((uintptr_t)data_buf + iov[0].MPL_IOV_LEN -
			sizeof(MPIDI_CH3_Pkt_eager_send_t));

            *num_bytes_ptr += sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager);
            avail -= sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager);
        } else {
            MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req *fast_header;
            vstart = v->buffer;

            DEBUG_PRINT
                ("[send: fill buf], head cached, head_flag %p, vstart %p, length %d\n",
                 &v->head_flag, vstart,
                 len - sizeof(MPIDI_CH3_Pkt_eager_send_t) + 
		 sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req));
             
            fast_header = vstart;
            fast_header->type = MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ;
            fast_header->bytes_in_pkt = len - sizeof(MPIDI_CH3_Pkt_eager_send_t);
            fast_header->seqnum = seq_num;
            fast_header->sender_req_id = header->sender_req_id;
            cached->sender_req_id = header->sender_req_id;
            v->pheader = fast_header;
            data_buf =
                (void *) ((unsigned long) vstart +
                          sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req));
	    if (iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_eager_send_t)) 
	      MPIU_Memcpy(data_buf, (void *)((uintptr_t)iov[0].MPL_IOV_BUF +
			   sizeof(MPIDI_CH3_Pkt_eager_send_t)), 
			   iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_eager_send_t));

	    data_buf = (void *)((uintptr_t)data_buf + iov[0].MPL_IOV_LEN -
			sizeof(MPIDI_CH3_Pkt_eager_send_t));

            *num_bytes_ptr += sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req);
            avail -= sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req);
        }
    } else
#endif
    {
        vstart = v->buffer;
        DEBUG_PRINT
            ("[send: fill buf], head not cached, v %p, vstart %p, length %d, header size %d\n",
             v, vstart, len, iov[0].MPL_IOV_LEN);
        MPIU_Memcpy(vstart, header, iov[0].MPL_IOV_LEN);
#ifndef MV2_DISABLE_HEADER_CACHING 
        if (header->type == MPIDI_CH3_PKT_EAGER_SEND &&
            ((len - sizeof(MPIDI_CH3_Pkt_eager_send_t)) <= MAX_SIZE_WITH_HEADER_CACHING)) {
            MPIU_Memcpy(cached, header, sizeof(MPIDI_CH3_Pkt_eager_send_t));
            ++vc->mrail.rfp.cached_miss;
        }
#endif
        data_buf = (void *) ((unsigned long) vstart + iov[0].MPL_IOV_LEN);
        *num_bytes_ptr += iov[0].MPL_IOV_LEN;
        avail -= iov[0].MPL_IOV_LEN;
        v->pheader = vstart;
    }

    
    /* We have filled the header, it is time to fit in the actual data */
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && n_iov > 1 && is_device_buffer(iov[1].MPL_IOV_BUF)) {
        /* in the case of GPU buffers, there is only one data iov, if data is non-contiguous
         * it should have been packed before this */
        MPIU_Assert(n_iov == 2);

        MPIU_Memcpy_CUDA(data_buf,
                iov[1].MPL_IOV_BUF,
                iov[1].MPL_IOV_LEN,
                cudaMemcpyDeviceToHost);
        *num_bytes_ptr += iov[1].MPL_IOV_LEN;
        avail -= iov[1].MPL_IOV_LEN;

        MPIU_Assert(avail >= 0);
    } else
#endif
    {
        for (i = 1; i < n_iov; i++) {
            if (avail >= iov[i].MPL_IOV_LEN) {
              MPIU_Memcpy(data_buf, iov[i].MPL_IOV_BUF, iov[i].MPL_IOV_LEN);
                data_buf = (void *) ((unsigned long) data_buf + iov[i].MPL_IOV_LEN);
                *num_bytes_ptr += iov[i].MPL_IOV_LEN;
                avail -= iov[i].MPL_IOV_LEN;
            } else if (avail > 0) {
              MPIU_Memcpy(data_buf, iov[i].MPL_IOV_BUF, avail);
                data_buf = (void *) ((unsigned long) data_buf + avail);
                *num_bytes_ptr += avail;
                avail = 0;
                break;
            } else break;
        }
    }

    DEBUG_PRINT("[send: fill buf], num bytes copied %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_FILL_START_BUF);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Fast_rdma_send_complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/* INOUT: num_bytes_ptr holds the pkt_len as input parameter */
static inline int MPIDI_CH3I_MRAILI_Fast_rdma_send_complete(MPIDI_VC_t * vc,
                                              MPL_IOV * iov,
                                              int n_iov,
                                              int *num_bytes_ptr,
                                              vbuf ** vbuf_handle)
{
    int rail;
    int  post_len;
    char cq_overflow = 0;
    VBUF_FLAG_TYPE flag;
    vbuf *v =
        &(vc->mrail.rfp.RDMA_send_buf[vc->mrail.rfp.phead_RDMA_send]);
    char *rstart;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_SEND_COMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_SEND_COMPLETE);

    rail = MRAILI_Send_select_rail(vc);
    MRAILI_Fast_rdma_fill_start_buf(vc, iov, n_iov, num_bytes_ptr);

    post_len = *num_bytes_ptr;
    rstart = vc->mrail.rfp.remote_RDMA_buf +
            (vc->mrail.rfp.phead_RDMA_send * rdma_fp_buffer_size);
    DEBUG_PRINT("[send: rdma_send] local vbuf %p, remote start %p, align size %d\n",
               v, rstart, post_len);

    if (++(vc->mrail.rfp.phead_RDMA_send) >= num_rdma_buffer)
        vc->mrail.rfp.phead_RDMA_send = 0;

    v->rail = rail;
    v->padding = BUSY_FLAG;

    /* requirements for coalescing */
    ++vc->mrail.outstanding_eager_vbufs;
    v->eager = 1;
    v->vc = (void *) vc;

    /* set tail flag with the size of the content */
    if ((int) *(VBUF_FLAG_TYPE *) (v->buffer + post_len) == post_len) {
        flag = (VBUF_FLAG_TYPE) (post_len + FAST_RDMA_ALT_TAG);
    } else {
        flag = (VBUF_FLAG_TYPE) post_len;
    }
    /* set head flag */
    *v->head_flag = (VBUF_FLAG_TYPE) flag;
    /* set tail flag */    
    *((VBUF_FLAG_TYPE *)(v->buffer + post_len)) = flag;

    DEBUG_PRINT("incrementing the outstanding eager vbufs: RFP %d\n", vc->mrail.outstanding_eager_vbufs);

    /* generate a completion, following statements should have been executed during
     * initialization */
    post_len += VBUF_FAST_RDMA_EXTRA_BYTES;

    DEBUG_PRINT("[send: rdma_send] lkey %p, rkey %p, len %d, flag %d\n",
                vc->mrail.rfp.RDMA_send_buf_mr[vc->mrail.rails[rail].hca_index]->lkey,
                vc->mrail.rfp.RDMA_remote_buf_rkey, post_len, *v->head_flag);

    VBUF_SET_RDMA_ADDR_KEY(v, post_len, v->head_flag,
            vc->mrail.rfp.RDMA_send_buf_mr[vc->mrail.rails[rail].hca_index]->lkey, rstart,
            vc->mrail.rfp.RDMA_remote_buf_rkey[vc->mrail.rails[rail].hca_index]);

    XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
    FLUSH_RAIL(vc, rail);
#ifdef CRC_CHECK
    p->crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
                              *v->head_flag - sizeof *p);
#endif

    cq_overflow = check_cq_overflow(vc, rail);

    if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
        --vc->mrail.rails[rail].send_wqes_avail;
        *vbuf_handle = v;

        IBV_POST_SR(v, vc, rail, "ibv_post_sr (post_fast_rdma)");
        DEBUG_PRINT("[send:post rdma] desc posted\n");
    } else {
        DEBUG_PRINT("[send: rdma_send] Warning! no send wqe or send cq available\n");
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        *vbuf_handle = v;
        return MPI_MRAIL_MSG_QUEUED;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_SEND_COMPLETE);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Fast_rdma_ok
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_CH3I_MRAILI_Fast_rdma_ok(MPIDI_VC_t * vc, MPIDI_msg_sz_t len)
{
    int i = 0;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_OK);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_OK);

    if(unlikely(vc->tmp_dpmvc)) {
        return 0;
    }
    
#ifdef _ENABLE_UD_
    if(rdma_enable_hybrid)
    {
            if(unlikely(!(vc->mrail.state & MRAILI_RC_CONNECTED))) {
                    return 0;
            }
    }
#endif /* _ENABLE_UD_ */

    if (unlikely(len > MRAIL_MAX_RDMA_FP_SIZE)) {
        return 0;
    }

    if (unlikely(num_rdma_buffer < 2
        || vc->mrail.rfp.phead_RDMA_send == vc->mrail.rfp.ptail_RDMA_send
        || vc->mrail.rfp.RDMA_send_buf[vc->mrail.rfp.phead_RDMA_send].padding == BUSY_FLAG
        || MRAILI_Coalesce_ok(vc, 0))) /* We can only coalesce with send/recv. */
    {
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_OK);
        return 0;
    }

    if (unlikely(!mv2_MPIDI_CH3I_RDMA_Process.has_srq)) {
        for (i = 0; i < rdma_num_rails; i++)
        {
            if (vc->mrail.srp.credits[i].backlog.len != 0)
            {
                MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_OK);
    	    return 0;
            }
        }
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FAST_RDMA_OK);
    return 1;
} 

#undef FUNCNAME
#define FUNCNAME mv2_post_srq_buffers
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_post_srq_buffers(int num_bufs, int hca_num)
{
    int i = 0;
    vbuf* v = NULL;
    struct ibv_recv_wr* bad_wr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_POST_SRQ_BUFFERS);
    MPIDI_FUNC_ENTER(MPID_STATE_POST_SRQ_BUFFERS);

    if (num_bufs > mv2_srq_fill_size)
    {
        ibv_va_error_abort(
            GEN_ASSERT_ERR,
            "Try to post %d to SRQ, max %d\n",
            num_bufs,
            mv2_srq_fill_size);
    }

    for (; i < num_bufs; ++i)
    {
        if ((v = get_vbuf_by_offset(MV2_RECV_VBUF_POOL_OFFSET)) == NULL)
        {
            break;
        }

        VBUF_INIT_RECV(
            v,
            VBUF_BUFFER_SIZE,
            hca_num * rdma_num_ports * rdma_num_qp_per_port);
            v->transport = IB_TRANSPORT_RC;

        if (ibv_post_srq_recv(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[hca_num], &v->desc.u.rr, &bad_wr))
        {
            MRAILI_Release_vbuf(v);
            break;
        }
    }

    DEBUG_PRINT("Posted %d buffers to SRQ\n",num_bufs);

    MPIDI_FUNC_EXIT(MPID_STATE_POST_SRQ_BUFFERS);
    return i;
}

#ifdef _ENABLE_UD_
#undef FUNCNAME
#define FUNCNAME mv2_post_ud_recv_buffers
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_post_ud_recv_buffers(int num_bufs, mv2_ud_ctx_t *ud_ctx)
{
    int i = 0,ret = 0;
    vbuf* v = NULL;
    struct ibv_recv_wr* bad_wr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_POST_RECV_BUFFERS);
    MPIDI_FUNC_ENTER(MPID_STATE_POST_RECV_BUFFERS);

    if (num_bufs > rdma_default_max_ud_recv_wqe)
    {
        ibv_va_error_abort(
                GEN_ASSERT_ERR,
                "Try to post %d to UD recv buffers, max %d\n",
                num_bufs, rdma_default_max_ud_recv_wqe);
    }

    for (i = 0; i < num_bufs; ++i)
    {
        MV2_GET_AND_INIT_UD_VBUF(v);
        if (v == NULL)
        {
            break;
        }

        vbuf_init_ud_recv(v, rdma_default_ud_mtu, ud_ctx->hca_num);
        v->transport = IB_TRANSPORT_UD;
        if (ud_ctx->qp->srq) {
            ret = ibv_post_srq_recv(ud_ctx->qp->srq, &v->desc.u.rr, &bad_wr);
        } else {
            ret = ibv_post_recv(ud_ctx->qp, &v->desc.u.rr, &bad_wr);
        }
        if (ret)
        {
            MRAILI_Release_vbuf(v);
            break;
        }
    }

    PRINT_DEBUG(DEBUG_UD_verbose>0 ,"Posted %d buffers of size:%d to UD QP on HCA %d\n",
                num_bufs, rdma_default_ud_mtu, ud_ctx->hca_num);

    MPIDI_FUNC_EXIT(MPID_STATE_POST_RECV_BUFFERS);
    return i;
}

#undef FUNCNAME
#undef FUNCNAME
#define FUNCNAME post_hybrid
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int post_hybrid_send(MPIDI_VC_t* vc, vbuf* v, int rail)
{
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    MPIDI_STATE_DECL(MPID_STATE_POST_HYBRID_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_POST_HYBRID_SEND);

    switch (v->transport) {
        case IB_TRANSPORT_UD:
            vc->mrail.rely.total_messages++;
            /* Enable RC conection if total no of msgs on UD channel reachd a
             * threshold and total rc connections less than threshold  
             */
            if (!(vc->mrail.state & (MRAILI_RC_CONNECTED | MRAILI_RC_CONNECTING)) 
                && (rdma_ud_num_msg_limit)
                && (vc->mrail.rely.total_messages > rdma_ud_num_msg_limit)
                && ((mv2_MPIDI_CH3I_RDMA_Process.rc_connections + rdma_hybrid_pending_rc_conn)
                    < rdma_hybrid_max_rc_conn)
                && vc->mrail.rely.ext_window.head == NULL
                && !(vc->state == MPIDI_VC_STATE_LOCAL_CLOSE || vc->state == MPIDI_VC_STATE_CLOSE_ACKED)) {
                /* This is hack to create RC channel usig CM protocol.
                ** Need to handle this by sending REQ/REP on UD channel itself
                */
                vc->ch.state = MPIDI_CH3I_VC_STATE_UNCONNECTED;
#ifdef _ENABLE_XRC_
                if(USE_XRC) {
                    VC_XST_CLR (vc, XF_SEND_IDLE);
                }
#endif
                PRINT_DEBUG(DEBUG_UD_verbose>1, "Connection initiated to :%d\n", vc->pg_rank);
                MV2_HYBRID_SET_RC_CONN_INITIATED(vc);
            } 
            post_ud_send(vc, v, rail, NULL);
            break;
        case IB_TRANSPORT_RC:
            MPIU_Assert(vc->mrail.state & MRAILI_RC_CONNECTED);
            if(proc->has_srq) {
                post_srq_send(vc, v, rail);
            } else {
                post_send(vc, v, rail);
            }
            break;
        default:
            PRINT_DEBUG(DEBUG_UD_verbose>1,"Invalid IB transport protocol\n");
            return -1;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_POST_HYBRID_SEND);
    return 0;
}
#endif /* _ENABLE_UD_ */

#undef FUNCNAME
#define FUNCNAME mv2_eager_fast_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_eager_fast_send(MPIDI_VC_t* vc, const void *buf,
                        MPIDI_msg_sz_t data_sz, int rank, int tag,
                        MPID_Comm *comm, int context_offset, MPID_Request **sreq_p)
{
    int rail = 0;
    int retval = 0;
    vbuf* v = NULL;
    int len = 0;
    void *ptr = NULL;
    MPID_Seqnum_t seqnum;
    MPIDI_CH3_Pkt_t *upkt = NULL;
    MPIDI_CH3_Pkt_eager_send_t *eager_pkt = NULL;

    rail = MRAILI_Send_select_rail(vc);

    /* Get VBUF */
    MRAILI_Get_buffer(vc, v, data_sz+sizeof(MPIDI_CH3_Pkt_eager_send_t));

    /* Point header to start of buffer */
    upkt = (MPIDI_CH3_Pkt_t *) v->buffer;
    eager_pkt = &((*upkt).eager_send);

    /* Create packet header */
    MPIDI_Pkt_init(eager_pkt, MPIDI_CH3_PKT_EAGER_SEND);
    eager_pkt->data_sz                 = data_sz;
    eager_pkt->match.parts.tag         = tag;
    eager_pkt->match.parts.rank        = comm->rank;
    eager_pkt->match.parts.context_id  = comm->context_id + context_offset;

    /* Set sequence number */
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);

    /* Copy data */
    ptr = (void*) v->buffer + sizeof(MPIDI_CH3_Pkt_eager_send_t);

    memcpy(ptr, buf, data_sz);
    /* Compute size of pkt */
    len = sizeof(MPIDI_CH3_Pkt_eager_send_t) + data_sz;

    /* Initialize other vbuf parameters */
    vbuf_init_send(v, len, rail);

    /* Send the packet */
    retval = mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);

    return retval;
}

#undef FUNCNAME
#define FUNCNAME mv2_eager_fast_coalesce_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_eager_fast_coalesce_send(MPIDI_VC_t* vc, const void *buf,
                        MPIDI_msg_sz_t data_sz, int rank, int tag,
                        MPID_Comm *comm, int context_offset, MPID_Request **sreq_p)
{
    int retval = 0;
    vbuf* v = NULL;
    int len = 0;
    void *ptr = NULL;
    MPID_Seqnum_t seqnum;
    MPIDI_CH3_Pkt_t *upkt = NULL;
    MPIDI_CH3_Pkt_eager_send_t *eager_pkt = NULL;

    /* Get VBUF */
    v = MRAILI_Get_Vbuf(vc, data_sz+sizeof(MPIDI_CH3_Pkt_eager_send_t));

    /* Point header to start of buffer */
    upkt = (MPIDI_CH3_Pkt_t *) (v->buffer + v->content_size);
    eager_pkt = &((*upkt).eager_send);

    /* Create packet header */
    MPIDI_Pkt_init(eager_pkt, MPIDI_CH3_PKT_EAGER_SEND);
    eager_pkt->data_sz                 = data_sz;
    eager_pkt->match.parts.tag         = tag;
    eager_pkt->match.parts.rank        = comm->rank;
    eager_pkt->match.parts.context_id  = comm->context_id + context_offset;

    /* Set sequence number */
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);

    /* Copy data */
    ptr = (void*) v->buffer + v->content_size + sizeof(MPIDI_CH3_Pkt_eager_send_t);

    memcpy(ptr, buf, data_sz);
    /* Compute size of pkt */
    len = sizeof(MPIDI_CH3_Pkt_eager_send_t) + data_sz;

    /* Update length */
    v->content_size += len;

    /* send the buffer if we aren't trying to coalesce it */
    if(likely(vc->mrail.coalesce_vbuf != v))  {
        /* Initialize other vbuf parameters */
        vbuf_init_send(v, len, v->rail);
        /* Send the packet */
        retval = mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, v->rail);
    } else {
        MPIDI_CH3I_MRAILI_Pkt_comm_header *p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *)
            (v->buffer + v->content_size - len);

        PACKET_SET_CREDIT(p, vc, v->rail);
#ifdef CRC_CHECK
        p->crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
                                  v->desc.sg_entry.length - sizeof *p);
#endif
        v->vc                = (void *) vc;
        p->rail        = v->rail;
#ifdef _ENABLE_UD_
        if(rdma_enable_hybrid) {
                p->src.rank    = MPIDI_Process.my_pg_rank;
        } else
#endif
        {
                p->src.vc_addr = vc->mrail.remote_vc_addr;
        }
    }

    return retval;
}

#undef FUNCNAME
#define FUNCNAME mv2_eager_fast_rfp_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int mv2_eager_fast_rfp_send(MPIDI_VC_t* vc, const void *buf,
                        MPIDI_msg_sz_t data_sz, int rank, int tag,
                        MPID_Comm *comm, int context_offset, MPID_Request **sreq_p)
{
    /* For short send n_iov is always 2 */
    int n_iov = 2;
    MPID_Seqnum_t seqnum;
    vbuf *buf_handle = NULL;
    int num_bytes_ptr = 0;
    MPL_IOV iov[2];
    MPIDI_CH3_Pkt_t upkt;
    MPIDI_CH3_Pkt_eager_send_t * const eager_pkt = &upkt.eager_send;

    if (unlikely(!MPIDI_CH3I_MRAILI_Fast_rdma_ok(vc, data_sz+sizeof(*eager_pkt)))) {
        return vc->eager_fast_fn(vc, buf, data_sz, rank,
                                tag, comm, context_offset, sreq_p);
    }

    /* Create packet header */
    MPIDI_Pkt_init(eager_pkt, MPIDI_CH3_PKT_EAGER_SEND);
    eager_pkt->data_sz                 = data_sz;
    eager_pkt->match.parts.tag         = tag;
    eager_pkt->match.parts.rank        = comm->rank;
    eager_pkt->match.parts.context_id  = comm->context_id + context_offset;

    /* Create IOV (header) */
    iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)eager_pkt;
    iov[0].MPL_IOV_LEN = sizeof(*eager_pkt);
    /* Create IOV (data) */
    iov[1].MPL_IOV_BUF = (MPL_IOV_BUF_CAST) buf;
    iov[1].MPL_IOV_LEN = data_sz;

    /* Compute size of pkt */
    num_bytes_ptr = iov[0].MPL_IOV_LEN + iov[1].MPL_IOV_LEN;

    /* Set sequence number */
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);

    return MPIDI_CH3I_MRAILI_Fast_rdma_send_complete(vc, iov,
                n_iov, &num_bytes_ptr, &buf_handle);
}

#undef FUNCNAME
#define FUNCNAME post_srq_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int post_srq_send(MPIDI_VC_t* vc, vbuf* v, int rail)
{
    char cq_overflow = 0;
    MPIDI_CH3I_MRAILI_Pkt_comm_header *p = v->pheader;
    PACKET_SET_CREDIT(p, vc, rail);

    MPIDI_STATE_DECL(MPID_STATE_POST_SRQ_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_POST_SRQ_SEND);

    v->vc = (void *) vc;
    p->rail        = rail;
#ifdef _ENABLE_UD_
    if(rdma_enable_hybrid) {
            p->src.rank    = MPIDI_Process.my_pg_rank;
            while (vc->mrail.rails[rail].qp_hndl->state != IBV_QPS_RTS) {
                    MPID_Progress_test();
            }
    } else
#endif
    {
            p->src.vc_addr = vc->mrail.remote_vc_addr;
    }
    MPIU_Assert(v->transport == IB_TRANSPORT_RC);
    
    if (p->type == MPIDI_CH3_PKT_NOOP) {
        v->seqnum = p->seqnum = -1;
    } else {
        v->seqnum = p->seqnum = vc->mrail.seqnum_next_tosend;
        vc->mrail.seqnum_next_tosend++;
    }
    
    p->acknum = vc->mrail.seqnum_next_toack;
    MARK_ACK_COMPLETED(vc);
    
    XRC_FILL_SRQN_FIX_CONN (v, vc, rail);

    FLUSH_RAIL(vc, rail);

    cq_overflow = check_cq_overflow(vc, rail);

    if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
        --vc->mrail.rails[rail].send_wqes_avail;

        IBV_POST_SR(v, vc, rail, "ibv_post_sr (post_send_desc)");
    } else {
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        MPIDI_FUNC_EXIT(MPID_STATE_POST_SRQ_SEND);
        return MPI_MRAIL_MSG_QUEUED;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_POST_SRQ_SEND);
    return 0;
}

#undef FUNCNAME
#define FUNCNAME post_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int post_send(MPIDI_VC_t * vc, vbuf * v, int rail)
{
    char cq_overflow = 0;
    MPIDI_CH3I_MRAILI_Pkt_comm_header *p = v->pheader;

    MPIDI_STATE_DECL(MPID_STATE_POST_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_POST_SEND);
    DEBUG_PRINT(
                "[post send] credit %d,type noop %d, "
                "backlog %d, wqe %d, nb will be %d\n",
                vc->mrail.srp.credits[rail].remote_credit,
                p->type == MPIDI_CH3_PKT_NOOP, 
                vc->mrail.srp.credits[0].backlog.len,
                vc->mrail.rails[rail].send_wqes_avail,
                v->desc.sg_entry.length);

    v->vc = (void *) vc;
    p->rail        = rail;
#ifdef _ENABLE_UD_
    if(rdma_enable_hybrid) {
            p->src.rank = MPIDI_Process.my_pg_rank;
    } else
#endif
    {
            p->src.vc_addr = vc->mrail.remote_vc_addr;
    }

    MPIU_Assert(v->transport == IB_TRANSPORT_RC);
   
    if (p->type == MPIDI_CH3_PKT_NOOP) {
        v->seqnum = p->seqnum = -1;
    } else {
        v->seqnum = p->seqnum = vc->mrail.seqnum_next_tosend;
        vc->mrail.seqnum_next_tosend++;
    }
    p->acknum = vc->mrail.seqnum_next_toack;
    MARK_ACK_COMPLETED(vc);

    PRINT_DEBUG(DEBUG_UD_verbose>1, "sending seqnum:%d acknum:%d\n",p->seqnum,p->acknum);

    if (vc->mrail.srp.credits[rail].remote_credit > 0
        || p->type == MPIDI_CH3_PKT_NOOP) {

        PACKET_SET_CREDIT(p, vc, rail);
#ifdef CRC_CHECK
	p->crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
				  v->desc.sg_entry.length - sizeof *p );
#endif
        if (p->type != MPIDI_CH3_PKT_NOOP)
        {
            --vc->mrail.srp.credits[rail].remote_credit;
        }

        v->vc = (void *) vc;

        XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
        FLUSH_RAIL(vc, rail);

        cq_overflow = check_cq_overflow(vc, rail);

        if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
            --vc->mrail.rails[rail].send_wqes_avail;
            IBV_POST_SR(v, vc, rail, "ibv_post_sr (post_send_desc)");
        } else {
            MRAILI_Ext_sendq_enqueue(vc, rail, v);
            MPIDI_FUNC_EXIT(MPID_STATE_POST_SEND);
            return MPI_MRAIL_MSG_QUEUED;
        }
    }
    else
    {
        ibv_backlog_queue_t *q = &(vc->mrail.srp.credits[rail].backlog);
        BACKLOG_ENQUEUE(q, v);
        MPIDI_FUNC_EXIT(MPID_STATE_POST_SEND);
        return MPI_MRAIL_MSG_QUEUED;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_POST_SEND);
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_Fill_start_buffer
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Fill_start_buffer(vbuf * v,
                             MPL_IOV * iov,
                             int n_iov)
{
    int i = 0;
    int avail = 0;
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        avail = ((vbuf_pool_t*)v->pool_index)->buf_size - v->content_size;
    } else 
#endif
    {
        avail = VBUF_BUFFER_SIZE - v->content_size;
    }
    void *ptr = (v->buffer + v->content_size);
    int len = 0;
#ifdef _ENABLE_UD_
    if( rdma_enable_hybrid && v->transport == IB_TRANSPORT_UD) {
        avail = MRAIL_MAX_UD_SIZE - v->content_size;
    }
#endif

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_FILL_START_BUFFER);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_FILL_START_BUFFER);

    DEBUG_PRINT("buffer: %p, content size: %d\n", v->buffer, v->content_size);

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && n_iov > 1 && is_device_buffer(iov[1].MPL_IOV_BUF)) {
        /* in the case of GPU buffers, there is only one data iov, if data is non-contiguous
         * it should have been packed before this */
        MPIU_Assert(n_iov == 2);

        MPIU_Memcpy(ptr, iov[0].MPL_IOV_BUF,
                (iov[0].MPL_IOV_LEN));
        len += (iov[0].MPL_IOV_LEN);
        avail -= (iov[0].MPL_IOV_LEN);
        ptr = (void *) ((unsigned long) ptr + iov[0].MPL_IOV_LEN);

        if (avail >= iov[1].MPL_IOV_LEN) {
            MPIU_Memcpy_CUDA(ptr,
                    iov[1].MPL_IOV_BUF,
                    iov[1].MPL_IOV_LEN,
                    cudaMemcpyDeviceToHost);
            len += iov[1].MPL_IOV_LEN;
        } else {
            MPIU_Memcpy_CUDA(ptr,
                    iov[1].MPL_IOV_BUF,
                    avail,
                    cudaMemcpyDeviceToHost);
            len += avail;
            avail = 0;
        }
    } else 
#endif
    {
        for (i = 0; i < n_iov; i++) {
            DEBUG_PRINT("[fill buf]avail %d, len %d\n", avail,
                    iov[i].MPL_IOV_LEN);
            if (avail >= iov[i].MPL_IOV_LEN) {
                DEBUG_PRINT("[fill buf] cpy ptr %p\n", ptr);
                MPIU_Memcpy(ptr, iov[i].MPL_IOV_BUF,
                        (iov[i].MPL_IOV_LEN));
                len += (iov[i].MPL_IOV_LEN);
                avail -= (iov[i].MPL_IOV_LEN);
                ptr = (void *) ((unsigned long) ptr + iov[i].MPL_IOV_LEN);
            } else {
                MPIU_Memcpy(ptr, iov[i].MPL_IOV_BUF, avail);
                len += avail;
                avail = 0;
                break;
            }
        }
    }
    v->content_size += len;

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_FILL_START_BUFFER);
    return len;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_Get_Vbuf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline vbuf * MRAILI_Get_Vbuf(MPIDI_VC_t * vc, size_t pkt_len)
{
    int rail = 0;
    vbuf* temp_v = NULL;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_GET_VBUF);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_GET_VBUF);

    if (unlikely(NULL != vc->mrail.coalesce_vbuf)) {
        int coalesc_buf_size = 0;
#if defined(_ENABLE_UD_) || defined(_MCST_SUPPORT_)
        if (!vc->mrail.coalesce_vbuf->pool_index) {
            coalesc_buf_size = MRAIL_MAX_UD_SIZE;
        } else
#endif
        {
            coalesc_buf_size = ((vbuf_pool_t*)vc->mrail.coalesce_vbuf->pool_index)->buf_size;
        }

        if((coalesc_buf_size - vc->mrail.coalesce_vbuf->content_size) 
                >= pkt_len)
        {
            DEBUG_PRINT("returning back a coalesce buffer\n");
            return vc->mrail.coalesce_vbuf;
        } else {
            FLUSH_SQUEUE(vc);
            vc->mrail.coalesce_vbuf = NULL;
            DEBUG_PRINT("Send out the coalesce vbuf\n");
        }
    }

    rail = MRAILI_Send_select_rail(vc);
    /* if there already wasn't a vbuf that could
     * hold our packet we need to allocate a 
     * new one
     */
    if (likely(NULL == temp_v)) {
        /* are we trying to coalesce? If so, place
         * it as the new coalesce vbuf and add it
         * to the extended sendq
         */

        if(unlikely(MRAILI_Coalesce_ok(vc, rail)) &&
            (pkt_len*2 <= DEFAULT_MEDIUM_VBUF_SIZE)) {
            MRAILI_Get_buffer(vc, temp_v, DEFAULT_MEDIUM_VBUF_SIZE);
            vc->mrail.coalesce_vbuf = temp_v;

            temp_v->seqnum = vc->mrail.seqnum_next_tosend;
            vc->mrail.seqnum_next_tosend++;

            temp_v->coalesce = 1;
            temp_v->rail = rail;
            MRAILI_Ext_sendq_enqueue(vc, temp_v->rail, temp_v); 
            DEBUG_PRINT("coalesce is ok\n");

            if(!mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
                --vc->mrail.srp.credits[temp_v->rail].remote_credit;
            }

        } else {
            MRAILI_Get_buffer(vc, temp_v, pkt_len);
            DEBUG_PRINT("coalesce not ok\n");
        }

        DEBUG_PRINT("buffer is %p\n", temp_v->buffer);
        DEBUG_PRINT("pheader buffer is %p\n", temp_v->pheader);

        temp_v->rail = rail;
        temp_v->eager = 1;
        temp_v->content_size = 0;

        DEBUG_PRINT("incrementing the outstanding eager vbufs: eager %d\n",
                vc->mrail.outstanding_eager_vbufs);

        if (temp_v->transport == IB_TRANSPORT_RC)
            ++vc->mrail.outstanding_eager_vbufs;
    }

    MPIU_Assert(temp_v != NULL);

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_GET_VBUF);
    return temp_v;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Eager_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAILI_Eager_send(MPIDI_VC_t * vc,
                                 MPL_IOV * iov,
                                 int n_iov,
                                 size_t pkt_len,
                                 int *num_bytes_ptr,
                                 vbuf **buf_handle)
{
    vbuf * v;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_EAGER_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_EAGER_SEND);

    /* first we check if we can take the RDMA FP */
    if(likely(MPIDI_CH3I_MRAILI_Fast_rdma_ok(vc, pkt_len))) {
    
        *num_bytes_ptr = pkt_len;
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_EAGER_SEND);
        return MPIDI_CH3I_MRAILI_Fast_rdma_send_complete(vc, iov,
                n_iov, num_bytes_ptr, buf_handle);
    } 

    /* otherwise we can always take the send/recv path */
    v = MRAILI_Get_Vbuf(vc, pkt_len);

    DEBUG_PRINT("[eager send]vbuf addr %p, buffer: %p\n", v, v->buffer);
    *num_bytes_ptr = MRAILI_Fill_start_buffer(v, iov, n_iov);
   
#ifdef CKPT
    /* this won't work properly at the moment... 
     *
     * My guess is that if vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE
     * just have Coalesce_ok return 0 -- then you'll always get a new vbuf
     * (actually there are a few other things to change as well...)
     */

    if (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE) {
        /*MPIDI_CH3I_MRAILI_Pkt_comm_header * p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *) v->pheader;*/
        MPIDI_CH3I_CR_msg_log_queue_entry_t *entry;
        if (rdma_use_coalesce) {
            entry = MSG_LOG_QUEUE_TAIL(vc);
            if (entry->buf == v) /*since the vbuf is already filled, no need to queue it again*/
            {
                PRINT_DEBUG(DEBUG_FT_verbose, "coalesced buffer\n");
                return MPI_MRAIL_MSG_QUEUED;
            }
        }
        entry = (MPIDI_CH3I_CR_msg_log_queue_entry_t *) MPIU_Malloc(sizeof(MPIDI_CH3I_CR_msg_log_queue_entry_t));
        entry->buf = v;
        entry->len = *num_bytes_ptr;
        MSG_LOG_ENQUEUE(vc, entry);
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_EAGER_SEND);
        return MPI_MRAIL_MSG_QUEUED;
    }
#endif

    /* send the buffer if we aren't trying to coalesce it */
    if(likely(vc->mrail.coalesce_vbuf != v))  {
        DEBUG_PRINT("[eager send] len %d, selected rail hca %d, rail %d\n",
                *num_bytes_ptr, vc->mrail.rails[v->rail].hca_index, v->rail);
        vbuf_init_send(v, *num_bytes_ptr, v->rail);
        mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, v->rail);
    } else {
        MPIDI_CH3I_MRAILI_Pkt_comm_header *p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *)
            (v->buffer + v->content_size - *num_bytes_ptr);

        PACKET_SET_CREDIT(p, vc, v->rail);
#ifdef CRC_CHECK
	p->crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
                                  v->desc.sg_entry.length - sizeof *p);
#endif
        v->vc                = (void *) vc;
        p->rail        = v->rail;
#ifdef _ENABLE_UD_
	if(rdma_enable_hybrid) {
            p->src.rank    = MPIDI_Process.my_pg_rank;
	} else
#endif
    {
            p->src.vc_addr = vc->mrail.remote_vc_addr;
	}
    }

    *buf_handle = v;

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_EAGER_SEND);
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_rget_finish
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAILI_rget_finish(MPIDI_VC_t * vc,
                                 MPL_IOV * iov,
                                 int n_iov,
                                 int *num_bytes_ptr, vbuf ** buf_handle, 
                                 int rail)
{
    vbuf *v;
    int mpi_errno;
    size_t nbytes = MAX(DEFAULT_MEDIUM_VBUF_SIZE, *num_bytes_ptr);

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_RGET_FINISH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_RGET_FINISH);

    if (likely(nbytes <= DEFAULT_MEDIUM_VBUF_SIZE)) {
        GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_MEDIUM_DATA_VBUF_POOL_OFFSET);
    } else {
        GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_LARGE_DATA_VBUF_POOL_OFFSET);
    }
    *buf_handle = v;
    *num_bytes_ptr = MRAILI_Fill_start_buffer(v, iov, n_iov);

    vbuf_init_send(v, *num_bytes_ptr, rail);

    mpi_errno = mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_RGET_FINISH); 
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_rput_complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAILI_rput_complete(MPIDI_VC_t * vc,
                                 MPL_IOV * iov,
                                 int n_iov,
                                 int *num_bytes_ptr, vbuf ** buf_handle, 
                                 int rail)
{
    vbuf * v;
    int mpi_errno;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_RPUT_COMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_RPUT_COMPLETE);

    MRAILI_Get_buffer(vc, v, iov->MPL_IOV_LEN);
    *buf_handle = v;
    DEBUG_PRINT("[eager send]vbuf addr %p\n", v);
    *num_bytes_ptr = MRAILI_Fill_start_buffer(v, iov, n_iov);

    DEBUG_PRINT("[eager send] len %d, selected rail hca %d, rail %d\n",
                *num_bytes_ptr, vc->mrail.rails[rail].hca_index, rail);

    vbuf_init_send(v, *num_bytes_ptr, rail);

    mpi_errno = mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_RPUT_COMPLETE);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_Backlog_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Backlog_send(MPIDI_VC_t * vc, int rail)
{
    char cq_overflow = 0;
    ibv_backlog_queue_t *q;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_BACKLOG_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_BACKLOG_SEND);

    q = &vc->mrail.srp.credits[rail].backlog;

#ifdef CKPT
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        PRINT_ERROR("[%s, %d] CKPT has_srq error\n", __FILE__, __LINE__  );
        exit(EXIT_FAILURE);
    }
#endif

    while ((q->len > 0)
           && (vc->mrail.srp.credits[rail].remote_credit > 0)) {
        vbuf *v = NULL;
        MPIDI_CH3I_MRAILI_Pkt_comm_header *p;
        MPIU_Assert(q->vbuf_head != NULL);
        BACKLOG_DEQUEUE(q, v);

        /* Assumes packet header is at beginning of packet structure */
        p = (MPIDI_CH3I_MRAILI_Pkt_comm_header *) v->pheader;

        PACKET_SET_CREDIT(p, vc, rail);
#ifdef CRC_CHECK
	p->mrail.crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
                                  v->desc.sg_entry.length - sizeof *p);
#endif
        --vc->mrail.srp.credits[rail].remote_credit;

        if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
#ifdef _ENABLE_UD_
		if(rdma_enable_hybrid) {
                p->src.rank    = MPIDI_Process.my_pg_rank;
		} else
#endif
        {
                p->src.vc_addr = vc->mrail.remote_vc_addr;
		}
            p->rail        = rail;
        }

     	v->vc = vc;
        v->rail = rail;

        XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
        FLUSH_RAIL(vc, rail);
 
        cq_overflow = check_cq_overflow(vc, rail);

        if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
            --vc->mrail.rails[rail].send_wqes_avail;

            IBV_POST_SR(v, vc, rail,
                        "ibv_post_sr (MRAILI_Backlog_send)");
        } else {
            MRAILI_Ext_sendq_enqueue(vc, rail, v);
            continue;
        }
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_BACKLOG_SEND);
    return 0;
}


#undef FUNCNAME
#define FUNCNAME MRAILI_Flush_wqe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Flush_wqe(MPIDI_VC_t *vc, vbuf *v , int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_FLUSH_WQE);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_FLUSH_WQE);
    FLUSH_RAIL(vc, rail);
    if (!vc->mrail.rails[rail].send_wqes_avail)
    {
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_FLUSH_WQE);
        return MPI_MRAIL_MSG_QUEUED;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_FLUSH_WQE);
    return 0;
}
#undef FUNCNAME
#define FUNCNAME MRAILI_Process_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Process_send(void *vbuf_addr)
{
    int mpi_errno = MPI_SUCCESS;

    vbuf            *v = vbuf_addr;
    MPIDI_CH3I_MRAILI_Pkt_comm_header *p;
    MPIDI_VC_t      *vc;
    MPIDI_VC_t      *orig_vc;
    MPID_Request    *req;
    double          time_taken;
    int             complete;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_PROCESS_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_PROCESS_SEND);

    vc  = v->vc;
    p = v->pheader;
#ifdef _ENABLE_XRC_
    if (USE_XRC && VC_XST_ISSET (vc, XF_INDIRECT_CONN)) {
        orig_vc = vc->ch.orig_vc;
    }
    else 
#endif
    {
        orig_vc = vc;
    }
    if (v->transport == IB_TRANSPORT_RC) {
        if (v->padding == RDMA_ONE_SIDED) {
            ++(orig_vc->mrail.rails[v->rail].send_wqes_avail);
            if (orig_vc->mrail.rails[v->rail].ext_sendq_head) {
                MRAILI_Ext_sendq_send(orig_vc, v->rail);
            }

            if ((mpi_errno = MRAILI_Handle_one_sided_completions(v)) != MPI_SUCCESS)
            {
                MPIR_ERR_POP(mpi_errno);
            }

            MRAILI_Release_vbuf(v);
            goto fn_exit;
        }

    
        ++orig_vc->mrail.rails[v->rail].send_wqes_avail;


        if(vc->free_vc) {
            if(vc->mrail.rails[v->rail].send_wqes_avail == rdma_default_max_send_wqe)   {
                if (v->padding == NORMAL_VBUF_FLAG) {
                    DEBUG_PRINT("[process send] normal flag, free vbuf\n");
                    MRAILI_Release_vbuf(v);
                } else {
                    v->padding = FREE_FLAG;
                }

                MPIU_Memset(vc, 0, sizeof(MPIDI_VC_t));
                MPIU_Free(vc); 
                mpi_errno = MPI_SUCCESS;
                goto fn_exit;
            }
        }

        if(v->eager) {
            --vc->mrail.outstanding_eager_vbufs;
            DEBUG_PRINT("Eager, decrementing to: %d\n", vc->mrail.outstanding_eager_vbufs);

            if(vc->mrail.outstanding_eager_vbufs < 
                    rdma_coalesce_threshold) {
                DEBUG_PRINT("Flushing coalesced\n", v);
                FLUSH_SQUEUE(vc);
            }
            v->eager = 0;
        }
 
        if (orig_vc->mrail.rails[v->rail].ext_sendq_head) {
            MRAILI_Ext_sendq_send(orig_vc, v->rail);
        }

        if(v->padding == COLL_VBUF_FLAG) { 
            MRAILI_Release_vbuf(v);
            goto fn_exit;
        } 

        if (v->padding == RPUT_VBUF_FLAG) {

            req = (MPID_Request *)v->sreq;

            PRINT_DEBUG(DEBUG_RNDV_verbose, "Processing RPUT completion "
                    "req: %p, protocol: %d, local: %d, remote: %d\n",
                    req, req->mrail.protocol, req->mrail.local_complete, req->mrail.remote_complete);

            /* HSAM is Activated */
            if (mv2_MPIDI_CH3I_RDMA_Process.has_hsam) {
                req = (MPID_Request *)v->sreq;
                MPIU_Assert(req != NULL);
                get_wall_time(&time_taken);
                req->mrail.stripe_finish_time[v->rail] = 
                    time_taken;
            }

#ifdef _ENABLE_CUDA_
            if (rdma_enable_cuda
                && v->orig_vbuf != NULL) {
                vbuf *orig_vbuf = (vbuf *) (v->orig_vbuf);
                orig_vbuf->finish_count++;
                if (orig_vbuf->finish_count == rdma_num_rails) {
                    MRAILI_Release_vbuf(orig_vbuf);
                }
            }
#endif
            MRAILI_Release_vbuf(v);
            goto fn_exit;
        }
        if (v->padding == RGET_VBUF_FLAG) {

            req = (MPID_Request *)v->sreq;

            /* HSAM is Activated */
            if (mv2_MPIDI_CH3I_RDMA_Process.has_hsam) {
                MPIU_Assert(req != NULL);
                get_wall_time(&time_taken);
                /* Record the time only the first time a data transfer
                 * is scheduled on this rail
                 * this may occur for very large size messages */

                /* SS: The value in measuring time is a double.
                 * As long as it is below some epsilon value, it
                 * can be considered same as zero */
                if(req->mrail.stripe_finish_time[v->rail] < ERROR_EPSILON) {
                    req->mrail.stripe_finish_time[v->rail] = 
                        time_taken;
                }
            }

            ++req->mrail.local_complete;
            PRINT_DEBUG(DEBUG_RNDV_verbose, "Processing RGET completion "
                    "req: %p, protocol: %d, local: %d, remote: %d\n",
                    req, req->mrail.protocol, req->mrail.local_complete, req->mrail.remote_complete);

            /* If the message size if less than the striping threshold, send a
             * finish message immediately
             *
             * If HSAM is defined, wait for rdma_num_rails / stripe_factor
             * number of completions before sending the finish message.
             * After sending the finish message, adjust the weights of different
             * paths
             *
             * If HSAM is not defined, wait for rdma_num_rails completions
             * before sending the finish message
             */

            if(req->mrail.rndv_buf_sz > rdma_large_msg_rail_sharing_threshold) {
                if(mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
                        (req->mrail.local_complete == 
                         req->mrail.num_rdma_read_completions )) { 

                    MRAILI_RDMA_Get_finish(vc, 
                            (MPID_Request *) v->sreq, v->rail);

                    adjust_weights(v->vc, req->mrail.stripe_start_time,
                            req->mrail.stripe_finish_time, 
                            req->mrail.initial_weight);                       

                } else if (!mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
                        (req->mrail.local_complete == 
                         req->mrail.num_rdma_read_completions)) {

                    MRAILI_RDMA_Get_finish(vc,
                            (MPID_Request *) v->sreq, v->rail);
                }
            } else {
                MRAILI_RDMA_Get_finish(vc,
                        (MPID_Request *) v->sreq, v->rail);
            }

            MRAILI_Release_vbuf(v);
            goto fn_exit;
        }
        if (v->padding == CREDIT_VBUF_FLAG) {
            PRINT_DEBUG(DEBUG_XRC_verbose>0, "CREDIT Vbuf\n");
            --orig_vc->mrail.rails[v->rail].send_wqes_avail;
            goto fn_exit;
        }
    }
    
    switch (p->type) {
#ifdef CKPT
    case MPIDI_CH3_PKT_CM_SUSPEND:
    case MPIDI_CH3_PKT_CM_REACTIVATION_DONE:
        MPIDI_CH3I_CM_Handle_send_completion(vc, p->type,v);
        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        }
        break;
    case MPIDI_CH3_PKT_CR_REMOTE_UPDATE:
        MPIDI_CH3I_CR_Handle_send_completion(vc, p->type,v);
        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        }
        break;
#endif        
#ifndef MV2_DISABLE_HEADER_CACHING 
    case MPIDI_CH3_PKT_FAST_EAGER_SEND:
    case MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ:
#endif
#if defined(USE_EAGER_SHORT)
    case MPIDI_CH3_PKT_EAGERSHORT_SEND:
#endif /* defined(USE_EAGER_SHORT) */
    case MPIDI_CH3_PKT_EAGER_SEND:
    case MPIDI_CH3_PKT_EAGER_SYNC_SEND: 
    case MPIDI_CH3_PKT_PACKETIZED_SEND_DATA:
    case MPIDI_CH3_PKT_RNDV_R3_DATA:
    case MPIDI_CH3_PKT_READY_SEND:
    case MPIDI_CH3_PKT_PUT:
    case MPIDI_CH3_PKT_PUT_IMMED:
    case MPIDI_CH3_PKT_ACCUMULATE:
    case MPIDI_CH3_PKT_ACCUMULATE_IMMED:
        req = v->sreq;
        v->sreq = NULL;
        DEBUG_PRINT("[process send] complete for eager msg, req %p\n",
                    req);
        if (req != NULL) {
            MPIDI_CH3U_Handle_send_req(vc, req, &complete);

            DEBUG_PRINT("[process send] req not null\n");
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }
        }
        if (v->padding == NORMAL_VBUF_FLAG) {
            DEBUG_PRINT("[process send] normal flag, free vbuf\n");
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
    case MPIDI_CH3_PKT_RPUT_FINISH:
        req = (MPID_Request *) (v->sreq);
        if (req == NULL) {
            ibv_va_error_abort(GEN_EXIT_ERR,
                    "s == NULL, s is the send, v is %p "
                    "handler of the rput finish", v);
        }

#ifdef _ENABLE_CUDA_
        int process_rput_finish = 0;
        MPIDI_CH3_Pkt_rput_finish_t *rput_pkt = 
                        (MPIDI_CH3_Pkt_rput_finish_t *) v->buffer;
        if (rdma_enable_cuda) {
            if (req->mrail.cuda_transfer_mode == NONE 
                        || rput_pkt->cuda_pipeline_finish) {
                process_rput_finish = 1;
            }
        }
        if (!rdma_enable_cuda || process_rput_finish)
#endif
        {

        ++req->mrail.local_complete;
        if (req->mrail.local_complete == rdma_num_rails) {
            req->mrail.local_complete = UINT32_MAX;
        }
        PRINT_DEBUG(DEBUG_RNDV_verbose, "Processing RPUT FIN completion "
                "req: %p, protocol: %d, local: %d, remote: %d\n",
                req, req->mrail.protocol, req->mrail.local_complete, req->mrail.remote_complete);

        if(MPIDI_CH3I_MRAIL_Finish_request(req)) {

            if (req->mrail.d_entry != NULL) {
                dreg_unregister(req->mrail.d_entry);
                req->mrail.d_entry = NULL;
            }

            if(mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
               ((req->mrail.rndv_buf_sz > rdma_large_msg_rail_sharing_threshold))) {

                /* Adjust the weights of different paths according to the
                 * timings obtained for the stripes */

                adjust_weights(v->vc, req->mrail.stripe_start_time,
                        req->mrail.stripe_finish_time, 
                        req->mrail.initial_weight);
            }
            
            MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);        
            req->mrail.d_entry = NULL;
            MPIDI_CH3U_Handle_send_req(vc, req, &complete);

            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, 
                        "Get incomplete eager send request\n");
            }
        }
        }

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
    case MPIDI_CH3_PKT_GET_RESP:
    case MPIDI_CH3_PKT_GET_RESP_IMMED:
        DEBUG_PRINT("[process send] get get respond finish\n");
        req = (MPID_Request *) (v->sreq);
        v->sreq = NULL;
        if (NULL != req) {
            if (MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol) {
                if (req->mrail.d_entry != NULL) {
                    dreg_unregister(req->mrail.d_entry);
                    req->mrail.d_entry = NULL;
                }
                MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);
                req->mrail.d_entry = NULL;
            }

            MPIDI_CH3U_Handle_send_req(vc, req, &complete);
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }
        }

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;

    case MPIDI_CH3_PKT_RGET_FINISH:

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }

        break;
#if defined(_MCST_SUPPORT_)
    case MPIDI_CH3_PKT_MCST:
    case MPIDI_CH3_PKT_MCST_INIT:
        PRINT_DEBUG(DEBUG_MCST_verbose > 4, 
                "mcast send completion\n");
        mcast_ctx->ud_ctx->send_wqes_avail++;
        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
    case MPIDI_CH3_PKT_MCST_NACK:
        if (mcast_use_mcast_nack) {
            mcast_ctx->ud_ctx->send_wqes_avail++;
        }
    case MPIDI_CH3_PKT_MCST_INIT_ACK:
        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
    
#endif
    case MPIDI_CH3_PKT_NOOP:
    case MPIDI_CH3_PKT_ADDRESS:
    case MPIDI_CH3_PKT_ADDRESS_REPLY:
    case MPIDI_CH3_PKT_CM_ESTABLISH:
    case MPIDI_CH3_PKT_PACKETIZED_SEND_START:
    case MPIDI_CH3_PKT_RNDV_REQ_TO_SEND:
    case MPIDI_CH3_PKT_RNDV_READY_REQ_TO_SEND:
    case MPIDI_CH3_PKT_RNDV_CLR_TO_SEND:
    case MPIDI_CH3_PKT_EAGER_SYNC_ACK:
    case MPIDI_CH3_PKT_CANCEL_SEND_REQ:
    case MPIDI_CH3_PKT_CANCEL_SEND_RESP:
    case MPIDI_CH3_PKT_PUT_RNDV:
    case MPIDI_CH3_PKT_RMA_RNDV_CLR_TO_SEND:
    case MPIDI_CH3_PKT_CUDA_CTS_CONTI:
    case MPIDI_CH3_PKT_GET:
    case MPIDI_CH3_PKT_GET_RNDV:
    case MPIDI_CH3_PKT_ACCUMULATE_RNDV:
    case MPIDI_CH3_PKT_GET_ACCUM:
    case MPIDI_CH3_PKT_LOCK:
    case MPIDI_CH3_PKT_LOCK_ACK:
    case MPIDI_CH3_PKT_LOCK_OP_ACK:
    case MPIDI_CH3_PKT_UNLOCK:
    case MPIDI_CH3_PKT_FLUSH:
    case MPIDI_CH3_PKT_ACK:
    case MPIDI_CH3_PKT_DECR_AT_COUNTER:
    case MPIDI_CH3_PKT_FOP:
    case MPIDI_CH3_PKT_FOP_RESP:
    case MPIDI_CH3_PKT_FOP_RESP_IMMED:
    case MPIDI_CH3_PKT_FOP_IMMED:
    case MPIDI_CH3_PKT_CAS_IMMED:
    case MPIDI_CH3_PKT_CAS_RESP_IMMED:
    case MPIDI_CH3_PKT_GET_ACCUM_RNDV:
    case MPIDI_CH3_PKT_GET_ACCUM_IMMED:
    case MPIDI_CH3_PKT_GET_ACCUM_RESP_IMMED:
    case MPIDI_CH3_PKT_FLOW_CNTL_UPDATE:
    case MPIDI_CH3_PKT_RNDV_R3_ACK:
    case MPIDI_CH3_PKT_ZCOPY_FINISH:
    case MPIDI_CH3_PKT_ZCOPY_ACK:
        DEBUG_PRINT("[process send] get %d\n", p->type);
        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        }
        else v->padding = FREE_FLAG;
        break;
   case MPIDI_CH3_PKT_GET_ACCUM_RESP:
        req = v->sreq;
        v->sreq = NULL;
        if (NULL != req) {
            MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(req);

            MPIDI_CH3U_Handle_send_req(vc, req, &complete);
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }
        }

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
   case MPIDI_CH3_PKT_CLOSE:  /*24*/
        DEBUG_PRINT("[process send] get %d\n", p->type);
        vc->pending_close_ops -= 1;
        if (vc->disconnect == 1 && vc->pending_close_ops == 0)
        {
            mpi_errno = MPIDI_CH3_Connection_terminate(vc);
            if(mpi_errno)
            {
              MPIR_ERR_POP(mpi_errno);
            }
        }

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        }
        else {
            v->padding = FREE_FLAG;
        }
        break;
    default:
        dump_vbuf("unknown packet (send finished)", v);
        ibv_va_error_abort(IBV_STATUS_ERR,
                         "Unknown packet type %d in "
                         "MRAILI_Process_send MPIDI_CH3_PKT_FOP: %d", p->type, MPIDI_CH3_PKT_FOP);
    }
    DEBUG_PRINT("return from process send\n");

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_PROCESS_SEND);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
#undef FUNCNAME
#define FUNCNAME MRAILI_Send_noop
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MRAILI_Send_noop(MPIDI_VC_t * c, int rail)
{
    /* always send a noop when it is needed even if there is a backlog.
     * noops do not consume credits.
     * this is necessary to avoid credit deadlock.
     * RNR NAK will protect us if receiver is low on buffers.
     * by doing this we can force a noop ahead of any other queued packets.
     */

    vbuf* v = get_vbuf_by_offset(MV2_RECV_VBUF_POOL_OFFSET);

    MPIDI_CH3I_MRAILI_Pkt_noop* p = (MPIDI_CH3I_MRAILI_Pkt_noop *) v->pheader;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_SEND_NOOP);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_SEND_NOOP);

    p->type = MPIDI_CH3_PKT_NOOP;
    vbuf_init_send(v, sizeof(MPIDI_CH3I_MRAILI_Pkt_noop), rail);
    mv2_MPIDI_CH3I_RDMA_Process.post_send(c, v, rail);
    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_SEND_NOOP);
}

#undef FUNCNAME
#define FUNCNAME MRAILI_Send_noop_if_needed
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Send_noop_if_needed(MPIDI_VC_t * vc, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_SEND_NOOP_IF_NEEDED);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_SEND_NOOP_IF_NEEDED);

    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq
     || vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE)
	return MPI_SUCCESS;

    DEBUG_PRINT( "[ibv_send]local credit %d, rdma redit %d\n",
        vc->mrail.srp.credits[rail].local_credit,
        vc->mrail.rfp.rdma_credit);

    if (vc->mrail.srp.credits[rail].local_credit >=
        rdma_dynamic_credit_threshold
        || vc->mrail.rfp.rdma_credit > num_rdma_buffer / 2
        || (vc->mrail.srp.credits[rail].remote_cc <=
            rdma_credit_preserve
            && vc->mrail.srp.credits[rail].local_credit >=
            rdma_credit_notify_threshold)
        ) {
        MRAILI_Send_noop(vc, rail);
    } 
    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_SEND_NOOP_IF_NEEDED);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_RDMA_Get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MRAILI_RDMA_Get(   MPIDI_VC_t * vc, vbuf *v,
                        char * local_addr, uint32_t lkey,
                        char * remote_addr, uint32_t rkey,
                        int nbytes, int rail
                    )
{
    char cq_overflow = 0;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_RDMA_GET);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_RDMA_GET);

    DEBUG_PRINT("MRAILI_RDMA_Get: RDMA Read, "
            "remote addr %p, rkey %p, nbytes %d, hca %d\n",
            remote_addr, rkey, nbytes, vc->mrail.rails[rail].hca_index);

    vbuf_init_rget(v, (void *)local_addr, lkey,
                   remote_addr, rkey, nbytes, rail);
    
    v->vc = (void *)vc;

    XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
    
    cq_overflow = check_cq_overflow(vc, rail);

    if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
        --vc->mrail.rails[rail].send_wqes_avail;
        IBV_POST_SR(v, vc, rail, "MRAILI_RDMA_Get");
    } else {
        MRAILI_Ext_sendq_enqueue(vc,rail, v);
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_RDMA_GET);
    return;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_RDMA_Put
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MRAILI_RDMA_Put(   MPIDI_VC_t * vc, vbuf *v,
                        char * local_addr, uint32_t lkey,
                        char * remote_addr, uint32_t rkey,
                        int nbytes, int rail
                    )
{
    char cq_overflow = 0;

    MPIDI_STATE_DECL(MPID_STATE_MRAILI_RDMA_PUT);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_RDMA_PUT);

    DEBUG_PRINT("MRAILI_RDMA_Put: RDMA write, "
            "remote addr %p, rkey %p, nbytes %d, hca %d\n",
            remote_addr, rkey, nbytes, vc->mrail.rails[rail].hca_index);

    vbuf_init_rput(v, (void *)local_addr, lkey,
                   remote_addr, rkey, nbytes, rail);
    
    v->vc = (void *)vc;
    XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
 
    cq_overflow = check_cq_overflow(vc, rail);

    if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
        --vc->mrail.rails[rail].send_wqes_avail;
        IBV_POST_SR(v, vc, rail, "MRAILI_RDMA_Put");
    } else {
        MRAILI_Ext_sendq_enqueue(vc,rail, v);
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_RDMA_PUT);
    return;
}


void vbuf_address_send(MPIDI_VC_t *vc)
{
    int rail, i;

    vbuf* v = NULL;
    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    MPIDI_CH3_Pkt_address_t* p = (MPIDI_CH3_Pkt_address_t *) v->pheader;

    rail = MRAILI_Send_select_rail(vc);
    p->type = MPIDI_CH3_PKT_ADDRESS;
    p->rdma_address = (unsigned long)vc->mrail.rfp.RDMA_recv_buf_DMA;

    for (i = 0; i < rdma_num_hcas; i++) {    
	DEBUG_PRINT("mr %p\n", vc->mrail.rfp.RDMA_recv_buf_mr[i]);
	p->rdma_hndl[i]   = vc->mrail.rfp.RDMA_recv_buf_mr[i]->rkey;
    }
    vbuf_init_send(v, sizeof(MPIDI_CH3_Pkt_address_t), rail);
    mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);
}

void vbuf_address_reply_send(MPIDI_VC_t *vc, uint8_t data)
{
    int rail;

    vbuf *v = NULL;
    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    MPIDI_CH3_Pkt_address_reply_t *p = (MPIDI_CH3_Pkt_address_reply_t *) v->pheader;

    rail = MRAILI_Send_select_rail(vc);
    p->type = MPIDI_CH3_PKT_ADDRESS_REPLY;
    p->reply_data = data;
    
    vbuf_init_send(v, sizeof(MPIDI_CH3_Pkt_address_reply_t), rail);
    mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);
}


int mv2_shm_coll_post_send(vbuf *v, int rail, MPIDI_VC_t * vc)
{ 
    char cq_overflow = 0;
    int mpi_errno = MPI_SUCCESS;

    v->rail = rail; 

    cq_overflow = check_cq_overflow(vc, rail);

    if (likely(vc->mrail.rails[rail].send_wqes_avail > 0 && !cq_overflow)) {
        --vc->mrail.rails[rail].send_wqes_avail;

        IBV_POST_SR(v, vc, rail, "ibv_post_sr (post_fast_rdma)");
        DEBUG_PRINT("[send:post rdma] desc posted\n");
    } else {
        DEBUG_PRINT("[send: rdma_send] Warning! no send wqe or send cq available\n");
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        mpi_errno = MPI_MRAIL_MSG_QUEUED;
    }

    return mpi_errno; 
}

void mv2_shm_coll_prepare_post_send(uint64_t local_rdma_addr, uint64_t remote_rdma_addr, 
                      uint32_t local_rdma_key, uint32_t remote_rdma_key, 
                      int len, int rail, MPIDI_VC_t * vc)
{
    vbuf *v=NULL;
    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    if (likely(len <= rdma_max_inline_size)) {
        v->desc.u.sr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    } else {
        v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    }
    (v)->desc.u.sr.wr.rdma.remote_addr = (uintptr_t) (remote_rdma_addr);
    (v)->desc.u.sr.wr.rdma.rkey = (remote_rdma_key);
    v->desc.u.sr.wr_id = (uintptr_t) v;
    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    (v)->desc.sg_entry.length = len;

    (v)->desc.sg_entry.lkey = (local_rdma_key);
    (v)->desc.sg_entry.addr =  (uintptr_t) (local_rdma_addr);
    (v)->padding = COLL_VBUF_FLAG;
    (v)->vc   = vc;
    XRC_FILL_SRQN_FIX_CONN (v, vc, rail);
    mv2_shm_coll_post_send(v, rail, vc);

    return;
}

int mv2_shm_coll_reg_buffer(void *buffer, int size, struct ibv_mr *mem_handle[], 
                           int *buffer_registered)
{
   int i=0;
   int mpi_errno = MPI_SUCCESS;

    for ( i = 0 ; i < rdma_num_hcas; i ++ ) {
        mem_handle[i]  = (struct ibv_mr *) register_memory(buffer, size, i);

        if (!mem_handle[i]) {
            /* de-register already registered with other hcas*/
            for (i = i-1; i >=0 ; --i)
            {
                if (mem_handle[i] != NULL) {
                    deregister_memory(mem_handle[i]);
                }
            }
            *buffer_registered = 0;
        }
    }
    *buffer_registered = 1;

    return mpi_errno;
}

int mv2_shm_coll_dereg_buffer(struct ibv_mr *mem_handle[])
{ 
   int i=0, mpi_errno = MPI_SUCCESS;
   for ( i = 0 ; i < rdma_num_hcas; i ++ ) {
       if (mem_handle[i] != NULL) {
           if (deregister_memory(mem_handle[i])) { 
               ibv_error_abort(IBV_RETURN_ERR,
                                        "deregistration failed\n");
           }
           mem_handle[i] = NULL;
       }
   }
   return mpi_errno; 
}
