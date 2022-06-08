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
#include "rdma_impl.h"
#include "ibv_impl.h"
#include "vbuf.h"
#include "mv2_ud.h"
#include "dreg.h"
#include "mpiutil.h"

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

#ifdef _ENABLE_UD_
#define MV2_GET_RNDV_QP(_rqp, _proc)                            \
do {                                                            \
    _rqp = (_proc)->zcopy_info.rndv_qp_pool_free_head;          \
    (_proc)->zcopy_info.rndv_qp_pool_free_head =                \
        ((_proc)->zcopy_info.rndv_qp_pool_free_head)->next;     \
    _rqp->next = NULL;                                          \
} while(0)

#define MV2_RELEASE_RNDV_QP(_rqp, _proc)                        \
do {                                                            \
    if ((_proc)->zcopy_info.rndv_qp_pool_free_head != NULL) {   \
        _rqp->next = (_proc)->zcopy_info.rndv_qp_pool_free_head;\
    } else {                                                    \
        _rqp->next = NULL;                                      \
    }                                                           \
    (_proc)->zcopy_info.rndv_qp_pool_free_head = _rqp;          \
} while(0)

#define MV2_RNDV_ZCOPY_PUSH_SEGMENT (32)

#undef FUNCNAME 
#define FUNCNAME MPIDI_CH3_Rendezvous_zcopy_resend_cts
#undef FCNAME       
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_CH3_Rendezvous_zcopy_resend_cts(MPIDI_VC_t * vc, 
                            MPID_Request *rreq, mv2_rndv_qp_t *rqp)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_CH3_Pkt_t upkt;
    MPIDI_CH3_Pkt_rndv_clr_to_send_t *cts_pkt = &upkt.rndv_clr_to_send;
    MPID_Request *cts_req;


    MPIDI_Pkt_init(cts_pkt, MPIDI_CH3_PKT_RNDV_CLR_TO_SEND);
    cts_pkt->sender_req_id = rreq->dev.sender_req_id;
    cts_pkt->receiver_req_id = rreq->handle;

    if (rreq->dev.OnDataAvail == NULL)
    {
        cts_pkt->recv_sz = rreq->dev.iov[0].MPL_IOV_LEN;
        if (rreq->dev.iov_count > 1)
        {
            int k = 1;
            for (; k < rreq->dev.iov_count; ++k)
            {
                cts_pkt->recv_sz += rreq->dev.iov[k].MPL_IOV_LEN;
            }
        }
    }
    else
    {
        cts_pkt->recv_sz = rreq->dev.segment_size;
    }

    MPIDI_CH3I_MRAIL_SET_PKT_RNDV(cts_pkt, rreq);
    MPIU_Assert(cts_pkt->rndv.protocol == MV2_RNDV_PROTOCOL_UD_ZCOPY);

    cts_pkt->rndv.rndv_qpn = rqp->ud_qp->qp_num;
    cts_pkt->rndv.hca_index = rqp->hca_num;

    mpi_errno = MPIDI_CH3_iStartMsg(vc, cts_pkt, sizeof(*cts_pkt), &cts_req);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,
                "**ch3|ctspkt");
    }
    if (cts_req != NULL) {
        MPID_Request_release(cts_req);
    }
fn_fail:
    return mpi_errno;
}

static inline void MRAILI_Rndv_send_zcopy_finish(MPIDI_VC_t * vc,  
                    MPID_Request * sreq, mv2_ud_zcopy_info_t *zcopy_info)
{
    int hca_index = sreq->mrail.hca_index;
    vbuf *v;
    MPIDI_CH3_Pkt_zcopy_finish_t *zcopy_finish;
    
    MV2_GET_AND_INIT_UD_VBUF(v);
    zcopy_finish = v->pheader;
    
    PRINT_DEBUG(DEBUG_ZCY_verbose>1, "sending zcopy finish message to:%d\n", vc->pg_rank);

    MPIDI_Pkt_init(zcopy_finish, MPIDI_CH3_PKT_ZCOPY_FINISH);
    zcopy_finish->receiver_req_id = sreq->mrail.partner_id;
    zcopy_finish->hca_index = hca_index;

    v->flags |= UD_VBUF_RETRY_ALWAYS;

    vbuf_init_send(v, sizeof(MPIDI_CH3_Pkt_zcopy_finish_t), hca_index);
    /* need to send on same UD qp on which zcopy data transfered */
    vc->mrail.rely.total_messages++;
    post_ud_send(vc, v, hca_index, zcopy_info->rndv_ud_qps[hca_index]);
}

static inline void MRAILI_Rndv_send_zcopy_ack(MPIDI_VC_t * vc,  MPID_Request * rreq)
{
    int hca_index;
    vbuf *v;
    MPIDI_CH3_Pkt_zcopy_ack_t *zcopy_ack;
    
    MV2_GET_AND_INIT_UD_VBUF(v);
    zcopy_ack = v->pheader;
    hca_index = ((mv2_rndv_qp_t *) rreq->mrail.rndv_qp_entry)->hca_num;
    
    PRINT_DEBUG(DEBUG_ZCY_verbose>1, "sending zcopy ack message to:%d\n", vc->pg_rank);
    MPIDI_Pkt_init(zcopy_ack, MPIDI_CH3_PKT_ZCOPY_ACK);
    zcopy_ack->sender_req_id = rreq->dev.sender_req_id;
    
    vbuf_init_send(v, sizeof(MPIDI_CH3_Pkt_zcopy_ack_t), hca_index);
    /* need to send on same UD qp on which zcopy data transfered */
    vc->mrail.rely.total_messages++;
    post_ud_send(vc, v, hca_index, NULL);
    
}

static inline void mv2_flush_zcopy_rndv_qp(mv2_rndv_qp_t *rqp, int num_to_flush)
{
    int ne, count;
    struct ibv_qp_attr qp_attr;
    struct ibv_wc wc;
   
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_ERR;

    /* Transition to error state to flush remaining buffers */
    if(ibv_modify_qp(rqp->ud_qp, &qp_attr, IBV_QP_STATE)) {
        ibv_error_abort(IBV_RETURN_ERR, "Error in changing QP state to err\n");
    }
        
    /* Flush remaining buffers */
    count = 0;
    do {
        ne = ibv_poll_cq(rqp->ud_cq, 1, &wc);
        count += ne;
    } while(count < num_to_flush);

    
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RESET;

    if(ibv_modify_qp(rqp->ud_qp, &qp_attr, IBV_QP_STATE)) {
        ibv_error_abort(IBV_RETURN_ERR, "Error in changing QP state to err\n");
    }

    /* transiotion to RTS state */
    if(mv2_ud_qp_transition(rqp->ud_qp, rqp->hca_num)) {
        ibv_error_abort(IBV_RETURN_ERR, "Error in changing RNDV UD QP transition\n");
    }
}

static inline void mv2_ud_post_zcopy_recv(MPID_Request *req, mv2_ud_zcopy_info_t *zcopy_info)
{
    int i = 0;
    int bytes_to_post = 0, curr_len = 0;
    int  posted_buffers = 0, posts_required = 0;
    struct ibv_recv_wr *bad_wr = NULL;
    struct ibv_recv_wr rr[32];
    struct ibv_sge sge_entry[64];

    mv2_rndv_qp_t *rqp = (mv2_rndv_qp_t *) req->mrail.rndv_qp_entry;
    posts_required = ((req->mrail.rndv_buf_sz + MRAIL_MAX_UD_SIZE - 1) / MRAIL_MAX_UD_SIZE);
    if (posts_required <= 0) {
        PRINT_DEBUG(DEBUG_ZCY_verbose>0 ,"Posted zero or less buffers :%d\n", posts_required);
    }

    while (curr_len < req->mrail.rndv_buf_sz) {
        for (i=0; i<32; i++) {
            MPIU_Assert(posted_buffers < posts_required);
            bytes_to_post = 
                MIN(MRAIL_MAX_UD_SIZE, (req->mrail.rndv_buf_sz - curr_len));
            if (i > 0) {
               rr[i-1].next = &(rr[i]);
            }
        
            rr[i].next = NULL;
            rr[i].wr_id = posted_buffers;
            rr[i].num_sge = 2;
            rr[i].sg_list = &(sge_entry[i * 2]);
            
            sge_entry[i * 2].addr = (uintptr_t) zcopy_info->grh_buf;
            sge_entry[i * 2].length = 40;
            sge_entry[i * 2].lkey = ((dreg_entry *)
                    (zcopy_info->grh_mr))->memhandle[rqp->hca_num]->lkey;
             
            sge_entry[i * 2 + 1].addr = 
                            (uintptr_t) (req->mrail.rndv_buf + curr_len);
            sge_entry[i * 2 + 1].length = bytes_to_post;
            sge_entry[i * 2 + 1].lkey =  ((dreg_entry *) 
                        (req->mrail.d_entry))->memhandle[rqp->hca_num]->lkey;

            curr_len += bytes_to_post;
            posted_buffers++;
            if (curr_len >=  req->mrail.rndv_buf_sz) {
                break;
            }
        }

        if (ibv_post_recv(rqp->ud_qp, rr, &bad_wr)) {
            ibv_error_abort(IBV_RETURN_ERR,"Failed to post zcopy rndv buf\n");
        }
    }
   
    PRINT_DEBUG(DEBUG_ZCY_verbose>0 ,"Posted zcopy recv buffers:%d\n", posted_buffers);

    MPIU_Assert(curr_len == req->mrail.rndv_buf_sz);

}

void MPIDI_CH3I_MRAIL_Prepare_rndv_zcopy(MPIDI_VC_t * vc, MPID_Request * req)
{
    mv2_rndv_qp_t *rqp = NULL;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    
    /* Make sure free QPs available */
    if (proc->zcopy_info.rndv_qp_pool_free_head == NULL) {
        PRINT_DEBUG(DEBUG_ZCY_verbose>2, "No free rndv QP, fall back to R3. remote:%d\n", vc->pg_rank);
        req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
        proc->zcopy_info.no_free_rndv_qp++;
        return;
    }
    
    MPIDI_CH3I_MRAIL_Prepare_rndv(vc, req);
 
    /* return if selected protocl is R3 */
    if (req->mrail.protocol == MV2_RNDV_PROTOCOL_R3) {
        return;
    }
    
    PRINT_DEBUG(DEBUG_ZCY_verbose>1, "Received RTS. Preparing zcopy rndv remote:%d\n", vc->pg_rank);

    MPIU_Assert(req->mrail.protocol == MV2_RNDV_PROTOCOL_UD_ZCOPY);

    MV2_GET_RNDV_QP(rqp, proc);
    req->mrail.rndv_qp_entry = rqp;

    mv2_ud_post_zcopy_recv(req, &proc->zcopy_info);
}

void mv2_ud_zcopy_poll_cq(mv2_ud_zcopy_info_t *zcopy_info, mv2_ud_ctx_t *ud_ctx,
                                vbuf *resend_buf, int hca_index, int *found)
{
    int ne, i;
    struct ibv_wc wc_list[MV2_RNDV_ZCOPY_PUSH_SEGMENT];

    ne = ibv_poll_cq(zcopy_info->rndv_ud_cqs[hca_index], MV2_RNDV_ZCOPY_PUSH_SEGMENT, wc_list);
    if (ne < 0) {
        ibv_error_abort(IBV_RETURN_ERR, "Error in polling RNDV CQ\n");
    } else if (ne > 0) {
        for (i=0; i< ne; i++) {
            if (wc_list[i].status != IBV_WC_SUCCESS) {
                ibv_va_error_abort(IBV_STATUS_ERR, "Error in UD RNDV"
                        " ibv_poll got completion with"
                        " error code %d, wr_id: %lu\n",
                        wc_list[i].status, wc_list[i].wr_id);
            }
            /* handle zcopy finish control message */
            if (wc_list[i].wr_id > (uint64_t)MV2_RNDV_ZCOPY_PUSH_SEGMENT) {
                vbuf *v = (vbuf *) ((uintptr_t) wc_list[i].wr_id);
                v->flags &= ~(UD_VBUF_SEND_INPROGRESS);
                if (v->flags & UD_VBUF_FREE_PENIDING && v != resend_buf) {
                    v->flags &= ~(UD_VBUF_FREE_PENIDING);
                    v->flags &= ~(UD_VBUF_RETRY_ALWAYS);
                    MRAILI_Release_vbuf(v);
                }
                ud_ctx->send_wqes_avail++;
            } else {
                ud_ctx->send_wqes_avail += (int)wc_list[i].wr_id;
            }
        }
        *found = 1;
    } else {
        *found = 0;
    }
}

void MPIDI_CH3I_MRAILI_Rendezvous_zcopy_push(MPIDI_VC_t * vc, 
                    MPID_Request * sreq, mv2_ud_zcopy_info_t *zcopy_info)
{
    int i, hca_index, max_segments, seqnum = 0;
    int posts_required, bytes_to_post;
    mv2_ud_ctx_t *ud_ctx;
    struct ibv_send_wr sr[MV2_RNDV_ZCOPY_PUSH_SEGMENT], *bad_wr;
    struct ibv_sge sg_entry[MV2_RNDV_ZCOPY_PUSH_SEGMENT];
    
    PRINT_DEBUG(DEBUG_ZCY_verbose>0, "ZCOPY rndv push remote qpn:%d hca:%d remote:%d\n",
                    sreq->mrail.remote_qpn, sreq->mrail.hca_index, vc->pg_rank);

    MPIU_Assert(sreq->mrail.d_entry != NULL);
    hca_index = sreq->mrail.hca_index;
    ud_ctx = zcopy_info->rndv_ud_qps[hca_index];
    
    posts_required = ((sreq->mrail.rndv_buf_sz + MRAIL_MAX_UD_SIZE - 1) / MRAIL_MAX_UD_SIZE);

    if (posts_required <= 0) {
        PRINT_DEBUG(DEBUG_ZCY_verbose>0 ,"Posted zero or less buffers :%d remote:%d\n", 
                                                posts_required, vc->pg_rank);
    }
    
    max_segments = MIN (posts_required, MV2_RNDV_ZCOPY_PUSH_SEGMENT);

    for (i=0; i<max_segments; i++) {
        if (i == (max_segments-1)) {
            /* Only last work request sends a complition */
            sr[i].send_flags = IBV_SEND_SIGNALED;
            sr[i].next = NULL;
            sr[i].wr_id = (uint64_t) max_segments;
        } else {
            sr[i].send_flags = 0;
            sr[i].next = &(sr[i+1]);
            sr[i].wr_id = (uint64_t) 0;
        }

        sr[i].num_sge = 1;
        sr[i].wr.ud.remote_qkey = 0;

        sr[i].opcode   = IBV_WR_SEND_WITH_IMM;
        sr[i].wr.ud.remote_qpn  = sreq->mrail.remote_qpn;
        sr[i].wr.ud.ah = vc->mrail.ud[hca_index].ah;
    }

    while (sreq->mrail.rndv_buf_off < sreq->mrail.rndv_buf_sz) {
        int found = 0;
        
        do {
            mv2_ud_zcopy_poll_cq(zcopy_info, ud_ctx, NULL, hca_index, &found);
        } while( ud_ctx->send_wqes_avail < (max_segments+1) ||found);

        for (i=0; i<max_segments; i++) {
            bytes_to_post =  MIN(sreq->mrail.rndv_buf_sz - 
                                sreq->mrail.rndv_buf_off, MRAIL_MAX_UD_SIZE);
            sr[i].sg_list = &(sg_entry[i]);
            sr[i].imm_data = seqnum++;
            sg_entry[i].addr   = (uint64_t) (uintptr_t) ((char *)
                            sreq->mrail.rndv_buf + sreq->mrail.rndv_buf_off);
            sg_entry[i].length = bytes_to_post;
            sg_entry[i].lkey   = ((dreg_entry *)
                            sreq->mrail.d_entry)->memhandle[hca_index]->lkey;
            
            ud_ctx->send_wqes_avail --;

            sreq->mrail.rndv_buf_off += bytes_to_post;
            
            if (sreq->mrail.rndv_buf_off == sreq->mrail.rndv_buf_sz) {
                sr[i].next = NULL;
                sr[i].send_flags = IBV_SEND_SIGNALED;
                sr[i].wr_id = (uint64_t) (i + 1);
                break;
            }
        }

        if(ibv_post_send(ud_ctx->qp, sr, &bad_wr)) {
            ibv_va_error_abort(IBV_RETURN_ERR,"Error in posting UD RNDV QP %d %lu",
                     ud_ctx->send_wqes_avail, bad_wr->wr_id);
        }
    }

    PRINT_DEBUG(DEBUG_ZCY_verbose>1, "Posted rndv send posts:%d to:%d\n", seqnum, vc->pg_rank);
    
    MPIU_Assert(sreq->mrail.rndv_buf_off == sreq->mrail.rndv_buf_sz);
    /* send finish msg*/
    MRAILI_Rndv_send_zcopy_finish(vc, sreq, zcopy_info);
    sreq->mrail.nearly_complete = 1;
}

void MPIDI_CH3_Rendezvous_zcopy_finish(MPIDI_VC_t * vc,
                             MPIDI_CH3_Pkt_zcopy_finish_t * zcopy_finish)
{
    int i, ne, posted_buffers, count = 0, complete;
    int out_of_order = 0, next_to_recv = 0, empty = 0;
    struct ibv_wc *wc; 
    MPID_Request *rreq;
    mv2_rndv_qp_t *rqp;

    MPID_Request_get_ptr(zcopy_finish->receiver_req_id, rreq);
    rqp = (mv2_rndv_qp_t *) rreq->mrail.rndv_qp_entry;
    
    posted_buffers = ((rreq->mrail.rndv_buf_sz + MRAIL_MAX_UD_SIZE - 1) / MRAIL_MAX_UD_SIZE);
    wc = (struct ibv_wc *) MPIU_Malloc (sizeof(struct ibv_wc) * posted_buffers);
    
    do {
        ne = ibv_poll_cq(rqp->ud_cq, posted_buffers - 1, wc);
        if (ne < 0) {
            ibv_error_abort(IBV_RETURN_ERR, "Error in polling recv RNDV UD QP\n");
        } else if (ne > 0) {
            for (i=0; i<ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    ibv_va_error_abort(IBV_RETURN_ERR, "Got error completion "
                    "in RNDV UD recv error. code: %d, wr_id: %lu", 
                    wc[i].status, wc[i].wr_id);
                } else {
                    if(IBV_WC_RECV == wc[i].opcode) {
                        if(wc[i].imm_data != next_to_recv) {
                            PRINT_DEBUG(DEBUG_ZCY_verbose>0,"Out of Order Rndv"
                            "recv'ed:%d expected:%d remote:%d\n", 
                            wc[i].imm_data, next_to_recv, vc->pg_rank);
                            out_of_order = 1;
                        }
                        next_to_recv++;
                    }
                }
                count++;
            }
        } else {
            empty = 1;
        }
    } while (!empty && posted_buffers != count);

    PRINT_DEBUG(DEBUG_ZCY_verbose>1, "Done polling RNDV UD. got %d of %d. remote:%d\n",
                                    count, posted_buffers, vc->pg_rank);
    
    if (count == posted_buffers && !out_of_order) {
        /* send zcopy ack */
        MRAILI_Rndv_send_zcopy_ack(vc, rreq);
        
        if (rreq->mrail.rndv_buf_alloc == 1) {
            MPIDI_CH3_Rendezvous_unpack_data(vc, rreq);
        } else {
            rreq->mrail.rndv_buf = NULL;
        }
            
        MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(rreq);
        MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
        if (complete == TRUE) {
             vc->ch.recv_active = NULL;
        } else {
            ibv_error_abort(IBV_RETURN_ERR, "Error in UD RNDV completion\n");
        }
        
        MV2_RELEASE_RNDV_QP(rqp, &mv2_MPIDI_CH3I_RDMA_Process);
        
    } else {
        PRINT_DEBUG(DEBUG_ZCY_verbose>1, "zcopy Rndv recv failed. "
            "posted: %d recv'ed: %d out_of_order:%d remote:%d\n",
                posted_buffers, count, out_of_order, vc->pg_rank);
        if (posted_buffers != count) {
            MPIU_Assert(posted_buffers > count);
            mv2_flush_zcopy_rndv_qp(rqp, posted_buffers - count);
            mv2_ud_post_zcopy_recv(rreq, &mv2_MPIDI_CH3I_RDMA_Process.zcopy_info);
            MPIDI_CH3_Rendezvous_zcopy_resend_cts(vc, rreq, rqp);
        }
    }
        
    MPIU_Free(wc);
        
}

void MPIDI_CH3_Rendezvous_zcopy_ack(MPIDI_VC_t * vc,
                             MPIDI_CH3_Pkt_zcopy_ack_t * zcopy_ack)
{
    int complete;
    MPID_Request *sreq;

    MPID_Request_get_ptr(zcopy_ack->sender_req_id, sreq);
    
    MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(sreq);
    MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
    MPIU_Assert(complete == TRUE);
}

#endif /* _ENABLE_UD_ */
