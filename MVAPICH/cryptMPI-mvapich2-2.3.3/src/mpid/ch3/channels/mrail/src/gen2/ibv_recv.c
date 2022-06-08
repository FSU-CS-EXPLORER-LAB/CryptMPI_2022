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

#include "rdma_impl.h"
#include "upmi.h"
#include "mpiutil.h"
#include "cm.h"
#ifdef _ENABLE_UD_
#include "mv2_ud.h"
#endif

#define SET_CREDIT(header, vc, rail, transport)                             \
{                                                                           \
    if (transport  == IB_TRANSPORT_RC)  {                                   \
        vc->mrail.rfp.ptail_RDMA_send += header->rdma_credit;               \
        if (vc->mrail.rfp.ptail_RDMA_send >= num_rdma_buffer)               \
            vc->mrail.rfp.ptail_RDMA_send -= num_rdma_buffer;               \
        vc->mrail.srp.credits[rail].remote_cc = header->remote_credit;      \
        vc->mrail.srp.credits[rail].remote_credit += header->vbuf_credit;   \
    } else {                                                                \
    }                                                                       \
}

#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);  fflush(stderr);                   \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

/* FIXME: Ideally the header size should be determined by high level macros,
 * instead of hacking the message header at the device layer */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAIL_Parse_header
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAIL_Parse_header(MPIDI_VC_t * vc,
                                  vbuf * v, void **pkt, int *header_size)
{
    void *vstart;
    MPIDI_CH3I_MRAILI_Pkt_comm_header *header;
#ifdef CRC_CHECK
    unsigned long crc;
#endif
    int mpi_errno = MPI_SUCCESS;
    int ret;
    MPIDI_STATE_DECL(MPIDI_STATE_CH3I_MRAIL_PARSE_HEADER);
    MPIDI_FUNC_ENTER(MPIDI_STATE_CH3I_MRAIL_PARSE_HEADER);

    DEBUG_PRINT("[parse header] vbuf address %p\n", v);
    vstart = v->pheader;
    header = vstart;
    DEBUG_PRINT("[parse header] header type %d\n", header->type);

    /* set it to the header size by default */
    *header_size = MPIDI_CH3_Pkt_size_index[header->type];
#ifdef CRC_CHECK
    crc = update_crc(1, (void *)((uintptr_t)header+sizeof *header),
                     v->content_size - sizeof *header);
    if (crc != header->crc) {
	int rank; UPMI_GET_RANK(&rank);
	MPL_error_printf(stderr, "CRC mismatch, get %lx, should be %lx "
		"type %d, ocntent size %d\n", 
		crc, header->crc, header->type, v->content_size);
        exit(EXIT_FAILURE);
    }
#endif
    switch (header->type) {
#ifndef MV2_DISABLE_HEADER_CACHING 
    case (MPIDI_CH3_PKT_FAST_EAGER_SEND):
    case (MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ):
        {
            MPIDI_CH3I_MRAILI_Pkt_fast_eager *fast_header = vstart;
            MPIDI_CH3_Pkt_eager_send_t *eager_header =
                (MPIDI_CH3_Pkt_eager_send_t *) vc->mrail.rfp.
                cached_incoming;

            if (MPIDI_CH3_PKT_FAST_EAGER_SEND == header->type) {
                *header_size = sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager);
            } else {
                *header_size =
                    sizeof(MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req);
                eager_header->sender_req_id =
                    ((MPIDI_CH3I_MRAILI_Pkt_fast_eager_with_req *)
                     vstart)->sender_req_id;
            }

            DEBUG_PRINT("[receiver side] cached credit %d\n",
                        eager_header->rdma_credit);

            eager_header->data_sz = fast_header->bytes_in_pkt;
            eager_header->seqnum = fast_header->seqnum;

            *pkt = (void *) eager_header;
            DEBUG_PRINT
                ("[recv: parse header] faster headersize returned %d\n",
                 *header_size);
        }
        break;
#endif
#ifdef USE_EAGER_SHORT
    case (MPIDI_CH3_PKT_EAGERSHORT_SEND):
        {
            *pkt = vstart;
            *header_size = sizeof(MPIDI_CH3_Pkt_eagershort_send_t);
        }
        break;
#endif /*USE_EAGER_SHORT*/
    case (MPIDI_CH3_PKT_EAGER_SEND):
        {
            DEBUG_PRINT("[recv: parse header] pkt eager send\n");
#ifndef MV2_DISABLE_HEADER_CACHING 
            if (v->padding != NORMAL_VBUF_FLAG &&
                ((v->content_size - sizeof(MPIDI_CH3_Pkt_eager_send_t)) <= MAX_SIZE_WITH_HEADER_CACHING )) {
                /* Only cache header if the packet is from RdMA path 
                 * XXXX: what is R3_FLAG? 
                 */
                MPIU_Memcpy((vc->mrail.rfp.cached_incoming), vstart,
                       sizeof(MPIDI_CH3_Pkt_eager_send_t));
            }
#endif
            *pkt = (MPIDI_CH3_Pkt_t *) vstart;
#if 0
            if (v->padding == NORMAL_VBUF_FLAG)
#endif
            *header_size = sizeof(MPIDI_CH3_Pkt_eager_send_t);
            DEBUG_PRINT("[recv: parse header] headersize returned %d\n",
                        *header_size);
        }
        break;
    case (MPIDI_CH3_PKT_RNDV_REQ_TO_SEND):
    case (MPIDI_CH3_PKT_RNDV_READY_REQ_TO_SEND):
    case (MPIDI_CH3_PKT_RNDV_CLR_TO_SEND):
    case (MPIDI_CH3_PKT_RMA_RNDV_CLR_TO_SEND):
    case (MPIDI_CH3_PKT_CUDA_CTS_CONTI):
    case (MPIDI_CH3_PKT_RPUT_FINISH):
    case (MPIDI_CH3_PKT_ZCOPY_FINISH):
    case (MPIDI_CH3_PKT_ZCOPY_ACK):
    case (MPIDI_CH3_PKT_MCST_NACK):
    case (MPIDI_CH3_PKT_MCST_INIT_ACK):
    case (MPIDI_CH3_PKT_NOOP):
    case MPIDI_CH3_PKT_EAGER_SYNC_ACK:
    case MPIDI_CH3_PKT_CANCEL_SEND_REQ:
    case MPIDI_CH3_PKT_CANCEL_SEND_RESP:
#ifdef CKPT
    case MPIDI_CH3_PKT_CM_SUSPEND:
    case MPIDI_CH3_PKT_CM_REACTIVATION_DONE:
    case MPIDI_CH3_PKT_CR_REMOTE_UPDATE:
#endif
        {
            *pkt = vstart;
        }
        break;
    case MPIDI_CH3_PKT_RNDV_R3_ACK: 
        {
            *pkt = vstart;
        }
        goto fn_exit;
    case MPIDI_CH3_PKT_ADDRESS:
	{
	    *pkt = vstart;
	    MPIDI_CH3I_MRAILI_Recv_addr(vc, vstart);
	    break;
	}
    case MPIDI_CH3_PKT_ADDRESS_REPLY:
    {
        *pkt = vstart;
        MPIDI_CH3I_MRAILI_Recv_addr_reply(vc, vstart);
        break;
    }
    case MPIDI_CH3_PKT_CM_ESTABLISH:
        {
            *pkt = vstart;
            *header_size = sizeof(MPIDI_CH3_Pkt_cm_establish_t);
            break;
        }
    case MPIDI_CH3_PKT_PACKETIZED_SEND_START:
        {
            *pkt = vstart;
            *header_size = sizeof(MPIDI_CH3_Pkt_packetized_send_start_t);
            break;
        }
    case MPIDI_CH3_PKT_PACKETIZED_SEND_DATA:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_packetized_send_data_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_RNDV_R3_DATA:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_EAGER_SYNC_SEND:
    case MPIDI_CH3_PKT_READY_SEND:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_send_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_PUT_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_put_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_PUT:
        {
            /*Put uses MPIDI_CH3_Pkt_t type*/
            *header_size = sizeof(MPIDI_CH3_Pkt_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_PUT_RNDV:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_put_rndv_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_RNDV:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_rndv_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_RESP:
    case MPIDI_CH3_PKT_GET_RESP_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_resp_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_ACCUMULATE_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_accum_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_ACCUMULATE:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_ACCUMULATE_RNDV:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_accum_rndv_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_ACCUM_RNDV:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_accum_rndv_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_LOCK:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_lock_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_LOCK_ACK: 
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_lock_ack_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_LOCK_OP_ACK: 
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_lock_op_ack_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_UNLOCK:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_unlock_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_FLUSH:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_flush_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_ACK:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_ack_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_DECR_AT_COUNTER: 
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_decr_at_counter_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_FOP_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_fop_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_FOP:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_FOP_RESP:
    case MPIDI_CH3_PKT_FOP_RESP_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_fop_resp_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_CAS_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_cas_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_CAS_RESP_IMMED: 
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_cas_resp_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_ACCUM_IMMED:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_accum_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_ACCUM:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_ACCUM_RESP:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_accum_resp_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_GET_ACCUM_RESP_IMMED: 
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_get_accum_resp_t);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_FLOW_CNTL_UPDATE:
        {
            *header_size = sizeof(MPIDI_CH3I_MRAILI_Pkt_flow_cntl);
            *pkt = vstart;
            break;
        }
    case MPIDI_CH3_PKT_CLOSE:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_close_t);
            *pkt = vstart;
        }
        break;
    case MPIDI_CH3_PKT_RGET_FINISH:
        {
            *header_size = sizeof(MPIDI_CH3_Pkt_rget_finish_t);
            *pkt = vstart;
            break;
        }
    default:
        {
            /* Header is corrupted if control has reached here in prototype */
            /* */
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno,
                    MPI_ERR_OTHER,
                    "**fail",
                    "**fail %s %d", 
                    "Control shouldn't reach here "
                    "in prototype, header %d\n",
                    header->type);
        }
    }

    PRINT_DEBUG(DEBUG_CHM_verbose>1, "Before set credit, vc: %p, v->rail: %d, "
                "pkt: %p, pheader: %p\n", vc, v->rail, pkt, v->pheader);

    SET_CREDIT((&(((MPIDI_CH3_Pkt_t *) 
                        (*pkt))->eager_send)), vc, (v->rail),v->transport);


    if (vc->mrail.srp.credits[v->rail].remote_credit > 0 &&
        vc->mrail.srp.credits[v->rail].backlog.len > 0) {
        MRAILI_Backlog_send(vc, v->rail);
    }

    /* if any credits remain, schedule rendezvous progress */
    if (((vc->mrail.srp.credits[v->rail].remote_credit > 0 
            || (vc->mrail.rfp.ptail_RDMA_send != 
                vc->mrail.rfp.phead_RDMA_send))
        )
        && (vc->mrail.sreq_head != NULL)) {
        PUSH_FLOWLIST(vc);
    }

    if (vc->mrail.state & MRAILI_RC_CONNECTED
            && v->transport == IB_TRANSPORT_RC
            && vc->mrail.rfp.RDMA_recv_buf == NULL
            && num_rdma_buffer && !vc->mrail.rfp.rdma_failed) {
        if ((mv2_MPIDI_CH3I_RDMA_Process.polling_group_size + rdma_pending_conn_request) <
                rdma_polling_set_limit) {
            vc->mrail.rfp.eager_start_cnt++;
            if (rdma_polling_set_threshold < 
                    vc->mrail.rfp.eager_start_cnt) {
                MPICM_lock();
#ifdef _ENABLE_XRC_
                if (xrc_rdmafp_init &&
                        USE_XRC && VC_XST_ISUNSET (vc, XF_SEND_IDLE)) {
                    if (VC_XSTS_ISUNSET (vc, XF_START_RDMAFP | 
                                XF_CONN_CLOSING | XF_DPM_INI)) {
                        PRINT_DEBUG(DEBUG_XRC_verbose>0, "Trying to FP to %d st: %d xr: 0x%08x", 
                                vc->pg_rank, vc->ch.state, vc->ch.xrc_flags);
                        VC_XST_SET (vc, XF_START_RDMAFP);
                        MPICM_unlock();
                        MPIDI_CH3I_CM_Connect (vc);
                        goto fn_exit;
                    }
                }
                else if (!USE_XRC || 
                        (xrc_rdmafp_init && 
                        VC_XSTS_ISUNSET(vc, 
                            XF_DPM_INI | XF_CONN_CLOSING | XF_START_RDMAFP)
                        && VC_XSTS_ISSET (vc, XF_SEND_IDLE | XF_RECV_IDLE)
                        && header->type != MPIDI_CH3_PKT_ADDRESS))
#endif
                {
                    DEBUG_PRINT("FP to %d (IDLE)\n", vc->pg_rank);
                    MPICM_unlock();
                    ret = vbuf_fast_rdma_alloc(vc, 1);
                    if (ret == MPI_SUCCESS) {
                        vbuf_address_send(vc);
                        rdma_pending_conn_request++;
                        vc->mrail.state |=  MRAILI_RFP_CONNECTING;
                    } else {
                        vc->mrail.rfp.rdma_failed = 1;
                    }
                    goto fn_exit;
                }
                MPICM_unlock();
            }
        }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPIDI_STATE_CH3I_MRAIL_PARSE_HEADER);
    return mpi_errno;

fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAIL_Fill_Request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAIL_Fill_Request(MPID_Request * req, vbuf * v,
                                  int header_size, int *nb)
{
    MPL_IOV    *iov;
    int         n_iov;
    size_t      len_avail;
    void        *data_buf;
    int         i;
    MPIDI_STATE_DECL(MPIDI_STATE_CH3I_MRAIL_FILL_REQUEST);
    MPIDI_FUNC_ENTER(MPIDI_STATE_CH3I_MRAIL_FILL_REQUEST);

    len_avail 	= v->content_size - header_size;
    iov 	= (req == NULL) ? NULL : req->dev.iov;
    n_iov 	= (req == NULL) ? 0 : req->dev.iov_count;
    data_buf    = (void *) ((uintptr_t) v->pheader + header_size);

    DEBUG_PRINT
        ("[recv:fill request] total len %d, head len %d, n iov %d\n",
         v->content_size, header_size, n_iov);



#ifdef _ENABLE_CUDA_
    if ( rdma_enable_cuda && is_device_buffer(iov[0].MPL_IOV_BUF)) {

        *nb = 0;
        MPIU_Assert(req->dev.iov_offset == 0 && n_iov == 1);
        MPIU_Memcpy_CUDA(iov[0].MPL_IOV_BUF, 
                data_buf, 
                iov[0].MPL_IOV_LEN,
                cudaMemcpyHostToDevice);
        *nb += iov[0].MPL_IOV_LEN;
        len_avail -= iov[0].MPL_IOV_LEN;
    } else {
#endif

    *nb = 0;
    for (i = req->dev.iov_offset; i < n_iov; i++) {
        if (len_avail >= (MPIDI_msg_sz_t) iov[i].MPL_IOV_LEN
            && iov[i].MPL_IOV_LEN != 0) {
            MPIU_Memcpy(iov[i].MPL_IOV_BUF, data_buf, iov[i].MPL_IOV_LEN);
            data_buf = (void *) ((uintptr_t) data_buf + iov[i].MPL_IOV_LEN);
            len_avail -= iov[i].MPL_IOV_LEN;
            *nb += iov[i].MPL_IOV_LEN;
        } else if (len_avail > 0) {
            MPIU_Memcpy(iov[i].MPL_IOV_BUF, data_buf, len_avail);
            *nb += len_avail;
            break;
        }
    }
#ifdef _ENABLE_CUDA_
    }
#endif
    v->content_consumed = header_size + *nb;

    DEBUG_PRINT
        ("[recv:fill request] about to return form request, nb %d\n", *nb);
    MPIDI_FUNC_EXIT(MPIDI_STATE_CH3I_MRAIL_FILL_REQUEST);
    return MPI_SUCCESS;
}

void MPIDI_CH3I_MRAIL_Release_vbuf(vbuf * v)
{
    v->eager = 0;
    v->coalesce = 0;
    v->content_size = 0;

    if (v->padding == NORMAL_VBUF_FLAG || v->padding == RPUT_VBUF_FLAG)
        MRAILI_Release_vbuf(v);
    else {
        MRAILI_Release_recv_rdma(v);
        MRAILI_Send_noop_if_needed((MPIDI_VC_t *) v->vc, v->rail);
    }
}

int MPIDI_CH3I_MRAILI_Recv_addr(MPIDI_VC_t * vc, void *vstart)
{
    MPIDI_CH3_Pkt_address_t *pkt = vstart;
    int i;
    int ret;
#ifdef _ENABLE_XRC_
    if (USE_XRC && (0 == xrc_rdmafp_init || 
            VC_XST_ISSET (vc, XF_CONN_CLOSING)))
        return MPI_ERR_INTERN;
#endif

    DEBUG_PRINT("set rdma address, dma address %p\n",
            (void *)pkt->rdma_address);

    /* check if it has accepted max allowing connections */
    if (rdma_fp_sendconn_accepted >= rdma_polling_set_limit)
    {
        vbuf_address_reply_send(vc, RDMA_FP_MAX_SEND_CONN_REACHED);
        goto fn_exit;
    }

    if (pkt->rdma_address != 0) {
	    /* Allocating the send vbufs for the eager RDMA flow */
        ret = vbuf_fast_rdma_alloc(vc, 0);
        if (ret == MPI_SUCCESS) {
	        for (i = 0; i < rdma_num_hcas; i ++) {
	            vc->mrail.rfp.RDMA_remote_buf_rkey[i] = pkt->rdma_hndl[i];
	        }
	        vc->mrail.rfp.remote_RDMA_buf = (void *)pkt->rdma_address;
            vbuf_address_reply_send(vc, RDMA_FP_SUCCESS);
            rdma_fp_sendconn_accepted++;
        } else {
            vbuf_address_reply_send(vc, RDMA_FP_SENDBUFF_ALLOC_FAILED);
            return -1;
        } 
    }
fn_exit:
    return MPI_SUCCESS;
}

int MPIDI_CH3I_MRAILI_Recv_addr_reply(MPIDI_VC_t * vc, void *vstart)
{
    int hca_index;
    int ret;
    MPIDI_CH3_Pkt_address_reply_t *pkt = vstart;
    DEBUG_PRINT("Received addr reply packet. reply data :%d\n", pkt->reply_data);
    
    if (pkt->reply_data == RDMA_FP_SENDBUFF_ALLOC_FAILED 
        || pkt->reply_data == RDMA_FP_MAX_SEND_CONN_REACHED) {

        DEBUG_PRINT("RDMA FP setup failed. clean up recv buffers\n ");
    
        if (!mv2_rdma_fast_path_preallocate_buffers) {
            /* de-regster the recv buffers */
            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                if (vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]) {
                    ret = deregister_memory(vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]);
                    if (ret) {
                        MPL_error_printf("Failed to deregister mr (%d)\n", ret);
                    } else {
                        vc->mrail.rfp.RDMA_recv_buf_mr[hca_index] = NULL;
                    }
                }
            }
            /* deallocate recv RDMA buffers */
            if (vc->mrail.rfp.RDMA_recv_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf_DMA);
                vc->mrail.rfp.RDMA_recv_buf_DMA = NULL;
            }

            /* deallocate vbuf struct buffers */
            if (vc->mrail.rfp.RDMA_recv_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf);
                vc->mrail.rfp.RDMA_recv_buf = NULL;
            }
        } else {
            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                vc->mrail.rfp.RDMA_recv_buf_mr[hca_index] = NULL;
            }
            vc->mrail.rfp.RDMA_recv_buf_DMA = NULL;
            vc->mrail.rfp.RDMA_recv_buf = NULL;
        }
        
        /* set flag to mark that FP setup is failed/rejected. 
        we sholdn't try further on this vc */
        vc->mrail.rfp.rdma_failed = 1;

    } else if (pkt->reply_data == RDMA_FP_SUCCESS) {
            
        /* set pointers */
        vc->mrail.rfp.p_RDMA_recv = 0;
        vc->mrail.rfp.p_RDMA_recv_tail = num_rdma_buffer - 1;

        /* Add the connection to the RDMA polling list */
        MPIU_Assert(mv2_MPIDI_CH3I_RDMA_Process.polling_group_size < rdma_polling_set_limit);

        mv2_MPIDI_CH3I_RDMA_Process.polling_set
            [mv2_MPIDI_CH3I_RDMA_Process.polling_group_size] = vc;
        mv2_MPIDI_CH3I_RDMA_Process.polling_group_size++;

        vc->mrail.cmanager.num_channels      += 1;
        vc->mrail.cmanager.num_local_pollings = 1;
        vc->mrail.rfp.in_polling_set          = 1;
        vc->mrail.state &= ~(MRAILI_RFP_CONNECTING);
        vc->mrail.state |= MRAILI_RFP_CONNECTED;
        if (mv2_use_eager_fast_send &&
            !(SMP_INIT && (vc->smp.local_nodes >= 0))) {
            vc->eager_fast_rfp_fn = mv2_eager_fast_rfp_send;
        }
    } else {
        ibv_va_error_abort(GEN_EXIT_ERR,
                "Invalid reply data received. reply_data: pkt->reply_data%d\n",
                                                              pkt->reply_data);
    }
    
    rdma_pending_conn_request--;
    
    return MPI_SUCCESS;
}

