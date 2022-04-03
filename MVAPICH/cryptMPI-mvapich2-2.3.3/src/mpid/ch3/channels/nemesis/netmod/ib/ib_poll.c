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

#include "upmi.h"

#define _GNU_SOURCE
#include "ib_vbuf.h"
#include "ib_process.h"
#include "ib_vc.h"
#include "ib_recv.h"
#include "ib_send.h"
#include "mpidi_ch3_impl.h"
#include "ib_poll.h"
#include "ib_errors.h"
#include "mpid_nem_impl.h"
#include "ib_lmt.h"
#include "ib_srq.h"

#define PKTARRAY_SIZE (MPIDI_CH3_PKT_END_ALL+1)
static MPIDI_CH3_PktHandler_Fcn *pktArray[PKTARRAY_SIZE];

/* This is called whenever a vc has been closed as a result of an
   error.  The VC is put into an error state and cannot be opened
   again. */
#undef FUNCNAME
#define FUNCNAME error_closed
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int error_closed(MPIDI_VC_t *const vc)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_ERROR_CLOSED);

    MPIDI_FUNC_ENTER(MPID_STATE_ERROR_CLOSED);

    VC_FIELD(vc, state) = MPID_NEM_IB_VC_STATE_ERROR;

    mpi_errno = MPIDI_CH3U_Handle_connection(vc, MPIDI_VC_EVENT_TERMINATED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

 fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_ERROR_CLOSED);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}


static inline vbuf * MPIDI_CH3I_RDMA_poll(MPIDI_VC_t * vc)
{
    vbuf *v = NULL;
    volatile VBUF_FLAG_TYPE *head;

    if (num_rdma_buffer == 0)
        return NULL;

    v = &(VC_FIELD(vc, connection)->rfp.RDMA_recv_buf[VC_FIELD(vc, connection)->rfp.p_RDMA_recv]);
    head = v->head_flag;

    if (*head && VC_FIELD(vc, connection)->rfp.p_RDMA_recv != VC_FIELD(vc, connection)->rfp.p_RDMA_recv_tail) {
        /* advance receive pointer */
        if (++(VC_FIELD(vc, connection)->rfp.p_RDMA_recv) >= num_rdma_buffer) {
            VC_FIELD(vc, connection)->rfp.p_RDMA_recv = 0;
        }
        v->iheader = v->buffer;
        DEBUG_PRINT("[recv: poll rdma] recv %d, tail %d, size %d\n",
            vc->pg_rank,
            VC_FIELD(vc, connection)->rfp.p_RDMA_recv, VC_FIELD(vc, connection)->rfp.p_RDMA_recv_tail, *head);
        v->pheader = v->iheader + IB_PKT_HEADER_LENGTH;
        v->content_size = (*head & FAST_RDMA_SIZE_MASK) - IB_PKT_HEADER_LENGTH;
    } else {
        v = NULL;
    }
    return v;
}




#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_handle_read_individual
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_handle_read_individual(MPIDI_VC_t* vc, vbuf* buffer, int* header_type)
{
    int mpi_errno = MPI_SUCCESS;
    int header_size = 0;
    MPIDI_CH3_Pkt_t* header = NULL;
    int packetized_recv = 0;

    /* Step one, ask lower level to provide header */
    /*  save header at req->dev.pending_pkt, and return the header size */
    /*  ??TODO: Possibly just return the address of the header */

    DEBUG_PRINT("[handle read] pheader: %p\n", buffer->pheader);

    mpi_errno = MPIDI_nem_ib_parse_header(vc, buffer, (void *)(&header), &header_size);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }


    *header_type = header->type;
    MPID_Request* req = VC_FIELD(vc, recv_active);

    MPID_nem_lmt_pkthandler_init(pktArray, PKTARRAY_SIZE);

#ifdef ENABLE_CHECKPOINTING
    MPIDI_nem_ckpt_pkthandler_init(pktArray, PKTARRAY_SIZE);
#endif 

    MPIDI_msg_sz_t lmt_len;
    switch(((MPIDI_nem_ib_pkt_comm_header* )buffer->iheader)->type)
    {   
        case MPIDI_NEM_PKT_LMT_RTS:
            lmt_len = sizeof(MPIDI_CH3_Pkt_t) + sizeof(MPID_nem_ib_lmt_cookie);
            mpi_errno = pktArray[header->type](vc, buffer->pheader,
                                     &(lmt_len), &req);
            goto fn_exit;

        case MPIDI_NEM_PKT_LMT_CTS:
            lmt_len = sizeof(MPIDI_CH3_Pkt_t) + sizeof(MPID_nem_ib_lmt_cookie);
            mpi_errno = pktArray[header->type](vc, buffer->pheader,
                                     &(lmt_len), &req);
            goto fn_exit;
        case MPIDI_NEM_PKT_LMT_DONE:
            lmt_len = sizeof(MPIDI_CH3_Pkt_t);
            mpi_errno = pktArray[header->type](vc, buffer->pheader,
                                     &(lmt_len), &req);
            goto fn_exit;
#ifdef ENABLE_CHECKPOINTING
        case MPIDI_NEM_PKT_CKPT_MARKER:
        case MPIDI_NEM_IB_PKT_UNPAUSE:
            lmt_len = sizeof(MPIDI_CH3_Pkt_t);
            mpi_errno = pktArray[header->type](vc, buffer->pheader,
                                     &(lmt_len), &req);
            goto fn_exit;
#endif
    }

    switch(((MPIDI_nem_ib_pkt_comm_header* )buffer->iheader)->type)
    {
    case MPIDI_CH3_PKT_NOOP:
    case MPIDI_CH3_PKT_ADDRESS:
    case MPIDI_CH3_PKT_ADDRESS_REPLY: 
    case MPIDI_CH3_PKT_RNDV_R3_ACK:
            DEBUG_PRINT("ADDRESS or NOOP received, don't need to proceed\n");
        goto fn_exit;
/* packetized codes */
    case MPIDI_CH3_PKT_PACKETIZED_SEND_DATA:
            DEBUG_PRINT("Packetized data received, don't need to proceed\n");
            MPIDI_nem_ib_packetized_recv_data(vc, buffer);
        goto fn_exit;
    case MPIDI_CH3_PKT_RNDV_R3_DATA:
            DEBUG_PRINT("R3 data received, don't need to proceed\n");
            MPIDI_nem_ib_lmt_r3_recv_data(vc,buffer);
        goto fn_exit;
/* packetized codes */
    case MPIDI_CH3_PKT_PACKETIZED_SEND_START:
            packetized_recv = 1;
            header_size += ((MPIDI_CH3_Pkt_packetized_send_start_t*) header)->origin_head_size;
#if defined(MPIDI_MRAILI_COALESCE_ENABLED)
            buffer->content_consumed = header_size;
#endif /* defined(MPIDI_MRAILI_COALESCE_ENABLED) */
            header = (void *)((uintptr_t) header + sizeof(MPIDI_CH3_Pkt_packetized_send_start_t));
        break;
    default:
        break;
    }


    MPIDI_msg_sz_t buflen = sizeof(MPIDI_CH3_Pkt_t);

    /* Step two, load request according to the header content */
    if ((mpi_errno = MPIDI_CH3U_Handle_recv_pkt(
        vc,
        (void*) header,
        &buflen,
        &VC_FIELD(vc, recv_active))) != MPI_SUCCESS)
    {
        MPIR_ERR_POP(mpi_errno);
    }

    /* DEBUG_PRINT("[recv: progress] about to fill request, recv_active %p\n", vc->ch.recv_active); */

    if (VC_FIELD(vc, recv_active) != NULL)
    {
        /* Step three, ask lower level to fill the request */
        /*      request is vc->ch.recv_active */

        /* packetized codes */
        if (packetized_recv == 1)
        {
            if ((mpi_errno = MPIDI_nem_ib_packetized_recv_req(
                    vc,
                    VC_FIELD(vc, recv_active))) != MPI_SUCCESS)
            {
                MPIR_ERR_POP(mpi_errno);
            }
        }

        int nb;

        if ((mpi_errno = MPIDI_nem_ib_fill_request(
                VC_FIELD(vc, recv_active),
                buffer,
                header_size,
                &nb)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        req = VC_FIELD(vc, recv_active);
        DEBUG_PRINT(
            "[recv: handle read] nb %d, iov n %d, len %d, VBUFSIZE %d\n",
            nb,
            req->dev.iov_count,
            req->dev.iov[0].MPL_IOV_LEN,
            VBUF_BUFFER_SIZE
        );


        /* ch3_progress.c */
        if (MPIDI_nem_ib_request_adjust_iov(req, nb))
        {
            /* Read operation complete */
            DEBUG_PRINT("[recv: handle read] adjust iov correct\n");
            int complete;

            if ((mpi_errno = MPIDI_CH3U_Handle_recv_req(
                    vc,
                    req,
                    &complete)) != MPI_SUCCESS)
            {
                MPIR_ERR_POP(mpi_errno);
            }

            DEBUG_PRINT("[recv: handle read] adjust req fine, complete %d\n", complete);

            while (!complete)
            {
                header_size += nb;

                /* Fill request again */
                if ((mpi_errno = MPIDI_nem_ib_fill_request(
                        req,
                        buffer,
                        header_size,
                        &nb)) != MPI_SUCCESS)
                {
                    MPIR_ERR_POP(mpi_errno);
                }

                if (!MPIDI_nem_ib_request_adjust_iov(req, nb))
                {
                    if (!packetized_recv)
                    {
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    }

                    goto fn_exit;
                }

                if ((mpi_errno = MPIDI_CH3U_Handle_recv_req(
                        vc,
                        req,
                        &complete)
                    ) != MPI_SUCCESS)
                {
                    MPIR_ERR_POP(mpi_errno);
                }
            }

            /* If the communication is packetized, we are expecing more packets for the
             * request. We encounter an error if the request finishes at this stage */
            /* packetized codes */
            if (packetized_recv)
            {
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            }
        }
        else if (!packetized_recv)
        {
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        }
#if defined(DEBUG)
        else
        {
            DEBUG_PRINT("unfinished req left to packetized send\n");
        }
#endif /* defined(DEBUG) */
        VC_FIELD(vc, recv_active) = NULL;
    }
#if defined(DEBUG)
    else
    {
        /* we are getting a 0 byte msg header */
    }
#endif /* if defined(DEBUG) */

fn_fail:
fn_exit:
    DEBUG_PRINT("exiting handle read\n");
    return mpi_errno;
}



/**
 * ib read progress function
 * @param vc_pptr pointer of pointer of vc
 * @param v_ptr pointer of vbuf
 * @param is_blocking flag of blocking polling
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_read_progress
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_read_progress(MPIDI_VC_t ** vc_pptr, vbuf ** v_ptr, int is_blocking)
{
    static MPIDI_VC_t   *pending_vc = NULL;
    int         type;
    MPIDI_VC_t  *recv_vc_ptr;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_IB_READ_PROGRESS);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_IB_READ_PROGRESS);

    if (pending_vc != NULL) {
        type =
            MPIDI_nem_ib_waiting_msg(pending_vc, v_ptr, is_blocking);
        if (type == T_CHANNEL_CONTROL_MSG_ARRIVE) {
            if((void *) pending_vc != (*v_ptr)->vc) {
                fprintf(stderr, "mismatch %p %p\n", pending_vc,
                        (*v_ptr)->vc);
            }
            MPIU_Assert((void *) pending_vc == (*v_ptr)->vc);
            *vc_pptr = pending_vc;
        } else if(type == T_CHANNEL_EXACT_ARRIVE) {
            *vc_pptr = pending_vc;
            pending_vc = NULL;
        }
        goto fn_exit;
    }

    type = MPIDI_nem_ib_get_next_vbuf(vc_pptr, v_ptr);
    if (type != T_CHANNEL_NO_ARRIVE) {
        goto fn_exit;
    }

    /* local polling has finished, now we need to start global subchannel polling
     * For convenience, at this stage, we by default refer to the global polling channel
     * as the send recv channel on each of the queue pair
     * TODO: we may extend this interface to support other format of channel polling */
    /* Interface cq_poll requires that if *v_ptr is exactly the next packet
     * don't enqueue * v_ptr.
     * if *v_ptr is not initialized as NULL, Cq_poll will return exactly only the
     * packet on *v_ptr
     * TODO: Is this the optimal way?
     */

    type = MPIDI_nem_ib_cq_poll(v_ptr, NULL, 0, is_blocking);
    if (type != T_CHANNEL_NO_ARRIVE) {
        recv_vc_ptr = (*v_ptr)->vc;
        *vc_pptr = recv_vc_ptr;
        switch (type) {
        case (T_CHANNEL_EXACT_ARRIVE):
            DEBUG_PRINT("Get one packet with exact seq num\n");
            break;
        case (T_CHANNEL_OUT_OF_ORDER_ARRIVE):
            /* It is possible that *v_ptr points to a vbuf that contains later pkts send/recv
             * may return vbuf from any connection */
            /* DEBUG_PRINT("get out of order progress seqnum %d, expect %d\n",
                        (*v_ptr)->seqnumber, recv_vc_ptr->seqnum_recv); */

            type =
                MPIDI_nem_ib_waiting_msg(recv_vc_ptr, v_ptr, 1);
            if (type == T_CHANNEL_CONTROL_MSG_ARRIVE) {
                pending_vc = recv_vc_ptr;
            } else if (T_CHANNEL_EXACT_ARRIVE == type) {
                DEBUG_PRINT("Get out of order delivered msg\n");
            } else {
                fprintf(stderr, "Error recving run return type\n");
                exit(1);
            }
            break;
        case (T_CHANNEL_CONTROL_MSG_ARRIVE):
            DEBUG_PRINT("Get one control msg\n");
            break;
        default:
            /* Error here */
            break;
        }
        goto fn_exit;
    }
  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_IB_READ_PROGRESS);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_handle_read
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_handle_read(MPIDI_VC_t * vc, vbuf * buffer)
{
    int mpi_errno = MPI_SUCCESS;
    int header_type;

/* coalesce codes */
#if defined(MPIDI_MRAILI_COALESCE_ENABLED)
    /* we don't know how many packets may be combined, so
     * we check after reading a packet to see if there are
     * more bytes yet to be consumed
     */

    buffer->content_consumed = 0;
    unsigned char* original_buffer = buffer->buffer;

    DEBUG_PRINT("[handle read] buffer %p\n", buffer);

    int total_consumed = 0;

    /* TODO: Refactor this algorithm so the first flag is not used. */
    int first = 1;

    do
    {
        vc->ch.recv_active = vc->ch.req;
        buffer->content_consumed = 0;

        if ((mpi_errno = handle_read_individual(vc, buffer, &header_type)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        buffer->pheader = (void *) ((uintptr_t) buffer->pheader + buffer->content_consumed);
        total_consumed += buffer->content_consumed;

        if (MPIDI_CH3I_Seq(header_type) && !first)
        {
            ++vc->seqnum_recv;
        }

        first = 0;
    }
    while (total_consumed != buffer->content_size);

    DEBUG_PRINT("Finished with buffer -- size: %d, consumed: %d\n", buffer->content_size, buffer->content_consumed);

    /* put the original buffer address back */
    buffer->buffer = original_buffer;
    buffer->pheader = original_buffer;

    DEBUG_PRINT("buffer set to: %p\n", buffer->buffer);
#else /* defined(MPIDI_MRAILI_COALESCE_ENABLED) */
    if ((mpi_errno = MPIDI_nem_ib_handle_read_individual(vc, buffer, &header_type)) != MPI_SUCCESS)
    {
        MPIR_ERR_POP(mpi_errno);
    }
#endif /* defined(MPIDI_MRAILI_COALESCE_ENABLED) */

fn_fail:
    /* by this point we can always free the vbuf */
    MPIDI_nem_ib_release_vbuf(buffer);

    return mpi_errno;
}

/**
 * poll function
 * @param in_blocking_poll flag to blocking polling
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_poll
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_poll(int in_blocking_poll)
{

    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_IB_POLL);
    MPIDI_FUNC_ENTER(MPID_STATE_IB_POLL);

    MPIDI_VC_t *vc_ptr = NULL;
    vbuf *buffer = NULL;

    if((mpi_errno = MPIDI_nem_ib_read_progress(&vc_ptr, &buffer, in_blocking_poll)) != MPI_SUCCESS)
    {
        MPIR_ERR_POP(mpi_errno);
    }

    if (vc_ptr == NULL) {
    /* spin_count code */
    } else {
        if ((mpi_errno = MPIDI_nem_ib_handle_read(vc_ptr, buffer)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    if (flowlist) {
        MPID_nem_lmt_ib_process_rndv();
    }

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_IB_POLL);
    return mpi_errno;
}

/**
 * get seq number from vbuf according to the header type
 * need to update MPIDI_CH3_Pkt_type
 *
 */
static inline int GetSeqNumVbuf(vbuf * buf)
{
    if (NULL == buf) {
        return PKT_IS_NULL;
    }

    switch(((MPIDI_nem_ib_pkt_comm_header *)buf->iheader)->type) {
        case MPIDI_CH3_PKT_EAGER_SEND:
#if defined(USE_EAGER_SHORT)
        case MPIDI_CH3_PKT_EAGERSHORT_SEND:
#endif
        case MPIDI_CH3_PKT_READY_SEND:
        case MPIDI_CH3_PKT_EAGER_SYNC_SEND:
        case MPIDI_CH3_PKT_EAGER_SYNC_ACK:
        case MPIDI_CH3_PKT_RNDV_REQ_TO_SEND:
        case MPIDI_CH3_PKT_RNDV_CLR_TO_SEND:
        case MPIDI_CH3_PKT_RNDV_SEND:
        case MPIDI_CH3_PKT_PACKETIZED_SEND_START:
        case MPIDI_CH3_PKT_PACKETIZED_SEND_DATA:
        case MPIDI_CH3_PKT_RNDV_R3_DATA:
        case MPIDI_CH3_PKT_CANCEL_SEND_REQ:
        case MPIDI_CH3_PKT_CANCEL_SEND_RESP:
        case MPIDI_CH3_PKT_PUT:
        case MPIDI_CH3_PKT_PUT_IMMED:
        case MPIDI_CH3_PKT_GET:
        case MPIDI_CH3_PKT_GET_RESP:
        case MPIDI_CH3_PKT_ACCUMULATE:
        case MPIDI_CH3_PKT_LOCK:
        case MPIDI_CH3_PKT_FLUSH:
        case MPIDI_CH3_PKT_UNLOCK:
        case MPIDI_CH3_PKT_GET_ACCUM:
        case MPIDI_CH3_PKT_GET_ACCUM_RESP:
        case MPIDI_CH3_PKT_FOP:
        case MPIDI_CH3_PKT_FOP_RESP:
        case MPIDI_CH3_PKT_FLOW_CNTL_UPDATE:
        case MPIDI_CH3_PKT_CLOSE:
        case MPIDI_NEM_PKT_LMT_RTS:
        case MPIDI_NEM_PKT_LMT_CTS:
        case MPIDI_NEM_PKT_LMT_DONE:
        case MPIDI_NEM_PKT_LMT_COOKIE:
#ifdef ENABLE_CHECKPOINTING
        case MPIDI_NEM_PKT_CKPT_MARKER:
#endif
            {
                return ((MPIDI_nem_ib_pkt_comm_header *)(buf->iheader))->seqnum;
            }
#ifndef MV2_DISABLE_HEADER_CACHING 
        case MPIDI_CH3_PKT_FAST_EAGER_SEND:
        case MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ:
            {
                return ((MPIDI_nem_ib_pkt_fast_eager *)(buf->iheader))->seqnum;
            }
#endif
        default:
            return PKT_NO_SEQ_NUM;
    }
}


static inline int PKT_IS_NOOP(void* v)
{
        return ((MPIDI_nem_ib_pkt_comm_header*) ((vbuf*) v)->iheader)->type == MPIDI_CH3_PKT_NOOP;
}

static int MPIDI_nem_ib_test_pkt(vbuf **vbuf_handle)
{
    int type = T_CHANNEL_NO_ARRIVE;
    while (arriving_head) {
        type = MPIDI_nem_ib_waiting_msg(arriving_head->vc, vbuf_handle, 0);
        if (type == T_CHANNEL_NO_ARRIVE) {
            arriving_head->inqueue = 0;
            arriving_head = arriving_head->next_arriving;
            if (!arriving_head)
                arriving_tail = NULL;
        } else {
            break;
        }
    }

    return type;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_MPIDI_nem_ib_cq_poll
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_cq_poll(vbuf **vbuf_handle,
        MPIDI_VC_t * vc_req, int receiving, int is_blocking)
{
    int ne, ret, err = 0;
    *vbuf_handle = NULL;
    int num_cqs = 0;
    int is_send_completion;
    struct ibv_wc wc;
    vbuf* v;
    MPIDI_VC_t *vc = NULL;
    int i = 0;
    int needed;
    int type = T_CHANNEL_NO_ARRIVE;
    struct ibv_cq *ev_cq;
    struct ibv_cq *chosen_cq;
    static unsigned long nspin = 0;
    void *ev_ctx;

    needed = 0;

    if (!receiving && !vc_req) {
        type = MPIDI_nem_ib_test_pkt(vbuf_handle);
        if (type == T_CHANNEL_EXACT_ARRIVE
                || type == T_CHANNEL_CONTROL_MSG_ARRIVE)
            goto fn_exit;
    }

    /*
    if ((process_info.hca_type == CHELSIO_T3) &&
        (process_info.cluster_size != VERY_SMALL_CLUSTER)) {
        num_cqs = 2;
    } else 
    */
    {
        num_cqs = 1;
    }

    for (i=0; i < ib_hca_num_hcas; ++i) {
        int cq_choice;
        for (cq_choice = 0; cq_choice < num_cqs; ++cq_choice) {
				if (1 == num_cqs) {
                    chosen_cq = hca_list[i].cq_hndl;
                } else {
                    if (0 == cq_choice) {
                        chosen_cq = hca_list[i].send_cq_hndl;
                    } else {
                        chosen_cq = hca_list[i].recv_cq_hndl;
                    }
                }
                ne = ibv_poll_cq(chosen_cq, 1, &wc);
                if (ne < 0 ) {
                    ibv_error_abort(IBV_RETURN_ERR, "Fail to poll cq\n");
                } else if (ne) {
                    v = (vbuf *) ((uintptr_t) wc.wr_id);

                    vc = (MPIDI_VC_t *) (v->vc);

                    if (wc.status != IBV_WC_SUCCESS) {
                        if (wc.opcode == IBV_WC_SEND ||
                                wc.opcode == IBV_WC_RDMA_WRITE ) {
                            fprintf(stderr, "send desc error\n");
                        } else {
                            fprintf(stderr, "recv desc error, %d\n", wc.opcode);
                        }
                        error_closed(vc);
                        MRAILI_Release_vbuf(v);
                        goto fn_exit;
                    }

                if (2 == num_cqs) {
                    if (0 == cq_choice) {
                        if (process_info.global_used_send_cq) {
                            process_info.global_used_send_cq--;
                        } else {
                            ibv_va_error_abort(IBV_STATUS_ERR,
                                                "trying to decrement send cq"
                                                " count already at 0");
                        }
                    } else {
                        if (process_info.global_used_recv_cq) {
                            process_info.global_used_recv_cq--;
                        } else {
                            ibv_va_error_abort(IBV_STATUS_ERR,
                                                "trying to decrement recv cq"
                                                " count already at 0");
                        }
                    }
                }
                    is_send_completion = (wc.opcode == IBV_WC_SEND
                        || wc.opcode == IBV_WC_RDMA_WRITE
                        || wc.opcode == IBV_WC_RDMA_READ);

                    /* srq codes */
                    if(!is_send_completion && process_info.has_srq) {
                        vc = (void *)(unsigned long)
                            (((MPIDI_nem_ib_pkt_comm_header *)
                              ((vbuf *)v)->iheader)->vc_addr);
                        v->vc = vc;
                        v->rail = ((MPIDI_nem_ib_pkt_comm_header *)
                                ((vbuf*)v)->iheader)->rail;
                    }

                    /* get the VC and increase its wqe */
                    if (is_send_completion) {
                        DEBUG_PRINT("[device_Check] process send, v %p\n", v);
                        /* ib_send.c */
                        MRAILI_Process_send(v);
                        type = T_CHANNEL_NO_ARRIVE;
                        *vbuf_handle = NULL;
                    } else if ((NULL == vc_req || vc_req == vc) && 0 == receiving ){
                        /* In this case, we should return the vbuf
                         * any way if it is next expected*/
                        int seqnum = GetSeqNumVbuf(v);

                        *vbuf_handle = v;
                        v->content_size = wc.byte_len - IB_PKT_HEADER_LENGTH;

                        /* srq codes */
                        if (process_info.has_srq) {
                            pthread_spin_lock(&srq_info.
                                    srq_post_spin_lock);

                            if(v->padding == NORMAL_VBUF_FLAG) {
                                /* Can only be from SRQ path */
                                --srq_info.posted_bufs[i];
                            }

                            if(srq_info.posted_bufs[i] <=
                                    rdma_credit_preserve) {
                                /* Need to post more to the SRQ */
                                srq_info.posted_bufs[i] +=
                                    MPIDI_nem_ib_post_srq_buffers(mv2_srq_fill_size -
                                        srq_info.posted_bufs[i], i);

                            }

                            pthread_spin_unlock(&srq_info.
                                    srq_post_spin_lock);

                            /* Check if we need to release the SRQ limit thread */
                            if (srq_info.
                                    srq_zero_post_counter[i] >= 1) {
                                pthread_mutex_lock(
                                        &srq_info.
                                        srq_post_mutex_lock[i]);
                                srq_info.srq_zero_post_counter[i] = 0;
                                pthread_cond_signal(&srq_info.
                                        srq_post_cond[i]);
                                pthread_mutex_unlock(
                                        &srq_info.
                                        srq_post_mutex_lock[i]);
                            }

                        }
                        else
                        {
                            VC_FIELD(vc, connection)->srp.credits[v->rail].preposts--;

                            needed = rdma_prepost_depth + rdma_prepost_noop_extra
                                     + MIN(rdma_prepost_rendezvous_extra,
                                        VC_FIELD(vc, connection)->srp.credits[v->rail].
                                           rendezvous_packets_expected);
                        }
                        DEBUG_PRINT("[cq_poll]: get seqnum = %d, expect = %d, seqnum in ib = %d, "
                                " v->content_size = %d, type = %d\n", seqnum, 
                                VC_FIELD(vc, seqnum_recv), ((MPIDI_nem_ib_pkt_comm_header *)v->iheader)->seqnum, v->content_size,
                                ((MPIDI_nem_ib_pkt_comm_header *)v->iheader)->type);

                        if (seqnum == PKT_NO_SEQ_NUM){
                            type = T_CHANNEL_CONTROL_MSG_ARRIVE;
                        } else if (seqnum == VC_FIELD( vc, seqnum_recv) ) {
                        	VC_FIELD( vc, seqnum_recv)++;
                            type = T_CHANNEL_EXACT_ARRIVE;
                            DEBUG_PRINT("[channel manager] get one with exact seqnum\n");
                        } else {
                            type = T_CHANNEL_OUT_OF_ORDER_ARRIVE;
                            VQUEUE_ENQUEUE(VC_FIELD(vc, cmanager),
                                    INDEX_GLOBAL(VC_FIELD(vc, cmanager), v->rail), v);
                            DEBUG_PRINT("get recv %d (expected %d)\n", seqnum, VC_FIELD( vc, seqnum_recv));
                        }

                        /* srq codes */
                        if (!process_info.has_srq) {
                            if (PKT_IS_NOOP(v)) {
                                PREPOST_VBUF_RECV(vc, v->rail);
                                /* noops don't count for credits */
                                --VC_FIELD(vc, connection)->srp.credits[v->rail].local_credit;
                            }
                            else if (VC_FIELD(vc, connection)->srp.credits[v->rail].preposts < rdma_rq_size &&
                                     VC_FIELD(vc, connection)->srp.credits[v->rail].preposts +
                                     rdma_prepost_threshold < needed)
                            {
                                do {
                                    PREPOST_VBUF_RECV(vc, v->rail);
                                } while (VC_FIELD(vc, connection)->srp.credits[v->rail].preposts < rdma_rq_size &&
                                    VC_FIELD(vc, connection)->srp.credits[v->rail].preposts < needed);
                            }

                            MRAILI_Send_noop_if_needed(vc, v->rail);
                        }

                        if (type == T_CHANNEL_CONTROL_MSG_ARRIVE ||
                                type == T_CHANNEL_EXACT_ARRIVE ||
                                type == T_CHANNEL_OUT_OF_ORDER_ARRIVE) {
                            goto fn_exit;
                        }
                    } else {
                        /* Commenting out the assert - possible coding error
                         * MPIU_Assert(0);
                         */
                        /* Now since this is not the packet we want, we have to enqueue it */
                        type = T_CHANNEL_OUT_OF_ORDER_ARRIVE;
                        *vbuf_handle = NULL;
                        v->content_size = wc.byte_len - IB_PKT_HEADER_LENGTH;
                        VQUEUE_ENQUEUE(VC_FIELD(vc, cmanager),
                                INDEX_GLOBAL(&VC_FIELD(vc, cmanager), v->rail),
                                v);

                        /* srq codes */
                        if (process_info.has_srq) {
                            pthread_spin_lock(&srq_info.srq_post_spin_lock);

                            if(v->padding == NORMAL_VBUF_FLAG) {
                                /* Can only be from SRQ path */
                                --srq_info.posted_bufs[i];
                            }

                            if(srq_info.posted_bufs[i] <= rdma_credit_preserve) {
                                /* Need to post more to the SRQ */
                                srq_info.posted_bufs[i] +=
                                    MPIDI_nem_ib_post_srq_buffers(mv2_srq_fill_size -
                                        srq_info.posted_bufs[i], i);

                            }

                            pthread_spin_unlock(&srq_info.
                                    srq_post_spin_lock);
                        } else
                        {
                            VC_FIELD(vc, connection)->srp.credits[v->rail].preposts--;

                            needed = rdma_prepost_depth + rdma_prepost_noop_extra
                                     + MIN(rdma_prepost_rendezvous_extra,
                                           VC_FIELD(vc, connection)->srp.credits[v->rail].
                                           rendezvous_packets_expected);

                            if (PKT_IS_NOOP(v)) {
                                PREPOST_VBUF_RECV(vc, v->rail);
                                VC_FIELD(vc, connection)->srp.credits[v->rail].local_credit--;
                            }
                            else
                            if (VC_FIELD(vc, connection)->srp.credits[v->rail].preposts <
                                    rdma_rq_size &&
                                    VC_FIELD(vc, connection)->srp.credits[v->rail].preposts +
                                    rdma_prepost_threshold < needed) {
                                do {
                                    PREPOST_VBUF_RECV(vc, v->rail);
                                } while (VC_FIELD(vc, connection)->srp.credits[v->rail].preposts
                                         < rdma_rq_size &&
                                         VC_FIELD(vc, connection)->srp.credits[v->rail].preposts <
                                        needed);
                            }
                            MRAILI_Send_noop_if_needed(vc, v->rail);
                        }
                    }
                } else {
                    *vbuf_handle = NULL;
                    type = T_CHANNEL_NO_ARRIVE;
                    nspin++;

                    /* Blocking mode progress */
                    if(rdma_use_blocking && is_blocking && nspin >= rdma_blocking_spin_count_threshold) {
                        /* Okay ... spun long enough, now time to go to sleep! */

        #if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                        MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_unlock(&MPIR_ThreadInfo.global_mutex, &err);
                        MPIU_THREAD_CHECK_END
        #endif
                        do {
                            ret = ibv_get_cq_event(
                                    hca_list[i].comp_channel,
                                    &ev_cq, &ev_ctx);
                            if (ret && errno != EINTR) {
                                ibv_va_error_abort(IBV_RETURN_ERR,
                                        "Failed to get cq event: %d\n", ret);
                            }
                        } while (ret && errno == EINTR);
        #if (MPICH_THREAD_LEVEL == MPI_THREAD_MULTIPLE)
                        MPIU_THREAD_CHECK_BEGIN
                        MPID_Thread_mutex_lock(&MPIR_ThreadInfo.global_mutex, &err);
                        MPIU_THREAD_CHECK_END
        #endif

                    if (num_cqs == 1) {
                                if (ev_cq != hca_list[i].cq_hndl) {
                                    ibv_error_abort(IBV_STATUS_ERR,
                                             "Event in unknown CQ\n");
                                }

                           ibv_ack_cq_events(hca_list[i].cq_hndl, 1);

                                if (ibv_req_notify_cq(
                                    hca_list[i].cq_hndl, 0)) {
                                    ibv_error_abort(IBV_RETURN_ERR,
                                            "Couldn't request for CQ notification\n");
                                }
                    } else {
                                if (ev_cq == hca_list[i].send_cq_hndl) {
                                ibv_ack_cq_events(
                                    hca_list[i].send_cq_hndl, 1);

                                    if (ibv_req_notify_cq(
                                  hca_list[i].send_cq_hndl, 0)) {
                                        ibv_error_abort(IBV_RETURN_ERR,
                                           "Couldn't request for CQ notification\n");
                                    }
                        } else if (ev_cq ==
                                    hca_list[i].recv_cq_hndl) {
                                ibv_ack_cq_events(
                                    hca_list[i].recv_cq_hndl, 1);

                                    if (ibv_req_notify_cq(
                                  hca_list[i].recv_cq_hndl, 0)) {
                                        ibv_error_abort(IBV_RETURN_ERR,
                                           "Couldn't request for CQ notification\n");
                                    }
                                } else {
                                   ibv_error_abort(IBV_STATUS_ERR,
                                             "Event in unknown CQ\n");
                        }
                    }
                        nspin = 0;
                    }
                }
        }
    }
fn_exit:
    return type;
}


int MPIDI_nem_ib_get_next_vbuf(MPIDI_VC_t** vc_ptr, vbuf** vbuf_ptr)
{
    *vc_ptr = NULL;
    *vbuf_ptr = NULL;
    VBUF_FLAG_TYPE size;
    int type = MPIDI_nem_ib_test_pkt(vbuf_ptr);

    switch(type)
    {
    case T_CHANNEL_CONTROL_MSG_ARRIVE:
    case T_CHANNEL_EXACT_ARRIVE:
            *vc_ptr = (*vbuf_ptr)->vc;
        goto fn_exit;
    case T_CHANNEL_OUT_OF_ORDER_ARRIVE:
            type = T_CHANNEL_NO_ARRIVE;
            *vbuf_ptr = NULL;
        break;
    }

    if (num_rdma_buffer == 0)
    {
        goto fn_exit;
    }

    int i = 0;
    MPIDI_VC_t* vc = NULL;
    int seq;
    vbuf* v = NULL;
    volatile VBUF_FLAG_TYPE* tail = NULL;
    volatile VBUF_FLAG_TYPE* head = NULL;


    /* no msg is queued, poll rdma polling set */
    for (i=0; i < process_info.polling_group_size; ++i)
    {
        vc = process_info.polling_set[i];
        seq  = GetSeqNumVbuf(VC_FIELD(vc, cmanager)->msg_channels[INDEX_LOCAL(VC_FIELD(vc, cmanager),0)].v_queue_head);

        /* rfp codes */
        if (seq == PKT_IS_NULL)
        {
            v = &(VC_FIELD(vc, connection)->rfp.RDMA_recv_buf[VC_FIELD(vc, connection)->rfp.p_RDMA_recv]);
            head = v->head_flag;

            if (*head && VC_FIELD(vc, connection)->rfp.p_RDMA_recv != VC_FIELD(vc, connection)->rfp.p_RDMA_recv_tail)
            {
                size = (*head & FAST_RDMA_SIZE_MASK);
                tail = (VBUF_FLAG_TYPE *) (v->buffer + size);
                /* If the tail has not received yet, than go ahead and
                ** poll next connection */
                if (*head != *tail) {
                    continue;
                }

                DEBUG_PRINT("Get one!\n");

                if (++VC_FIELD(vc, connection)->rfp.p_RDMA_recv >= num_rdma_buffer)
                {
                    VC_FIELD(vc, connection)->rfp.p_RDMA_recv = 0;
                }

                v->iheader = v->buffer;
                v->pheader = v->iheader + IB_PKT_HEADER_LENGTH;
                v->content_size = size - IB_PKT_HEADER_LENGTH;
                *head = 0;

                seq = GetSeqNumVbuf(v);
                DEBUG_PRINT("[get rfp packet] seq = %d, expect %d, seqnum in ib = %d, v->iheader->type = %d\n", seq, 
                VC_FIELD(vc, seqnum_recv), ((MPIDI_nem_ib_pkt_comm_header *)v->iheader)->seqnum, 
                ((MPIDI_nem_ib_pkt_comm_header *)v->iheader)->type);

                if (seq == VC_FIELD(vc, seqnum_recv))
                {
                    type = T_CHANNEL_EXACT_ARRIVE;
                    ++VC_FIELD(vc, seqnum_recv);
                    *vbuf_ptr = v;
                    *vc_ptr = v->vc;
                    goto fn_exit;
                }
                else if (seq == PKT_NO_SEQ_NUM)
                {
                    type = T_CHANNEL_CONTROL_MSG_ARRIVE;
                    *vbuf_ptr = v;
                    *vc_ptr = v->vc;
                    goto fn_exit;
                }
                else
                {
                    VQUEUE_ENQUEUE(VC_FIELD(vc, cmanager), INDEX_LOCAL(VC_FIELD(vc, cmanager), 0), v);
                    continue;
                }
            }
            else
            {
                continue;
            }
        }

        if (seq == VC_FIELD( vc, seqnum_recv))
        {
            *vbuf_ptr = VQUEUE_DEQUEUE(VC_FIELD(vc, cmanager), INDEX_LOCAL(VC_FIELD(vc, cmanager), 0));
            *vc_ptr = (*vbuf_ptr)->vc;
            VC_FIELD( vc, seqnum_recv)++;
            type = T_CHANNEL_EXACT_ARRIVE;
            goto fn_exit;
        }
        else if (seq == PKT_NO_SEQ_NUM)
        {
            *vbuf_ptr = VQUEUE_DEQUEUE(VC_FIELD(vc, cmanager), INDEX_LOCAL(VC_FIELD(vc, cmanager), 0));
            *vc_ptr = (*vbuf_ptr)->vc;
            type = T_CHANNEL_CONTROL_MSG_ARRIVE;
            goto fn_exit;
        }
    }

fn_exit:
    return type;
}
/**
 * if there is msg have been received
 */
int MPIDI_nem_ib_waiting_msg(MPIDI_VC_t * vc, vbuf ** vbuf_handle, int blocking)
{
    MPID_nem_ib_channel_manager * cmanager = VC_FIELD(vc, cmanager);
    int i = 0;
    int seq;
    int seq_expected = VC_FIELD( vc, seqnum_recv);
    int type = T_CHANNEL_NO_ARRIVE;

    *vbuf_handle = NULL;

    if (blocking) {
        DEBUG_PRINT("{entering} solve_out_of_order next expected %d, channel %d, head %p (%d)\n",
                VC_FIELD( vc, seqnum_recv), cmanager->num_channels,
                cmanager->msg_channels[0].v_queue_head,
                GetSeqNumVbuf(cmanager->msg_channels[0].v_queue_head));
    }

    for (i=0; i < cmanager->num_channels; ++i)
    {
        seq = GetSeqNumVbuf(cmanager->msg_channels[i].v_queue_head);
        if (seq == seq_expected) {
            *vbuf_handle = VQUEUE_DEQUEUE(cmanager, i);
            type = T_CHANNEL_EXACT_ARRIVE;
            VC_FIELD( vc, seqnum_recv)++;
            goto fn_exit;
        } else if (PKT_NO_SEQ_NUM == seq) {
            *vbuf_handle = VQUEUE_DEQUEUE(cmanager, i);
            type = T_CHANNEL_CONTROL_MSG_ARRIVE;
            goto fn_exit;
        } else if (PKT_IS_NULL == seq) {
            /* Do nothing */
        } else {
            *vbuf_handle = cmanager->msg_channels[i].v_queue_head;
            type = T_CHANNEL_OUT_OF_ORDER_ARRIVE;
        }
    }

    /* Obviously the packet with correct sequence hasn't arrived */
    while (blocking) {
        /* poll local subrails*/
        for (i = 0; i < cmanager->num_local_pollings; ++i) {
            seq = GetSeqNumVbuf(cmanager->msg_channels[INDEX_LOCAL(cmanager,i)].v_queue_head);
            if (seq == seq_expected) {
                *vbuf_handle = VQUEUE_DEQUEUE(cmanager, INDEX_LOCAL(cmanager,i));
                VC_FIELD( vc, seqnum_recv)++;
                type = T_CHANNEL_EXACT_ARRIVE;
                goto fn_exit;
            } else if (seq == PKT_NO_SEQ_NUM) {
                *vbuf_handle = VQUEUE_DEQUEUE(cmanager, INDEX_LOCAL(cmanager,i));
                type = T_CHANNEL_CONTROL_MSG_ARRIVE;
                goto fn_exit;
            }
            /* rfp codes */
            else if (VC_FIELD(vc, connection)->rfp.in_polling_set) {
                *vbuf_handle = MPIDI_CH3I_RDMA_poll(vc);
                seq = GetSeqNumVbuf(*vbuf_handle);
                if (seq == seq_expected) {
                    type = T_CHANNEL_EXACT_ARRIVE;
                    VC_FIELD(vc, seqnum_recv)++;
                    goto fn_exit;
                }
                else if( seq == PKT_NO_SEQ_NUM) {
                    type = T_CHANNEL_CONTROL_MSG_ARRIVE;
                    goto fn_exit;
                } else if (*vbuf_handle != NULL){
                    VQUEUE_ENQUEUE(cmanager, INDEX_LOCAL(cmanager,i), *vbuf_handle);
                    *vbuf_handle = NULL;
                }
            }
        }

        type = MPIDI_nem_ib_cq_poll(vbuf_handle, vc, 0, blocking);
        if (type != T_CHANNEL_NO_ARRIVE) {
            switch(type) {
                case (T_CHANNEL_EXACT_ARRIVE):
                    goto fn_exit;
                case (T_CHANNEL_OUT_OF_ORDER_ARRIVE):
                    continue;
                case (T_CHANNEL_CONTROL_MSG_ARRIVE):
                    goto fn_exit;
                default:
                    ibv_error_abort(GEN_ASSERT_ERR, "Unexpected return type\n");
                    break;
            }
        } else {

        }
    }
fn_exit:
    if (blocking) {
        DEBUG_PRINT("{return} solve_out_of_order, type %d, next expected %d\n",
                type, VC_FIELD( vc, seqnum_recv));
    }
    return type;
}

/*
 * MPIDI_CH3I_Request_adjust_iov()
 *
 * Adjust the iovec in the request by the supplied number of bytes.  If the
 * iovec has been consumed, return true; otherwise return
 * false.
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_request_adjust_iov
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_request_adjust_iov(MPID_Request * req, MPIDI_msg_sz_t nb)
{
    int offset = req->dev.iov_offset;
    const int count = req->dev.iov_count;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_REQUEST_ADJUST_IOV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_REQUEST_ADJUST_IOV);

    while (offset < count) {
        if (req->dev.iov[offset].MPL_IOV_LEN <= (unsigned int) nb) {
            nb -= req->dev.iov[offset].MPL_IOV_LEN;
            ++offset;
        } else {
            req->dev.iov[offset].MPL_IOV_BUF =
                ((char *) req->dev.iov[offset].MPL_IOV_BUF) + nb;
            req->dev.iov[offset].MPL_IOV_LEN -= nb;
            req->dev.iov_offset = offset;
            MPIDI_DBG_PRINTF((60, FCNAME, "adjust_iov returning FALSE"));
            MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_REQUEST_ADJUST_IOV);
            return FALSE;
        }
    }

    req->dev.iov_offset = 0;

    MPIDI_DBG_PRINTF((60, FCNAME, "adjust_iov returning TRUE"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_REQUEST_ADJUST_IOV);
    return TRUE;
}

