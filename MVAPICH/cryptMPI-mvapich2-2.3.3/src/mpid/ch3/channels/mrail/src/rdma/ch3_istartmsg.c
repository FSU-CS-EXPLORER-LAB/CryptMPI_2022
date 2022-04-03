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

#include "mpidi_ch3_impl.h"
#include "rdma_impl.h"

#ifdef MPICH_DBG_OUTPUT
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#endif

#undef DEBUG_PRINT
#define DEBUG_PRINT(args...)                                  \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);                                    \
} while (0)

#ifndef DEBUG
#undef DEBUG_PRINT
#define DEBUG_PRINT(args...)
#endif

#undef FUNCNAME
#define FUNCNAME create_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPID_Request * create_request(void * hdr, MPIDI_msg_sz_t hdr_sz,
					    MPIU_Size_t nb)
{
    MPIDI_STATE_DECL(MPID_STATE_CREATE_REQUEST);
    MPIDI_FUNC_ENTER(MPID_STATE_CREATE_REQUEST);

    MPID_Request* sreq = MPID_Request_create();
    /* --BEGIN ERROR HANDLING-- */
    if (sreq == NULL)
    {
        MPIDI_FUNC_EXIT(MPID_STATE_CREATE_REQUEST);
        return NULL;
    }
    /* --END ERROR HANDLING-- */
    MPIU_Object_set_ref(sreq, 2);
    sreq->kind = MPID_REQUEST_SEND;
    MV2_INC_NUM_POSTED_SEND();
#ifdef _ENABLE_CUDA_
    sreq->dev.pending_pkt = MPIU_Malloc(hdr_sz - nb);
    MPIU_Memcpy(sreq->dev.pending_pkt, (char *)hdr + nb, hdr_sz - nb);
    sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)((char *)sreq->dev.pending_pkt);
#else
    MPIU_Memcpy(&sreq->dev.pending_pkt, hdr, sizeof(MPIDI_CH3_Pkt_t));
    sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)((char *) &sreq->dev.pending_pkt + nb);
#endif
    sreq->ch.reqtype = REQUEST_NORMAL;
    sreq->dev.iov[0].MPL_IOV_LEN = hdr_sz - nb;
    sreq->dev.iov_count = 1;
    sreq->dev.OnDataAvail = 0;

    MPIDI_FUNC_EXIT(MPID_STATE_CREATE_REQUEST);
    return sreq;
}

int MPIDI_CH3_SMP_iStartMsg(MPIDI_VC_t * vc, void *pkt,
                                          MPIDI_msg_sz_t pkt_sz,
                                          MPID_Request ** sreq_ptr);
/*
 * MPIDI_CH3_iStartMsg() attempts to send the message immediately.  If the
 * entire message is successfully sent, then NULL is returned.  Otherwise a
 * request is allocated, the header is copied into the request, and a pointer
 * to the request is returned.  An error condition also results in a request be
 * allocated and the errror being returned in the status field of the
 * request.
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iStartMsg
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_iStartMsg(MPIDI_VC_t * vc, void *pkt, MPIDI_msg_sz_t pkt_sz,
                        MPID_Request ** sreq_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    MPL_IOV iov[1];
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_ISTARTMSG);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_ISTARTMSG);

    MPIU_DBG_PRINTF(("ch3_istartmsg\n"));
    MPIDI_DBG_PRINTF((50, FCNAME, "entering"));

#if defined(CKPT)
	MPIDI_CH3I_CR_lock();
#endif

    /* If send queue is empty attempt to send
       data, queuing any unsent data. */
    if (SMP_INIT && vc->smp.local_nodes >= 0 &&
        vc->smp.local_nodes != g_smpi.my_local_id) {
        mpi_errno = MPIDI_CH3_SMP_iStartMsg(vc, pkt, pkt_sz,sreq_ptr);
        MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_ISTARTMSG);
#ifdef CKPT
	MPIDI_CH3I_CR_unlock();
#endif
        return(mpi_errno);
    }

#ifdef CKPT
    /*Detect whether the packet is CTS*/
    MPIDI_CH3_Pkt_t *upkt = (MPIDI_CH3_Pkt_t *)pkt;
    if (upkt->type == MPIDI_CH3_PKT_RNDV_CLR_TO_SEND) {
        MPIDI_CH3_Pkt_rndv_clr_to_send_t * cts_pkt = &(upkt->rndv_clr_to_send);
        if (cts_pkt->rndv.protocol == MV2_RNDV_PROTOCOL_RPUT) {
            /*If using rput protocol, keep track of the request*/
            MPID_Request *rreq;
            MPID_Request_get_ptr(cts_pkt->receiver_req_id, rreq);
            MPIDI_CH3I_CR_req_enqueue(rreq, vc);
        }
    }
#endif

    /*CM code*/
    if ((vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE 
#ifdef _ENABLE_XRC_
            || (USE_XRC && VC_XST_ISUNSET (vc, XF_SEND_IDLE))
#endif
            ) || !MPIDI_CH3I_CM_SendQ_empty(vc)) {
        /*Request need to be queued*/
        MPIDI_DBG_PRINTF((55, FCNAME, "not connected, enqueuing"));
        sreq = create_request(pkt, pkt_sz, 0);
        MPIDI_CH3I_CM_SendQ_enqueue(vc, sreq);
        if (vc->ch.state == MPIDI_CH3I_VC_STATE_UNCONNECTED)  {
            MPIDI_CH3I_CM_Connect(vc);
        }
        goto fn_exit;
    }

    if (MPIDI_CH3I_SendQ_empty(vc)) {   /* MT */
        int nb;
        int pkt_len;
        vbuf *buf;

        /* MT - need some signalling to lock down our right to use the
           channel, thus insuring that the progress engine does also try to
           write */

        iov[0].MPL_IOV_BUF = pkt;
        iov[0].MPL_IOV_LEN = pkt_sz;
        pkt_len = pkt_sz;

        /* TODO: Codes to send pkt through send/recv path */
        mpi_errno =
            MPIDI_CH3I_MRAILI_Eager_send(vc, iov, 1, pkt_len, &nb, &buf);
        DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno, nb);

        if (mpi_errno == MPI_SUCCESS) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
            goto fn_exit;
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            /* fast rdma ok but cannot send: there is no send wqe available */
            /* sreq = create_request(pkt, pkt_sz, 0);
            buf->sreq = (void *) sreq;   */ 
            mpi_errno = MPI_SUCCESS;
            goto fn_exit;
        } else {
            sreq = MPID_Request_create();
            if (sreq == NULL) {
                mpi_errno =
                    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
                                         FCNAME, __LINE__,
                                         MPI_ERR_OTHER, "**nomem", 0);
                goto fn_exit;
            }
            sreq->kind = MPID_REQUEST_SEND;
            MV2_INC_NUM_POSTED_SEND();
            MPID_cc_set(&sreq->cc, 0);
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
            PRINT_DEBUG(DEBUG_SHM_verbose>1,
                    "Enqueue send to rank: %d, sreq: %p, type: %d, ch.reqtype: %d\n",
                    vc->pg_rank, sreq, MPIDI_Request_get_type(sreq), sreq->ch.reqtype);
        }
    } else {
        sreq = create_request(pkt, pkt_sz, 0);
        MPIDI_CH3I_SendQ_enqueue(vc, sreq);
        PRINT_DEBUG(DEBUG_SHM_verbose>1,
                "Eqnueue send to rank: %d, sreq: %p, type: %d, ch.reqtype: %d\n",
                vc->pg_rank, sreq, MPIDI_Request_get_type(sreq), sreq->ch.reqtype);
    }

  fn_exit:
    *sreq_ptr = sreq;
#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif

    DEBUG_PRINT("Exiting istartmsg\n");
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_ISTARTMSG);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_SMP_iStartMsg
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_SMP_iStartMsg(MPIDI_VC_t * vc, void *pkt,
                                          MPIDI_msg_sz_t pkt_sz,
                                          MPID_Request ** sreq_ptr)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSG);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSG);
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    MPL_IOV iov[1];
    MPIDI_CH3_Pkt_send_t *pkt_header;

    DEBUG_PRINT("entering ch3_istartmsg\n");

    pkt_header = (MPIDI_CH3_Pkt_send_t *)pkt;

    /* If send queue is empty attempt to send
       data, queuing any unsent data. */
    if (MPIDI_CH3I_SMP_SendQ_empty(vc)) {       /* MT */
        int nb;

        /* MT - need some signalling to lock down our right to use the
           channel, thus insuring that the progress engine does also try to
           write */

        iov[0].MPL_IOV_BUF = pkt;
        iov[0].MPL_IOV_LEN = pkt_sz;

        if (pkt_header->type == MPIDI_CH3_PKT_RNDV_R3_DATA)
        {
            MPIDI_CH3I_SMP_writev_rndv_header(vc, iov, 1, &nb);
        }
        else
        {
            MPIDI_CH3I_SMP_writev(vc, iov, 1, &nb);
        }
#ifdef CKPT
		MPIDI_CH3I_MRAILI_Pkt_comm_header* p = (MPIDI_CH3I_MRAILI_Pkt_comm_header*)pkt;
		if( p->type >= MPIDI_CH3_PKT_CM_SUSPEND && 
			p->type<= MPIDI_CH3_PKT_CR_REMOTE_UPDATE ) {
			DEBUG_PRINT("%s [%d => %d]: imm-write pkt %s(%d), ret nb=%d,pkt-size=%d\n", __func__, 
			MPIDI_Process.my_pg_rank, vc->pg_rank,  MPIDI_CH3_Pkt_type_to_string[p->type],
			p->type, nb, pkt_sz );
		}
#endif  // CKPT
        if (nb != pkt_sz) 
        {
            sreq = create_request(pkt, pkt_sz, nb);
            if(pkt_header->type == MPIDI_CH3_PKT_RNDV_R3_DATA)
            { 
                sreq->ch.reqtype = REQUEST_RNDV_R3_HEADER;
            }

            MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
            vc->smp.send_active = sreq;

            PRINT_DEBUG(DEBUG_SHM_verbose>1,
                    "send to %d delayed, request enqueued: %p, type: %d, pkt_sz: %ld, ch.reqtype: %d\n",
                    vc->pg_rank, sreq, MPIDI_Request_get_type(sreq), pkt_sz, sreq->ch.reqtype);
        }
#if defined(DEBUG)
        else
        {
            DEBUG_PRINT("data sent immediately.\n");
        }
#endif /* defined(DEBUG) */
    } else {
        sreq = create_request(pkt, pkt_sz, 0);
        if(pkt_header->type == MPIDI_CH3_PKT_RNDV_R3_DATA) {
            sreq->ch.reqtype = REQUEST_RNDV_R3_HEADER;
        }

        MPIDI_CH3I_SMP_SendQ_enqueue(vc, sreq);
        PRINT_DEBUG(DEBUG_SHM_verbose>1,
                "send to %d delayed, request enqueued: %p, type: %d, pkt_sz: %ld, ch.reqtype: %d\n",
                vc->pg_rank, sreq, MPIDI_Request_get_type(sreq), pkt_sz, sreq->ch.reqtype);
#ifdef CKPT  
		MPIDI_CH3I_MRAILI_Pkt_comm_header* p = (MPIDI_CH3I_MRAILI_Pkt_comm_header*)pkt;
		if( p->type >= MPIDI_CH3_PKT_CM_SUSPEND && 
			p->type<= MPIDI_CH3_PKT_CR_REMOTE_UPDATE ) 
		{
			DEBUG_PRINT("%s [%d => %d]: Enqueue:  pkt %s(%d), pkt-size=%d\n", __func__, 
				MPIDI_Process.my_pg_rank, vc->pg_rank,  MPIDI_CH3_Pkt_type_to_string[p->type],
				p->type, pkt_sz );
		} 
#endif   // end of CKPT
    }

fn_exit:
    *sreq_ptr = sreq;
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSG);
    return mpi_errno;

#ifndef CHANNEL_MRAIL
fn_fail:
#endif
    goto fn_exit;
}

