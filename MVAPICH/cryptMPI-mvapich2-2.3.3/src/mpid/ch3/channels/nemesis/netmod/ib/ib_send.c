/*!\file */
/*
 *  (C) 2006 by Argonne National Laboratory.
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

#define _GNU_SOURCE
#include "mpiimpl.h"
#include "mpidimpl.h"
#include "upmi.h"
#include "mpid_nem_impl.h"

#include "ib_device.h"
#include "ib_cm.h"
#include "ib_vc.h"
#include "ib_vbuf.h"
#include "ib_param.h"
#include "ib_send.h"
#include "ib_errors.h"
#include "ib_process.h"
#include "ib_lmt.h"
#include "ib_srq.h"
/* add ib_poll.h for MPIDI_nem_ib_request_adjust_iov */
#include "ib_poll.h"

#define SENDQ_EMPTY(q) GENERIC_Q_EMPTY (q)
#define SENDQ_HEAD(q) GENERIC_Q_HEAD (q)
#define SENDQ_ENQUEUE(qp, ep) GENERIC_Q_ENQUEUE (qp, ep, dev.next)
#define SENDQ_DEQUEUE(qp, ep) GENERIC_Q_DEQUEUE (qp, ep, dev.next)

#define Calculate_IOV_len(_iov, _n_iov, _len)                   \
{   int _i; (_len) = 0;                                         \
    for (_i = 0; _i < (_n_iov); _i ++) {                        \
        (_len) += (_iov)[_i].MPL_IOV_LEN;                      \
    }                                                           \
}


#define GET_SEQ_NUM(_pkt, _seqnum) \
{   \
    if (NULL == (_pkt)) \
        _seqnum = -101;    \
    switch (((MPIDI_nem_ib_pkt_comm_header *)(_pkt))->type) { \
    case MPIDI_CH3_PKT_EAGER_SEND:  \
    case MPIDI_CH3_PKT_READY_SEND:  \
    case MPIDI_CH3_PKT_EAGER_SYNC_SEND: \
    case MPIDI_NEM_PKT_LMT_RTS:    \
    case MPIDI_NEM_PKT_LMT_CTS:  \
    case MPIDI_NEM_PKT_LMT_DONE: \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_send_t *) (_pkt))->seqnum;    \
            break;  \
        }   \
    case MPIDI_CH3_PKT_RNDV_CLR_TO_SEND:    \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_rndv_clr_to_send_t *) (_pkt))->seqnum; \
            break; \
        }   \
    default:    \
        _seqnum =  -1;  \
    }   \
}

#undef FUNCNAME
#define FUNCNAME create_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline MPID_Request * create_request(MPL_IOV * iov, int iov_count,
                        int iov_offset, MPIU_Size_t nb)
{
    MPID_Request * sreq;
    int i;
    MPIDI_STATE_DECL(MPID_STATE_CREATE_REQUEST);

    MPIDI_FUNC_ENTER(MPID_STATE_CREATE_REQUEST);

    sreq = MPID_Request_create();
    /* --BEGIN ERROR HANDLING-- */
    if (sreq == NULL)
        return NULL;
    /* --END ERROR HANDLING-- */
    MPIU_Object_set_ref(sreq, 2);
    sreq->kind = MPID_REQUEST_SEND;

    MPIU_Memcpy(sreq->dev.iov, iov, iov_count * sizeof(MPL_IOV));

    if (iov_offset == 0)
    {
    /*
        MPIU_Assert(iov[0].MPL_IOV_LEN == sizeof(MPIDI_CH3_Pkt_t));
    */
    
        MPIU_Memcpy(&sreq->dev.pending_pkt, iov[0].MPL_IOV_BUF, sizeof(MPIDI_CH3_Pkt_t));
        sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST) &sreq->dev.pending_pkt;
    }
    sreq->dev.iov[iov_offset].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)((char *)
            sreq->dev.iov[iov_offset].MPL_IOV_BUF + nb);
    sreq->dev.iov[iov_offset].MPL_IOV_LEN -= nb;
    sreq->dev.iov_offset = iov_offset;
    sreq->dev.iov_count = iov_count;
    sreq->dev.OnDataAvail = 0;

    MPIDI_FUNC_EXIT(MPID_STATE_CREATE_REQUEST);
    return sreq;
}


static inline void MPIDI_nem_ib_POST_SR(vbuf *_v, MPID_nem_ib_connection_t *_c, int _rail, char *err_string)
{
	int __ret;

	if((_v->desc.sg_entry.length <= rdma_max_inline_size)
			&& (_v->desc.u.sr.opcode != IBV_WR_RDMA_READ) )
	{
	   _v->desc.u.sr.send_flags = (enum ibv_send_flags)(IBV_SEND_SIGNALED | IBV_SEND_INLINE);


	} else {
		_v->desc.u.sr.send_flags = IBV_SEND_SIGNALED ;
	}

	if (_rail != _v->rail)
	{
			DEBUG_PRINT("[%s:%d] rail %d, vrail %d\n", __FILE__, __LINE__,(_rail), (_v)->rail);
			MPIU_Assert((_rail) == (_v)->rail);
	}

	process_info.global_used_send_cq++;

	__ret = ibv_post_send( _c->rails[_rail].qp_hndl, & (_v->desc.u.sr), &(_v->desc.y.bad_sr) );
	if(__ret) {
		ibv_error_abort(-1, err_string);
	}
}




/* to handle Send Q overflow, we maintain an extended send queue
 * above the HCA.  This permits use to have a virtually unlimited send Q depth
 * (limited by number of vbufs available for send)
 */
#undef FUNCNAME
#define FUNCNAME MRAILI_Ext_sendq_enqueue
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MRAILI_Ext_sendq_enqueue(MPIDI_VC_t *c,
                                            int rail,
                                            vbuf * v)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);

    v->desc.next = NULL;

    if (VC_FIELD(c, connection)->rails[rail].ext_sendq_head == NULL) {
        VC_FIELD(c, connection)->rails[rail].ext_sendq_head = v;
    } else {
        VC_FIELD(c, connection)->rails[rail].ext_sendq_tail->desc.next = v;
    }
    VC_FIELD(c, connection)->rails[rail].ext_sendq_tail = v;
    DEBUG_PRINT("[ibv_send] enqueue, head %p, tail %p\n",
            VC_FIELD(c, connection)->rails[rail].ext_sendq_head,
            VC_FIELD(c, connection)->rails[rail].ext_sendq_tail);
    
    VC_FIELD(c, connection)->rails[rail].ext_sendq_size++;
    if (VC_FIELD(c, connection)->rails[rail].ext_sendq_size > rdma_rndv_ext_sendq_size) {
        VC_FIELD(c, force_rndv) = 1;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_EXT_SENDQ_ENQUEUE);
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
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_EXT_SENDQ_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_EXT_SENDQ_SEND);

    vbuf *v;
    char no_cq_overflow = 1;
    int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);
    if ((NULL != hca_list[hca_num].send_cq_hndl) &&
        (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
        /* We are monitoring CQ's and there is CQ overflow */
        no_cq_overflow = 0;
    }

    while (VC_FIELD(c, connection)->rails[rail].send_wqes_avail
            && no_cq_overflow
            && VC_FIELD(c, connection)->rails[rail].ext_sendq_head) {
        v = VC_FIELD(c, connection)->rails[rail].ext_sendq_head;
        VC_FIELD(c, connection)->rails[rail].ext_sendq_head = v->desc.next;
        if (v == VC_FIELD(c, connection)->rails[rail].ext_sendq_tail) {
            VC_FIELD(c, connection)->rails[rail].ext_sendq_tail = NULL;
        }
        v->desc.next = NULL;
        VC_FIELD(c, connection)->rails[rail].send_wqes_avail--;
        VC_FIELD(c, connection)->rails[rail].ext_sendq_size--;


        MPIDI_nem_ib_POST_SR(v, VC_FIELD(c, connection), rail, "Mrail_post_sr (MRAILI_Ext_sendq_send)");
    }

    DEBUG_PRINT( "[ibv_send] dequeue, head %p, tail %p\n",
        VC_FIELD(c, connection)->rails[rail].ext_sendq_head,
        VC_FIELD(c, connection)->rails[rail].ext_sendq_tail);

    if (VC_FIELD(c, connection)->rails[rail].ext_sendq_size <= rdma_rndv_ext_sendq_size) {
        VC_FIELD(c, force_rndv) = 0;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_EXT_SENDQ_SEND);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_NEM_IB_PACKETIZED_SEND
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_packetized_send(MPIDI_VC_t * vc, MPID_Request * sreq)
{
    MPL_IOV iov[MPL_IOV_LIMIT + 1];

    MPIDI_CH3_Pkt_packetized_send_start_t send_start;
    MPIDI_CH3_Pkt_packetized_send_data_t pkt_head;

    vbuf *buf;
    int mpi_errno = MPI_SUCCESS;
    int n_iov;
    int pkt_len;
    int msg_buffered = 0;
    int nb;
    int complete;


    MPIDI_STATE_DECL(MPIDI_NEM_IB_PACKETIZED_SEND);
    MPIDI_FUNC_ENTER(MPIDI_NEM_IB_PACKETIZED_SEND);

    MPIDI_Pkt_init(&send_start, MPIDI_CH3_PKT_PACKETIZED_SEND_START);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_packetized_send_start_t);
    iov[0].MPL_IOV_BUF = (void*) &send_start;
    MPIU_Memcpy(&iov[1], sreq->dev.iov, sreq->dev.iov_count * sizeof(MPL_IOV));
    n_iov = 1 + sreq->dev.iov_count;

    /* origin head size contains size of the header in PACKETIZED_SEND_START pkt
    ** payload. In both iSendContig and iSendNonContig header size is constant
    ** i.e sizeof(MPIDI_CH3_Pkt_t)
    */
    send_start.origin_head_size = sizeof(MPIDI_CH3_Pkt_t);

    Calculate_IOV_len(iov, n_iov, pkt_len);
    
    mpi_errno =
        MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);

    if (MPI_SUCCESS != mpi_errno && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
        sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        MPID_Request_complete(sreq);
        goto fn_exit;
    } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
        msg_buffered = 1;
    }
    nb -= sizeof(MPIDI_CH3_Pkt_packetized_send_start_t) + IB_PKT_HEADER_LENGTH;

    MPIDI_Pkt_init(&pkt_head, MPIDI_CH3_PKT_PACKETIZED_SEND_DATA);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_packetized_send_data_t);
    iov[0].MPL_IOV_BUF = (void*) &pkt_head;

    do{
        while (!MPIDI_nem_ib_request_adjust_iov(sreq, nb)) {
            MPIDI_VC_FAI_send_seqnum(vc, seqnum);
            MPIDI_Pkt_set_seqnum(&pkt_head, seqnum);
            MPIDI_Request_set_seqnum(sreq, seqnum);

            MPIU_Memcpy((void *) &iov[1],
                   &sreq->dev.iov[sreq->dev.iov_offset],
                   (sreq->dev.iov_count -
                    sreq->dev.iov_offset) * sizeof(MPL_IOV));
            n_iov = sreq->dev.iov_count - sreq->dev.iov_offset + 1;

            Calculate_IOV_len(iov, n_iov, pkt_len);

            mpi_errno =
                MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb,
                        &buf);
            DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno,
                    nb);
            MPIU_Assert(NULL == buf->sreq);

            if (MPI_SUCCESS != mpi_errno
                && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
                sreq->status.MPI_ERROR = MPI_ERR_INTERN;
                MPID_Request_complete(sreq);
                goto fn_exit;
            } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                msg_buffered = 1;
            }

            nb -= sizeof(MPIDI_CH3_Pkt_packetized_send_data_t) + IB_PKT_HEADER_LENGTH;
        }
        if (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV) {
            MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
            nb = 0;
            complete = 0;
        } else {
            complete = 1;
        }
    } while (!complete);

    if (msg_buffered) {
        mpi_errno = MPI_MRAIL_MSG_QUEUED;
        buf->sreq = (void *) sreq;
    } else {
        MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
    }

fn_exit:
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPIDI_NEM_IB_PACKETIZED_SEND);
    return mpi_errno;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_fill_start_buffer
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_fill_start_buffer(vbuf * v,
                             MPL_IOV * iov,
                             int n_iov)
{
    MPIDI_STATE_DECL(MPID_STATE_FILL_START_BUFFER);
    MPIDI_FUNC_ENTER(MPID_STATE_FILL_START_BUFFER);

    int i = 0;
    int avail = VBUF_BUFFER_SIZE - v->content_size;
    void *ptr = (v->buffer + v->content_size);
    int len = 0;

    DEBUG_PRINT("buffer: %p, content size: %d\n", v->buffer, v->content_size);

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

    v->content_size += len;

    MPIDI_FUNC_EXIT(MPID_STATE_FILL_START_BUFFER);
    return len;
}

/**
 * Select a rail using a scheduling policy.
 */
int MPIDI_nem_ib_send_select_rail(MPIDI_VC_t *vc) {
	static int i = 0;
	if ( sm_scheduling == ROUND_ROBIN) {
		i = (i+1) % rdma_num_rails;
		return i;
	} else {
		return 0;
	}

}


#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_get_vbuf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
vbuf* MPIDI_nem_ib_get_vbuf(MPIDI_VC_t * vc, int pkt_len)
{
    MPIDI_STATE_DECL(MPID_STATE_NEM_IB_GET_VBUF);
    MPIDI_FUNC_ENTER(MPID_STATE_NEM_IB_GET_VBUF);

    vbuf* temp_v = NULL;

    /* if there already wasn't a vbuf that could
     * hold our packet we need to allocate a
     * new one
     */
    if(NULL == temp_v) {
        temp_v = get_vbuf();

        DEBUG_PRINT("buffer is %p\n", temp_v->buffer);
        DEBUG_PRINT("pheader buffer is %p\n", temp_v->pheader);

        /* vbuf members: rail eager content_size */
        temp_v->rail = MPIDI_nem_ib_send_select_rail(vc);
        temp_v->eager = 1;
        temp_v->content_size = IB_PKT_HEADER_LENGTH;

        DEBUG_PRINT("incrementing the outstanding eager vbufs: eager %d\n",
                VC_FIELD(vc, connection)->outstanding_eager_vbufs);

        VC_FIELD(vc, connection)->outstanding_eager_vbufs++;
    }
    MPIU_Assert(temp_v != NULL);

    MPIDI_FUNC_EXIT(MPID_STATE_NEM_IB_GET_VBUF);
    return temp_v;
}




#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_post_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_post_send(MPIDI_VC_t * vc, vbuf * v, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_IB_POST_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_IB_POST_SEND);

    char cq_overflow = 0;

    /* v->pheader */
    MPIDI_nem_ib_pkt_comm_header *p = v->iheader;
    DEBUG_PRINT(
                "[post send] credit %d,type %d, "
                "backlog %d, wqe %d, nb will be %d, seqnum = %d\n",
                VC_FIELD( vc, connection)->srp.credits[rail].remote_credit,
                p->type,
                VC_FIELD( vc, connection)->srp.credits[0].backlog.len,
                VC_FIELD( vc, connection)->rails[rail].send_wqes_avail,
                v->desc.sg_entry.length, p->seqnum);



    if (VC_FIELD( vc, connection)->srp.credits[rail].remote_credit > 0
        || p->type == MPIDI_CH3_PKT_NOOP
	) {

        /* in ib_cm.h */
        PACKET_SET_CREDIT(p, VC_FIELD( vc, connection), rail);

        if (p->type != MPIDI_CH3_PKT_NOOP)
        {
            VC_FIELD( vc, connection)->srp.credits[rail].remote_credit--;
        }

        v->vc = (void *) vc;

        int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);
        if ((NULL != hca_list[hca_num].send_cq_hndl) &&
            (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
            /*We are monitoring CQ's and there is CQ overflow*/
            cq_overflow = 1;
        }

        if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail || cq_overflow)
        {
            MRAILI_Ext_sendq_enqueue(vc, rail, v);
            MPIDI_FUNC_EXIT(MPID_STATE_IB_POST_SEND);
            return MPI_MRAIL_MSG_QUEUED;
        }
        else
        {
            VC_FIELD(vc, connection)->rails[rail].send_wqes_avail--;
            MPIDI_nem_ib_POST_SR(v, VC_FIELD( vc, connection), rail, "ibv_post_sr (post_send_desc)");
        }
    }
    else
    {
        /* credits is 0. no available buffer in remote */
        /* enqueue to backlog */
        ibv_backlog_queue_t *q = &(VC_FIELD(vc, connection)->srp.credits[rail].backlog);
        BACKLOG_ENQUEUE(q, v);
        MPIDI_FUNC_EXIT(MPID_STATE_IB_POST_SEND);
        return MPI_MRAIL_MSG_QUEUED;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_IB_POST_SEND);
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_eager_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_eager_send(MPIDI_VC_t * vc,
                        MPL_IOV * iov,
                        int n_iov,
                        int pkt_len,
                        int *num_bytes_ptr,
                        vbuf **buf_handle)
{
    int mpi_errno = MPI_SUCCESS;
    uint16_t seqnum;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_EAGER_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_EAGER_SEND);

    /* first we check if we can take the RDMA FP */
    if(MPIDI_nem_ib_fast_rdma_ok(vc, pkt_len)) {
        *num_bytes_ptr = pkt_len;
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_EAGER_SEND);
        return MPIDI_nem_ib_fast_rdma_send_complete(vc, iov, n_iov, num_bytes_ptr, buf_handle);
    }


    /* otherwise we can always take the send/recv path */
    vbuf* v = MPIDI_nem_ib_get_vbuf(vc, pkt_len);
    DEBUG_PRINT("[eager send]vbuf addr %p, buffer: %p\n", v, v->buffer);
    *num_bytes_ptr = MPIDI_nem_ib_fill_start_buffer(v, iov, n_iov) + IB_PKT_HEADER_LENGTH;

    /* send the buffer if we aren't trying to coalesce it */
        vbuf_init_send(v, *num_bytes_ptr, v->rail);

        /* whether post_send = post_srq_send or post_send is depending on
         * hca type and has_srq
         * currently directly using post_send
         */
        MPIDI_nem_ib_pkt_comm_header *p = v->iheader;
        p->type = ((MPIDI_CH3_Pkt_send_t*)v->pheader)->type;
        MPIDI_nem_ib_get_send_seqnum(vc, seqnum);
        MPIDI_nem_ib_set_seqnum(p, seqnum);
        process_info.post_send(vc, v, v->rail);

    *buf_handle = v;

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_EAGER_SEND);
    return mpi_errno;

}

#undef FUNCNAME
#define FUNCNAME isend_update_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static void isend_update_request(MPID_Request* sreq, void* pkt, int pkt_sz, int nb)
{
    MPIDI_STATE_DECL(MPID_STATE_ISEND_UPDATE_REQUEST);
    MPIDI_FUNC_ENTER(MPID_STATE_ISEND_UPDATE_REQUEST);
    MPIU_Memcpy(&sreq->dev.pending_pkt, pkt, sizeof(MPIDI_CH3_Pkt_t));
    sreq->dev.iov[0].MPL_IOV_BUF = (char *) &sreq->dev.pending_pkt + nb;
    sreq->dev.iov[0].MPL_IOV_LEN = pkt_sz - nb;
    sreq->dev.iov_count = 1;
    sreq->dev.iov_offset = 0;
    MPIDI_FUNC_EXIT(MPID_STATE_ISEND_UPDATE_REQUEST);

}


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_iSendContig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_iSendContig(MPIDI_VC_t *vc, MPID_Request *sreq, void *hdr, MPIDI_msg_sz_t hdr_sz,
                                void *data, MPIDI_msg_sz_t data_sz)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_ISENDCONTIGMSG);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_ISENDCONTIGMSG);

    MPL_IOV iov[2];
    int complete;
    int n_iov=2;
    void *databuf = NULL;

#ifdef ENABLE_CHECKPOINTING
    MPID_nem_ib_vc_area *vc_ib = VC_IB(vc);
    if (!MPID_nem_ib_vc_send_paused(vc_ib)) {
#endif
    /* The RDMA implementation uses a fixed length header, the size of which is the maximum of all possible packet headers */

    /* check whether sendq is empty */
    if (SENDQ_EMPTY(VC_FIELD(vc, send_queue))) {
        int nb;
        vbuf *buf;
        int pkt_len;
        MPIDI_DBG_PRINTF((55, FCNAME,
                          "send queue empty, attempting to write"));

        /* the channel use fixed header size*/
        hdr_sz = sizeof(MPIDI_CH3_Pkt_t);
        /* MT: need some signalling to lock down our right to use the channel, thus insuring that the progress engine does
           also try to write */
        iov[0].MPL_IOV_BUF = hdr;
        iov[0].MPL_IOV_LEN = hdr_sz;
        if (data_sz == 0) {
            n_iov = 1;
        } else {
        iov[1].MPL_IOV_BUF = data;
        iov[1].MPL_IOV_LEN = data_sz;
        }
        Calculate_IOV_len(iov, n_iov, pkt_len);

        if (pkt_len > MRAIL_MAX_EAGER_SIZE) {
          MPIU_Memcpy(sreq->dev.iov, iov, n_iov * sizeof(MPL_IOV));
            sreq->dev.iov_count = n_iov;
            mpi_errno = MPIDI_nem_ib_packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            }
            goto fn_exit;
        }

        if (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV) {
            /*reload iov */
            void *tmpbuf;
            int iter_iov;

            tmpbuf = MPIU_Malloc(sreq->dev.segment_size + pkt_len);
            databuf = tmpbuf;
            pkt_len = 0;
            /* First copy whatever has already been in iov set */
            for (iter_iov = 0; iter_iov < n_iov; iter_iov++) {
              MPIU_Memcpy(tmpbuf, iov[iter_iov].MPL_IOV_BUF,
                       iov[iter_iov].MPL_IOV_LEN);
                tmpbuf = (void *) ((unsigned long) tmpbuf +
                                   iov[iter_iov].MPL_IOV_LEN);
                pkt_len += iov[iter_iov].MPL_IOV_LEN;
            }
            DEBUG_PRINT("Pkt len after first stage %d\n", pkt_len);
            /* Second reload iov and copy */
            do {
                sreq->dev.iov_count = MPL_IOV_LIMIT;
                mpi_errno = MPIDI_CH3U_Request_load_send_iov(sreq,
                                                             sreq->dev.iov,
                                                             &sreq->dev.
                                                             iov_count);
                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno != MPI_SUCCESS) {
                    MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ch3|loadsendiov");
                }
                for (iter_iov = 0; iter_iov < sreq->dev.iov_count;
                     iter_iov++) {
                  MPIU_Memcpy(tmpbuf, sreq->dev.iov[iter_iov].MPL_IOV_BUF,
                           sreq->dev.iov[iter_iov].MPL_IOV_LEN);
                    tmpbuf =
                        (void *) ((unsigned long) tmpbuf +
                                  sreq->dev.iov[iter_iov].MPL_IOV_LEN);
                    pkt_len += sreq->dev.iov[iter_iov].MPL_IOV_LEN;
                }
            } while (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV);
            iov[0].MPL_IOV_BUF = databuf;
            iov[0].MPL_IOV_LEN = pkt_len;
            n_iov = 1;
        }
        if (pkt_len > MRAIL_MAX_EAGER_SIZE) {
          MPIU_Memcpy(sreq->dev.iov, iov, n_iov * sizeof(MPL_IOV));
            sreq->dev.iov_count = n_iov;
            mpi_errno = MPIDI_nem_ib_packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            }
            goto fn_exit;
        }

        DEBUG_PRINT("[send], n_iov: %d, pkt_len %d\n", n_iov, pkt_len);


        mpi_errno = MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);
        DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno, nb);

        if (mpi_errno == MPI_SUCCESS) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);

            if (nb == 0) {
                /* under layer cannot send out the msg because there is no credit or
                 * no send wqe available
                 DEBUG_PRINT("Send 0 bytes\n");
                 create_request(sreq, iov, n_iov, 0, 0);
                 MPIDI_CH3I_SendQ_enqueue(vc, sreq);
                 */
            } else {
                MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
                if (!complete) {
                    /* NOTE: dev.iov_count is used to detect completion instead of
                     * cc because the transfer may be complete, but
                       request may still be active (see MPI_Ssend()) */
                    SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
                    VC_FIELD(vc, send_active) = sreq;
                } else {
                    VC_FIELD(vc, send_active) = SENDQ_HEAD( VC_FIELD(vc, send_queue));
                }
            }
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            buf->sreq = (void *) sreq;
            mpi_errno = MPI_SUCCESS;
        } else {
            /* Connection just failed.  Mark the request complete and return an
             * error. */
            /* vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;*/
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
            /* MT - CH3U_Request_complete performs write barrier */
            MPID_Request_complete(sreq);

        }

    } else {
    	/* Send queue not empty, enqueuing */
        MPIDI_DBG_PRINTF((55, FCNAME, "send queue not empty, enqueuing"));
        isend_update_request(sreq, hdr, hdr_sz, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
    }
#ifdef ENABLE_CHECKPOINTING
    }
    else
    {
        isend_update_request(sreq, hdr, hdr_sz, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, paused_send_queue), sreq);
    }
#endif

fn_exit:
    if (databuf)
        MPIU_Free(databuf);
    /* eager poking */
    /* mpi_errno = MPID_Progress_test(); */
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_ISENDCONTIGMSG);
    return mpi_errno;
fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_iStartContigMsg
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_iStartContigMsg(MPIDI_VC_t *vc, void *hdr, MPIDI_msg_sz_t hdr_sz, void *data, MPIDI_msg_sz_t data_sz, 
                                    MPID_Request **sreq_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    MPL_IOV iov[2];
    int n_iov=2;
    int nb;
    int pkt_len;
    vbuf *buf;

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG);

    /* MT - need some signalling to lock down our right to use the
       channel, thus insuring that the progress engine does also try to
       write */

    /* the channel use fixed header size*/
    hdr_sz = sizeof(MPIDI_CH3_Pkt_t);
    iov[0].MPL_IOV_BUF = hdr;
    iov[0].MPL_IOV_LEN = hdr_sz;
    if(data_sz == 0) {
        n_iov = 1;
        pkt_len = hdr_sz;
    } else {
        iov[1].MPL_IOV_BUF = data;
        iov[1].MPL_IOV_LEN = data_sz;
        Calculate_IOV_len(iov, n_iov, pkt_len);
    }

#ifdef ENABLE_CHECKPOINTING
    MPID_nem_ib_vc_area *vc_ib = VC_IB(vc);
    if (!MPID_nem_ib_vc_send_paused(vc_ib)) {
#endif

    if (SENDQ_EMPTY(VC_FIELD(vc, send_queue))) {

        if (pkt_len > MRAIL_MAX_EAGER_SIZE) {
            sreq = create_request(iov, n_iov, 0, 0);
            mpi_errno = MPIDI_nem_ib_packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            }
            goto fn_exit;
        }


        mpi_errno = MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);
        DEBUG_PRINT("[istartmsg] mpierr %d, nb %d\n", mpi_errno, nb);

        if (mpi_errno == MPI_SUCCESS) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
        /* eager poking */
        /* if (nb == pkt_len)
            mpi_errno = MPID_Progress_test();
         */
            goto fn_exit;
        /*rfp codes*/
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            /* fast rdma ok but cannot send: there is no send wqe available */
            sreq = create_request(iov, n_iov, 0, 0);
            buf->sreq = (void *) sreq;
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
            sreq->cc = 0;
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        }
    } else { /* if(!SENDQ_EMPTY(VC_FIELD(vc, send_queue))) */
        MPIDI_DBG_PRINTF((55, FCNAME,
                          "send in progress, request enqueued"));

        sreq = create_request(iov, n_iov, 0, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
    }
#ifdef ENABLE_CHECKPOINTING
    }
    else /* if(MPID_nem_ib_vc_send_paused(vc)) */
    {
        sreq = create_request(iov, n_iov, 0, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, paused_send_queue), sreq);        
    }
#endif

fn_exit:
    *sreq_ptr = sreq;

    DEBUG_PRINT("Exiting istartmsg\n");
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG);
    return mpi_errno;
}

#ifdef ENABLE_CHECKPOINTING
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_iStartContigMsg_paused
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_iStartContigMsg_paused(MPIDI_VC_t *vc, void *hdr, MPIDI_msg_sz_t hdr_sz, void *data, MPIDI_msg_sz_t data_sz,
                                    MPID_Request **sreq_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    MPL_IOV iov[2];
    int n_iov=2;

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG_PAUSED);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG_PAUSED);

    int nb;
    int pkt_len;
    vbuf *buf;

    /* MT - need some signalling to lock down our right to use the
       channel, thus insuring that the progress engine does also try to
       write */

    /* the channel use fixed header size*/
    hdr_sz = sizeof(MPIDI_CH3_Pkt_t);
    iov[0].MPL_IOV_BUF = hdr;
    iov[0].MPL_IOV_LEN = hdr_sz;
    if(data_sz == 0) {
        n_iov = 1;
        pkt_len = hdr_sz;
    } else {
        iov[1].MPL_IOV_BUF = data;
        iov[1].MPL_IOV_LEN = data_sz;
        Calculate_IOV_len(iov, n_iov, pkt_len);
    }

    if (SENDQ_EMPTY(VC_FIELD(vc, send_queue)))
    {

        if (pkt_len > MRAIL_MAX_EAGER_SIZE) {
            sreq = create_request(iov, n_iov, 0, 0);
            mpi_errno = MPIDI_nem_ib_packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            }
            goto fn_exit;
        }


        mpi_errno = MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);
        mpi_errno = MPI_SUCCESS
        DEBUG_PRINT("[istartmsg_paused] mpierr %d, nb %d\n", mpi_errno, nb);
        if (mpi_errno == MPI_SUCCESS) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
        /* eager poking */
        /* if (nb == pkt_len)
           mpi_errno = MPID_Progress_test();
         */
            goto fn_exit;
        /*rfp codes*/
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            /* fast rdma ok but cannot send: there is no send wqe available */
            sreq = create_request(iov, n_iov, 0, 0);
            buf->sreq = (void *) sreq;
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
            sreq->cc = 0;
            /* TODO: Create an appropriate error message based on the value of errno
             */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        }
    } else {
        MPIDI_DBG_PRINTF((55, FCNAME,
                          "send in progress, request enqueued"));
        sreq = create_request(iov, n_iov, 0, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
    }

fn_exit:
    *sreq_ptr = sreq;

    DEBUG_PRINT("Exiting istartmsg_paused\n");
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_ISTARTCONTIGMSG_PAUSED);
    return mpi_errno;
}
#endif /* ENABLE_CHECKPOINTING */


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_iSendNoncontig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_iSendNoncontig (MPIDI_VC_t *vc, MPID_Request *sreq, void *header, MPIDI_msg_sz_t hdr_sz)
{
    int mpi_errno = MPI_SUCCESS;
    int i, n_iov, pkt_len, complete = 0, nb = 0;
    MPL_IOV iov[MPL_IOV_LIMIT];
    void *tmpbuf = NULL, *databuf = NULL;
    vbuf *vbuf;

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_ISENDNONCONTIG);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_ISENDNONCONTIG);

    iov[0].MPL_IOV_BUF = header;
    /* the channel use fixed header size*/
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);

#ifdef ENABLE_CHECKPOINTING
    MPID_nem_ib_vc_area *vc_ib = VC_IB(vc);
    if (!MPID_nem_ib_vc_send_paused(vc_ib)) {
#endif

    if (SENDQ_EMPTY(VC_FIELD(vc, send_queue))) {

        n_iov = MPL_IOV_LIMIT - 1;
        mpi_errno = MPIDI_CH3U_Request_load_send_iov(sreq, &iov[1], &n_iov);
        if (MPI_SUCCESS != mpi_errno) {
           /* --BEGIN ERROR HANDLING-- */
           MPIU_Object_set_ref(sreq, 0);
           MPIDI_CH3_Request_destroy(sreq);
           MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ch3|loadsendiov");
           /* --END ERROR HANDLING-- */
        }
        n_iov++;

        if (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV) {
            /*reload iov, allocate packing buffer of size =  data size + header size*/
            tmpbuf = MPIU_Malloc(sreq->dev.segment_size + sizeof(MPIDI_CH3_Pkt_t));
            databuf = tmpbuf;

            pkt_len = 0;
            /* First copy whatever has already been in iov set */
            for (i = 0; i < n_iov; i++) {
                MPIU_Memcpy(tmpbuf, iov[i].MPL_IOV_BUF, iov[i].MPL_IOV_LEN);
                tmpbuf = (void *)((unsigned long) tmpbuf + iov[i].MPL_IOV_LEN);
                pkt_len += iov[i].MPL_IOV_LEN;
            }
  
            /* Second reload iov and copy */
            do {
                n_iov = MPL_IOV_LIMIT;
                mpi_errno = MPIDI_CH3U_Request_load_send_iov(sreq,
                                                             sreq->dev.iov,
                                                             &n_iov);
                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno != MPI_SUCCESS) {
                    MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ch3|loadsendiov");
                }
                /* --END ERROR HANDLING-- */

                for (i = 0; i < n_iov; i++) {
                    MPIU_Memcpy(tmpbuf, sreq->dev.iov[i].MPL_IOV_BUF,
                                     sreq->dev.iov[i].MPL_IOV_LEN);
                    tmpbuf = (void *) ((unsigned long) tmpbuf +
                                     sreq->dev.iov[i].MPL_IOV_LEN);
                    pkt_len += sreq->dev.iov[i].MPL_IOV_LEN;
                }
            } while (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV);

            /*check if all the data has been packed*/
            MPIU_Assert(pkt_len == (sreq->dev.segment_size + sizeof(MPIDI_CH3_Pkt_t)));

            iov[0].MPL_IOV_BUF = databuf;
            iov[0].MPL_IOV_LEN = pkt_len;
            n_iov = 1;
        } else {
            Calculate_IOV_len(iov, n_iov, pkt_len);
        }

        if (pkt_len > MRAIL_MAX_EAGER_SIZE) {
            MPIU_Memcpy(sreq->dev.iov, iov, n_iov * sizeof(MPL_IOV));
            sreq->dev.iov_count = n_iov;
            mpi_errno = MPIDI_nem_ib_packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            } else if (MPI_SUCCESS != mpi_errno) {
                MPIU_Object_set_ref(sreq, 0);
                MPIDI_CH3_Request_destroy(sreq);
                MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nem|packetizedsend");
            }
            goto fn_exit;
        }

        mpi_errno = MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &vbuf);
        DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno, nb);
        if (MPI_SUCCESS == mpi_errno) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
            if (nb != 0) {
                MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
                if (!complete) {
                    SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
                    VC_FIELD(vc, send_active) = sreq;
                } else {
                    VC_FIELD(vc, send_active) = SENDQ_HEAD( VC_FIELD(vc, send_queue));
                }
            }
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            vbuf->sreq = (void *) sreq;
            mpi_errno = MPI_SUCCESS;
        } else {
            MPIU_Object_set_ref(sreq, 0);
            MPIDI_CH3_Request_destroy(sreq);
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nem|ibeagersend");
        }
    } else {
       /* Send queue not empty, enqueuing */
        MPIDI_DBG_PRINTF((55, FCNAME, "send queue not empty, enqueuing"));
        isend_update_request(sreq, header, hdr_sz, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, send_queue), sreq);
    }
#ifdef ENABLE_CHECKPOINTING
    }
    else
    {
        isend_update_request(sreq, header, hdr_sz, 0);
        SENDQ_ENQUEUE(&VC_FIELD(vc, paused_send_queue), sreq);
    }
#endif

fn_exit:
    if (databuf)
        MPIU_Free(databuf);
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_ISENDNONCONTIG);
    return mpi_errno;
fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MRAILI_Backlog_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Backlog_send(MPIDI_VC_t * vc, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_BACKLOG_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_BACKLOG_SEND);

    ibv_backlog_queue_t *q = &VC_FIELD(vc, connection)->srp.credits[rail].backlog;
    char cq_overflow = 0;

    while ((q->len > 0)
           && (VC_FIELD(vc, connection)->srp.credits[rail].remote_credit > 0)) {
        vbuf *v = NULL;
        MPIDI_nem_ib_pkt_comm_header *p;
        MPIU_Assert(q->vbuf_head != NULL);
        BACKLOG_DEQUEUE(q, v);

        /* Assumes packet header is at beginning of packet structure */
        p = (MPIDI_nem_ib_pkt_comm_header *) v->iheader;

        PACKET_SET_CREDIT(p, VC_FIELD(vc, connection), rail);
        --VC_FIELD(vc, connection)->srp.credits[rail].remote_credit;

        /* srq codes */
        if (process_info.has_srq) {
            p->vc_addr = VC_FIELD(vc, connection)->remote_vc_addr;
            p->rail        = rail;
        }

        v->vc = vc;
        v->rail = rail;

        int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);
        if ((NULL != hca_list[hca_num].send_cq_hndl) &&
            (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
            /* We are monitoring CQ's and there is CQ overflow */
            cq_overflow = 1;
        }

        if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail || cq_overflow) {
            MRAILI_Ext_sendq_enqueue(vc, rail, v);
            continue;
        }
        --VC_FIELD(vc, connection)->rails[rail].send_wqes_avail;

        MPIDI_nem_ib_POST_SR(v, VC_FIELD(vc, connection), rail, "ibv_post_sr (MRAILI_Backlog_push)");
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_BACKLOG_SEND);
    return 0;
}


#undef FUNCNAME
#define FUNCNAME MRAILI_Process_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MRAILI_Process_send(void *vbuf_addr)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_PROCESS_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_PROCESS_SEND);
    int mpi_errno = MPI_SUCCESS;

    vbuf            *v = vbuf_addr;
    MPIDI_nem_ib_pkt_comm_header *p;
    MPIDI_VC_t      *vc;
    MPIDI_VC_t      *orig_vc;
    MPID_Request    *req;
    int             complete;

    vc  = v->vc;
    {
        orig_vc = vc;
    }

    p   = v->iheader;

    VC_FIELD(orig_vc, connection)->rails[v->rail].send_wqes_avail++;

    if(VC_FIELD(vc, free_vc)) {
        if(VC_FIELD(vc, connection)->rails[v->rail].send_wqes_avail == rdma_default_max_send_wqe) {
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
        VC_FIELD(vc, connection)->outstanding_eager_vbufs--;
        /* DEBUG_PRINT("Eager, decrementing to: %d\n", v,
                vc->mrail.outstanding_eager_vbufs); */
        if(VC_FIELD(vc, connection)->outstanding_eager_vbufs <
                rdma_coalesce_threshold) {
            DEBUG_PRINT("Flushing coalesced\n", v);
        }
        v->eager = 0;
    }

    /* ext_sendq_head */
    if (VC_FIELD(orig_vc, connection)->rails[v->rail].ext_sendq_head) {
        MRAILI_Ext_sendq_send(orig_vc, v->rail);
    }

    if(v->padding == COLL_VBUF_FLAG) {
        MRAILI_Release_vbuf(v);
        goto fn_exit;
    }

    /* RPUT codes */
    
    if (v->padding == RPUT_VBUF_FLAG) {
        req = (MPID_Request *)v->sreq;
        MPIU_Assert(req != NULL);

        MRAILI_Release_vbuf(v);
        goto fn_exit;
    }
    

    if (v->padding == CREDIT_VBUF_FLAG) {
        VC_FIELD(orig_vc, connection)->rails[v->rail].send_wqes_avail--;
        goto fn_exit;
    }
    switch (p->type) {
    /*header caching codes */
#ifndef MV2_DISABLE_HEADER_CACHING 
    case MPIDI_CH3_PKT_FAST_EAGER_SEND:
    case MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ:
#endif
    case MPIDI_CH3_PKT_EAGER_SEND:
#if defined(USE_EAGER_SHORT)
    case MPIDI_CH3_PKT_EAGERSHORT_SEND:
#endif
    case MPIDI_CH3_PKT_EAGER_SYNC_SEND:
    case MPIDI_CH3_PKT_PACKETIZED_SEND_DATA:
    case MPIDI_CH3_PKT_RNDV_R3_DATA:
    case MPIDI_CH3_PKT_READY_SEND:
    case MPIDI_CH3_PKT_PUT:
    case MPIDI_CH3_PKT_PUT_IMMED:
    case MPIDI_CH3_PKT_ACCUMULATE:
    case MPIDI_CH3_PKT_FOP:
    case MPIDI_CH3_PKT_FOP_RESP:
    case MPIDI_CH3_PKT_FLUSH:
    case MPIDI_CH3_PKT_UNLOCK:
    case MPIDI_CH3_PKT_GET_ACCUM:
    case MPIDI_CH3_PKT_GET_ACCUM_RESP:
        req = v->sreq;
        v->sreq = NULL;
        DEBUG_PRINT("[process send type: %d] complete for eager msg, req %p, seqnum = %d\n",
                    p->type, req, p->seqnum);
        if (req != NULL) {
            MPIDI_CH3U_Handle_send_req(vc, req, &complete);

            DEBUG_PRINT("[process send] req not null\n");
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }
        }
       

        if (v->padding == NORMAL_VBUF_FLAG) {
            DEBUG_PRINT("[process send] normal flag, free vbuf\n");
            /* vbuf.c */
            MRAILI_Release_vbuf(v);
        } else {
            v->padding = FREE_FLAG;
        }
        break;
    case MPIDI_NEM_PKT_LMT_DONE:
            {
            MPID_nem_pkt_lmt_done_t *done = NULL;
            MPID_Request *orig_req;
            done = (MPID_nem_pkt_lmt_done_t *)(v->pheader);
            MPID_Request_get_ptr(done->sreq_id, orig_req);
            /* MPID_Request_get_ptr(done->req_id, orig_req); */
            if (orig_req == NULL){
                ibv_va_error_abort(GEN_EXIT_ERR,
                        "s == NULL, s is the send, v is %p "
                        "handler of the rput finish", v);
            }
 
            if (REQ_FIELD(orig_req, d_entry) != NULL) {
                dreg_unregister(REQ_FIELD(orig_req, d_entry)); 
                REQ_FIELD(orig_req, d_entry) = NULL;
            }
            int dt_contig;
            MPIDI_msg_sz_t data_sz ATTRIBUTE((unused));
            MPI_Aint dt_true_lb ATTRIBUTE((unused));
            MPID_Datatype * dt_ptr;
            MPIDI_Datatype_get_info(orig_req->dev.user_count, orig_req->dev.datatype, dt_contig, data_sz, dt_ptr, dt_true_lb);
            if(!dt_contig)
            if(REQ_FIELD(orig_req, rndv_buf) != NULL)
                MPIU_Free(REQ_FIELD(orig_req, rndv_buf));
            MPIDI_CH3U_Handle_send_req(vc, orig_req, &complete);
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }

           if (v->padding == NORMAL_VBUF_FLAG)
                MRAILI_Release_vbuf(v);
            else v->padding = FREE_FLAG;
            
           break;
           }
    case MPIDI_CH3_PKT_GET_RESP:
        DEBUG_PRINT("[process send] get get respond finish\n");
        req = (MPID_Request *) (v->sreq);
        v->sreq = NULL;
        if (NULL != req) {

        MPIDI_CH3U_Handle_send_req(vc, req, &complete);
            if (complete != TRUE) {
                ibv_error_abort(IBV_STATUS_ERR, "Get incomplete eager send request\n");
            }
        }
        if (v->padding == NORMAL_VBUF_FLAG)
            MRAILI_Release_vbuf(v);
        else v->padding = FREE_FLAG;
        break;

    case MPIDI_CH3_PKT_NOOP:
    case MPIDI_CH3_PKT_ADDRESS:
    case MPIDI_CH3_PKT_ADDRESS_REPLY:
    case MPIDI_CH3_PKT_PACKETIZED_SEND_START:
    case MPIDI_CH3_PKT_RNDV_CLR_TO_SEND:
    case MPIDI_CH3_PKT_EAGER_SYNC_ACK:
    case MPIDI_CH3_PKT_CANCEL_SEND_REQ:
    case MPIDI_CH3_PKT_CANCEL_SEND_RESP:
    case MPIDI_CH3_PKT_GET:
    case MPIDI_CH3_PKT_LOCK:
    case MPIDI_CH3_PKT_LOCK_ACK:
    case MPIDI_CH3_PKT_LOCK_OP_ACK:
    case MPIDI_CH3_PKT_ACK:
    case MPIDI_CH3_PKT_FOP_RESP_IMMED:
    case MPIDI_CH3_PKT_FOP_IMMED:
    case MPIDI_CH3_PKT_CAS_IMMED:
    case MPIDI_CH3_PKT_CAS_RESP_IMMED:
    case MPIDI_CH3_PKT_GET_ACCUM_IMMED:
    case MPIDI_CH3_PKT_FLOW_CNTL_UPDATE:
    case MPIDI_CH3_PKT_CLOSE:  /*24*/
    case MPIDI_NEM_PKT_LMT_RTS:
    case MPIDI_NEM_PKT_LMT_CTS:
    case MPIDI_NEM_PKT_LMT_COOKIE:
    case MPIDI_CH3_PKT_RNDV_R3_ACK:
#ifdef ENABLE_CHECKPOINTING
    case MPIDI_NEM_PKT_CKPT_MARKER: 
    case MPIDI_NEM_IB_PKT_UNPAUSE: 
#endif
        DEBUG_PRINT("[process send] get %d\n", p->type);

        if (v->padding == NORMAL_VBUF_FLAG) {
            MRAILI_Release_vbuf(v);
        }
        else v->padding = FREE_FLAG;
        break;
    default:
        dump_vbuf("unknown packet (send finished)", v);
        ibv_va_error_abort(IBV_STATUS_ERR,
                         "Unknown packet type %d in "
                         "MRAILI_Process_send", p->type);
    }
    DEBUG_PRINT("return from process send\n");

fn_exit:
MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_PROCESS_SEND);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MRAILI_Send_noop
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MRAILI_Send_noop(MPIDI_VC_t * c, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_MRAILI_SEND_NOOP);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAILI_SEND_NOOP);

    /* always send a noop when it is needed even if there is a backlog.
     * noops do not consume credits.
     * this is necessary to avoid credit deadlock.
     * RNR NAK will protect us if receiver is low on buffers.
     * by doing this we can force a noop ahead of any other queued packets.
     */

    vbuf* v = get_vbuf();
    MPIDI_nem_ib_pkt_noop* p = (MPIDI_nem_ib_pkt_noop *) v->iheader;
    p->type = MPIDI_CH3_PKT_NOOP;
    vbuf_init_send(v, sizeof(MPIDI_nem_ib_pkt_noop), rail);
    process_info.post_send(c, v, rail);
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

    /* srq codes */
    if (process_info.has_srq)
        return MPI_SUCCESS;

    if (VC_FIELD(vc, connection)->srp.credits[rail].local_credit >=
        rdma_dynamic_credit_threshold
        /*rfp codes */
        || VC_FIELD(vc, connection)->rfp.rdma_credit > num_rdma_buffer / 2
        || (VC_FIELD(vc, connection)->srp.credits[rail].remote_cc <=
            rdma_credit_preserve
            && VC_FIELD(vc, connection)->srp.credits[rail].local_credit >=
            rdma_credit_notify_threshold)
        ) {
        MRAILI_Send_noop(vc, rail);
    }
    MPIDI_FUNC_EXIT(MPID_STATE_MRAILI_SEND_NOOP_IF_NEEDED);
    return MPI_SUCCESS;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_fast_rdma_fill_start_buf
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/* INOUT: num_bytes_ptr holds the pkt_len as input parameter */
int MPIDI_nem_ib_fast_rdma_fill_start_buf(MPIDI_VC_t * vc,
                                    MPL_IOV * iov, int n_iov,
                                    int *num_bytes_ptr)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_FILL_START_BUF);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_FILL_START_BUF);

    /* FIXME: Here we assume that iov holds a packet header */
#ifndef MV2_DISABLE_HEADER_CACHING
    MPIDI_CH3_Pkt_send_t *cached =  VC_FIELD(vc, connection)->rfp.cached_outgoing;
    MPIDI_nem_ib_pkt_comm_header *cached_iheader = VC_FIELD(vc, connection)->rfp.cached_outgoing_iheader;
#endif
    MPIDI_CH3_Pkt_send_t *header;
    vbuf *v = &(VC_FIELD(vc, connection)->rfp.RDMA_send_buf[VC_FIELD(vc, connection)->rfp.phead_RDMA_send]);
    void *vstart;
    void *data_buf;

    int len = *num_bytes_ptr, avail = 0;
    int i;
    header = iov[0].MPL_IOV_BUF;
    avail   = len;
    /*PACKET_SET_RDMA_CREDIT delayed. need to get iheader position first*/
    /* it's also need to be added in !MV2_DISABLE_HEADER_CACHING situation*/
    *num_bytes_ptr = 0;
    v->sreq = NULL;

    DEBUG_PRINT("Header info, tag %d, rank %d, context_id %d\n", 
            header->match.parts.tag, header->match.parts.rank, header->match.parts.context_id);
#ifndef MV2_DISABLE_HEADER_CACHING
    if ((header->type == MPIDI_CH3_PKT_EAGER_SEND) &&
        (len - sizeof(MPIDI_CH3_Pkt_t) <= MAX_SIZE_WITH_HEADER_CACHING) &&
        (header->match.parts.tag == cached->match.parts.tag) &&
        (header->match.parts.rank == cached->match.parts.rank) &&
        (header->match.parts.context_id == cached->match.parts.context_id) &&
        (0 == cached_iheader->vbuf_credit) &&
        (0 == cached_iheader->remote_credit) &&
        (VC_FIELD(vc, connection)->rfp.rdma_credit == cached_iheader->rdma_credit)) {
        /* change the header contents */
        ++VC_FIELD(vc, connection)->rfp.cached_hit;
        VC_FIELD(vc, connection)->rfp.rdma_credit = 0;

        if (header->sender_req_id == cached->sender_req_id) {
            MPIDI_nem_ib_pkt_fast_eager *fast_header;
            vstart = v->buffer;

            DEBUG_PRINT 
                ("[send: fill buf], head cached, head_flag %p, vstart %p, length %d",
                 &v->head_flag, vstart,
                 len - sizeof(MPIDI_CH3_Pkt_t) + 
         sizeof(MPIDI_nem_ib_pkt_fast_eager));

            fast_header = vstart;
            fast_header->type = MPIDI_CH3_PKT_FAST_EAGER_SEND;
            fast_header->bytes_in_pkt = len - sizeof(MPIDI_CH3_Pkt_t);
            v->pheader = fast_header;
            v->iheader = fast_header;
            data_buf = (void *) ((unsigned long) vstart +
                                 sizeof(MPIDI_nem_ib_pkt_fast_eager));

        if (iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_t))
          MPIU_Memcpy(data_buf, (void *)((uintptr_t)iov[0].MPL_IOV_BUF +
               sizeof(MPIDI_CH3_Pkt_t)),
               iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_t));

        data_buf = (void *)((uintptr_t)data_buf + iov[0].MPL_IOV_LEN -
            sizeof(MPIDI_CH3_Pkt_t));

            *num_bytes_ptr += sizeof(MPIDI_nem_ib_pkt_fast_eager);
            avail -= sizeof(MPIDI_nem_ib_pkt_fast_eager);
        } else {
            MPIDI_nem_ib_pkt_fast_eager_with_req *fast_header;
            vstart = v->buffer;
            DEBUG_PRINT 
                ("[send: fill buf], head cached, head_flag %p, vstart %p, length %d\n",
                 &v->head_flag, vstart,
                 len - sizeof(MPIDI_CH3_Pkt_t) +
         sizeof(MPIDI_nem_ib_pkt_fast_eager_with_req));

            fast_header = vstart;
            fast_header->type = MPIDI_CH3_PKT_FAST_EAGER_SEND_WITH_REQ;
            fast_header->bytes_in_pkt = len - sizeof(MPIDI_CH3_Pkt_t);
            fast_header->sender_req_id = header->sender_req_id;
            cached->sender_req_id = header->sender_req_id;
            v->pheader = fast_header;
            v->iheader = fast_header;
            data_buf =
                (void *) ((unsigned long) vstart +
                          sizeof(MPIDI_nem_ib_pkt_fast_eager_with_req));
        if (iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_t))
          MPIU_Memcpy(data_buf, (void *)((uintptr_t)iov[0].MPL_IOV_BUF +
               sizeof(MPIDI_CH3_Pkt_t)),
               iov[0].MPL_IOV_LEN - sizeof(MPIDI_CH3_Pkt_t));

        data_buf = (void *)((uintptr_t)data_buf + iov[0].MPL_IOV_LEN -
            sizeof(MPIDI_CH3_Pkt_t));

            *num_bytes_ptr += sizeof(MPIDI_nem_ib_pkt_fast_eager_with_req);
            avail -= sizeof(MPIDI_nem_ib_pkt_fast_eager_with_req);
        }
    } else
#endif
    {
    vstart = v->buffer;
    v->iheader = vstart;
    MPIDI_nem_ib_pkt_comm_header *p = v->iheader;
    PACKET_SET_RDMA_CREDIT(p, VC_FIELD(vc, connection));

        vstart = (void *) ((unsigned long) vstart + IB_PKT_HEADER_LENGTH);
        p->type = header->type;
        MPIU_Memcpy(vstart, header, iov[0].MPL_IOV_LEN);
#ifndef MV2_DISABLE_HEADER_CACHING
        if (header->type == MPIDI_CH3_PKT_EAGER_SEND && 
            (len - sizeof(MPIDI_CH3_Pkt_t) <= MAX_SIZE_WITH_HEADER_CACHING) ) {
          MPIU_Memcpy(cached, header, sizeof(MPIDI_CH3_Pkt_eager_send_t));
          MPIU_Memcpy(cached_iheader, p, sizeof(MPIDI_nem_ib_pkt_comm_header));
        }
        ++VC_FIELD(vc, connection)->rfp.cached_miss;
#endif
        data_buf = (void *) ((unsigned long) vstart + iov[0].MPL_IOV_LEN);
        *num_bytes_ptr += iov[0].MPL_IOV_LEN + IB_PKT_HEADER_LENGTH;
        avail -= iov[0].MPL_IOV_LEN;
        v->pheader = vstart;
    }

    /* We have filled the header, it is time to fit in the actual data */
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

    DEBUG_PRINT("[send: fill buf], num bytes copied %d\n", *num_bytes_ptr);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_FILL_START_BUF);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_fast_rdma_send_complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/* INOUT: num_bytes_ptr holds the pkt_len as input parameter */
int MPIDI_nem_ib_fast_rdma_send_complete(MPIDI_VC_t * vc,
                                              MPL_IOV * iov,
                                              int n_iov,
                                              int *num_bytes_ptr,
                                              vbuf ** vbuf_handle)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_SEND_COMPLETE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_SEND_COMPLETE);
    int rail;
    int  post_len;
    char cq_overflow = 0;
    VBUF_FLAG_TYPE flag;
    vbuf *v =
        &(VC_FIELD(vc, connection)->rfp.RDMA_send_buf[VC_FIELD(vc, connection)->rfp.phead_RDMA_send]);
    char *rstart;
    uint16_t seqnum;

    rail = MPIDI_nem_ib_send_select_rail(vc);
    MPIDI_nem_ib_fast_rdma_fill_start_buf(vc, iov, n_iov, num_bytes_ptr);

    MPIDI_nem_ib_pkt_comm_header *p = v->iheader;
    MPIDI_nem_ib_get_send_seqnum(vc, seqnum);
    MPIDI_nem_ib_set_seqnum(p, seqnum);

    post_len = *num_bytes_ptr;
    rstart = VC_FIELD(vc, connection)->rfp.remote_RDMA_buf +
            (VC_FIELD(vc, connection)->rfp.phead_RDMA_send * rdma_fp_buffer_size);
    DEBUG_PRINT("[send: rdma_send] local vbuf %p, remote start %p, size %d, iheader = %p\n",
               v, rstart, post_len, v->iheader);

    if (++(VC_FIELD(vc, connection)->rfp.phead_RDMA_send) >= num_rdma_buffer)
        VC_FIELD(vc, connection)->rfp.phead_RDMA_send = 0;

    v->rail = rail;
    v->padding = BUSY_FLAG;

    /* requirements for coalescing */
    ++VC_FIELD(vc, connection)->outstanding_eager_vbufs;
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

    DEBUG_PRINT("incrementing the outstanding eager vbufs: RFP %d\n", VC_FIELD(vc, connection)->outstanding_eager_vbufs);

    /* generate a completion, following statements should have been executed during
     * initialization */
    post_len += VBUF_FAST_RDMA_EXTRA_BYTES;

    DEBUG_PRINT("[send: rdma_send] lkey %p, rkey %p, len %d, flag %d\n",
                VC_FIELD(vc, connection)->rfp.RDMA_send_buf_mr[VC_FIELD(vc, connection)->rails[rail].hca_index]->lkey,
                VC_FIELD(vc, connection)->rfp.RDMA_remote_buf_rkey, post_len, *v->head_flag);

    VBUF_SET_RDMA_ADDR_KEY(v, post_len, v->head_flag,
            VC_FIELD(vc, connection)->rfp.RDMA_send_buf_mr[VC_FIELD(vc, connection)->rails[rail].hca_index]->lkey, rstart,
            VC_FIELD(vc, connection)->rfp.RDMA_remote_buf_rkey[VC_FIELD(vc, connection)->rails[rail].hca_index]);

#ifdef CRC_CHECK
    p->mrail.crc = update_crc(1, (void *)((uintptr_t)p+sizeof *p),
                              *v->head_flag - sizeof *p);
#endif

    if ((NULL != hca_list[VC_FIELD(vc, connection)->rails[rail].hca_index].send_cq_hndl) &&
        (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
        /* We are monitoring CQ's and there is CQ overflow */
        cq_overflow = 1;
    }

    if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail || cq_overflow) {
        DEBUG_PRINT("[send: rdma_send] Warning! no send wqe or send cq available\n");
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        *vbuf_handle = v;
        return MPI_MRAIL_MSG_QUEUED;
    } else {
        --VC_FIELD(vc, connection)->rails[rail].send_wqes_avail;
        *vbuf_handle = v;

        MPIDI_nem_ib_POST_SR(v, VC_FIELD(vc, connection), rail, "ibv_post_sr (post_fast_rdma)");
        DEBUG_PRINT("[send:post rdma] desc posted\n");
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_SEND_COMPLETE);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_fast_rdma_ok
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_fast_rdma_ok(MPIDI_VC_t * vc, int len)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_OK);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_OK);

    if (len > MRAIL_MAX_RDMA_FP_SIZE) {
        return 0;
    }

    if (num_rdma_buffer < 2
        || VC_FIELD(vc, connection)->rfp.phead_RDMA_send == VC_FIELD(vc, connection)->rfp.ptail_RDMA_send
        || VC_FIELD(vc, connection)->rfp.RDMA_send_buf[VC_FIELD(vc, connection)->rfp.phead_RDMA_send].padding == BUSY_FLAG
        )
    {
        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_OK);
        return 0;
    }

    int i = 0;

    for (; i < rdma_num_rails; i++)
    {
        if (VC_FIELD(vc, connection)->srp.credits[i].backlog.len != 0)
        {
            MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_OK);
            return 0;
        }
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_FAST_RDMA_OK);
    return 1;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_post_srq_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_post_srq_send(MPIDI_VC_t* vc, vbuf* v, int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_IB_POST_SRQ_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_IB_POST_SRQ_SEND);
    int hca_num = rail / (ib_hca_num_ports * rdma_num_qp_per_port);
    char cq_overflow = 0;
    MPIDI_nem_ib_pkt_comm_header *p = v->iheader;
    PACKET_SET_CREDIT(p, VC_FIELD( vc, connection), rail);

    v->vc = (void *) vc;
    p->vc_addr = VC_FIELD(vc, connection)->remote_vc_addr;
    p->rail        = rail;

    if ((NULL != hca_list[hca_num].send_cq_hndl) &&
        (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
        /* We are monitoring CQ's and there is CQ overflow */
        cq_overflow = 1;
    }

    if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail || cq_overflow) {
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        MPIDI_FUNC_EXIT(MPID_STATE_IB_POST_SRQ_SEND);
        return MPI_MRAIL_MSG_QUEUED;
    }

    --VC_FIELD(vc, connection)->rails[rail].send_wqes_avail;
    MPIDI_nem_ib_POST_SR(v, VC_FIELD( vc, connection), rail, "ibv_post_sr (post_send_desc)");

    pthread_spin_lock(&srq_info.srq_post_spin_lock);

    if(srq_info.posted_bufs[hca_num] <= rdma_credit_preserve) {
        srq_info.posted_bufs[hca_num] +=
            MPIDI_nem_ib_post_srq_buffers(mv2_srq_fill_size -
                    srq_info.posted_bufs[hca_num],
                    hca_num);
    }

    pthread_spin_unlock(&srq_info.srq_post_spin_lock);

    MPIDI_FUNC_EXIT(MPID_STATE_IB_POST_SRQ_SEND);
    return 0;
}

void vbuf_address_send(MPIDI_VC_t *vc)
{                   
    int rail, i;    
                    
    vbuf* v = get_vbuf();
    MPIDI_nem_ib_pkt_address* p = (MPIDI_nem_ib_pkt_address *) v->pheader;
    MPIDI_nem_ib_pkt_comm_header *pi = v->iheader;
                
    rail = MPIDI_nem_ib_send_select_rail(vc);
    pi->type = MPIDI_CH3_PKT_ADDRESS;
    p->type        = 0;
    p->rdma_address = (unsigned long)VC_FIELD(vc, connection)->rfp.RDMA_recv_buf_DMA;
    DEBUG_PRINT("send rdma address, dma address %p\n",
            (void *)p->rdma_address);

    for (i = 0; i < ib_hca_num_hcas; i++) {
    DEBUG_PRINT("mr %p\n", VC_FIELD(vc, connection)->rfp.RDMA_recv_buf_mr[i]);
    p->rdma_hndl[i]   = VC_FIELD(vc, connection)->rfp.RDMA_recv_buf_mr[i]->rkey;
    }
    vbuf_init_send(v, sizeof(MPIDI_nem_ib_pkt_address)+IB_PKT_HEADER_LENGTH, rail);
    process_info.post_send(vc, v, rail);
}

void vbuf_address_reply_send(MPIDI_VC_t *vc, uint8_t data)
{
    int rail;

    vbuf *v = get_vbuf();
    MPIDI_nem_ib_pkt_address_reply *p = (MPIDI_nem_ib_pkt_address_reply *) v->pheader;
    MPIDI_nem_ib_pkt_comm_header *pi = v->iheader;

    rail = MPIDI_nem_ib_send_select_rail(vc);
    pi->type = MPIDI_CH3_PKT_ADDRESS_REPLY;
    p->type     = 0;
    p->reply_data = data;

    vbuf_init_send(v, sizeof(MPIDI_nem_ib_pkt_address_reply)+IB_PKT_HEADER_LENGTH, rail);
    process_info.post_send(vc, v, rail);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_lmt_r3_ack_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_lmt_r3_ack_send(MPIDI_VC_t *vc)
{
    vbuf *v;
    int mpi_errno;
    int rail;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_LMT_R3_ACK_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_LMT_R3_ACK_SEND);

    v = get_vbuf();
    MPIDI_CH3_Pkt_rndv_r3_ack_t* p = (MPIDI_CH3_Pkt_rndv_r3_ack_t*) v->pheader;
    MPIDI_nem_ib_pkt_comm_header *pi = v->iheader;
    rail = MPIDI_nem_ib_send_select_rail(vc);

    pi->type = MPIDI_CH3_PKT_RNDV_R3_ACK;
    p->type = 0;

    p->ack_data = VC_FIELD(vc, received_r3_data);
    VC_FIELD(vc, received_r3_data) = 0;

    vbuf_init_send(v, sizeof(MPIDI_CH3_Pkt_rndv_r3_ack_t) + IB_PKT_HEADER_LENGTH, rail);

    mpi_errno = process_info.post_send(vc, v, rail);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_LMT_R3_ACK_SEND);
    return mpi_errno;
}

#ifdef ENABLE_CHECKPOINTING
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_send_queued
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_send_queued(MPIDI_VC_t *vc, MPIDI_nem_ib_request_queue_t *send_queue)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq;
    MPIDI_msg_sz_t offset;
    MPL_IOV *iov;
    int complete;

    vbuf *buf;
    int nb, n_iov, pkt_len = 0;

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_SEND_QUEUED);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_SEND_QUEUED);

    MPIU_DBG_MSG_P(CH3_CHANNEL, VERBOSE, "vc = %p", vc);
    MPIU_Assert(vc != NULL);

    if (MPIDI_CH3I_Sendq_empty(*send_queue))
    goto fn_exit;

    while (!MPIDI_CH3I_Sendq_empty(*send_queue))
    {
        sreq = MPIDI_CH3I_Sendq_head(*send_queue);
        MPIU_DBG_MSG_P(CH3_CHANNEL, VERBOSE, "Sending %p", sreq);

        iov = &sreq->dev.iov[sreq->dev.iov_offset];
        n_iov = sreq->dev.iov_count;

        Calculate_IOV_len(iov, n_iov, pkt_len);


        mpi_errno = MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);

        if (mpi_errno == MPI_SUCCESS)
        {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
            goto fn_exit;
        /*rfp codes*/
        }
        else if (MPI_MRAIL_MSG_QUEUED == mpi_errno)
        {
            /* fast rdma ok but cannot send: there is no send wqe available */
            sreq = create_request(iov, n_iov, 0, 0);
            buf->sreq = (void *) sreq;
            mpi_errno = MPI_SUCCESS;
            goto fn_exit;
        }
        else
        {
            sreq = MPID_Request_create();
            if (sreq == NULL)
            {
                mpi_errno =
                    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
                                         FCNAME, __LINE__,
                                         MPI_ERR_OTHER, "**nomem", 0);
                goto fn_exit;
            }
            sreq->kind = MPID_REQUEST_SEND;
            sreq->cc = 0;
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        }


        complete = 1;
        for (iov = &sreq->dev.iov[sreq->dev.iov_offset]; iov < &sreq->dev.iov[sreq->dev.iov_offset + sreq->dev.iov_count]; ++iov)
        {
            if (offset < iov->MPL_IOV_LEN)
            {
                iov->MPL_IOV_BUF = (char *)iov->MPL_IOV_BUF + offset;
                iov->MPL_IOV_LEN -= offset;
                /* iov_count should be equal to the number of iov's remaining */
                sreq->dev.iov_count -= ((iov - sreq->dev.iov) - sreq->dev.iov_offset);
                sreq->dev.iov_offset = iov - sreq->dev.iov;
                complete = 0;
                break;
            }
            offset -= iov->MPL_IOV_LEN;
        }
        if (!complete)
        {
            /* writev couldn't write the entire iov, give up for now */
            break;
        }
        else
        {
            /* sent whole message */
            int (*reqFn)(MPIDI_VC_t *, MPID_Request *, int *);

            reqFn = sreq->dev.OnDataAvail;
            if (!reqFn)
            {
                MPIU_Assert(MPIDI_Request_get_type(sreq) != MPIDI_REQUEST_TYPE_GET_RESP);
                MPID_Request_complete(sreq);
                MPIU_DBG_MSG(CH3_CHANNEL, VERBOSE, ".... complete");
                MPIDI_CH3I_Sendq_dequeue(send_queue, &sreq);
                continue;
            }

            complete = 0;
            mpi_errno = reqFn(vc, sreq, &complete);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            if (complete)
            {
                MPIU_DBG_MSG(CH3_CHANNEL, VERBOSE, ".... complete");
                MPIDI_CH3I_Sendq_dequeue(send_queue, &sreq);
                continue;
            }
            sreq->dev.iov_offset = 0;
        }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_SEND_QUEUED);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


#endif

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
       }
   }
   return mpi_errno;
}

int mv2_shm_coll_post_send(vbuf *v, int rail, MPIDI_VC_t * vc)
{
   char no_cq_overflow = 1;
   int mpi_errno = MPI_SUCCESS;

   v->rail = rail;

    int hca_num = rail / (rdma_num_rails / ib_hca_num_hcas);
    if ((NULL != hca_list[hca_num].send_cq_hndl) &&
        (process_info.global_used_send_cq >= rdma_default_max_cq_size)) {
        /* We are monitoring CQ's and there is CQ overflow */
        no_cq_overflow = 0;
    }

    if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail || !no_cq_overflow) {
        DEBUG_PRINT("[send: rdma_send] Warning! no send wqe or send cq available\n");
        MRAILI_Ext_sendq_enqueue(vc, rail, v);
        return MPI_MRAIL_MSG_QUEUED;
    } else {
        --VC_FIELD(vc, connection)->rails[rail].send_wqes_avail;

        MPIDI_nem_ib_POST_SR(v, VC_FIELD(vc, connection), rail, "ibv_post_sr (post_fast_rdma)");
        DEBUG_PRINT("[send:post rdma] desc posted\n");
    }

    return mpi_errno;
}

void mv2_shm_coll_prepare_post_send(uint64_t local_rdma_addr, uint64_t remote_rdma_addr,
                      uint32_t local_rdma_key, uint32_t remote_rdma_key,
                      int len, int rail, MPIDI_VC_t * vc)
{
    vbuf *v=NULL;
    v = get_vbuf();
    v->desc.u.sr.next = NULL;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
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
    mv2_shm_coll_post_send(v, rail, vc);

    return;
}
