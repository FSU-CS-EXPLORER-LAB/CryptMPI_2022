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
#include "mpiutil.h"

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

#define GET_SEQ_NUM(_pkt, _seqnum) \
{   \
    if (NULL == (_pkt)) \
        _seqnum = -101;    \
    switch (((MPIDI_CH3I_MRAILI_Pkt_comm_header *)(_pkt))->type) { \
    case MPIDI_CH3_PKT_EAGER_SEND:  \
    case MPIDI_CH3_PKT_READY_SEND:  \
    case MPIDI_CH3_PKT_EAGER_SYNC_SEND: \
    case MPIDI_CH3_PKT_RNDV_REQ_TO_SEND:    \
    case MPIDI_CH3_PKT_RNDV_READY_REQ_TO_SEND:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_send_t *) (_pkt))->seqnum;    \
            break;  \
        }   \
    case MPIDI_CH3_PKT_RNDV_CLR_TO_SEND:    \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_rndv_clr_to_send_t *) (_pkt))->seqnum;    \
            break; \
        }   \
    case MPIDI_CH3_PKT_GET_RNDV:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_get_rndv_t *) (_pkt))->seqnum;    \
            break; \
        }   \
    case MPIDI_CH3_PKT_PUT_RNDV:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_put_rndv_t *) (_pkt))->seqnum;    \
            break; \
        }   \
    case MPIDI_CH3_PKT_ACCUMULATE_RNDV:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_accum_rndv_t *) (_pkt))->seqnum;    \
            break; \
        }   \
    case MPIDI_CH3_PKT_GET_ACCUM_RNDV:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_accum_rndv_t *) (_pkt))->seqnum;    \
            break; \
        }   \
     case MPIDI_CH3_PKT_GET_RESP:  \
        {   \
            _seqnum = ((MPIDI_CH3_Pkt_get_resp_t *) (_pkt))->seqnum;    \
            break; \
        }   \
    default:    \
        _seqnum =  -1;  \
    }   \
}

static MPL_IOV iov[MPL_IOV_LIMIT + 1];

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Packetized_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Packetized_send(MPIDI_VC_t * vc, MPID_Request * sreq)
{
    MPIDI_CH3_Pkt_packetized_send_start_t send_start;
    MPIDI_CH3_Pkt_packetized_send_data_t pkt_head;

    vbuf *buf;
    int mpi_errno = MPI_SUCCESS;
    int n_iov;
    int msg_buffered = 0;
    int nb;
    int complete;
    int pkt_len;
    int seqnum;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_SENDV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_SENDV);
    MPIU_DBG_PRINTF(("ch3_isendv\n"));
    MPIDI_DBG_PRINTF((50, FCNAME, "entering"));

    MPIDI_Pkt_init(&send_start, MPIDI_CH3_PKT_PACKETIZED_SEND_START);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_packetized_send_start_t);
    iov[0].MPL_IOV_BUF = (void*) &send_start;
    MPIU_Memcpy(&iov[1], sreq->dev.iov, sreq->dev.iov_count * sizeof(MPL_IOV));
    n_iov = 1 + sreq->dev.iov_count;

    GET_SEQ_NUM(sreq->dev.iov[0].MPL_IOV_BUF, seqnum);
    
    if (-1 == seqnum) {
        MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    }
    MPIDI_Pkt_set_seqnum(&send_start, seqnum);
    MPIDI_Request_set_seqnum(sreq, seqnum);
    send_start.origin_head_size = sreq->dev.iov[0].MPL_IOV_LEN;

    Calculate_IOV_len(iov, n_iov, pkt_len);

    mpi_errno =
        MPIDI_CH3I_MRAILI_Eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);
    DEBUG_PRINT("[pkt send] mpierr %d, nb %d\n", mpi_errno, nb);

    if (MPI_SUCCESS != mpi_errno && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
        vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
        sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        MPID_Request_complete(sreq);
        goto fn_exit;
    } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
        msg_buffered = 1;
    }
    nb -= sizeof(MPIDI_CH3_Pkt_packetized_send_start_t);

    MPIDI_Pkt_init(&pkt_head, MPIDI_CH3_PKT_PACKETIZED_SEND_DATA);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_packetized_send_data_t);
    iov[0].MPL_IOV_BUF = (void*) &pkt_head;

    do {
        while (!MPIDI_CH3I_Request_adjust_iov(sreq, nb)) {
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
                MPIDI_CH3I_MRAILI_Eager_send(vc, iov, n_iov, pkt_len, &nb,
                        &buf);
            DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno,
                    nb);
            MPIU_Assert(NULL == buf->sreq);

            if (MPI_SUCCESS != mpi_errno
                && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
                vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
                sreq->status.MPI_ERROR = MPI_ERR_INTERN;
                MPID_Request_complete(sreq);
                goto fn_exit;
            } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                msg_buffered = 1;
            }

            nb -= sizeof(MPIDI_CH3_Pkt_packetized_send_data_t);
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
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_SENDV);
    return mpi_errno;

}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Packetized_recv_req
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Packetized_recv_req(MPIDI_VC_t * vc, MPID_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_REQ);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_REQ);
    if (NULL == vc->mrail.packetized_recv) {
        vc->mrail.packetized_recv = (void *) rreq;
    } else {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER,
                                 "**fail", 0);
    }
    DEBUG_PRINT("Add rreq %p to packetized recv\n", rreq);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_REQ);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Packetized_recv_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Packetized_recv_data(MPIDI_VC_t * vc, vbuf *v)
{
    int mpi_errno = MPI_SUCCESS;
    int skipsize = sizeof(MPIDI_CH3_Pkt_packetized_send_data_t);
    int nb, complete;
    MPID_Request *rreq = vc->mrail.packetized_recv;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_DATA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_DATA);

    if (NULL == vc->mrail.packetized_recv) {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER,
                                 "**fail", 0);
        goto fn_exit;
    }

    DEBUG_PRINT("[pkt recv], rreq %p, offset %d, count %d\n", rreq, rreq->dev.iov_offset,
            rreq->dev.iov_count);

    mpi_errno = MPIDI_CH3I_MRAIL_Fill_Request(rreq, v, skipsize, &nb);
    if (mpi_errno != MPI_SUCCESS) {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER, "**fail",
                                 0);
        goto fn_exit;
    }
    skipsize += nb;
    DEBUG_PRINT("[pkt recv], skipsize %d, filled size %d\n", skipsize, nb);

    if (MPIDI_CH3I_Request_adjust_iov(rreq, nb)) {
        mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
        DEBUG_PRINT("[recv: handle read] adjust req fine, complete %d\n",
                    complete);
        if (mpi_errno != MPI_SUCCESS) {
            mpi_errno =
                MPIR_Err_create_code(mpi_errno,
                                     MPIR_ERR_RECOVERABLE, FCNAME,
                                     __LINE__, MPI_ERR_OTHER, "**fail", 0);
            goto fn_exit;
        }
        while (complete != TRUE) {
            mpi_errno = MPIDI_CH3I_MRAIL_Fill_Request(rreq, v, skipsize, &nb);
            if (mpi_errno != MPI_SUCCESS) {
                mpi_errno =
                    MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER, "**fail",
                                 0);
                goto fn_exit;
            }
            if (!MPIDI_CH3I_Request_adjust_iov(rreq, nb)) {
                goto fn_exit;
            }
            skipsize += nb;

            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
            DEBUG_PRINT("[recv: handle read] adjust req fine, complete %d\n",
                    complete);
            if (mpi_errno != MPI_SUCCESS) {
                mpi_errno =
                    MPIR_Err_create_code(mpi_errno,
                                     MPIR_ERR_RECOVERABLE, FCNAME,
                                     __LINE__, MPI_ERR_OTHER, "**fail", 0);
                goto fn_exit;
            }
        }
        if (TRUE == complete) {
            vc->mrail.packetized_recv = NULL;
        }
    }
  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_PACKETIZED_RECV_DATA);
    return mpi_errno;
}
