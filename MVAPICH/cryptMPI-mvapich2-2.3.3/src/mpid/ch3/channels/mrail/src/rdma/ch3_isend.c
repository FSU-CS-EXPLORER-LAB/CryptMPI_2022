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


#undef FUNCNAME
#define FUNCNAME isend_update_request
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static void isend_update_request(MPID_Request* sreq, void* pkt, int pkt_sz, int nb)
{
    MPIDI_STATE_DECL(MPID_STATE_ISEND_UPDATE_REQUEST);
    MPIDI_FUNC_ENTER(MPID_STATE_ISEND_UPDATE_REQUEST);
#ifdef _ENABLE_CUDA_
    sreq->dev.pending_pkt = MPIU_Malloc(pkt_sz - nb);
    MPIU_Memcpy(sreq->dev.pending_pkt, (char *) pkt+ nb,  pkt_sz - nb);
    sreq->dev.iov[0].MPL_IOV_BUF = (char *) sreq->dev.pending_pkt;
#else
    MPIU_Memcpy(&sreq->dev.pending_pkt, pkt, sizeof(MPIDI_CH3_Pkt_t));
    sreq->dev.iov[0].MPL_IOV_BUF = (char *) &sreq->dev.pending_pkt + nb;
#endif
    sreq->dev.iov[0].MPL_IOV_LEN = pkt_sz - nb;
    sreq->dev.iov_count = 1;
    sreq->dev.iov_offset = 0;
    MPIDI_FUNC_EXIT(MPID_STATE_ISEND_UPDATE_REQUEST);
}

#define DEBUG_PRINT(args...)

static int MPIDI_CH3_SMP_iSend(MPIDI_VC_t * vc, MPID_Request * sreq, void *pkt,
                        MPIDI_msg_sz_t pkt_sz);

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iSend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_iSend(MPIDI_VC_t * vc, MPID_Request * sreq, void *pkt,
                    MPIDI_msg_sz_t pkt_sz)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_ISEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_ISEND);
    int mpi_errno = MPI_SUCCESS;
    MPL_IOV iov[1];
    int complete;

    MPIU_DBG_PRINTF(("ch3_isend\n"));
    MPIDI_DBG_PRINTF((50, FCNAME, "entering"));

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif

    if (SMP_INIT && vc->smp.local_nodes >= 0 &&
        vc->smp.local_nodes != g_smpi.my_local_id)
    {
        mpi_errno = MPIDI_CH3_SMP_iSend(vc, sreq, pkt, pkt_sz);
        if (mpi_errno != MPI_SUCCESS) MPIR_ERR_POP(mpi_errno);
        goto fn_exit;
    }

    /*CM code*/
    if ((vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE
#ifdef _ENABLE_XRC_
            || (USE_XRC && VC_XST_ISUNSET (vc, XF_SEND_IDLE))
#endif
            ) || !MPIDI_CH3I_CM_SendQ_empty(vc))
    {
        /*Request need to be queued*/
        MPIDI_DBG_PRINTF((55, FCNAME, "not connected, enqueuing"));
        isend_update_request(sreq, pkt, pkt_sz, 0);
        MPIDI_CH3I_CM_SendQ_enqueue(vc, sreq);
        if (vc->ch.state == MPIDI_CH3I_VC_STATE_UNCONNECTED)
        {
            MPIDI_CH3I_CM_Connect(vc);
        }
        goto fn_exit;
    }

    /* The RDMA implementation uses a fixed length header, the size of which is the maximum of all possible packet headers */

    if (MPIDI_CH3I_SendQ_empty(vc)) {   /* MT */
        int nb;
        vbuf *buf;
        MPIDI_DBG_PRINTF((55, FCNAME,
                          "send queue empty, attempting to write"));

        /* MT: need some signalling to lock down our right to use the channel, thus insuring that the progress engine does
           also try to write */

        iov[0].MPL_IOV_BUF = pkt;
        iov[0].MPL_IOV_LEN = pkt_sz;

        mpi_errno =
            MPIDI_CH3I_MRAILI_Eager_send(vc, iov, 1, pkt_sz, &nb, &buf);
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
                     * cc
                     * because the transfer may be complete, but
                     request may still be active (see MPI_Ssend()) */
                    MPIDI_CH3I_SendQ_enqueue_head(vc, sreq);
                    vc->ch.send_active = sreq;
                } else {
                    vc->ch.send_active = MPIDI_CH3I_SendQ_head(vc);
                }

            }
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            buf->sreq = (void *) sreq;
            mpi_errno = MPI_SUCCESS;
        } else {
            /* Connection just failed.  Mark the request complete and return an
             * error. */
            vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
            /* MT - CH3U_Request_complete performs write barrier */
            MPID_Request_complete(sreq);

        }
        goto fn_exit;

    } else {
        MPIDI_DBG_PRINTF((55, FCNAME, "send queue not empty, enqueuing"));
        isend_update_request(sreq, pkt, pkt_sz, 0);
        MPIDI_CH3I_SendQ_enqueue(vc, sreq);
    }

fn_exit:
#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_ISEND);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iSend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3_SMP_iSend(MPIDI_VC_t * vc, MPID_Request * sreq, void *pkt,
                        MPIDI_msg_sz_t pkt_sz)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_SMP_ISEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_SMP_ISEND);

    DEBUG_PRINT("entering ch3_isend\n");

    if (MPIDI_CH3I_SMP_SendQ_empty(vc)) {       /* MT */
        int nb;
        MPL_IOV iov[1];
        iov[0].MPL_IOV_BUF = pkt;
        iov[0].MPL_IOV_LEN = pkt_sz;

        MPIDI_CH3I_SMP_writev(vc, iov, 1, &nb);
        DEBUG_PRINT("wrote %d bytes\n", nb);

        if (nb == pkt_sz)
        {
            int complete;
            DEBUG_PRINT("write complete, calling MPIDI_CH3U_Handle_send_req()\n");
            mpi_errno = MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
            if (mpi_errno != MPI_SUCCESS) MPIR_ERR_POP(mpi_errno);

            if (!complete)
            {
                /* NOTE: dev.iov_count is used to detect completion instead of cc because
                 * the transfer may be complete, but the
                 * request may still be active (see MPI_Ssend())
                 */
                MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
                vc->smp.send_active = sreq;
            }
            else
            {
                vc->smp.send_active = MPIDI_CH3I_SMP_SendQ_head(vc);
            }
        }
        else
        {
            MPIDI_DBG_PRINTF((55, FCNAME, "partial write, enqueuing at head"));
            isend_update_request(sreq, pkt, pkt_sz, nb);
            MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
            vc->smp.send_active = sreq;
        }
    }
    else
    {
        DEBUG_PRINT("send queue not empty, enqueuing\n");
        isend_update_request(sreq, pkt, pkt_sz, 0);
        MPIDI_CH3I_SMP_SendQ_enqueue(vc, sreq);
    }

fn_exit:
   MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_SMP_ISEND);
   return mpi_errno;
fn_fail:
   goto fn_exit;
}
