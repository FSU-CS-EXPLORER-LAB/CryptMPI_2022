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
static inline MPID_Request * create_request(MPL_IOV * iov, int iov_count, 
					    int iov_offset, MPIU_Size_t nb)
{
    MPID_Request * sreq;
    MPIDI_STATE_DECL(MPID_STATE_CREATE_REQUEST);

    MPIDI_FUNC_ENTER(MPID_STATE_CREATE_REQUEST);

    sreq = MPID_Request_create();
    /* --BEGIN ERROR HANDLING-- */
    if (sreq == NULL)
        return NULL;
    /* --END ERROR HANDLING-- */
    MPIU_Object_set_ref(sreq, 2);
    sreq->kind = MPID_REQUEST_SEND;
    MV2_INC_NUM_POSTED_SEND();

    MPIU_Memcpy(sreq->dev.iov, iov, iov_count * sizeof(MPL_IOV));

    if (iov_offset == 0)
    {
	/*
        MPIU_Assert(iov[0].MPL_IOV_LEN == sizeof(MPIDI_CH3_Pkt_t));
	*/
#ifdef _ENABLE_CUDA_
        sreq->dev.pending_pkt = MPIU_Malloc(iov[0].MPL_IOV_LEN);
        MPIU_Memcpy(sreq->dev.pending_pkt, iov[0].MPL_IOV_BUF, iov[0].MPL_IOV_LEN);
        sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)sreq->dev.pending_pkt;
#else
        MPIU_Memcpy(&sreq->dev.pending_pkt, iov[0].MPL_IOV_BUF, sizeof(MPIDI_CH3_Pkt_t));
        sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST) &sreq->dev.pending_pkt;
#endif
    }
    sreq->ch.reqtype = REQUEST_NORMAL;
    sreq->dev.iov[iov_offset].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)((char *)
			sreq->dev.iov[iov_offset].MPL_IOV_BUF + nb);
    sreq->dev.iov[iov_offset].MPL_IOV_LEN -= nb;
    sreq->dev.iov_offset = iov_offset;
    sreq->dev.iov_count = iov_count;
    sreq->dev.OnDataAvail = 0;

    MPIDI_FUNC_EXIT(MPID_STATE_CREATE_REQUEST);
    return sreq;
}

static void MPIDI_CH3_SMP_iStartMsgv(MPIDI_VC_t * vc, MPL_IOV * iov,
                                           int n_iov,
                                           MPID_Request ** sreq_ptr);
/*
 * MPIDI_CH3_iStartMsgv() attempts to send the message immediately.  If the
 * entire message is successfully sent, then NULL is returned.  Otherwise a
 * request is allocated, the iovec and the first buffer pointed to by the
 * iovec (which is assumed to be a MPIDI_CH3_Pkt_t) are copied into the
 * request, and a pointer to the request is returned.  An error condition also
 * results in a request be allocated and the errror being returned in the
 * status field of the request.
 */

/* XXX - What do we do if MPIDI_CH3_Request_create() returns NULL???  If
   MPIDI_CH3_iStartMsgv() returns NULL, the calling code assumes the request
   completely successfully, but the reality is that we couldn't allocate the
   memory for a request.  This seems like a flaw in the CH3 API. */

/* NOTE - The completion action associated with a request created by
   CH3_iStartMsgv() is alway MPIDI_CH3_CA_COMPLETE.  This implies that
   CH3_iStartMsgv() can only be used when the entire message can be described
   by a single iovec of size MPL_IOV_LIMIT. */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iStartMsgv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_iStartMsgv(MPIDI_VC_t * vc, MPL_IOV * iov, int n_iov,
                         MPID_Request ** sreq_ptr)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_ISTARTMSGV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_ISTARTMSGV);
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    vbuf *buf;
    DEBUG_PRINT("ch3_istartmsgv, header %d\n", ((MPIDI_CH3_Pkt_t *)iov[0].MPL_IOV_BUF)->type);
    MPIDI_DBG_PRINTF((50, FCNAME, "entering"));
#ifdef MPICH_DBG_OUTPUT
    if (n_iov > MPL_IOV_LIMIT
        || iov[0].MPL_IOV_LEN > sizeof(MPIDI_CH3_Pkt_t))
    {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**arg");
    }
#endif

    MPIDI_DBG_Print_packet((MPIDI_CH3_Pkt_t *) iov[0].MPL_IOV_BUF);

    DEBUG_PRINT("remote local nodes %d, myid %d\n", 
                vc->smp.local_nodes, g_smpi.my_local_id);

#if defined(CKPT)
    MPIDI_CH3I_CR_lock();
#endif

    if (SMP_INIT && vc->smp.local_nodes >= 0 &&
        vc->smp.local_nodes != g_smpi.my_local_id) {
        MPIDI_CH3_SMP_iStartMsgv(vc, iov, n_iov, sreq_ptr);
        MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
#ifdef CKPT
	MPIDI_CH3I_CR_unlock();
#endif
        return(mpi_errno);
    }
#ifdef _ENABLE_UD_
    int len;
    Calculate_IOV_len(iov, n_iov, len);
#endif    
    /*CM code*/
    if ((vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE 
#ifdef _ENABLE_XRC_
            || (USE_XRC && VC_XST_ISUNSET (vc, XF_SEND_IDLE))
#endif
            ) || !MPIDI_CH3I_CM_SendQ_empty(vc)) {
        /*Request need to be queued*/
        MPIDI_DBG_PRINTF((55, FCNAME, "not connected, enqueuing"));
        sreq = create_request(iov, n_iov, 0, 0);
        MPIDI_CH3I_CM_SendQ_enqueue(vc, sreq);
        if (vc->ch.state == MPIDI_CH3I_VC_STATE_UNCONNECTED)  {
            mpi_errno = MPIDI_CH3I_CM_Connect(vc);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        goto fn_exit;
    }

    /* If send queue is empty attempt to send
       data, queuing any unsent data. */
    if (MPIDI_CH3I_SendQ_empty(vc)) {   /* MT */
        int nb;
        int pkt_len;

        /* MT - need some signalling to lock down our right to use the
           channel, thus insuring that the progress engine does also try to
           write */
        Calculate_IOV_len(iov, n_iov, pkt_len);
        DEBUG_PRINT("[send], n_iov: %d, pkt_len %d\n", n_iov,
                    pkt_len);
        
        if (pkt_len > MRAIL_MAX_EAGER_SIZE
#ifdef _ENABLE_UD_
         || (!(vc->mrail.state & MRAILI_RC_CONNECTED) && pkt_len > MRAIL_MAX_UD_SIZE)
#endif
        ) {
            sreq = create_request(iov, n_iov, 0, 0);
            mpi_errno = MPIDI_CH3_Packetized_send(vc, sreq);
            if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                mpi_errno = MPI_SUCCESS;
            }
            goto fn_exit;
        }

        /* TODO: Codes to send pkt through send/recv path */
        mpi_errno =
            MPIDI_CH3I_MRAILI_Eager_send(vc, iov, n_iov, pkt_len, &nb, &buf);
        DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno, nb);

        if (mpi_errno == MPI_SUCCESS) {
            DEBUG_PRINT("[send path] eager send return %d bytes\n", nb);
        } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
            /* fast rdma ok but cannot send: there is no send wqe available */
            sreq = create_request(iov, n_iov, 0, 0);
            buf->sreq = (void *) sreq;
            mpi_errno = MPI_SUCCESS;
        } else {
            sreq = MPID_Request_create();
            if (sreq == NULL) {
                MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
            }
            sreq->kind = MPID_REQUEST_SEND;
            MV2_INC_NUM_POSTED_SEND();
            MPID_cc_set(&sreq->cc, 0);
            /* TODO: Create an appropriate error message based on the value of errno
             * */
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
        }
        goto fn_exit;
    }
    sreq = create_request(iov, n_iov, 0, 0);
    MPIDI_CH3I_SendQ_enqueue(vc, sreq);
fn_exit:
    *sreq_ptr = sreq;
#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif

    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_ISTARTMSGV);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_SMP_iStartMsgv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static void MPIDI_CH3_SMP_iStartMsgv(MPIDI_VC_t * vc,
                                    MPL_IOV * iov, int n_iov,
                                    MPID_Request ** sreq_ptr)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSGV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSGV);
    MPID_Request *sreq = NULL;

    DEBUG_PRINT("entering ch3_smp_istartmsgv\n");

    /* If send queue is empty attempt to send
       data, queuing any unsent data. */
    if (MPIDI_CH3I_SMP_SendQ_empty(vc)) {
        int nb;
        /* MT - need some signalling to lock down our right to use the
           channel, thus insuring that the progress engine does also try to
           write */
        MPIDI_CH3I_SMP_writev(vc, iov, n_iov, &nb);
        int offset = 0;
        DEBUG_PRINT("ch3_smp_istartmsgv: writev returned %d bytes\n", nb);

        while (offset < n_iov) {
            if (nb >= (int) iov[offset].MPL_IOV_LEN) {
                nb -= iov[offset].MPL_IOV_LEN;
                ++offset;
            } else {
                DEBUG_PRINT("ch3_istartmsgv: shm_writev did not complete the send, allocating request\n");
                sreq = create_request(iov, n_iov, offset, nb);
                MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
                vc->smp.send_active = sreq;
                break;
            } 
        }
    } else {
        sreq = create_request(iov, n_iov, 0, 0);
        MPIDI_CH3I_SMP_SendQ_enqueue(vc, sreq);
    }

    *sreq_ptr = sreq;
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_SMP_ISTARTMSGV);
}
