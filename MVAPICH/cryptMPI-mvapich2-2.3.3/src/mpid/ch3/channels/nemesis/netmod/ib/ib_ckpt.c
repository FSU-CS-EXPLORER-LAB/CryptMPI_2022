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

#include "mpid_nem_impl.h"
#include "ib_init.h"
#include "ib_vc.h"
#include "ib_send.h"

#undef DEBUG_PRINT
#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT(args...)                                      \
do {                                                              \
    int __rank;                                                   \
    UPMI_GET_RANK(&__rank);                                        \
    fprintf(stderr, "[%d][%s:%d] ", __rank, __FILE__, __LINE__);  \
    fprintf(stderr, args);                                        \
    fflush(stderr);                                               \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif


#ifdef ENABLE_CHECKPOINTING

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_ckpt_pause_send_vc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_ckpt_pause_send_vc(MPIDI_VC_t *vc)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_nem_ib_vc_area *vc_ib = VC_IB(vc);
    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_CKPT_PAUSE_SEND_VC);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_CKPT_PAUSE_SEND_VC);

    vc_ib->send_paused = TRUE;

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_CKPT_PAUSE_SEND_VC);
    return mpi_errno;
fn_fail:

    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_pkt_unpause_handler
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_pkt_unpause_handler(MPIDI_VC_t *vc, MPIDI_CH3_Pkt_t *pkt, MPIDI_msg_sz_t *buflen, MPID_Request **rreqp)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_nem_ib_vc_area *vc_ib = VC_IB(vc);
    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_CKPT_UNPAUSE_HANDLER);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_CKPT_UNPAUSE_HANDLER);

    vc_ib->send_paused = FALSE;

    /* There may be a unpause message in the send queue.  If so, just enqueue everything on the send queue. */
    if (MPIDI_CH3I_Sendq_empty(vc_ib->send_queue))
        mpi_errno = MPID_nem_ib_send_queued(vc, (MPIDI_nem_ib_request_queue_t *)&vc_ib->paused_send_queue); /*Need to implement for IB*/
        
    /* if anything is left on the paused queue, put it on the send queue and wait for the reconnect */
    if (!MPIDI_CH3I_Sendq_empty(vc_ib->paused_send_queue)) {
        
        MPIDI_CH3I_Sendq_enqueue_multiple_no_refcount(&vc_ib->send_queue, vc_ib->paused_send_queue.head, vc_ib->paused_send_queue.tail);
        vc_ib->paused_send_queue.head = vc_ib->paused_send_queue.tail = NULL;
    }

fn_exit:
    *buflen = sizeof(MPIDI_CH3_Pkt_t);

    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_CKPT_UNPAUSE_HANDLER);
    return mpi_errno;
fn_fail:

    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_ckpt_continue_vc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_ckpt_continue_vc(MPIDI_VC_t *vc)
{
    int mpi_errno = MPI_SUCCESS;

    MPID_PKT_DECL_CAST(upkt, MPIDI_nem_ib_pkt_unpause_t, unpause_pkt);
    MPID_Request *unpause_req;
    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_CKPT_CONTINUE_VC);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_CKPT_CONTINUE_VC);

    unpause_pkt->type = MPIDI_NEM_PKT_NETMOD;
    unpause_pkt->subtype = MPIDI_NEM_IB_PKT_UNPAUSE;

    fprintf(stderr,"Calling MPID_nem_ib_iStartContigMsg_paused\n");
    mpi_errno = MPID_nem_ib_iStartContigMsg_paused(vc, &upkt, sizeof(MPIDI_nem_ib_pkt_unpause_t), NULL, 0, &unpause_req);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    if (unpause_req)
    {
        if (unpause_req->status.MPI_ERROR) MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPID_Request_release(unpause_req);
        if (mpi_errno) goto fn_fail;
    }

    fprintf(stderr,"Exiting MPID_nem_ib_ckpt_continue_vc\n");

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_CKPT_CONTINUE_VC);
    return mpi_errno;
fn_fail:

    goto fn_exit;
}



#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_ckpt_restart_vc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_ckpt_restart_vc(MPIDI_VC_t *vc)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_CH3_Pkt_t upkt;
    MPIDI_nem_ib_pkt_unpause_t * const pkt = (MPIDI_nem_ib_pkt_unpause_t *)&upkt;
    MPID_Request *sreq;
    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_CKPT_RESTART_VC);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_CKPT_RESTART_VC);

    pkt->type = MPIDI_NEM_PKT_NETMOD;
    pkt->subtype = MPIDI_NEM_IB_PKT_UNPAUSE;

    mpi_errno = MPID_nem_ib_iStartContigMsg_paused(vc, pkt, sizeof(pkt), NULL, 0, &sreq);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    if (sreq != NULL) {
        if (sreq->status.MPI_ERROR != MPI_SUCCESS) {
            mpi_errno = sreq->status.MPI_ERROR;
            MPID_Request_release(sreq);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        }
        MPID_Request_release(sreq);
    }
    
fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_CKPT_RESTART_VC);
    return mpi_errno;
fn_fail:

    goto fn_exit;
}

#endif /* ENABLE_CHECKPOINTING */
