/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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

#include "mpidimpl.h"
#if defined(CHANNEL_MRAIL)
#include "dreg.h"
#endif

#undef FUNCNAME
#define FUNCNAME MPID_Cancel_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Cancel_recv(MPID_Request * rreq)
{
    int netmod_cancelled = TRUE;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_MPID_CANCEL_RECV);
    
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_CANCEL_RECV);
    
    MPIU_Assert(rreq->kind == MPID_REQUEST_RECV);

#if defined (CHANNEL_PSM)
    rreq->psm_flags |= PSM_RECV_CANCEL;
    if(psm_do_cancel(rreq) == MPI_SUCCESS) {
        MPID_cc_set(rreq->cc_ptr, 0);
        MPID_Request_release(rreq);
    }
    goto fn_exit;
#endif

#if defined(CHANNEL_MRAIL)
    /* OSU-MPI2 requires extra step to finish rndv request */ 
    MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(rreq);
#endif /* defined(CHANNEL_MRAIL) */

    /* If the netmod has its own cancel_recv function, we need to call
       it here. ANYSOURCE cancels (netmod and otherwise) are handled by
       MPIDI_CH3U_Recvq_DP below. */
#ifdef ENABLE_COMM_OVERRIDES
    if (rreq->dev.match.parts.rank != MPI_ANY_SOURCE)
    {
        MPIDI_VC_t *vc;
        MPIDI_Comm_get_vc_set_active(rreq->comm, rreq->dev.match.parts.rank, &vc);
        if (vc->comm_ops && vc->comm_ops->cancel_recv)
            netmod_cancelled = !vc->comm_ops->cancel_recv(NULL, rreq);
    }
#endif

    if (netmod_cancelled && MPIDI_CH3U_Recvq_DP(rreq))
    {
	MPIU_DBG_MSG_P(CH3_OTHER,VERBOSE,
		       "request 0x%08x cancelled", rreq->handle);
        MPIR_STATUS_SET_CANCEL_BIT(rreq->status, TRUE);
        MPIR_STATUS_SET_COUNT(rreq->status, 0);
        mpi_errno = MPID_Request_complete(rreq);
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    else
    {
	MPIU_DBG_MSG_P(CH3_OTHER,VERBOSE,
	    "request 0x%08x already matched, unable to cancel", rreq->handle);
    }

 fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_CANCEL_RECV);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}
