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

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2,unexpected_recvs_rendezvous);
MPIR_T_PVAR_ULONG2_COUNTER_BUCKET_DECL_EXTERN(MV2,mv2_pt2pt_mpid_recv); 

#undef FUNCNAME
#define FUNCNAME MPID_Recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Recv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	      MPID_Comm * comm, int context_offset,
	      MPI_Status * status, MPID_Request ** request)
{
    MPIR_T_PVAR_COUNTER_BUCKET_INC(MV2,mv2_pt2pt_mpid_recv,count,datatype);

    /* FIXME: in the common case, we want to simply complete the message
       and make as few updates as possible.
       Note in addition that this routine is used only by MPI_Recv (a
       blocking routine; the intent of the interface (which returns 
       a request) was to simplify the handling of the case where the
       message was not found in the unexpected queue. */

    int mpi_errno = MPI_SUCCESS;
    MPID_Request * rreq;
    int found;
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    int is_devbuf = 0;
#endif
    MPIDI_STATE_DECL(MPID_STATE_MPID_RECV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_RECV);

    MPIU_DBG_MSG_FMT(CH3_OTHER,VERBOSE,(MPIU_DBG_FDEST,
                      "rank=%d, tag=%d, context=%d", rank, tag,
		      comm->recvcontext_id + context_offset));
    
    if (rank == MPI_PROC_NULL)
    {
	MPIR_Status_set_procnull(status);
	rreq = NULL;
	goto fn_exit;
    }

    /* Check to make sure the communicator hasn't already been revoked */
    if (comm->revoked &&
            MPIR_AGREE_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask) &&
            MPIR_SHRINK_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask)) {
        MPIR_ERR_SETANDJUMP(mpi_errno,MPIX_ERR_REVOKED,"**revoked");
    }

    /* psm buffers unexpected messages internally, so we don't need the ch3
       unexpected queue. If data is contig just call the blocking MPIDI_CH3_Recv
       If data is non-contig,... */

#if defined(CHANNEL_PSM)
    MPI_Aint dt_true_lb;
    MPID_Datatype *dt_ptr;
    MPIDI_msg_sz_t data_sz;
    MPIDI_msg_sz_t pksz;
    int dt_contig;
    void *pkbuf;

    MPIDI_Datatype_get_info(count, datatype, dt_contig, data_sz, dt_ptr, dt_true_lb);
    if(dt_contig) {
        mpi_errno = MPIDI_CH3_Recv(rank, tag, comm->recvcontext_id + context_offset,
                (char *)buf + dt_true_lb, data_sz, status, request);
        rreq = *request;
    } else {
        PSMSG(fprintf(stderr, "non-contig recv for psm\n"));
        MPIR_Pack_size_impl(count, datatype, &pksz);
        if(count == 0) {
            pksz = 0;
        }
        pkbuf = MPIU_Malloc(pksz);
        if(!pkbuf) {
        	MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_NO_MEM, "**nomem");
        }
        mpi_errno = MPIDI_CH3_Recv(rank, tag, comm->recvcontext_id + context_offset,
                pkbuf, pksz, status, request);
        rreq = *request;
        if(rreq && (rreq->psm_flags & PSM_NON_BLOCKING_RECV)) {
            rreq->psm_flags |= PSM_NON_CONTIG_REQ;
            rreq->pkbuf = pkbuf;
            rreq->pksz = pksz;
            rreq->dev.user_buf = (char *)buf;// + dt_true_lb;
            rreq->dev.user_count = count;
            rreq->dev.datatype = datatype;
        } else {
            mpi_errno = psm_do_unpack(count, datatype, comm, pkbuf, pksz, 
                        (char *)buf/* + dt_true_lb*/ , data_sz);
            MPIU_Free(pkbuf);
        }
    }
    goto fn_exit;
#endif

    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    rreq = MPIDI_CH3U_Recvq_FDU_or_AEP(rank, tag, 
				       comm->recvcontext_id + context_offset,
                                       comm, buf, count, datatype, &found);
    if (rreq == NULL) {
	MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
	MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,"**nomemreq");
    }
    MPIDI_Request_set_type(rreq, MPIDI_REQUEST_TYPE_RECV);

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        if (is_device_buffer(buf)) {
            /* buf is in the GPU device memory */
            rreq->mrail.cuda_transfer_mode = DEVICE_TO_DEVICE;
        } else {
            /* buf is in the host memory*/
            rreq->mrail.cuda_transfer_mode = NONE;
        }
    }
#endif

    if (found)
    {
	MPIDI_VC_t * vc;

	/* Message was found in the unexepected queue */
	MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"request found in unexpected queue");

	/* Release the message queue - we've removed this request from 
	   the queue already */
	MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
	if (MPIDI_Request_get_msg_type(rreq) == MPIDI_REQUEST_EAGER_MSG)
	{
	    int recv_pending;
	    
	    /* This is an eager message. */
	    MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"eager message in the request");

	    if (MPIDI_Request_get_sync_send_flag(rreq))
	    {
		MPIDI_Comm_get_vc_set_active(comm, rreq->dev.match.parts.rank, &vc);
        MPIR_ERR_CHKANDJUMP1(vc->state == MPIDI_VC_STATE_MORIBUND, mpi_errno, MPIX_ERR_PROC_FAILED, "**comm_fail", "**comm_fail %d", rreq->dev.match.parts.rank);
		mpi_errno = MPIDI_CH3_EagerSyncAck( vc, rreq );
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	    }
	    
            /* the request was found in the unexpected queue, so it has a
               recv_pending_count of at least 1, corresponding to this matching */
            MPIDI_Request_decr_pending(rreq);
            MPIDI_Request_check_pending(rreq, &recv_pending);

            if (MPID_Request_is_complete(rreq)) {
                /* is it ever possible to have (cc==0 && recv_pending>0) ? */
                MPIU_Assert(!recv_pending);

                /* All of the data has arrived, we need to unpack the data and 
                   then free the buffer and the request. */
                if (rreq->dev.recv_data_sz > 0)
                {
                    MPIDI_CH3U_Request_unpack_uebuf(rreq);
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
                    is_devbuf = is_device_buffer((void *)rreq->dev.tmpbuf);
                    if (is_devbuf)
                    {
                        cudaFree(rreq->dev.tmpbuf);
                    }
                    else
#endif
                    {
                       MPIU_Free(rreq->dev.tmpbuf);
                    }
                }

                mpi_errno = rreq->status.MPI_ERROR;
                if (status != MPI_STATUS_IGNORE)
                {
                    *status = rreq->status;
                }

                MPID_Request_release(rreq);
                rreq = NULL;

                goto fn_exit;
            }
	    else
	    {
                /* there should never be outstanding completion events for an unexpected
                 * recv without also having a "pending recv" */
                MPIU_Assert(recv_pending);

		/* The data is still being transfered across the net.  
		   We'll leave it to the progress engine to handle once the
		   entire message has arrived. */
		if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN)
		{
		    MPID_Datatype_get_ptr(datatype, rreq->dev.datatype_ptr);
		    MPID_Datatype_add_ref(rreq->dev.datatype_ptr);
		}
	    }
	}
	else if (MPIDI_Request_get_msg_type(rreq) == MPIDI_REQUEST_RNDV_MSG)
	{
        MPIR_T_PVAR_COUNTER_INC(MV2, unexpected_recvs_rendezvous, 1);
#if defined(CHANNEL_MRAIL)
        MPIDI_Comm_get_vc(comm, rreq->dev.match.parts.rank, &vc);
        mpi_errno = MPIDI_CH3_RecvRndv( vc, rreq );
#else
        MPIDI_Comm_get_vc_set_active(comm, rreq->dev.match.parts.rank, &vc);
        MPIR_ERR_CHKANDJUMP1(vc->state == MPIDI_VC_STATE_MORIBUND, mpi_errno, MPIX_ERR_PROC_FAILED, "**comm_fail", "**comm_fail %d", rreq->dev.match.parts.rank);
        mpi_errno = vc->rndvRecv_fn( vc, rreq );
#endif
	    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	    if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN)
	    {
		MPID_Datatype_get_ptr(datatype, rreq->dev.datatype_ptr);
		MPID_Datatype_add_ref(rreq->dev.datatype_ptr);
	    }
	}
	else if (MPIDI_Request_get_msg_type(rreq) == MPIDI_REQUEST_SELF_MSG)
	{
	    mpi_errno = MPIDI_CH3_RecvFromSelf( rreq, buf, count, datatype );
	    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	    if (status != MPI_STATUS_IGNORE)
	    {
		*status = rreq->status;
	    }
	}
	else
	{
	    /* --BEGIN ERROR HANDLING-- */
#ifdef HAVE_ERROR_CHECKING
            int msg_type = MPIDI_Request_get_msg_type(rreq);
#endif
            MPID_Request_release(rreq);
	    rreq = NULL;
	    MPIR_ERR_SETANDJUMP1(mpi_errno,MPI_ERR_INTERN, "**ch3|badmsgtype",
                                 "**ch3|badmsgtype %d", msg_type);
	    /* --END ERROR HANDLING-- */
	}
    }
    else
    {
	/* Message has yet to arrived.  The request has been placed on the 
	   list of posted receive requests and populated with
           information supplied in the arguments. */
	MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"request allocated in posted queue");

	/* FIXME: We do not need to add a datatype reference if
	   the request is blocking.  This is currently added because
	   of the actions that are taken when a request is freed. 
	   (specifically, the datatype and comm both have their refs
	   decremented, and are freed if the refs are zero) */
	if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN)
	{
	    MPID_Datatype_get_ptr(datatype, rreq->dev.datatype_ptr);
	    MPID_Datatype_add_ref(rreq->dev.datatype_ptr);
	}

	rreq->dev.recv_pending_count = 1;
	/* We must wait until here to exit the msgqueue critical section
	   on this request (we needed to set the recv_pending_count
	   and the datatype pointer) */
        MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    }

  fn_exit:
    *request = rreq;
    MPIU_DBG_STMT(CH3_OTHER,VERBOSE,
    if (rreq)
    {
	MPIU_DBG_MSG_P(CH3_OTHER,VERBOSE,
		       "request allocated, handle=0x%08x", rreq->handle);
    }
    else
    {
	MPIU_DBG_MSG(CH3_OTHER,VERBOSE,
		     "operation complete, no requests allocated");
    });

 fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_RECV);
    return mpi_errno;
}

#if defined (CHANNEL_PSM)
int psm_do_unpack(int count, MPI_Datatype datatype, MPID_Comm *comm, 
                  void *pkbuf, int pksz, void *inbuf, MPIDI_msg_sz_t data_sz)
{
    MPI_Aint first = 0, last = data_sz;
    MPID_Segment *segp;

    segp = MPID_Segment_alloc();
    if(segp == NULL) {
        return MPI_ERR_NO_MEM;
    }

    MPID_Segment_init(inbuf, count, datatype, segp, 0);
    MPID_Segment_unpack(segp, first, &last, pkbuf);
    MPID_Segment_free(segp);
    PSMSG(fprintf(stderr, "PSM Unpack done\n"));
    return MPI_SUCCESS;
}
#endif
