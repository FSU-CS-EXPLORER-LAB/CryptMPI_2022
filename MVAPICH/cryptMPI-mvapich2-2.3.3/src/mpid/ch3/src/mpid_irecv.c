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
MPIR_T_PVAR_ULONG2_COUNTER_BUCKET_DECL_EXTERN(MV2,mv2_pt2pt_mpid_irecv);

#undef FUNCNAME
#define FUNCNAME MPID_Irecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Irecv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	       MPID_Comm * comm, int context_offset,
               MPID_Request ** request)
{
    MPIR_T_PVAR_COUNTER_BUCKET_INC(MV2,mv2_pt2pt_mpid_irecv,count,datatype);

    MPID_Request * rreq;
    int found;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPID_IRECV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_IRECV);

    MPIU_DBG_MSG_FMT(CH3_OTHER,VERBOSE,(MPIU_DBG_FDEST,
			"rank=%d, tag=%d, context=%d", 
			rank, tag, comm->recvcontext_id + context_offset));

    if (rank == MPI_PROC_NULL)
    {
        MPIDI_Request_create_null_rreq(rreq, mpi_errno, goto fn_fail);
        MV2_INC_NUM_POSTED_RECV();
        goto fn_exit;
    }

    /* Check to make sure the communicator hasn't already been revoked */
    if (comm->revoked &&
            MPIR_AGREE_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask) &&
            MPIR_SHRINK_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask)) {
        MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"Comm has been revoked. Returning from MPID_IRECV.");
        MPIR_ERR_SETANDJUMP(mpi_errno,MPIX_ERR_REVOKED,"**revoked");
    }

#if defined (CHANNEL_PSM) /* psm: post request to psm library and return request
                           to MPI_Recv. */
    MPI_Aint dt_true_lb;
    MPID_Datatype *dt_ptr;
    MPIDI_msg_sz_t data_sz;
    MPIDI_msg_sz_t pksz;
    int dt_contig;
    void *pkbuf;

    MPIDI_Datatype_get_info(count, datatype, dt_contig, data_sz, dt_ptr, dt_true_lb);
	rreq = MPID_Request_create();
    MPIU_Object_set_ref(rreq, 2);
	rreq->kind = MPID_REQUEST_RECV;
    MPIR_Comm_add_ref(comm);
    rreq->comm = comm;
    rreq->dev.user_buf	 = (char *)buf ;//+ dt_true_lb;
    rreq->dev.user_count = count;
    rreq->dev.datatype	 = datatype;
	rreq->dev.match.parts.tag	   = tag;
	rreq->dev.match.parts.rank	   = rank;
	rreq->dev.match.parts.context_id = comm->recvcontext_id + context_offset;
    if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN) {
        MPID_Datatype_get_ptr(datatype, rreq->dev.datatype_ptr);
		MPID_Datatype_add_ref(rreq->dev.datatype_ptr);
        rreq->psm_flags |= PSM_NEED_DTYPE_RELEASE;
    }
 
    if(dt_contig) {
        mpi_errno = MPIDI_CH3_iRecv(rank, tag, comm->recvcontext_id + context_offset,
                (char *)buf + dt_true_lb, data_sz, rreq);
    } else {
        PSMSG(fprintf(stderr, "non-contig I-recv for psm\n"));
        MPIR_Pack_size_impl(count, datatype, &pksz);
        pkbuf = MPIU_Malloc(pksz);
        if(!pkbuf) {
        	MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_NO_MEM, "**nomem");
        }
        rreq->pksz = pksz;
        rreq->pkbuf = pkbuf;
        rreq->psm_flags |= PSM_NON_CONTIG_REQ;
        mpi_errno = MPIDI_CH3_iRecv(rank, tag, comm->recvcontext_id + context_offset,
                pkbuf, pksz, rreq);
        if(mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    goto fn_exit;
#endif /* CHANNEL_PSM */

    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    rreq = MPIDI_CH3U_Recvq_FDU_or_AEP(rank, tag, 
				       comm->recvcontext_id + context_offset,
                                       comm, buf, count, datatype, &found);
    if (rreq == NULL)
    {
	MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
	MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,"**nomemreq");
    }
    MPIDI_Request_set_type(rreq, MPIDI_REQUEST_TYPE_IRECV);

#ifdef _ENABLE_CUDA_
    int device_buf = 0;
    if (rdma_enable_cuda) {
        device_buf = is_device_buffer(buf);
    }
#endif
    if (found)
    {
	MPIDI_VC_t * vc;

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        if (device_buf) {
            /* buf is in the GPU device memory */
            rreq->mrail.cuda_transfer_mode = DEVICE_TO_DEVICE;
        } else {
            /* buf is in the host memory*/
            rreq->mrail.cuda_transfer_mode = NONE;
        }
    }
#endif
	
	/* Message was found in the unexepected queue */
	MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"request found in unexpected queue");

	/* Release the message queue - we've removed this request from 
	   the queue already */
	MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);

	if (MPIDI_Request_get_msg_type(rreq) == MPIDI_REQUEST_EAGER_MSG)
	{
	    int recv_pending;
	    
	    /* This is an eager message */
	    MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"eager message in the request");
	    
	    /* If this is a eager synchronous message, then we need to send an 
	       acknowledgement back to the sender. */
	    if (MPIDI_Request_get_sync_send_flag(rreq))
	    {
		MPIDI_Comm_get_vc_set_active(comm, rreq->dev.match.parts.rank, &vc);
		mpi_errno = MPIDI_CH3_EagerSyncAck( vc, rreq );
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	    }

            /* the request was found in the unexpected queue, so it has a
               recv_pending_count of at least 1 */
            MPIDI_Request_decr_pending(rreq);
            MPIDI_Request_check_pending(rreq, &recv_pending);

            if (MPID_Request_is_complete(rreq)) {
                /* is it ever possible to have (cc==0 && recv_pending>0) ? */
                MPIU_Assert(!recv_pending);

                /* All of the data has arrived, we need to copy the data and 
                   then free the buffer. */
                if (rreq->dev.recv_data_sz > 0)
                {
                    MPIDI_CH3U_Request_unpack_uebuf(rreq);
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
                if (rdma_enable_cuda && rdma_cuda_smp_ipc 
                        && is_device_buffer(rreq->dev.tmpbuf)) {
                   cudaFree(rreq->dev.tmpbuf);
                }
                else
#endif
                {
                    MPIU_Free(rreq->dev.tmpbuf);
                }

                }

                mpi_errno = rreq->status.MPI_ERROR;
                goto fn_exit;
            }
	    else
	    {
                /* there should never be outstanding completion events for an unexpected
                 * recv without also having a "pending recv" */
                MPIU_Assert(recv_pending);
		/* The data is still being transfered across the net.  We'll 
		   leave it to the progress engine to handle once the
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
         MPIDI_Comm_get_vc_set_active(comm, rreq->dev.match.parts.rank, &vc);

#ifdef _ENABLE_CUDA_
        if (rdma_enable_cuda) {
            if (device_buf) {
                /* buf is in the GPU device memory */
                rreq->mrail.cuda_transfer_mode = DEVICE_TO_DEVICE;
            } else {
                /* buf is in the host memory*/
                rreq->mrail.cuda_transfer_mode = NONE;
            }
        }
#endif


	
#if defined(CHANNEL_MRAIL)
        mpi_errno = MPIDI_CH3_RecvRndv( vc, rreq );
#else
	    mpi_errno = vc->rndvRecv_fn( vc, rreq );
#endif
	    if (mpi_errno) MPIR_ERR_POP( mpi_errno );
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
	
	if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN)
	{
	    MPID_Datatype_get_ptr(datatype, rreq->dev.datatype_ptr);
	    MPID_Datatype_add_ref(rreq->dev.datatype_ptr);
	}

#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda) {
        if (device_buf) {
            /* buf is in the GPU device memory */
            rreq->mrail.cuda_transfer_mode = DEVICE_TO_DEVICE;
        } else {
            /* buf is in the host memory*/
            rreq->mrail.cuda_transfer_mode = NONE;
        }
    }
#endif

	rreq->dev.recv_pending_count = 1;

	/* We must wait until here to exit the msgqueue critical section
	   on this request (we needed to set the recv_pending_count
	   and the datatype pointer) */
        MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    }

  fn_exit:
    *request = rreq;
    MPIU_DBG_MSG_P(CH3_OTHER,VERBOSE,"request allocated, handle=0x%08x", 
		   rreq->handle);

 fn_fail:
    MPIU_DBG_MSG_D(CH3_OTHER,VERBOSE,"IRECV errno: 0x%08x", mpi_errno);
    MPIU_DBG_MSG_D(CH3_OTHER,VERBOSE,"(class: %d)", MPIR_ERR_GET_CLASS(mpi_errno));
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_IRECV);
    return mpi_errno;
}
