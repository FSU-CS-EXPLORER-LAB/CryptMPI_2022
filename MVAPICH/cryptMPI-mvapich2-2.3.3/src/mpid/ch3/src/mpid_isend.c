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

MPIR_T_PVAR_ULONG2_COUNTER_BUCKET_DECL_EXTERN(MV2,mv2_pt2pt_mpid_isend);

/* FIXME: HOMOGENEOUS SYSTEMS ONLY -- no data conversion is performed */

/* FIXME: The routines MPID_Isend, MPID_Issend, MPID_Irsend are nearly 
   identical. It would be better if these did roughly:

   MPID_Irsend -> always eager send (with ready mode for error detection)
   MPID_Issend -> always rendezvous send
   MPID_Isend  -> chose eager/rendezvous based on a threshold (and consider
   making the threshold configurable at either compile time (for best 
   low-latency performance) or run-time (for application tuning).

   Then the 3 routines share little code, particularly if the eager/rendezvous 
   implementations are in their own routine
   */
/*
 * MPID_Isend()
 */
#ifdef CHANNEL_MRAIL
extern MPID_Request *mv2_dummy_request;
#endif

#undef FUNCNAME
#define FUNCNAME MPID_Isend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Isend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
	       int tag, MPID_Comm * comm, int context_offset,
               MPID_Request ** request)
{

    MPIR_T_PVAR_COUNTER_BUCKET_INC(MV2,mv2_pt2pt_mpid_isend,count,datatype);

    MPIDI_msg_sz_t data_sz;
    int dt_contig;
    MPI_Aint dt_true_lb;
    MPID_Datatype * dt_ptr;
    MPID_Request * sreq = NULL;
    MPIDI_VC_t * vc=0;
#if defined(MPID_USE_SEQUENCE_NUMBERS)
    MPID_Seqnum_t seqnum;
#endif    
#if !defined(CHANNEL_MRAIL)
    int eager_threshold = -1;
#endif
#ifdef _ENABLE_CUDA_
    int cuda_transfer_mode = NONE;
#endif 
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPID_ISEND);

    MPIDI_FUNC_ENTER(MPID_STATE_MPID_ISEND);

    MPIU_DBG_MSG_FMT(CH3_OTHER,VERBOSE,(MPIU_DBG_FDEST,
                  "rank=%d, tag=%d, context=%d", 
                  rank, tag, comm->context_id + context_offset));

    /* Check to make sure the communicator hasn't already been revoked */
    if (comm->revoked &&
            MPIR_AGREE_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask) &&
            MPIR_SHRINK_TAG != MPIR_TAG_MASK_ERROR_BITS(tag & ~MPIR_Process.tagged_coll_mask)) {
        MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"Communicator revoked. MPID_ISEND returning");
        MPIR_ERR_SETANDJUMP(mpi_errno,MPIX_ERR_REVOKED,"**revoked");
    }
    
    if (rank == comm->rank && comm->comm_kind != MPID_INTERCOMM)
    {
#if defined (CHANNEL_PSM)
        goto skip_self_send; /* psm will internally do self-send, no special
                                handling is needed here */
#endif /* CHANNEL_PSM */          
	mpi_errno = MPIDI_Isend_self(buf, count, datatype, rank, tag, comm, 
			    context_offset, MPIDI_REQUEST_TYPE_SEND, &sreq);
	goto fn_exit;
    }
#if defined (CHANNEL_PSM)
skip_self_send:
#endif

    if (rank != MPI_PROC_NULL) {
        MPIDI_Comm_get_vc_set_active(comm, rank, &vc);
#ifdef ENABLE_COMM_OVERRIDES
        /* this needs to come before the sreq is created, since the override
         * function is responsible for creating its own request */
        if (vc->comm_ops && vc->comm_ops->isend)
        {
            mpi_errno = vc->comm_ops->isend( vc, buf, count, datatype, rank, tag, comm, context_offset, &sreq);
            goto fn_exit;
        }
#endif
    }

#ifndef CHANNEL_MRAIL
    MPIDI_Request_create_sreq(sreq, mpi_errno, goto fn_exit);
    MPIDI_Request_set_type(sreq, MPIDI_REQUEST_TYPE_SEND);
#endif

    if (rank == MPI_PROC_NULL)
    {
#ifdef CHANNEL_MRAIL
        sreq = mv2_dummy_request;
#else
        MPIU_Object_set_ref(sreq, 1);
        MPID_cc_set(&sreq->cc, 0);
#endif
        goto fn_exit;
    }

    MPIDI_Datatype_get_info(count, datatype, dt_contig, data_sz, dt_ptr, 
			                dt_true_lb);
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) { 
        if (is_device_buffer((void *)buf)) {
            /* buf is in the GPU device memory */
            cuda_transfer_mode = DEVICE_TO_DEVICE;
        } else {
            /* buf is in the main memory */
            cuda_transfer_mode = NONE;
        }
    }
#endif

#ifdef USE_EAGER_SHORT
    if ((data_sz + sizeof(MPIDI_CH3_Pkt_eager_send_t) <= vc->eager_fast_max_msg_sz) &&
        vc->eager_fast_fn && dt_contig
#ifdef _ENABLE_CUDA_
        && cuda_transfer_mode == NONE
#endif
#ifdef CKPT
        && vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE
#endif /* CKPT */
        ) {
        mpi_errno = MPIDI_CH3_EagerContigShortSend(&sreq,
                           MPIDI_CH3_PKT_EAGERSHORT_SEND,
                           (char *)buf + dt_true_lb,
                           data_sz, rank, tag, comm,
                           context_offset );
#ifdef CHANNEL_MRAIL
        if (sreq == NULL) {
            sreq = mv2_dummy_request;
        }
#endif
        goto fn_exit;
    }
#endif

    if (sreq == NULL) {
        MPIDI_Request_create_sreq(sreq, mpi_errno, goto fn_exit);
    }
    MPIDI_Request_set_type(sreq, MPIDI_REQUEST_TYPE_ISEND);

    if (data_sz == 0)
    {
#if defined (CHANNEL_PSM)
        goto eager_send;
#endif /* CHANNEL_PSM */
	MPIDI_CH3_Pkt_t upkt;
	MPIDI_CH3_Pkt_eager_send_t * const eager_pkt = &upkt.eager_send;

	MPIDI_Request_set_msg_type(sreq, MPIDI_REQUEST_EAGER_MSG);
	sreq->dev.OnDataAvail = 0;
	    
	MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"sending zero length message");
	MPIDI_Pkt_init(eager_pkt, MPIDI_CH3_PKT_EAGER_SEND);
	eager_pkt->match.parts.rank = comm->rank;
	eager_pkt->match.parts.tag = tag;
	eager_pkt->match.parts.context_id = comm->context_id + context_offset;
	eager_pkt->sender_req_id = sreq->handle;
	eager_pkt->data_sz = 0;
	
	MPIDI_VC_FAI_send_seqnum(vc, seqnum);
	MPIDI_Pkt_set_seqnum(eager_pkt, seqnum);
	MPIDI_Request_set_seqnum(sreq, seqnum);
	
	MPID_THREAD_CS_ENTER(POBJ, vc->pobj_mutex);
	mpi_errno = MPIDI_CH3_iSend(vc, sreq, eager_pkt, sizeof(*eager_pkt));
	MPID_THREAD_CS_EXIT(POBJ, vc->pobj_mutex);
	/* --BEGIN ERROR HANDLING-- */
	if (mpi_errno != MPI_SUCCESS)
	{
            MPID_Request_release(sreq);
	    sreq = NULL;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**ch3|eagermsg");
	    goto fn_exit;
	}
	/* --END ERROR HANDLING-- */

	goto fn_exit;
    }

#if defined (CHANNEL_PSM)
    if(HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN) {
        sreq->dev.datatype_ptr = dt_ptr;
        MPID_Datatype_add_ref(dt_ptr);
        sreq->psm_flags |= PSM_NEED_DTYPE_RELEASE;
    }
    if(vc->force_eager)
        goto eager_send;
#endif /* CHANNEL_PSM */

#if defined(CHANNEL_MRAIL)
    int i;
    for (i = 0 ; i < rdma_num_extra_polls; i++)
    {
        if (rdma_global_ext_sendq_size > 1)
            MPID_Progress_test();
    }
#endif
    /* FIXME: flow control: limit number of outstanding eager messsages 
       containing data and need to be buffered by the receiver */
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        if(HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN &&
                sreq->dev.datatype_ptr == NULL) {
            sreq->dev.datatype_ptr = dt_ptr;
            MPID_Datatype_add_ref(dt_ptr);
        }
        sreq->mrail.cuda_transfer_mode = cuda_transfer_mode;

        /*forces rndv for some IPC based CUDA transfers*/
#ifdef HAVE_CUDA_IPC
        if (vc->smp.local_rank != -1 &&
            sreq->mrail.cuda_transfer_mode != NONE &&
            rdma_cuda_ipc) { 

            /*initialize IPC buffered channel if not initialized*/
            if (rdma_cuda_dynamic_init && 
                cuda_initialized && 
                vc->smp.can_access_peer == CUDA_IPC_UNINITIALIZED) {
                cudaipc_init_dynamic (vc);
            }

            if (vc->smp.can_access_peer == CUDA_IPC_ENABLED &&
                cudaipc_stage_buffered &&
                dt_contig &&
                data_sz >= rdma_cuda_ipc_threshold)  {
                /* force RNDV for CUDA transfers when buffered CUDA IPC is enabled or
                 * if rdma_cuda_smp_ipc is set off */
                if (!rdma_cuda_smp_ipc) {
                    goto rndv_send;
                }
            }
        }
#endif

        /*forces rndv for non IPC based CUDA transfers*/
        if (SMP_INIT && 
            vc->smp.local_rank != -1 &&
            sreq->mrail.cuda_transfer_mode != NONE) {
#ifdef HAVE_CUDA_IPC
            if (rdma_cuda_ipc == 0 || 
                vc->smp.can_access_peer != CUDA_IPC_ENABLED) 
#endif
            {
                goto rndv_send;
            }
        }

    }
#endif

#if defined(CHANNEL_MRAIL)
    if (data_sz + sizeof(MPIDI_CH3_Pkt_eager_send_t) <=	vc->eager_max_msg_sz
        && !vc->force_rndv)
#else /* defined(CHANNEL_MRAIL) */
    MPIDI_CH3_GET_EAGER_THRESHOLD(&eager_threshold, comm, vc);

    /* FIXME: flow control: limit number of outstanding eager messages
       containing data and need to be buffered by the receiver */
    if (data_sz + sizeof(MPIDI_CH3_Pkt_eager_send_t) <= eager_threshold)
#endif /* defined(CHANNEL_MRAIL) */
    {
#if defined (CHANNEL_PSM)
eager_send:
#endif /* _OSU_PSM */
        if (dt_contig) 
        {
            mpi_errno = MPIDI_CH3_EagerContigIsend( &sreq, 
                                MPIDI_CH3_PKT_EAGER_SEND,
                                (char*)buf + dt_true_lb, 
                                data_sz, rank, tag, 
                                comm, context_offset );
        } 
        else 
        {
#if defined (CHANNEL_PSM)
            sreq->psm_flags |= PSM_NON_BLOCKING_SEND;
#endif
            mpi_errno = MPIDI_CH3_EagerNoncontigSend( &sreq, 
                                                          MPIDI_CH3_PKT_EAGER_SEND,
                                                          buf, count, datatype,
                                                          data_sz, rank, tag, 
                                                          comm, context_offset );
#if defined (CHANNEL_PSM)
            goto fn_exit;
#endif            
            /* If we're not complete, then add a reference to the datatype */
            if (sreq && sreq->dev.OnDataAvail && sreq->dev.datatype_ptr == NULL) {
                sreq->dev.datatype_ptr = dt_ptr;
                MPID_Datatype_add_ref(dt_ptr);
            }
        }
    }
    else
    {
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
rndv_send:
#endif
	/* Note that the sreq was created above */
	MPIDI_Request_set_msg_type( sreq, MPIDI_REQUEST_RNDV_MSG );

	mpi_errno = vc->rndvSend_fn( &sreq, buf, count, datatype, dt_contig,
                                     data_sz, dt_true_lb, rank, tag, comm, 
                                     context_offset );
	/* FIXME: fill temporary IOV or pack temporary buffer after send to 
	   hide some latency.  This requires synchronization
           because the CTS packet could arrive and be processed before the 
	   above iStartmsg completes (depending on the progress
           engine, threads, etc.). */
#if defined(CHANNEL_MRAIL)
        /* rndv transfers need to process CTS packet to initiate the actual RDMA transfer */
        MPID_Progress_test();
#endif /* defined(CHANNEL_MRAIL) */
	
	if (sreq && dt_ptr != NULL && sreq->dev.datatype_ptr == NULL)
	{
	    sreq->dev.datatype_ptr = dt_ptr;
	    MPID_Datatype_add_ref(dt_ptr);
	}
    }

  fn_exit:
    *request = sreq;

#if defined(CHANNEL_MRAIL)
    for (i = 0 ; i < rdma_num_extra_polls; i++)
    {
        if (rdma_global_ext_sendq_size > 1)
            MPID_Progress_test();
    }
#endif

    MPIU_DBG_STMT(CH3_OTHER,VERBOSE,
    {
	if (sreq != NULL)
	{
	    MPIU_DBG_MSG_P(CH3_OTHER,VERBOSE,"request allocated, handle=0x%08x", sreq->handle);
	}
    }
		  );
    
  fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_ISEND);
    return mpi_errno;
}
