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

#include "mpid_mrail_rndv.h"
#include "mpidimpl.h"

/*
 * This file contains the implementation of the rendezvous protocol
 * for MPI point-to-point messaging.
 */

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_RndvSend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/* MPID_MRAIL_RndvSend - Send a request to perform a rendezvous send */
int MPID_MRAIL_RndvSend (
    MPID_Request** sreq_p,
    const void* buf,
    MPI_Aint count,
    MPI_Datatype datatype,
    int dt_contig,
    MPIDI_msg_sz_t data_sz,
    MPI_Aint dt_true_lb,
    int rank,
    int tag,
    MPID_Comm* comm,
    int context_offset)
{
    MPIDI_CH3_Pkt_t upkt;
    MPIDI_CH3_Pkt_rndv_req_to_send_t * const rts_pkt = &upkt.rndv_req_to_send;
    MPIDI_VC_t * vc;
    MPID_Request *sreq =*sreq_p;
    int          mpi_errno = MPI_SUCCESS;
#if defined(MPID_USE_SEQUENCE_NUMBERS)
    MPID_Seqnum_t seqnum;
#endif /* defined(MPID_USE_SEQUENCE_NUMBERS) */
    MPIDI_STATE_DECL(MPID_STATE_MRAIL_RNDVSEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAIL_RNDVSEND);
	
    PRINT_DEBUG(DEBUG_RNDV_verbose,
            "Rndv Send to rank: %d, tag: %d, context: %d, buf: %p, size: %ld, contig: %d\n",
            rank, tag, comm->context_id + context_offset, buf, data_sz, dt_contig);
    MPIU_DBG_MSG_D(CH3_OTHER,VERBOSE,
		   "sending rndv RTS, data_sz=" MPIDI_MSG_SZ_FMT, data_sz);
	    
    sreq->partner_request = NULL;
	
    MPIDI_Pkt_init(rts_pkt, MPIDI_CH3_PKT_RNDV_REQ_TO_SEND);
    rts_pkt->match.parts.rank	      = comm->rank;
    rts_pkt->match.parts.tag	      = tag;
    rts_pkt->match.parts.context_id = comm->context_id + context_offset;
    rts_pkt->sender_req_id    = sreq->handle;
    rts_pkt->data_sz	      = data_sz;
#if defined(_ENABLE_CUDA_)
    if(rdma_enable_cuda) {
        rts_pkt->rndv.cuda_transfer_mode = sreq->mrail.cuda_transfer_mode;
    }
#endif
    MPIDI_Comm_get_vc(comm, rank, &vc);
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(rts_pkt, seqnum);
    MPIDI_Request_set_seqnum(sreq, seqnum);

    MPIU_DBG_MSGPKT(vc,tag,rts_pkt->match.parts.context_id,rank,data_sz,"Rndv");

    if (dt_contig) 
    {
	MPIU_DBG_MSG_D(CH3_OTHER,VERBOSE,"  contiguous rndv data, data_sz="
		       MPIDI_MSG_SZ_FMT, data_sz);
		
	sreq->dev.OnDataAvail = 0;
	
	sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST) ((char*)sreq->dev.user_buf + dt_true_lb);
	sreq->dev.iov[0].MPL_IOV_LEN = data_sz;
	sreq->dev.iov_count = 1;
    }
    else
    {
	sreq->dev.segment_ptr = MPID_Segment_alloc( );
	/* if (!sreq->dev.segment_ptr) { MPIR_ERR_POP(); } */
	MPID_Segment_init(sreq->dev.user_buf, sreq->dev.user_count,
			  sreq->dev.datatype, sreq->dev.segment_ptr, 0);
	sreq->dev.iov_count = MPL_IOV_LIMIT;
	sreq->dev.segment_first = 0;
	sreq->dev.segment_size = data_sz;
	/* One the initial load of a send iov req, set the OnFinal action (null
	   for point-to-point) */
	sreq->dev.OnFinal = 0;
	mpi_errno = MPIDI_CH3U_Request_load_send_iov(sreq, &sreq->dev.iov[0],
						     &sreq->dev.iov_count);
    /* Fallback to R3 for non-contig transfers */
    if (IS_VC_SMP(vc)) {
        sreq->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
        MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(sreq);
    }
#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && sreq->dev.OnDataAvail == 
                        MPIDI_CH3_ReqHandler_pack_cudabuf) {
        int complete ATTRIBUTE((unused));
        MPIDI_CH3_ReqHandler_pack_cudabuf_stream(vc, sreq, &complete, (void *) stream_d2h);
        sreq->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)sreq->dev.tmpbuf;
        sreq->dev.iov[0].MPL_IOV_LEN = sreq->dev.segment_size;
        sreq->dev.iov_count = 1;
        sreq->dev.OnDataAvail = 0;
        /* When processing datatype operations, we can end up with cases where
         * one IOV is on the host while the other is on the device. This catches
         * this situation. In particular, when we are using MPI_BOTTOM based
         * scheme in MPIR_Igather_binomial, MPIR_Igather_binomial_MV2,
         * MPIR_Gather_intra we can hit this situation. */
        rts_pkt->rndv.cuda_transfer_mode = sreq->mrail.cuda_transfer_mode = DEVICE_TO_DEVICE;
    }
#endif
	/* --BEGIN ERROR HANDLING-- */
	if (mpi_errno != MPI_SUCCESS)
	{
	    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
					     FCNAME, __LINE__, MPI_ERR_OTHER,
					     "**ch3|loadsendiov", 0);
	    goto fn_exit;
	}
	/* --END ERROR HANDLING-- */
    }
    mpi_errno = MPIDI_CH3_iStartRndvMsg (vc, sreq, &upkt);
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno != MPI_SUCCESS)
    {
	MPIU_Object_set_ref(sreq, 0);
	MPIDI_CH3_Request_destroy(sreq);
	*sreq_p = NULL;
	mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
					 FCNAME, __LINE__, MPI_ERR_OTHER,
					 "**ch3|rtspkt", 0);
	goto fn_exit;
    }
    /* --END ERROR HANDLING-- */
    
 fn_exit:

    MPIDI_FUNC_EXIT(MPID_STATE_MRAIL_RNDVSEND);
    return mpi_errno;
}

/*
 * This routine processes a rendezvous message once the message is matched.
 * It is used in mpid_recv and mpid_irecv.
 */
#undef FUNCNAME
#define FUNCNAME MPID_MRAIL_RndvRecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_MRAIL_RndvRecv (MPIDI_VC_t* vc, MPID_Request* rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MRAIL_RNDVRECV);
    MPIDI_FUNC_ENTER(MPID_STATE_MRAIL_RNDVRECV);
    /* A rendezvous request-to-send (RTS) message has arrived.  We need
       to send a CTS message to the remote process. */
    
    PRINT_DEBUG(DEBUG_RNDV_verbose,
            "Rndv Recv from rank: %d, tag: %d, context: %d, buf: %p, size: %ld\n",
            rreq->dev.match.parts.rank, rreq->dev.match.parts.tag,
            rreq->dev.match.parts.context_id, rreq->dev.user_buf, rreq->dev.recv_data_sz);

    if (rreq->dev.recv_data_sz == 0) {
	MPID_Request_complete(rreq);
    }
    else {
	mpi_errno = MPIDI_CH3U_Post_data_receive_found(rreq);
	if (mpi_errno != MPI_SUCCESS) {
	    MPIR_ERR_SETANDJUMP1(mpi_errno,MPI_ERR_OTHER,
				 "**ch3|postrecv",
				 "**ch3|postrecv %s",
				 "MPIDI_CH3_PKT_RNDV_REQ_TO_SEND");
	}
    }

    if (MPIDI_CH3_RECV_REQ_IS_READ(rreq)) {
        mpi_errno = MPIDI_CH3_RecvRndv(vc, rreq);
    } else {
        mpi_errno = MPIDI_CH3_iStartRndvTransfer (vc, rreq);
    }

    if (mpi_errno != MPI_SUCCESS) {
	MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,
				"**ch3|ctspkt");
    }

 fn_fail:    
    MPIDI_FUNC_EXIT(MPID_STATE_MRAIL_RNDVRECV);
    return mpi_errno;
}
