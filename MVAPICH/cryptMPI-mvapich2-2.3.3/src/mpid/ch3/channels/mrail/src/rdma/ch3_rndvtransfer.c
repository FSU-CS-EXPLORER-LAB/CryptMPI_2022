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
#include "vbuf.h"
#include "upmi.h"
#include "mpiutil.h"
#include "rdma_impl.h"
#include "smp_smpi.h"

#include "dreg.h"

static int MPIDI_CH3_SMP_Rendezvous_push(MPIDI_VC_t *, MPID_Request *);

#if defined(_SMP_CMA_)
static int MPIDI_CH3_CMA_Rendezvous_push(MPIDI_VC_t *, MPID_Request *);
extern int MPIDI_CH3I_SMP_do_cma_put(MPIDI_VC_t * vc, const void *src, void *dst, ssize_t len);
extern int MPIDI_CH3I_SMP_do_cma_get(MPIDI_VC_t * vc, const void *src, void *dst, ssize_t len);
#endif

MPIDI_VC_t *flowlist;

#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);                                    \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Prepare_rndv_get
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_Prepare_rndv_get(MPIDI_VC_t * vc,
                               MPID_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_GET);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_GET);

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif

    MPIU_Assert(MV2_RNDV_PROTOCOL_RGET == rreq->mrail.protocol);

    MPIDI_CH3I_MRAIL_Prepare_rndv(vc, rreq);

#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_GET);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Prepare_rndv_cts
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Prepare_rndv_cts(MPIDI_VC_t * vc,
                               MPIDI_CH3_Pkt_rndv_clr_to_send_t * cts_pkt,
                               MPID_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_CTS);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_CTS);

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif

#if defined(_ENABLE_CUDA_)
    if(rdma_enable_cuda && rreq->mrail.cuda_transfer_mode != NONE) {
        cts_pkt->rndv.cuda_transfer_mode = DEVICE_TO_DEVICE;
    } else {
        cts_pkt->rndv.cuda_transfer_mode = NONE;
    }
#endif

    if (IS_VC_SMP(vc) && cts_pkt->type == MPIDI_CH3_PKT_RMA_RNDV_CLR_TO_SEND) {
        rreq->mrail.protocol   = MV2_RNDV_PROTOCOL_R3;
        cts_pkt->rndv.protocol = MV2_RNDV_PROTOCOL_R3;
    }

    switch (rreq->mrail.protocol) {
    case MV2_RNDV_PROTOCOL_R3:
        {
            cts_pkt->rndv.protocol = MV2_RNDV_PROTOCOL_R3;
            /*MRAILI_Prepost_R3(); */
            break;
        }
    case MV2_RNDV_PROTOCOL_RPUT:
        {
            MPIDI_CH3I_MRAIL_Prepare_rndv(vc, rreq);
            MPIDI_CH3I_MRAIL_SET_PKT_RNDV(cts_pkt, rreq);
            MPIDI_CH3I_MRAIL_REVERT_RPUT(rreq);
            break;
        }
    case MV2_RNDV_PROTOCOL_RGET:
        {
            int rank;
            UPMI_GET_RANK(&rank);
            fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);
            fprintf(stderr, "RGET preparing CTS?\n");
            mpi_errno = -1;
            break;
        }
#ifdef _ENABLE_UD_
    case MV2_RNDV_PROTOCOL_UD_ZCOPY:
        {
            MPIDI_CH3I_MRAIL_Prepare_rndv_zcopy(vc, rreq);
            MPIDI_CH3I_MRAIL_SET_PKT_RNDV(cts_pkt, rreq);
            if (rreq->mrail.protocol == MV2_RNDV_PROTOCOL_UD_ZCOPY) {
                cts_pkt->rndv.rndv_qpn = ((mv2_rndv_qp_t *) 
                        rreq->mrail.rndv_qp_entry)->ud_qp->qp_num;
                cts_pkt->rndv.hca_index = ((mv2_rndv_qp_t *)
                        rreq->mrail.rndv_qp_entry)->hca_num;
            }
            break;
        }
#endif
    default:
        {
            int rank;
            UPMI_GET_RANK(&rank);
            fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);
            fprintf(stderr,
                    "Unknown protocol %d type from rndv req to send\n",
		    rreq->mrail.protocol);
            mpi_errno = -1;
            break;
        }
    }

#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PREPARE_RNDV_CTS);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iStartRndvTransfer
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_iStartRndvTransfer(MPIDI_VC_t * vc, MPID_Request * rreq)
{
    MPIDI_CH3_Pkt_t upkt;
    MPIDI_CH3_Pkt_rndv_clr_to_send_t *cts_pkt = &upkt.rndv_clr_to_send;
    MPID_Request *cts_req;
    MPID_Seqnum_t seqnum;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_ISTARTRNDVTRANSFER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_ISTARTRNDVTRANSFER);

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif
       
    MPIDI_Pkt_init(cts_pkt, MPIDI_CH3_PKT_RNDV_CLR_TO_SEND);
    if (rreq->dev.iov_count == 1 && rreq->dev.OnDataAvail == NULL)
	cts_pkt->recv_sz = rreq->dev.iov[0].MPL_IOV_LEN;
    else
	cts_pkt->recv_sz = rreq->dev.segment_size;
    
    cts_pkt->sender_req_id = rreq->dev.sender_req_id;
    cts_pkt->receiver_req_id = rreq->handle;
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(cts_pkt, seqnum);    

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda  &&
        ((rreq->mrail.cuda_transfer_mode != NONE &&
            (vc->smp.local_nodes == -1))
#ifdef HAVE_CUDA_IPC
       || (rdma_cuda_ipc && cudaipc_stage_buffered && 
           rreq->mrail.cuda_transfer_mode != NONE &&
           vc->smp.can_access_peer == CUDA_IPC_ENABLED) ||
           (rreq->mrail.protocol == MV2_RNDV_PROTOCOL_CUDAIPC)
#endif
       ))
    { 
       mpi_errno = MPIDI_CH3_Prepare_rndv_cts_cuda(vc, cts_pkt, rreq);
    } else 
#endif
    {
        mpi_errno = MPIDI_CH3_Prepare_rndv_cts(vc, cts_pkt, rreq);
    }

    if (mpi_errno != MPI_SUCCESS) {
	mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
					 FCNAME, __LINE__,
					 MPI_ERR_OTHER, "**ch3|ctspkt", 0);
	goto fn_exit;
    }

    mpi_errno = MPIDI_CH3_iStartMsg(vc, cts_pkt, sizeof(*cts_pkt), &cts_req);
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno != MPI_SUCCESS) {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER, "**ch3|ctspkt", 0);
        goto fn_exit;
    }
    /* --END ERROR HANDLING-- */
    if (cts_req != NULL) {
        MPID_Request_release(cts_req);
    }

  fn_exit:
#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_ISTARTRNDVTRANSFER);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rndv_transfer
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rndv_transfer(MPIDI_VC_t * vc,
        MPID_Request * sreq,
        MPID_Request * rreq,
        MPIDI_CH3_Pkt_rndv_clr_to_send_t * cts_pkt,
        MPIDI_CH3_Pkt_rndv_req_to_send_t * rts_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_CH3I_MRAILI_Rndv_info_t *rndv;        /* contains remote info */
    MPID_Request * req;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_TRANSFER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_TRANSFER);


#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif
        
    DEBUG_PRINT("Get rndv reply, add to list\n");

    /* This function can adapt to either read
     * or write based on the value of sreq or
     * rreq. */
    if(sreq) {
        req = sreq;
    } else {
        req = rreq;
    }

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && sreq) {
        /* Local data is on host, but remote side replied indicating that its
         * buffers reside on device. If this is an intra-node transfer, we
         * require cuda_transfer_mode to be set so that we can choose not to do
         * CMA/LiMIC-based transfers if source or target is on device */
        if ((sreq->mrail.cuda_transfer_mode == NONE) && (vc->smp.local_nodes >= 0)) {
            if (cts_pkt->rndv.cuda_transfer_mode != NONE) {
                req->mrail.cuda_transfer_mode = HOST_TO_DEVICE;
            } 
        }

        /* Local data is on device, but remote side replied indicating that its
         * buffers reside on host or is unable to transfer to device directly.
         * If this is an intra-node transfer, we require cuda_transfer_mode to
         * be set so that we can choose not to do CMA/LiMIC-based transfers if
         * source or target is on device */
        if ((sreq->mrail.cuda_transfer_mode == DEVICE_TO_DEVICE) && (vc->smp.local_nodes >= 0)) {
            req->mrail.cuda_transfer_mode = DEVICE_TO_HOST;
        }
    }
#endif

    switch (req->mrail.protocol)
    {
    case MV2_RNDV_PROTOCOL_RPUT:
            rndv = (cts_pkt == NULL) ? NULL : &cts_pkt->rndv;
            sreq->mrail.partner_id = cts_pkt->receiver_req_id;
            MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(sreq, rndv);
        break;
    case MV2_RNDV_PROTOCOL_R3:
            rndv = (cts_pkt == NULL) ? NULL : &cts_pkt->rndv;
            sreq->mrail.partner_id = cts_pkt->receiver_req_id;
            MPIU_Assert(rndv->protocol == MV2_RNDV_PROTOCOL_R3);
        break;
    case MV2_RNDV_PROTOCOL_RGET:
            rndv = (rts_pkt == NULL) ? ((cts_pkt == NULL) ? NULL : &cts_pkt->rndv) : &rts_pkt->rndv;
            MPIU_Assert (rndv != NULL);
            if (sreq != NULL && cts_pkt != NULL) sreq->mrail.partner_id = cts_pkt->receiver_req_id;
            MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(req, rndv);
        break;
#ifdef _ENABLE_UD_
    case MV2_RNDV_PROTOCOL_UD_ZCOPY:
            rndv = (cts_pkt == NULL) ? NULL : &cts_pkt->rndv;
            sreq->mrail.partner_id = cts_pkt->receiver_req_id;
            sreq->mrail.rndv_buf_off = 0;
            sreq->mrail.nearly_complete = 0;
            MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(sreq, rndv);
        break;
#endif
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    case MV2_RNDV_PROTOCOL_CUDAIPC:
            if (cts_pkt->rndv.protocol != MV2_RNDV_PROTOCOL_CUDAIPC) {
                sreq->mrail.protocol = cts_pkt->rndv.protocol;
                MPIU_Assert(sreq->mrail.protocol == MV2_RNDV_PROTOCOL_R3);
            }
            rndv = (cts_pkt == NULL) ? NULL : &cts_pkt->rndv;
            sreq->mrail.partner_id = cts_pkt->receiver_req_id;
        break;
#endif
    default:
            mpi_errno = MPIR_Err_create_code(
                0,
                MPIR_ERR_FATAL,
                FCNAME,
                __LINE__,
                MPI_ERR_OTHER,
                "**fail",
                "**fail %s",
                "unknown protocol");
#if defined(CKPT)
            MPIDI_CH3I_CR_unlock();
#endif /* defined(CKPT) */
        return mpi_errno;
    }

    RENDEZVOUS_IN_PROGRESS(vc, req);
    /*
     * this is where all rendezvous transfers are started,
     * so it is the only place we need to set this kludgy
     * field
     */

    req->mrail.nearly_complete = 0;

    PUSH_FLOWLIST(vc);

#if defined(CKPT)
    MPIDI_CH3I_CR_unlock();
#endif /* defined(CKPT) */

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_TRANSFER);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_push
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvous_push(MPIDI_VC_t * vc, MPID_Request * sreq)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_PUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_PUSH);

    if (SMP_INIT
        && vc->smp.local_nodes >= 0
#if defined(_ENABLE_CUDA_)
        && sreq->mrail.protocol != MV2_RNDV_PROTOCOL_CUDAIPC
#endif
        && vc->smp.local_nodes != g_smpi.my_local_id)
    {
        MPIDI_CH3_SMP_Rendezvous_push(vc, sreq);
        return MPI_SUCCESS;
    }

    switch (sreq->mrail.protocol)
    {
    case MV2_RNDV_PROTOCOL_RPUT:
            MPIDI_CH3I_MRAILI_Rendezvous_rput_push(vc, sreq);
        break;
    case MV2_RNDV_PROTOCOL_RGET:
            MPIDI_CH3I_MRAILI_Rendezvous_rget_push(vc, sreq);
        break;
#ifdef _ENABLE_UD_
    case MV2_RNDV_PROTOCOL_UD_ZCOPY:
            MPIDI_CH3I_MRAILI_Rendezvous_zcopy_push(vc, sreq,
                        &(mv2_MPIDI_CH3I_RDMA_Process.zcopy_info));
        break;
#endif
#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    case MV2_RNDV_PROTOCOL_CUDAIPC:
            MPIDI_CH3_CUDAIPC_Rendezvous_push(vc, sreq);
        break;
#endif
    default:
            MPIDI_CH3_Rendezvous_r3_push(vc, sreq);
        break;
    }
    
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_PUSH);
    return MPI_SUCCESS;
}

#if defined(_SMP_CMA_)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_CMA_Rendezvous_push
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3_CMA_Rendezvous_push(MPIDI_VC_t * vc,
                                                MPID_Request * sreq)
{
    int mpi_errno = MPI_SUCCESS;
    int complete = 0, rail = -1;
    void *src, *dst;
    ssize_t len, offset  ATTRIBUTE((unused));
    int type  ATTRIBUTE((unused)) = MPIDI_Request_get_type(sreq);

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_CMA_RNDV_PUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_CMA_RNDV_PUSH);

    PRINT_DEBUG(DEBUG_RNDV_verbose>2,
            "req type: %d, protocol: %d, partner_id: %08x, iov count: %d, offset %lu, len: %lu\n", 
            type, sreq->mrail.protocol, sreq->mrail.partner_id, sreq->dev.iov_count,
            sreq->dev.iov_offset, sreq->dev.iov[0].MPL_IOV_LEN);

    /* Non-contig sends are handled using the R3 protocol */
    MPIU_Assert(sreq->dev.iov_count == 1);
    MPIU_Assert(sreq->mrail.protocol == MV2_RNDV_PROTOCOL_RPUT ||
                sreq->mrail.protocol == MV2_RNDV_PROTOCOL_RGET );

    if (sreq->mrail.protocol == MV2_RNDV_PROTOCOL_RPUT) {
        src = sreq->dev.iov[0].MPL_IOV_BUF;
        len = sreq->dev.iov[0].MPL_IOV_LEN;
        dst = sreq->mrail.remote_addr;

        mpi_errno = MPIDI_CH3I_SMP_do_cma_put(vc, src, dst, len);
        if (MPI_SUCCESS != mpi_errno) {
            vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
            MPID_Request_complete(sreq);
            return mpi_errno;
        }

        MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
        MRAILI_RDMA_Put_finish(vc, sreq, rail);
        sreq->mrail.nearly_complete = 1;
    } else if (sreq->mrail.protocol == MV2_RNDV_PROTOCOL_RGET) {
        dst = sreq->dev.iov[0].MPL_IOV_BUF;
        len = sreq->dev.iov[0].MPL_IOV_LEN;
        src = sreq->mrail.remote_addr;

        mpi_errno = MPIDI_CH3I_SMP_do_cma_get(vc, src, dst, len);
        if (MPI_SUCCESS != mpi_errno) {
            vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
            sreq->status.MPI_ERROR = MPI_ERR_INTERN;
            MPID_Request_complete(sreq);
            return mpi_errno;
        }

        sreq->mrail.nearly_complete = 1;
        sreq->mrail.num_rdma_read_completions = 1;
        MRAILI_RDMA_Get_finish(vc, sreq, rail);
    } else {
         mpi_errno =
             MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL, FCNAME,
                              __LINE__, MPI_ERR_OTHER, "**notimpl", 0);
        return mpi_errno;
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_CMA_RNDV_PUSH);
    return mpi_errno;
}
#endif

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_SMP_Rendezvous_push
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3_SMP_Rendezvous_push(MPIDI_VC_t * vc,
                                                MPID_Request * sreq)
{
    int nb;
    int complete = 0;
    int seqnum;
    int mpi_errno;
    MPIDI_CH3_Pkt_rndv_r3_data_t pkt_head;
    MPID_Request * send_req;
#if defined (_ENABLE_CUDA_)
    int iov_isdev = 0;
#endif

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SMP_RNDV_PUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SMP_RNDV_PUSH);

    if (sreq->mrail.protocol != MV2_RNDV_PROTOCOL_R3
#if defined (_ENABLE_CUDA_)
            && sreq->mrail.cuda_transfer_mode == NONE
#endif
       ) {
#if defined(_SMP_CMA_)
        if(g_smp_use_cma) {
            return MPIDI_CH3_CMA_Rendezvous_push(vc, sreq);
        }
#endif
    }

    MPIDI_Pkt_init(&pkt_head, MPIDI_CH3_PKT_RNDV_R3_DATA);
    pkt_head.receiver_req_id = sreq->mrail.partner_id;
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(&pkt_head, seqnum);
    MPIDI_Request_set_seqnum(sreq, seqnum);

#if defined(_SMP_CMA_) || defined(_SMP_LIMIC_)
#if defined(_SMP_CMA_)
    int use_cma = g_smp_use_cma; 
#else
    int use_cma = 0;
#endif
#if defined(_SMP_LIMIC_)
    int use_limic = g_smp_use_limic2;
#endif 

    /* Use cma for contiguous data 
     * Use shared memory for non-contiguous data
     */
    pkt_head.csend_req_id = NULL;
    pkt_head.send_req_id = NULL;

#if defined(_SMP_CMA_)
    if (use_cma && (!g_smp_max_switch || 
            (g_smp_max_switch && sreq->dev.iov[0].MPL_IOV_LEN < s_smp_cma_max_size))
            && sreq->dev.OnDataAvail != MPIDI_CH3_ReqHandler_SendReloadIOV
            && sreq->dev.iov_count == 1
#if defined(_ENABLE_CUDA_)
            && (sreq->mrail.cuda_transfer_mode == NONE)
#endif
       )
    {
        pkt_head.csend_req_id = sreq;
        pkt_head.send_req_id = NULL;
    }
#endif

#if defined(_SMP_LIMIC_)
    if (use_limic && (!g_smp_max_switch ||
            (g_smp_max_switch && sreq->dev.iov[0].MPL_IOV_LEN < s_smp_limic2_max_size))
            && sreq->dev.OnDataAvail != MPIDI_CH3_ReqHandler_SendReloadIOV
            && sreq->dev.iov_count == 1
#if defined(_ENABLE_CUDA_)
            && (sreq->mrail.cuda_transfer_mode == NONE)
#endif
       )
    {
        pkt_head.csend_req_id = NULL;
        pkt_head.send_req_id = sreq;
    }
#endif

#endif

    PRINT_DEBUG(DEBUG_RNDV_verbose>1,
            "Sending R3 Data to %d, sreq: %p, partner: %08x, niov: %d, cma: %p, limic: %p\n",
            vc->pg_rank, sreq, sreq->mrail.partner_id, sreq->dev.iov_count,
            pkt_head.csend_req_id, pkt_head.send_req_id);

    mpi_errno = MPIDI_CH3_iStartMsg(vc, &pkt_head,
                                    sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t),
                                    &send_req);

    if (mpi_errno != MPI_SUCCESS) {
         MPIU_Object_set_ref(sreq, 0);
         MPIDI_CH3_Request_destroy(sreq);
         sreq = NULL;
         mpi_errno =
             MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL, FCNAME,
                              __LINE__, MPI_ERR_OTHER, "**ch3|rtspkt",
                              0);
        return mpi_errno;
    }
    /* --END ERROR HANDLING-- */
    if (send_req != NULL) {
        DEBUG_PRINT("r3 packet not sent \n");
        MPID_Request_release(send_req);
    }

#if defined(_SMP_LIMIC_) || defined(_SMP_CMA_)
      if (pkt_head.send_req_id || pkt_head.csend_req_id) {
        sreq->mrail.nearly_complete = 1;
        return MPI_SUCCESS;
    }
#endif

    PRINT_DEBUG(DEBUG_RNDV_verbose>1, "R3 Data sent to %d, sreq: %p\n", vc->pg_rank, sreq);
    if (MPIDI_CH3I_SMP_SendQ_empty(vc)) {
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda && s_smp_cuda_pipeline) {
            iov_isdev = is_device_buffer((void *) sreq->dev.iov[sreq->dev.iov_offset].MPL_IOV_BUF);
        }
#endif

        vc->smp.send_current_pkt_type = SMP_RNDV_MSG;
        for (;;) {
            PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                    "sreq: %p, iov count: %d, offset %lu, len[0]: %lu\n",
                    sreq, sreq->dev.iov_count, sreq->dev.iov_offset,
                    sreq->dev.iov[0].MPL_IOV_LEN);

            if (vc->smp.send_current_pkt_type == SMP_RNDV_MSG) {
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) { 
                    mpi_errno = MPIDI_CH3I_SMP_writev_rndv_data_cuda(vc,
                          sreq,
                          &sreq->dev.iov[sreq->dev.iov_offset],
                          sreq->dev.iov_count - sreq->dev.iov_offset,
                          &nb,
                          0/*is_cont*/);
                } else
#endif
                {
                    mpi_errno = MPIDI_CH3I_SMP_writev_rndv_data(vc, 
                                    sreq,
                                    &sreq->dev.iov[sreq->dev.iov_offset], 
                                    sreq->dev.iov_count - sreq->dev.iov_offset,
                                    &nb);
                }
            } else {
                MPIU_Assert(vc->smp.send_current_pkt_type == SMP_RNDV_MSG_CONT);
#if defined(_ENABLE_CUDA_)
                if (iov_isdev) {
                    mpi_errno = MPIDI_CH3I_SMP_writev_rndv_data_cuda(vc,
                          sreq,
                          &sreq->dev.iov[sreq->dev.iov_offset],
                          sreq->dev.iov_count - sreq->dev.iov_offset,
                          &nb,
                          1/*is_cont*/);
                } else
#endif
                {
                    MPIDI_CH3I_SMP_writev_rndv_data_cont(vc,
                                    sreq,
                                    &sreq->dev.iov[sreq->dev.iov_offset],
                                    sreq->dev.iov_count - sreq->dev.iov_offset,
                                    &nb);
                }
            }

            if (MPI_SUCCESS != mpi_errno) {
                vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
                sreq->status.MPI_ERROR = MPI_ERR_INTERN;
                MPID_Request_complete(sreq);
                return mpi_errno;
            }

            if (nb > 0) {
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "Wrote R3 data, dest: %d, sreq: %p, bytes: %d\n",
                        vc->pg_rank, sreq, nb);
                if (MPIDI_CH3I_Request_adjust_iov(sreq, nb)) {
#if defined(_ENABLE_CUDA_)
                    if (iov_isdev) {
                        /* the request is completed when the device to 
                         * host copy is complete*/
                        sreq->mrail.nearly_complete = 1;
                        break;
                    } 
#endif

                    MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
                    if (complete) {
                        sreq->mrail.nearly_complete = 1;
                        break;
                    } else {
                        vc->smp.send_current_pkt_type = SMP_RNDV_MSG_CONT;
                    }
                } else {
                    sreq->ch.reqtype = REQUEST_RNDV_R3_DATA;
                    MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
                    vc->smp.send_active = sreq;
                    sreq->mrail.nearly_complete = 1;
                    vc->smp.send_current_pkt_type = SMP_RNDV_MSG_CONT;
                    MV2_INC_NUM_POSTED_SEND();
                    PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                            "Enqueue next R3 data, dest: %d, sreq: %p\n",
                            vc->pg_rank, sreq);
                    break;
                }
            } else {
                sreq->ch.reqtype = REQUEST_RNDV_R3_DATA;
                MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, sreq);
                vc->smp.send_active = sreq;
                sreq->mrail.nearly_complete = 1;
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "Enqueue R3 data, dest: %d, sreq: %p\n",
                        vc->pg_rank, sreq);
                break;
            }
        }
    } else {
        sreq->ch.reqtype = REQUEST_RNDV_R3_DATA;
        MPIDI_CH3I_SMP_SendQ_enqueue(vc, sreq);
        sreq->mrail.nearly_complete = 1;
        PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                "Enqueue R3 data, dest: %d, sreq: %p\n",
                vc->pg_rank, sreq);
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SMP_RNDV_PUSH);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_r3_push
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3_Rendezvous_r3_push(MPIDI_VC_t * vc, MPID_Request * sreq)
{
    vbuf *buf;
    MPL_IOV iov[MPL_IOV_LIMIT + 1] = {0};
    int n_iov;
    int msg_buffered = 0;
    int nb = 0;
    int complete = 0;
    int seqnum;
    int finished = 0;
    int mpi_errno;
    int wait_for_rndv_r3_ack = 0;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_R3_PUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_R3_PUSH);

    MPIDI_CH3_Pkt_rndv_r3_data_t pkt_head;

    MPIDI_Pkt_init(&pkt_head, MPIDI_CH3_PKT_RNDV_R3_DATA);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
    iov[0].MPL_IOV_BUF = (void*) &pkt_head;
    pkt_head.receiver_req_id = sreq->mrail.partner_id;

    do {
        do {
	    /* stop sending more R3 data to avoid SRQ flooding at receiver */
            if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
                if (vc->ch.pending_r3_data >= rdma_max_r3_pending_data) {
                    wait_for_rndv_r3_ack = 1;
                    break;
                }
            }		

            MPIDI_VC_FAI_send_seqnum(vc, seqnum);
            MPIDI_Pkt_set_seqnum(&pkt_head, seqnum);
            MPIDI_Request_set_seqnum(sreq, seqnum);

            MPIU_Memcpy((void *) &iov[1],
                   &sreq->dev.iov[sreq->dev.iov_offset],
                   (sreq->dev.iov_count -
                    sreq->dev.iov_offset) * sizeof(MPL_IOV));
            n_iov = sreq->dev.iov_count - sreq->dev.iov_offset + 1;

            DEBUG_PRINT("iov count (sreq): %d, offset %d, len[1] %d\n",
                        sreq->dev.iov_count, sreq->dev.iov_offset,
                        sreq->dev.iov[0].MPL_IOV_LEN);

            {
                int i = 0;
                size_t  total_len = 0;
                for (i = 0; i < n_iov; i++) {
                    total_len += (iov[i].MPL_IOV_LEN);
                }

                mpi_errno =
                    MPIDI_CH3I_MRAILI_Eager_send(vc, iov, n_iov, 
                        total_len, &nb, &buf);
            }

            DEBUG_PRINT("[istartmsgv] mpierr %d, nb %d\n", mpi_errno,
                    nb);

            if (MPI_SUCCESS != mpi_errno
                && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
                vc->ch.state = MPIDI_CH3I_VC_STATE_FAILED;
                sreq->status.MPI_ERROR = MPI_ERR_INTERN;
                MPID_Request_complete(sreq);
                return;
            } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                msg_buffered = 1;
            }

            nb -= sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
            finished = MPIDI_CH3I_Request_adjust_iov(sreq, nb);
            DEBUG_PRINT("ajust iov finish: %d\n", finished);
            vc->ch.pending_r3_data += nb;
        } while (!finished/* && !msg_buffered*/);

        if (wait_for_rndv_r3_ack) {
            break;
        }
        if (finished && sreq->dev.OnDataAvail ==
			MPIDI_CH3_ReqHandler_SendReloadIOV) {
            MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
            nb = 0;
            complete = 0;
        } else if (finished) {
            complete = 1;
        }
    } while (/* 1 != msg_buffered && */0 == complete);

    DEBUG_PRINT("exit loop with complete %d, msg_buffered %d wiat %d pending data:%d \n", complete,
                msg_buffered, wait_for_rndv_r3_ack, vc->ch.pending_r3_data);

    if (wait_for_rndv_r3_ack) { //|| 0 == complete && 1 == msg_buffered) {
        sreq->mrail.nearly_complete = 0;
    } else if (1 == msg_buffered) {
        buf->sreq = (void *) sreq;
        sreq->mrail.nearly_complete = 1;
    } else {
        buf->sreq = NULL;
        MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
        sreq->mrail.nearly_complete = 1;
    }

    if (sreq->mrail.nearly_complete) {
        DEBUG_PRINT("R3 PUSH completed\n");
    } else {
        DEBUG_PRINT("Send Max R3 Pending Data. waiting for ACK\n");
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_R3_PUSH);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Process_rndv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3I_MRAILI_Process_rndv()
{
    MPID_Request *sreq;
    MPIDI_VC_t *pending_flowlist = NULL, *temp_vc = NULL;
    int need_vc_enqueue = 0;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PROCESS_RNDV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PROCESS_RNDV);
    while (flowlist) {
        /* Push on the the first ongoing receive with
         * MPIDI_CH3_Rendezvous_push. If the receive
         * finishes, it will advance the shandle_head
         * pointer on the connection.
         *
         * xxx the side effect of MPIDI_CH3_Rendezvous_push is
         * bad practice. Find a way to do this so the logic
         * is obvious.
         */
#ifdef CKPT
        /*If vc is suspended, ignore this flow and move on*/
        if (flowlist->ch.state != MPIDI_CH3I_VC_STATE_IDLE) {
            POP_FLOWLIST();/*VC will be push back when state becomes MPIDI_CH3I_VC_STATE_IDLE*/
            continue;
        }
#endif

#ifdef _ENABLE_XRC_
        if (USE_XRC && VC_XSTS_ISUNSET (flowlist, XF_SMP_VC | 
                    XF_DPM_INI | XF_SEND_IDLE | XF_UD_CONNECTED)) {
            PRINT_DEBUG(DEBUG_XRC_verbose>0, "No conn for RNDV! 0x%08x\n", flowlist->ch.xrc_flags);
            MPIDI_CH3I_CM_Connect(flowlist);
            POP_FLOWLIST();
            continue;
        }
#endif
        sreq = flowlist->mrail.sreq_head;
        while (sreq != NULL) {
#ifdef CKPT
            if (flowlist->ch.rput_stop
             && MV2_RNDV_PROTOCOL_RPUT == sreq->mrail.protocol) {
                break; /*VC will be push back when the rput_stop becomes 0*/
            }
#endif
            MPIDI_CH3_Rendezvous_push(flowlist, sreq);
            PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                    "after rndv push, sreq: %p, nearly_complete: %d, local_complete: %d, remote_complete: %d\n",
                    sreq, sreq->mrail.nearly_complete, sreq->mrail.local_complete, sreq->mrail.remote_complete);
            if (1 != sreq->mrail.nearly_complete) {
                break;
            }
            PRINT_DEBUG(DEBUG_RNDV_verbose, "sreq: %p, protocol: %d, "
                    "nearly_complete: %d, local_complete: %d, remote_complete: %d\n", 
                    sreq, sreq->mrail.protocol, sreq->mrail.nearly_complete,
                    sreq->mrail.local_complete, sreq->mrail.remote_complete);
            RENDEZVOUS_DONE(flowlist);
            sreq = flowlist->mrail.sreq_head;
        }

        temp_vc = flowlist;
        need_vc_enqueue = 0;
        if (sreq && 1 != sreq->mrail.nearly_complete) {
            need_vc_enqueue = 1;
        }

#if defined (_ENABLE_CUDA_) && defined (HAVE_CUDA_IPC)
        if(rdma_enable_cuda) {
            sreq = flowlist->mrail.cudaipc_sreq_head;
            while (sreq != NULL) {
                MPIDI_CH3_CUDAIPC_Rendezvous_recv(flowlist, sreq);
                if (1 != sreq->mrail.nearly_complete) {
                    if (!need_vc_enqueue) {
                        need_vc_enqueue = 1;
                    }
                    break;
                }
                CUDAIPC_RECV_DONE(flowlist);
                sreq = flowlist->mrail.cudaipc_sreq_head;
            }
        }
#endif 

        /* now move on to the next connection */
        POP_FLOWLIST();

        if (need_vc_enqueue) {    
            ADD_PENDING_FLOWLIST(temp_vc, pending_flowlist);
        }
    }

    while(pending_flowlist) {
        /* push pending vc to the flowlist */
        REMOVE_PENDING_FLOWLIST(temp_vc, pending_flowlist)
        PUSH_FLOWLIST(temp_vc);
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PROCESS_RNDV);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvouz_r3_recv_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvouz_r3_recv_data(MPIDI_VC_t * vc, vbuf * buffer)
{
    int mpi_errno = MPI_SUCCESS;
    int skipsize = sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
    int nb, complete;
    MPID_Request *rreq;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_R3_RCV_DATA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_R3_RCV_DATA);
    MPID_Request_get_ptr(((MPIDI_CH3_Pkt_rndv_r3_data_t *) (buffer->
                                                            pheader))->
                         receiver_req_id, rreq);

    if (!(MV2_RNDV_PROTOCOL_R3 == rreq->mrail.protocol ||
          MV2_RNDV_PROTOCOL_RPUT == rreq->mrail.protocol)) {
        int rank;
        UPMI_GET_RANK(&rank);

        DEBUG_PRINT( "[rank %d]get wrong req protocol, req %08x, protocol %d\n", rank,
            rreq, rreq->mrail.protocol);
        MPIU_Assert(MV2_RNDV_PROTOCOL_R3 == rreq->mrail.protocol ||
               MV2_RNDV_PROTOCOL_RPUT == rreq->mrail.protocol);
    }

    rreq->mrail.protocol = MV2_RNDV_PROTOCOL_R3;

    mpi_errno = MPIDI_CH3I_MRAIL_Fill_Request(rreq, buffer, skipsize, &nb);
    if (mpi_errno != MPI_SUCCESS)
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno,
            MPIR_ERR_FATAL,
            FCNAME,
            __LINE__,
            MPI_ERR_OTHER,
            "**fail",
            0);
        goto fn_exit;
    }

    vc->ch.received_r3_data += nb;
    skipsize += nb;
    DEBUG_PRINT("[recv r3: handle read] filled request nb is %d\n", nb);

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
            mpi_errno =
                MPIDI_CH3I_MRAIL_Fill_Request(rreq, buffer, skipsize, &nb);
            
            if (mpi_errno != MPI_SUCCESS)
            {
                mpi_errno = MPIR_Err_create_code(
                    mpi_errno,
                    MPIR_ERR_FATAL,
                    FCNAME,
                    __LINE__,
                    MPI_ERR_OTHER,
                    "**fail",
                    0);
                goto fn_exit;
            }
            vc->ch.received_r3_data += nb;
            if (!MPIDI_CH3I_Request_adjust_iov(rreq, nb)) {
                goto fn_exit;
            }
            skipsize += nb;

            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
            DEBUG_PRINT
                ("[recv: handle read] adjust req fine, complete %d\n",
                 complete);
            if (mpi_errno != MPI_SUCCESS) {
                mpi_errno =
                    MPIR_Err_create_code(mpi_errno,
                                         MPIR_ERR_RECOVERABLE, FCNAME,
                                         __LINE__, MPI_ERR_OTHER, "**fail",
                                         0);
                goto fn_exit;
            }
        }
        if (TRUE == complete) {
            rreq->mrail.protocol = MV2_RNDV_PROTOCOL_RENDEZVOUS_UNSPECIFIED;
        }
    }
  fn_exit:
    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        if ( vc->ch.received_r3_data >= rdma_max_r3_pending_data) {
            DEBUG_PRINT("recved data: %d send ack\n", vc->ch.received_r3_data );
            MPIDI_CH3I_MRAILI_Rendezvous_r3_ack_send(vc);
        }
    }

    DEBUG_PRINT("Successfully return from r3 recv\n");
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_R3_RCV_DATA);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvouz_r3_ack_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_CH3_Rendezvouz_r3_ack_recv(MPIDI_VC_t * vc, 
				MPIDI_CH3_Pkt_rndv_r3_ack_t *r3ack_pkt)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_R3_ACK_RECV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_R3_ACK_RECV);

    DEBUG_PRINT("Received R3 Ack %d\n", r3ack_pkt->ack_data);
    vc->ch.pending_r3_data -= r3ack_pkt->ack_data;
    MPIU_Assert(vc->ch.pending_r3_data == 0);
    PUSH_FLOWLIST(vc);
    
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_R3_ACK_RECV);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_rget_send_finish
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvous_rget_send_finish(MPIDI_VC_t * vc,
                                     MPIDI_CH3_Pkt_rget_finish_t *rget_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    int complete;
    MPID_Request *sreq;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_RGET_SEND_FINISH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_RGET_SEND_FINISH);

    MPID_Request_get_ptr(rget_pkt->sender_req_id, sreq);
    PRINT_DEBUG(DEBUG_RNDV_verbose,
            "Received RGET finish, sreq: %p, protocol: %d, local: %d, remote: %d\n",
            sreq, sreq->mrail.protocol, sreq->mrail.local_complete, sreq->mrail.remote_complete);

#if defined (_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
    cudaError_t cudaerr = cudaSuccess;

    if (rdma_enable_cuda && sreq->mrail.ipc_cuda_event) {
        cudaerr = cudaStreamWaitEvent(0, sreq->mrail.ipc_cuda_event->event, 0);
        if (cudaerr != cudaSuccess) {
            ibv_error_abort(IBV_RETURN_ERR,"cudaStreamWaitEvent failed\n");
        }
    
        if (sreq->mrail.ipc_cuda_event) {
            release_cudaipc_event(sreq->mrail.ipc_cuda_event);
        }
        sreq->mrail.ipc_cuda_event = NULL;
    }
#endif

    sreq->mrail.remote_complete = UINT32_MAX;
    if (!MPIDI_CH3I_MRAIL_Finish_request(sreq)) {
        return MPI_SUCCESS;
    }

    MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(sreq);

#if 0
    if(mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
            ((req->mrail.rndv_buf_sz > rdma_large_msg_rail_sharing_threshold))) {

        /* Adjust the weights of different paths according to the
         * timings obtained for the stripes */

        adjust_weights(v->vc, req->mrail.stripe_start_time,
                req->mrail.stripe_finish_time, req->mrail.initial_weight);
    }
#endif

    MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);

    if (complete != TRUE)
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno,
            MPIR_ERR_FATAL,
            FCNAME,
            __LINE__,
            MPI_ERR_OTHER,
            "**fail",
            0);
        goto fn_exit;
    }

#if defined(CKPT)
    MPIDI_CH3I_CR_req_dequeue(sreq);
#endif /* defined(CKPT) */

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_RGET_SEND_FINISH);
    return mpi_errno;

}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_rget_recv_finish
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvous_rget_recv_finish(MPIDI_VC_t * vc,
                                     MPID_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    int complete;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RGET_RECV_FINISH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RGET_RECV_FINISH);

    if (!MPIDI_CH3I_MRAIL_Finish_request(rreq))
    {
        return MPI_SUCCESS;
    }

    if (rreq->mrail.rndv_buf_alloc == 1)
    {
        /* If we are using datatype, then need to unpack data from tmpbuf */
        int iter = 0;
        int copied = 0;
        uintptr_t buf = (uintptr_t) rreq->mrail.rndv_buf;

        for (; iter < rreq->dev.iov_count; ++iter)
        {
          MPIU_Memcpy(rreq->dev.iov[iter].MPL_IOV_BUF,
                   (void *) buf, rreq->dev.iov[iter].MPL_IOV_LEN);
            buf += rreq->dev.iov[iter].MPL_IOV_LEN;
            copied += rreq->dev.iov[iter].MPL_IOV_LEN;
        }

        MPIDI_CH3I_Request_adjust_iov(rreq, copied);

        while (rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_UnpackSRBufReloadIOV
            || rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_ReloadIOV)
        {
            /* XXX: dev.ca should only be CA_COMPLETE? */
            /* end of XXX */
            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);

            if (mpi_errno != MPI_SUCCESS || complete == TRUE)
            {
                mpi_errno = MPIR_Err_create_code(
                    mpi_errno,
                    MPIR_ERR_FATAL,
                    FCNAME,
                    __LINE__,
                    MPI_ERR_OTHER,
                    "**fail",
                    0);
                goto fn_exit;
            }

            copied = 0;

            for (iter = 0; iter < rreq->dev.iov_count; ++iter)
            {
              MPIU_Memcpy(rreq->dev.iov[iter].MPL_IOV_BUF,
                       (void *) buf, rreq->dev.iov[iter].MPL_IOV_LEN);
                buf += rreq->dev.iov[iter].MPL_IOV_LEN;
                copied += rreq->dev.iov[iter].MPL_IOV_LEN;
            }

            MPIDI_CH3I_Request_adjust_iov(rreq, copied);
        }
    }
    else
    {
        rreq->mrail.rndv_buf = NULL;
    }

#if defined(CKPT)
    MPIDI_CH3I_CR_req_dequeue(rreq);
#endif /* defined(CKPT) */

    MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(rreq);

    mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);

    if (mpi_errno != MPI_SUCCESS)
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno,
                                 MPIR_ERR_RECOVERABLE, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**fail", 0);
    }

    if (complete)
    {
        vc->ch.recv_active = NULL;
    }
    else
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER, "**fail", 0);
        goto fn_exit;
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RGET_RECV_FINISH);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_unpack_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvous_unpack_data(MPIDI_VC_t *vc, MPID_Request *rreq) 
{
        /* If we are using datatype, then need to unpack data from tmpbuf */
        int iter = 0;
        MPIDI_msg_sz_t copied = 0;
        int mpi_errno = MPI_SUCCESS;
        int complete;
        uintptr_t buf = (uintptr_t) rreq->mrail.rndv_buf;
    
        MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_UNPACK_DATA);
        MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_UNPACK_DATA);

        for (iter=0; iter < rreq->dev.iov_count; ++iter)
        {
          MPIU_Memcpy(rreq->dev.iov[iter].MPL_IOV_BUF,
                   (void *) buf, rreq->dev.iov[iter].MPL_IOV_LEN);
            buf += rreq->dev.iov[iter].MPL_IOV_LEN;
            copied += rreq->dev.iov[iter].MPL_IOV_LEN;
        }

        MPIDI_CH3I_Request_adjust_iov(rreq, copied);

        while (rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_UnpackSRBufReloadIOV
            || rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_ReloadIOV)
        {
            /* XXX: dev.ca should only be CA_COMPLETE? */
            /* end of XXX */
            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno != MPI_SUCCESS || complete == TRUE)
            {
                mpi_errno = MPIR_Err_create_code(
                    mpi_errno,
                    MPIR_ERR_FATAL,
                    FCNAME,
                    __LINE__,
                    MPI_ERR_OTHER,
                    "**fail",
                    0);
                goto fn_exit;
            }

            copied = 0;

            for (iter = 0; iter < rreq->dev.iov_count; ++iter)
            {
              MPIU_Memcpy(rreq->dev.iov[iter].MPL_IOV_BUF,
                       (void *) buf, rreq->dev.iov[iter].MPL_IOV_LEN);
                buf += rreq->dev.iov[iter].MPL_IOV_LEN;
                copied += rreq->dev.iov[iter].MPL_IOV_LEN;
            }

            MPIDI_CH3I_Request_adjust_iov(rreq, copied);
        }
  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_UNPACK_DATA);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Rendezvous_rput_finish
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Rendezvous_rput_finish(MPIDI_VC_t * vc,
                                     MPIDI_CH3_Pkt_rput_finish_t * rf_pkt)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *rreq;
    int complete;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_RPUT_FINISH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_RPUT_FINISH);

    MPID_Request_get_ptr(rf_pkt->receiver_req_id, rreq);
    PRINT_DEBUG(DEBUG_RNDV_verbose, "Received RPUT finish, rreq: %p, protocol: %d, local: %d, remote: %d\n",
            rreq, rreq->mrail.protocol, rreq->mrail.local_complete, rreq->mrail.remote_complete);

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && rreq->mrail.cuda_transfer_mode != NONE 
        && rreq->mrail.cuda_transfer_mode != DEVICE_TO_HOST) {
        if (MPIDI_CH3I_MRAILI_Process_cuda_finish(vc, rreq, rf_pkt) != 1) {
            goto fn_exit;
        }
    } else 
#endif
    {
        if (IS_VC_SMP(vc)) {
            rreq->mrail.remote_complete = UINT32_MAX;
        } else {
            rreq->mrail.remote_complete++;
            if (rreq->mrail.remote_complete == rdma_num_rails) {
                rreq->mrail.remote_complete = UINT32_MAX;
            }
        }
        if (!MPIDI_CH3I_MRAIL_Finish_request(rreq))
        {
            return MPI_SUCCESS;
        } 
    }

    if (rreq->mrail.rndv_buf_alloc == 1){
        MPIDI_CH3_Rendezvous_unpack_data(vc, rreq);
    } else {
        rreq->mrail.rndv_buf = NULL;
    }

#if defined(CKPT)
    MPIDI_CH3I_CR_req_dequeue(rreq);
#endif /* defined(CKPT) */

    if (rreq->mrail.remote_addr == NULL) {
        MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(rreq);
    }

    mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
    if (mpi_errno != MPI_SUCCESS)
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno,
                                 MPIR_ERR_RECOVERABLE, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**fail", 0);
    }

    PRINT_DEBUG(DEBUG_RNDV_verbose, "rreq: %p, complete: %d\n", rreq, complete);
    if (complete)
    {
        vc->ch.recv_active = NULL;
    }
    else
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                                 FCNAME, __LINE__,
                                 MPI_ERR_OTHER, "**fail", 0);
        goto fn_exit;
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_RPUT_FINISH);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Get_rndv_push
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Get_rndv_push(MPIDI_VC_t * vc,
                            MPIDI_CH3_Pkt_t *resp_pkt,
                            MPID_Request * req)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_GET_RNDV_PUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_GET_RNDV_PUSH);

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif

    if (resp_pkt->type ==  MPIDI_CH3_PKT_GET_RESP) { 
        MPIDI_CH3_Pkt_get_resp_t *get_resp_pkt = (MPIDI_CH3_Pkt_get_resp_t *) resp_pkt;

        if (MV2_RNDV_PROTOCOL_R3 == req->mrail.protocol) {
            req->mrail.partner_id = get_resp_pkt->request_handle;
            if (vc->smp.local_nodes < 0) {
 	       MPIDI_VC_revoke_seqnum_send(vc, get_resp_pkt->seqnum);
            }
            RENDEZVOUS_IN_PROGRESS(vc, req);
            req->mrail.nearly_complete = 0;
            PUSH_FLOWLIST(vc);
        } else {
            MPL_IOV iov;
            MPIDI_CH3I_MRAILI_Rndv_info_t rndv;
 
            iov.MPL_IOV_BUF = (void*) get_resp_pkt;
            iov.MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_get_resp_t);
            get_resp_pkt->protocol = MV2_RNDV_PROTOCOL_RPUT;
 
            MPIDI_CH3I_MRAIL_SET_REMOTE_RNDV_INFO(&rndv, req);

            MPIDI_CH3I_MRAILI_Get_rndv_rput(vc, req, &rndv, &iov);
 
            if (MV2_RNDV_PROTOCOL_R3 == req->mrail.protocol) {
                req->mrail.partner_id = get_resp_pkt->request_handle;
                if (vc->smp.local_nodes < 0) {
 	                MPIDI_VC_revoke_seqnum_send(vc, get_resp_pkt->seqnum);
                }
                RENDEZVOUS_IN_PROGRESS(vc, req);
                req->mrail.nearly_complete = 0;
                PUSH_FLOWLIST(vc);
            }
        }
    } else {
        MPIDI_CH3_Pkt_get_accum_resp_t *get_accum_resp_pkt = (MPIDI_CH3_Pkt_get_accum_resp_t *)resp_pkt;

        if (MV2_RNDV_PROTOCOL_R3 == req->mrail.protocol) {
            req->mrail.partner_id = get_accum_resp_pkt->request_handle;
            if (vc->smp.local_nodes < 0) {
                MPIDI_VC_revoke_seqnum_send(vc, get_accum_resp_pkt->seqnum);
            }
            RENDEZVOUS_IN_PROGRESS(vc, req);
            req->mrail.nearly_complete = 0;
            PUSH_FLOWLIST(vc);
        } else {
            MPL_IOV iov;
            MPIDI_CH3I_MRAILI_Rndv_info_t rndv;

            iov.MPL_IOV_BUF = (void*) get_accum_resp_pkt;
            iov.MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_get_accum_resp_t);
            get_accum_resp_pkt->protocol = MV2_RNDV_PROTOCOL_RPUT;

            MPIDI_CH3I_MRAIL_SET_REMOTE_RNDV_INFO(&rndv, req);
            MPIDI_CH3I_MRAILI_Get_rndv_rput(vc, req, &rndv, &iov);

            if (MV2_RNDV_PROTOCOL_R3 == req->mrail.protocol) {
                req->mrail.partner_id = get_accum_resp_pkt->request_handle;
                if (vc->smp.local_nodes < 0) {
                    MPIDI_VC_revoke_seqnum_send(vc, get_accum_resp_pkt->seqnum);
                }

                RENDEZVOUS_IN_PROGRESS(vc, req);
                req->mrail.nearly_complete = 0;
                PUSH_FLOWLIST(vc);
            }
        } 
    }

#ifdef CKPT
    MPIDI_CH3I_CR_unlock();
#endif

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_GET_RNDV_PUSH);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Get_rndv_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Get_rndv_recv(MPIDI_VC_t * vc, MPID_Request * req)
{
    int mpi_errno = MPI_SUCCESS;
    int complete;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RNDV_RECV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RNDV_RECV);

    MPIU_Assert(req->mrail.protocol == MV2_RNDV_PROTOCOL_RPUT);

#ifdef CKPT
    MPIDI_CH3I_CR_lock();
#endif

    if (req->mrail.rndv_buf_alloc == 1)
    {
        /* If we are using datatype, then need to unpack data from tmpbuf */
        int iter = 0;
        uintptr_t buf = (uintptr_t) req->mrail.rndv_buf;

        for (iter=0; iter < req->dev.iov_count; ++iter)
        {
          MPIU_Memcpy(req->dev.iov[iter].MPL_IOV_BUF,
                   (void *) buf, req->dev.iov[iter].MPL_IOV_LEN);
            buf += req->dev.iov[iter].MPL_IOV_LEN;
        }

        while (req->dev.OnDataAvail == MPIDI_CH3_ReqHandler_UnpackSRBufReloadIOV
            || req->dev.OnDataAvail == MPIDI_CH3_ReqHandler_ReloadIOV)
        {
            /* mpi_errno = MPIDI_CH3U_Request_load_recv_iov(req); */
            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, req, &complete);

            if (mpi_errno != MPI_SUCCESS)
            {
                goto fn_exit;
            }

            for (iter = 0; iter < req->dev.iov_count; ++iter)
            {
              MPIU_Memcpy(req->dev.iov[iter].MPL_IOV_BUF,
                       (void *) buf, req->dev.iov[iter].MPL_IOV_LEN);
                buf += req->dev.iov[iter].MPL_IOV_LEN;
            }
        }
    }
    else
    {
        req->mrail.rndv_buf = NULL;
    }

    MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(req);

    mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, req, &complete);

    if (mpi_errno != MPI_SUCCESS)
    {
        goto fn_exit;
    }

    MPIU_Assert(complete == TRUE);

  fn_exit:
#if defined(CKPT)
    MPIDI_CH3I_CR_unlock();
#endif /* defined(CKPT) */
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RNDV_RECV);
    return mpi_errno;
}

