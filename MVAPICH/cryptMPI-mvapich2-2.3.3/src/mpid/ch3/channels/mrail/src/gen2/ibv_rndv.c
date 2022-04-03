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
#include <sys/time.h>
#include "mpichconf.h"
#include <mpimem.h>
#include "rdma_impl.h"
#include "mpiimpl.h"
#include "ibv_impl.h"
#include "vbuf.h"
#include "dreg.h"
#include "mpiutil.h"

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

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

void get_sorted_index(MPIDI_VC_t *vc, int *b);

int MPIDI_CH3I_MRAIL_Prepare_rndv(MPIDI_VC_t * vc, MPID_Request * req)
{
    dreg_entry *reg_entry = NULL;
    PRINT_DEBUG(DEBUG_RNDV_verbose>1,
            "vc->rank: %d, req: %p, protocol: %d, recv size %ld, segsize %ld, iov count %d\n",
            vc->pg_rank, req, req->mrail.protocol, req->dev.recv_data_sz, req->dev.segment_size, req->dev.iov_count);

    if (IS_VC_SMP(vc)) {
        if (req->dev.iov_count > 1) {
            req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
            MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);
        } else if (req->mrail.protocol == MV2_RNDV_PROTOCOL_RENDEZVOUS_UNSPECIFIED) {
            req->mrail.protocol = smp_rndv_protocol;
        }
        if (req->mrail.protocol == MV2_RNDV_PROTOCOL_R3) {
            return 0;
        }
    }
#ifdef _ENABLE_UD_
    else if (rdma_enable_hybrid && (!(vc->mrail.state & MRAILI_RC_CONNECTED) || 
                    req->mrail.protocol == MV2_RNDV_PROTOCOL_UD_ZCOPY)) {
#ifdef _ENABLE_XRC_
        if (USE_XRC && !(vc->mrail.state & MRAILI_RC_CONNECTED) &&
                (req->mrail.protocol == MV2_RNDV_PROTOCOL_RPUT ||
                req->mrail.protocol == MV2_RNDV_PROTOCOL_RGET)) {
            req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
            return 0;
        }
#endif        
        req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
        if (rdma_use_ud_zcopy) {
            req->mrail.protocol = MV2_RNDV_PROTOCOL_UD_ZCOPY;
        }
    }
#endif
    else {
        req->mrail.protocol = rdma_rndv_protocol;
    }

    /* Step 1: ready for user space (user buffer or pack) */
    if (1 == req->dev.iov_count && (req->dev.OnDataAvail == NULL 
        || req->dev.OnDataAvail == req->dev.OnFinal
        || req->dev.OnDataAvail == MPIDI_CH3_ReqHandler_UnpackSRBufComplete
#if defined(_ENABLE_CUDA_)
        || req->dev.OnDataAvail == MPIDI_CH3_ReqHandler_unpack_cudabuf
#endif 
        ))
    {
        req->mrail.rndv_buf = req->dev.iov[0].MPL_IOV_BUF;
        req->mrail.rndv_buf_sz = req->dev.iov[0].MPL_IOV_LEN;
        req->mrail.rndv_buf_alloc = 0;
    } else {
        req->mrail.rndv_buf_sz = req->dev.segment_size;
        req->mrail.rndv_buf = MPIU_Malloc(req->mrail.rndv_buf_sz);

        if (req->mrail.rndv_buf == NULL) {

            /* fall back to r3 if cannot allocate tmp buf */

            DEBUG_PRINT("[rndv sent] set info: cannot allocate space\n");
            req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
            req->mrail.rndv_buf_sz = 0;
        } else {
            req->mrail.rndv_buf_alloc = 1;
        }
    }
    req->mrail.rndv_buf_off = 0;

    /* Step 1.5: If use R3 for smaller messages */
    if (req->mrail.rndv_buf_sz <= MPIDI_CH3_R3_THRESHOLD(vc)
#ifdef _ENABLE_CUDA_
        && !rdma_enable_cuda
#endif
        ) {
        PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                "Using R3, rndv_buf_sz: %ld, rdma_r3_threshold: %d\n",
                req->mrail.rndv_buf_sz, rdma_r3_threshold);
        req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
        MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);
    }
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && req->mrail.cuda_transfer_mode != NONE) {
        if ( (MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol) ||
                (MV2_RNDV_PROTOCOL_RGET == req->mrail.protocol) ) {
            return 1;
        } else {
            return 0;
        }
    }
#endif
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid && (req->mrail.rndv_buf_sz < rdma_ud_zcopy_threshold
     || req->mrail.rndv_buf_sz > (MRAIL_MAX_UD_SIZE * rdma_ud_zcopy_rq_size))) {
            /*len <= (rdma_default_ud_mtu * 4096) */
        req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
        MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);
    }
#endif

    /* Step 2: try register and decide the protocol */
    /* The additional check for intra-node peer has been added to handle 
       case when RGET is used for CUDA IPC communication when shared memory 
       is enabled. In this case registration cache is not initialized and 
       hence dreg_register leads to a hang. Better separation of thsee cases
       might be possible */
    if (
#ifdef _ENABLE_CUDA_
        (!rdma_enable_cuda || !SMP_INIT || vc->smp.local_nodes == -1) && 
#endif
        (MV2_RNDV_PROTOCOL_RPUT == req->mrail.protocol ||
            MV2_RNDV_PROTOCOL_RGET == req->mrail.protocol ||
                MV2_RNDV_PROTOCOL_UD_ZCOPY == req->mrail.protocol)) {
        if (IS_VC_SMP(vc)) {
            PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                    "SMP vc, not registering. rank: %d, buf size: %ld, addr: %p, protocol: %d\n",
                    vc->pg_rank, req->mrail.rndv_buf_sz, req->mrail.rndv_buf, req->mrail.protocol);
            req->mrail.d_entry = NULL;
        } else {
            PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                    "Remote vc, registering. rank: %d, buf size: %ld, addr: %p\n",
                    vc->pg_rank, req->mrail.rndv_buf_sz, req->mrail.rndv_buf);
            reg_entry =
                dreg_register(req->mrail.rndv_buf, req->mrail.rndv_buf_sz);
            if (NULL == reg_entry) {
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "registration failed for buf: %p, using R3\n", req->mrail.rndv_buf);
                req->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
                MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req);
                /*MRAILI_Prepost_R3(); */
            } else {
                req->mrail.d_entry = reg_entry;
                PRINT_DEBUG(DEBUG_RNDV_verbose>1,
                        "registration success for buf: %p, d_entry: %p\n",
                        req->mrail.rndv_buf, req->mrail.d_entry);
            }
        }

        req->mrail.local_complete  = 0;
        req->mrail.remote_complete = 0;
        req->mrail.d_entry = reg_entry;
        return 1;
    } else {
        req->mrail.local_complete  = 0;
        req->mrail.remote_complete = 0;
    }

    return 0;
}

int MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(MPID_Request * sreq, 
        /* contains local info */
        MPIDI_CH3I_MRAILI_Rndv_info_t *rndv)
{
    int hca_index;

    if (rndv->protocol == MV2_RNDV_PROTOCOL_R3) {
        if (sreq->mrail.d_entry != NULL) {
            dreg_unregister(sreq->mrail.d_entry);
            sreq->mrail.d_entry = NULL;
        }
        if (1 == sreq->mrail.rndv_buf_alloc
                && NULL != sreq->mrail.rndv_buf) {
            MPIU_Free(sreq->mrail.rndv_buf);
            sreq->mrail.rndv_buf_alloc = 0;
            sreq->mrail.rndv_buf = NULL;
        }
        sreq->mrail.remote_addr = NULL;
        /* Initialize this completion counter to 0
         * required for even striping */
        sreq->mrail.local_complete = 0;

        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index ++)
            sreq->mrail.rkey[hca_index] = 0;
        sreq->mrail.protocol = MV2_RNDV_PROTOCOL_R3;
    } else {
#ifdef _ENABLE_UD_
        if(rndv->protocol == MV2_RNDV_PROTOCOL_UD_ZCOPY) {
            PRINT_DEBUG(DEBUG_ZCY_verbose>0, "Received CTS.remote qpn:%d\n", rndv->rndv_qpn);
            sreq->mrail.remote_qpn = rndv->rndv_qpn;
            sreq->mrail.hca_index = rndv->hca_index;
        }
        /* TODO: Can we avoid dev.iov copy for zcopy */
#endif
#ifdef _ENABLE_CUDA_
        int i;
        if( rdma_enable_cuda && sreq->mrail.cuda_transfer_mode != NONE) {
            sreq->mrail.cts_received = 1;
            sreq->mrail.num_send_cuda_copy = 0;
	    if (rndv->cuda_transfer_mode == NONE || rndv->cuda_transfer_mode == DEVICE_TO_HOST) {
                sreq->mrail.cuda_transfer_mode = DEVICE_TO_HOST;
            }
            if (sreq->mrail.cuda_transfer_mode == DEVICE_TO_DEVICE
                || sreq->mrail.cuda_transfer_mode == HOST_TO_DEVICE) {
                for (i = 0; i < rndv->num_cuda_blocks; i++) {
                    sreq->mrail.cuda_remote_addr[i] = rndv->buffer_addr[i];
                    for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                        sreq->mrail.cuda_remote_rkey[i][hca_index] = 
                            rndv->buffer_rkey[i][hca_index];
                    }
                    sreq->mrail.num_remote_cuda_pending = rndv->num_cuda_blocks;
                    sreq->mrail.cuda_block_offset = rndv->cuda_block_offset;
                    sreq->mrail.num_remote_cuda_done = 0;
                }
            } else if (sreq->mrail.cuda_transfer_mode == DEVICE_TO_HOST) {
                sreq->mrail.num_remote_cuda_pending = 
                        ROUNDUP(sreq->mrail.rndv_buf_sz, rdma_cuda_block_size);
                sreq->mrail.cuda_block_offset = 0;
                sreq->mrail.num_remote_cuda_done = 0;
                sreq->mrail.remote_addr = rndv->buf_addr;
                for (hca_index = 0; hca_index < rdma_num_hcas; hca_index ++)
                    sreq->mrail.rkey[hca_index] = rndv->rkey[hca_index];
            } else {
                /* Code should not enter here */
		MPIU_Assert(0);
            }
        } else
#endif
        {
            sreq->mrail.remote_addr = rndv->buf_addr;
            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index ++)
            sreq->mrail.rkey[hca_index] = rndv->rkey[hca_index];
        }

        DEBUG_PRINT("[add rndv list] addr %p, key %p\n",
                sreq->mrail.remote_addr,
                sreq->mrail.rkey[0]);
        if (1 == sreq->mrail.rndv_buf_alloc) {
            int mpi_errno = MPI_SUCCESS;
            int i;
            uintptr_t buf;

            buf = (uintptr_t) sreq->mrail.rndv_buf;
            for (i = 0; i < sreq->dev.iov_count; i++) {
                MPIU_Memcpy((void *) buf, sreq->dev.iov[i].MPL_IOV_BUF,
                        sreq->dev.iov[i].MPL_IOV_LEN);
                buf += sreq->dev.iov[i].MPL_IOV_LEN;
            }

            /* TODO: Following part is a workaround to deal with 
             * datatype with large number of segments. 
             * We check if the datatype has finished 
             * loading and reload if not.
             * May be better interface with 
             * upper layer should be considered */

            while (sreq->dev.OnDataAvail == 
                    MPIDI_CH3_ReqHandler_SendReloadIOV) {
                sreq->dev.iov_count = MPL_IOV_LIMIT;
                mpi_errno =
                    MPIDI_CH3U_Request_load_send_iov(sreq,
                            sreq->dev.iov,
                            &sreq->dev.iov_count);
                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno != MPI_SUCCESS) {
                    ibv_error_abort(IBV_STATUS_ERR, "Reload iov error");
                }
                for (i = 0; i < sreq->dev.iov_count; i++) {
                   MPIU_Memcpy((void *) buf, sreq->dev.iov[i].MPL_IOV_BUF,
                            sreq->dev.iov[i].MPL_IOV_LEN);
                    buf += sreq->dev.iov[i].MPL_IOV_LEN;
                }
            }
        }
    }
    return MPI_SUCCESS;
}

void MRAILI_RDMA_Put_finish(MPIDI_VC_t * vc, 
        MPID_Request * sreq, int rail)
{
    MPIDI_CH3_Pkt_rput_finish_t rput_pkt;
    MPL_IOV iov;
    int n_iov = 1;
    int nb;
    vbuf *buf;
    MPID_Seqnum_t seqnum;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_Pkt_init(&rput_pkt, MPIDI_CH3_PKT_RPUT_FINISH);
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda) {
        rput_pkt.is_cuda = 0;
    }
#endif
    rput_pkt.receiver_req_id = sreq->mrail.partner_id;
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(&rput_pkt, seqnum); 

    PRINT_DEBUG(DEBUG_RNDV_verbose>1,
            "Sending RPUT FINISH to %d, sreq: %p, rreq: %08x, rail: %d\n",
            vc->pg_rank, sreq, sreq->mrail.partner_id, rail);

    if (IS_VC_SMP(vc)) {
        MPID_Request *new_req = NULL; 
        sreq->mrail.local_complete = UINT32_MAX;

        mpi_errno = 
            MPIDI_CH3_SMP_iStartMsg(vc, &rput_pkt, 
                    sizeof(MPIDI_CH3_Pkt_rput_finish_t), &new_req);
        if (mpi_errno != MPI_SUCCESS) { 
            ibv_error_abort(IBV_STATUS_ERR,
                    "Failed sending rput finish through SMP channel \n");
        }

        if (new_req != NULL) {
            MPID_Request_release(new_req);
        }
    } else { 
        iov.MPL_IOV_BUF = &rput_pkt;
        iov.MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_rput_finish_t);

        int rc = MPIDI_CH3I_MRAILI_rput_complete(vc, &iov, n_iov, &nb, &buf, rail);

        if (rc != 0 && rc != MPI_MRAIL_MSG_QUEUED)
        { 
            ibv_error_abort(IBV_STATUS_ERR,
                    "Cannot send rput through send/recv path");
        }

        buf->sreq = (void *) sreq;

        /* mark MPI send complete when VIA send completes */

        DEBUG_PRINT("VBUF ASSOCIATED: %p, %08x\n", buf, buf->desc.u.sr.wr_id);
    }
}

void MRAILI_RDMA_Get_finish(MPIDI_VC_t * vc, 
        MPID_Request * rreq, int rail)
{
    MPIDI_CH3_Pkt_rget_finish_t rget_pkt;
    MPL_IOV iov;
    int n_iov = 1;
    int nb;
    int mpi_errno = MPI_SUCCESS;
    MPID_Seqnum_t seqnum;

    MPIDI_Pkt_init(&rget_pkt, MPIDI_CH3_PKT_RGET_FINISH);
    rget_pkt.sender_req_id = rreq->dev.sender_req_id;
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(&rget_pkt, seqnum); 

    rreq->mrail.local_complete = UINT32_MAX;
    PRINT_DEBUG(DEBUG_RNDV_verbose>1,
            "Sending RGET FINISH to %d, rreq: %p, sreq: %08x, rail: %d\n",
            vc->pg_rank, rreq, rreq->dev.sender_req_id, rail);

    if (IS_VC_SMP(vc)) {
        MPID_Request *new_req = NULL; 

        mpi_errno = 
            MPIDI_CH3_SMP_iStartMsg(vc, &rget_pkt, 
                    sizeof(MPIDI_CH3_Pkt_rget_finish_t), &new_req);
        if (mpi_errno != MPI_SUCCESS) { 
            ibv_error_abort(IBV_STATUS_ERR,
                    "Failed sending rget finish through SMP channel \n");
        }

        if (new_req != NULL) {
            MPID_Request_release(new_req);
        }

        MPIDI_CH3_Rendezvous_rget_recv_finish(vc, rreq); 
    } else { 
        vbuf *buf;

        iov.MPL_IOV_BUF = &rget_pkt;
        iov.MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_rget_finish_t);

        mpi_errno =
            MPIDI_CH3I_MRAILI_rget_finish(vc, &iov, n_iov, &nb, &buf, rail);
        if (mpi_errno != MPI_SUCCESS && 
                mpi_errno != MPI_MRAIL_MSG_QUEUED) {
            ibv_error_abort(IBV_STATUS_ERR,
                    "Cannot send rget finish through send/recv path");
        }

        MPIDI_CH3_Rendezvous_rget_recv_finish(vc, rreq);
        DEBUG_PRINT("VBUF ASSOCIATED: %p, %08x\n", buf, buf->desc.u.sr.wr_id);
    }

}

void MPIDI_CH3I_MRAILI_Rendezvous_rget_push(MPIDI_VC_t * vc,
        MPID_Request * rreq)
{
    vbuf *v;
    int queued = 0;
    int ext_sendq_size = 0;
    int rail, disp, s_total, inc;
    int rail_index;
    MPIDI_msg_sz_t nbytes;

    int count_rail;

    int mapped[MAX_NUM_SUBRAILS];
    int actual_index[MAX_NUM_SUBRAILS];

    double time;

    if (rreq->mrail.rndv_buf_off == 0) {
        rreq->mrail.num_rdma_read_completions = 0;
        rreq->mrail.local_complete = 0;
    }

    for(rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
        if(mv2_MPIDI_CH3I_RDMA_Process.has_apm && apm_tester) {
            perform_manual_apm(vc->mrail.rails[rail_index].qp_hndl);
        }
    }


#if defined(DEBUG)
    if (rreq->mrail.rndv_buf_sz > 0) {
        MPIU_Assert(rreq->mrail.d_entry != NULL);
        MPIU_Assert(rreq->mrail.remote_addr != NULL);
    }
#endif /* defined(DEBUG) */

    /* Use the HSAM Functionality */
    if(mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
            (rreq->mrail.rndv_buf_sz > rdma_large_msg_rail_sharing_threshold)) {

      MPIU_Memset(mapped, 0, rdma_num_rails * sizeof(int));
      MPIU_Memset(actual_index, 0, rdma_num_rails * sizeof(int));

        get_sorted_index(vc, actual_index);
    
        /* Get the wall-time, internally defined function */
        get_wall_time(&time);

        /* Set the start time for the stripe and the 
         * finish time to be zero*/ 

        rreq->mrail.stripe_start_time = time;

        for(rail = 0; rail < rdma_num_rails; rail++) {

            rreq->mrail.initial_weight[rail] = 
                vc->mrail.rails[rail].s_weight;
            rreq->mrail.stripe_finish_time[rail] = 0;
        }

    }

    while (rreq->mrail.rndv_buf_off < 
            rreq->mrail.rndv_buf_sz) {
        nbytes = rreq->mrail.rndv_buf_sz - rreq->mrail.rndv_buf_off;

        if (nbytes > mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize) {
            nbytes = mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize;
        }

        DEBUG_PRINT("[buffer content]: %02x,%02x,%02x, "
                "offset %d, remote buf %p\n",
                ((char *) rreq->mrail.rndv_buf)[0],
                ((char *) rreq->mrail.rndv_buf)[1],
                ((char *) rreq->mrail.rndv_buf)[2],
                rreq->mrail.rndv_buf_off, rreq->mrail.remote_addr);
        
        if (nbytes <= rdma_large_msg_rail_sharing_threshold) {
            rail = MRAILI_Send_select_rail(vc);
            /* Get current number of pending entries */
            GET_EXT_SENDQ_SIZE(vc, rail, ext_sendq_size);
            if (ext_sendq_size >= rdma_rndv_ext_sendq_size) {
                break;
            }

            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = rreq;

            MRAILI_RDMA_Get(vc, v,
                    (char *) (rreq->mrail.rndv_buf) +
                    rreq->mrail.rndv_buf_off,
                    ((dreg_entry *)rreq->mrail.d_entry)->
                    memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                    (char *) (rreq->mrail.remote_addr) +
                    rreq->mrail.rndv_buf_off,
                    rreq->mrail.rkey[vc->mrail.rails[rail].hca_index],
                    nbytes, rail);

            rreq->mrail.num_rdma_read_completions++;

        } else if(!mv2_MPIDI_CH3I_RDMA_Process.has_hsam) {
            inc = nbytes / rdma_num_rails;
            for(rail = 0; rail < rdma_num_rails; rail++) {
                /* Get current number of pending entries */
                GET_EXT_SENDQ_SIZE(vc, rail, ext_sendq_size);
                if (ext_sendq_size >= rdma_rndv_ext_sendq_size) {
                    queued = 1;
                    break;
                }
            }
            if (queued) {
                break;
            }
            
            for(rail = 0; rail < rdma_num_rails - 1; rail++) {
                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                v->sreq = rreq;
                MRAILI_RDMA_Get(vc, v,
                        (char *) (rreq->mrail.rndv_buf) +
                        rreq->mrail.rndv_buf_off + rail * inc,
                        ((dreg_entry *)rreq->mrail.d_entry)->
                        memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                        (char *) (rreq->mrail.remote_addr) +
                        rreq->mrail.rndv_buf_off + rail * inc,
                        rreq->mrail.rkey[vc->mrail.rails[rail].hca_index], 
                        inc, rail);
                rreq->mrail.num_rdma_read_completions++;
                /* Send the finish message immediately after the data */  
            }
            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = rreq;
            MRAILI_RDMA_Get(vc, v,
                    (char *) (rreq->mrail.rndv_buf) +
                    rreq->mrail.rndv_buf_off + inc * (rdma_num_rails - 1),
                    ((dreg_entry *)rreq->mrail.d_entry)->
                    memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                    (char *) (rreq->mrail.remote_addr) +
                    rreq->mrail.rndv_buf_off + inc * (rdma_num_rails - 1),
                    rreq->mrail.rkey[vc->mrail.rails[rail].hca_index], 
                    nbytes - (rdma_num_rails - 1) * inc, rail);
            rreq->mrail.num_rdma_read_completions++;

        } else {
            rail = 0;
            count_rail = 0;
           
            s_total = 0;

            while(count_rail <( rdma_num_rails / stripe_factor)) {
                if(vc->mrail.rails[actual_index[rail]].s_weight > 0) {
                    s_total += vc->mrail.rails[actual_index[rail]].s_weight;
                    mapped[count_rail] = actual_index[rail];
                    count_rail++;
                }
                rail = (rail + 1) % rdma_num_rails;
            }
            
            disp = 0;

            for(count_rail = 0; count_rail
                    < ((rdma_num_rails / stripe_factor) - 1);
                    count_rail++) {

                inc = vc->mrail.rails[mapped[count_rail]].s_weight *
                    (nbytes / s_total);
                
                if (inc <= 0) { 
                    continue;
                }
                
                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                v->sreq = rreq;
                MRAILI_RDMA_Get(vc, v,
                        (char *) (rreq->mrail.rndv_buf) +
                        rreq->mrail.rndv_buf_off + disp,
                        ((dreg_entry *)rreq->mrail.d_entry)->
                        memhandle[vc->mrail.
                        rails[mapped[count_rail]].hca_index]->lkey,
                        (char *) (rreq->mrail.remote_addr) +
                        rreq->mrail.rndv_buf_off + disp,
                        rreq->mrail.rkey[vc->mrail.
                        rails[mapped[count_rail]].hca_index],
                        inc, mapped[count_rail]);

                rreq->mrail.num_rdma_read_completions++;
                /* Send the finish message immediately after the data */
                disp += inc;
            }

            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = rreq;
            MRAILI_RDMA_Get(vc, v,
                    (char *) (rreq->mrail.rndv_buf) +
                    rreq->mrail.rndv_buf_off + disp,
                    ((dreg_entry *)rreq->mrail.d_entry)->
                    memhandle[vc->mrail.
                    rails[mapped[count_rail]].hca_index]->lkey,
                    (char *) (rreq->mrail.remote_addr) +
                    rreq->mrail.rndv_buf_off + disp,
                    rreq->mrail.rkey[vc->mrail.
                    rails[mapped[count_rail]].hca_index],
                    nbytes - disp, mapped[count_rail]);
            rreq->mrail.num_rdma_read_completions++;

        }
        /* Send the finish message immediately after the data */  
        rreq->mrail.rndv_buf_off += nbytes; 
    }

    /* Send the finish message through the rails, when all data is sent */
    if (rreq->mrail.rndv_buf_off == rreq->mrail.rndv_buf_sz) {
        rreq->mrail.nearly_complete = 1;
    } else { 
        rreq->mrail.nearly_complete = 0;
    }
}


/* Algorithm:
 *
 * if(size) is less than striping threshold
 *      select only one rail and send a message through this
 *  otherwise
 *      if HSAM is defined, use the best stripe_factor rails
 *      and send the message through them
 *  else
 *      stripe the message evenly through all paths 
 */

void MPIDI_CH3I_MRAILI_Rendezvous_rput_push(MPIDI_VC_t * vc,
        MPID_Request * sreq)
{
    vbuf *v;
    int rail, disp, s_total;
    MPIDI_msg_sz_t nbytes, inc;
    int rail_index;

    int count_rail;

    int mapped[MAX_NUM_SUBRAILS];
    int actual_index[MAX_NUM_SUBRAILS];
    int queued = 0;
    int ext_sendq_size = 0;

    double time;

    for(rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
        if(mv2_MPIDI_CH3I_RDMA_Process.has_apm && apm_tester) {
            perform_manual_apm(vc->mrail.rails[rail_index].qp_hndl);
        }
    }
    
    sreq->mrail.local_complete = 0;

#if defined(DEBUG)
    if (sreq->mrail.rndv_buf_sz > 0) {
        MPIU_Assert(sreq->mrail.d_entry != NULL);
        MPIU_Assert(sreq->mrail.remote_addr != NULL);
    }
#endif /* defined(DEBUG) */

    /* Use the HSAM Functionality */
    if(mv2_MPIDI_CH3I_RDMA_Process.has_hsam && 
            (sreq->mrail.rndv_buf_sz > rdma_large_msg_rail_sharing_threshold)) {

      MPIU_Memset(mapped, 0, rdma_num_rails * sizeof(int));
      MPIU_Memset(actual_index, 0, rdma_num_rails * sizeof(int));

        get_sorted_index(vc, actual_index);
    
        /* Get the wall-time, internally defined function */
        get_wall_time(&time);

        /* Set the start time for the stripe and the 
         * finish time to be zero*/ 

        sreq->mrail.stripe_start_time = time;

        for(rail = 0; rail < rdma_num_rails; rail++) {

            sreq->mrail.initial_weight[rail] = 
                vc->mrail.rails[rail].s_weight;
            sreq->mrail.stripe_finish_time[rail] = 0;
        }

    }

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && sreq->mrail.cuda_transfer_mode != NONE) {
        MPIDI_CH3I_MRAILI_Rendezvous_rput_push_cuda(vc, sreq);
        return;
    }
#endif
    while (sreq->mrail.rndv_buf_off < sreq->mrail.rndv_buf_sz) {
        nbytes = sreq->mrail.rndv_buf_sz - sreq->mrail.rndv_buf_off;

        if (nbytes > mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize) {
            nbytes = mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize;
        }

        DEBUG_PRINT("[buffer content]: %02x,%02x,%02x, offset %d, remote buf %p\n",
                ((char *) sreq->mrail.rndv_buf)[0],
                ((char *) sreq->mrail.rndv_buf)[1],
                ((char *) sreq->mrail.rndv_buf)[2],
                sreq->mrail.rndv_buf_off, sreq->mrail.remote_addr);
        
        if (nbytes <= rdma_large_msg_rail_sharing_threshold) {
            rail = MRAILI_Send_select_rail(vc);

            /* Get current number of pending entries */
            GET_EXT_SENDQ_SIZE(vc, rail, ext_sendq_size);
            if (ext_sendq_size >= rdma_rndv_ext_sendq_size) {
                break;
            }

            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = sreq;

            MRAILI_RDMA_Put(vc, v,
                    (char *) (sreq->mrail.rndv_buf) +
                    sreq->mrail.rndv_buf_off,
                    ((dreg_entry *)sreq->mrail.d_entry)->
                    memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                    (char *) (sreq->mrail.remote_addr) +
                    sreq->mrail.rndv_buf_off,
                    sreq->mrail.rkey[vc->mrail.rails[rail].hca_index],
                    nbytes, rail);
           
        } else if(!mv2_MPIDI_CH3I_RDMA_Process.has_hsam) {
            for(rail = 0; rail < rdma_num_rails; rail++) {
                /* Get current number of pending entries */
                GET_EXT_SENDQ_SIZE(vc, rail, ext_sendq_size);
                if (ext_sendq_size >= rdma_rndv_ext_sendq_size) {
                    queued = 1;
                    break;
                }
            }
            if (queued) {
                break;
            }

            inc = nbytes / rdma_num_rails;
            
            for(rail = 0; rail < rdma_num_rails - 1; rail++) {
                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                v->sreq = sreq;
                MRAILI_RDMA_Put(vc, v,
                        (char *) (sreq->mrail.rndv_buf) +
                        sreq->mrail.rndv_buf_off + rail * inc,
                        ((dreg_entry *)sreq->mrail.d_entry)->
                        memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                        (char *) (sreq->mrail.remote_addr) +
                        sreq->mrail.rndv_buf_off + rail * inc,
                        sreq->mrail.rkey[vc->mrail.rails[rail].hca_index], 
                        inc, rail);
                /* Send the finish message immediately after the data */  
            }
            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = sreq;
            MRAILI_RDMA_Put(vc, v,
                    (char *) (sreq->mrail.rndv_buf) +
                    sreq->mrail.rndv_buf_off + inc * (rdma_num_rails - 1),
                    ((dreg_entry *)sreq->mrail.d_entry)->
                    memhandle[vc->mrail.rails[rail].hca_index]->lkey,
                    (char *) (sreq->mrail.remote_addr) +
                    sreq->mrail.rndv_buf_off + inc * (rdma_num_rails - 1),
                    sreq->mrail.rkey[vc->mrail.rails[rail].hca_index], 
                    nbytes - (rdma_num_rails - 1) * inc, rail);

        } else {
            rail = 0;
            count_rail = 0;
           
            s_total = 0;

            while(count_rail <( rdma_num_rails / stripe_factor)) {
                if(vc->mrail.rails[actual_index[rail]].s_weight > 0) {
                    s_total += vc->mrail.rails[actual_index[rail]].s_weight;
                    mapped[count_rail] = actual_index[rail];
                    count_rail++;
                }
                rail = (rail + 1) % rdma_num_rails;
            }
            
            disp = 0;

            for(count_rail = 0; count_rail
                    < ((rdma_num_rails / stripe_factor) - 1);
                    count_rail++) {

                inc = vc->mrail.rails[mapped[count_rail]].s_weight *
                    (nbytes / s_total);

                if (inc <= 0) { 
                    continue;
                    
                }
                
                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                v->sreq = sreq;
                MRAILI_RDMA_Put(vc, v,
                        (char *) (sreq->mrail.rndv_buf) +
                        sreq->mrail.rndv_buf_off + disp,
                        ((dreg_entry *)sreq->mrail.d_entry)->
                        memhandle[vc->mrail.
                        rails[mapped[count_rail]].hca_index]->lkey,
                        (char *) (sreq->mrail.remote_addr) +
                        sreq->mrail.rndv_buf_off + disp,
                        sreq->mrail.rkey[vc->mrail.
                        rails[mapped[count_rail]].hca_index],
                        inc, mapped[count_rail]);

                /* Send the finish message immediately after the data */
                disp += inc;
            }

            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            v->sreq = sreq;
            MRAILI_RDMA_Put(vc, v,
                    (char *) (sreq->mrail.rndv_buf) +
                    sreq->mrail.rndv_buf_off + disp,
                    ((dreg_entry *)sreq->mrail.d_entry)->
                    memhandle[vc->mrail.
                    rails[mapped[count_rail]].hca_index]->lkey,
                    (char *) (sreq->mrail.remote_addr) +
                    sreq->mrail.rndv_buf_off + disp,
                    sreq->mrail.rkey[vc->mrail.
                    rails[mapped[count_rail]].hca_index],
                    nbytes - disp, mapped[count_rail]);


        }
        sreq->mrail.rndv_buf_off += nbytes; 
    }

    if( sreq->mrail.rndv_buf_off != sreq->mrail.rndv_buf_sz ) {
          DEBUG_PRINT("%s: [%d -> %d]: rndv_buf_off %d != rndv_buf_sz %d putting vc back in flowlist\n", __func__, 
          MPIDI_Process.my_pg_rank, vc->pg_rank, 
          sreq->mrail.rndv_buf_off, sreq->mrail.rndv_buf_sz  );
    }

    /* Send the finish message through the rails, when all data is sent */
    if (sreq->mrail.rndv_buf_off == sreq->mrail.rndv_buf_sz) {
        for(rail = 0; rail < rdma_num_rails; rail++) { 
            MRAILI_RDMA_Put_finish(vc, sreq, rail);
        }
        sreq->mrail.nearly_complete = 1;
    } else { 
        sreq->mrail.nearly_complete = 0;
    }
}

int MPIDI_CH3I_MRAILI_Rendezvous_r3_ack_send(MPIDI_VC_t *vc)
{
    vbuf *v;
    int mpi_errno;
    MPL_IOV iov;
    int total_len;
    int rail;
    MPID_Seqnum_t seqnum;

    MRAILI_Get_buffer(vc, v, rdma_vbuf_total_size);
    rail = MRAILI_Send_select_rail(vc);
 
    MPIDI_CH3_Pkt_rndv_r3_ack_t r3_ack;
    MPIDI_Pkt_init(&r3_ack, MPIDI_CH3_PKT_RNDV_R3_ACK);
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(&r3_ack, seqnum);

    r3_ack.ack_data = vc->ch.received_r3_data;
    vc->ch.received_r3_data = 0;

    iov.MPL_IOV_BUF = &r3_ack;
    iov.MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_rndv_r3_ack_t);

    total_len = MRAILI_Fill_start_buffer(v, &iov, 1);
    DEBUG_PRINT("[[eager send] len %d vbuf: %p\n",total_len, v);
    vbuf_init_send(v, total_len, rail);
    
    mpi_errno = mv2_MPIDI_CH3I_RDMA_Process.post_send(vc, v, rail);

    return mpi_errno;
}

/* Algorithm:
 * if (message size < striping threshold)
 *     mark as complete, independent of the rendezvous protocol
 * 
 * if (rendezvous protocol == RDMA Read)
 *     only one finish is expected, mark as complete
 *
 * if (rendezvous protocol == RDMA Write)
 *     rdma_num_rails finish messages are expected
 *     check this condition and mark complete accordingly
 */

int MPIDI_CH3I_MRAIL_Finish_request(MPID_Request *rreq)
{
    PRINT_DEBUG(DEBUG_RNDV_verbose > 1,
            "rreq: %p, protocol: %d, local_complete: %d, remote_complete: %d\n",
            rreq, rreq->mrail.protocol, rreq->mrail.local_complete, rreq->mrail.remote_complete);

    MPIU_Assert(rreq->mrail.local_complete  == UINT32_MAX || rreq->mrail.local_complete  <= rdma_num_rails);
    MPIU_Assert(rreq->mrail.remote_complete == UINT32_MAX || rreq->mrail.remote_complete <= rdma_num_rails);

    switch(rreq->mrail.protocol) {
        case MV2_RNDV_PROTOCOL_RGET:
        case MV2_RNDV_PROTOCOL_RPUT:
            return (rreq->mrail.local_complete == UINT32_MAX || rreq->mrail.remote_complete == UINT32_MAX);
        default:
            break;
    }

    return 1;
}

/* Get the sorted indices for the given array */
void get_sorted_index(MPIDI_VC_t *vc, int *b)
{               
    int *taken;

    int i, j, max = -1, index = 0;

    taken = (int *) MPIU_Malloc(sizeof(int) * rdma_num_rails);

    /* Sanity */ 
    MPIU_Memset(taken, 0, sizeof(int) * rdma_num_rails);
    
    /* Sort the array */
    for(i = 0; i < rdma_num_rails; i++) {
        for(j = 0; j < rdma_num_rails; j++) {
            if((vc->mrail.rails[j].s_weight >= max)
                    && (taken[j] != -1)) {
                max = vc->mrail.rails[j].s_weight;
                index = j;
            }
        }
        taken[index] = -1;
        b[i] = index;

        MPIU_Assert((index >= 0) && (index < rdma_num_rails));
        max = -1;
    }

    /* Free the buffer */
    MPIU_Free(taken);
}



#undef FUNCNAME 
#define FUNCNAME adjust_weights 
#undef FCNAME       
#define FCNAME MPL_QUOTE(FUNCNAME)

void adjust_weights(MPIDI_VC_t *vc, double start_time,
    double *finish_time,
    double *init_weight)
{
    int i;
    double bw[MAX_NUM_SUBRAILS];
    double bw_total = 0;
    int weight_assigned = 0;
    int count_rails_used = 0;
    int rail_used[MAX_NUM_SUBRAILS];

    MPIU_Memset(rail_used, 0, sizeof(int) * MAX_NUM_SUBRAILS);

    for (i = 0; i < rdma_num_rails; i++) {

        /* This rail was used at all */
        if(finish_time[i] > 0) {
            finish_time[i] -= start_time;
            MPIU_Assert(finish_time[i] > 0);
            finish_time[i] /= 100;
            bw[i] = (init_weight[i]) / (double)(finish_time[i]);
            bw_total += bw[i];
            rail_used[i] = 1;
            count_rails_used++;
        }
    }

    for (i = 0; i < rdma_num_rails; i++) {
        /* Only update if the path is used */
        if(rail_used[i]){
        
        /* Use a linear model for path updates to tolerate
         * jitter from the network */

            vc->mrail.rails[i].s_weight =
               (int) (alpha * ((count_rails_used * DYNAMIC_TOTAL_WEIGHT *
                        bw[i] / (bw_total * rdma_num_rails)))
                + (1 - alpha) * vc->mrail.rails[i].s_weight);
            MPIU_Assert(vc->mrail.rails[i].s_weight > 0);
        
        }
        if ( vc->mrail.rails[i].s_weight >= 0) {
            weight_assigned += vc->mrail.rails[i].s_weight;
        }
    }
}

/* User defined function for wall-time */
void get_wall_time(double *t)
{
    struct timeval tv;
    static int initialized = 0;
    static int sec_base;
   
    gettimeofday(&tv, NULL);
   
    if (!initialized) {
        sec_base = tv.tv_sec;
        initialized = 1;
    }
    
    *t = (double) (tv.tv_sec - sec_base) * 1.0 + 
        (double) tv.tv_usec * 1.0e-6;
}
