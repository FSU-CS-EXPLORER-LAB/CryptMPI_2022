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

#include "upmi.h"

#include "mpid_nem_impl.h"
#include "ib_lmt.h"
#include "ib_vbuf.h"
#include "ib_send.h"
#include "ib_recv.h"
#include "ib_rdma.h"
#include "ib_errors.h"
#include "mpidimpl.h"
#include "ib_param.h"
#include "ib_hca.h"
/* add ib_poll.h for MPIDI_nem_ib_request_adjust_iov */
#include "ib_poll.h"

/** these are supposed to be in ib_param.c **/

int MPID_nem_ib_lmt_start_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq, MPL_IOV s_cookie);
int MPIDI_CH3U_Post_data_receive_found_temp(MPID_Request * rreq);
int MPIDI_CH3I_MRAIL_Prepare_rndv(MPIDI_VC_t * vc, MPID_Request * req);
int MPIDI_NEM_Prepare_ib_lmt_r_cookie(MPIDI_VC_t * vc,
                               MPID_nem_ib_lmt_cookie *r_cookie_buf,
                               MPID_Request * rreq);
int MPID_nem_ib_lmt_done_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq);

#define IBV_RETURN_ERR   -3

#define MPIDI_CH3I_MRAIL_SET_COOKIE_RNDV(_cookie_buf, _req)               \
{                                                               \
    int _i;                                                     \
    (_cookie_buf)->protocol = REQ_FIELD(_req, protocol);             \
    if ( (MV2_LMT_PROTOCOL_RPUT == (_cookie_buf)->protocol) ||       \
            (MV2_LMT_PROTOCOL_RGET == (_cookie_buf)->protocol) ) {   \
        for (_i = 0; _i < ib_hca_num_hcas; _i ++) {               \
            (_cookie_buf)->rkey[_i] =                             \
            ((REQ_FIELD(_req, d_entry))->memhandle[_i])->rkey;       \
        }                                                       \
        (_cookie_buf)->buf_addr = REQ_FIELD(_req, rndv_buf);         \
    }                                                           \
}           


#define MPIDI_CH3I_MRAIL_REVERT_RPUT(_sreq)                     \
{                                                               \
    if (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(_sreq, protocol))          \
        REQ_FIELD(_sreq, protocol) = MV2_LMT_PROTOCOL_RPUT;           \
} 

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_lmt_start_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_lmt_start_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq, MPL_IOV s_cookie)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_nem_ib_lmt_cookie *rndv_info, *r_cookie_buf;
    int r_cookie_len;

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_LMT_START_RECV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_LMT_START_RECV);

    r_cookie_buf = MPIU_Malloc(sizeof(MPID_nem_ib_lmt_cookie));
    rndv_info = (MPID_nem_ib_lmt_cookie *)s_cookie.MPL_IOV_BUF;
    MPIDI_NEM_RNDV_SET_REQ_INFO(rreq,rndv_info);    
    
    /* whether this should be integrated here ?? */
    if (rreq->dev.recv_data_sz == 0) {
        MPID_Request_complete(rreq);
    }
    else {
        /* put the rreq->user_buf to rreq->dev.iov[] */
        mpi_errno = MPIDI_CH3U_Post_data_receive_found_temp(rreq);
                /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno != MPI_SUCCESS && rreq != NULL) {
              MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,
                                    "**nem|loadsendiov");
        }
    }

    if(MPIDI_NEM_RNDV_PROTOCOL_IS_READ(rreq)) 
    {
/*         mpi_errno = MPIDI_CH3_Prepare_rndv_get(vc, rreq);
         if (mpi_errno != MPI_SUCCESS) {
             MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,"**ch3|rndv");
         }

         mpi_errno = MPIDI_CH3_Rndv_transfer(vc,
                  NULL, rreq, NULL, rts_pkt);                             //rts_pkt needs to be modified
         if (mpi_errno != MPI_SUCCESS && rreq != NULL) {
              MPIR_ERR_SETANDJUMP(mpi_errno,
                     MPI_ERR_OTHER,"**ch3|senddata");
         }
*/
        MPL_error_printf("Function not implemented");
        exit( EXIT_FAILURE );
    } else {
        mpi_errno = MPIDI_NEM_Prepare_ib_lmt_r_cookie(VC, r_cookie_buf, rreq);
        if (mpi_errno != MPI_SUCCESS)
        {
            MPIR_ERR_SETANDJUMP(mpi_errno,MPI_ERR_OTHER,"**nem_ib|rndv");
        }
        
        r_cookie_len = sizeof(MPID_nem_ib_lmt_cookie);
        MPID_nem_lmt_send_CTS(VC, rreq, r_cookie_buf, r_cookie_len);
        
    }
    
 fn_exit:
    MPIU_Free(r_cookie_buf);
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_LMT_START_RECV);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}



/*      
 * This function is used to post a receive operation on a request for the 
 * next data to arrive.  In turn, this request is attached to a virtual
 * connection.
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3U_Post_data_receive_found_temp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3U_Post_data_receive_found_temp(MPID_Request * rreq)
{
    int dt_contig;
    MPI_Aint dt_true_lb;
    MPIDI_msg_sz_t userbuf_sz ATTRIBUTE((unused));
    MPID_Datatype * dt_ptr = NULL;
    MPIDI_msg_sz_t data_sz;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3U_IB_POST_DATA_RECEIVE_FOUND);
            
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3U_IB_POST_DATA_RECEIVE_FOUND);
    
    MPIDI_Datatype_get_info(rreq->dev.user_count, rreq->dev.datatype,
                            dt_contig, userbuf_sz, dt_ptr, dt_true_lb);
/*                
    if (rreq->dev.recv_data_sz <= userbuf_sz) {
        data_sz = rreq->dev.recv_data_sz;
    }
    else {
        MPIU_DBG_MSG_FMT(CH3_OTHER,VERBOSE,(MPIU_DBG_FDEST,
               "receive buffer too small; message truncated, msg_sz=" MPIDI_MSG_SZ_FMT ", userbuf_sz="
                                            MPIDI_MSG_SZ_FMT,
                                 rreq->dev.recv_data_sz, userbuf_sz));
        rreq->status.MPI_ERROR = MPIR_Err_create_code(MPI_SUCCESS,
                     MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_TRUNCATE,
                     "**truncate", "**truncate %d %d %d %d",
                     rreq->status.MPI_SOURCE, rreq->status.MPI_TAG,
                     rreq->dev.recv_data_sz, userbuf_sz );
        rreq->status.count = userbuf_sz;
        data_sz = userbuf_sz;
    }
*/
    data_sz = rreq->ch.lmt_data_sz;

    if (dt_contig && data_sz == rreq->dev.recv_data_sz)
    {
        /* user buffer is contiguous and large enough to store the
           entire message.  However, we haven't yet *read* the data 
           (this code describes how to read the data into the destination) */
        MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"IOV loaded for contiguous read");
        rreq->dev.iov[0].MPL_IOV_BUF =
            (MPL_IOV_BUF_CAST)((char*)(rreq->dev.user_buf) + dt_true_lb);
        rreq->dev.iov[0].MPL_IOV_LEN = data_sz;
        rreq->dev.iov_count = 1;
        /* FIXME: We want to set the OnDataAvail to the appropriate 
           function, which depends on whether this is an RMA 
           request or a pt-to-pt request. */
        rreq->dev.OnDataAvail = 0;
    }
    else {
        /* user buffer is not contiguous or is too small to hold
           the entire message */
        MPIU_DBG_MSG(CH3_OTHER,VERBOSE,"IOV loaded for non-contiguous read");
        rreq->dev.segment_ptr = MPID_Segment_alloc( );
        MPIR_ERR_CHKANDJUMP1((rreq->dev.segment_ptr == NULL), mpi_errno, MPI_ERR_OTHER, "**nomem", "**nomem %s", "MPID_Segment_alloc");
        MPID_Segment_init(rreq->dev.user_buf, rreq->dev.user_count,
                          rreq->dev.datatype, rreq->dev.segment_ptr, 0);
        rreq->dev.segment_first = 0;
        rreq->dev.segment_size = data_sz;
        mpi_errno = MPIDI_CH3U_Request_load_recv_iov(rreq);
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno,MPI_ERR_OTHER,
                                     "**ch3|loadrecviov");
        }
    }

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3U_IB_POST_DATA_RECEIVE_FOUND);
    return mpi_errno;
}


int MPIDI_CH3I_MRAIL_Prepare_rndv(MPIDI_VC_t * vc, MPID_Request * req)
{
    struct dreg_entry *d_entry;
    DEBUG_PRINT ("[prepare cts] rput protocol, recv size %d, segsize %d, io count %d\n",
        req->dev.recv_data_sz, req->dev.segment_size, req->dev.iov_count);

    if (MV2_LMT_PROTOCOL_RPUT == rdma_rndv_protocol) {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_RPUT;
    } else if (MV2_LMT_PROTOCOL_RGET == rdma_rndv_protocol) {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_RGET;
    } else {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
    }

    /* Step 1: ready for user space (user buffer or pack) */
    if (1 == req->dev.iov_count && (req->dev.OnDataAvail == NULL ||
                (req->dev.OnDataAvail == req->dev.OnFinal) ||
                (req->dev.OnDataAvail ==
                 MPIDI_CH3_ReqHandler_UnpackSRBufComplete))) {
        REQ_FIELD(req, rndv_buf) = req->dev.iov[0].MPL_IOV_BUF;
        REQ_FIELD(req, rndv_buf_sz) = req->dev.iov[0].MPL_IOV_LEN;
        REQ_FIELD(req, rndv_buf_alloc) = 0;
    } else {
        REQ_FIELD(req, rndv_buf_sz) = req->dev.segment_size;
        REQ_FIELD(req, rndv_buf) = MPIU_Malloc(REQ_FIELD(req, rndv_buf_sz));

        if (REQ_FIELD(req, rndv_buf) == NULL) {

            /* fall back to r3 if cannot allocate tmp buf */

            DEBUG_PRINT("[rndv sent] set info: cannot allocate space\n");
            REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
            REQ_FIELD(req, rndv_buf_sz) = 0;
        } else {
            REQ_FIELD(req, rndv_buf_alloc) = 1;
        }
    }
    REQ_FIELD(req, rndv_buf_off) = 0;

    /* Step 1.5: If use R3 for smaller messages */
    if (REQ_FIELD(req, rndv_buf_sz) <= rdma_r3_threshold)
    {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
    }

    /* Step 2: try register and decide the protocol */

    if ( (MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(req, protocol)) ||
            (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(req, protocol)) ) {
        DEBUG_PRINT("[cts] size registered %d, addr %p\n",
                REQ_FIELD(req, rndv_buf_sz), REQ_FIELD(req, rndv_buf));
        d_entry = dreg_register(REQ_FIELD(req, rndv_buf), REQ_FIELD(req, rndv_buf_sz));

        if (NULL == d_entry) {
            REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
            if (1 == REQ_FIELD(req, rndv_buf_alloc)) {
                MPIU_Free(REQ_FIELD(req, rndv_buf));
                REQ_FIELD(req, rndv_buf_alloc) = 0;
                REQ_FIELD(req, rndv_buf_sz) = 0;
                REQ_FIELD(req, rndv_buf) = NULL;
            }
            REQ_FIELD(req, rndv_buf_alloc) = 0;
            /* MPIU_Assert(0); */
            /*MRAILI_Prepost_R3(); */
        }
        DEBUG_PRINT("[prepare cts] register success\n");
    }

    if ( (MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(req, protocol)) ||
            (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(req, protocol)) ) {
        REQ_FIELD(req, completion_counter) = 0;
        REQ_FIELD(req, d_entry) = d_entry;
        return 1;
    } else {
        return 0;
    }
}

#if 0
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
            rndv = (rts_pkt == NULL) ? NULL : &rts_pkt->rndv;
            MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(rreq, rndv);
        break;
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
#endif  /* defined(CKPT) */
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
#endif  /* defined(CKPT) */

    return MPI_SUCCESS;
}
#endif


#undef FUNCNAME
#define FUNCNAME MPIDI_NEM_Prepare_ib_lmt_r_cookie
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_NEM_Prepare_ib_lmt_r_cookie(MPIDI_VC_t * vc,
                               MPID_nem_ib_lmt_cookie *r_cookie_buf,
                               MPID_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
                                                  
    switch (REQ_FIELD(rreq, protocol)) {
    case MV2_LMT_PROTOCOL_R3:
        {
            r_cookie_buf->protocol = MV2_LMT_PROTOCOL_R3;
            /*MRAILI_Prepost_R3(); */
            break;
        }
    case MV2_LMT_PROTOCOL_RPUT:
        {
            MPIDI_CH3I_MRAIL_Prepare_rndv(vc, rreq);
            MPIDI_CH3I_MRAIL_SET_COOKIE_RNDV(r_cookie_buf, rreq);
            MPIDI_CH3I_MRAIL_REVERT_RPUT(rreq);
            break;
        }
    case MV2_LMT_PROTOCOL_RGET:
        {
            int rank;
            UPMI_GET_RANK(&rank);
            fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);
            fprintf(stderr, "RGET preparing CTS?\n");
            mpi_errno = -1;
            exit(EXIT_FAILURE);

            break;
        }
    default:
        {
            int rank;
            UPMI_GET_RANK(&rank);
            fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);
            fprintf(stderr,
                    "Unknown protocol %d type from rndv req to send\n",
                    REQ_FIELD(rreq, protocol));
            mpi_errno = -1;
            exit(EXIT_FAILURE);

            break;
        }
    }

    return mpi_errno;
}


int MPIDI_nem_ib_Finish_request(MPID_Request *rreq)
{
    REQ_FIELD(rreq,completion_counter)++;

    if(REQ_FIELD(rreq,protocol) == MV2_LMT_PROTOCOL_RGET) {
        return 1;
    }

    if(REQ_FIELD(rreq,completion_counter) < rdma_num_rails) {
        return 0;
    }

    return 1;
}



#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_lmt_done_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_lmt_done_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq){
    int mpi_errno = MPI_SUCCESS;
    int complete;

    if (!MPIDI_nem_ib_Finish_request(rreq))
    {
        return MPI_SUCCESS;
    }

    if (REQ_FIELD(rreq,rndv_buf_alloc) == 1)
    {
        /* If we are using datatype, then need to unpack data from tmpbuf */
        int iter = 0;
        int copied = 0;
        uintptr_t buf = (uintptr_t) REQ_FIELD(rreq,rndv_buf);

        for (iter=0; iter < rreq->dev.iov_count; ++iter)
        {
          MPIU_Memcpy(rreq->dev.iov[iter].MPL_IOV_BUF,
                   (void *) buf, rreq->dev.iov[iter].MPL_IOV_LEN);
            buf += rreq->dev.iov[iter].MPL_IOV_LEN;
            copied += rreq->dev.iov[iter].MPL_IOV_LEN;
        }

        MPIDI_nem_ib_request_adjust_iov(rreq, copied);

        while (rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_UnpackSRBufReloadIOV
            || rreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_ReloadIOV)
        {
            mpi_errno = MPIDI_CH3U_Handle_recv_req(VC, rreq, &complete);

            /*** added ***/
            if (TRUE == complete && MV2_LMT_PROTOCOL_R3 == REQ_FIELD(rreq, protocol)) {
                MPIDI_NEM_IB_RREQ_RNDV_FINISH(rreq);
            }
            else if (PARTIAL_COMPLETION == complete) {
                complete = 1;
            }

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

            MPIDI_nem_ib_request_adjust_iov(rreq, copied);
        }
    }
    else
    {
        REQ_FIELD(rreq, rndv_buf) = NULL;
    }
 
    MPIDI_NEM_IB_RREQ_RNDV_FINISH(rreq);  
    if ((mpi_errno = MPIDI_CH3U_Handle_recv_req(
                    VC,
                    rreq,
                    &complete)) != MPI_SUCCESS)
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno,
                                 MPIR_ERR_RECOVERABLE, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**fail", 0);
    }

    if (complete)
    {
        VC_FIELD(VC, recv_active) = NULL;
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
    return 0;
}



#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_lmt_r3_recv_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_lmt_r3_recv_data(MPIDI_VC_t * vc, vbuf * buffer)
{                       
    int mpi_errno = MPI_SUCCESS;
    int skipsize = sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
    int nb, complete;
    MPID_Request *rreq;
    MPID_Request_get_ptr(((MPIDI_CH3_Pkt_rndv_r3_data_t *) (buffer->
                                                            pheader))->
                         receiver_req_id, rreq);

    if (!(MV2_LMT_PROTOCOL_R3 == REQ_FIELD(rreq, protocol) ||
          MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(rreq, protocol))) {
        int rank;
        UPMI_GET_RANK(&rank);
        
       DEBUG_PRINT( "[rank %d]get wrong req protocol, req %p, protocol %d\n", rank,
            rreq, REQ_FIELD(rreq, protocol));
        MPIU_Assert(MV2_LMT_PROTOCOL_R3 == REQ_FIELD(rreq, protocol) ||
               MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(rreq, protocol));
    }   

    REQ_FIELD(rreq, protocol) = MV2_LMT_PROTOCOL_R3;
 
    mpi_errno = MPIDI_nem_ib_fill_request(rreq, buffer, skipsize, &nb);
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
    
    VC_FIELD(vc, received_r3_data) += nb;
    skipsize += nb;
    DEBUG_PRINT("[recv r3: handle read] filled request nb is %d\n", nb);
    
    if(MPIDI_nem_ib_request_adjust_iov(rreq,nb)) {
        mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);

       if (TRUE == complete && MV2_LMT_PROTOCOL_R3 == REQ_FIELD(rreq, protocol)) {
            MPIDI_NEM_IB_RREQ_RNDV_FINISH(rreq);
        }
        else if (PARTIAL_COMPLETION == complete) {
            complete = 1;
        }

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
            mpi_errno = MPIDI_nem_ib_fill_request(rreq, buffer, skipsize, &nb);

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

            VC_FIELD(vc, received_r3_data) += nb;
            if(!MPIDI_nem_ib_request_adjust_iov(rreq,nb)) {
                goto fn_exit;
            }
            skipsize += nb;

            mpi_errno = MPIDI_CH3U_Handle_recv_req(vc, rreq, &complete);
            /** added  **/
            if (TRUE == complete && MV2_LMT_PROTOCOL_R3 == REQ_FIELD(rreq, protocol)) {
                MPIDI_NEM_IB_RREQ_RNDV_FINISH(rreq);
            }
            else if (PARTIAL_COMPLETION == complete) {
                complete = 1;
            }
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
            REQ_FIELD(rreq, protocol) = MV2_LMT_PROTOCOL_RENDEZVOUS_UNSPECIFIED;
        }
    }
  fn_exit:
    if (process_info.has_srq) {
        if ( VC_FIELD(vc, received_r3_data) >= rdma_max_r3_pending_data) {
            MPIDI_nem_ib_lmt_r3_ack_send(vc);
        }
    }
    DEBUG_PRINT("Successfully return from r3 recv\n");
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_lmt_r3_recv_ack 
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPIDI_nem_ib_lmt_r3_recv_ack(MPIDI_VC_t * vc,
                               void *vstart)
{
    MPIDI_CH3_Pkt_rndv_r3_ack_t* p = vstart;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_RNDV_R3_ACK_RECV);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_RNDV_R3_ACK_RECV);

    DEBUG_PRINT("Received R3 Ack %d\n", p->ack_data);
    VC_FIELD(vc, pending_r3_data) -= p->ack_data;
    MPIU_Assert(VC_FIELD(vc, pending_r3_data) == 0);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_RNDV_R3_ACK_RECV);
}
