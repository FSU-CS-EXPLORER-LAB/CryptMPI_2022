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
#include "ib_lmt.h"
#include "ib_vbuf.h"
#include "ib_send.h"
#include "ib_param.h"
#include "ib_process.h"
#include "ib_errors.h"
/* add ib_poll.h for MPIDI_nem_ib_request_adjust_iov */
#include "ib_poll.h"

#define MAXTRANSFERSIZE 4*1024*1024

#define Calculate_IOV_len(_iov, _n_iov, _len)                   \
{   int _i; (_len) = 0;                                         \
    for (_i = 0; _i < (_n_iov); _i ++) {                        \
        (_len) += (_iov)[_i].MPL_IOV_LEN;                      \
    }                                                           \
}

MPIDI_VC_t *flowlist;

static inline void MPIDI_nem_ib_POST_SR(vbuf *_v, MPID_nem_ib_connection_t *_c,
int _rail, char *err_string)
{
    int __ret;

    if((_v->desc.sg_entry.length <= rdma_max_inline_size)
            && (_v->desc.u.sr.opcode != IBV_WR_RDMA_READ) )
    {
       _v->desc.u.sr.send_flags = (enum ibv_send_flags)(IBV_SEND_SIGNALED |
IBV_SEND_INLINE);


    } else {
        _v->desc.u.sr.send_flags = IBV_SEND_SIGNALED ;
    }

    if (_rail != _v->rail)
    {
            DEBUG_PRINT(stderr, "[%s:%d] rail %d, vrail %d\n", __FILE__,
__LINE__,(_rail), (_v)->rail);
            MPIU_Assert((_rail) == (_v)->rail);
    }

    process_info.global_used_send_cq++;

    __ret = ibv_post_send( _c->rails[_rail].qp_hndl, & (_v->desc.u.sr),
&(_v->desc.y.bad_sr) );
    if(__ret) {
        ibv_error_abort(-1, err_string);
    }
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_rndv_initiate
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_lmt_ib_initiate_lmt(MPIDI_VC_t *vc, MPIDI_CH3_Pkt_t *pkt,
                                           struct MPID_Request *req) 
{
   int dt_contig;
   MPIDI_msg_sz_t data_sz;
   MPI_Aint dt_true_lb;
   MPID_Datatype * dt_ptr; 
   struct dreg_entry *d_entry;
   int mpi_errno = MPI_SUCCESS;

   MPID_nem_pkt_lmt_rts_t *rts_pkt = (MPID_nem_pkt_lmt_rts_t *) pkt;
   MPID_nem_ib_lmt_cookie *cookie = MPIU_Malloc(sizeof(MPID_nem_ib_lmt_cookie));

   /*all variable declarations must be done before state declaration*/
   MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_LMT_IB_INITIATE_LMT);
   MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_LMT_IB_INITIATE_LMT);

   /*get information about the datatype*/
   MPIDI_Datatype_get_info(req->dev.user_count, req->dev.datatype, dt_contig, data_sz, dt_ptr,
                                           dt_true_lb);

   /*protocol is set to RPUT by default for now*/
   if (MV2_LMT_PROTOCOL_RPUT == rdma_rndv_protocol) {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_RPUT;
   } else if (MV2_LMT_PROTOCOL_RGET == rdma_rndv_protocol) {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_RGET;
   } else {
        REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
   }

   if (dt_contig) {

        /*need to look into where OnDataAvail is used for non-contig*/
        req->dev.OnDataAvail = 0;
        req->dev.iov[0].MPL_IOV_BUF = (MPL_IOV_BUF_CAST)
                             ((char*)req->dev.user_buf + dt_true_lb);
        req->dev.iov[0].MPL_IOV_LEN = data_sz;
        req->dev.iov_count = 1;

        REQ_FIELD(req, rndv_buf) = req->dev.iov[0].MPL_IOV_BUF;
        REQ_FIELD(req, rndv_buf_sz) = req->dev.iov[0].MPL_IOV_LEN;
        REQ_FIELD(req, rndv_buf_alloc) = 0;

   } else {

        req->dev.segment_ptr = MPID_Segment_alloc();
        MPID_Segment_init(req->dev.user_buf, req->dev.user_count,
               req->dev.datatype, req->dev.segment_ptr, 0);
        req->dev.iov_count = MPL_IOV_LIMIT;
        req->dev.segment_first = 0;
        req->dev.segment_size = data_sz; 
        /* One the initial load of a send iov req, set the OnFinal action (null
        for point-to-point) */
        req->dev.OnDataAvail = 0;
        req->dev.OnFinal = 0;
        mpi_errno = MPIDI_CH3U_Request_load_send_iov(req, &req->dev.iov[0],
                             &req->dev.iov_count);
        if (mpi_errno != MPI_SUCCESS) {
          mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                         FCNAME, __LINE__, MPI_ERR_OTHER,
                         "**ch3|loadsendiov", 0);
          goto fn_exit;
        }

        REQ_FIELD(req, rndv_buf_sz) = req->dev.segment_size;
        REQ_FIELD(req, rndv_buf) = MPIU_Malloc(REQ_FIELD(req, rndv_buf_sz));
        if (REQ_FIELD(req, rndv_buf) == NULL) {
            /* fall back to r3 if cannot allocate tmp buf */
            REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
            REQ_FIELD(req, rndv_buf_sz) = 0;
            REQ_FIELD(req, rndv_buf_alloc) = 0;
        } else {
            REQ_FIELD(req, rndv_buf_alloc) = 1;
        }

   }
   REQ_FIELD(req, rndv_buf_off) = 0; 

   if ( (MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(req, protocol)) ||
            (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(req, protocol)) ) {

        /*use this if dreg_register has been defined and replace memhandle
        reg_entry*/
        d_entry = dreg_register(REQ_FIELD(req, rndv_buf), REQ_FIELD(req, rndv_buf_sz));
        if (NULL != d_entry) { 
            REQ_FIELD(req, completion_counter) = 0;
            REQ_FIELD(req, d_entry) = d_entry; 
        }
        else {
            REQ_FIELD(req, protocol) = MV2_LMT_PROTOCOL_R3;
            /*buffer allocated in non-contig case must be freed here*/
        }  

    }

   /*filling in the cookie*/
   cookie->protocol = REQ_FIELD(req, protocol);    
   if ((MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(req, protocol)) ||     
            (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(req, protocol))) {
        cookie->buf_addr= REQ_FIELD(req, rndv_buf);
        /*rkey for all hcas' need to be filled out*/
        cookie->rkey[0] = REQ_FIELD(req, d_entry)->memhandle[0]->rkey;
   }

   MPID_nem_lmt_send_RTS(vc, rts_pkt, cookie, sizeof(MPID_nem_ib_lmt_cookie));
 fn_exit:
 fn_fail:
    MPIU_Free(cookie);
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_LMT_IB_INITIATE_LMT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME vbuf_init_rndv_rput
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void vbuf_init_rndv_rput(
    vbuf* v,
    void* local_address,
    uint32_t lkey,
    void* remote_addr,
    uint32_t rkey,
    MPIDI_msg_sz_t len,
    int rail)
{
    MPIDI_STATE_DECL(MPID_STATE_IB_VBUF_INIT_RPUT);
    MPIDI_FUNC_ENTER(MPID_STATE_IB_VBUF_INIT_RPUT);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    v->desc.u.sr.opcode = IBV_WR_RDMA_WRITE;
    v->desc.u.sr.wr_id = (uintptr_t) v;

    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.wr.rdma.remote_addr = (uintptr_t)(remote_addr);
    v->desc.u.sr.wr.rdma.rkey = rkey;

    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = lkey;
    v->desc.sg_entry.addr = (uintptr_t)(local_address);
    v->padding = RPUT_VBUF_FLAG;
    v->rail = rail;

    MPIDI_FUNC_EXIT(MPID_STATE_IB_VBUF_INIT_RPUT);
}

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_rput
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_rput(struct MPIDI_VC *vc, struct MPID_Request *sreq)
{
  vbuf *v;  
  MPIDI_msg_sz_t nbytes;
  int rail = 0, hca_num = 0;
  int mpi_errno = MPI_SUCCESS;
  int cq_overflow = 0;
  int complete = 0;
  int i;
  uintptr_t buf;

  /*all variable declarations must be done before state declaration*/
  MPIDI_STATE_DECL(MPID_STATE_MPIDI_NEM_IB_RPUT);
  MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_NEM_IB_RPUT);

  /*if rndv_buf_off is not zero, the data would have already been copied*/
  if (REQ_FIELD(sreq, rndv_buf_alloc) == 1 && 
      REQ_FIELD(sreq, rndv_buf_off) == 0) {

      buf = (uintptr_t) REQ_FIELD(sreq, rndv_buf);
      do {

         for (i = 0; i < sreq->dev.iov_count; i++) {
             MPIU_Memcpy((void *) buf, sreq->dev.iov[i].MPL_IOV_BUF,
                       sreq->dev.iov[i].MPL_IOV_LEN);
             buf += sreq->dev.iov[i].MPL_IOV_LEN;
         }

         if (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV) {
             complete = 0;
             sreq->dev.iov_count = MPL_IOV_LIMIT;
             mpi_errno = MPIDI_CH3U_Request_load_send_iov(sreq,
                          sreq->dev.iov,
                          &sreq->dev.iov_count);
             if (mpi_errno != MPI_SUCCESS) {
                 mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL,
                            FCNAME, __LINE__, MPI_ERR_OTHER,
                            "**ch3|loadsendiov", 0);
                  goto fn_exit;
             }
         } else {
             complete = 1;
         }
      } while(!complete);
  }

  while (REQ_FIELD(sreq, rndv_buf_off) <
            REQ_FIELD(sreq, rndv_buf_sz)) {

      nbytes = REQ_FIELD(sreq, rndv_buf_sz) - REQ_FIELD(sreq, rndv_buf_off);

      if (nbytes > MAXTRANSFERSIZE) {
            nbytes = MAXTRANSFERSIZE;
      }

      /*we need to handle striping over multiple rails here
      for now we use just one rail*/
      v = get_vbuf();
      v->sreq = sreq;
      v->vc = vc;

      vbuf_init_rndv_rput(v, (void *) REQ_FIELD(sreq, rndv_buf) +
                    REQ_FIELD(sreq, rndv_buf_off),
                    (REQ_FIELD(sreq, d_entry)->memhandle[hca_num])->lkey,
                    REQ_FIELD(sreq, remote_addr)  +
                    REQ_FIELD(sreq, rndv_buf_off), 
                    REQ_FIELD(sreq, rkey[hca_num]),
                    nbytes, rail);

      if ((NULL != hca_list[hca_num].send_cq_hndl) &&
            (VC_FIELD(vc, connection)->rails[rail].used_send_cq >=
            rdma_default_max_cq_size)) {
            /* We are monitoring CQ's and there is CQ overflow */
            cq_overflow = 1;
      }

      if (!VC_FIELD(vc, connection)->rails[rail].send_wqes_avail ||
              cq_overflow)
      {
            REQ_FIELD(sreq, rndv_buf_off) += nbytes;
            MRAILI_Ext_sendq_enqueue(vc, rail, v);
            MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_RPUT);
            if (REQ_FIELD(sreq, rndv_buf_off) <
                REQ_FIELD(sreq, rndv_buf_sz)) { 
                return MPI_MRAIL_MSG_QUEUED;
            } else {
                return mpi_errno; 
            }
      }
      else
      {
            VC_FIELD(vc, connection)->rails[rail].send_wqes_avail--;
            MPIDI_nem_ib_POST_SR(v, VC_FIELD( vc, connection), rail,
                    "ibv_post_sr (post_send_desc)");
      }

      REQ_FIELD(sreq, rndv_buf_off) += nbytes;

   }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_NEM_IB_RPUT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_NEM_IB_R3_SEND
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_nem_ib_r3_send(MPIDI_VC_t * vc, MPID_Request * sreq)
{
    vbuf *buf;
    int mpi_errno = MPI_SUCCESS;
    int n_iov;
    int pkt_len;
    int msg_buffered = 0;
    int nb;
    int complete;
    MPL_IOV iov[MPL_IOV_LIMIT + 1];
    MPIDI_CH3_Pkt_rndv_r3_data_t pkt_head;

    MPIDI_STATE_DECL(MPIDI_NEM_IB_R3_SEND);
    MPIDI_FUNC_ENTER(MPIDI_NEM_IB_R3_SEND);

    MPIDI_Pkt_init(&pkt_head, MPIDI_CH3_PKT_RNDV_R3_DATA);
    iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t);
    iov[0].MPL_IOV_BUF = (void*) &pkt_head;
    pkt_head.receiver_req_id = (sreq)->ch.lmt_req_id;

    do{
        do{
        /* stop sending more R3 data to avoid SRQ flooding at receiver */
            if (process_info.has_srq) {
                while (VC_FIELD(vc, pending_r3_data) >= rdma_max_r3_pending_data) {
                   MPID_nem_ib_poll(FALSE); 
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

            Calculate_IOV_len(iov, n_iov, pkt_len);

            mpi_errno =
                MPIDI_nem_ib_eager_send(vc, iov, n_iov, pkt_len, &nb,
                        &buf);

            if (MPI_SUCCESS != mpi_errno
                && MPI_MRAIL_MSG_QUEUED != mpi_errno) {
                sreq->status.MPI_ERROR = MPI_ERR_INTERN;
                MPID_Request_complete(sreq);
                goto fn_exit;
            } else if (MPI_MRAIL_MSG_QUEUED == mpi_errno) {
                msg_buffered = 1;
            }

            nb -= sizeof(MPIDI_CH3_Pkt_rndv_r3_data_t) + IB_PKT_HEADER_LENGTH;
            VC_FIELD(vc, pending_r3_data) += nb;
        } while (!MPIDI_nem_ib_request_adjust_iov(sreq, nb));

        if (sreq->dev.OnDataAvail == MPIDI_CH3_ReqHandler_SendReloadIOV) {
            MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
            nb = 0;
            complete = 0;
        } else {
            complete = 1;
        }

    } while (!complete);

    if (msg_buffered) {
        mpi_errno = MPI_MRAIL_MSG_QUEUED;
        buf->sreq = (void *) sreq;
    } else {
        buf->sreq = NULL;
        MPIDI_CH3U_Handle_send_req(vc, sreq, &complete);
    }

fn_exit:
    MPIDI_DBG_PRINTF((50, FCNAME, "exiting"));
    MPIDI_FUNC_EXIT(MPIDI_NEM_IB_R3_SEND);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_rndv_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_lmt_ib_start_send(struct MPIDI_VC *vc, struct MPID_Request *sreq,
                                           MPL_IOV r_cookie)
{

   /*all variable declarations must be done before state declaration*/
   int mpi_errno = MPI_SUCCESS;
   MPID_nem_ib_lmt_cookie *cookie = (MPID_nem_ib_lmt_cookie *)r_cookie.MPL_IOV_BUF;

   MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_LMT_IB_START_SEND);
   MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_LMT_IB_START_SEND);

   /*There is an additional check that is done in the
   current MPIDI_CH3_PktHandler_RndvClrToSend function. 
   Is it required here?*/

   /*check for receive size and print warning or update the rndv size*/

   if(cookie->protocol == MV2_LMT_PROTOCOL_R3) {

      if (REQ_FIELD(sreq, d_entry) != NULL) {
           dreg_unregister(REQ_FIELD(sreq, d_entry));
           REQ_FIELD(sreq, d_entry) = NULL;
      }
      if (1 == REQ_FIELD(sreq, rndv_buf_alloc)
               && NULL != REQ_FIELD(sreq, rndv_buf)) {
           MPIU_Free(REQ_FIELD(sreq, rndv_buf));
           REQ_FIELD(sreq, rndv_buf_alloc) = 0;
           REQ_FIELD(sreq, rndv_buf) = NULL;
      }
      REQ_FIELD(sreq, protocol) = MV2_LMT_PROTOCOL_R3;

   }
   
   switch (REQ_FIELD(sreq, protocol))
   {
       case MV2_LMT_PROTOCOL_RPUT:
             /*multirail is not handled yet*/
             REQ_FIELD(sreq, remote_addr) = cookie->buf_addr;
             REQ_FIELD(sreq, rkey[0]) = cookie->rkey[0];
             mpi_errno = MPIDI_nem_ib_rput(vc, sreq);
             if (mpi_errno == MPI_SUCCESS) {  
                 MPID_nem_lmt_send_DONE(vc, sreq);
             } else if (mpi_errno == MPI_MRAIL_MSG_QUEUED) {
                 RENDEZVOUS_IN_PROGRESS(vc, sreq); 
                 PUSH_FLOWLIST(vc); 
                 mpi_errno = MPI_SUCCESS;
             } else {  
                goto fn_fail;
             }
             break;
       case MV2_LMT_PROTOCOL_R3:
            MPIDI_nem_ib_r3_send(vc, sreq);  
            break;
       case MV2_LMT_PROTOCOL_RGET:
            /*we should not get a CTS for RGET protocol*/
            MPL_error_printf( "CTS for RGET protocol" );
            exit( EXIT_FAILURE );
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
            goto fn_fail;
   }

 fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_LMT_IB_START_SEND);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_lmt_ib_process_rndv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void MPID_nem_lmt_ib_process_rndv()
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request *sreq = NULL;
    MPIDI_VC_t *pending_flowlist = NULL, *temp_vc = NULL;
    int need_vc_enqueue = 0;

    MPIDI_STATE_DECL(MPID_NEM_LMT_IB_PROCESS_RNDV);
    MPIDI_FUNC_ENTER(MPID_NEM_LMT_IB_PROCESS_RNDV);
    while (flowlist) {
        need_vc_enqueue = 0;
        sreq = (MPID_Request *)(VC_FIELD(flowlist,connection)->sreq_head);
        while (sreq != NULL) {
            MPIU_Assert (MV2_LMT_PROTOCOL_RPUT == REQ_FIELD(sreq, protocol));
            mpi_errno = MPIDI_nem_ib_rput(flowlist, sreq);
            if (mpi_errno == MPI_SUCCESS) {
                MPID_nem_lmt_send_DONE(flowlist, sreq);
                RENDEZVOUS_DONE(flowlist);
                sreq = (MPID_Request *)(VC_FIELD(flowlist,connection)->sreq_head);
            } else {
                MPIU_Assert(mpi_errno == MPI_MRAIL_MSG_QUEUED);
                temp_vc = flowlist;
                need_vc_enqueue = 1; 
                break;
            }
        }

        POP_FLOWLIST();

        /*try progresssing next connection*/
        if (need_vc_enqueue) { 
            ADD_PENDING_FLOWLIST(temp_vc, pending_flowlist);
        }
    }

    while(pending_flowlist) {
        /* push pending vc to the flowlist */
        REMOVE_PENDING_FLOWLIST(temp_vc, pending_flowlist)
        PUSH_FLOWLIST(temp_vc);
    }

fn_fail:
    MPIDI_FUNC_EXIT(MPID_NEM_LMT_IB_PROCESS_RNDV);
}
