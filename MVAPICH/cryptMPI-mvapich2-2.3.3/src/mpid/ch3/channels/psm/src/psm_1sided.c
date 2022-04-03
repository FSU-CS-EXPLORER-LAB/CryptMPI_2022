/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include "psmpriv.h"
#include "psm_vbuf.h"
#include "mpidrma.h"

/* Choose correct type for PSM tag, either psm2_mq_tag_t or uint64_t */
#if PSM_VERNO >= PSM_2_1_VERSION
    #define MAKE_1SIDED_SELECTOR(_rt, _rts) do {              \
        MAKE_PSM_SELECTOR(_rt, MPID_CONTEXT_PSMCTRL, 0, 0);   \
        _rts.tag0 = MQ_TAGSEL_ANY_TAG;                        \
        _rts.tag1 = MQ_TAGSEL_ANY_SOURCE;                     \
        _rts.tag2 = MQ_TAGSEL_ALL;                            \
    } while (0)
#else
    #define MAKE_1SIDED_SELECTOR(_rt, _rts) do {               \
        _rt = 0;                                               \
        MAKE_PSM_SELECTOR(_rt, MPID_CONTEXT_PSMCTRL, 0, 0);    \
        _rts = (MQ_TAGSEL_ANY_SOURCE & MQ_TAGSEL_ANY_TAG);     \
    } while (0)
#endif

psm_prepost_list_t *psm_prepost_list_head = NULL;

static void psm_1sided_recv(MPID_Request *req, void *ptr);
static void *psm_gen_packbuf(MPID_Request *rreq, MPID_Request *dtreq);
static void psm_init_tag();
static MPID_Request *psm_1sc_putacc_rndvrecv(MPID_Request *, MPIDI_msg_sz_t, MPID_Request **,
                                      void *, int, int, MPIDI_msg_sz_t, MPIDI_VC_t *);
static MPIDI_CH3_PktHandler_Fcn *psm_pkthndl[MPIDI_CH3_PKT_END_CH3+1];

/* notes:
   psm does not support one-sided natively. one-sided msgs (put/get/acc)
   are sent on a control-context on which vbufs are pre-posted. If 
   msgs are small, the data is sent inline with the control packet.
   If data is large, the control-packet sends the (tag,src_rank) and the
   target posts a RNVD receive on that tag,rank on a special RNDV context.
   psm_1sided_input processes all incoming one-sided packets.
    
   the rank used in the one-sided call (MPI_Put,MPI_Get) is w.r.t to the
   communicator the window operates on. a rank-mapping array created at window
   creation time remaps the rank w.r.t the communicator to a rank w.r.t
   COMM_WORLD.    
  */ 
    
void psm_init_1sided()
{
    /* psm uses CH3 packet handlers for one-sided */
    MPIDI_CH3_PktHandler_Init(psm_pkthndl, MPIDI_CH3_PKT_END_CH3+1);
    psm_init_tag();
}

void psm_prepost_1sc()
{
    int i;
    MPID_Request *req;
    vbuf *ptr;

    if(psm_tot_pposted_recvs >= MAX_PREPOST)
        return;

    psm_prepost_list_t *plptr = (psm_prepost_list_t *)MPIU_Malloc(
        sizeof(psm_prepost_list_t));

    for(i = 0; i < RCV_PREPOST; i++) {
        req = psm_create_req();
        req->kind = MPID_REQUEST_RECV;
        req->psm_flags |= PSM_1SIDED_PREPOST;
        ptr = psm_get_vbuf();
        req->vbufptr = ptr;
        ptr->req = (void*) req;
        plptr->prepost_array[i] = ptr;
        psm_1sided_recv(req, ptr->buffer); 
    }

    plptr->next = psm_prepost_list_head;
    psm_prepost_list_head = plptr;

    PRINT_DEBUG(DEBUG_1SC_verbose>0, "pre-posted recv buffers\n");
}

void psm_release_prepost_1sc()
{
    int i;
    MPID_Request *req;
    psm_prepost_list_t *plptr = psm_prepost_list_head, *plpre;

    while (plptr) {
        for (i = 0; i < RCV_PREPOST; i++) {
            if (plptr->prepost_array[i]) {
                req = (MPID_Request*) plptr->prepost_array[i]->req;
                MPIDI_CH3_Request_destroy(req);
            }
        }
        plpre = plptr;
        plptr = plptr->next;
        MPIU_Free(plpre);
    }
}

static void psm_1sided_recv(MPID_Request *req, void *ptr)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t rtag ,rtagsel;
    #else
        uint64_t rtag, rtagsel;
    #endif
    MAKE_1SIDED_SELECTOR(rtag, rtagsel);

    assert(req->psm_flags & PSM_1SIDED_PREPOST);
    req->psm_flags = 0;
    req->psm_flags |= PSM_1SIDED_PREPOST;
    ++psm_tot_pposted_recvs;
    _psm_enter_;
    PSM_IRECV(psmdev_cw.mq, rtag, rtagsel, MQ_FLAGS_NONE, ptr,
                 PSM_VBUFSZ, req, &(req->mqreq));
    _psm_exit_;
}

static PSM_ERROR_T psm_iput(int dest, void *buf, MPIDI_msg_sz_t buflen, MPID_Request *req, int src)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t stag;
    #else
        uint64_t stag = 0;
    #endif
    PSM_ERROR_T psmerr;

    MAKE_PSM_SELECTOR(stag, MPID_CONTEXT_PSMCTRL, 0, src);
    _psm_enter_;
    if ((unlikely(buflen > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_ISEND(&req, dest, buf, buflen,
                    stag, MQ_FLAGS_NONE);
    } else {
        psmerr = PSM_ISEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                    MQ_FLAGS_NONE, stag, buf, buflen, req, &(req->mqreq));
    }
    _psm_exit_;
    return psmerr;
}

static PSM_ERROR_T psm_iget_rndvsend(MPID_Request *req, int dest, void *buf, MPIDI_msg_sz_t buflen,
                       int tag, int src)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t stag;
    #else
        uint64_t stag = 0;
    #endif
    PSM_ERROR_T psmerr;

    MAKE_PSM_SELECTOR(stag, MPID_CONTEXT_RNDVPSM, tag, src);
    _psm_enter_;
    if ((unlikely(buflen > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_ISEND(&req, dest, buf, buflen,
                    stag, MQ_FLAGS_NONE);
    } else {
        psmerr = PSM_ISEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                    MQ_FLAGS_NONE, stag, buf, buflen, req, &(req->mqreq));
    }
    _psm_exit_;
    return psmerr;
}

void psm_iput_rndv(int dest, void *buf, MPIDI_msg_sz_t buflen, int tag, int src, MPID_Request **rptr)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t stag;
    #else
        uint64_t stag = 0;
    #endif

    PSM_ERROR_T psmerr ATTRIBUTE((unused));
    MPID_Request *rndvreq = NULL;

    rndvreq = psm_create_req();
    rndvreq->kind = MPID_REQUEST_SEND;
    rndvreq->psm_flags |= PSM_RNDVSEND_REQ;
    /* FIXME: this is not an efficient way
     *        however, if we use 'buf', which is from rptr's pack buffer,
     *        it causes memory leak or data corruption */
    if (NULL != (*rptr) && (*rptr)->psm_flags & PSM_1SIDED_NON_CONTIG_REQ) {
        rndvreq->pkbuf = MPIU_Malloc(buflen);
        MPIU_Memcpy(rndvreq->pkbuf, buf, buflen);
        buf = rndvreq->pkbuf;
        rndvreq->psm_flags |= (PSM_1SIDED_NON_CONTIG_REQ | PSM_PACK_BUF_FREE);
    }
    *rptr = rndvreq;
    PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv send len %zu tag %d dest %d I-am %d\n", buflen, tag, dest, src);
  
    MAKE_PSM_SELECTOR(stag, MPID_CONTEXT_RNDVPSM, tag, src);
    _psm_enter_;
    if ((unlikely(buflen > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_ISEND(rptr, dest, buf, buflen,
                    stag, MQ_FLAGS_NONE);
    } else {
        psmerr = PSM_ISEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                MQ_FLAGS_NONE, stag, buf, buflen, rndvreq, &(rndvreq->mqreq));
    }
    _psm_exit_;
}

/* used for fop, cas, fop response, cas resposne */
int psm_1sided_atomicpkt(MPIDI_CH3_Pkt_t *pkt, MPL_IOV *iov, int iov_n, int rank,
                             int srank, MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    int mpi_errno = MPI_SUCCESS;
    int i;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req;

    if (*rptr == NULL) 
        req = psm_create_req();
    else
        req = (*rptr);

    req->kind = MPID_REQUEST_SEND;
    if(pkt->type == MPIDI_CH3_PKT_FOP_RESP) {
        req->psm_flags |= PSM_FOPRESP_REQ;
    }
    req->psm_flags |= PSM_1SIDED_PUTREQ;
    vptr = psm_get_vbuf();
    req->vbufptr = vptr;
    vptr->req = (void*) req;

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }

    if(buflen <= PSM_VBUFSZ) {
        off = vptr->buffer;
       
        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
        }
        psm_iput(rank, vptr->buffer, buflen, req, srank);
        ++psm_tot_eager_puts;
    } 
    return mpi_errno;
}

/* copy iov into a single vbuf, post send to target rank,
   using 1-sided context id */

int psm_1sided_putpkt(MPIDI_CH3_Pkt_put_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    int mpi_errno = MPI_SUCCESS;
    int rank, i;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req;
    int inuse = 0;

    if (*rptr == NULL) {
        req = psm_create_req();
        *rptr = req;
    } else
        req = (*rptr);

    req->kind = MPID_REQUEST_SEND;
    req->psm_flags |= PSM_1SIDED_PUTREQ;
    vptr = psm_get_vbuf();
    req->vbufptr = vptr;
    vptr->req = (void*) req;
    rank = pkt->mapped_trank;

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }

    /* eager PUT */
    if(buflen <= PSM_VBUFSZ) {
        off = vptr->buffer;
        pkt->rndv_mode = 0;
       
        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
        }
        psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);
        ++psm_tot_eager_puts;
    } else { /* rndv PUT */
        off = vptr->buffer;
        pkt->rndv_mode = 1;
        pkt->rndv_tag = psm_get_rndvtag();
        pkt->rndv_len = iov[iov_n-1].MPL_IOV_LEN;
        buflen = 0;
        
        /* decrease header req's ref_count, since CH3 only checks the rndv one.*/
        MPIU_Object_release_ref(req, &inuse);

        /* last iov is the packet */
        for(i = 0; i < (iov_n-1); i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
            buflen = buflen + len;
        }
        psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);
        iovp = (void *)iov[iov_n-1].MPL_IOV_BUF;
        len = iov[iov_n-1].MPL_IOV_LEN;
        psm_iput_rndv(rank, iovp, len, pkt->rndv_tag, pkt->mapped_srank, rptr);
        ++psm_tot_rndv_puts;
    }
    return mpi_errno;
}

/* copy iov into a single vbuf, post send to target rank,
   using 1-sided context id */

int psm_1sided_accumpkt(MPIDI_CH3_Pkt_accum_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    int rank, i;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req;
    int inuse = 0;

    if (*rptr == NULL) {
        req = psm_create_req();
        *rptr = req;
    } else
        req = (*rptr);

    req->kind = MPID_REQUEST_SEND;
    req->psm_flags |= PSM_1SIDED_PUTREQ;
    vptr = psm_get_vbuf();
    req->vbufptr = vptr;
    vptr->req = (void*) req;
    rank = pkt->mapped_trank;

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }

    /* eager PUT */
    if(buflen <= PSM_VBUFSZ) {
        off = vptr->buffer;
        pkt->rndv_mode = 0;
        pkt->stream_mode = 0;
       
        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
        }
        psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);
    } else { /* rndv PUT */
        off = vptr->buffer;

        if (pkt->flags & MPIDI_CH3_PKT_FLAG_RMA_STREAM) {
            pkt->stream_mode = 1;
        } else {
            pkt->stream_mode = 0;
        }

        pkt->rndv_mode = 1;
        pkt->rndv_tag = psm_get_rndvtag();
        pkt->rndv_len = iov[iov_n-1].MPL_IOV_LEN;
        buflen = 0;
        
        /* decrease header req's ref_count, since CH3 only checks the rndv one.*/
        MPIU_Object_release_ref(req, &inuse);

        /* last iov is the packet */
        for(i = 0; i < (iov_n-1); i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
            buflen = buflen + len;
        }
       psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);
        iovp = (void *)iov[iov_n-1].MPL_IOV_BUF;
        len = iov[iov_n-1].MPL_IOV_LEN;
        psm_iput_rndv(rank, iovp, len, pkt->rndv_tag, pkt->mapped_srank, rptr);
    }
    ++psm_tot_accs;
    return mpi_errno;
}

int psm_1sided_getaccumpkt(MPIDI_CH3_Pkt_get_accum_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    int rank, i, last_stream_unit;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req;
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t rtag, rtagsel;
    #else
        uint64_t rtag, rtagsel;
    #endif
    PSM_ERROR_T psmerr;
    int inuse = 0;

    if (*rptr != NULL)
        last_stream_unit = (*rptr)->last_stream_unit;

    if (*rptr == NULL) {
        req = psm_create_req();
        *rptr = req;
    } else
        req = (*rptr);

    req->kind = MPID_REQUEST_SEND;
    req->psm_flags |= PSM_1SIDED_PUTREQ;
    vptr = psm_get_vbuf();
    req->vbufptr = vptr;
    vptr->req = (void*) req;
    rank = pkt->mapped_trank;

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }

    /* eager PUT */
    if(buflen <= PSM_VBUFSZ) {
        off = vptr->buffer;
        pkt->rndv_mode = 0;
        pkt->stream_mode = 0;
       
        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = (void *)((uintptr_t)off + len);
        }
        psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);
    } else { /* rndv GET ACCUM */

         if (pkt->flags & MPIDI_CH3_PKT_FLAG_RMA_STREAM) {
             pkt->stream_mode = 1;
         } else {
             pkt->stream_mode = 0;
             last_stream_unit = 1;
         }

        off = vptr->buffer;
        pkt->rndv_mode = 1;
        pkt->rndv_tag = psm_get_rndvtag();
        pkt->rndv_len = iov[iov_n-1].MPL_IOV_LEN;

        /*tag for resp packet*/
        pkt->resp_rndv_tag = psm_get_rndvtag();

        /* decrease header req's ref_count, since CH3 only checks the rndv one.*/
        MPIU_Object_release_ref(req, &inuse);

        /* last iov is the packet */
        buflen = 0;
        for(i = 0; i < (iov_n-1); i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
            buflen = buflen + len;
        }
        psm_iput(rank, vptr->buffer, buflen, req, pkt->mapped_srank);

        iovp = (void *)iov[iov_n-1].MPL_IOV_BUF;
        len = iov[iov_n-1].MPL_IOV_LEN;
        psm_iput_rndv(rank, iovp, len, pkt->rndv_tag, pkt->mapped_srank, rptr);

        /*post rndv recieve for response*/
        MPID_Request *resp_req = NULL, *orig_resp_req = NULL;

        MPID_Request_get_ptr(pkt->request_handle, orig_resp_req);
        if(!MPIR_DATATYPE_IS_PREDEFINED(orig_resp_req->dev.datatype)) {
            if(!orig_resp_req->dev.datatype_ptr->is_contig) {
                 MPI_Aint result_type_size;
                 MPID_Datatype_get_size_macro(orig_resp_req->dev.datatype, result_type_size);

                 orig_resp_req->dev.real_user_buf = orig_resp_req->dev.user_buf;
                 orig_resp_req->dev.user_buf = MPIU_Malloc(orig_resp_req->dev.user_count*result_type_size);
                 orig_resp_req->psm_flags |= PSM_RNDVRECV_GET_PACKED;
            }
        }

        resp_req = psm_create_req();
        resp_req->kind = MPID_REQUEST_RECV;
        resp_req->psm_flags |= PSM_GETACCUM_GET_RNDV_REQ;
        resp_req->last_stream_unit = last_stream_unit;
        if(orig_resp_req->psm_flags & PSM_RNDVRECV_GET_PACKED) {
            resp_req->psm_flags |= PSM_RNDVRECV_GET_PACKED;
            orig_resp_req->psm_flags &= ~PSM_RNDVRECV_GET_PACKED;
        }
        resp_req->savedreq = orig_resp_req;

        resp_req->from_rank = pkt->target_rank; /* provide rank in the RMA window (may be the same as global rank) */
        resp_req->request_completed_cb = MPIDI_CH3_Req_handler_rma_op_complete;
        resp_req->dev.source_win_handle = pkt->source_win_handle;
        resp_req->dev.rma_target_ptr = pkt->target_ptr;

        if (pkt->flags & MPIDI_CH3_PKT_FLAG_RMA_LOCK_SHARED ||
            pkt->flags & MPIDI_CH3_PKT_FLAG_RMA_LOCK_EXCLUSIVE) {
            resp_req->is_piggyback = 1;
        }

        #if PSM_VERNO >= PSM_2_1_VERSION
            rtagsel.tag0 = MQ_TAGSEL_ALL;
            rtagsel.tag1 = MQ_TAGSEL_ALL;
            rtagsel.tag2 = MQ_TAGSEL_ALL;
        #else
            rtag = 0;
            rtagsel = MQ_TAGSEL_ALL;
        #endif
        MAKE_PSM_SELECTOR(rtag, MPID_CONTEXT_RNDVPSM, pkt->resp_rndv_tag,
                  pkt->mapped_trank);

        _psm_enter_;
        if ((unlikely(pkt->rndv_len > ipath_max_transfer_size))) {
            psmerr = PSM_LARGE_IRECV(orig_resp_req->dev.user_buf, pkt->rndv_len,
                        &resp_req, rtag, rtagsel);
        } else {
            psmerr = PSM_IRECV(psmdev_cw.mq, rtag, rtagsel, MQ_FLAGS_NONE,
                        orig_resp_req->dev.user_buf, pkt->rndv_len, resp_req,
                        &(resp_req->mqreq));
        }
        _psm_exit_;
        if(unlikely(psmerr != PSM_OK)) {
            printf("ERROR: rndv recv failed\n");
        }
    }

    ++psm_tot_accs;
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_1sided_getaccumresppkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_1sided_getaccumresppkt(MPIDI_CH3_Pkt_get_accum_resp_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req = (*rptr);
    PSM_ERROR_T psmerr;
    int mpi_errno = MPI_SUCCESS, i;

    req->psm_flags |= PSM_GETACCUMRESP_REQ;

    if(!pkt->rndv_mode) {
        req->psm_flags |= PSM_CONTROL_PKTREQ;
        vptr = psm_get_vbuf();
        req->vbufptr = vptr;
        vptr->req = (void*) req;
        off = vptr->buffer;

        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = (void *) ((uintptr_t)off + len);
            buflen = buflen + len;
        }

        psmerr = psm_iput(pkt->mapped_trank, vptr->buffer, buflen, req,
                pkt->mapped_srank);
    } else {
        if (req->is_piggyback) {
            MPID_Request *sreq;
            vptr = psm_get_vbuf();
            sreq = psm_create_req();
            sreq->psm_flags |= PSM_CONTROL_PKTREQ;
            sreq->vbufptr = vptr;
            vptr->req = (void*) sreq;
            off = vptr->buffer;
            buflen = 0;

            for(i = 0; i < (iov_n - 1); i++) {
                iovp = (void *)iov[i].MPL_IOV_BUF;
                len = iov[i].MPL_IOV_LEN;
                memcpy(off, iovp, len);
                off = off + len;
                buflen = buflen + len;
            }

            psm_iput(pkt->mapped_trank, vptr->buffer, buflen, sreq,
                    pkt->mapped_srank);
        }

        iovp = (void *)iov[iov_n-1].MPL_IOV_BUF;
        len = iov[iov_n-1].MPL_IOV_LEN;
        assert(len == pkt->rndv_len);

        psmerr = psm_iget_rndvsend(req, pkt->mapped_trank, iovp, len,
                                   pkt->rndv_tag, pkt->mapped_srank);
    }

    if(unlikely(psmerr != PSM_OK)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_INTERN, "**fail");
    }

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_1sided_getpkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_1sided_getpkt(MPIDI_CH3_Pkt_get_t *pkt, MPL_IOV *iov, int iov_n,
        MPID_Request **rptr) 
{
    MPIDI_msg_sz_t buflen = 0, len;
    int mpi_errno = MPI_SUCCESS, i;
    void *off, *iovp;
    PSM_ERROR_T psmerr;
    vbuf *vptr;
    MPID_Request *req = (*rptr);

    req->psm_flags |= (PSM_GETPKT_REQ | PSM_CONTROL_PKTREQ);
    MPIU_Object_add_ref(req);    

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }
    assert(buflen <= PSM_VBUFSZ);

    vptr = psm_get_vbuf();
    req->vbufptr = vptr;
    vptr->req = (void*) req;
    off = vptr->buffer;
    
    for(i = 0; i < iov_n; i++) {
        iovp = (void *)iov[i].MPL_IOV_BUF;
        len = iov[i].MPL_IOV_LEN;
        memcpy(off, iovp, len);
        off = off + len;
    }
    psmerr = psm_iput(pkt->mapped_trank, vptr->buffer, buflen, req,
            pkt->mapped_srank);
    if(unlikely(psmerr != PSM_OK)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_INTERN, "**fail");
    }

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_1sided_getresppkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_1sided_getresppkt(MPIDI_CH3_Pkt_get_resp_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr)
{
    vbuf *vptr;
    void *iovp, *off;
    MPIDI_msg_sz_t buflen = 0, len;
    MPID_Request *req = (*rptr);
    PSM_ERROR_T psmerr;
    int mpi_errno = MPI_SUCCESS, i;

    req->psm_flags |= PSM_GETRESP_REQ;

    for(i = 0; i < iov_n; i++) {
        buflen = buflen + iov[i].MPL_IOV_LEN;
    }

    PRINT_DEBUG(DEBUG_1SC_verbose>1, "get-resp packet length %zu\n", buflen);
    /* eager get response */
    if(!pkt->rndv_mode) {
        req->psm_flags |= PSM_CONTROL_PKTREQ;
        vptr = psm_get_vbuf();
        req->vbufptr = vptr;
        vptr->req = (void*) req;
        off = vptr->buffer;
       
        for(i = 0; i < iov_n; i++) {
            iovp = (void *)iov[i].MPL_IOV_BUF;
            len = iov[i].MPL_IOV_LEN;
            memcpy(off, iovp, len);
            off = off + len;
        }

        psmerr = psm_iput(pkt->mapped_trank, vptr->buffer, buflen, req,
                pkt->mapped_srank);
    } else {
        if (req->is_piggyback) {
            MPID_Request *sreq;
            vptr = psm_get_vbuf();
            sreq = psm_create_req();
            sreq->psm_flags |= PSM_CONTROL_PKTREQ;
            sreq->vbufptr = vptr;
            vptr->req = (void*) sreq;
            off = vptr->buffer;
            buflen = 0;

            for(i = 0; i < (iov_n - 1); i++) {
                iovp = (void *)iov[i].MPL_IOV_BUF;
                len = iov[i].MPL_IOV_LEN;
                memcpy(off, iovp, len);
                off = off + len;
                buflen = buflen + len;
            }

            psm_iput(pkt->mapped_trank, vptr->buffer, buflen, sreq,
                    pkt->mapped_srank);
        }

        iovp = (void *)iov[iov_n-1].MPL_IOV_BUF;
        len = iov[iov_n-1].MPL_IOV_LEN;
        assert(len == pkt->rndv_len);
    
        psmerr = psm_iget_rndvsend(req, pkt->mapped_trank, iovp, len,
                                   pkt->rndv_tag, pkt->mapped_srank);

    }
    /* Keep track of pack buffer */
    if ((req->psm_flags & PSM_1SIDED_NON_CONTIG_REQ) && iov[iov_n-1].MPL_IOV_BUF != req->pkbuf) {
        req->psm_flags |= PSM_PACK_BUF_FREE;
        req->pkbuf = iov[iov_n-1].MPL_IOV_BUF;
    }

    if(unlikely(psmerr != PSM_OK)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_INTERN, "**fail");
    }

    return mpi_errno;
}

/*  incoming one-sided packet processing.
    Put:
        if(RNDV_PUT)
            create a new recv_req with the tag peer sent,
            post request on the RNDV context
        else
            call put-handler
            re-post the vbuf
        fi

    Accumulate:
        if(RNDV_ACCUM)
            we need to create a new request to receive the data.
            create a tmpbuf for this size and post a recv/
            Once, receive completes call packet handler, but with
            data-copied flag to do only the accumulate 
        else
            call accum-handler
            re-post the vbuf
        fi    

    Get:
        if(small_get)
            call the get_handler. It will send out data using iStartMsgv
            on 1-sided context using a vbuf.
        else


        fi
    
    Lock:
        Call lock-handler function
    Lock_granted:
        set win_ptr->lock_granted to 1. Someone is waiting for this.
            

        
*/

#define GET_VC(_vc, _whndl, _vcindex)    do {               \
    MPID_Win *win_ptr;                                      \
    MPID_Win_get_ptr(_whndl, win_ptr);                      \
    MPIDI_Comm_get_vc(win_ptr->comm_ptr, _vcindex, &_vc);   \
} while(0)

#define __check(_str, _tp)                     do { \
    if(_tp == MPIDI_CH3_PKT_##_str) goto do_##_str; \
} while(0)  

#define _SECTION(TP)                                            \
    PRINT_DEBUG(DEBUG_1SC_verbose>1, "Section handles "#TP"\n"); \
    do_##TP:                                                  

int psm_1sided_input(MPID_Request *req, MPIDI_msg_sz_t inlen)
{
    MPIDI_CH3_Pkt_t *pkt;
    MPID_Request *temp_req;
    vbuf  *vbptr;
    void *ptr;
    MPIDI_msg_sz_t msg = inlen;
    MPIDI_VC_t *vc;

    vbptr = req->vbufptr;
    ptr = vbptr->buffer;
    pkt = (MPIDI_CH3_Pkt_t *) ptr;

    __check(PUT,                  pkt->type);
    __check(PUT_IMMED,            pkt->type);
    __check(GET,                  pkt->type);
    __check(GET_RESP,             pkt->type);
    __check(GET_RESP_IMMED,       pkt->type);
    __check(ACCUMULATE,           pkt->type);
    __check(ACCUMULATE_IMMED,     pkt->type);
    __check(GET_ACCUM,            pkt->type);
    __check(GET_ACCUM_IMMED,      pkt->type);
    __check(GET_ACCUM_RESP,       pkt->type);
    __check(GET_ACCUM_RESP_IMMED, pkt->type);
    __check(CAS_IMMED,            pkt->type);
    __check(CAS_RESP_IMMED,       pkt->type);
    __check(FOP,                  pkt->type);
    __check(FOP_IMMED,            pkt->type);
    __check(FOP_RESP,             pkt->type);
    __check(FOP_RESP_IMMED,       pkt->type);
    __check(LOCK,                 pkt->type);
    __check(UNLOCK,               pkt->type);
    __check(FLUSH,                pkt->type);
    __check(LOCK_ACK,             pkt->type);
    __check(ACK,                  pkt->type);
    __check(DECR_AT_COUNTER,      pkt->type);
    __check(LOCK_OP_ACK,          pkt->type);
    goto errpkt;


    /* handle put 
       if data is small, it is received in vbuf. call the PUT packet handler to
       complete the operation. If the source-datatype was non-contiguous we
       would have packed it. In this case, the packet handler will unpack the
       data

       if data is large, the packet handler is called to parse datatype info.
       if target-datatype is non-contiguous create a packing buffer of required
       size and post the RNDV receive on this buffer. Once RNDV receive
       completes, unpack the data into correct address */
    {
        _SECTION(PUT);
        _SECTION(PUT_IMMED);
        MPIDI_CH3_Pkt_put_t *putpkt = (MPIDI_CH3_Pkt_put_t *) pkt;
        if(!putpkt->rndv_mode) { /* eager put */
            GET_VC(vc, putpkt->target_win_handle, putpkt->source_rank);
            vc->ch.recv_active = req;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "put packet from %d\n", vc->pg_rank);
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            goto end;
        } else {                /* large put */
            MPID_Request *nreq = NULL;
            MPID_Win *win_ptr = NULL;

            MPID_Win_get_ptr(putpkt->target_win_handle, win_ptr); 
            win_ptr->outstanding_rma++;

            GET_VC(vc, putpkt->target_win_handle, putpkt->source_rank);
            vc->ch.recv_active = req;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "large put packet from %d\n", vc->pg_rank);
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            psm_1sc_putacc_rndvrecv(req, inlen, &nreq, putpkt->addr,
                            putpkt->rndv_tag, putpkt->mapped_srank,
                            putpkt->rndv_len, vc);
            nreq->psm_flags |= PSM_RNDVRECV_PUT_REQ;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv_put request. posted recv %p\n", nreq);
            goto end_2;
        }

    }

    {
        _SECTION(GET);
        MPIDI_CH3_Pkt_get_t *getpkt = (MPIDI_CH3_Pkt_get_t *) pkt;
        GET_VC(vc, getpkt->target_win_handle, getpkt->source_rank);
        vc->ch.recv_active = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "get packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;
    }

    {
        _SECTION(GET_RESP);
        _SECTION(GET_RESP_IMMED);
        MPIDI_CH3_Pkt_get_resp_t *resppkt = (MPIDI_CH3_Pkt_get_resp_t *) pkt;
        GET_VC(vc, resppkt->target_win_handle, resppkt->source_rank);
        vc->ch.recv_active = req;
        req->dev.target_win_handle = resppkt->source_win_handle;
        req->dev.source_win_handle = resppkt->target_win_handle;
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;
    }

    {
        _SECTION(ACCUMULATE);
        _SECTION(ACCUMULATE_IMMED);
        MPIDI_CH3_Pkt_accum_t *acpkt = (MPIDI_CH3_Pkt_accum_t *) pkt;
        if(!acpkt->rndv_mode) { /* eager accumulate */
            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            vc->ch.recv_active = req;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "accum packet from %d\n", vc->pg_rank);
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            goto end;           /* large accumulate */
        } else if(!acpkt->stream_mode){
            MPID_Request *nreq = NULL;
            MPID_Win *win_ptr = NULL; 

            MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr); 
            win_ptr->outstanding_rma++; 

            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            req->psm_flags |= PSM_RNDV_ACCUM_REQ;
            vc->ch.recv_active = req;
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            nreq = vc->ch.recv_active;

            nreq = psm_1sc_putacc_rndvrecv(req, inlen, &nreq, 
                                    nreq->dev.user_buf, acpkt->rndv_tag,
                                    acpkt->mapped_srank, acpkt->rndv_len, vc);
            nreq->psm_flags |= PSM_RNDVRECV_ACCUM_REQ;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv_accum request. posted recv %p\n", nreq);

            goto end_2;
        } else {
            MPID_Request *nreq = NULL;
            MPID_Win *win_ptr = NULL; 
            int complete = TRUE;

            MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr); 
            win_ptr->outstanding_rma++; 

            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            req->psm_flags |= PSM_RNDV_ACCUM_REQ;
            vc->ch.recv_active = req;
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            nreq = vc->ch.recv_active;

            if(MPIR_DATATYPE_IS_PREDEFINED(nreq->dev.datatype)) {
                /*for contiguous buffer, ext_hdr has been copied insided accumulate packet handler function*/
                MPIU_Memcpy(nreq->dev.ext_hdr_ptr, (char *)ptr + sizeof(MPIDI_CH3_Pkt_t), nreq->dev.ext_hdr_sz);
                MPIDI_CH3_ReqHandler_AccumMetadataRecvComplete(vc, nreq, &complete);
            }

            nreq = psm_1sc_putacc_rndvrecv(req, inlen, &nreq, 
                                    nreq->dev.user_buf, acpkt->rndv_tag,
                                    acpkt->mapped_srank, acpkt->rndv_len, vc);
            nreq->psm_flags |= PSM_RNDVRECV_ACCUM_REQ;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv_accum request. posted recv %p\n", nreq);

            goto end_2;
        }
    }

    {
        _SECTION(GET_ACCUM);
        _SECTION(GET_ACCUM_IMMED);
        MPIDI_CH3_Pkt_get_accum_t *acpkt = (MPIDI_CH3_Pkt_get_accum_t *) pkt;

        if(!acpkt->rndv_mode) {
            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            vc->ch.recv_active = req;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "get accum packet from %d\n", vc->pg_rank);
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
            goto end;           /* large accumulate */
        } else if (!acpkt->stream_mode) {
            MPID_Request *nreq = NULL;
            MPID_Win *win_ptr = NULL;

            MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr);
            win_ptr->outstanding_rma++;
        
            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            req->psm_flags |= PSM_RNDV_ACCUM_REQ;
            vc->ch.recv_active = req;
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));

            nreq = vc->ch.recv_active;
            nreq = psm_1sc_putacc_rndvrecv(req, inlen, &nreq,
                                    nreq->dev.user_buf, acpkt->rndv_tag,
                                    acpkt->mapped_srank, acpkt->rndv_len, vc);
            nreq->psm_flags |= PSM_GETACCUM_RNDV_REQ;

            nreq->resp_rndv_tag = acpkt->resp_rndv_tag;

            PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv_accum request. posted recv %p\n", nreq);
            goto end_2; 
        } else {
            int complete = TRUE;

            MPID_Request *nreq = NULL;
            MPID_Win *win_ptr = NULL;

            MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr);
            win_ptr->outstanding_rma++;
        
            GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
            req->psm_flags |= PSM_RNDV_ACCUM_REQ;
            vc->ch.recv_active = req;
            psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));

            nreq = vc->ch.recv_active;

            if(MPIR_DATATYPE_IS_PREDEFINED(nreq->dev.datatype)) {
                MPIU_Memcpy(nreq->dev.ext_hdr_ptr, (char *)ptr + sizeof(MPIDI_CH3_Pkt_t), nreq->dev.ext_hdr_sz);
                MPIDI_CH3_ReqHandler_GaccumMetadataRecvComplete(vc, nreq, &complete);
            }

            nreq = psm_1sc_putacc_rndvrecv(req, inlen, &nreq,
                                    nreq->dev.user_buf, acpkt->rndv_tag,
                                    acpkt->mapped_srank, acpkt->rndv_len, vc);
            nreq->psm_flags |= PSM_GETACCUM_RNDV_REQ;

            nreq->resp_rndv_tag = acpkt->resp_rndv_tag;

            PRINT_DEBUG(DEBUG_1SC_verbose>1, "rndv_accum request. posted recv %p\n", nreq);
            goto end_2; 
        }
 
    }

    {
        _SECTION(GET_ACCUM_RESP);
        _SECTION(GET_ACCUM_RESP_IMMED);
        MPIDI_CH3_Pkt_get_accum_resp_t *acpkt = (MPIDI_CH3_Pkt_get_accum_resp_t *) pkt;

        GET_VC(vc, acpkt->target_win_handle, acpkt->source_rank);
        vc->ch.recv_active = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "get accum packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));

        goto end;          
    }

    {
        _SECTION(FOP);
        _SECTION(FOP_IMMED);
        MPIDI_CH3_Pkt_fop_t *foppkt = (MPIDI_CH3_Pkt_fop_t *) pkt;
        GET_VC(vc, foppkt->target_win_handle, foppkt->source_rank);
        vc->ch.recv_active = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "fop packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;           
    }

    {
        _SECTION(FOP_RESP);
        _SECTION(FOP_RESP_IMMED);
        temp_req = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "fop resp packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](NULL, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &temp_req);
        goto end;           
    }

    {
        _SECTION(CAS_IMMED);
        MPIDI_CH3_Pkt_cas_t *caspkt = (MPIDI_CH3_Pkt_cas_t *) pkt;
        GET_VC(vc, caspkt->target_win_handle, caspkt->source_rank);
        vc->ch.recv_active = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "cas packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;           
    }

    {
        _SECTION(CAS_RESP_IMMED);
        temp_req = req;
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "cas resp packet from %d\n", vc->pg_rank);
        psm_pkthndl[pkt->type](NULL, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &temp_req);
        goto end;           
    }

    /* handle lock */
    {
        _SECTION(LOCK);
        MPIDI_CH3_Pkt_lock_t *lockpkt = (MPIDI_CH3_Pkt_lock_t *) pkt;
        GET_VC(vc, lockpkt->target_win_handle, lockpkt->source_rank);
        vc->ch.recv_active = req;
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "lock request from [%d]\n", vc->pg_rank);
        goto end;
    }

    /* handle unlock */
    {
        _SECTION(UNLOCK);
        MPIDI_CH3_Pkt_unlock_t *unlockpkt = (MPIDI_CH3_Pkt_unlock_t *) pkt;
        GET_VC(vc, unlockpkt->target_win_handle, unlockpkt->source_rank);
        vc->ch.recv_active = req;
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "ulock request from [%d]\n", vc->pg_rank);
        goto end;
    }


    /* handle flush */
    {
        _SECTION(FLUSH);
        MPIDI_CH3_Pkt_flush_t *flushpkt = (MPIDI_CH3_Pkt_flush_t *) pkt;
        if (flushpkt->target_win_handle != MPI_WIN_NULL) { 
            GET_VC(vc, flushpkt->target_win_handle, flushpkt->target_rank);
        }
        temp_req = req;
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &temp_req);
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "flush request from [%d]\n", vc->pg_rank);
        goto end;
    }


    /* handle lock ack */
    {
        _SECTION(LOCK_ACK);
        MPIDI_CH3_Pkt_lock_ack_t *grpkt = (MPIDI_CH3_Pkt_lock_ack_t *) pkt;
        MPID_Win *win_ptr;
        MPID_Win_get_ptr(grpkt->source_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, grpkt->target_rank, &vc);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "granted lock\n");
        goto end;
    }


    /* handle ack */
    {
        _SECTION(ACK);
        MPIDI_CH3_Pkt_ack_t *grpkt = (MPIDI_CH3_Pkt_ack_t *) pkt;
        MPID_Win *win_ptr;
        MPID_Win_get_ptr(grpkt->source_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, grpkt->target_rank, &vc);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "ack packet\n");
        goto end;
    }
    /* handle decr_at_counter */
    {
        _SECTION(DECR_AT_COUNTER);
        MPIDI_CH3_Pkt_decr_at_counter_t *grpkt = (MPIDI_CH3_Pkt_decr_at_counter_t *) pkt;
        MPID_Win *win_ptr;
        MPID_Win_get_ptr(grpkt->source_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, grpkt->target_rank, &vc);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;
    }
    /* handle lock_op_ack */
    {
        _SECTION(LOCK_OP_ACK);
        MPIDI_CH3_Pkt_lock_op_ack_t *grpkt = (MPIDI_CH3_Pkt_lock_op_ack_t *) pkt;
        MPID_Win *win_ptr;
        MPID_Win_get_ptr(grpkt->source_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, grpkt->target_rank, &vc);
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
        goto end;
    }
errpkt:    
    fprintf(stderr, "Unknown packet type %d, name: %s\n", pkt->type, MPIDI_CH3_Pkt_type_to_string[pkt->type]);
    fprintf(stderr, "Request flags are %x\n", req->psm_flags);
    fprintf(stderr, "Length of message was %ld\n", inlen);
    fprintf(stderr, "I should not be here. Poof!\n");
    fflush(stderr);

end:    
    /* Reset req before re-posting */
    vbptr->req = req;

    /* re-post the vbuf */
    psm_1sided_recv(req, ptr);

end_2:    
    return MPI_SUCCESS;
}
#undef _SECTION
#undef __check
#undef GET_VC

/* a large request has completed */
int psm_complete_rndvrecv(MPID_Request *req, MPIDI_msg_sz_t inlen)
{
    /* the put pkt request was stored in tmpbuf */
    MPID_Request *putreq;
    MPIDI_CH3_Pkt_t *pkt;
    MPIDI_VC_t *vc;
    MPID_Win *win_ptr;
    vbuf *vbptr;
    MPIDI_msg_sz_t msg = req->pktlen;
    int complete = TRUE;

    /* if request counter is greater than 1, then the transaction is not finished;
     * cannot release the request yet */
    if (unlikely(MPID_cc_get(req->cc) > 1)) {
        return MPID_Request_complete(req);
    }

    putreq = req->savedreq;
    vbptr = putreq->vbufptr;
    pkt = (MPIDI_CH3_Pkt_t *) vbptr->buffer;
    if(req->psm_flags & PSM_RNDVRECV_PUT_REQ) {
        MPIDI_CH3_Pkt_put_t *putpkt;
        putpkt = (MPIDI_CH3_Pkt_put_t *) pkt;
        MPID_Win_get_ptr(putpkt->target_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, putpkt->source_rank, &vc);
        vc->ch.recv_active = putreq;
        if(req->psm_flags & PSM_RNDVRECV_NC_REQ) {
            /* we've received it to a pack-buf. Unpack it now */
            MPID_Request *treq = req->pending_req;
            psm_do_unpack(treq->dev.user_count, treq->dev.datatype, NULL,
                    req->pkbuf, req->pksz, treq->dev.user_buf, inlen);
            /* treq had dataloop et al free it now */
            MPID_Request_complete(treq);
            req->psm_flags |= PSM_PACK_BUF_FREE;
        }
        putreq->psm_flags |= PSM_RNDVPUT_COMPLETED;
        win_ptr->outstanding_rma--;
        psm_pkthndl[pkt->type](vc, pkt, (((char*)(pkt))+sizeof(MPIDI_CH3_Pkt_t)), &msg, &(vc->ch.recv_active));
    } else if(req->psm_flags & PSM_RNDVRECV_ACCUM_REQ) {
        MPIDI_CH3_Pkt_accum_t *acpkt;
        acpkt = (MPIDI_CH3_Pkt_accum_t *) pkt;
        MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, acpkt->source_rank, &vc);
        vc->ch.recv_active = req;
        if(req->psm_flags & PSM_RNDVRECV_NC_REQ) {
            MPIU_Memcpy(req->dev.user_buf, req->pkbuf, inlen);
            req->psm_flags |= PSM_PACK_BUF_FREE;
        }
        req->psm_flags |= PSM_RNDVPUT_COMPLETED;
        win_ptr->outstanding_rma--;
        MPIDI_CH3_ReqHandler_AccumRecvComplete(vc, req, &complete);
    } else if (req->psm_flags & PSM_GETACCUM_RNDV_REQ) {
        MPIDI_CH3_Pkt_get_accum_t *acpkt;
        acpkt = (MPIDI_CH3_Pkt_get_accum_t *) pkt;
        MPID_Win_get_ptr(acpkt->target_win_handle, win_ptr);
        MPIDI_Comm_get_vc(win_ptr->comm_ptr, acpkt->source_rank, &vc);
        vc->ch.recv_active = req;
        if(req->psm_flags & PSM_RNDVRECV_NC_REQ) {
            /* Copy the received data to the user_buf, it will be unpacked later */
            MPIU_Memcpy(req->dev.user_buf, req->pkbuf, inlen);
            req->psm_flags |= PSM_PACK_BUF_FREE;
            /* Copy the datatype info. if we use different req to perform RNDV transfer */
            MPID_Request *treq = req->pending_req;
            if (unlikely(treq != req)) {
                req->dev.user_count = treq->dev.user_count;
                req->dev.datatype = treq->dev.datatype;
                req->dev.datatype_ptr = treq->dev.datatype_ptr;
            }
        }
        req->psm_flags |= PSM_RNDVPUT_COMPLETED;
        win_ptr->outstanding_rma--;
        MPIDI_CH3_ReqHandler_GaccumRecvComplete(vc, req, &complete);
    }

    /* free the rndv request */
    req->psm_flags &= ~PSM_RNDVPUT_COMPLETED;
    req->savedreq = NULL;
    MPIU_Object_set_ref(req, 0);
    MPIDI_CH3_Request_destroy(req);

    /* Reset req before re-posting */
    vbptr->req = putreq;

    /* repost the original put-vbuf*/
    psm_1sided_recv(putreq, vbptr->buffer);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME psm_1sc_get_rndvrecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_1sc_get_rndvrecv(MPID_Request *savreq, MPIDI_CH3_Pkt_t *pkt, int from_rank)
{
    MPIDI_CH3_Pkt_get_t *getpkt = (MPIDI_CH3_Pkt_get_t *) pkt;
    MPID_Request *req;
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t rtag, rtagsel;
    #else
        uint64_t rtag, rtagsel;
    #endif
    PSM_ERROR_T psmerr;
    int mpi_errno = MPI_SUCCESS;

    req = psm_create_req();
    req->from_rank = getpkt->target_rank;
    req->kind = MPID_REQUEST_RECV;
    req->psm_flags |= PSM_RNDVRECV_GET_REQ;
    req->request_completed_cb = MPIDI_CH3_Req_handler_rma_op_complete;
    if (getpkt->flags & MPIDI_CH3_PKT_FLAG_RMA_LOCK_SHARED ||
        getpkt->flags & MPIDI_CH3_PKT_FLAG_RMA_LOCK_EXCLUSIVE) {
        req->is_piggyback = 1;
    }
    req->dev.source_win_handle = savreq->dev.source_win_handle;
    req->dev.rma_target_ptr = savreq->dev.rma_target_ptr;
    ++psm_tot_rndv_gets;
    
    /* save the get-request. GET is complete only when the original req is
     * completed. */
    req->savedreq = savreq;
    if(savreq->psm_flags & PSM_RNDVRECV_GET_PACKED) {
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "GET: origin needs unpack\n");
        req->psm_flags |= PSM_RNDVRECV_GET_PACKED;
        savreq->psm_flags &= ~PSM_RNDVRECV_GET_PACKED;
    }
    #if PSM_VERNO >= PSM_2_1_VERSION
        rtagsel.tag0 = MQ_TAGSEL_ALL;
        rtagsel.tag1 = MQ_TAGSEL_ALL;
        rtagsel.tag2 = MQ_TAGSEL_ALL;
    #else
        rtag = 0;
        rtagsel = MQ_TAGSEL_ALL;
    #endif
    MAKE_PSM_SELECTOR(rtag, MPID_CONTEXT_RNDVPSM, getpkt->rndv_tag,
                      from_rank);
    /* ch3u_rma_sync.c saved the origin_addr in dev.user_buf */
    _psm_enter_;
    if ((unlikely(getpkt->rndv_len > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_IRECV(savreq->dev.user_buf,
                        getpkt->rndv_len, &req, rtag, rtagsel);
    } else {
        psmerr = PSM_IRECV(psmdev_cw.mq, rtag, rtagsel, MQ_FLAGS_NONE,
                          savreq->dev.user_buf, getpkt->rndv_len, req, &(req->mqreq));
    }
    _psm_exit_;
    if(unlikely(psmerr != PSM_OK)) {
        mpi_errno = psm_map_error(psmerr);
        MPIR_ERR_POP(mpi_errno);
    }

fn_fail:
    return mpi_errno;    
}

static MPID_Request *psm_1sc_putacc_rndvrecv(MPID_Request *putreq, MPIDI_msg_sz_t putlen,
                     MPID_Request **nreq, void *useraddr, int rndv_tag, 
                     int source_rank, MPIDI_msg_sz_t rndv_len, MPIDI_VC_t *vc)
{
    MPID_Request *req = *nreq;
    MPID_Request *preq = vc->ch.recv_active;
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t rtag, rtagsel;
    #else
        uint64_t rtag, rtagsel;
    #endif
    PSM_ERROR_T psmerr;

    if(req == NULL) {
        req = psm_create_req();
        *nreq = req;
    }

    MPIU_Object_set_ref(req, 2);
    req->kind = MPID_REQUEST_RECV;
    req->savedreq = putreq;
    req->pktlen = putlen;

    #if PSM_VERNO >= PSM_2_1_VERSION
        rtagsel.tag0 = MQ_TAGSEL_ALL;
        rtagsel.tag1 = MQ_TAGSEL_ALL;
        rtagsel.tag2 = MQ_TAGSEL_ALL;
    #else
        rtag = 0;
        rtagsel = MQ_TAGSEL_ALL;
    #endif
    /* pkt->target_rank is my rank */
    MAKE_PSM_SELECTOR(rtag, MPID_CONTEXT_RNDVPSM, rndv_tag,
                      source_rank);

    /* if we're receiving non-contig addtitional processing needed */
    if(!MPIR_DATATYPE_IS_PREDEFINED(preq->dev.datatype)) {
        useraddr = psm_gen_packbuf(req, preq);
        rndv_len = req->pksz;
        req->psm_flags |= PSM_RNDVRECV_NC_REQ;
        req->psm_flags |= PSM_1SIDED_NON_CONTIG_REQ;
        /* we need the datatype info. keep the req pending */
        req->pending_req = preq;
    }
	 
    _psm_enter_;
    if ((unlikely(rndv_len > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_IRECV(useraddr, rndv_len,
                        &req, rtag, rtagsel);
    } else {
        psmerr = PSM_IRECV(psmdev_cw.mq, rtag, rtagsel, MQ_FLAGS_NONE,
                        useraddr, rndv_len, req, &(req->mqreq));
    }
    _psm_exit_;
    if(unlikely(psmerr != PSM_OK)) {
        printf("ERROR: rndv recv failed\n");
    }
    return req;
}

#undef FUNCNAME
#define FUNCNAME psm_send_1sided_ctrlpkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_send_1sided_ctrlpkt(MPID_Request **rptr, int dest, void *buf, 
                            MPIDI_msg_sz_t buflen, int src, int create_req)
{
    MPID_Request *req = *rptr;
    vbuf *vb;
    int mpi_errno = MPI_SUCCESS;
    PSM_ERROR_T psmerr;

    if(create_req) {
        req = psm_create_req();
        *rptr = req;
    }
    req->kind = MPID_REQUEST_SEND;
    req->psm_flags |= PSM_CONTROL_PKTREQ;
    
    vb = psm_get_vbuf();
    if(unlikely(vb == NULL)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_NO_MEM, "**nomem");
        goto fn_fail;
    }

    req->vbufptr = vb;
    vb->req = (void*) req;
    memcpy(vb->buffer, buf, buflen);

    psmerr = psm_iput(dest, vb->buffer, buflen, req, src);
    if(unlikely(psmerr != PSM_OK)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_INTERN, "**fail");
        goto fn_fail;
    }

fn_fail:    
    return mpi_errno;
}

/* if response is packed, unpack */

int psm_getresp_rndv_complete(MPID_Request *req, MPIDI_msg_sz_t inlen)
{
    int mpi_errno = MPI_SUCCESS;

    if(req->psm_flags & PSM_RNDVRECV_GET_PACKED) {
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "GET RDNV: did unpack\n");
        MPID_Request *savq = req->savedreq;
        psm_do_unpack(savq->dev.user_count, savq->dev.datatype, NULL, savq->dev.user_buf,
                0, savq->dev.real_user_buf, inlen);
        MPIU_Free(savq->dev.user_buf);

        if (((req->psm_flags & PSM_RNDVRECV_GET_REQ) && !req->is_piggyback) 
            || ((req->psm_flags & PSM_GETACCUM_GET_RNDV_REQ) && req->last_stream_unit
                 && !req->is_piggyback)) {
            MPID_Win *win_ptr;
            MPID_Win_get_ptr(req->dev.source_win_handle, win_ptr);
            mpi_errno = MPIDI_CH3I_RMA_Handle_ack(win_ptr, req->from_rank);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }

        /* complete the control request and decrease ref_count,
         * thus it can be freed in CH3. */
        MPID_Request_complete(savq);

        MPIU_Object_set_ref(req, 0);
        MPIDI_CH3_Request_destroy(req);
    } else {
        /* if request counter is greater than 1, then the transaction is not finished;
         * cannot release the request yet */
        if (unlikely(MPID_cc_get(req->cc) > 1)) {
            mpi_errno = MPID_Request_complete(req);
            goto fn_exit;
        }
        if (req->request_completed_cb != NULL) {
            mpi_errno = req->request_completed_cb(req);
            if (mpi_errno != MPI_SUCCESS) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        if (((req->psm_flags & PSM_RNDVRECV_GET_REQ) && !req->is_piggyback) 
            || ((req->psm_flags & PSM_GETACCUM_GET_RNDV_REQ) && req->last_stream_unit
                 && !req->is_piggyback)) {
            MPID_Win *win_ptr;
            MPID_Win_get_ptr(req->dev.source_win_handle, win_ptr);
            mpi_errno = MPIDI_CH3I_RMA_Handle_ack(win_ptr, req->from_rank);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }

        MPID_cc_set(req->savedreq->cc_ptr, 0);
	    MPID_Request_release(req->savedreq);
        MPIU_Object_set_ref(req, 0);
        MPIDI_CH3_Request_destroy(req);
    }
fn_exit:
    return MPI_SUCCESS;

fn_fail:
    PRINT_DEBUG(DEBUG_1SC_verbose>1, "request error\n");
    return mpi_errno;
}
/* get-response has completed. Call into receive handlers to decrement
 * my-counter */

int psm_getresp_complete(MPID_Request *req) 
{
    int complete = TRUE;
    MPIDI_VC_t *vc = (MPIDI_VC_t *) req->pkbuf;
    MPIDI_CH3_ReqHandler_GetSendComplete(vc, req, &complete);
    return MPI_SUCCESS;
}

int psm_fopresp_complete(MPID_Request *req) 
{
    int complete = TRUE;
    MPIDI_VC_t *vc = (MPIDI_VC_t *) req->pkbuf;
    MPIDI_CH3_ReqHandler_FOPSendComplete(vc, req, &complete);
    return MPI_SUCCESS;
}

int psm_getaccumresp_complete(MPID_Request *req) 
{
    int complete = TRUE;
    MPIDI_VC_t *vc = (MPIDI_VC_t *) req->pkbuf;
    MPIDI_CH3_ReqHandler_GaccumSendComplete(vc, req, &complete);
    return MPI_SUCCESS;
}

static void *psm_gen_packbuf(MPID_Request *rreq, MPID_Request *dtreq) 
{
    int typesize;

    MPID_Datatype_get_size_macro(dtreq->dev.datatype, typesize);
    rreq->pksz = dtreq->dev.user_count * typesize;
    
    rreq->pkbuf = MPIU_Malloc(rreq->pksz);
    return rreq->pkbuf;
}

static int gbl_rndv_tag;
static pthread_spinlock_t taglock;
static void psm_init_tag()
{
    pthread_spin_init(&taglock, 0);
    gbl_rndv_tag = 0;
}
/* get a new tag for rndv message sending. rndv target will wait on this tag */
int psm_get_rndvtag()
{
    int an_alarmingly_long_variable;
    pthread_spin_lock(&taglock);
    ++gbl_rndv_tag;
    an_alarmingly_long_variable = gbl_rndv_tag;
    pthread_spin_unlock(&taglock);
    return an_alarmingly_long_variable;
}
