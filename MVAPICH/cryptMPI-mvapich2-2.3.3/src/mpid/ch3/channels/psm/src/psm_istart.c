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

/* 
   iov[0] contains the packet information which includes tag/destrank/context_id
   to use. Extract iov[0], cast it to a packet type and use it */

#define PSM_ESTABLISH_CONNECTION(peer)          \
do {                                            \
    if(unlikely(!PSM_ADDR_RESOLVED(peer))) {    \
        mpi_errno = psm_connect_peer(peer);     \
        if (mpi_errno != MPI_SUCCESS) {         \
            MPIR_ERR_POP(mpi_errno);            \
        }                                       \
    }                                           \
} while(0)

#undef FUNCNAME
#define FUNCNAME psm_istartmsgv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_istartmsgv(MPIDI_VC_t *vc, MPL_IOV *iov, int iov_n, MPID_Request **rptr)
{
    MPIDI_CH3_Pkt_t *genpkt;
    void *buf;
    int psmerr, mpi_errno = MPI_SUCCESS;
    MPIDI_msg_sz_t buflen;
    
    PSM_ESTABLISH_CONNECTION(vc->pg_rank);
    assert(iov_n > 0);
    genpkt = (MPIDI_CH3_Pkt_t *) iov[0].MPL_IOV_BUF;

    switch(genpkt->type) {
        case MPIDI_CH3_PKT_PUT: 
        {
            /* packet handlers expect generic packet size */
            MPIDI_CH3_Pkt_put_t *putpkt = (MPIDI_CH3_Pkt_put_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "mpi put to %d\n", putpkt->mapped_trank);
            mpi_errno = psm_1sided_putpkt(putpkt, iov, iov_n, rptr); //ODOT: err
            goto fn_exit;
        }
        case MPIDI_CH3_PKT_ACCUMULATE: {
            MPIDI_CH3_Pkt_accum_t *acpkt = (MPIDI_CH3_Pkt_accum_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "mpi accum to %d\n", acpkt->mapped_trank);
            mpi_errno = psm_1sided_accumpkt(acpkt, iov, iov_n, rptr); //ODOT: error handle
            goto fn_exit;
        }
        case MPIDI_CH3_PKT_GET_ACCUM: {
            MPIDI_CH3_Pkt_get_accum_t *acpkt = (MPIDI_CH3_Pkt_get_accum_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "mpi get_accum to %d\n", acpkt->mapped_trank);
            mpi_errno = psm_1sided_getaccumpkt(acpkt, iov, iov_n, rptr); //ODOT: error handle
            goto fn_exit;
        }
        case MPIDI_CH3_PKT_GET_ACCUM_RESP: 
        case MPIDI_CH3_PKT_GET_ACCUM_RESP_IMMED: 
        {
            MPIDI_CH3_Pkt_get_accum_resp_t *acpkt = (MPIDI_CH3_Pkt_get_accum_resp_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "mpi get_accum to %d\n", acpkt->mapped_trank);
            mpi_errno = psm_1sided_getaccumresppkt(acpkt, iov, iov_n, rptr); //ODOT: error handle
            goto fn_exit;
        }
        case MPIDI_CH3_PKT_GET_RESP: 
        case MPIDI_CH3_PKT_GET_RESP_IMMED:
        {
            MPIDI_CH3_Pkt_get_resp_t *resppkt = (MPIDI_CH3_Pkt_get_resp_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "mpi response for get from %d\n", resppkt->mapped_trank);
            (*rptr)->pkbuf = vc;
            mpi_errno = psm_1sided_getresppkt(resppkt, iov, iov_n, rptr);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            goto fn_exit;
        }
        case MPIDI_CH3_PKT_GET: {
            MPIDI_CH3_Pkt_get_t *getpkt = (MPIDI_CH3_Pkt_get_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "mpi get to %d\n", getpkt->mapped_trank);
            mpi_errno = psm_1sided_getpkt(getpkt, iov, iov_n, rptr);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            if(getpkt->rndv_mode) {
                mpi_errno = psm_1sc_get_rndvrecv((*rptr), (MPIDI_CH3_Pkt_t *)getpkt,
                        getpkt->mapped_trank);
            }
            goto fn_exit;
        }

        case MPIDI_CH3_PKT_FOP:{
            MPIDI_CH3_Pkt_t * pkt = (MPIDI_CH3_Pkt_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            mpi_errno = psm_1sided_atomicpkt(pkt, iov, iov_n,
                    pkt->fop.mapped_trank, pkt->fop.mapped_srank, rptr);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            goto fn_exit;
         }
        case MPIDI_CH3_PKT_FOP_RESP: {
            MPIDI_CH3_Pkt_t * pkt = (MPIDI_CH3_Pkt_t *) genpkt;
            iov[0].MPL_IOV_LEN = sizeof(MPIDI_CH3_Pkt_t);
            (*rptr)->pkbuf = vc;
            mpi_errno = psm_1sided_atomicpkt(pkt, iov, iov_n,
                    pkt->fop_resp.mapped_trank, pkt->fop_resp.mapped_srank, rptr);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            goto fn_exit;
        }
        default: {
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "sending packet type %d\n", genpkt->type);
            if(genpkt->type != MPIDI_CH3_PKT_EAGER_SEND) {
                PSM_ERR_ABORT("unknown control packet type %d, name: %s\n", genpkt->type, MPIDI_CH3_Pkt_type_to_string[genpkt->type]);
            }                
            assert(genpkt->type == MPIDI_CH3_PKT_EAGER_SEND);                                     
            break;                                        
        }
    }

    /* 2-sided packet handling */
    MPIDI_CH3_Pkt_send_t *pkt;

    pkt = (MPIDI_CH3_Pkt_send_t *) iov[0].MPL_IOV_BUF;
    buf = (void *) iov[1].MPL_IOV_BUF;
    buflen = iov[1].MPL_IOV_LEN;
    psmerr = psm_send_pkt(rptr, pkt->match, vc->pg_rank, buf, buflen);
    if(unlikely(psmerr != PSM_OK)) {
        mpi_errno = psm_map_error(psmerr);
    }

fn_exit:
    if(unlikely(mpi_errno != MPI_SUCCESS)) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_INTERN, "**fail");        
    }

fn_fail:    
    return mpi_errno;

}

#undef FUNCNAME
#define FUNCNAME psm_istartmsg
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_istartmsg(MPIDI_VC_t *vc, void *upkt, MPIDI_msg_sz_t pkt_sz, MPID_Request **rptr)
{
    MPIDI_CH3_Pkt_t* genpkt = upkt;
    void *buf;
    int psmerr, src, trank = -1;
    MPIDI_msg_sz_t buflen;
    int mpi_errno = MPI_SUCCESS;

    PSM_ESTABLISH_CONNECTION(vc->pg_rank);
    buf = upkt;
    buflen = pkt_sz;

    switch(genpkt->type) {
        case MPIDI_CH3_PKT_GET:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_get_t *) genpkt)->mapped_srank;
            (*rptr)->psm_flags |= PSM_GETPKT_REQ;
            trank = ((MPIDI_CH3_Pkt_get_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 0);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            if(((MPIDI_CH3_Pkt_get_t *) genpkt)->rndv_mode) {
                mpi_errno = psm_1sc_get_rndvrecv((*rptr), genpkt, trank);
            } else {
                ++psm_tot_eager_gets;
            }
            break;

        case MPIDI_CH3_PKT_PUT_IMMED:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            ((MPIDI_CH3_Pkt_put_t *) genpkt)->rndv_mode = 0;
            src = ((MPIDI_CH3_Pkt_put_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_put_t *) genpkt)->mapped_trank;
            ++psm_tot_eager_puts;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_ACCUMULATE_IMMED: 
            buflen = sizeof(MPIDI_CH3_Pkt_t);
             ((MPIDI_CH3_Pkt_accum_t *) genpkt)->rndv_mode = 0;
            src = ((MPIDI_CH3_Pkt_accum_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_accum_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_GET_ACCUM_IMMED: 
            buflen = sizeof(MPIDI_CH3_Pkt_t);
             ((MPIDI_CH3_Pkt_get_accum_t *) genpkt)->rndv_mode = 0;
            src = ((MPIDI_CH3_Pkt_get_accum_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_get_accum_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_FOP_IMMED:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_fop_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_fop_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_CAS_IMMED: 
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_cas_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_cas_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_CAS_RESP_IMMED:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_cas_resp_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_cas_resp_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_FOP_RESP: 
        case MPIDI_CH3_PKT_FOP_RESP_IMMED:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_fop_resp_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_fop_resp_t *) genpkt)->mapped_trank;
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_LOCK:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_lock_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_lock_t *) genpkt)->mapped_trank;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "Sending LOCK packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_LOCK_ACK:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_lock_ack_t *) genpkt)->mapped_srank;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "Sending LOCK ACK packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, vc->pg_rank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_ACK:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_ack_t *) genpkt)->mapped_srank;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "Sending ACK packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, vc->pg_rank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_UNLOCK:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_unlock_t *) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_unlock_t *) genpkt)->mapped_trank;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "Sending UNLOCK packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_FLUSH:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_flush_t *) genpkt)->mapped_srank;
            PRINT_DEBUG(DEBUG_1SC_verbose>1, "Sending FLUSH packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, vc->pg_rank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        case MPIDI_CH3_PKT_DECR_AT_COUNTER:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_decr_at_counter_t*) genpkt)->mapped_srank;
            trank = ((MPIDI_CH3_Pkt_decr_at_counter_t *) genpkt)->mapped_trank;
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "Sending DECR_AT_COUNTER packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;
        
        case MPIDI_CH3_PKT_LOCK_OP_ACK:
            buflen = sizeof(MPIDI_CH3_Pkt_t);
            src = ((MPIDI_CH3_Pkt_lock_op_ack_t*) genpkt)->target_rank;
            trank = ((MPIDI_CH3_Pkt_lock_op_ack_t *) genpkt)->mapped_trank;
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "Sending LOCK_OP_ACK  packet to %d from %d\n", trank, src);
            mpi_errno = psm_send_1sided_ctrlpkt(rptr, trank, buf, buflen, src, 1);
            if(unlikely(mpi_errno != MPI_SUCCESS)) {
                MPIR_ERR_POP(mpi_errno);
            }
            break;

        default: {
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "sending packet type %d\n", genpkt->type);
            if(genpkt->type != MPIDI_CH3_PKT_EAGER_SEND) {
                PSM_ERR_ABORT("unknown control packet type %d, name: %s\n", genpkt->type, MPIDI_CH3_Pkt_type_to_string[genpkt->type]);
            }                
            assert(genpkt->type == MPIDI_CH3_PKT_EAGER_SEND);                                     

            MPIDI_CH3_Pkt_send_t *pkt = (MPIDI_CH3_Pkt_send_t *) genpkt;
            psmerr = psm_send_pkt(rptr, pkt->match, vc->pg_rank, buf, buflen);
            if(unlikely(psmerr != PSM_OK)) {
                mpi_errno = psm_map_error(psmerr);
            }
            break;
        }
    }
    
fn_exit:
    if(unlikely(mpi_errno != MPI_SUCCESS)) {
        MPIR_ERR_POP(mpi_errno);
    }
    return mpi_errno;
fn_fail:    
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME psm_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_send(MPIDI_VC_t *vc, MPIDI_Message_match match, MPID_Request *sreq)
{
    int psmerr;
    int mpi_errno = MPI_SUCCESS;

    PRINT_DEBUG(DEBUG_CHM_verbose>1, "simple PSM_Send issued\n");
    PSM_ESTABLISH_CONNECTION(vc->pg_rank);
    psmerr = psm_send_pkt(&sreq, match, vc->pg_rank, sreq->pkbuf, sreq->pksz);
    if(unlikely(psmerr != PSM_OK)) {
        mpi_errno = psm_map_error(psmerr);
        MPIR_ERR_POP(mpi_errno);
    }

fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_isend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_isend(MPIDI_VC_t *vc, MPIDI_Message_match match, MPID_Request *sreq)
{
    int psmerr;
    int mpi_errno = MPI_SUCCESS;

    PRINT_DEBUG(DEBUG_CHM_verbose>1, "simple non-blocking PSM_Send issued\n");
    PSM_ESTABLISH_CONNECTION(vc->pg_rank);
    
    psmerr = psm_isend_pkt(sreq, match, vc->pg_rank, sreq->pkbuf, 
                           sreq->pksz);
    if(unlikely(psmerr != PSM_OK)) {
        mpi_errno = psm_map_error(psmerr);
        MPIR_ERR_POP(mpi_errno);
    }

fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_isendv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_isendv(MPIDI_VC_t *vc, MPL_IOV *iov, int iov_n, MPID_Request *rptr)
{
    MPIDI_CH3_Pkt_send_t *pkt;
    void *buf;
    int psmerr, mpi_errno = MPI_SUCCESS;
    MPIDI_msg_sz_t buflen;

    assert(iov_n > 0);
    pkt = (MPIDI_CH3_Pkt_send_t *) iov[0].MPL_IOV_BUF;
    buf = (void *) iov[1].MPL_IOV_BUF;
    buflen = iov[1].MPL_IOV_LEN;
    PSM_ESTABLISH_CONNECTION(vc->pg_rank);

    psmerr = psm_isend_pkt(rptr, pkt->match, vc->pg_rank, buf, buflen);
    if(unlikely(psmerr != PSM_OK)) {
        mpi_errno = psm_map_error(psmerr);
        MPIR_ERR_POP(mpi_errno);
    }

fn_fail:    
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME psm_send_noncontig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_send_noncontig(MPIDI_VC_t *vc, MPID_Request *sreq, 
                       MPIDI_Message_match match)
{
    int mpi_errno, inuse;

    if(sreq->psm_flags & PSM_NON_BLOCKING_SEND) {
        mpi_errno = psm_isend(vc, match, sreq);
        if(unlikely(mpi_errno != MPI_SUCCESS)) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    else {
        mpi_errno = psm_send(vc, match, sreq);
        if(unlikely(mpi_errno != MPI_SUCCESS)) {
            MPIR_ERR_POP(mpi_errno);
        }

        if(!(sreq->psm_flags & PSM_NON_BLOCKING_SEND)) {
            MPID_cc_set(sreq->cc_ptr, 0);
            MPIU_Object_release_ref(sreq, &inuse);
        }
    }

fn_fail:
    return mpi_errno;
}
#undef FUNCNAME
#define FUNCNAME psm_map_error
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_map_error(PSM_ERROR_T psmerr)
{
    if(psmerr == PSM_OK) {
        return MPI_SUCCESS;
    }

    fprintf(stderr, "psm error: %s\n", PSM_ERROR_GET_STRING(psmerr));
    fflush(stderr);
    return MPI_ERR_INTERN;
}
