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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "psmpriv.h"
#include "mpidpre.h"

/* send packet: if Ssend call, add MQ flag.
                if MT issue isend and return with flag 
                if ST do blocking send, update cc_ptr
*/

#undef FUNCNAME
#define FUNCNAME psm_large_msg_isend_pkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
#if PSM_VERNO >= PSM_2_1_VERSION
    PSM_ERROR_T psm_large_msg_isend_pkt(MPID_Request **rptr, int dest, void *buf, MPIDI_msg_sz_t buflen, psm2_mq_tag_t *stag, uint32_t flags)
#else
    PSM_ERROR_T psm_large_msg_isend_pkt(MPID_Request **rptr, int dest, void *buf, MPIDI_msg_sz_t buflen, uint64_t stag, uint32_t flags)
#endif
{
    PSM_ERROR_T psmerr;
    MPID_Request *req = *rptr;
    int i = 0, steps = 0, balance = 0;
    int obj_ref = 0, cc_cnt = 0;

    /* Compute the number of chunks */
    steps = buflen / ipath_max_transfer_size;
    balance = buflen % ipath_max_transfer_size;

    PRINT_DEBUG(DEBUG_CHM_verbose>1,
            "PSM large send, buflen: %llu, max_size: %llu, steps: %d, balance: %d\n",
           (long long unsigned int)buflen, (long long unsigned int)ipath_max_transfer_size, steps, balance);

    /* Sanity check */
    MPIU_Assert(steps > 0);
    MPIU_Assert(balance >= 0);

    /* Get current object reference count and completion count */
    cc_cnt  = *(req->cc_ptr);
    obj_ref = MPIU_Object_get_ref(req);

    /* Increment obj ref count and comp count by number of chunks */
    cc_cnt  += steps - (balance == 0 ? 1 : 0);
    obj_ref += steps - (balance == 0 ? 1 : 0);

    /* Update object reference count and completion count */
    MPID_cc_set(req->cc_ptr, cc_cnt);
    MPIU_Object_set_ref(req, obj_ref);

    for (i = 0; i < steps; i++) {
        psmerr = PSM_ISEND_PTR(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                    flags, stag, buf, ipath_max_transfer_size, req, &(req->mqreq));
        buf += ipath_max_transfer_size;
    }
    if (balance) {
        psmerr = PSM_ISEND_PTR(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                    flags, stag, buf, balance, req, &(req->mqreq));
    }

    return psmerr;
}

#undef FUNCNAME
#define FUNCNAME psm_send_pkt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
PSM_ERROR_T psm_send_pkt(MPID_Request **rptr, MPIDI_Message_match m,
                 int dest, void *buf, MPIDI_msg_sz_t buflen)
{
    PSM_ERROR_T psmerr;
    uint32_t flags = MQ_FLAGS_NONE;
    MPID_Request *req = *rptr;
    uint8_t blocking = 1;
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t stag;
    #else
        uint64_t stag = 0;
    #endif

    MAKE_PSM_SELECTOR(stag, m.parts.context_id, m.parts.tag, m.parts.rank);
    if(req && req->psm_flags & PSM_SYNC_SEND) {
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "sync send psm\n");
        flags = PSM_MQ_FLAG_SENDSYNC;
        blocking = 0;
    }

    #if PSM_VERNO >= PSM_2_1_VERSION
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm2_mq_send: ctx = %d tag = %d\n", m.parts.context_id, m.parts.tag);
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm2_mq_send: dst = %d src = %d\n", dest, m.parts.rank);
    #else
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm_mq_send: ctx = %d tag = %d\n", m.parts.context_id, m.parts.tag);
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm_mq_send: dst = %d src = %d\n", dest, m.parts.rank);
    #endif

    if(blocking && !CAN_BLK_PSM(buflen))
        blocking = 0;

    if((blocking) && (likely(buflen <= ipath_max_transfer_size))){
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "blocking send\n");
        _psm_enter_;
        psmerr = PSM_SEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                flags, stag, buf, buflen);
        _psm_exit_;
        if(req) {
            MPID_cc_set(req->cc_ptr, 0);
        }
    } else {
        if(!req) {
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm_send_pkt created new req\n");
            req = psm_create_req();
            req->kind = MPID_REQUEST_SEND;
            *rptr = req;
        }

        req->psm_flags |= PSM_NON_BLOCKING_SEND;
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "nb send posted for blocking mpi_send\n");
        _psm_enter_;
        if ((unlikely(buflen > ipath_max_transfer_size))) {
            psmerr = PSM_LARGE_ISEND(rptr, dest, buf, buflen, stag, flags);
        } else {
            psmerr = PSM_ISEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                        flags, stag, buf, buflen, req, &(req->mqreq));
        }
        _psm_exit_;
        ++psm_tot_sends;
    }

    return psmerr;
}

/* isend:
        if issend, append MQ flag
        issue isend 
*/

PSM_ERROR_T psm_isend_pkt(MPID_Request *req, MPIDI_Message_match m,
                  int dest, void *buf, MPIDI_msg_sz_t buflen)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t stag;
    #else
        uint64_t stag = 0;
    #endif
    uint32_t flags = MQ_FLAGS_NONE;
    PSM_ERROR_T psmerr;

    MAKE_PSM_SELECTOR(stag, m.parts.context_id, m.parts.tag, m.parts.rank);
    assert(req);
    if(req->psm_flags & PSM_SYNC_SEND) {
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "sync Isend psm\n");
        flags = PSM_MQ_FLAG_SENDSYNC;
    }

    assert(dest < psmdev_cw.pg_size);
    #if PSM_VERNO >= PSM_2_1_VERSION
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm2_mq_isend: ctx = %d tag = %d\n", m.parts.context_id, m.parts.tag);
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm2_mq_isend: dst = %d src = %d\n", dest, m.parts.rank);
    #else
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm_mq_isend: ctx = %d tag = %d\n", m.parts.context_id, m.parts.tag);
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "psm_mq_isend: dst = %d src = %d\n", dest, m.parts.rank);
    #endif

    _psm_enter_;
    if ((unlikely(buflen > ipath_max_transfer_size))) {
        psmerr = PSM_LARGE_ISEND(&req, dest, buf, buflen, stag, flags);
    } else {
        psmerr = PSM_ISEND(psmdev_cw.mq, psmdev_cw.epaddrs[dest],
                    flags, stag, buf, buflen, req, &(req->mqreq));
    }
    _psm_exit_;
    ++psm_tot_sends;
    return psmerr;
}

/* create a new MPID_Request */

MPID_Request * psm_create_req()
{
    MPID_Request *req = MPID_Request_create();
    MPIU_Object_set_ref(req, 2);
    return req;
}
