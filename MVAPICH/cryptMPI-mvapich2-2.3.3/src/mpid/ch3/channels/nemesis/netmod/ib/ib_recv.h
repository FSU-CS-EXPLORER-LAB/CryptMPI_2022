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

#ifndef IB_RECV_H
#define IB_RECV_H

#include "ib_vbuf.h"
#include "ib_vc.h"
#include "ib_errors.h"
int MPIDI_nem_ib_parse_header(MPIDI_VC_t * vc,
                                  vbuf * v, void **pkt, int *header_size);
int MPIDI_nem_ib_fill_request(MPID_Request * req, vbuf * v,
                                  int header_size, int *nb);
void MPIDI_nem_ib_release_vbuf(vbuf * v);
int MPIDI_nem_ib_recv_addr(MPIDI_VC_t * vc, void *vstart);
int MPIDI_nem_ib_recv_addr_reply(MPIDI_VC_t * vc, void *vstart);
int MPIDI_nem_ib_packetized_recv_data(MPIDI_VC_t * vc, vbuf *v);
int MPIDI_nem_ib_packetized_recv_req(MPIDI_VC_t * vc, MPID_Request * rreq);

#define IBV_POST_RR(_c,_vbuf,_rail) {                           \
    int __ret;                                                  \
    _vbuf->vc = (void *)_c;                                     \
    VC_FIELD((_c), connection)->rails[(_rail)].used_recv_cq++;  \
    __ret = ibv_post_recv(VC_FIELD((_c), connection)->rails[(_rail)].qp_hndl,   \
                          &((_vbuf)->desc.u.rr),                  \
            &((_vbuf)->desc.y.bad_rr));                           \
    if (__ret) {                                                \
        ibv_error_abort(-1,                      \
            "ibv_post_recv err");          \
    }                                                           \
}

/* was in rdma_impl.h */
#define PREPOST_VBUF_RECV(_c, _subrail)  {                      \
    vbuf *__v = get_vbuf();                                     \
    vbuf_init_recv(__v, VBUF_BUFFER_SIZE, _subrail);            \
    IBV_POST_RR(_c, __v, (_subrail));                           \
    VC_FIELD((_c), connection)->srp.credits[(_subrail)].local_credit++;\
    VC_FIELD((_c), connection)->srp.credits[(_subrail)].preposts++;\
}

#endif
