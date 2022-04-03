/*!\file */
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

#ifndef IB_LMT_H
#define IB_LMT_H

#include "mpid_nem_impl.h"
#include "mpidi_ch3_pre.h"
#include "ib_hca.h"
#include "ib_vc.h"
#include "dreg.h"
#include <infiniband/verbs.h>

#define PARTIAL_COMPLETION 4

typedef enum {
    MV2_LMT_PROTOCOL_RENDEZVOUS_UNSPECIFIED = 0,
    MV2_LMT_PROTOCOL_EAGER,
    MV2_LMT_PROTOCOL_R3,
    MV2_LMT_PROTOCOL_RPUT,
    MV2_LMT_PROTOCOL_RGET,
} MPID_nem_ib_rndv_protocol_t;


typedef struct{
#if defined(MPID_USE_SEQUENCE_NUMBERS)
    MPID_Seqnum_t seqnum;
#endif /* defined(MPID_USE_SEQUENCE_NUMBERS) */
    MPIDI_msg_sz_t recv_sz;

   /* MPIDI_CH3I_MRAILI_IBA_PKT_DECL (new MPIDI_nem_ib_pkt_comm_header_t)*/
    uint8_t  vbuf_credit;
    uint8_t  remote_credit;
    uint8_t  rdma_credit;
    uint8_t  rail;
    union {
        uint32_t smp_index;
        uint64_t vc_addr;
    } src;
#ifdef CRC_CHECK
    unsigned long crc;
#endif

    /* rndv (MPIDI_CH3I_MRAILI_RNDV_INFO_DECL) in original rts,cts pkt, now used as cookie */
    MPID_nem_ib_rndv_protocol_t  protocol;
    void                *buf_addr;
    uint32_t            rkey[MAX_NUM_HCAS];
} MPID_nem_ib_lmt_cookie;   /*used like pkt */


typedef struct{
    MPI_Request partner_id;
    uint8_t rndv_buf_alloc;
    /*why do we need to store buf and size here though we have already store it
    inside request*/
    void * rndv_buf;
    MPIDI_msg_sz_t rndv_buf_sz;
    MPIDI_msg_sz_t rndv_buf_off;
    MPID_nem_ib_rndv_protocol_t protocol;
    /*temporary holder for registered memory*/
    struct dreg_entry *d_entry;
    void     *remote_addr;
    uint32_t rkey[MAX_NUM_HCAS];
    uint8_t  nearly_complete;
    uint8_t  num_rdma_read_completions;
    uint32_t completion_counter;
    /*hold the next request in flowlist*/
    void *next_inflow;
} __attribute__((__packed__)) MPID_nem_ib_rndv_info;


#define REQ_FIELD(req, field) (((MPID_nem_ib_rndv_info *)(&(req)->ch.netmod_area))->field)

#define MPIDI_NEM_RNDV_SET_REQ_INFO(rreq, rndv_info) \
{                                                   \
    REQ_FIELD(rreq, protocol) = rndv_info->protocol;  \
}

#define MPIDI_NEM_RNDV_PROTOCOL_IS_READ(rreq) \
    (MV2_LMT_PROTOCOL_RGET == REQ_FIELD(rreq, protocol))


#define MPIDI_nem_ib_lmt_cookie_set_seqnum(rndv_info_, seqnum_)  \
{                                           \
  (rndv_info_)->seqnum = (seqnum_);             \
}   

#define MPIDI_NEM_IB_RREQ_RNDV_FINISH(rreq)                         \
{                                                                    \
    if (rreq != NULL) {                                                 \
        if (REQ_FIELD(rreq, d_entry) != NULL) {                      \
            dreg_unregister(REQ_FIELD(rreq, d_entry));       \
            REQ_FIELD(rreq, d_entry) = NULL;                   \
        }                                                       \
        if (1 == REQ_FIELD(rreq, rndv_buf_alloc)                \
            && REQ_FIELD(rreq, rndv_buf) != NULL) {             \
            MPIU_Free(REQ_FIELD(rreq, rndv_buf));                    \
            REQ_FIELD(rreq, rndv_buf) = NULL;                        \
            REQ_FIELD(rreq, rndv_buf_off) = REQ_FIELD(rreq, rndv_buf_sz) = 0; \
            REQ_FIELD(rreq, rndv_buf_alloc) = 0;                     \
        }  else {                                               \
            REQ_FIELD(rreq, rndv_buf_off) = REQ_FIELD(rreq, rndv_buf_sz) = 0; \
        }                                                       \
        REQ_FIELD(rreq, protocol) = MV2_LMT_PROTOCOL_RENDEZVOUS_UNSPECIFIED; \
    }                                                           \
}

int MPID_nem_lmt_ib_initiate_lmt(MPIDI_VC_t *vc, MPIDI_CH3_Pkt_t *rts_pkt,
                                           struct MPID_Request *req);
int MPID_nem_lmt_ib_start_send(struct MPIDI_VC *vc, struct MPID_Request *sreq,
                                           MPL_IOV r_cookie);
int MPID_nem_ib_lmt_start_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq, MPL_IOV s_cookie);
int MPID_nem_ib_lmt_done_recv(struct MPIDI_VC *VC, struct MPID_Request *rreq);
int MPIDI_nem_ib_lmt_r3_recv_data(MPIDI_VC_t * vc, vbuf * buffer);
void MPIDI_nem_ib_lmt_r3_recv_ack(MPIDI_VC_t * vc,
                               void* vstart);
void MPID_nem_lmt_ib_process_rndv(void);
#endif
