/*! \file */
/*
 *  (C) 2006 by Argonne National Laboratory.
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

#define _GNU_SOURCE
#include "mpiimpl.h"
#include "mpidimpl.h"
#include "upmi.h"
#include "ib_device.h"
#include "ib_vc.h"
#include "ib_send.h"
#include "ib_recv.h"
/* TODO: move srq to a independent structure in order to get 
 * rid of include ib_process.h here*/
#include "ib_process.h"
#include "ib_lmt.h"

#define PKTARRAY_SIZE (MPIDI_CH3_PKT_END_ALL+1)
#ifdef ENABLE_CHECKPOINTING
static MPIDI_CH3_PktHandler_Fcn *pktArray[PKTARRAY_SIZE];
#endif

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_vc_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * Initialize the virtual channel.
 * The connections should be initialized before this function.
 *
 * \param vc an already instantiated Virtual Channel structure.
 */
int MPID_nem_ib_vc_init (MPIDI_VC_t *vc )
        /*const char *business_card)*/
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_CH3I_VC *vc_ch = &vc->ch;
    int pg_size;
    int i;
    UPMI_GET_SIZE(&pg_size);

    MPIDI_STATE_DECL(MPID_STATE_MPID_NEM_IB_VC_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPID_NEM_IB_VC_INIT);

    /* override nocontig send function */
    vc->sendNoncontig_fn = MPID_nem_ib_iSendNoncontig;

    vc_ch->iStartContigMsg    = MPID_nem_ib_iStartContigMsg;
    vc_ch->iSendContig        = MPID_nem_ib_iSendContig;
#ifdef ENABLE_CHECKPOINTING
    vc_ch->ckpt_pause_send_vc = MPID_nem_ib_ckpt_pause_send_vc;
    vc_ch->ckpt_continue_vc   = MPID_nem_ib_ckpt_continue_vc;
    vc_ch->ckpt_restart_vc    = MPID_nem_ib_ckpt_restart_vc;
    
    pktArray[MPIDI_NEM_IB_PKT_UNPAUSE] = MPID_nem_ib_pkt_unpause_handler;
#endif


    /*
    int ud_qpn, dlid, out_len, i;
    struct ibv_ah_attr attr;
    uint64_t guid;
    */


    VC_FIELD(vc, qp) = NULL;
    VC_FIELD(vc, free_vc) = 0;
    VC_FIELD(vc, in_queue) = 0;
    VC_FIELD(vc, seqnum_recv) = 0;
    VC_FIELD(vc, seqnum_send) = 0;
    VC_FIELD(vc, pending_r3_data) = 0;
    VC_FIELD(vc, received_r3_data) = 0;

    VC_FIELD(vc, conn_status) = MPID_NEM_IB_CONN_NONE;


    VC_FIELD(vc, connection) = &conn_info.connections[vc->pg_rank];
    VC_FIELD(vc, cmanager) = &cmanagers[vc->pg_rank];
    VC_FIELD(vc, cmanager)->vc = (void *)vc;
    VC_FIELD(vc, force_rndv) = 0;

    VC_FIELD(vc, send_queue).head = VC_FIELD(vc, send_queue).tail = NULL;

    /* MRAILI_Init_vc is added here */
    VC_FIELD(vc, connection)->rfp.phead_RDMA_send = 0;
    VC_FIELD(vc, connection)->rfp.ptail_RDMA_send = 0;
    VC_FIELD(vc, connection)->rfp.p_RDMA_recv = 0;
    VC_FIELD(vc, connection)->rfp.p_RDMA_recv_tail = 0;
    VC_FIELD(vc, connection)->rfp.rdma_failed = 0;

    VC_FIELD(vc, connection)->next_packet_expected  = 0;
    VC_FIELD(vc, connection)->next_packet_tosend    = 0;
    VC_FIELD(vc, connection)->outstanding_eager_vbufs = 0;
    VC_FIELD(vc, connection)->coalesce_vbuf         = NULL;
    
    VC_FIELD(vc, connection)->rfp.rdma_credit = 0;
#ifndef MV2_DISABLE_HEADER_CACHING 
    VC_FIELD(vc, connection)->rfp.cached_miss   = 0;
    VC_FIELD(vc, connection)->rfp.cached_hit    = 0;
    VC_FIELD(vc, connection)->rfp.cached_incoming = MPIU_Malloc (sizeof(MPIDI_CH3_Pkt_send_t));
    VC_FIELD(vc, connection)->rfp.cached_outgoing = MPIU_Malloc (sizeof(MPIDI_CH3_Pkt_send_t));
    VC_FIELD(vc, connection)->rfp.cached_incoming_iheader = MPIU_Malloc (sizeof(MPIDI_nem_ib_pkt_comm_header));
    VC_FIELD(vc, connection)->rfp.cached_outgoing_iheader = MPIU_Malloc (sizeof(MPIDI_nem_ib_pkt_comm_header));
    memset(VC_FIELD(vc, connection)->rfp.cached_outgoing, 0, sizeof(MPIDI_CH3_Pkt_send_t));
    memset(VC_FIELD(vc, connection)->rfp.cached_incoming, 0, sizeof(MPIDI_CH3_Pkt_send_t));
    memset(VC_FIELD(vc, connection)->rfp.cached_outgoing_iheader, 0, sizeof(MPIDI_nem_ib_pkt_comm_header));
    memset(VC_FIELD(vc, connection)->rfp.cached_incoming_iheader, 0, sizeof(MPIDI_nem_ib_pkt_comm_header));
#endif

    if (pg_size < rdma_eager_limit)
        VC_FIELD(vc, connection)->rfp.eager_start_cnt = rdma_polling_set_threshold + 1;
    else
        VC_FIELD(vc, connection)->rfp.eager_start_cnt = 0;

    VC_FIELD(vc, connection)->rfp.in_polling_set = 0;

    VC_FIELD(vc, connection)->sreq_head = NULL;
    VC_FIELD(vc, connection)->sreq_tail = NULL;
    VC_FIELD(vc, connection)->nextflow  = NULL;
    VC_FIELD(vc, connection)->inflow    = 0;

    /* PREPOST_VBUF_RECV previously in MPIDI_MRAIL_Vc */
    for(i = 0; i < rdma_num_rails; i++) {
        int hca_index = i / (rdma_num_rails / ib_hca_num_hcas);
        VC_FIELD(vc, connection)->rails[i].send_wqes_avail    = rdma_default_max_send_wqe;
        VC_FIELD(vc, connection)->rails[i].ext_sendq_head     = NULL;
        VC_FIELD(vc, connection)->rails[i].ext_sendq_tail     = NULL;
        VC_FIELD(vc, connection)->rails[i].ext_sendq_size     = 0;
        VC_FIELD(vc, connection)->rails[i].used_send_cq       = 0;
        VC_FIELD(vc, connection)->rails[i].used_recv_cq       = 0;
        VC_FIELD(vc, connection)->rails[i].hca_index          = hca_index;
        int k;
        if (!process_info.has_srq) {
            for (k = 0; k < rdma_initial_prepost_depth; k++) {
                PREPOST_VBUF_RECV(vc, i);
            }
        }

        VC_FIELD(vc, connection)->srp.credits[i].remote_credit     = rdma_initial_credits;
        VC_FIELD(vc, connection)->srp.credits[i].remote_cc         = rdma_initial_credits;
        VC_FIELD(vc, connection)->srp.credits[i].local_credit      = 0;
        VC_FIELD(vc, connection)->srp.credits[i].preposts          = rdma_initial_prepost_depth;


        if (!process_info.has_srq) {
            VC_FIELD(vc, connection)->srp.credits[i].initialized   = (uint8_t)
                (rdma_prepost_depth == rdma_initial_prepost_depth);
        } else {
            VC_FIELD(vc, connection)->srp.credits[i].initialized = 1;
            VC_FIELD(vc, connection)->srp.credits[i].pending_r3_sends = 0;
        }

        VC_FIELD(vc, connection)->srp.credits[i].backlog.len       = 0;
        VC_FIELD(vc, connection)->srp.credits[i].backlog.vbuf_head = NULL;
        VC_FIELD(vc, connection)->srp.credits[i].backlog.vbuf_tail = NULL;
    
        VC_FIELD(vc, connection)->srp.credits[i].rendezvous_packets_expected = 0;

        VC_FIELD(vc, connection)->rails[i].s_weight =
                DYNAMIC_TOTAL_WEIGHT / rdma_num_rails;

    }

    VC_FIELD(vc, connection)->rails[rdma_num_rails - 1].s_weight =
            DYNAMIC_TOTAL_WEIGHT -
            (DYNAMIC_TOTAL_WEIGHT / rdma_num_rails) *
            (rdma_num_rails - 1);

    vc->eager_max_msg_sz = rdma_iba_eager_threshold;
    vc_ch->lmt_initiate_lmt  = MPID_nem_lmt_ib_initiate_lmt;  
    vc_ch->lmt_start_recv    = MPID_nem_ib_lmt_start_recv;
    vc_ch->lmt_start_send    = MPID_nem_lmt_ib_start_send;
    vc_ch->lmt_handle_cookie = NULL;
    vc_ch->lmt_done_send     = NULL;
    vc_ch->lmt_done_recv     = MPID_nem_ib_lmt_done_recv;

    MPIDI_FUNC_EXIT(MPID_STATE_MPID_NEM_IB_VC_INIT);
    return mpi_errno;
}



#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_vc_destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * Free any resources associated with this VC here.
 */
int MPID_nem_ib_vc_destroy(MPIDI_VC_t *vc)
{
    int mpi_errno = MPI_SUCCESS;
    VC_FIELD(vc, connection) = NULL;
       return mpi_errno;
}


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_vc_terminate
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_vc_terminate (MPIDI_VC_t *vc)
{
    return MPIDI_CH3U_Handle_connection(vc, MPIDI_VC_EVENT_TERMINATED);
}
