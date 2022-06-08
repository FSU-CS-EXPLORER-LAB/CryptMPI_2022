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
#include "mpiutil.h"
#include <stdio.h>
#include "rdma_impl.h"

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

MPIDI_VC_t *mv2_read_progress_pending_vc = NULL;

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_read_progress
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_read_progress(MPIDI_VC_t ** vc_pptr, vbuf ** v_ptr, int *rdmafp_found, int is_blocking)
{
    int 	type;
    MPIDI_VC_t 	*recv_vc_ptr;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_READ_PROGRESS);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_READ_PROGRESS);

    *vc_pptr = NULL;
    *v_ptr = NULL;

    /* Blocking for message on one VC */
    if (mv2_read_progress_pending_vc != NULL) {
        type = MPIDI_CH3I_MRAILI_Waiting_msg(mv2_read_progress_pending_vc, v_ptr, 1);
        if (type == T_CHANNEL_CONTROL_MSG_ARRIVE) {
            if((void *) mv2_read_progress_pending_vc != (*v_ptr)->vc) {
                fprintf(stderr, "mismatch %p %p\n", mv2_read_progress_pending_vc,
                        (*v_ptr)->vc);
            }
            MPIU_Assert((void *) mv2_read_progress_pending_vc == (*v_ptr)->vc);
            *vc_pptr = mv2_read_progress_pending_vc;
        } else if(type == T_CHANNEL_EXACT_ARRIVE) {
            *vc_pptr = mv2_read_progress_pending_vc;
            mv2_read_progress_pending_vc = NULL;
            DEBUG_PRINT("will return seqnum %d\n",
                        ((MPIDI_CH3_Pkt_rndv_req_to_send_t *) (*v_ptr)->
                         pheader)->seqnum);
        } else if (type == T_CHANNEL_OUT_OF_ORDER_ARRIVE) {
            /* Reqd pkt has not arrived yet. Poll CQ for it */
        } else {
            /* T_CHANNEL_NO_ARRIVE - no packets left  in queue */
        }
        goto fn_exit;
    }

    /* Poll RDMA Fast path channel */
    if (likely(mv2_MPIDI_CH3I_RDMA_Process.polling_group_size)) {
        type = MPIDI_CH3I_MRAILI_Get_next_vbuf(vc_pptr, v_ptr);
        if (type != T_CHANNEL_NO_ARRIVE) {
            if (rdmafp_found != NULL ) {
                *rdmafp_found = 1;
            }
            goto fn_exit;
        }
    }
    /* local polling has finished, now we need to start global subchannel polling 
     * For convenience, at this stage, we by default refer to the global polling channel 
     * as the send recv channel on each of the queue pair
     * TODO: we may extend this interface to support other format of channel polling */
    /* Interface cq_poll requires that if *v_ptr is exactly the next packet 
     * don't enqueue * v_ptr. 
     * if *v_ptr is not initialized as NULL, Cq_poll will return exactly only the
     * packet on *v_ptr
     * TODO: Is this the optimal way?
     */
    type = MPIDI_CH3I_MRAILI_Cq_poll(v_ptr, NULL, 0, is_blocking);
    if (type != T_CHANNEL_NO_ARRIVE) {
        recv_vc_ptr = (*v_ptr)->vc;
        *vc_pptr = recv_vc_ptr;
        switch (type) {
        case (T_CHANNEL_EXACT_ARRIVE):
            DEBUG_PRINT("Get one packet with exact seq num\n");
            break;
        case (T_CHANNEL_OUT_OF_ORDER_ARRIVE):
            /* It is possible that *v_ptr points to a vbuf that contains later pkts send/recv 
             * may return vbuf from any connection */
            DEBUG_PRINT("get out of order progress seqnum %d, expect %d\n",
                        ((MPIDI_CH3_Pkt_rndv_req_to_send_t *) *
                         v_ptr)->seqnum, recv_vc_ptr->seqnum_recv);

            type =
                MPIDI_CH3I_MRAILI_Waiting_msg(recv_vc_ptr, v_ptr, 1);
            if (type == T_CHANNEL_CONTROL_MSG_ARRIVE) {
                mv2_read_progress_pending_vc = recv_vc_ptr;
            } else if (T_CHANNEL_EXACT_ARRIVE == type) {
                DEBUG_PRINT("Get out of order delivered msg\n");
            } else {
                PRINT_ERROR("Error recving run return type\n");
                exit(EXIT_FAILURE);
            }
            break;
        case (T_CHANNEL_CONTROL_MSG_ARRIVE):
            DEBUG_PRINT("Get one control msg\n");
            break;
        default:
            /* Error here */
            break;
        }
        goto fn_exit;
    }

  fn_exit:

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_READ_PROGRESS);
    return MPI_SUCCESS;
}

/* non-blocking functions */

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_post_read
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_post_read(MPIDI_VC_t * vc, void *buf, int len)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_POST_READ);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_POST_READ);
    MPIDI_DBG_PRINTF((60, FCNAME, "entering"));
    vc->ch.read.total = 0;
    vc->ch.read.buffer = buf;
    vc->ch.read.bufflen = len;
    vc->ch.read.use_iov = FALSE;
    vc->ch.read_state = MPIDI_CH3I_READ_STATE_READING;
#ifdef USE_RDMA_UNEX
    if (vc->ch.unex_list)
        shmi_read_unex(vc);
#endif
    MPIU_DBG_PRINTF(("post_read: len = %d\n", len));
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_POST_READ);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_post_readv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_post_readv(MPIDI_VC_t * vc, MPL_IOV * iov, int n)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_POST_READV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_POST_READV);
    MPIDI_DBG_PRINTF((60, FCNAME, "entering"));
    /* strip any trailing empty buffers */
    while (n && iov[n - 1].MPL_IOV_LEN == 0)
        n--;
    vc->ch.read.total = 0;
    vc->ch.read.iov = iov;
    vc->ch.read.iovlen = n;
    vc->ch.read.index = 0;
    vc->ch.read.use_iov = TRUE;
    vc->ch.read_state = MPIDI_CH3I_READ_STATE_READING;
#ifdef USE_RDMA_UNEX
    if (vc->ch.unex_list)
        shmi_readv_unex(vc);
#endif
#ifdef MPICH_DBG_OUTPUT
    while (n) {
        MPIU_DBG_PRINTF(("post_readv: iov[%d].len = %d\n", n - 1,
                         iov[n - 1].MPL_IOV_LEN));
        n--;
    }
#endif

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_POST_READV);
    return MPI_SUCCESS;
}
