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

#ifndef IB_POLL_H
#define IB_POLL_H

#include "ib_vbuf.h"

int MPID_nem_ib_poll(int in_blocking_progress);

int MPIDI_nem_ib_cq_poll(vbuf **vbuf_handle, MPIDI_VC_t * vc_req, int receiving, int is_blocking);
int MPIDI_nem_ib_get_next_vbuf(MPIDI_VC_t** vc_ptr, vbuf** vbuf_ptr);
int MPIDI_nem_ib_waiting_msg(MPIDI_VC_t * vc, vbuf ** vbuf_handle, int blocking);
int MPIDI_nem_ib_request_adjust_iov(MPID_Request * req, MPIDI_msg_sz_t nb);

#endif
