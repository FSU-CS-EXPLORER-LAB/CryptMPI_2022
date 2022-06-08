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

#ifndef IB_RDMA_H
#define IB_RDMA_H

#define _GNU_SOURCE
#include "mpid_nem_impl.h"
#include <infiniband/verbs.h>
/* add ib_vc.h, ib_hca.h for vbuf_fast_rdma_alloc function*/
#include "ib_vc.h"
#include "ib_hca.h"

int vbuf_fast_rdma_alloc(MPIDI_VC_t * c, int dir);
int deregister_memory(struct ibv_mr * mr); 
struct ibv_mr * register_memory(void * buf, size_t len, int hca_num);
#endif
