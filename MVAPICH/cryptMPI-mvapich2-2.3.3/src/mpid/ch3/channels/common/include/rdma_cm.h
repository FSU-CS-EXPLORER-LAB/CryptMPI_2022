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

#ifndef MVAPICH2_GEN2_RDMA_CM_H
#define MVAPICH2_GEN2_RDMA_CM_H

#include "mpichconf.h"

#if defined(RDMA_CM)

#include <stdio.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <semaphore.h>
#include <pthread.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#ifdef _MULTI_SUBNET_SUPPORT_
#include <infiniband/ib.h>
#endif /*_MULTI_SUBNET_SUPPORT_*/
#include "rdma_impl.h"

extern int g_num_smp_peers;
extern int *rdma_cm_host_list;
#ifdef _MULTI_SUBNET_SUPPORT_
extern union ibv_gid *rdma_cm_host_gid_list;
#endif /*_MULTI_SUBNET_SUPPORT_*/

/* Initiate all active connect requests */
int rdma_cm_connect_all(int pg_rank, MPIDI_PG_t *pg);

/* Initiate single active connect request */
int rdma_cm_connect_to_server(MPIDI_VC_t *vc, int ipnum, int rail_index);

/* Exchange the ip information with all the processes */
int rdma_cm_get_hostnames(int pg_rank, MPIDI_PG_t *pg);

/* Initialize rdma_cm resources + cm_ids + bind port + connection thrd */
int ib_init_rdma_cm(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc,
		     int pg_rank, int pg_size);

/* Finalize rdma_cm specific resources */
void ib_finalize_rdma_cm(int pg_rank, MPIDI_PG_t *pg);

#endif /* defined(RDMA_CM) */

#endif  /* MVAPICH2_GEN2_CM_H */
