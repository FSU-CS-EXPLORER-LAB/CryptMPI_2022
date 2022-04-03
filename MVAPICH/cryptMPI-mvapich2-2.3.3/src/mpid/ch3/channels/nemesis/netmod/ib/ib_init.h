/*!\file */
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

#ifndef IB_INIT_H
#define IB_INIT_H

#define _GNU_SOURCE
#include "mpid_nem_impl.h"
#include "ib_ds_queue.h"
#include <infiniband/verbs.h>

#define FREE_SEND_QUEUE_ELEMENTS  MPID_NEM_NUM_CELLS

extern MPID_nem_queue_ptr_t MPID_nem_module_ib_recv_queue;
extern MPID_nem_queue_ptr_t MPID_nem_module_ib_free_queue;

extern MPID_nem_queue_ptr_t MPID_nem_process_recv_queue;
extern MPID_nem_queue_ptr_t MPID_nem_process_free_queue;

extern struct ibv_mr *proc_elements_mr;
extern struct ibv_mr *module_elements_mr;



int MPID_nem_ib_init (MPIDI_PG_t *pg_p,
                             int pg_rank,
                             char **bc_val_p,
                             int *val_max_sz_p);

int MPID_nem_ib_get_business_card (int my_rank, char **bc_val_p, int *val_max_sz_p);
int MPID_nem_ib_ckpt_shutdown (void);
int MPID_nem_ib_connect_to_root (const char *business_card, MPIDI_VC_t *new_vc);

#ifdef ENABLE_CHECKPOINTING
typedef struct MPIDI_nem_ib_pkt_unpause
{
    MPIDI_CH3_Pkt_type_t type;
    unsigned subtype;
} MPIDI_nem_ib_pkt_unpause_t;
#endif

typedef GENERIC_Q_DECL(struct MPID_Request) MPIDI_nem_ib_request_queue_t;

#endif /* IB_INIT_H */
