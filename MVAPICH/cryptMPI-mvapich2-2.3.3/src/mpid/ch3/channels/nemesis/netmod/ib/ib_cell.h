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

#ifndef IB_CELL_H
#define IB_CELL_H

#include "ib_ds_queue.h"

typedef struct {
    union {
        struct ibv_send_wr s_wr;
        struct ibv_recv_wr r_wr;
    } u;
    struct ibv_sge sg_list;
} MPID_nem_ib_descriptor_t;

typedef struct {
    MPID_nem_cell_ptr_t     nem_cell;
    int                     datalen;
    MPID_nem_ib_queue_elem_t *qe;
    MPIDI_VC_t              *vc;
    MPID_nem_ib_descriptor_t desc;
} MPID_nem_ib_cell_elem_t;

typedef struct _ib_cell_pool {
    MPID_nem_ib_queue_t          *queue;
    int                                 ncells;
    pthread_spinlock_t                 lock;
} MPID_nem_ib_cell_pool_t;


int MPID_nem_ib_init_cell_pool(int n);
void MPID_nem_ib_finalize_cell_pool();

int MPID_nem_ib_get_cell( MPID_nem_ib_cell_elem_t **e);
void MPID_nem_ib_return_cell( MPID_nem_ib_cell_elem_t *ce);
void MPID_nem_ib_prep_cell_recv( MPID_nem_ib_cell_elem_t *ce, void* buf);
void MPID_nem_ib_prep_cell_send( MPID_nem_ib_cell_elem_t *ce, void* buf, uint32_t len);
int MPID_nem_ib_add_cells(int n);

extern MPID_nem_ib_cell_pool_t MPID_nem_ib_cell_pool;

#endif
