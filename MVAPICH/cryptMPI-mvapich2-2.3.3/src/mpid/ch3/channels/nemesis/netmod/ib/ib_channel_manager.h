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

#ifndef IB_CHANNEL_MANAGER_H
#define IB_CHANNEL_MANAGER_H

#include "ib_hca.h"
#include "ib_vbuf.h"

typedef struct MPID_nem_ib_channel_manager_t {
    int     num_channels;
    int     num_local_pollings;

    struct {
        vbuf    *v_queue_head;
        vbuf    *v_queue_tail;
        int     len;
    } *msg_channels;

    struct MPID_nem_ib_channel_manager_t *next_arriving;
    int    inqueue;

    struct MPID_nem_ib_channel_manager_t *prev;
    struct MPID_nem_ib_channel_manager_t *next;   /* for msg queue */
    int    pending_vbuf;

    void *vc;
} MPID_nem_ib_channel_manager;

/**
 * The list of all cmanagers.
 */
extern MPID_nem_ib_channel_manager *cmanagers;
extern MPID_nem_ib_channel_manager *arriving_head;
extern MPID_nem_ib_channel_manager *arriving_tail;

#define INDEX_GLOBAL(_cmanager,_global_index) (_global_index) 

#define INDEX_LOCAL(_cmanager,_local_index) \
    (((_cmanager)->num_channels - (_cmanager)->num_local_pollings) + (_local_index))

/* return type predefinition */
#define T_CHANNEL_NO_ARRIVE 0   
#define T_CHANNEL_EXACT_ARRIVE 1
#define T_CHANNEL_OUT_OF_ORDER_ARRIVE 2
#define T_CHANNEL_CONTROL_MSG_ARRIVE 3
#define T_CHANNEL_ERROR -1

int MPIDI_nem_ib_init_cmanager(int size, int rank);
void CHANNEL_ENQUEUE(MPID_nem_ib_channel_manager* cmanager);
void VQUEUE_ENQUEUE(MPID_nem_ib_channel_manager* cmanager, int index, vbuf* v);
vbuf* VQUEUE_DEQUEUE(MPID_nem_ib_channel_manager* cmanager, int index);
#endif
