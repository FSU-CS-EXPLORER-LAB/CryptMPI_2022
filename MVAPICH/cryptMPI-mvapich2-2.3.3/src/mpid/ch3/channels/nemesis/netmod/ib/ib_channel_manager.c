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

#include <infiniband/verbs.h>

#include "mpidimpl.h"
#include "upmi.h"

#include "ib_channel_manager.h"
#include "ib_errors.h"

MPID_nem_ib_channel_manager *cmanagers;

MPID_nem_ib_channel_manager *arriving_head = NULL;
MPID_nem_ib_channel_manager *arriving_tail = NULL;

#undef FUNCNAME
#define FUNCNAME MPIDI_nem_ib_init_cmanager
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * init cmanager_info
 *
 */
int MPIDI_nem_ib_init_cmanager(int rank, int size)
{
    int i;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_INIT_CMANAGER);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_INIT_CMANAGER);

    cmanagers = (MPID_nem_ib_channel_manager *)MPIU_Malloc(size * sizeof(MPID_nem_ib_channel_manager));

    memset(cmanagers, 0, size * sizeof(MPID_nem_ib_channel_manager));

    for (i = 0; i < size; i++) {
        if (i == rank)
            continue;

        cmanagers[i].num_channels          = rdma_num_rails;
        cmanagers[i].num_local_pollings    = 0;
        cmanagers[i].msg_channels = MPIU_Malloc(sizeof *cmanagers[i].msg_channels * (cmanagers[i].num_channels + 1));
        if (!cmanagers[i].msg_channels) {
            ibv_error_abort(GEN_EXIT_ERR, "No resource for msg channels\n");
        }

        memset(cmanagers[i].msg_channels, 0, sizeof *cmanagers[i].msg_channels * (cmanagers[i].num_channels + 1));

        cmanagers[i].next_arriving = NULL;
        cmanagers[i].inqueue       = 0;
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_INIT_CMANAGER);
    return MPI_SUCCESS;
}

void CHANNEL_ENQUEUE(MPID_nem_ib_channel_manager* cmanager)
{
    if (arriving_tail == NULL)
    {
        arriving_head = arriving_tail = cmanager;
        cmanager->next_arriving = NULL;
    }
    else
    {
        arriving_tail->next_arriving = cmanager;
        cmanager->next_arriving = NULL;
        arriving_tail = cmanager;
    }

    cmanager->inqueue = 1;
}

void VQUEUE_ENQUEUE(MPID_nem_ib_channel_manager* cmanager, int index, vbuf* v)
{
    v->desc.next = NULL;

    if (cmanager->msg_channels[index].v_queue_tail == NULL)
    {
        cmanager->msg_channels[index].v_queue_head = v;
    }
    else
    {
        cmanager->msg_channels[index].v_queue_tail->desc.next = v;
    }

    cmanager->msg_channels[index].v_queue_tail = v;
    ++cmanager->msg_channels[index].len;

    if (!cmanager->inqueue)
    {
        CHANNEL_ENQUEUE(cmanager);
    }
}

vbuf* VQUEUE_DEQUEUE(MPID_nem_ib_channel_manager* cmanager, int index)
{
    vbuf* v = cmanager->msg_channels[index].v_queue_head;
    cmanager->msg_channels[index].v_queue_head = v->desc.next;

    if (v == cmanager->msg_channels[index].v_queue_tail)
    {
        cmanager->msg_channels[index].v_queue_tail = NULL;
    }

    --cmanager->msg_channels[index].len;
    v->desc.next = NULL;
    return v;
}
