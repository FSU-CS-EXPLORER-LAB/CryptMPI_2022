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

#include "mpidimpl.h"
#include "ib_ds_queue.h"



#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_queue_init(
        MPID_nem_ib_queue_t **q)
{
    int mpi_errno = MPI_SUCCESS;
    /*int i;*/

    MPIU_Assert(NULL != q);

    *q = MPIU_Malloc(sizeof(MPID_nem_ib_queue_t));

    if(NULL == *q) {
        MPIU_CHKMEM_SETERR(mpi_errno,
                sizeof(MPID_nem_ib_queue_t),
                "IB Module Queue");
    }

    memset(*q, 0, sizeof(MPID_nem_ib_queue_t));

    /*   fn_exit: */
           return mpi_errno;
    /*   fn_fail:
           goto fn_exit;
           */
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_new_elem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_queue_new_elem(
        MPID_nem_ib_queue_elem_t **e, void *init_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    (*e) = MPIU_Malloc(sizeof(MPID_nem_ib_queue_elem_t));

    if (NULL == *e) {
        MPIU_CHKMEM_SETERR (mpi_errno,
                sizeof(MPID_nem_ib_queue_elem_t),
                "IB module queue elem");
    }

    (*e)->data = init_ptr;

    /*   fn_exit: */
           return mpi_errno;
    /*   fn_fail:
           goto fn_exit;
           */
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_empty
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_queue_empty(
        MPID_nem_ib_queue_t *q)
{
    return (NULL == q->head);
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_dequeue
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPID_nem_ib_queue_dequeue(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t **e)
{
    *e = q->head;

    if(*e) {
        q->head = q->head->next;

        if(NULL == q->head) {
            q->tail = NULL;
        }
    }
}


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_enqueue
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPID_nem_ib_queue_enqueue(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t *e)
{
    if(NULL == q->tail) {
        q->head = e;
    } else {
        q->tail->next = e;
    }

    q->tail = e;
    e->next = NULL;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_free
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPID_nem_ib_queue_free(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t *e)
{
    e->next = q->free_queue;
    q->free_queue = e;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_alloc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_queue_alloc(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t **e)
{
    int mpi_errno = MPI_SUCCESS;

    if(q->free_queue) {
        *e = q->free_queue;
        q->free_queue = q->free_queue->next;
    } else {
        *e = MPIU_Malloc(sizeof(MPID_nem_ib_queue_elem_t));
        if(NULL == *e) {
            MPIU_CHKMEM_SETERR(mpi_errno,
                    sizeof(MPID_nem_ib_queue_elem_t),
                    "IB Module Queue Element");
        }
    }

    /*   fn_exit: */
           return mpi_errno;
    /*   fn_fail:
           goto fn_exit;
           */
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_queue_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPID_nem_ib_queue_finalize(
        MPID_nem_ib_queue_t *q)
{
    MPID_nem_ib_queue_elem_t *e;

    while(!MPID_nem_ib_queue_empty(q)) {

        MPID_nem_ib_queue_dequeue(q, &e);

        MPIU_Free(e);
    }

    MPIU_Free(q);
}
