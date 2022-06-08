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

#ifndef IB_DS_QUEUE_H
#define IB_DS_QUEUE_H

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>


typedef struct _ib_queue_elem {
    void                    *data;
    struct _ib_queue_elem   *next;
} MPID_nem_ib_queue_elem_t, *MPID_nem_ib_queue_elem_ptr_t;

typedef struct _ib_queue {
    MPID_nem_ib_queue_elem_t *head;
    MPID_nem_ib_queue_elem_t *tail;
    MPID_nem_ib_queue_elem_t *free_queue;
} MPID_nem_ib_queue_t, *MPID_nem_ib_queue_ptr_t;

/** TODO MOVE IN ib_send.h */
extern MPID_nem_ib_queue_ptr_t MPID_nem_ib_vc_queue;

int MPID_nem_ib_queue_init(MPID_nem_ib_queue_t**);

int MPID_nem_ib_queue_new_elem(
        MPID_nem_ib_queue_elem_t **, void *init_ptr);

int MPID_nem_ib_queue_empty(MPID_nem_ib_queue_t *q);

void MPID_nem_ib_queue_dequeue(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t **e);

void MPID_nem_ib_queue_enqueue(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t *e);

void MPID_nem_ib_queue_free(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t *e);

int MPID_nem_ib_queue_alloc(
        MPID_nem_ib_queue_t *q,
        MPID_nem_ib_queue_elem_t **e);

void MPID_nem_ib_queue_finalize(
        MPID_nem_ib_queue_t *q);


#endif  /* IB_DS_QUEUE_H */
