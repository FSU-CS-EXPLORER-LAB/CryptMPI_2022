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

#ifndef WORK_QUEUE_H
#define WORK_QUEUE_H

#define  MAX_QUEUE_LENGTH    (1024)

#include <semaphore.h>

#define WORKELEM_SIZE (512)

typedef struct work_elem {
    unsigned char data[WORKELEM_SIZE];
    int size;                   // data-size
    unsigned long arg1;
    unsigned long arg2;
    unsigned long arg3;
} __attribute__ ((packed)) work_elem_t;

typedef struct work_queue {

    int qlength;                // length of queue

    unsigned long head;         // q-head: always grab an elem from head
    unsigned long tail;         // q-tail:  always insert new elem to tail

    sem_t sem;                  // control how many available items in the queue
    pthread_mutex_t lock;       // serialize access to the queue

    struct work_elem queue[MAX_QUEUE_LENGTH];

    char name[16];
} work_queue_t;

struct work_queue *create_queue(char *name);
int destroy_queue(struct work_queue *wq);

int workqueue_enqueue(struct work_queue *wq, void *elem, int elem_size, unsigned long arg1, unsigned long arg2);

int workqueue_enqueue3(struct work_queue *wq, void *elem, int elem_size, unsigned long arg1, unsigned long arg2, unsigned long arg3);

int workqueue_dequeue(struct work_queue *wq, struct work_elem *we);

int workqueue_empty(struct work_queue *wq);
int workqueue_full(struct work_queue *wq);

void dump_work_elem(struct work_elem *we);
void dump_queue(struct work_queue *wq);

#endif                          // end of #ifndef WORK_QUEUE_H
