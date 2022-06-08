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

#include <mpichconf.h>

#ifdef CR_AGGRE

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#include <string.h>

#include <pthread.h>
#include <semaphore.h>
#include <infiniband/verbs.h>

#include "debug.h"
#include "work_queue.h"

#include "ib_comm.h"

struct work_queue *create_queue(char *name)
{
    struct work_queue *wq = (struct work_queue *) malloc(sizeof(struct work_queue));

    if (!wq) {
        error("fail to alloc queue...\n");
        return NULL;
    }
    memset(wq, 0, sizeof(*wq));

    // now, head=tail=0, empty queue
    wq->qlength = MAX_QUEUE_LENGTH;

    // init the sem to 0 ( no avail elems now )
    if (sem_init(&wq->sem, 0, 0) != 0) {
        error("fail to init sem...\n");
        perror("fail to init sem");
        return NULL;
    }
    // init the lock to be "unlocked"
    if (pthread_mutex_init(&wq->lock, NULL) != 0) {
        error("fail to init mutex...\n");
        perror("init mutex...");
        goto err_out_1;
    }

    strncpy(wq->name, name, 15);
    wq->name[15] = 0;

    return wq;

  err_out_1:
    sem_destroy(&wq->sem);
    free(wq);
    return NULL;
}

int destroy_queue(struct work_queue *wq)
{
    sem_destroy(&wq->sem);
    pthread_mutex_destroy(&wq->lock);

    free(wq);
    return 0;
}

void dump_queue(struct work_queue *wq)
{
    printf("\n========= work-queue %s: \n", wq->name);

    pthread_mutex_lock(&wq->lock);
    printf(" qlength = %d, head=%lu, tail=%lu\n", wq->qlength, wq->head, wq->tail);
    pthread_mutex_unlock(&wq->lock);

}

/*
enqueue an elem to "tail" of the wq, giving 3 args
*/
int workqueue_enqueue3(struct work_queue *wq, void *elem, int elem_size, unsigned long arg1, unsigned long arg2, unsigned long arg3)
{
    struct work_elem *ep;

    pthread_mutex_lock(&wq->lock);

    /// should check if queue full? 
    /// if (wq->tail - wq->head >= MAX_QUEUE_LENGTH)
    ep = wq->queue + (wq->tail % MAX_QUEUE_LENGTH);
    wq->tail++;

    if (elem_size > WORKELEM_SIZE) {
        error("Error: enq elem-size=%d too big...\n", elem_size);
        return -1;
    }
    if (elem)
        memcpy(ep->data, elem, elem_size);
    ep->size = elem_size;
    ep->arg1 = arg1;
    ep->arg2 = arg2;
    ep->arg3 = arg3;

    pthread_mutex_unlock(&wq->lock);
    /// one more valid elem in the queue
    sem_post(&wq->sem);

    return 0;
}

/*
enqueue an elem to "tail" of the wq
*/
int workqueue_enqueue(struct work_queue *wq, void *elem, int elem_size, unsigned long arg1, unsigned long arg2)
{
    return workqueue_enqueue3(wq, elem, elem_size, arg1, arg2, 0);

/*    
    struct work_elem* ep;
    
    pthread_mutex_lock( &wq->lock );
    
    ep = wq->queue + (wq->tail%MAX_QUEUE_LENGTH);
    wq->tail++;
    
    if( elem ) memcpy( ep->data, elem, elem_size );
    ep->arg1 = arg1;
    ep->arg2 = arg2;    
        
    pthread_mutex_unlock( &wq->lock );
    /// one more valid elem in the queue
    sem_post( &wq->sem );
    
    return 0;
*/
}

/*
Retrieve an elem from head of queue
*/
int workqueue_dequeue(struct work_queue *wq, struct work_elem *we)
{
    // grab a valid elem from the queue
    sem_wait(&wq->sem);

    /// serialize access to queue
    pthread_mutex_lock(&wq->lock);

    memcpy(we, wq->queue + (wq->head % MAX_QUEUE_LENGTH), sizeof(*we));
    wq->head++;

    pthread_mutex_unlock(&wq->lock);

    return 0;
}

int workqueue_empty(struct work_queue *wq)
{
    int ret;

    pthread_mutex_lock(&wq->lock);
    ret = (wq->head == wq->tail);
    pthread_mutex_lock(&wq->lock);
    return ret;
}

int workqueue_full(struct work_queue *wq)
{
    int ret;

    pthread_mutex_lock(&wq->lock);
    ret = (wq->tail - wq->head >= MAX_QUEUE_LENGTH);
    pthread_mutex_lock(&wq->lock);
    return ret;
}

void dump_work_elem(struct work_elem *we)
{
    struct ib_packet *pkt = (struct ib_packet *) we->data;

    printf("elem: rbuf_id=%d, rprocid=%d, rckptid=%d, size=%lu, off=%lu, lbuf_id=%d\n", pkt->RR.rbuf_id, pkt->RR.rprocid, pkt->RR.rckptid, pkt->RR.size, pkt->RR.offset, pkt->RR.lbuf_id);
    printf("    arg1= %lu, arg2= %lu\n", we->arg1, we->arg2);

}

#endif
