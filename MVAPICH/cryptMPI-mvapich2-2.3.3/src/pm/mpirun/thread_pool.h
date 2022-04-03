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

#ifndef __THREAD_POOL__
#define __THREAD_POOL__

#include <semaphore.h>

#include "bitmap.h"
#include "work_queue.h"

#define MAX_THREAD_NUM    (256)

typedef struct thread_pool {
    int num_threads;            // num of threads in this pool
    int usage_cnt;              // how many usages on the pool

    struct work_queue *queue;   // threads fetch work-rqst from this queue
    pthread_t thread[MAX_THREAD_NUM];   // array of threads

    /// don't need:  mutex and bitmap
    pthread_mutex_t mutex;      //lock to protect the pool
    struct bitmap bitmap;       // bitmap of free threads     

//  sem_t       sem[MAX_THREAD_NUM];   // block/unblock a thread    
//  uint64_t    arg1[MAX_THREAD_NUM];   // arg1 for each thread
//  uint64_t    arg2[MAX_THREAD_NUM];   // arg2 for each thread

    char name[16];
} thread_pool_t;

//struct thread_pool*  tp_init_thread_pool(struct thread_pool* pool,  int numthr, void*(func)(void*), char* name );
struct thread_pool *tp_create_thread_pool(int numthr, void *(func) (void *), char *name);
int tp_destroy_thread_pool(struct thread_pool *pool);

int tp_get_thread(struct thread_pool *pool);
void tp_free_thread(struct thread_pool *pool, int tid);

void tp_wake_all_threads(struct thread_pool *pool);

#endif                          // __THREAD_POOL__
