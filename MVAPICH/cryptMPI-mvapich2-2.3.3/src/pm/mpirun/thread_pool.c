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

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <netinet/in.h>
#include <byteswap.h>
#include <inttypes.h>
#include <signal.h>

#include <string.h>             //     #define _GNU_SOURCE

#include <infiniband/verbs.h>

#include "ib_comm.h"
#include "debug.h"
#include "thread_pool.h"
#include "ibutil.h"
#include "work_queue.h"

//struct thread_pool*  tp_init_thread_pool(struct thread_pool* pool,  int numthr, void*(func)(void*), char* name )
struct thread_pool *tp_create_thread_pool(int numthr, void *(func) (void *), char *name)
{
    struct thread_pool *pool;

    int i;

    if (numthr > MAX_THREAD_NUM) {
        printf("Error!! numthr %d > MAX-THREAD-NUM\n", numthr);
        return NULL;
    }
    /// alloc a thread-pool 
    pool = (struct thread_pool *) malloc(sizeof(struct thread_pool));
    if (!pool) {
        printf("Fail to create thrpool %s\n", name);
        return NULL;
    }
    memset(pool, 0, sizeof(struct thread_pool));

    // set thread-pool name 
    strncpy(pool->name, name, 15);
    pool->name[15] = 0;

    pool->num_threads = numthr;

    // work-queue associated with thread-pool
    pool->queue = create_queue("RR-queue");
    if (pool->queue == NULL) {
        error("Fail to create queue\n");
        goto err_out_1;
    }
    dump_queue(pool->queue);

    /// the mutex and bitmap is not necessary now...    
    pthread_mutex_init(&pool->mutex, NULL);

    /// now bitmap is no longer needed in a thread-pool, replaced by a work-queue 
    bmp_init(&pool->bitmap, numthr, 1); // initially all bits are "1": free
    //bmp_dump( &pool->bitmap);

    /// create the thread-pool
    for (i = 0; i < numthr; i++) {
        //sem_init( &pool->sem[i], 0, 0); // init the sem-count to be 0     
        //pool->arg1[i] = 0;
        //pool->arg2[i] = 0;
        if (pthread_create(&pool->thread[i], NULL, func, (void *) pool) != 0) {
            error("Error creating thread %d in pool %s\n", i, name);
        }
    }

    //dbg("tpool %s: init bitmap= %s\n", pool->name, int_to_binary(pool->bitmap) );
    printf("Have created tpool %s: thr-num = %d\n", pool->name, pool->num_threads);

    return pool;

  err_out_1:
    tp_destroy_thread_pool(pool);
    return NULL;
}

int tp_destroy_thread_pool(struct thread_pool *pool)    // , struct work_queue* wq )
{
    int i = 0;

/*    /// tell all threads to exit
    for(i=0; i<pool->num_threads; i++){
        pool->arg1[i] = arg_invalid;  // pass an invalid args, tell thread i to terminate
        sem_post( &pool->sem[i] );    // awake thread i
        //dbg(" resume thread %d\n", i);
    }    
*/

    printf("start to destroy thread_pool \"%s\" ...\n", pool->name);
    for (i = 0; i < pool->num_threads; i++) {
        workqueue_enqueue(pool->queue, NULL, 0, arg_invalid, arg_invalid);
    }

    /// wait for all thread to terminate
    for (i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->thread[i], NULL);
        //dbg(" join thread %d\n", i);
    }

    ///////// mutex and bitmap are not necessary now
    pthread_mutex_destroy(&pool->mutex);
    //bmp_destroy( & pool->bitmap);
    ///////////////

    for (i = 0; i < pool->num_threads; i++) {
        //sem_destroy( &pool->sem[i] );
    }

    if (pool->queue) {
        destroy_queue(pool->queue);
        pool->queue = NULL;
    }

    free(pool);
    printf("thread_pool \"%s\" released...\n", pool->name);

    return i;                   // return num of threads terminated
}

void tp_wake_all_threads(struct thread_pool *pool)
{
    int i;
    for (i = 0; i < pool->num_threads; i++) {
        workqueue_enqueue(pool->queue, NULL, 0, arg_invalid, arg_invalid);
    }
}

int tp_get_thread(struct thread_pool *pool)
{
    //static int cnt = 0;   
    //int i = cnt % pool->num_threads;
    //cnt++;
    int i;
    pthread_mutex_lock(&pool->mutex);
/*    i = ffs(pool->bitmap); // the lsb is bit 1
    if( i>= 1 ){
        pool->bitmap &= ~(1<<(i-1));
    }    */

    i = bmp_ffs_and_toggle(&pool->bitmap);  // the LSB offset is 0
    pool->usage_cnt++;

    //dbg("tpool %s: get tp %d, bitmap is:: \n", pool->name, i); 
    dbg("tpool %s: get tp %d, usg=%d\n", pool->name, i, pool->usage_cnt);
    //bmp_dump( &pool->bitmap);

    pthread_mutex_unlock(&pool->mutex);
    //dbg(" get thread %d\n", i); //-1);
    return i;                   // the LSB is index at 0, // so minus 1
}

/*
Return the thread(tid) to thread_pool as free
*/
void tp_free_thread(struct thread_pool *pool, int tid)
{
    pthread_mutex_lock(&pool->mutex);

    //pool->bitmap |= (1<<(tid));
    bmp_set_bit(&pool->bitmap, tid);    // mark the bit "tid" to "1" as free

    dbg("tpool %s: free tpid %d, bitmap== \n", pool->name, tid);    //, pool->bitmap);
    //bmp_dump( &pool->bitmap);

    pthread_mutex_unlock(&pool->mutex);
    //dbg(" free thread %d\n", tid);

}

#endif
