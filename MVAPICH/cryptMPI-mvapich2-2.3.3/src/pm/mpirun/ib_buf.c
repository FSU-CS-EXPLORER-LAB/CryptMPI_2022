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

#include <infiniband/verbs.h>

#include "ib_comm.h"
// #include "ibutil.h"
#include "debug.h"

void *ib_buffer_slot_addr(struct ib_buffer *buf, int slot)
{
    if (slot > buf->num_slot) {
        error("Error!! slot %d > buf-slot %d\n", slot, buf->num_slot);
        return NULL;
    }
    return (buf->addr + ((unsigned long) slot) * buf->slot_size);
}

void dump_ib_buffer(struct ib_buffer *buf)
{
    printf("\n===========  dump ib_buffer: %s\n", buf->name);
    printf("\ttotal %d slots, free-slots=%d, total size %ld, slot-size=%d\n", buf->num_slot, buf->free_slots, buf->size, buf->slot_size);
    printf("\tbitmap = \n");
    bmp_dump(&buf->bitmap);
    printf("\n==============================\n");
}

struct ib_buffer *create_ib_buffer(int size, int slot_size, char *name)
{
    int i;

    struct ib_buffer *buf = malloc(sizeof(struct ib_buffer));

    memset(buf, 0, sizeof(struct ib_buffer));

    buf->size = size;
    buf->slot_size = slot_size;
    buf->num_slot = size / slot_size;   // num_slot;
    buf->free_slots = buf->num_slot;

    // alloc actual mem for this buf
    buf->addr = memalign(PAGE_SIZE, size);

    // init the bitmap for this buf, all bits are inited to "1"
    if (bmp_init(&buf->bitmap, buf->num_slot, 1) < 0) {
        error("Error!: Fail to create ib_buf %s\n", name);
        goto err_out_1;
    }
    //memset( buf->state,SLOT_FREE, BUF_SLOT_COUNT ); // all slots are free now

    // init the buf-mutex
    pthread_mutex_init(&buf->mutex, NULL);

    // init locks for each slot 
    for (i = 0; i < buf->num_slot; i++) {
        pthread_mutex_init(&buf->lock[i], NULL);
    }

    // init the buf_semaphore   
    dbg("ib-buf (%s): init sem to %d\n", name, buf->num_slot);
    sem_init(&buf->buf_sem, 0, buf->num_slot);  // init the sem-count to be 0

    strcpy(buf->name, name);

    printf("Has created ib-buf \"%s\", size=%d, slot-size %d, %d slots \n", name, size, slot_size, buf->num_slot);
    return buf;

  err_out_1:
    free(buf->addr);
    free(buf);
    return NULL;
}

/*
    free all resources in this IB_buffer
*/
void free_ib_buffer(struct ib_buffer *buf)
{
    int i;

    dbg(" ----> free buf \"%s\" \n", buf->name);

    sem_destroy(&buf->buf_sem);

    // release the mutex
    pthread_mutex_destroy(&buf->mutex);

    for (i = 0; i < buf->num_slot; i++) {
        pthread_mutex_destroy(&buf->lock[i]);
    }

    // destroy the bitmap       
    bmp_destroy(&buf->bitmap);

    /// dereg the mem-region
    if (buf->mr)
        ibv_dereg_mr(buf->mr);

    free(buf->addr);
    free(buf);

    //dbg(" <----- buf %s\n", buf->name );
}

/*
Get a free slot in this buffer. The slot size = buf->slot_size.
the address of this slot is returned via (*addr).
if "expect"=1, need put a lock at the allocated slot
*/
int get_buf_slot(struct ib_buffer *buf, void **addr, int expect)
{
    int i;

// search_free_slot:
    sem_wait(&buf->buf_sem);    // dec the free-slot by 1, may block here

    pthread_mutex_lock(&buf->mutex);

    ///////////////
    i = bmp_ffs_and_toggle(&buf->bitmap);
    if (i >= 0)                 // has grab slot i
    {
        buf->free_slots--;
        pthread_mutex_unlock(&buf->mutex);

        if (expect)
            pthread_mutex_lock(&buf->lock[i]);
        if (addr) {
            *addr = (buf->addr + i * buf->slot_size);
        }
        return i;
    }
/*    else{    /// this should never happen!!
        pthread_mutex_unlock( &buf->mutex );
        error(" Error at buf-slot at %s\n", buf->name );
        goto search_free_slot;
    }    */
    /////////////////// 
    /*
       for(i=0; i< buf->num_slot; i++ )
       {
       if( buf->state[i] == SLOT_FREE )// find a free slot
       {
       buf->state[i] = SLOT_INUSE;
       if( expect )
       pthread_mutex_lock( & buf->lock[i] );
       buf->free_slots--;
       pthread_mutex_unlock( &buf->mutex );

       //dbg("get slot %d from buf \"%s\"\n", i, buf->name);
       if( addr ){
       *addr = (buf->addr + i*buf->slot_size);
       }
       return i;
       }
       } */

    pthread_mutex_unlock(&buf->mutex);
    error(" Error when alloc buf-slot at %s\n", buf->name);
    return -1;                  // this shouldn't happen!!   
}

/*
Free a slot, return it to buf.  
If "expect" =1, unlock the related mtx, to wake up any threads waiting on it
*/
int free_buf_slot(struct ib_buffer *buf, int slot, int expect)
{
    pthread_mutex_lock(&buf->mutex);

    //buf->state[slot] = SLOT_FREE;
    if (bmp_get_pos(&buf->bitmap, slot) == 0) {
        bmp_set_bit(&buf->bitmap, slot);
    } else {
        //dbg("========= buf %p:  slot %d already free...\n", buf, slot);
    }
    buf->free_slots++;
    ////////////

    if (expect)
        pthread_mutex_unlock(&buf->lock[slot]);

    pthread_mutex_unlock(&buf->mutex);

    sem_post(&buf->buf_sem);    // increase the slot-count by 1

    //dbg("Has freed \"%s\" slot %d\n", buf->name, slot);
    return 0;
}

#endif
