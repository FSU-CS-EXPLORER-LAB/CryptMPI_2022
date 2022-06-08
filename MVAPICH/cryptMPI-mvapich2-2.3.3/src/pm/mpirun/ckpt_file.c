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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <infiniband/verbs.h>
//#include "ib_comm.h"

#include <semaphore.h>

#include "ib_comm.h"
#include "ib_buf.h"

#include "list.h"
#include "debug.h"

#include "ckpt_file.h"

extern struct ib_HCA hca;

/***
Allocate a struct representing a new ckpt_file, 
and init some elems of the struct
***/
ckpt_file_t *alloc_ckpt_file()
{
    ckpt_file_t *cfile = malloc(sizeof(ckpt_file_t));

    memset(cfile, 0, sizeof(*cfile));

    MV2_INIT_LIST_HEAD(&cfile->list);
    MV2_INIT_LIST_HEAD(&cfile->chunk_list);

    pthread_mutex_init(&cfile->mutex, NULL);
    pthread_mutex_init(&cfile->io_mtx, NULL);

    sem_init(&cfile->sem, 0, 0);
    gettimeofday(&cfile->tstart, NULL);

    return cfile;
}

int free_ckpt_file(ckpt_file_t * cfile)
{
    /////////// if chunk-list not empty, should free them...
    if (!mv2_list_empty(&cfile->chunk_list)) {
        error("free cfile: ckpt-%d-proc-%d: chunk-list not empty!!!\n", cfile->ckpt_id, cfile->proc_rank);
        ckpt_chunk_t *ck;
        struct list_head *pos, *n;
        /*mv2_list_for_each_entry( ck, &(cfile->chunk_list), list )
           {
           //// free the buf-chunk
           mv2_list_del( &ck->list );
           ckpt_free_chunk(ck);
           free(ck);
           } */
        mv2_list_for_each_safe(pos, n, &(cfile->chunk_list)) {
            ck = mv2_list_entry(pos, ckpt_chunk_t, list);
            mv2_list_del(&ck->list);
            ckpt_free_chunk(ck);
            free(ck);
        }
    }
    //////////////////////

    sem_destroy(&cfile->sem);
    pthread_mutex_destroy(&cfile->mutex);
    if (cfile->fd > 0)
        close(cfile->fd);
    cfile->fd = -1;
    free(cfile);
    return 0;
}

ckpt_chunk_t *alloc_ckpt_chunk()
{
    ckpt_chunk_t *chk = malloc(sizeof(ckpt_chunk_t));

    memset(chk, 0, sizeof(*chk));

    return chk;
}

void ckpt_free_chunk(ckpt_chunk_t * chunk)
{
    free_buf_slot(hca.rdma_buf, chunk->bufid, 0);
    //free(chunk);
}

/***
Check buf-content. 
***/
unsigned long check_chunk_content(ckpt_file_t * cfile, ckpt_chunk_t * chunk, unsigned long size)
{
    unsigned long i = 0, err = 0;
    unsigned char *buf = chunk->buf;
    //unsigned long size = chunk->size;

    if (!buf || chunk->bufid < 0) {
        error("**** Empty buf! (%s) %ld@%ld: lbuf-id=%d\n", cfile->filename, size, chunk->offset, chunk->bufid);
        return -1;
    }

    for (i = 0; i < size; i++) {
        if (buf[i] != (unsigned char) (chunk->offset + i))
            err++;
    }
    dbg("****  (%s) %ld@%ld: lbuf-id=%d:  found %ld err(s)\n", cfile->filename, size, chunk->offset, chunk->bufid, err);

    return err;
}

/**
io-thr has finished RR a chunk, now store this chunk to the corresponding ckpt-file.
rrpkt: the ib-pkt that containing infor about this chunk
**/
int add_chunk_to_ckpt_file(ckpt_file_t * cfile, struct ib_packet *rrpkt)
{
    int flag;

    ckpt_chunk_t *ck = NULL;

    /// init a new chunk
    ckpt_chunk_t *chunk = malloc(sizeof(ckpt_chunk_t));
    chunk->ckpt_id = rrpkt->RR.rckptid;
    chunk->proc_rank = rrpkt->RR.rprocid;
    chunk->offset = rrpkt->RR.offset;
    chunk->size = rrpkt->RR.size;
    chunk->buf = (void *) rrpkt->RR.laddr;  // local RR buf-chunk addr
    chunk->bufid = rrpkt->RR.lbuf_id;   // local RR buf-chunk  id
    chunk->curr_pos = 0;
    MV2_INIT_LIST_HEAD(&chunk->list);

    //// check chunk contents
    //check_chunk_content( cfile, chunk, chunk->size );

    //////
    pthread_mutex_lock(&cfile->mutex);

    //// search in ascending order, find a place to insert this new chunk
    flag = 0;
    mv2_list_for_each_entry(ck, &(cfile->chunk_list), list) {
        if (chunk->offset < ck->offset) {
            flag = 1;
            // insert chunk before this "ck"
            mv2_list_add(&chunk->list, ck->list.prev);
            break;
        }
    }
    if (!flag) {                // should insert to tail of list
        mv2_list_add_tail(&chunk->list, &(cfile->chunk_list));
    }

    cfile->rcv_pending_RR_num--;
    cfile->rcv_ready_chunk_num++;
    cfile->write_size += rrpkt->RR.size;

    pthread_mutex_unlock(&cfile->mutex);

    /// notify anyone blocking in the file
    sem_post(&cfile->sem);

    return 0;
}

void dump_ckpt_file(ckpt_file_t * cfile)
{
    ckpt_chunk_t *ck;
    int i = 0;
    //pthread_mutex_lock( &cfile->mutex );
    printf("ckpt-file (%s)(ckptid %d, proc-rank %d): ref=%d, has_last=%d, can-release=%d, adv_size=%ld "
           "chunklist:: \n", cfile->filename, cfile->ckpt_id, cfile->proc_rank, atomic_read(&cfile->ref), cfile->has_last_chunk, cfile->can_release, cfile->adv_size);

    mv2_list_for_each_entry(ck, &(cfile->chunk_list), list) {
        printf("\tChunk %d:  bufid %d, %ld@%ld, curr_pos %lu\n", i, ck->bufid, ck->size, ck->offset, ck->curr_pos);
        i++;
    }
//  pthread_mutex_unlock( &cfile->mutex );
}

void dump_chunk(ckpt_chunk_t * ck)
{
    printf("\tChunk:  offset %lu, size %lu, curr_pos %lu\n", ck->offset, ck->size, ck->curr_pos);
}

/**
cr_restart needs a data-chunk starting at (offset) from the ckpt_file.
It may block here waiting for the availability of that chunk
**/
ckpt_chunk_t *get_chunk_from_ckpt_file(ckpt_file_t * cfile, int offset)
{
    ckpt_chunk_t *ck = NULL;
    int found = 0;

    while (!found)              // must block until we get a chunk with that offset
    {
        pthread_mutex_lock(&cfile->mutex);

        mv2_list_for_each_entry(ck, &(cfile->chunk_list), list) {
            //dbg("\tsee a ck @ (%d,%d) \n", ck->offset, ck->size );
            if (ck->offset == offset) { // find the chunk
                mv2_list_del(&ck->list);    // rm the chunk from chunk-list
                pthread_mutex_unlock(&cfile->mutex);
                //found = 1;
                return ck;
            }
        }

        // cannot find the chunk, Is this ckpt-file finished?
        if (cfile->has_last_chunk && cfile->rcv_pending_RR_num == 0) {
            // this ckpt-file is complete, no more chunks...
            dbg(" -----------  cfile (%d, %d) is completed...\n", cfile->ckpt_id, cfile->proc_rank);
            pthread_mutex_unlock(&cfile->mutex);
            return NULL;
        }
        // haven't found the chunk, wait
        pthread_mutex_unlock(&cfile->mutex);
        dbg("cannot find chunk(%d, %d)@offset %d, wait...\n", cfile->ckpt_id, cfile->proc_rank, offset);
        sem_wait(&cfile->sem);
    }

    return NULL;
}

#endif
