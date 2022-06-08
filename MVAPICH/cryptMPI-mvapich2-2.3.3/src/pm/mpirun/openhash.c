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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "list.h"
#include "atomic.h"

#include "debug.h"
#include "openhash.h"
//#include "ftb.h"

#include "ckpt_file.h"
#include "genhash.h"

/*
Create a hash-table with (name),  
size:  num of buckets in the table
*/
hash_table_t *create_hash_table(int size, char *name)
{
    int i;
    hash_table_t *ht = (hash_table_t *) malloc(sizeof(hash_table_t));

    ht->buckets = (hash_bucket_t *) malloc(size * sizeof(hash_bucket_t));
    // need safety-check

    ht->length = size;
    strncpy(ht->name, name, 32);
    ht->name[31] = 0;

    for (i = 0; i < size; i++) {
        pthread_mutex_init(&(ht->buckets[i].mutex), NULL);
        MV2_INIT_LIST_HEAD(&(ht->buckets[i].list));
    }

    printf("Have created hash-table \"%s\" with %d buckets\n", name, size);
    return ht;

}

void destroy_hash_table(hash_table_t * ht)
{
    int i;

    dump_hash_table(ht);
    printf("Destroy hash-table \"%s\"...\n", ht->name);

    for (i = 0; i < ht->length; i++) {
        pthread_mutex_destroy(&(ht->buckets[i].mutex));
    }
    free(ht->buckets);
    free(ht);

}

void dump_hash_table(hash_table_t * ht)
{
    int i, flag;
    hash_bucket_t *bkt;
    ckpt_file_t *cfile;

    printf("Dump hash-table \"%s\"...\n", ht->name);

    for (i = 0; i < ht->length; i++) {
        bkt = ht->buckets + i;  // get bucket i
        flag = 1;

        /// walk through bucket i       
        pthread_mutex_lock(&bkt->mutex);

        if (mv2_list_empty(&bkt->list))
            continue;

        mv2_list_for_each_entry(cfile, &(bkt->list), list) {
            if (flag) {
                printf("================   bucket %d:\n", i);
                flag = 0;
            }
            dump_ckpt_file(cfile);
            //printf("\tckpt-%d-rank-%d : chunk_num %d, chunk_to_write %d, has_last_chunk %d\n", 
            //  cfile->ckpt_id, cfile->proc_rank, 
            //  cfile->chunk_num, cfile->chunk_to_write, cfile->has_last_chunk );
        }
        pthread_mutex_unlock(&bkt->mutex);

    }

    printf("============= finish hash-table \"%s\"...\n", ht->name);

}

/**
create a new record representing a ckpt-file {filename, or (ckptid, proc-rank)}, 
then insert this new record into hashtable.
If this record already exists, return it. 
This func is called by fuse_open()
**/
#if    HASH_FROM_FILENAME
ckpt_file_t *hash_table_create_record(hash_table_t * ht, const char *fname)
#else
ckpt_file_t *hash_table_create_record(hash_table_t * ht, int ckpt_id, int proc_rank)
#endif
{
#if    HASH_FROM_FILENAME
    unsigned long x = rabinhash(fname, strlen(fname));
    int namelen = strlen(fname);
#else
    unsigned long x = ((unsigned long) ckpt_id) * 0x9e370001UL + proc_rank;
#endif
    int idx = x % ht->length;

    hash_bucket_t *bkt = ht->buckets + idx;

    ckpt_file_t *cfile = NULL;

    /// lock the bucket
    pthread_mutex_lock(&bkt->mutex);

    /// search for given key in the bucket
    mv2_list_for_each_entry(cfile, &(bkt->list), list) {
#if    HASH_FROM_FILENAME
        if (cfile->filename_len == namelen && strncmp(cfile->filename, fname, namelen) == 0)
#else
        if (cfile->ckpt_id == ckpt_id && cfile->proc_rank == proc_rank)
#endif
        {                       // the record already exists, (created by ib-loop thr)
            pthread_mutex_unlock(&bkt->mutex);
            //dbg("the cfile (%d,%d) exists\n",ckpt_id, proc_rank);
            atomic_inc(&cfile->ref);
            cfile->adv_size = 0;
            return cfile;
        }
    }

    /// if the key not exist, insert it 
    cfile = alloc_ckpt_file();
#if    HASH_FROM_FILENAME
    cfile->filename_len = namelen;
    strncpy(cfile->filename, fname, namelen + 1);
#else
    cfile->ckpt_id = ckpt_id;
    cfile->proc_rank = proc_rank;
#endif

    cfile->rcv_pending_RR_num = 0;
    cfile->rcv_ready_chunk_num = 0;
    cfile->rcv_total_chunk_num = 0;
    cfile->curr_chunk = NULL;
    atomic_set(&cfile->ref, 1);
    // insert this new record into this bucket-tail
    mv2_list_add_tail(&cfile->list, &bkt->list);

    /// unlock the bucket
    pthread_mutex_unlock(&bkt->mutex);
#if    HASH_FROM_FILENAME
    dbg("has created a new cfile (%s)\n", fname);
#else
    dbg("has created a new cfile (%d,%d)\n", ckpt_id, proc_rank);
#endif

    return cfile;
}

/*
Given the key about a ckpt-chunk-data(ckpt id, proc rank, is-last-chunk), 
see if this ckpt file is in hash table, and add a reference to that record.

Called by ib-loop-thread.

NOTE::: caller should be serialized , so be called by the IB-thread.  IB thread poll the RQ, 
and for each chunk-RR-rqst, add a reference to the ckpt-file, then add a work-item to the work-queue.
The io-thread-pool then fetch work-item from the work-queue, and deals with each chunk
*/
#if    HASH_FROM_FILENAME
ckpt_file_t *hash_table_get_record(hash_table_t * ht, const char *fname, int is_last_chunk)
#else
ckpt_file_t *hash_table_get_record(hash_table_t * ht, int ckpt_id, int proc_rank, int is_last_chunk)
#endif
{
#if    HASH_FROM_FILENAME
    unsigned long x = rabinhash(fname, strlen(fname));
    int namelen = strlen(fname);
#else
    unsigned long x = ((unsigned long) ckpt_id) * 0x9e370001UL + proc_rank;
#endif
    int idx = x % ht->length;

    hash_bucket_t *bkt = ht->buckets + idx;

    ckpt_file_t *cfile = NULL;
    int flag = 0;

    /// lock the bucket
    pthread_mutex_lock(&bkt->mutex);

    /// search for given key in the bucket
    mv2_list_for_each_entry(cfile, &(bkt->list), list) {
#if    HASH_FROM_FILENAME
        if (cfile->filename_len == namelen && strncmp(cfile->filename, fname, namelen) == 0)
#else
        if (cfile->ckpt_id == ckpt_id && cfile->proc_rank == proc_rank)
#endif
        {
            flag = 1;
            break;
        }
    }

    if (flag == 1) {            // if the key already exists, add reference
        pthread_mutex_unlock(&bkt->mutex);

        atomic_inc(&cfile->ref);
        /// lock-unlock before accessing the shared record
        pthread_mutex_lock(&cfile->mutex);
        if (is_last_chunk)
            cfile->has_last_chunk = is_last_chunk;
        cfile->rcv_pending_RR_num++;
        cfile->rcv_total_chunk_num++;
        pthread_mutex_unlock(&cfile->mutex);
        //dbg("the cfile (%d,%d) exists\n",ckpt_id, proc_rank);
        return cfile;
    } else {                    /// if the key not exist, create a new one

        /// if the key not exist, insert it 
        cfile = alloc_ckpt_file();
#if    HASH_FROM_FILENAME
        cfile->filename_len = namelen;
        strncpy(cfile->filename, fname, namelen + 1);
#else
        cfile->ckpt_id = ckpt_id;
        cfile->proc_rank = proc_rank;
#endif

        cfile->rcv_pending_RR_num = 1;
        cfile->rcv_ready_chunk_num = 0;
        cfile->rcv_total_chunk_num = 1;
        cfile->curr_chunk = NULL;
        cfile->has_last_chunk = is_last_chunk;
        atomic_set(&cfile->ref, 1);

        // don't open the file in the IB-loop. 
        // This loop is in critical path, thus cannot invoke any blocking calls.  
        // Let the first io-thread open the file

        // insert this new record into this bucket-tail
        mv2_list_add_tail(&cfile->list, &bkt->list);
    }
    /// If comes here: has inserted a new record into hash-table
    pthread_mutex_unlock(&bkt->mutex);
#if    HASH_FROM_FILENAME
    dbg("Create a new cfile (%s)\n", fname);
#else
    dbg("Create a new cfile (%d,%d)\n", ckpt_id, proc_rank);
#endif
    return cfile;
}

/*
find an entry matching (ckpt id, proc rank), decrease its reference "chunk_to_write" for this entry
If all chunks belonging to this entry have finished, including the last_chunk, 
then remove this entry from hash table
Input: 
    is_release:  0: called from io-thr; 1: called by fuse-release to close the file
Return: 
    0:  after deref, some other chunks remains to be written, 
            or the last-chunk hasn't been written
    1:  all chunks of this ckpt-file have been written. Have released this record
*/
static int ckfile_finish_cnt = 0;
int hash_table_put_record(hash_table_t * ht, ckpt_file_t * cfile, int is_release)
{
    pthread_mutex_lock(&cfile->mutex);
    //dbg("will release ckfile: %s, ref=%d, is_rel=%d\n",
    //        cfile->filename, atomic_read(&cfile->ref), is_release );
    //dump_ckpt_file(cfile);
    dbg("Enter:: cfile %s: ref = %d, fd=%d, can-rel=%d, has-last-chunk=%d\n", cfile->filename, atomic_read(&cfile->ref), cfile->fd, cfile->can_release, cfile->has_last_chunk);

    if (atomic_read(&cfile->ref) > 0)
        atomic_dec(&cfile->ref);

    if (atomic_read(&cfile->ref) > 0)   //!atomic_dec_and_test( &cfile->ref) )  // ref > 0
    {                           // the ref not zero
        //if( !is_release ) // calling from io-thr
        //cfile->rcv_pending_RR_num--;
        //dbg("cfile (%d, %d) ref = %d, still active\n", cfile->ckpt_id, cfile->proc_rank, cfile->ref );        
        pthread_mutex_unlock(&cfile->mutex);
        return 0;
    }

    if (!is_release) {
        dbg("******  cfile %s: (%d, %d) ref = %d, but not release\n", __func__, cfile->ckpt_id, cfile->proc_rank, cfile->ref);
        pthread_mutex_unlock(&cfile->mutex);
        return 0;
    }
    // data isn't complete, and OS still has reference, do nothing
    //if(!cfile->can_release && !cfile->has_last_chunk ) /// (can-rel=0, last-chunk=0)
    if (!cfile->can_release ||  // user hasn't call close()
        (cfile->adv_size > 0 && !cfile->has_last_chunk) // user has posted some data to file, but hasn't finished writing
        )                       ///// (can-rel=0, last-chunk=0)
    {
        dbg("can't rel(%s):: cfile->adv_size=%d, cfile->can_rel=%d, cfile->has_last_chk=%d\n", cfile->filename, cfile->adv_size, cfile->can_release, cfile->has_last_chunk);
        pthread_mutex_unlock(&cfile->mutex);
        return 0;
    }
    /// Has finished all data, but OS still has reference  (can-rel=0, last-chunk=1)
    // can free all data, close the "fd", but keep the "cfile" struct
    if (!cfile->can_release) {
        if (!mv2_list_empty(&cfile->chunk_list)) {
            error("free cfile: ckpt-%d-proc-%d: chunk-list not empty!!!\n", cfile->ckpt_id, cfile->proc_rank);
            ckpt_chunk_t *ck;
            struct list_head *pos, *n;
            mv2_list_for_each_safe(pos, n, &(cfile->chunk_list)) {
                ck = mv2_list_entry(pos, ckpt_chunk_t, list);
                mv2_list_del(&ck->list);
                ckpt_free_chunk(ck);
                free(ck);
            }
        }
        if (cfile->fd > 0) {
            dbg("close file-fd %s\n", cfile->filename);
            close(cfile->fd);
            cfile->fd = -1;
        }
        pthread_mutex_unlock(&cfile->mutex);
        return 0;
    }
    ///// OS has no reference to it. (can-rel=1)
    ///  Will free the file, regardless of last-chunk
    //dbg("cfile %s: (%d, %d) ref = %d, will free it\n", 
    //  cfile->filename, cfile->ckpt_id, cfile->proc_rank, cfile->ref );

    /// remove the cfile from hash-table
    mv2_list_del(&cfile->list);

    /// free the ckpt-file record
    dbg("\n*** Now, free -- (%s)(%d,%d) : total chunk_num %d, pending chunks %d, has_last_chunk %d, adv-size=%d\n\n",
        cfile->filename, cfile->ckpt_id, cfile->proc_rank, cfile->rcv_total_chunk_num, cfile->rcv_pending_RR_num, cfile->has_last_chunk, cfile->adv_size);

    free_ckpt_file(cfile);
    return 1;

#if    HASH_FROM_FILENAME
    unsigned long x = rabinhash(cfile->filename, cfile->filename_len);
#else
    unsigned long x = ((unsigned long) cfile->ckpt_id) * 0x9e370001UL + cfile->proc_rank;
#endif

    int idx = x % ht->length;
    hash_bucket_t *bkt = ht->buckets + idx;

    ////////////////////////
    /////??? when is a ckpt-file finished: has_last_chunk=1 && rcv_pending_RR_num==0 && rcv_ready_chunk_num==0
    pthread_mutex_lock(&bkt->mutex);
    if (atomic_dec_and_test(&cfile->chunk_to_write) // true if: the counter becoems 0
        && cfile->has_last_chunk) {
        mv2_list_del(&cfile->list);
        /// unlock
        pthread_mutex_unlock(&bkt->mutex);

        printf("\tRelease ckpt-%d-rank-%d : chunk_num %d, chunk_to_write %d, has_last_chunk %d, size=%ld\n",
               cfile->ckpt_id, cfile->proc_rank, cfile->chunk_num, atomic_read(&cfile->chunk_to_write), cfile->has_last_chunk, cfile->write_size);
        /// close the ckpt-file
        if (cfile->fd) {
            fsync(cfile->fd);
            close(cfile->fd);
            ckfile_finish_cnt++;
            if (ckfile_finish_cnt == 8) {
                printf("%s: Has finished %d ckptfile, publish msg...\n", __func__, ckfile_finish_cnt);
                //ftb_publish_msg("CR_FTB_MIGRATE_PIC");
                ckfile_finish_cnt = 0;
            }
        }
        /// release the cfile-record
        pthread_mutex_destroy(&cfile->mutex);
        free(cfile);
        return 1;
    } else {
        pthread_mutex_unlock(&bkt->mutex);
        return 0;
    }
}

/**
Given file-name(fname), locate the bucket where this record resides;

if the record with given file-name exists, lock the bucket and return a pointer of the bucket;
if the record with (fname) doesn't exist, unlock the bucket and return NULL;
**/
hash_bucket_t *htbl_find_lock_bucket(hash_table_t * ht, const char *fname, int namelen)
{
#if    HASH_FROM_FILENAME
    unsigned long x = rabinhash(fname, namelen);
#else
    unsigned long x = ((unsigned long) cfile->ckpt_id) * 0x9e370001UL + cfile->proc_rank;
#endif

    int idx = x % ht->length;
    hash_bucket_t *bkt = ht->buckets + idx;

    ckpt_file_t *cfile = NULL;
    int flag = 0;

    /// lock the bucket
    pthread_mutex_lock(&bkt->mutex);

    /// search for given key in the bucket
    mv2_list_for_each_entry(cfile, &(bkt->list), list) {
#if    HASH_FROM_FILENAME
        if (cfile->filename_len == namelen && strncmp(cfile->filename, fname, namelen) == 0)
#else
        if (cfile->ckpt_id == ckpt_id && cfile->proc_rank == proc_rank)
#endif
        {
            flag = 1;
            break;
        }
    }

    if (flag == 1) {
        // hold the bkt lock
        return bkt;
    } else {
        pthread_mutex_unlock(&bkt->mutex);
        return NULL;
    }

}

void htbl_unlock_bucket(hash_bucket_t * bkt)
{
    if (bkt)
        pthread_mutex_unlock(&bkt->mutex);
}

#endif
