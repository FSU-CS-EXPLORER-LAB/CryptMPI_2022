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

#ifndef OPENHASH_H
#define OPENHASH_H

#include <pthread.h>
#include <semaphore.h>

#include "list.h"
#include "atomic.h"
#include "ckpt_file.h"

#define     MAX_HASH_LENGTH    (128)

typedef struct hash_bucket_s {
    pthread_mutex_t mutex;
    struct list_head list;      // list of entries in a bucket

} hash_bucket_t;

typedef struct hash_table_s {
    int length;                 // length of the table,i.e., num of hash-buckets  
    hash_bucket_t *buckets;     // [MAX_HASH_LENGTH];  // 
    char name[32];

} hash_table_t;

//////////////////////////

hash_table_t *create_hash_table(int size, char *name);
void destroy_hash_table(hash_table_t * ht);

void dump_hash_table(hash_table_t * ht);

#if    HASH_FROM_FILENAME
ckpt_file_t *hash_table_create_record(hash_table_t * ht, const char *fname);
#else
ckpt_file_t *hash_table_create_record(hash_table_t * ht, int ckpt_id, int proc_rank);
#endif

#if    HASH_FROM_FILENAME
ckpt_file_t *hash_table_get_record(hash_table_t * ht, const char *fname, int is_last_chunk);
#else
ckpt_file_t *hash_table_get_record(hash_table_t * ht, int ckpt_id, int proc_rank, int is_last_chunk);
#endif

//int hash_table_put_record(hash_table_t* ht, ckpt_file_t* cfile);
int hash_table_put_record(hash_table_t * ht, ckpt_file_t * cfile, int is_release);

hash_bucket_t *htbl_find_lock_bucket(hash_table_t * ht, const char *fname, int namelen);
void htbl_unlock_bucket(hash_bucket_t * bkt);

#endif                          // OPENHASH_H
