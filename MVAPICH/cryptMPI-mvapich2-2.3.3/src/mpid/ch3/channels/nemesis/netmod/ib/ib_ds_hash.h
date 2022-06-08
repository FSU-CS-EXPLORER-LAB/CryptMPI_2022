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

#ifndef IB_DS_HASH_H
#define IB_DS_HASH_H

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>



/*
#define INIT_NEM_IB_PROC_DESC(_rank) {                          \
    if(gethostname(me.hostname, MAX_HOSTNAME_LEN)) {   \
        perror("gethostname");                                  \
    }                                                           \
    me.rank = _rank;                                            \
};

*/


/* A description of the process
 * is kept in this data structure.
 * Its useful to dereference when
 * printing our error/debug messages
 * /

typedef struct {
    char            hostname[MAX_HOSTNAME_LEN];
    int             rank;
} MPID_nem_ib_proc_desc_t;

MPID_nem_ib_proc_desc_t me;
*/

/**
 * Simple data structures for a Hash
 * table implementation
 */
typedef struct _hash_elem {
    void                    *data;
    uint32_t                uniq;
    uint64_t                key;
    struct _hash_elem       *prev;
    struct _hash_elem       *next;
} MPID_nem_ib_hash_elem_t, *MPID_nem_ib_hash_elem_ptr_t;

typedef struct {
    MPID_nem_ib_hash_elem_ptr_t      entries;
    uint32_t                                num_entries;
    pthread_mutex_t                         hash_table_lock;
} MPID_nem_ib_hash_table_t, *MPID_nem_ib_hash_table_ptr_t;



int MPID_nem_ib_init_hash_table(
        MPID_nem_ib_hash_table_ptr_t table,
        uint32_t nentries);

int MPID_nem_ib_insert_hash_elem(
        MPID_nem_ib_hash_table_ptr_t table,
        uint64_t key,
        void *data,
        uint32_t len);

void* MPID_nem_ib_lookup_hash_table(
        MPID_nem_ib_hash_table_ptr_t table,
        uint64_t key, uint32_t unique_id);

void MPID_nem_ib_finalize_hash_table(
        MPID_nem_ib_hash_table_ptr_t table);


#endif  /* IB_DS_HASH_H */
