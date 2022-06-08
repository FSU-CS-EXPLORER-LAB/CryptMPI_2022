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

#include "mpidimpl.h"
#include "ib_ds_hash.h"
#include "ib_errors.h"


/** hash - A simple hash function
 *  Too simple really, but this
 *  works great for now
 */

static uint32_t hash(uint64_t key, uint32_t size)
{
    uint32_t hash_value;

    hash_value = key % size;

    return hash_value;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_init_hash_table
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_init_hash_table(
        MPID_nem_ib_hash_table_ptr_t table,
        uint32_t nentries)
{
    int mpi_errno = MPI_SUCCESS;

    table->entries = MPIU_Malloc(
            sizeof(MPID_nem_ib_hash_elem_t) * nentries);
    table->num_entries = nentries;

    if(NULL == table->entries) {
        MPIU_CHKMEM_SETERR(mpi_errno,
                sizeof(MPID_nem_ib_hash_elem_t) * nentries,
                "IB Module Hash Table");
    }

    memset(table->entries, 0,
            sizeof(MPID_nem_ib_hash_elem_t) * nentries);

    pthread_mutex_init(&table->hash_table_lock, NULL);

    /*   fn_exit: */
           return mpi_errno;
    /*   fn_fail:
           goto fn_exit;
           */
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_insert_hash_elem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_insert_hash_elem(
        MPID_nem_ib_hash_table_ptr_t table,
        uint64_t key, void *data, uint32_t uniq)
{
    int mpi_errno = MPI_SUCCESS;
    uint32_t hash_index;
    MPID_nem_ib_hash_elem_ptr_t start_elem;
    MPID_nem_ib_hash_elem_ptr_t new_elem;
    MPID_nem_ib_hash_elem_ptr_t elem;

    MPIU_Assert(NULL != table);

    pthread_mutex_lock(&table->hash_table_lock);

    hash_index = hash(key, table->num_entries);

    /* Note that the first element is allocated
     * at the beginning, so this is guaranteed
     * to be non-null */
    start_elem = &table->entries[hash_index];

    MPIU_Assert(start_elem != NULL);

    /* Walk to end of list in this hash slot */
    elem = start_elem;
    while(elem->next != NULL) {
        elem = elem->next;
    }

    /* Insert the element */
    new_elem = MPIU_Malloc(sizeof(MPID_nem_ib_hash_elem_t));

    if(NULL == new_elem) {
        MPIU_CHKMEM_SETERR(mpi_errno,
                sizeof(MPID_nem_ib_hash_elem_t),
                "IB Module Hash Table New Element");
    }

    memset(new_elem, 0, sizeof(MPID_nem_ib_hash_elem_t));

    new_elem->data = data;
    new_elem->uniq = uniq;
    new_elem->key = key;
    new_elem->next = NULL;
    new_elem->prev = elem;

    elem->next = new_elem;

    NEM_IB_DBG("Inserted elem key %lu, uniq %u, hash index %u",
            key, uniq, hash_index);

    pthread_mutex_unlock(&table->hash_table_lock);

    /*   fn_exit: */
           return mpi_errno;
    /*   fn_fail:
           goto fn_exit;
           */
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_lookup_hash_table
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void* MPID_nem_ib_lookup_hash_table(
        MPID_nem_ib_hash_table_ptr_t table,
        uint64_t key, uint32_t unique_id)
{
    uint32_t hash_index;
    void *data = NULL;
    MPID_nem_ib_hash_elem_ptr_t start_elem;
    MPID_nem_ib_hash_elem_ptr_t elem;

    pthread_mutex_lock(&table->hash_table_lock);

    hash_index = hash(key, table->num_entries);

    NEM_IB_DBG("Got hash_index %u, key %lu, uniq %u",
            hash_index, key, unique_id);

    start_elem = &table->entries[hash_index];

    NEM_IB_DBG("");

    /* The first element is just a place holder */
    elem = start_elem->next;

    while(elem != NULL) {

        if((elem->key == key) && (elem->uniq == unique_id)) {
            data = elem->data;
            break;
        }
        elem = elem->next;
    }

    pthread_mutex_unlock(&table->hash_table_lock);

    return data;
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_finalize_hash_table
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPID_nem_ib_finalize_hash_table(
        MPID_nem_ib_hash_table_ptr_t table)
{
    int i;
    MPID_nem_ib_hash_elem_ptr_t start_elem;
    MPID_nem_ib_hash_elem_ptr_t elem, next_elem;

    pthread_mutex_lock(&table->hash_table_lock);

    MPIU_Assert(table->entries != NULL);

    for(i = 0; i < table->num_entries; i++) {

        start_elem = &table->entries[i];

        /* Walk through the list freeing
         * elements as we go */
        elem = start_elem->next;

        while(elem != NULL) {
            next_elem = elem->next;
            MPIU_Free(elem);
            elem = next_elem;
        }
    }

    pthread_mutex_unlock(&table->hash_table_lock);

    MPIU_Free(table->entries);
}


