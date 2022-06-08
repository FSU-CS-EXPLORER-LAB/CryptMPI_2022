/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#ifndef DB_SUFFIXLIST_H
#define DB_SUFFIXLIST_H 1

typedef struct  suffix_range_t *            sr_handle;
typedef struct  suffix_list_t *             sl_handle;
typedef struct  suffix_list_iterator_t *    sl_iterator;

sr_handle       sr_create       (char const * const, char const * const);
void            sr_destroy      (sr_handle);

sl_handle       sl_create       (sr_handle);
void            sl_destroy      (sl_handle);
sl_handle       sl_insert       (sl_handle, sr_handle);

sl_iterator     sli_create      (sl_handle);
char const *    sli_next        (sl_iterator);
int             sli_hasNext     (sl_iterator);

#endif
