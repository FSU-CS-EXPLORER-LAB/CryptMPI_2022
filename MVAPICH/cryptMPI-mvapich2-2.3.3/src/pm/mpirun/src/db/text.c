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

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static size_t nalloc = 0;
static size_t nmemb = 0;

static char ** db = NULL;

static int increase_allocation (void)
{
    size_t new_nalloc = (nalloc ? nalloc << 1 : 256);
    char ** new_db;

    if (nalloc > new_nalloc)  return -1;

    new_db = realloc(db, new_nalloc * sizeof(char *));
    if (NULL == new_db) return -1;

    db = new_db;
    nalloc = new_nalloc;

    return 0;
}

char const * db_add_text (char const * const text)
{
    if (NULL == text) return NULL;
    if (SIZE_MAX == nmemb) return NULL;

    if (nalloc == nmemb) {
        if (increase_allocation()) return NULL;
    }
    
    return db[nmemb++] = strdup(text);
}

void db_clear (void)
{
    int i = nmemb;

    nmemb = nalloc = 0;
    if (NULL == db) return;

    while (i--) free(db[i]);
    free(db);
    db = NULL;
}

