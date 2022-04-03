/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */
#ifndef MPIT_MV2_H
#define MPIT_MV2_H 1

void MPIT_REGISTER_MV2_VARIABLES (void);
void MPIT_MEM_REGISTER_PVARS (void);
void MPIT_FREE_MV2_VARIABLES (void);

void * MPIT_malloc (size_t, int, char const *);
void * MPIT_calloc (size_t, size_t, int, char const *);
void MPIT_free (void *, int, char const *);
char * MPIT_strdup (const char *, int, char const *);
void * MPIT_realloc (void *, size_t, int, char const *);
int MPIT_memalign (void **, size_t, size_t, int, char const *);

#endif
