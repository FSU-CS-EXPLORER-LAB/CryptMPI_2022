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

#ifndef _COMMON_CKPT_H
#define _COMMON_CKPT_H


#include "mpichconf.h"

#ifdef CKPT

#include "ftb_helper.h"

#define MAX_CR_MSG_LEN  256
#define CRU_MAX_KEY_LEN 64
#define CRU_MAX_VAL_LEN 64

#define DEFAULT_CHECKPOINT_FILENAME "/tmp/ckpt"
#define CR_SESSION_MAX  16

#define CR_MAX_FILENAME 256


// Defined in mpirun_util.c
extern char *CR_MPDU_getval(const char *, char *, int);
extern int CR_MPDU_parse_keyvals(char *);
extern int CR_MPDU_readline(int, char *, int);
extern int CR_MPDU_writeline(int, char *);

#ifdef CR_FTB


#endif                          /* CR_FTB */
#endif                          /* CKPT */

#endif                          /* _COMMON_CKPT_H */
