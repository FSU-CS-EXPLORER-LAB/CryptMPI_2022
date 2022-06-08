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

#ifndef __DEBUG__
#define __DEBUG__

#define MYDBG 0
// defined in mpispawn.c
extern int mt_id;
#if MYDBG
#define dbg(fmt, args... )    do{ \
    fprintf(stderr,"[mt_%d]: %s: " fmt, mt_id, __func__, ##args); fflush(stderr); }while(0)
#else
#define dbg(fmt, args... )
#endif

#define error(fmt, args... )            \
    do{    fprintf(stderr, "%s:   ", __func__ );    \
        fprintf(stderr, fmt, ## args); \
    }while(0)

#define err(fmt, args... )            \
    do{    fprintf( stderr, "%s:   ", __func__ );    \
        fprintf( stderr, fmt, ## args); \
    }while(0)

#undef MYDBG

//#define my_error(fmt, args...) printf(fmt , ## args)
//#define my_dbg(fmt, args...)  printf(fmt, ## args)

#endif                          /// end of __DEBUG__
