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
#include "crfs.h"

int crfs_mode;                  //= MODE_WRITEAGGRE; 
int mig_role;                   // = ROLE_INVAL;

int crfs_init_imp(int mode)
{
    void *p = 0;
    if (crfs_mode == MODE_WRITEAGGRE)
        ibcli_main_entry(p);
    else if (crfs_mode == MODE_MIG)
        ibsrv_main_entry(p);
    else {
        fprintf(stderr, "%s: Error: incorrect mode %d\n", __func__, crfs_mode);
        return -1;
    }

    return 0;
}

int crfs_destroy_imp(int mode)
{
    if (crfs_mode == MODE_WRITEAGGRE)
        ibcli_main_exit();
    else if (crfs_mode == MODE_MIG)
        ibsrv_main_exit();
    else {
        fprintf(stderr, "%s: Error: incorrect mode %d\n", __func__, crfs_mode);
        return -1;
    }

    return 0;
}

#endif
