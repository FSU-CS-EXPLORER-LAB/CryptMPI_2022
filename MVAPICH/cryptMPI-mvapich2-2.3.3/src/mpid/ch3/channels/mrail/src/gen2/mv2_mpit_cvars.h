/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include "mpichconf.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <infiniband/verbs.h>
#include "rdma_impl.h"
#include "vbuf.h"
#include "ibv_param.h"
#include "sysreport.h"
#include "smp_smpi.h"
#include "mv2_utils.h"
#include "upmi.h"
#include <inttypes.h>
 
#define MV2_CVAR_FATAL_ERR 1
#define MV2_CVAR_NONFATAL_ERR 0

/* structure to hold values to pass to access a CVAR */
typedef struct mv2_mpit_cvar_access {
    int cvar_index;
    char* cvar_name;
    MPI_T_cvar_handle cvar_handle;
    int default_cvar_value;
    int skip_if_default_has_set;
    int error_type;
    int check4_associate_env_conflict;
    char* env_name;
    char* env_conflict_error_msg;
    int check_max;
    int max_value;
    int check_min;
    int min_value;
    char* boundary_error_msg;
    int* skip;
    int* value;
} mv2_mpit_cvar_access_t;

void mv2_free_cvar_handles();
extern int mv2_read_and_check_cvar(mv2_mpit_cvar_access_t wrapper);
