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

#include "mpirun_dbg.h"

/* Start mpirun_rsh totalview integration */

struct MPIR_PROCDESC *MPIR_proctable = 0;
int MPIR_proctable_size = 0;
int MPIR_i_am_starter = 1;
int MPIR_debug_state = 0;
char *MPIR_dll_name = "MVAPICH2";

/* End mpirun_rsh totalview integration */

int debug_on = 0;
int param_debug = 0;
int use_totalview = 0;

/**
 *  Totalview intercepts MPIR_Breakpoint
 */
int MPIR_Breakpoint(void)
{
    return 0;
}
