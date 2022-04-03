/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
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

#include "mpidimpl.h"
#include "upmi.h"

/*
 * MPID_Get_universe_size - Get the universe size from the process manager
 *
 * Notes: The ch3 device requires that the PMI routines are used to 
 * communicate with the process manager.  If a channel wishes to 
 * bypass the standard PMI implementations, it is the responsibility of the
 * channel to provide an implementation of the PMI routines.
 */
#undef FUNCNAME
#define FUNCNAME MPID_Get_universe_size
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Get_universe_size(int  * universe_size)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno = UPMI_SUCCESS;

    pmi_errno = UPMI_GET_UNIVERSE_SIZE(universe_size);
    if (pmi_errno != UPMI_SUCCESS) {
	MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, 
			     "**pmi_get_universe_size",
			     "**pmi_get_universe_size %d", pmi_errno);
    }
    if (*universe_size < 0)
    {
	*universe_size = MPIR_UNIVERSE_SIZE_NOT_AVAILABLE;
    }
    
fn_exit:
    return mpi_errno;

    /* --BEGIN ERROR HANDLING-- */
fn_fail:
    *universe_size = MPIR_UNIVERSE_SIZE_NOT_AVAILABLE;
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
