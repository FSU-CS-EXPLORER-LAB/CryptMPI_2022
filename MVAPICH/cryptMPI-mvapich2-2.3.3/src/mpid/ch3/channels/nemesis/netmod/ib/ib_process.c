/*! \file */
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
/* #include "ib_utils.h" */


#include "ib_process.h"
#include "ib_hca.h"


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_init_process_info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 *  Initialize the process information.
 *
 * See alloc_process_init_info() in
 * mvapich2/trunk/src/mpid/ch3/channels/mrail/src/gen2/rdma_iba_priv.c
 * init mv2_nem_dev_info_t->conn_info
 *
 */
int MPID_nem_ib_init_process_info(int pg_rank, MPIDI_PG_t * pg_p)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_INIT_PROCESS_INFO);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_INIT_PROCESS_INFO);

    /* --------------------------- *
     * Set the process information.*
     * --------------------------- */
    memset(&process_info, 0, sizeof( MPID_nem_ib_process_info_t ));
    process_info.rank        = pg_rank;
    process_info.pg          = pg_p;
    process_info.rails       = 1;

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_INIT_PROCESS_INFO);
    return MPI_SUCCESS;

}

