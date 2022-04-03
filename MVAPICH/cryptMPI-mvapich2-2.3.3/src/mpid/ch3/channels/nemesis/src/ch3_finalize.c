/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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

/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpid_nem_impl.h"
#ifdef _OSU_MVAPICH_
#include "coll_shmem.h"
#endif /* _OSU_MVAPICH_ */

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Finalize(void)
{
    extern int finalize_coll_comm;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_FINALIZE);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_FINALIZE);

    mpi_errno = MPIDI_CH3I_Progress_finalize();
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
    mpi_errno = MPID_nem_finalize();
    if (mpi_errno) MPIR_ERR_POP (mpi_errno);

#ifdef _OSU_MVAPICH_
    if (mv2_enable_shmem_collectives || finalize_coll_comm == 1) {
        /* Freeing up shared memory collective resources*/
        mpi_errno = MPIDI_CH3I_SHMEM_COLL_finalize(MPID_nem_mem_region.local_rank,
                        MPID_nem_mem_region.num_local);
        if (mpi_errno) MPIR_ERR_POP (mpi_errno);

        MV2_collectives_arch_finalize();
    }
#endif

 fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_FINALIZE);
    return mpi_errno;
}
