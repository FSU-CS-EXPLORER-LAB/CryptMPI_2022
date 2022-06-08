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

#include "mpidi_ch3_impl.h"

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Comm_spawn
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Comm_spawn(const char *command, const char *argv[],
                         const int maxprocs, MPI_Info info, const int root,
                         MPID_Comm * comm, MPID_Comm * intercomm,
                         int array_of_errcodes[])
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_COMM_SPAWN);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_COMM_SPAWN);
    int mpi_errno =
        MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME,
                             __LINE__, MPI_ERR_OTHER, "**notimpl", 0);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_COMM_SPAWN);
    return mpi_errno;
}
