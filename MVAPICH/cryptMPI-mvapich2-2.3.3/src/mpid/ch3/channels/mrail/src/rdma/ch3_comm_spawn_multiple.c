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
#include "upmi.h"

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Comm_spawn_multiple
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Comm_spawn_multiple(int count, char **commands,
                                  char ***argvs, int *maxprocs,
                                  MPID_Info ** info_ptrs, int root,
                                  MPID_Comm * comm_ptr, MPID_Comm
                                  ** intercomm, int *errcodes)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_COMM_SPAWN_MULTIPLE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_COMM_SPAWN_MULTIPLE);
    int mpi_errno = MPI_SUCCESS;
    char port_name[MPI_MAX_PORT_NAME];
    int *info_keyval_sizes, i;
    PMI_keyval_t **info_keyval_vectors, preput_keyval_vector;

    if (comm_ptr->rank == root)
    {
        if ((info_keyval_sizes = (int *) MPIU_Malloc(count * sizeof(int))) == NULL)
        {
            MPIU_CHKMEM_SETERR(mpi_errno, count * sizeof(int), "key value sizes array");
        }

        /* TEMPORARILY set all user-provided info to NULL. PMI is not
           using it anyway. */
        for (i = 0; i < count; i++)
        {
            info_keyval_sizes[i] = 0;
        }

        info_keyval_vectors = NULL;

        if ((mpi_errno = MPIDI_CH3_Open_port(port_name)) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        preput_keyval_vector.key = "PARENT_ROOT_PORT_NAME";
        preput_keyval_vector.val = port_name;

        if ((mpi_errno = UPMI_JOB_SPAWN(
            count,
            (const char**) commands,
            NULL,
            (const char***) argvs,
            maxprocs,
            info_keyval_sizes,
            (const void**) info_keyval_vectors,
            1,
            (const void*) &preput_keyval_vector,
            NULL,
            0,
            errcodes)) != UPMI_SUCCESS)
        {
#ifdef HAVE_ERROR_CHECKING
            mpi_errno = MPIR_Err_create_code(
                MPI_SUCCESS,
                MPIR_ERR_FATAL,
                FCNAME,
                __LINE__,
                MPI_ERR_OTHER,
                "**pmi_spawn_multiple",
                "**pmi_spawn_multiple %d",
                mpi_errno);
#endif /* HAVE_ERROR_CHECKING */
        }

        MPIU_Free(info_keyval_sizes);
    }

    if ((mpi_errno = MPIDI_CH3_Comm_accept(
        port_name,
        root,
        comm_ptr,
        intercomm)) != MPI_SUCCESS)
    {
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_COMM_SPAWN_MULTIPLE);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}
