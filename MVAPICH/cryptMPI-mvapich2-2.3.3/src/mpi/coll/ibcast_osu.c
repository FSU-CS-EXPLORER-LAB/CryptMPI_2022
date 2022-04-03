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

#include "mpiimpl.h"
#include "collutil.h"

#include "ibcast_tuning.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)


int (*MV2_Ibcast_function) (void *buffer, int count, MPI_Datatype datatype, int root,
                            MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

int (*MV2_Ibcast_intra_node_function) (void *buffer, int count, MPI_Datatype datatype,
                                       int root, MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;


#undef FUNCNAME
#define FUNCNAME MPIR_Ibcast_tune_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Ibcast_tune_helper_MV2(void *buffer, int count, MPI_Datatype datatype,
                                       int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous ATTRIBUTE((unused));

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous);
    comm_size = comm_ptr->local_size;

    if (MV2_Ibcast_function == MPIR_Ibcast_scatter_rec_dbl_allgather) {
        if (MPIU_is_pof2(comm_size, NULL)) {
            mpi_errno = MPIR_Ibcast_scatter_rec_dbl_allgather(buffer, count,
                                                              datatype, root, comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }
        else {
            mpi_errno = MPIR_Ibcast_scatter_ring_allgather(buffer, count,
                                                           datatype, root, comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }
    }
    else {
        mpi_errno = MV2_Ibcast_function(buffer, count, datatype, root, comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Ibcast_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ibcast_intra_MV2(void *buffer, int count, MPI_Datatype datatype, int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous ATTRIBUTE((unused));
    MPI_Aint type_size, nbytes;
    
    int two_level_ibcast = 1;
    int range = 0;
    int range_threshold = 0;
    int range_threshold_intra = 0;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous); /* we don't handle the hetero case right now */

    comm_size = comm_ptr->local_size;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = type_size * count;

    // Search for some parameters regardless of whether subsequent selected
    // algorithm is 2-level or not
    
    // Search for the corresponding system size inside the tuning table
    while ((range < (mv2_size_ibcast_tuning_table - 1)) &&
           (comm_size > mv2_ibcast_thresholds_table[range].numproc)) {
        range++;
    }

    /* If tuning table size = 0, i.e the tables are not populated yet */
    /*
    if (0 == range) {
        if ((nbytes < MPIR_CVAR_BCAST_SHORT_MSG_SIZE) ||
            (comm_size < MPIR_CVAR_BCAST_MIN_PROCS))
            {
                mpi_errno = MPIR_Ibcast_binomial(buffer, count, datatype, root, comm_ptr, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
        else
            {
                if ((nbytes < MPIR_CVAR_BCAST_LONG_MSG_SIZE) && (MPIU_is_pof2(comm_size, NULL))) {
                    mpi_errno = MPIR_Ibcast_scatter_rec_dbl_allgather(buffer, count, datatype, root, comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
                else {
                    mpi_errno = MPIR_Ibcast_scatter_ring_allgather(buffer, count, datatype, root, comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
            }
        goto fn_exit;
    }
    */
    
    // Search for corresponding inter-leader function
    while ((range_threshold < (mv2_ibcast_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_ibcast_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_ibcast_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    // Search for corresponding intra-node function
    
    // Commenting this for the time being as none of
    // the algorithms are 2-level
    /*
    while ((range_threshold_intra <
            (mv2_ibcast_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_ibcast_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_ibcast_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }
    */

    MV2_Ibcast_function =
        mv2_ibcast_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Ibcast_function;

    MV2_Ibcast_intra_node_function =
        mv2_ibcast_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Ibcast_function;

    /* There are currently no two-level nb-bcast functions hence
       setting to 0 by default */
    two_level_ibcast = 
        mv2_ibcast_thresholds_table[range].is_two_level_ibcast[range_threshold]; 
    if (1 != two_level_ibcast) {
        mpi_errno = MPIR_Ibcast_tune_helper_MV2(buffer, count, datatype, root, comm_ptr, s);
    }
    else {
        /* Code path should not enter this with the current algorithms*/
    }

    return mpi_errno;
}
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */

#undef FUNCNAME
#define FUNCNAME MPIR_Ibcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ibcast_MV2(void *buffer, int count, MPI_Datatype datatype, int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPID_INTRACOMM) {
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
        mpi_errno = MPIR_Ibcast_intra_MV2(buffer, count, datatype, root, comm_ptr, s);
#else
        mpi_errno = MPIR_Ibcast_intra(buffer, count, datatype, root, comm_ptr, s);
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
    }
    else {
        mpi_errno = MPIR_Ibcast_inter(buffer, count, datatype, root, comm_ptr, s);
    }

    return mpi_errno;
}
