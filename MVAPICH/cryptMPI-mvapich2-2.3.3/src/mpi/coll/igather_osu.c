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

#include "igather_tuning.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)

int (*MV2_Igather_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

int (*MV2_Igather_intra_node_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;


#undef FUNCNAME
#define FUNCNAME MPIR_Igather_tune_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Igather_tune_helper_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int is_homogeneous ATTRIBUTE((unused));

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous);

    mpi_errno = MV2_Igather_function(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, root, comm_ptr, s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Igather_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Igather_intra_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous ATTRIBUTE((unused));
    MPI_Aint recvtype_size, sendtype_size, nbytes;
    
    int rank = comm_ptr->rank;
    int two_level_igather = 1;
    int range = 0;
    int range_threshold = 0;
    int range_threshold_intra = 0;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
    comm_size = comm_ptr->local_size;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous); /* we don't handle the hetero case right now */
    if (rank == root) {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcount * recvtype_size;
    } else {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcount * sendtype_size;
    }

    // Search for some parameters regardless of whether subsequent selected
    // algorithm is 2-level or not
    
    // Search for the corresponding system size inside the tuning table
    while ((range < (mv2_size_igather_tuning_table - 1)) &&
           (comm_size > mv2_igather_thresholds_table[range].numproc)) {
        range++;
    }
    
    // Search for corresponding inter-leader function
    while ((range_threshold < (mv2_igather_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_igather_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_igather_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    // Search for corresponding intra-node function
    
    // Commenting this for the time being as none of
    // the algorithms are 2-level
    /*
    while ((range_threshold_intra <
            (mv2_igather_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_igather_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_igather_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }
    */

    MV2_Igather_function =
        mv2_igather_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Igather_function;

    MV2_Igather_intra_node_function =
        mv2_igather_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Igather_function;

    /* There are currently no two-level nb-gather functions hence
       setting to 0 by default */
    two_level_igather = 
        mv2_igather_thresholds_table[range].is_two_level_igather[range_threshold]; 
    if (1 != two_level_igather) {
        mpi_errno = MPIR_Igather_tune_helper_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, root, comm_ptr, s);
    }
    else {
        /* Code path should not enter this with the current algorithms*/
    }

    return mpi_errno;
}
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */

#undef FUNCNAME
#define FUNCNAME MPIR_Igather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Igather_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPID_INTRACOMM) {    
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
        mpi_errno = MPIR_Igather_intra_MV2(sendbuf, sendcount, sendtype, recvbuf,
					   recvcount, recvtype, root, comm_ptr, s);
#else
        mpi_errno = MPIR_Igather_intra(sendbuf, sendcount, sendtype, recvbuf,
                                       recvcount, recvtype, root, comm_ptr, s);
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
    }
    else {
        mpi_errno = MPIR_Igather_inter(sendbuf, sendcount, sendtype, recvbuf,
                                       recvcount, recvtype, root, comm_ptr, s);
    }

    return mpi_errno;
}
