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

#include "iallgather_tuning.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)

int (*MV2_Iallgather_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

int (*MV2_Iallgather_intra_node_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;



#undef FUNCNAME
#define FUNCNAME MPIR_Iallgather_tune_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Iallgather_tune_helper_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int is_homogeneous  ATTRIBUTE((unused));
    int comm_size;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous);

    comm_size = comm_ptr->local_size;
    if ((comm_size & (comm_size - 1)) && MV2_Iallgather_function == MPIR_Iallgather_rec_dbl) {
        mpi_errno = MPIR_Iallgather_ring(sendbuf, sendcount, sendtype,
                                         recvbuf, recvcount, recvtype, comm_ptr, s);
    }
    else {
        mpi_errno = MV2_Iallgather_function(sendbuf, sendcount, sendtype, recvbuf,
                                            recvcount, recvtype, comm_ptr, s);
    }
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iallgather_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iallgather_intra_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous ATTRIBUTE((unused));
    MPI_Aint recvtype_size, nbytes;
    
    int two_level_iallgather = 1;
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

    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = recvtype_size * recvcount;

    // Search for some parameters regardless of whether subsequent selected
    // algorithm is 2-level or not
    
    // Search for the corresponding system size inside the tuning table
    while ((range < (mv2_size_iallgather_tuning_table - 1)) &&
           (comm_size > mv2_iallgather_thresholds_table[range].numproc)) {
        range++;
    }
    
    // Search for corresponding inter-leader function
    while ((range_threshold < (mv2_iallgather_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_iallgather_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_iallgather_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    // Search for corresponding intra-node function
    
    // Commenting this for the time being as none of
    // the algorithms are 2-level
    /*
    while ((range_threshold_intra <
            (mv2_iallgather_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_iallgather_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_iallgather_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }
    */

    MV2_Iallgather_function =
        mv2_iallgather_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Iallgather_function;

    MV2_Iallgather_intra_node_function =
        mv2_iallgather_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Iallgather_function;

    /* There are currently no two-level nb-allgather functions hence
       setting to 0 by default */
    two_level_iallgather = 
        mv2_iallgather_thresholds_table[range].is_two_level_iallgather[range_threshold]; 
    if (1 != two_level_iallgather) {
        mpi_errno = MPIR_Iallgather_tune_helper_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, comm_ptr, s);
    }
    else {
        /* Code path should not enter this with the current algorithms*/
    }

    return mpi_errno;
}
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */

#undef FUNCNAME
#define FUNCNAME MPIR_Iallgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iallgather_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPID_INTRACOMM) {    
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
      mpi_errno = MPIR_Iallgather_intra_MV2(sendbuf, sendcount, sendtype, recvbuf,
					    recvcount, recvtype, comm_ptr, s);
#else
      mpi_errno = MPIR_Iallgather_intra(sendbuf, sendcount, sendtype, recvbuf,
					recvcount, recvtype, comm_ptr, s);
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
    }
    else {
      mpi_errno = MPIR_Iallgather_inter(sendbuf, sendcount, sendtype, recvbuf,
					recvcount, recvtype, comm_ptr, s);
    }

    return mpi_errno;
}
