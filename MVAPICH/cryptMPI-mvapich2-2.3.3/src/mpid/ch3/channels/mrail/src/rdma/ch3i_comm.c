/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */
/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpidi_ch3_impl.h"
#include "mpiimpl.h"
#include "coll_shmem.h"

#define NULL_CONTEXT_ID -1

#undef FUNCNAME
#define FUNCNAME init_MV2_collops
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int init_MV2_collops (MPID_Comm *comm)
{
    int mpi_errno = MPI_SUCCESS;
    if (mv2_enable_ibcast) {
        comm->coll_fns->Ibcast_sched = MPIR_Ibcast_MV2;
    }
    if (mv2_enable_igather) {
        comm->coll_fns->Igather_sched = MPIR_Igather_MV2;
    }
    if (mv2_enable_iscatter) {
        comm->coll_fns->Iscatter_sched = MPIR_Iscatter_MV2;
    }
    if (mv2_enable_ireduce) {
        comm->coll_fns->Ireduce_sched = MPIR_Ireduce_MV2;
    }
    if (mv2_enable_ireduce_scatter) {
        comm->coll_fns->Ireduce_scatter_sched = MPIR_Ireduce_scatter_MV2;
    }
    if (mv2_enable_ialltoall) {
        comm->coll_fns->Ialltoall_sched = MPIR_Ialltoall_MV2;
    }
    if (mv2_enable_ialltoallv) {
        comm->coll_fns->Ialltoallv_sched = MPIR_Ialltoallv_MV2;
    }
    if (mv2_enable_iallgather) {
        comm->coll_fns->Iallgather_sched = MPIR_Iallgather_MV2;
    }
    if (mv2_enable_iallgatherv) {
        comm->coll_fns->Iallgatherv_sched = MPIR_Iallgatherv_MV2;
    }
    if (mv2_enable_iallreduce) {
        comm->coll_fns->Iallreduce_sched = MPIR_Iallreduce_MV2;
    }
    if (mv2_enable_ibarrier) {
        comm->coll_fns->Ibarrier_sched = MPIR_Ibarrier_MV2;
    }
    
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_comm_create
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_comm_create (MPID_Comm *comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_COMM_CREATE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_COMM_CREATE);

    /* Use the VC's eager threshold by default. */
    comm->dev.eager_max_msg_sz = -1;

    if(mv2_use_osu_collectives == 1 && 
        comm->comm_kind == MPID_INTRACOMM)  { 

        comm->coll_fns->Barrier = MPIR_Barrier_MV2;
        comm->coll_fns->Bcast = MPIR_Bcast_MV2;
        comm->coll_fns->Gather = MPIR_Gather_MV2;
        comm->coll_fns->Gatherv = MPIR_Gatherv_MV2;
        comm->coll_fns->Scatter = MPIR_Scatter_MV2;
        comm->coll_fns->Scatterv = MPIR_Scatterv;
        comm->coll_fns->Allgather = MPIR_Allgather_MV2; 
        comm->coll_fns->Allgatherv = MPIR_Allgatherv_MV2; 
        comm->coll_fns->Alltoall = MPIR_Alltoall_MV2; 
        comm->coll_fns->Alltoallv = MPIR_Alltoallv_MV2; 
        comm->coll_fns->Alltoallw = MPIR_Alltoallw; 
        comm->coll_fns->Reduce = MPIR_Reduce_MV2; 
        comm->coll_fns->Allreduce = MPIR_Allreduce_MV2;
        comm->coll_fns->Reduce_scatter = MPIR_Reduce_scatter_MV2;
        comm->coll_fns->Reduce_scatter_block = MPIR_Reduce_scatter_block_MV2;
        comm->coll_fns->Scan = MPIR_Scan_MV2;
        comm->coll_fns->Exscan = MPIR_Exscan_MV2;

    }
    
    if (mv2_use_osu_nb_collectives == 1 &&
        comm->comm_kind == MPID_INTRACOMM)  { 
      init_MV2_collops(comm);
    }
    MPIR_pof2_comm(comm, comm->local_size, comm->rank);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_COMM_CREATE);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_comm_destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_comm_destroy (MPID_Comm *comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_COMM_DESTROY);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_COMM_DESTROY);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_COMM_DESTROY);
    return mpi_errno;
}
