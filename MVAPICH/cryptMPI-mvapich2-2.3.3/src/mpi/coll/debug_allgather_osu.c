/* -*- Mode: C; c-basic-offset:4 ; -*- */
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
/*
*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/

#include "mpiimpl.h"
#include "secure_allgather.h"
#include "datatype.h"
#include "common_tuning.h"
#include "coll_shmem.h"
#include "allgather_tuning.h"



extern struct coll_runtime mv2_coll_param;
extern int allgather_tuning_algo;
extern int allgather_algo_num;
extern int use_2lvl_allgather;


MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_rd);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_bruck);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_ring);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_direct);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_directspread);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_gather_bcast);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_nonblocked);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_ring_nonblocked);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_direct);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_ring);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_multileader_rd_nonblocked);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_multileader_rd);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_2lvl_multileader_ring);



MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd_allgather_comm);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bruck);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_ring);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_direct);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_directspread);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_gather_bcast);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_nonblocked);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_nonblocked);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_direct);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_multileader_rd_nonblocked);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_multileader_rd);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_multileader_ring);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bruck_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_ring_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_directspread_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_nonblocked_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bruck_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_ring_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_directspread_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_nonblocked_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bruck_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_ring_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_directspread_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_nonblocked_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_rd_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bruck_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_ring_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_directspread_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_nonblocked_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_2lvl_ring_count_recv);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_count_recv);

#define FGP_SWITCH_FACTOR 4     /*Used to determine switch between naive and FGP
                                   design */

int (*MV2_Allgather_function)(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                             MPIR_Errflag_t *errflag);

int MPIR_Allgather_RD_Allgather_Comm_MV2(const void *sendbuf,
                                 int sendcount,
                                 MPI_Datatype sendtype,
                                 void *recvbuf,
                                 int recvcount,
                                 MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                 MPIR_Errflag_t *errflag)
{
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_rd_allgather_comm, 1);

    return 0;
}

int allgather_tuning(int comm_size, int pof2)
{

    char *value;
    if (pof2 == 1 && (value = getenv("MV2_ALLGATHER_RD_THRESHOLD")) != NULL) {
        /* pof2 case. User has set the run-time parameter
           "MV2_ALLGATHER_RD_THRESHOLD".
           * Just use that value */
        return mv2_coll_param.allgather_rd_threshold;
    }
    if (pof2 == 0 && (value = getenv("MV2_ALLGATHER_BRUCK_THRESHOLD")) != NULL) {
        /* Non-pof2 case. User has set the run-time parameter
           "MV2_ALLGATHER_BRUCK_THRESHOLD".
           * Just use that value */
        return mv2_coll_param.allgather_bruck_threshold;
    } else {
        /* User has not used any run-time parameters.
         */
        if (comm_size <= MV2_ALLGATHER_SMALL_SYSTEM_SIZE) {
            return mv2_tuning_table[ALLGATHER_ID][SMALL];
        } else if (comm_size > MV2_ALLGATHER_SMALL_SYSTEM_SIZE
                   && comm_size <= MV2_ALLGATHER_MEDIUM_SYSTEM_SIZE) {
            return mv2_tuning_table[ALLGATHER_ID][MEDIUM];
        } else {
            return mv2_tuning_table[ALLGATHER_ID][LARGE];
        }
    }
}

/* This is the default implementation of allgather. The algorithm is:

Algorithm: MPI_Allgather

For short messages and non-power-of-two no. of processes, we use
the algorithm from the Jehoshua Bruck et al IEEE TPDS Nov 97
paper. It is a variant of the disemmination algorithm for
barrier. It takes ceiling(lg p) steps.

Cost = lgp.alpha + n.((p-1)/p).beta
where n is total size of data gathered on each process.

For short or medium-size messages and power-of-two no. of
processes, we use the recursive doubling algorithm.

Cost = lgp.alpha + n.((p-1)/p).beta

TODO: On TCP, we may want to use recursive doubling instead of the Bruck
algorithm in all cases because of the pairwise-exchange property of
recursive doubling (see Benson et al paper in Euro PVM/MPI
2003).

It is interesting to note that either of the above algorithms for
MPI_Allgather has the same cost as the tree algorithm for MPI_Gather!

For long messages or medium-size messages and non-power-of-two
no. of processes, we use a ring algorithm. In the first step, each
process i sends its contribution to process i+1 and receives
the contribution from process i-1 (with wrap-around). From the
second step onwards, each process i forwards to process i+1 the
data it received from process i-1 in the previous step. This takes
a total of p-1 steps.

Cost = (p-1).alpha + n.((p-1)/p).beta

We use this algorithm instead of recursive doubling for long
messages because we find that this communication pattern (nearest
neighbor) performs twice as fast as recursive doubling for long
messages (on Myrinet and IBM SP).

Possible improvements:

End Algorithm: MPI_Allgather
*/
/* begin:nested */
/* not declared static because a machine-specific function may call this
one in some cases */

/* This implements an allgather via direct method, in which each
 * process sends directly to every other process.  All processes
 * start sending to rank 0 and work up in order. This is meant as
 * a base case */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_Direct_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_Direct_MV2(
    const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
          void *recvbuf, int recvcnt, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,direct);
    int i;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_direct, 1);

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_Allgather_Direct_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* allocate an array of request objects */
    MPID_Request **reqarray = NULL;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        2 * size * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    /* allocate an array of status objects */
    MPI_Status *starray = NULL;
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        2 * size * sizeof (MPI_Status),
                        mpi_errno, "starray");

    /* initialize our active request counter to 0 */
    int reqs = 0;

    /* copy our data to our receive buffer if needed */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcnt, sendtype,
            rbuf,    recvcnt, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* post receives */
    for (i = 0; i < size; i++) {
        /* our data is already in the receive buffer */
        if (i == rank) {
            continue;
        }

        /* compute pointer in receive buffer for incoming data from this rank */
        void* rbuf = (void*)((char*) recvbuf + i * recvcnt * recvtype_extent);

        /* post receive for data from this rank */
        MPIR_PVAR_INC(allgather, direct, recv, recvcnt, recvtype);
        mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
            i, MPIR_ALLGATHER_TAG, comm_ptr, &reqarray[reqs++]
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* TODO: consider placing a barrier here to ensure
     * receives are posted before sends, especially for large messages */
    //MPIR_Barrier_impl(comm_ptr);

    /* get parameters for sending data */
    const void* sbuf   = sendbuf;
    int scnt           = sendcnt;
    MPI_Datatype stype = sendtype;
    if (sendbuf == MPI_IN_PLACE) {
        /* use receive params if IN_PLACE */
        sbuf   = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
        scnt   = recvcnt;
        stype  = recvtype;
    }

    /* post sends */
    for (i = 0; i < size; i++) {
        /* no need to send to ourself */
        if (i == rank) {
            continue;
        }

        /* send data to this rank */
        MPIR_PVAR_INC(allgather, direct, send, scnt, stype); 
        mpi_errno = MPIC_Isend(sbuf, scnt, stype,
            i, MPIR_ALLGATHER_TAG, comm_ptr, &reqarray[reqs++], errflag
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* wait for all outstanding requests to complete */
    mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
    if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno == MPI_ERR_IN_STATUS) {
        for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                mpi_errno = starray[i].MPI_ERROR;
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
    }

    MPIU_CHKLMEM_FREEALL();
  fn_fail:
    MPIR_TIMER_END(coll,allgather,direct);
    return (mpi_errno);
}

/* This implements an allgather via direct method, in which each
 * process sends directly to every other process.  To spread the
 * load and avoid hot spots, processes starting by sending to the
 * rank one higher than their own.  This is meant as a base case
 * allgather, but it may actually be the fastest method in some cases. */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_DirectSpread_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_DirectSpread_MV2(
    const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
          void *recvbuf, int recvcnt, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,directspread);
    int i;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_directspread, 1);

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_Allgather_DirectSpread_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* allocate an array of request objects */
    MPID_Request **reqarray = NULL;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        2 * size * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    /* allocate an array of status objects */
    MPI_Status *starray = NULL;
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        2 * size * sizeof (MPI_Status),
                        mpi_errno, "starray");

    /* initialize our active request counter to 0 */
    int reqs = 0;

    /* copy our data to our receive buffer if needed */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcnt, sendtype,
            rbuf,    recvcnt, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* post receives */
    for (i = 1; i < size; i++) {
        /* compute source rank sending to us in this step */
        int src = rank - i;
        if (src < 0) {
            src += size;
        }

        /* get pointer to receive buffer for this rank */
        void* rbuf = (void*)((char*) recvbuf + src * recvcnt * recvtype_extent);

        /* post receive */
        MPIR_PVAR_INC(allgather, directspread, recv, recvcnt, recvtype);
        mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
            src, MPIR_ALLGATHER_TAG, comm_ptr, &reqarray[reqs++]
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* TODO: consider placing a barrier here to ensure
     * receives are posted before sends, especially for large messages */
    //MPIR_Barrier_impl(comm_ptr);

    /* get parameters for sending data */
    const void* sbuf   = sendbuf;
    int scnt           = sendcnt;
    MPI_Datatype stype = sendtype;
    if (sendbuf == MPI_IN_PLACE) {
        /* use receive params if IN_PLACE */
        sbuf   = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
        scnt   = recvcnt;
        stype  = recvtype;
    }

    for (i = 1; i < size; i++) {
        /* compute destination rank for this step */
        int dst = rank + i;
        if (dst >= size) {
            dst -= size;
        }

        /* post send to this destination rank */
        MPIR_PVAR_INC(allgather, directspread, send, scnt, stype); 
        mpi_errno = MPIC_Isend(sbuf, scnt, stype,
            dst, MPIR_ALLGATHER_TAG, comm_ptr, &reqarray[reqs++], errflag
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* wait on all outstanding requests */
    mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
    if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno == MPI_ERR_IN_STATUS) {
        for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                mpi_errno = starray[i].MPI_ERROR;
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
    }

    MPIU_CHKLMEM_FREEALL();
  fn_fail:
    MPIR_TIMER_END(coll,allgather,directspread);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_RD_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_RD_MV2(const void *sendbuf,
                          int sendcount,
                          MPI_Datatype sendtype,
                          void *recvbuf,
                          int recvcount,
                          MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                          MPIR_Errflag_t *errflag)
{


    MPIR_TIMER_START(coll,allgather,rd);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    int curr_cnt, dst;
    MPI_Status status;
    int mask, dst_tree_root, my_tree_root, is_homogeneous,
        send_offset, recv_offset, last_recv_cnt = 0, nprocs_completed, k,
        offset, tmp_mask, tree_root;
    int send_req_idx = 0;
#ifdef MPID_HAS_HETERO
    int position, tmp_buf_size, nbytes;
#endif

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_rd, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    if(rank==0)
        printf("MPIR_Allgather_RD_MV2\n");
    /************* For MPIC_Sendrecv_Plus ************
    MPID_Request *send_req_ptr[2*(comm_size-1)];
    int p;
    for(p=0; p<comm_size-1; ++p){
        (send_req_ptr[p])=NULL;
    }
    *************************************************/

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero) {
        is_homogeneous = 0;
    }
#endif

    if (is_homogeneous) {
        /* homogeneous. no need to pack into tmp_buf on each node. copy
         * local data into recvbuf */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       ((char *) recvbuf +
                                        rank * recvcount * recvtype_extent),
                                       recvcount, recvtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }


	    //	printf("starting curr_cnt for %d is %d\n", rank, curr_cnt);
        /******************** Added by Mehran  ********************/
        MPID_Node_id_t node_id, dst_node_id;
        int first_encrypted_index = -1, last_encrypted_index = -1; //Nothing encrypted so far
        char *in, *out, *rbuf, *sbuf;
        int recently_received=0;
        if(security_approach==2){
            MPID_Get_node_id(comm_ptr, rank, &node_id);
	        curr_cnt = 1;
        }else{
	        curr_cnt = recvcount;
	    }

        //printf("recvcount=%d & recvtype_extent=%d\n", recvcount, recvtype_extent);
        mask = 0x1;
        i = 0;
        
        while (mask < comm_size) {
            dst = rank ^ mask;

            /* find offset into send and recv buffers. zero out
             * the least significant "i" bits of rank and dst to
             * find root of src and dst subtrees. Use ranks of
             * roots as index to send from and recv into buffer */
	    
            dst_tree_root = dst >> i;
            dst_tree_root <<= i;

            my_tree_root = rank >> i;
            my_tree_root <<= i;

            /* FIXME: saving an MPI_Aint into an int */
            send_offset = my_tree_root * recvcount * recvtype_extent;
            recv_offset = dst_tree_root * recvcount * recvtype_extent;

            if (dst < comm_size) {
                if(security_approach==2){
                    //Naive+
		    
                    unsigned long  ciphertext_len = 0, count=0, in_size=0;
                    in_size = (unsigned long)(recvcount * recvtype_extent);
                    unsigned long max_out_len = (unsigned long) (16 + in_size);

                    if(comm_size - my_tree_root < curr_cnt)
                        curr_cnt = comm_size - my_tree_root;
                    MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                    if(node_id != dst_node_id){
                        //Inter-node
			            //printf("curr_cnt for %d is %d\n", rank, curr_cnt);
                        //encrypt all the unencrypted messages so far
                        /*if(last_encrypted_index == -1){
                            last_encrypted_index = my_tree_root;
			            }*/
                        //if(comm_size - my_tree_root < curr_cnt)
			            //curr_cnt = comm_size - my_tree_root;

                        int first_to_send = my_tree_root;
                        int last_to_send = my_tree_root + (int) (curr_cnt);
                        int enc_idx;
                        
                        //printf("last_to_send for %d is %d\n", rank, last_to_send);
			            //printf("%d is going to encrypt [%d, %d] - [%d, %d]\n", rank, first_to_send, last_to_send, first_encrypted_index, last_encrypted_index);
                        for(enc_idx = first_to_send; enc_idx<last_to_send; ++enc_idx){
                            bool already_encrypted = first_encrypted_index!= -1 && enc_idx >= first_encrypted_index && last_encrypted_index!= -1 && enc_idx <= last_encrypted_index;
                            if(!already_encrypted){
                                in = (char*)((char*) recvbuf + enc_idx * recvcount * recvtype_extent);
                                out = (char*)((char*) ciphertext_recvbuf + enc_idx * (recvcount * recvtype_extent + 16+12));
                                //printf("%d is going to encrypt %d\n", rank, enc_idx);
                                RAND_bytes(out, 12);
                                
                                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                            &ciphertext_len, max_out_len,
                                            out, 12, in, in_size,
                                            NULL, 0)){
                                    printf("Error in Naive+ encryption: allgather RD (Default)\n");
                                    fflush(stdout);
                                }
                            }//end if
                        }//end for
                        if(last_encrypted_index == -1 || last_to_send > last_encrypted_index){
                            last_encrypted_index = last_to_send-1;
                        }
                        if(first_encrypted_index == -1 || first_to_send < first_encrypted_index){
                            first_encrypted_index = first_to_send;
                        }
                        //--last_encrypted_index;
                        //printf("now first and last encrypted indices for %d are %d and %d\n", rank , first_encrypted_index, last_encrypted_index);

                        //set the send and recv buffers
                        
                        sbuf = (char*)((char*) ciphertext_recvbuf + my_tree_root * (recvcount * recvtype_extent + 16+12));
                        rbuf = (char*)((char*) ciphertext_recvbuf + dst_tree_root * (recvcount * recvtype_extent + 16+12));

                        //send recv
                        //printf("%d is going to send (I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt,   my_tree_root, dst, (comm_size - dst_tree_root), dst_tree_root);
			            //changed (comm_size - dst_tree_root) to curr_cnt
                        MPIR_PVAR_INC(allgather, rd, send, curr_cnt * (recvcount*recvtype_extent + 16+12), MPI_CHAR); 
                        MPIR_PVAR_INC(allgather, rd, recv, (comm_size - dst_tree_root) * (recvcount*recvtype_extent + 16+12), MPI_CHAR);
                        mpi_errno =
                            MPIC_Sendrecv(sbuf, curr_cnt * (recvcount*recvtype_extent + 16+12), 
                                    MPI_CHAR, dst, MPIR_ALLGATHER_TAG,
                                    rbuf, (comm_size - dst_tree_root) * (recvcount*recvtype_extent + 16+12), 
                                    MPI_CHAR, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /************* For MPIC_Sendrecv_Plus ************
                        mpi_errno =
                            MPIC_Sendrecv_Plus(sbuf, curr_cnt * (recvcount*recvtype_extent + 16+12), 
                                    MPI_CHAR, dst, MPIR_ALLGATHER_TAG,
                                    rbuf, (comm_size - dst_tree_root) * (recvcount*recvtype_extent + 16+12), 
                                    MPI_CHAR, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                        *************************************************/
                        if (mpi_errno) {
                            /* for communication errors, just record the error but
                            continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                        curr_cnt += recently_received;
                        //printf("%d received (I) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                        //decrypt the received messages
                        int decryption_index = dst_tree_root;
			            int last_to_decrypt = dst_tree_root + recently_received;
                        //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                        for(; decryption_index<last_to_decrypt; ++decryption_index){
                            in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent + 16+12));
                            out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                            //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                            if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                    in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                    NULL, 0)){

                                printf("Error in Naive+ decryption: allgather RD (default-I) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                                fflush(stdout);        
                            }
                        }


                    }else{
                        //Intra-Node
			            //printf("curr_cnt for %d is %d (intra-node)\n", rank, curr_cnt);
                        MPIR_PVAR_INC(allgather, rd, send, curr_cnt*recvcount, recvtype); 
                        MPIR_PVAR_INC(allgather, rd, recv, (comm_size - dst_tree_root)*recvcount, recvtype); 
                        mpi_errno =
                            MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                            curr_cnt*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG,
                                            ((char *) recvbuf + recv_offset),
                                            (comm_size - dst_tree_root)*recvcount, recvtype,
                                            dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /************* For MPIC_Sendrecv_Plus ************
                        mpi_errno =
                            MPIC_Sendrecv_Plus(((char *) recvbuf + send_offset),
                                            curr_cnt*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG,
                                            ((char *) recvbuf + recv_offset),
                                            (comm_size - dst_tree_root)*recvcount, recvtype,
                                            dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                        *************************************************/
                        if (mpi_errno) {
                            /* for communication errors, just record the error but
                            continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }

                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int) (last_recv_cnt/recvcount);
                        curr_cnt += recently_received;
                    }
                }//End Naive+
                else{
                    MPIR_PVAR_INC(allgather, rd, send, curr_cnt, recvtype); 
                    MPIR_PVAR_INC(allgather, rd, recv, (comm_size - dst_tree_root) * recvcount, recvtype); 
                    mpi_errno =
                        MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                        curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) recvbuf + recv_offset),
                                        (comm_size -
                                        dst_tree_root) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno =
                        MPIC_Sendrecv_Plus(((char *) recvbuf + send_offset),
                                        curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) recvbuf + recv_offset),
                                        (comm_size -
                                        dst_tree_root) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                    *************************************************/

                        
                    if (mpi_errno) {
                        /* for communication errors, just record the error but
                        continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        last_recv_cnt = 0;
                    }

                    MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                    curr_cnt += last_recv_cnt;
                }

            }

            /* if some processes in this process's subtree in this step
             * did not have any destination process to communicate with
             * because of non-power-of-two, we need to send them the
             * data that they would normally have received from those
             * processes. That is, the haves in this subtree must send to
             * the havenots. We use a logarithmic recursive-halfing algorithm
             * for this. */

            /* This part of the code will not currently be
             * executed because we are not using recursive
             * doubling for non power of two. Mark it as experimental
             * so that it doesn't show up as red in the coverage
             * tests. */
	    
            /* --BEGIN EXPERIMENTAL-- */
            if (dst_tree_root + mask > comm_size) {
		        
                nprocs_completed = comm_size - my_tree_root - mask;
                /* nprocs_completed is the number of processes in this
                 * subtree that have all the data. Send data to others
                 * in a tree fashion. First find root of current tree
                 * that is being divided into two. k is the number of
                 * least-significant bits in this process's rank that
                 * must be zeroed out to find the rank of the root */
                j = mask;
                k = 0;
                while (j) {
                    j >>= 1;
                    k++;
                }
                k--;

                /* FIXME: saving an MPI_Aint into an int */
                offset = recvcount * (my_tree_root + mask) * recvtype_extent;
                tmp_mask = mask >> 1;
                unsigned long  ciphertext_len = 0, count=0, in_size=0;
                in_size = (unsigned long)(recvcount * recvtype_extent);
                unsigned long max_out_len = (unsigned long) (16 + in_size);

                while (tmp_mask) {
                    dst = rank ^ tmp_mask;

                    tree_root = rank >> k;
                    tree_root <<= k;

                    /* send only if this proc has data and destination
                     * doesn't have data. at any step, multiple processes
                     * can send if they have the data */
                    if ((dst > rank) && (rank < tree_root + nprocs_completed)
                        && (dst >= tree_root + nprocs_completed)) {
                            if(security_approach==2){
                                //Naive+
                                if(comm_size - (my_tree_root+mask) < recently_received)
				                recently_received = comm_size - (my_tree_root+mask);
                                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                                if(node_id != dst_node_id){
                                    //Inter Node
                                    //if(comm_size - (my_tree_root+mask) < recently_received)
				                    //recently_received = comm_size - (my_tree_root+mask);

                                    int first_to_send = (my_tree_root + mask);
                                    int last_to_send = (my_tree_root + mask) + recently_received;
                                    int enc_idx;
                                    //printf("last_to_send (II) for %d is %d\n", rank, last_to_send);
                                    //printf("%d is going to encrypt %d - %d\n", rank, last_encrypted_index, last_to_send);
                                    for(enc_idx = first_to_send; enc_idx<last_to_send; ++enc_idx){
                                        bool already_encrypted = first_encrypted_index!= -1 && enc_idx >= first_encrypted_index && last_encrypted_index!= -1 && enc_idx <= last_encrypted_index;
                                        if(! already_encrypted){
                                            in = (char*)((char*) recvbuf + enc_idx * recvcount * recvtype_extent);
                                            out = (char*)((char*) ciphertext_recvbuf + enc_idx * (recvcount * recvtype_extent + 16+12));
                                            // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                                            RAND_bytes(out, 12);
                                            
                                            if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                                        &ciphertext_len, max_out_len,
                                                        out, 12, in, in_size,
                                                        NULL, 0)){
                                                printf("Error in Naive+ encryption: allgather RD (Default)\n");
                                                fflush(stdout);
                                            }
                                        }//end if
                                    }//end for
                                    if(last_encrypted_index == -1 || last_to_send > last_encrypted_index){
                                        last_encrypted_index = last_to_send -1;
                                    }
                                    if(first_encrypted_index == -1 || first_to_send < first_encrypted_index){
                                        first_encrypted_index = first_to_send;
                                    }

                                    sbuf = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask) * (recvcount * recvtype_extent + 16+12));
                                    
                                    //send
                                    MPIR_PVAR_INC(allgather, rd, send, recently_received * (recvcount*recvtype_extent + 16+12), MPI_CHAR); 
                                    //printf("%d is going to send (II) %d from %d to %d\n", rank, recently_received, (my_tree_root + mask), dst);

                                    mpi_errno =
                                        MPIC_Send(sbuf,
                                                recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                                    /************* For MPIC_Sendrecv_Plus ************
                                    mpi_errno =
                                        MPIC_Send_Plus(sbuf,
                                                recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                                    *************************************************/
                                    if (mpi_errno) {
                                        /* for communication errors, just record the error but
                                        continue */
                                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                        
                                    }
				    
                                    //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                                }else{
                                    //Intra Node
                                    MPIR_PVAR_INC(allgather, rd, send, recently_received*recvcount, recvtype); 
                                    mpi_errno =
                                        MPIC_Send(((char *) recvbuf + offset),
                                                    recently_received*recvcount, recvtype, dst,
                                                    MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                                    
                                    /************* For MPIC_Sendrecv_Plus ************
                                    mpi_errno =
                                        MPIC_Send_Plus(((char *) recvbuf + offset),
                                                    recently_received*recvcount, recvtype, dst,
                                                    MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                                    *************************************************/

                                    /* recently_received was set in the previous
                                    * receive. that's the amount of data to be
                                    * sent now. */
                                    if (mpi_errno) {
                                        /* for communication errors, just record the error
                                        but continue */
                                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    }
                                    
                                }


                            }//End Naive +
                            else{
                                MPIR_PVAR_INC(allgather, rd, send, last_recv_cnt, recvtype); 
                                mpi_errno =
                                    MPIC_Send(((char *) recvbuf + offset),
                                                last_recv_cnt, recvtype, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                                /************* For MPIC_Sendrecv_Plus ************
                                mpi_errno =
                                    MPIC_Send_Plus(((char *) recvbuf + offset),
                                                last_recv_cnt, recvtype, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                                *************************************************/

                                /* last_recv_cnt was set in the previous
                                * receive. that's the amount of data to be
                                * sent now. */
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                }
                            }
                        
                    }//End send condition
                    /* recv only if this proc. doesn't have data and sender
                     * has data */
                    else if ((dst < rank) &&
                             (dst < tree_root + nprocs_completed) &&
                             (rank >= tree_root + nprocs_completed)) {
			            if(security_approach==2){
                            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                            if(node_id != dst_node_id){
                                //Inter-Node

                                MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR);
				//printf("%d is going to recv (II) %d from %d @ %d\n", rank, (comm_size - (my_tree_root + mask)), dst, (my_tree_root + mask));
                                mpi_errno =
                                    MPIC_Recv(((char *) ciphertext_recvbuf + (my_tree_root + mask)*(recvcount*recvtype_extent + 16+12)),
                                                (comm_size -
                                                (my_tree_root +
                                                mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR,
                                                dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                                /* nprocs_completed is also equal to the
                                * no. of processes whose data we don't have */
				//printf("%d @ flag1 \n", rank);
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    last_recv_cnt = 0;
                                }
				//printf("%d @ flag2 \n", rank);
                                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                                int recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                                curr_cnt += recently_received;
                                //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                                //decrypt the received messages
                                int decryption_index = (my_tree_root + mask);
                                int last_to_decrypt = (my_tree_root + mask) + recently_received;
                                //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                                for(; decryption_index<last_to_decrypt; ++decryption_index){
                                    in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent + 16+12));
                                    out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                                    if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                            NULL, 0)){

                                        printf("Error in Naive+ decryption: allgather RD (default-II) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                                        fflush(stdout);        
                                    }
                                }


                            }else{
                                //intra-node
                                MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * recvcount, recvtype);
                                mpi_errno =
                                    MPIC_Recv(((char *) recvbuf + offset),
                                                (comm_size -
                                                (my_tree_root +
                                                mask)) * recvcount, recvtype,
                                                dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                                /* nprocs_completed is also equal to the
                                * no. of processes whose data we don't have */
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    last_recv_cnt = 0;
                                }
                                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                                curr_cnt += (int) (last_recv_cnt/recvcount);
                            }
                        }else{
                            MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * recvcount, recvtype);
                            mpi_errno =
                                MPIC_Recv(((char *) recvbuf + offset),
                                            (comm_size -
                                            (my_tree_root +
                                            mask)) * recvcount, recvtype,
                                            dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                            /* nprocs_completed is also equal to the
                            * no. of processes whose data we don't have */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                last_recv_cnt = 0;
                            }
                            MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                            curr_cnt += last_recv_cnt;
                        }
                        
                    }
                    tmp_mask >>= 1;
                    k--;
                }
            }
            /* --END EXPERIMENTAL-- */

            mask <<= 1;
            i++;
        }
    }
#ifdef MPID_HAS_HETERO
    else {
        /* heterogeneous. need to use temp. buffer. */

        MPIR_Pack_size_impl(recvcount * comm_size, recvtype, &tmp_buf_size);

        tmp_buf = MPIU_Malloc(tmp_buf_size);
        /* --BEGIN ERROR HANDLING-- */
        if (!tmp_buf) {
            mpi_errno =
                MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                     FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
            return mpi_errno;
        }
        /* --END ERROR HANDLING-- */

        /* calculate the value of nbytes, the number of bytes in packed
         * representation that each process contributes. We can't simply divide
         * tmp_buf_size by comm_size because tmp_buf_size is an upper
         * bound on the amount of memory required. (For example, for
         * a single integer, MPICH-1 returns pack_size=12.) Therefore, we
         * actually pack some data into tmp_buf and see by how much
         * 'position' is incremented. */

        position = 0;
        MPIR_Pack_impl(recvbuf, 1, recvtype, tmp_buf, tmp_buf_size, &position);
        nbytes = position * recvcount;

        /* pack local data into right location in tmp_buf */
        position = rank * nbytes;
        if (sendbuf != MPI_IN_PLACE) {
            MPIR_Pack_impl(sendbuf, sendcount, sendtype, tmp_buf,
                           tmp_buf_size, &position);
        } else {
            /* if in_place specified, local data is found in recvbuf */
            MPIR_Pack_impl(((char *) recvbuf + recvtype_extent * rank),
                           recvcount, recvtype, tmp_buf, tmp_buf_size, &position);
        }

        curr_cnt = nbytes;

        mask = 0x1;
        i = 0;
        while (mask < comm_size) {
            dst = rank ^ mask;

            /* find offset into send and recv buffers. zero out
             * the least significant "i" bits of rank and dst to
             * find root of src and dst subtrees. Use ranks of
             * roots as index to send from and recv into buffer. */

            dst_tree_root = dst >> i;
            dst_tree_root <<= i;

            my_tree_root = rank >> i;
            my_tree_root <<= i;

            send_offset = my_tree_root * nbytes;
            recv_offset = dst_tree_root * nbytes;

            if (dst < comm_size) {
                MPIR_PVAR_INC(allgather, rd, send, curr_cnt, MPI_BYTE); 
                MPIR_PVAR_INC(allgather, rd, recv, (tmp_buf_size - recv_offset), MPI_BYTE); 
                mpi_errno =
                    MPIC_Sendrecv(((char *) tmp_buf + send_offset),
                                     curr_cnt, MPI_BYTE, dst,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) tmp_buf + recv_offset),
                                     tmp_buf_size - recv_offset, MPI_BYTE,
                                     dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                       continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, MPI_BYTE, &last_recv_cnt);
                curr_cnt += last_recv_cnt;
            }

            /* if some processes in this process's subtree in this step
             * did not have any destination process to communicate with
             * because of non-power-of-two, we need to send them the
             * data that they would normally have received from those
             * processes. That is, the haves in this subtree must send to
             * the havenots. We use a logarithmic recursive-halfing
             * algorithm for this. */

            if (dst_tree_root + mask > comm_size) {
                nprocs_completed = comm_size - my_tree_root - mask;
                /* nprocs_completed is the number of processes in this
                 * subtree that have all the data. Send data to others
                 * in a tree fashion. First find root of current tree
                 * that is being divided into two. k is the number of
                 * least-significant bits in this process's rank that
                 * must be zeroed out to find the rank of the root */
                j = mask;
                k = 0;
                while (j) {
                    j >>= 1;
                    k++;
                }
                k--;

                offset = nbytes * (my_tree_root + mask);
                tmp_mask = mask >> 1;

                while (tmp_mask) {
                    dst = rank ^ tmp_mask;
                    tree_root = rank >> k;
                    tree_root <<= k;

                    /* send only if this proc has data and destination
                     * doesn't have data. at any step, multiple processes
                     * can send if they have the data */
                    if ((dst > rank) && (rank < tree_root + nprocs_completed)
                        && (dst >= tree_root + nprocs_completed)) {
                        MPIR_PVAR_INC(allgather, rd, send, last_recv_cnt, MPI_BYTE); 
                        mpi_errno =
                            MPIC_Send(((char *) tmp_buf + offset),
                                         last_recv_cnt, MPI_BYTE, dst,
                                         MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                        /* last_recv_cnt was set in the previous
                         * receive. that's the amount of data to be
                         * sent now. */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                               but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                    }
                    /* recv only if this proc. doesn't have data and sender
                     * has data */
                    else if ((dst < rank) &&
                             (dst < tree_root + nprocs_completed) &&
                             (rank >= tree_root + nprocs_completed)) {
                        MPIR_PVAR_INC(allgather, rd, recv, (tmp_buf_size - offset), MPI_BYTE);
                        mpi_errno = MPIC_Recv(((char *) tmp_buf + offset),
                                              tmp_buf_size - offset,
                                              MPI_BYTE, dst,
                                              MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                         * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                               but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, MPI_BYTE, &last_recv_cnt);
                        curr_cnt += last_recv_cnt;
                    }
                    tmp_mask >>= 1;
                    k--;
                }
            }
            mask <<= 1;
            i++;
        }

        position = 0;
        MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position, recvbuf,
                         recvcount * comm_size, recvtype);

        MPIU_Free(tmp_buf);
    }
#endif                          /* MPID_HAS_HETERO */
    //printf("%d Goodbye\n", rank);


    /*for(i=0; i<send_req_idx; ++i){
        mpi_errno = MPIC_Wait((send_req_ptr[i]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[i]->status.MPI_ERROR;
        }
        MPID_Request_release(send_req_ptr[i]);
    }*/
    


  fn_fail:
    MPIR_TIMER_END(coll,allgather,rd);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_Bruck_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_Bruck_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                             MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,bruck);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    int src, rem;
    void *tmp_buf;
    int curr_cnt, dst;
    int pof2 = 0;
    MPID_Node_id_t node_id, dst_node_id, src_node_id;
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_bruck, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    if(rank==0)
        printf("MPIR_Allgather_Bruck_MV2\n");
    /************* For MPIC_Sendrecv_Plus ************
    MPID_Request *send_req_ptr[comm_size-1];
    int p;
    for(p=0; p<comm_size-1; ++p){
        (send_req_ptr[p])=NULL;
    }
    *************************************************/
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* get true extent of recvtype */
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);
    recvbuf_extent =
        recvcount * comm_size * (MPIR_MAX(recvtype_true_extent, recvtype_extent));
    /* allocate a temporary buffer of the same size as recvbuf. */
    tmp_buf = MPIU_Malloc(recvbuf_extent);
    /* --BEGIN ERROR HANDLING-- */
    if (!tmp_buf) {
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**nomem", 0);
        return mpi_errno;
    }
    /* --END ERROR HANDLING-- */

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    /* copy local data to the top of tmp_buf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        mpi_errno = MPIR_Localcopy(((char *) recvbuf +
                                    rank * recvcount * recvtype_extent),
                                   recvcount, recvtype, tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /********************* Added by Mehran ***************/
    int last_encrypted_index = 0; //Nothing encrypted so far
    char *in, *out, *rbuf, *sbuf;
    int recently_received=0, s_msg_size, r_msg_size;
    if(security_approach==2){
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        curr_cnt = 1;
    }else{
        curr_cnt = recvcount;
    }
    int i=0;
    unsigned long  ciphertext_len = 0, count=0, in_size=0;
    in_size = (unsigned long)(recvcount * recvtype_extent);
    unsigned long max_out_len = (unsigned long) (16 + in_size);
    /* do the first \floor(\lg p) steps */
    pof2 = 1;
    while (pof2 <= comm_size / 2) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        if(security_approach==2){
            //Naive+
            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
            MPID_Get_node_id(comm_ptr, src, &src_node_id);
            if(node_id != dst_node_id){
                //encrypt from tmp_buf to ciphertext_recvbuf and set the send buffer from ciphertext_recvbuf and s_msg_size
                for(; last_encrypted_index<curr_cnt; ++last_encrypted_index){
                    in = (char*)((char*) tmp_buf + last_encrypted_index * recvcount * recvtype_extent);
                    out = (char*)((char*) ciphertext_recvbuf + last_encrypted_index * (recvcount * recvtype_extent + 16+12));
                    // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in Naive+ encryption: allgather Bruck (Default-I)\n");
                        fflush(stdout);
                    }

                }

                sbuf = (char*) ciphertext_recvbuf;
                s_msg_size = (recvcount * recvtype_extent + 16+12);

            }else{
                //set the send buffer from tmp_buf and s_msg_size
                sbuf = (char*) tmp_buf;
                s_msg_size = (recvcount * recvtype_extent);
            }
            if(node_id != src_node_id){
                //set the recvbuffer from ciphertext_recvbuf and r_msg_size
                rbuf = (char*)((char*) ciphertext_recvbuf + curr_cnt * (recvcount * recvtype_extent + 16+12));
                r_msg_size = (recvcount * recvtype_extent + 16+12);
            }else{
                //set the recv buffer from tmp_buf and r_msg_size
                rbuf = (char*)((char*) tmp_buf + curr_cnt * recvcount * recvtype_extent);
                r_msg_size = (recvcount * recvtype_extent);
            }

            MPIR_PVAR_INC(allgather, bruck, send, curr_cnt * s_msg_size, MPI_CHAR); 
            MPIR_PVAR_INC(allgather, bruck, recv, curr_cnt * r_msg_size, MPI_CHAR); 
            mpi_errno = MPIC_Sendrecv(sbuf, curr_cnt * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, curr_cnt * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, errflag);

            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(sbuf, curr_cnt * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, curr_cnt * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }


            if(node_id != src_node_id){
                //decrypt recvd messages from ciphertext_recvbuf to temp_buf
                int decryption_index = curr_cnt;
                for(; decryption_index<2*curr_cnt; ++decryption_index){
                    in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent + 16+12));
                    out = (char*)((char*) tmp_buf + decryption_index * recvcount * recvtype_extent);
                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                    if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){

                        printf("Error in Naive+ decryption: allgather Bruck (Default-I) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }

            }
        //end naive+
        }else{
            
            MPIR_PVAR_INC(allgather, bruck, send, curr_cnt, recvtype); 
            MPIR_PVAR_INC(allgather, bruck, recv, curr_cnt, recvtype); 
            mpi_errno = MPIC_Sendrecv(tmp_buf, curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) tmp_buf +
                                        curr_cnt * recvtype_extent), curr_cnt,
                                        recvtype, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, errflag);
            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(tmp_buf, curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) tmp_buf +
                                        curr_cnt * recvtype_extent), curr_cnt,
                                        recvtype, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/
            
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
        
        
        curr_cnt *= 2;
        pof2 *= 2;
    }//end while

    /* if comm_size is not a power of two, one more step is needed */

    rem = comm_size - pof2;
    if (rem) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;
        if(security_approach==2){
            //Naive+
            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
            MPID_Get_node_id(comm_ptr, src, &src_node_id);
            if(node_id != dst_node_id){
                //encrypt from tmp_buf to ciphertext_recvbuf and set the send buffer from ciphertext_recvbuf and s_msg_size
                for(; last_encrypted_index<rem; ++last_encrypted_index){
                    in = (char*)((char*) tmp_buf + last_encrypted_index * recvcount * recvtype_extent);
                    out = (char*)((char*) ciphertext_recvbuf + last_encrypted_index * (recvcount * recvtype_extent + 16+12));
                    // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in Naive+ encryption: allgather Bruck (Default-II)\n");
                        fflush(stdout);
                    }

                }

                sbuf = (char*) ciphertext_recvbuf;
                s_msg_size = (recvcount * recvtype_extent + 16+12);

            }else{
                //set the send buffer from tmp_buf and s_msg_size
                sbuf = (char*) tmp_buf;
                s_msg_size = (recvcount * recvtype_extent);
            }
            if(node_id != src_node_id){
                //set the recvbuffer from ciphertext_recvbuf and r_msg_size
                rbuf = (char*)((char*) ciphertext_recvbuf + curr_cnt * (recvcount * recvtype_extent + 16+12));
                r_msg_size = (recvcount * recvtype_extent + 16+12);
            }else{
                //set the recv buffer from tmp_buf and r_msg_size
                rbuf = (char*)((char*) tmp_buf + curr_cnt * recvcount * recvtype_extent);
                r_msg_size = (recvcount * recvtype_extent);
            }

            MPIR_PVAR_INC(allgather, bruck, send, rem * s_msg_size, MPI_CHAR); 
            MPIR_PVAR_INC(allgather, bruck, recv, rem * r_msg_size, MPI_CHAR); 
            mpi_errno = MPIC_Sendrecv(sbuf, rem * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, rem * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, errflag);

            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(sbuf, rem * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, rem * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/


            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }


            if(node_id != src_node_id){
                //decrypt recvd messages from ciphertext_recvbuf to temp_buf
                int decryption_index = curr_cnt;
                for(; decryption_index<curr_cnt+rem; ++decryption_index){
                    in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent + 16+12));
                    out = (char*)((char*) tmp_buf + decryption_index * recvcount * recvtype_extent);
                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                    if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){

                        printf("Error in Naive+ decryption: allgather Bruck (Default-II) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }

            }
        //end naive+
        }else{

            MPIR_PVAR_INC(allgather, bruck, send, rem * recvcount, recvtype); 
            MPIR_PVAR_INC(allgather, bruck, recv, rem * recvcount, recvtype); 
            mpi_errno = MPIC_Sendrecv(tmp_buf, rem * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG,
                                        ((char *) tmp_buf +
                                        curr_cnt * recvtype_extent),
                                        rem * recvcount, recvtype, src,
                                        MPIR_ALLGATHER_TAG, comm_ptr,
                                        MPI_STATUS_IGNORE, errflag);
            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(tmp_buf, rem * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG,
                                        ((char *) tmp_buf +
                                        curr_cnt * recvtype_extent),
                                        rem * recvcount, recvtype, src,
                                        MPIR_ALLGATHER_TAG, comm_ptr,
                                        MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }

    /* Rotate blocks in tmp_buf down by (rank) blocks and store
     * result in recvbuf. */

    mpi_errno = MPIR_Localcopy(tmp_buf, (comm_size - rank) * recvcount,
                               recvtype,
                               (char *) recvbuf +
                               rank * recvcount * recvtype_extent,
                               (comm_size - rank) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (rank) {
        mpi_errno = MPIR_Localcopy((char *) tmp_buf +
                                   (comm_size -
                                    rank) * recvcount * recvtype_extent,
                                   rank * recvcount, recvtype, recvbuf,
                                   rank * recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    void *tmp = (void*)(tmp_buf + recvtype_true_lb);
    MPIU_Free(tmp);
    /*int j;
    printf("%d finished and should wait for %d\n", rank, i);
    for(j=0; j<i; ++j){
	printf("%d is gonna wait for %d\n", rank, j);
        mpi_errno = MPIC_Wait((send_req_ptr[j]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[j]->status.MPI_ERROR;
        }
        MPID_Request_release(send_req_ptr[j]);
	printf("%d released %d\n", rank, j);
	}*/

  fn_fail:
    MPIR_TIMER_END(coll,allgather,bruck);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_Ring_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_Ring_MV2(const void *sendbuf,
                            int sendcount,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcount,
                            MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                            MPIR_Errflag_t *errflag)
{

    MPIR_TIMER_START(coll,allgather,ring);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    int left, right, jnext;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_ring, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    if(rank==0)
        printf("MPIR_Allgather_Ring_MV2\n");
    /************* For MPIC_Sendrecv_Plus ***************
    MPID_Request *send_req_ptr[comm_size-1];
    int p;
    for(p=0; p<comm_size-1; ++p){
        (send_req_ptr[p])=NULL;
    }
    ****************************************************/


    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent),
                                   recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /*
     * Now, send left to right.  This fills in the receive area in
     * reverse order.
     */
    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    /******************** Added by Mehran  ********************/
    MPID_Node_id_t node_id, send_node_id, recv_node_id;
    int node_gateway=0, send_idx=0;
    if(security_approach==2){
        
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        MPID_Get_node_id(comm_ptr, right, &send_node_id);
        MPID_Get_node_id(comm_ptr, left, &recv_node_id);
        if(node_id != send_node_id && node_id != recv_node_id){
            node_gateway=3;
        }else if(node_id != send_node_id){
            node_gateway=1;
        }else if(node_id != recv_node_id){
            node_gateway=2;
        }
        
    }
    /**********************************************************/



    j = rank;
    jnext = left;
    
    for (i = 1; i < comm_size; i++) {
        if(security_approach==2){//Naive PLus (Default)
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);
            char *in, *out, *sbuf, *rbuf;
            switch(node_gateway){
                case 1:
		    //printf("Case 1 in naive+ ring (default)\n");
                    //Encrypt before sending
                    in = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
                    out = (char*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));
                    RAND_bytes(out, 12); // 12 bytes of nonce

                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                        &ciphertext_len, max_out_len,
                                        out, 12,
                                        in, in_size,
                                        NULL, 0))
                    {
                        printf("Error in Naive+ encryption: allgather ring (Default)\n");
                        fflush(stdout);
                    }
                    MPIR_PVAR_INC(allgather, ring, send, recvcount*recvtype_extent+16+12, MPI_CHAR); 
                    MPIR_PVAR_INC(allgather, ring, recv, recvcount, recvtype); 
                    mpi_errno = MPIC_Sendrecv(out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    recvbuf + jnext * recvcount * recvtype_extent,
                                    recvcount, recvtype, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, errflag);
                    

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    recvbuf + jnext * recvcount * recvtype_extent,
                                    recvcount, recvtype, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/
                    send_idx = (send_idx+1)%2;
                    break;
                case 2:
                    //Decrypt after receiving
                    //printf("Case 2 in naive+ ring (default)\n");
                    sbuf = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
                    rbuf = (char*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));


                    MPIR_PVAR_INC(allgather, ring, send, recvcount, recvtype); 
                    MPIR_PVAR_INC(allgather, ring, recv, recvcount*recvtype_extent+16+12, MPI_CHAR); 
                    mpi_errno = MPIC_Sendrecv(sbuf, recvcount,
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount * recvtype_extent + 16+12,
                                    MPI_CHAR, left, MPIR_ALLGATHER_TAG,
                                    comm_ptr, MPI_STATUS_IGNORE, errflag);

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(sbuf, recvcount,
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount * recvtype_extent + 16+12,
                                    MPI_CHAR, left, MPIR_ALLGATHER_TAG,
                                    comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/


                    if(!EVP_AEAD_CTX_open(ctx, (recvbuf+jnext*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        rbuf, 12, rbuf+12, (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
			printf("Error in Naive+ decryption: allgather ring (default) I\n");
                            fflush(stdout);        
                        }
                    send_idx = (send_idx+1)%2;
                    break;
		case 3:
                    //Encrypt before sending and Decrypt after receiving
                    
                    //printf("Case 3 in naive+ ring (default)\n");
		    in = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
		    out = (char*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));
		    
		    rbuf = (char*)((char*) ciphertext_recvbuf + (send_idx+1)%2 * (recvcount * recvtype_extent + 12 + 16));
		    
		    RAND_bytes(out, 12); // 12 bytes of nonce
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                        &ciphertext_len, max_out_len,
                                        out, 12,
                                        in, in_size,
                                        NULL, 0))
                    {
                        printf("Error in Naive+ encryption: allgather ring (Default)\n");
                        fflush(stdout);
                    }
                    MPIR_PVAR_INC(allgather, ring, send, recvcount*recvtype_extent+16+12, MPI_CHAR); 
                    MPIR_PVAR_INC(allgather, ring, recv, recvcount*recvtype_extent+16+12, MPI_CHAR);
		    
                    mpi_errno = MPIC_Sendrecv((char *) out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount*recvtype_extent+16+12, MPI_CHAR, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, errflag);

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus((char *) out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount*recvtype_extent+16+12, MPI_CHAR, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/

		    //printf("%d is going to decrypt from %d to %d\n", rank, jnext * (recvcount * recvtype_extent + 12 + 16), jnext*recvcount*recvtype_extent);
                    if(!EVP_AEAD_CTX_open(ctx, (recvbuf+jnext*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        rbuf, 12, rbuf+12, (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                            printf("Error in Naive+ decryption: allgather ring (default) II \n");
                            fflush(stdout);        
                        }

                        send_idx = (send_idx+1)%2;


                    break;
                default:
                    //No encryption/decryption
		    //printf("Default case in naive+ ring (default)\n");
                    MPIR_PVAR_INC(allgather, ring, send, recvcount, recvtype); 
                    MPIR_PVAR_INC(allgather, ring, recv, recvcount, recvtype); 
                    mpi_errno = MPIC_Sendrecv(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/

                    break;

            }
            
        }else{//Not Naive+
            MPIR_PVAR_INC(allgather, ring, send, recvcount, recvtype); 
            MPIR_PVAR_INC(allgather, ring, recv, recvcount, recvtype); 
            mpi_errno = MPIC_Sendrecv(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);

            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
            **************************************************/

        }
        
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
    }
    
    /*for(j=0; j<comm_size-1; ++j){
        mpi_errno = MPIC_Wait((send_req_ptr[j]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[j]->status.MPI_ERROR;
        }
        MPID_Request_release(send_req_ptr[j]);
    }*/


  fn_fail:
    MPIR_TIMER_END(coll,allgather,ring);
    return (mpi_errno);
}

/* executes an allgather as a gather followed by a broadcast */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_gather_bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_gather_bcast_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,gather_bcast);
    int comm_size;
    int mpi_errno = MPI_SUCCESS;
    int gather_bcast_root = -1;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_gather_bcast, 1);

    comm_size = comm_ptr->local_size;
    if(comm_ptr->rank==0)
        printf("MPIR_Allgather_gather_bcast_MV2\n");
    MPIU_Assert(comm_ptr->dev.ch.shmem_coll_ok == 1);

    //Call gather (Calling mv2 version so that gather is tuned)
    //If IN_PLACE is used, gather is expected to handle it
    gather_bcast_root = 0;
    mpi_errno = MPIR_Gather_MV2(sendbuf, sendcount, sendtype,
				recvbuf, recvcount, recvtype,
				gather_bcast_root, comm_ptr, errflag);
    if (mpi_errno) {
	MPIR_ERR_POP(mpi_errno);
    }

    //gather_bcast_root has all data at this point

    //call bcast on the receive buffer
    mpi_errno = MPIR_Bcast_MV2(recvbuf, recvcount * comm_size, recvtype,
			       gather_bcast_root, comm_ptr, errflag);
    if (mpi_errno) {
	MPIR_ERR_POP(mpi_errno);
    }

  fn_fail:
    MPIR_TIMER_END(coll,allgather,gather_bcast);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_intra_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int comm_size;
    int mpi_errno = MPI_SUCCESS;
    int type_size;
    int comm_size_is_pof2 = 0;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0)) {
        return MPI_SUCCESS;
    }

    comm_size = comm_ptr->local_size;
    if(comm_ptr->rank==0)
        printf("MPIR_Allgather_intra_MV2\n");
    MPID_Datatype_get_size_macro(recvtype, type_size);

    /* check if comm_size is a power of two */
    comm_size_is_pof2 = comm_ptr->dev.ch.is_pof2;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    if ((recvcount * type_size <= allgather_tuning(comm_size, comm_size_is_pof2))
        && (comm_size_is_pof2 == 1)) {
        /* Short or medium size message and power-of-two no. of processes. Use
         * recursive doubling algorithm */
        mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                              recvcount, recvtype, comm_ptr, errflag);   
                    
    } else if (recvcount * type_size <= allgather_tuning(comm_size, comm_size_is_pof2)) {
        /* Short message and non-power-of-two no. of processes. Use
        * Bruck algorithm (see description above). */
        mpi_errno = MPIR_Allgather_Bruck_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                                 recvcount, recvtype, comm_ptr, errflag);

    } else {                
        /* long message or medium-size message and non-power-of-two
        * no. of processes. use ring algorithm. */
        mpi_errno = MPIR_Allgather_Ring_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                                recvcount, recvtype, comm_ptr, errflag);

    }
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int rank, size;
    int local_rank, local_size;
    int leader_comm_size = 0; 
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_extent = 0;  /* Datatype extent */
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr=NULL, *leader_commptr = NULL;

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    rank = comm_ptr->rank;
    size = comm_ptr->local_size; 
    if(rank==0)
        printf("MPIR_2lvl_Allgather_MV2\n");
    /* extract the rank,size information for the intra-node
     * communicator */
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_comm_size = leader_commptr->local_size;
    }

    /*If there is just one node, after gather itself,
     * root has all the data and it can do bcast*/
    if(local_rank == 0) {
        mpi_errno = MPIR_Gather_impl(sendbuf, sendcnt,sendtype, 
                                    (void*)((char*)recvbuf + (rank * recvcnt * recvtype_extent)), 
                                     recvcnt, recvtype,
                                     0, shmem_commptr, errflag);
    } else {
        /*Since in allgather all the processes could have 
         * its own data in place*/
        if(sendbuf == MPI_IN_PLACE) {
            mpi_errno = MPIR_Gather_impl((void*)((char*)recvbuf + (rank * recvcnt * recvtype_extent)), 
                                         recvcnt , recvtype, 
                                         recvbuf, recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcnt,sendtype, 
                                         recvbuf, recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        }
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Exchange the data between the node leaders*/
    if (local_rank == 0 && (leader_comm_size > 1)) {
        if(security_approach == 2){
            //NAIVE PLUS
	    //printf("Naive+\n");
            unsigned long count=0;
            unsigned long next, dest;
            unsigned int i;
            /*Step 1: Encryption*/


            int sendtype_sz, recvtype_sz;
            unsigned long  ciphertext_sendbuf_len = 0;
            sendtype_sz= recvtype_sz= 0;
            int var;
            var=MPI_Type_size(sendtype, &sendtype_sz);
            var=MPI_Type_size(recvtype, &recvtype_sz);

            int rank;
            rank = comm_ptr->rank;

            RAND_bytes(ciphertext_sendbuf, 12); // 12 bytes of nonce

            unsigned long t=0;
            t = (unsigned long)(local_size*sendtype_sz*sendcnt);
            unsigned long   max_out_len = (unsigned long) (16 + (local_size*sendtype_sz*sendcnt));

            if(!EVP_AEAD_CTX_seal(ctx, ciphertext_sendbuf+12,
                                &ciphertext_sendbuf_len, max_out_len,
                                ciphertext_sendbuf, 12,
                                recvbuf + (rank * recvcnt * recvtype_extent),  t,
                                NULL, 0))
            {
                    printf("Error in Naive+ encryption: allgather\n");
                    fflush(stdout);
            }
	    
            /*Step 2: Data exchange*/

            /*When data in each socket is different*/
            if (comm_ptr->dev.ch.is_uniform != 1) {

                int *displs = NULL;
                int *recvcnts = NULL;
                int *node_sizes;
                int i = 0;

                node_sizes = comm_ptr->dev.ch.node_sizes;

                displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                recvcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }
                recvcnts[0] = node_sizes[0] * recvcnt * recvtype_extent + 12 + 16;
                displs[0] = 0;

                for (i = 1; i < leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + (node_sizes[i - 1] * recvcnt * recvtype_extent + 12 + 16);
                    recvcnts[i] = node_sizes[i] * recvcnt * recvtype_extent + 12 + 16;
                }


                mpi_errno = MPIR_Allgatherv(ciphertext_sendbuf, (max_out_len+12), MPI_CHAR, 
                                        ciphertext_recvbuf, recvcnts, displs, MPI_CHAR,
                                        leader_commptr, errflag);
                MPIU_Free(displs);
                MPIU_Free(recvcnts);

                /*Step3: Decryption*/

                for( i = 0; i < leader_comm_size; i++){
                    next =(unsigned long )(displs[i]);
                    dest =(unsigned long )(node_sizes[i]*(recvcnt*recvtype_sz));
                    

                    if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                                    &count, (unsigned long )((node_sizes[i]*recvcnt*recvtype_sz)),
                                    (ciphertext_recvbuf+next), 12,
                                    (ciphertext_recvbuf+next+12), (unsigned long )((node_sizes[i]*recvcnt*recvtype_sz)+16),
                                    NULL, 0)){
                                printf("Decryption error in Naive+ allgather\n");fflush(stdout);        
                        }                               
                
                }


            } else {
                mpi_errno = MV2_Allgather_function(ciphertext_sendbuf, (max_out_len+12), MPI_CHAR,
                                                ciphertext_recvbuf, (max_out_len+12), MPI_CHAR,
                                                leader_commptr, errflag);


                /*Step3: Decryption*/
                for( i = 0; i < leader_comm_size; i++){
                    next =(unsigned long )(i*(max_out_len+12));
                    dest =(unsigned long )(i*(local_size*sendtype_sz*sendcnt));
                    

                    if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                                    &count, (unsigned long )((local_size*recvcnt*recvtype_sz)),
                                    (ciphertext_recvbuf+next), 12,
                                    (ciphertext_recvbuf+next+12), (unsigned long )((local_size*recvcnt*recvtype_sz)+16),
                                    NULL, 0)){
                                printf("Decryption error in Naive+ allgather\n");fflush(stdout);        
                        }                               
                
                }

            }

            


            /**********************     End of NAIVE PLUS ********************/
        }else{
            /*When data in each socket is different*/
            if (comm_ptr->dev.ch.is_uniform != 1) {

                int *displs = NULL;
                int *recvcnts = NULL;
                int *node_sizes;
                int i = 0;

                node_sizes = comm_ptr->dev.ch.node_sizes;

                displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                recvcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }
                recvcnts[0] = node_sizes[0] * recvcnt;
                displs[0] = 0;

                for (i = 1; i < leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + node_sizes[i - 1] * recvcnt;
                    recvcnts[i] = node_sizes[i] * recvcnt;
                }


                mpi_errno = MPIR_Allgatherv(MPI_IN_PLACE,
                                        (recvcnt*local_size),
                                        recvtype, 
                                        recvbuf, recvcnts,
                                        displs, recvtype,
                                        leader_commptr, errflag);
                MPIU_Free(displs);
                MPIU_Free(recvcnts);
            } else {
                mpi_errno = MV2_Allgather_function(MPI_IN_PLACE, 
                                                (recvcnt*local_size),
                                                recvtype,
                                                recvbuf, (recvcnt*local_size), recvtype,
                                                leader_commptr, errflag);

            }
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /*Bcast the entire data from node leaders to all other cores*/
    mpi_errno = MPIR_Bcast_impl (recvbuf, recvcnt * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
  
  fn_fail:
    return (mpi_errno);
}
/* end:nested */

/* This implements an allgather via a gather to leader processes,
 * an allgather across leaders, and a broadcast from each leader.
 * First all procs on a node execute a gather to the leader process.
 * Then the leaders exchange data via allgather/allgatherv.
 * Finally, the leaders broadcast the data to all procs on the node.
 * In this version, ranks do not need to be ordered, because the
 * leader will unpack data from a temporary buffer to the receive
 * buffer in the correct order before the broadcast. */
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_nonblocked_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_nonblocked_MV2(
    const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
          void *recvbuf, int recvcnt, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_nonblocked);
    int i;
    int mpi_errno = MPI_SUCCESS;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_nonblocked, 1);

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    int* node_sizes = comm_ptr->dev.ch.node_sizes;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_nonblocked_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* get true extent of recvtype */
    MPI_Aint recvtype_true_lb, recvtype_true_extent;
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    int local_rank = shmem_commptr->rank;
    int local_size = shmem_commptr->local_size;

    /* TODO: if ranks are in order, we can avoid the temp buffer and memcpy */
    int need_temp = 1;

    /* get info about communicator across node leaders, allocate temp buffer */
    MPID_Comm* leader_commptr = NULL;
    int leader_rank = -1;
    int leader_size = 0;
    void* tmpbuf = recvbuf;
    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        MPI_Comm leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_rank = leader_commptr->rank;
        leader_size = leader_commptr->local_size;

        /* allocate a temporary buffer */
        if (need_temp) {
            tmpbuf = MPIU_Malloc(size * recvcnt * recvtype_extent);
            if (!tmpbuf) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**nomem", 0);
                return mpi_errno;
            }

            /* adjust for potential negative lower bound in datatype */
            tmpbuf = (void*)((char*) tmpbuf - recvtype_true_lb);
        }
    }

    /* ----------------------------------------------
     * Gather data from procs on same node to leader
     * ---------------------------------------------- */

    /* gather data to leaders on each node */
    /* compute number of items to receive ahead of our spot in the buffer */
    MPI_Aint preceding_count = 0;
    for (i=0; i < leader_rank; i++) {
        preceding_count += node_sizes[i] * recvcnt;
    }
    
    if (local_rank == 0) {
    
        /* compute location to receive data from procs on our node */
        void* rbuf = (void*)((char*)tmpbuf + (preceding_count * recvtype_extent));

        /* gather data from procs on our node */
        if (sendbuf == MPI_IN_PLACE) {
            /* data is in our receive buffer indexed by our rank */
            void* sbuf = (void*)((char*)recvbuf + (rank * recvcnt * recvtype_extent));
            mpi_errno = MPIR_Gather_impl(sbuf, recvcnt, recvtype,
                                         rbuf, recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcnt, sendtype,
                                            rbuf, recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        }
    } else {
        /* send data to leader on our node */
        if (sendbuf == MPI_IN_PLACE) {
            /* data is in our receive buffer indexed by our rank */
            void* sbuf = (void*)((char*)recvbuf + (rank * recvcnt * recvtype_extent));
            mpi_errno = MPIR_Gather_impl(sbuf, recvcnt, recvtype,
                                         NULL, recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcnt, sendtype,
                                         NULL,    recvcnt, recvtype,
                                         0, shmem_commptr, errflag);
        }
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* ----------------------------------------------
     * Execute allgather across leaders
     * ---------------------------------------------- */

    /*If there is just one node, after gather itself,
     * root has all the data and it can do bcast*/

    /* Exchange the data between the node leaders */
    if (local_rank == 0 && (leader_size > 1)) {
        if(security_approach == 2){
            /************** NAIVE PLUS (NB)***************/
	        //printf("%d @ 0\n", rank);
            unsigned long count=0;
            unsigned long next, dest;
            unsigned int i;

            /*Step 1: Encryption*/
            int sendtype_sz, recvtype_sz;
            unsigned long  ciphertext_sendbuf_len = 0;
            sendtype_sz= recvtype_sz= 0;
            int var;
            var=MPI_Type_size(sendtype, &sendtype_sz);
            var=MPI_Type_size(recvtype, &recvtype_sz);

            RAND_bytes(ciphertext_sendbuf, 12); // 12 bytes of nonce

            unsigned long t=0;
            t = (unsigned long)(local_size*sendtype_sz*sendcnt);
            unsigned long   max_out_len = (unsigned long) (16 + (local_size*sendtype_sz*sendcnt));
	    
            if(!EVP_AEAD_CTX_seal(ctx, ciphertext_sendbuf+12,
                                &ciphertext_sendbuf_len, max_out_len,
                                ciphertext_sendbuf, 12,
                                tmpbuf + (preceding_count * recvtype_extent),  t,
                                NULL, 0))
            {
                    printf("Error in Naive+ encryption: allgather\n");
                    fflush(stdout);
            }
	        //printf("%d @ 1\n", rank);
            /*Step 2: Data exchange*/

            /*When data in each socket is different*/
            if (comm_ptr->dev.ch.is_uniform != 1) {
                int *displs = NULL;
                int *recvcnts = NULL;
                int *node_sizes;
                int i = 0;

                node_sizes = comm_ptr->dev.ch.node_sizes;

                displs = MPIU_Malloc(sizeof (int) * leader_size);
                recvcnts = MPIU_Malloc(sizeof (int) * leader_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }
                recvcnts[0] = node_sizes[0] * recvcnt * recvtype_extent + 12 + 16;
                displs[0] = 0;

                for (i = 1; i < leader_size; i++) {
                    displs[i] = displs[i - 1] + (node_sizes[i - 1] * recvcnt * recvtype_extent + 12 + 16);
                    recvcnts[i] = node_sizes[i] * recvcnt * recvtype_extent + 12 + 16;
                }

                mpi_errno = MPIR_Allgatherv_impl(ciphertext_sendbuf, (max_out_len+12), MPI_CHAR, 
                                        ciphertext_recvbuf, recvcnts, displs, MPI_CHAR,
                                        leader_commptr, errflag);
		    //MPIU_Free(displs);
                //MPIU_Free(recvcnts);
		    //printf("%d @ 2 (1)\n", rank);
                /*Step3: Decryption*/

                for( i = 0; i < leader_size; i++){
		    if(i!= leader_rank){
			
			next =(unsigned long )(displs[i]);
			dest =(unsigned long )(node_sizes[i]*(recvcnt*recvtype_sz));
                    
			//printf("%d is going to decrypt from %d and put at %d, node size = %d\n", rank, next,dest, node_sizes[i] );
			if(!EVP_AEAD_CTX_open(ctx, ((tmpbuf+dest)),
					      &count, (unsigned long )((node_sizes[i]*recvcnt*recvtype_sz)),
					      (ciphertext_recvbuf+next), 12,
					      (ciphertext_recvbuf+next+12), (unsigned long )((node_sizes[i]*recvcnt*recvtype_sz)+16), NULL, 0)){
			    printf("Decryption error in Naive+ allgather (NB) 1\n");fflush(stdout);        
                        }                               
		    }
                }
		//printf("%d @ 3 (1)\n", rank);
		MPIU_Free(recvcnts);
		MPIU_Free(displs);
            } else {
		
                mpi_errno = MPIR_Allgather_impl(ciphertext_sendbuf, (max_out_len+12), MPI_CHAR,
                                                ciphertext_recvbuf, (max_out_len+12), MPI_CHAR,
                                                leader_commptr, errflag);
		//printf("%d @ 2 (2)\n", rank);
		/*Step3: Decryption*/
                for( i = 0; i < leader_size; i++){
		    if(i != leader_rank){
			
			next =(unsigned long )(i*(max_out_len+12));
			dest =(unsigned long )(i*(local_size*sendtype_sz*sendcnt));
                    

			if(!EVP_AEAD_CTX_open(ctx, ((tmpbuf+dest)),
					      &count, (unsigned long )((local_size*recvcnt*recvtype_sz)),
					      (ciphertext_recvbuf+next), 12,
					      (ciphertext_recvbuf+next+12), (unsigned long )((local_size*recvcnt*recvtype_sz)+16), NULL, 0)){
			    printf("Decryption error in Naive+ allgather (NB) 2\n");fflush(stdout);        
                        }                               
		    }
                }
		//	printf("%d @ 3 (2)\n", rank);

            }






            /********* END OF NAIVE PLUS **********/
        }else{

            /* When data in each socket is different */
            if (comm_ptr->dev.ch.is_uniform != 1) {
                /* allocate memory for counts and displacements arrays */
                int* displs = MPIU_Malloc(sizeof (int) * leader_size);
                int* counts = MPIU_Malloc(sizeof (int) * leader_size);
                if (!displs || !counts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }

                /* set values in our counts and displacements arrays */
                displs[0] = 0;
                counts[0] = node_sizes[0] * recvcnt;
                for (i = 1; i < leader_size; i++) {
                    displs[i] = displs[i - 1] + node_sizes[i - 1] * recvcnt;
                    counts[i] = node_sizes[i] * recvcnt;
                }

                /* execute allgatherv across leader ranks */
                mpi_errno = MPIR_Allgatherv_impl(MPI_IN_PLACE, (recvcnt*local_size), recvtype,
                                                tmpbuf, counts, displs, recvtype,
                                                leader_commptr, errflag);

                /* free counts and displacements arrays */
                MPIU_Free(displs);
                MPIU_Free(counts);
		//printf("%d @ 4\n", rank);
            } else {
                /* execute allgather across leader ranks */
                mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, (recvcnt*local_size), recvtype,
                                                tmpbuf, (recvcnt*local_size), recvtype,
                                                leader_commptr, errflag);

            }
        }

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* ----------------------------------------------
     * Unpack data into receive buffer in correct order
     * ---------------------------------------------- */

    /* ----------------------------------------------
     * TODO: can avoid temp buffer and mem copy when in block
     * ---------------------------------------------- */
    /* ----------------------------------------------
     * TODO: faster memcpy available?
     * ---------------------------------------------- */

    /* leaders copy data from temp buffer to receive buffer in correct order */
    if (local_rank == 0 && need_temp) {
        /* point to start of temp buffer */
        char* sbuf = (char*)tmpbuf;

        /* copy data for each rank from temp buffer to receive buffer */
        for (i = 0; i < size; i++) {
            /* get next rank in list */
            int dstrank = comm_ptr->dev.ch.rank_list[i];

            /* compute position in receive buffer for this rank */
            void* rbuf = (void*)((char*)recvbuf + dstrank * recvcnt * recvtype_extent);

            /* copy data to its correct place */
            mpi_errno = MPIR_Localcopy(sbuf, recvcnt, recvtype,
                                       rbuf, recvcnt, recvtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* update pointer to next spot in temp buffer */
            sbuf += recvcnt * recvtype_extent;
        }
	
        /* free the temporary buffer if we allocated one */
        tmpbuf = (void*)((char*)tmpbuf + recvtype_true_lb);
        MPIU_Free(tmpbuf);
	

    }
    

    /* ----------------------------------------------
     * Broadcast receive buffer from leader to all procs on the node
     * ---------------------------------------------- */

    /* Bcast the entire data from node leaders to other ranks on node */
    mpi_errno = MPIR_Bcast_impl(recvbuf, recvcnt * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

  fn_fail:
    MPIR_TIMER_END(coll,allgather,2lvl_nonblocked);
    return (mpi_errno);
}




/**************** Added by Mehran *****************/
/**
 * 
 * In this function, first, we perform an inter-node 
 * RD allgather where all the ranks exchange 
 * their data with their peers on other nodes. 
 * Then, we perform inter-node allgather where all
 * the ranks at each node, exchange all the data
 * that they have using RD algorithm. This version is 
 * used when ranks are not block-ordered. So, we use a 
 * tmp buffer for allgather and at the end of the algorithm
 * we perform ordering.
 * 
 **/
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_multileader_rd_nonblocked);
    //    printf("MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2\n");
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, j, k, f;
    char *in, *out, *sbuf, *rbuf, *tmp_buf;
    MPI_Status status;
    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    
    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int comm_size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2\n");
    int p = shmem_commptr->local_size; // number of ranks per node
    int n = (int) (comm_size / p); // number of nodes
    
    int mask, dst_tree_root, my_tree_root, send_offset, recv_offset, send_req_idx = 0,
        last_recv_cnt = 0, nprocs_completed, offset, tmp_mask, out_index, dst,
         dst_node, my_node, tree_root, rank_index, dst_index, curr_cnt=1, recently_received;

    /************* For MPIC_Sendrecv_Plus ************
    MPID_Request *send_req_ptr[2*(comm_size-1)];
    int s;
    for(s=0; s<comm_size-1; ++s){
        (send_req_ptr[s])=NULL;
    }
    /*************************************************/
    //printf("%d  - %d started\n", recvcount, rank);
    /* get extent of receive type */
    MPI_Aint recvtype_extent, sendtype_extent, recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    recvbuf_extent =
        recvcount * comm_size * (MPIR_MAX(recvtype_true_extent, recvtype_extent));
    /* allocate a temporary buffer of the same size as recvbuf. */
    tmp_buf = MPIU_Malloc(recvbuf_extent);
    /* --BEGIN ERROR HANDLING-- */
    if (!tmp_buf) {
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**nomem", 0);
        return mpi_errno;
    }
    /* --END ERROR HANDLING-- */

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_multileader_rd_nonblocked, 1);

    int shared_mem_rank, shared_mem_dst; 
    shared_mem_rank = shmem_commptr->rank;

    /* Find the local node id*/
    MPID_Node_id_t node_id;    
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    my_node = node_id;
    rank_index = comm_ptr->dev.ch.rank_list_index;

    //    printf("NB - %d [%d] is in node %d and global_blocked = %d, equal_local_sizes = %d\n", rank, rank_index, my_node, comm_ptr->dev.ch.is_global_block, comm_ptr->dev.ch.equal_local_sizes);
    /* First, load the "local" version in the tmp_buf. */
    rbuf = ((char*) tmp_buf + rank_index * recvcount * recvtype_extent);
    

    if (sendbuf != MPI_IN_PLACE) {
        sbuf = ((char*) sendbuf);
	//if(recvcount==16)
	//printf("%d - %d copies from send buffer to %d at tmp_buffer\n", recvcount, rank, rank_index);
    }else{
        sbuf = ((char*) recvbuf + rank * recvcount * recvtype_extent);
	//	if(recvcount==16)
	//printf("%d - %d copies from %d at recv_buffer to %d at tmp_buffer\n", recvcount, rank, rank, rank_index);
    }

    /* copy data from send buffer to receive buffer */
    
    mpi_errno = MPIR_Localcopy(
        sendbuf, sendcount, sendtype,
        rbuf,    recvcount, recvtype
    );
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    
    
    if(security_approach==2){
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) tmp_buf + rank_index * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + my_node * (recvcount * recvtype_extent + 12 + 16));

        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long in_size=0;
        in_size = (unsigned long)(sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        //if(recvcount==16)
	//printf("%d - %d is going to encrypt from %d to %d\n", recvcount, rank, rank_index, my_node );
        if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, in_size,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather MultiLeader-RD-NB\n");
                fflush(stdout);
        }
    }else{
        /** 
         * Only for the Unencrypted version
         * Copy from recv_buf to ciphertext_recvbuf
         **/
        void* rbuf = (void*)((char*) tmp_buf + rank_index * recvcount * recvtype_extent);
        void* crbuf = (void*)((char*) ciphertext_recvbuf + my_node * recvcount * recvtype_extent);
        //if(recvcount==16)
	//printf("%d - %d is going to copy from %d to %d\n", recvcount, rank, rank_index, my_node );
        mpi_errno = MPIR_Localcopy(
            rbuf, sendcount, sendtype,
            crbuf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    
    /* Inter-Node RD*/
    
    mask = 0x1;
    i = 0;
    
    while (mask < comm_size) {
        dst_node = my_node ^ mask;
        dst_index = rank_index + (dst_node - my_node) * p;
        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = dst_node >> i;
        dst_tree_root <<= i;

        my_tree_root = my_node >> i;
        my_tree_root <<= i;

        if (dst_index < comm_size) { 
            dst = comm_ptr->dev.ch.rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/

            if(security_approach==2){
                send_offset = my_tree_root * (recvcount * recvtype_extent +16 +12);
                recv_offset = dst_tree_root * (recvcount * recvtype_extent +16 +12);
                //if(recvcount==16)
		//printf("%d - %d is going to send (Inter-I) %d from %d to %d and receive %d at %d ,send_req_idx=%d\n", recvcount, rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root, send_req_idx);
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (n - dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                /*mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);*/
		 mpi_errno =
                    MPIC_Sendrecv(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);

                    
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent+16+12));
                curr_cnt += recently_received;
                //if(recvcount==16)
		//printf("%d - %d received %d (%d) and curr_cunt = %d\n", recvcount, rank, last_recv_cnt, recently_received, curr_cnt);

                for(f=0; f<recently_received; ++f){
                    unsigned long count=0;
                    
                    in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)));
                    out_index = rank_index + (dst_tree_root+f-my_node) * p;
                    out = (char*)((char*) tmp_buf + out_index * recvcount * recvtype_extent);

                    //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                    in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                    NULL, 0)){

                        printf("Error in Naive+ decryption: allgather ML_RD (I) while %d tried to decrypt from %d to %d\n", rank, recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)), recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }// End for





            }else{//if not Naive+
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = my_tree_root * recvcount * recvtype_extent;
                recv_offset = dst_tree_root * recvcount * recvtype_extent;

                //if(recvcount==16)
		//printf("%d - %d is going to send (Inter-I) %d from %d to %d and receive %d at %d, send_req_idx=%d\n", recvcount, rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root, send_req_idx);

                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, curr_cnt*recvcount, recvtype); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (n - dst_tree_root) * recvcount, recvtype); 
                /*mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);*/
		mpi_errno =
                    MPIC_Sendrecv(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);

                    
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                curr_cnt += recently_received;

                //if(recvcount==16)
                //    printf("%d received %d (%d) and curr_cunt = %d\n", rank, last_recv_cnt, recently_received, curr_cnt);


                for(f=0; f<recently_received; ++f){
                    
                    in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * recvcount * recvtype_extent));
                    out_index = rank_index + (dst_tree_root+f-my_node) * p;
                    out = (char*)((char*) tmp_buf + out_index * recvcount * recvtype_extent);

                    //if(recvcount==16)
                      //  printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    mpi_errno = MPIR_Localcopy(
                        in, recvcount, recvtype,
                        out, recvcount, recvtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }// End for
            }//end else
            

        }//end if dst_index < comm_size

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */

    
        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > n) {
            
            nprocs_completed = n - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                dst_node = my_node ^ tmp_mask;
                dst_index = rank_index + (dst_node - my_node) * p;

                tree_root = my_node >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->dev.ch.rank_list[dst_index];
                
                if ((dst_node > my_node) && (my_node < tree_root + nprocs_completed)
                    && (dst_node >= tree_root + nprocs_completed)) {
                        if(security_approach==2){
                            //Naive+
                            // if(comm_size - (my_tree_root+mask) < recently_received)
                            //     recently_received = comm_size - (my_tree_root+mask);
                            

                            sbuf = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask) * (recvcount * recvtype_extent + 16+12));
                            
                            //send
                            MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, recently_received * (recvcount*recvtype_extent + 16+12), MPI_CHAR); 
                            //printf("%d  - %d is going to send (II) %d from %d to %d, send_req_idx=%d\n", recvcount, rank, recently_received, (my_tree_root + mask), dst, send_req_idx);
                            /*mpi_errno =
                                MPIC_Send_Plus(sbuf,
                                        recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);*/
			    mpi_errno =
                                MPIC_Send(sbuf,
                                        recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                            if (mpi_errno) {
                                /* for communication errors, just record the error but
                                continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                
                            }
                            
                            //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                            
                        }//End Naive +
                        else{
                            MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, last_recv_cnt, recvtype);
			    //printf("%d  - %d is going to send (II) %d from %d to %d, send_req_idx=%d\n", recvcount, rank, recently_received, (my_tree_root + mask), dst, send_req_idx);
                            /*mpi_errno =
                                MPIC_Send_Plus(((char *) ciphertext_recvbuf + offset),
                                            recently_received*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);*/

			    mpi_errno =
                                MPIC_Send(((char *) ciphertext_recvbuf + offset),
                                            recently_received*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);
			    
                            /* last_recv_cnt was set in the previous
                            * receive. that's the amount of data to be
                            * sent now. */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }

			
                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((dst_node < my_node) &&
                            (dst_node < tree_root + nprocs_completed) &&
                            (my_node >= tree_root + nprocs_completed)) {
                    if(security_approach==2){
                        
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (comm_size - (my_tree_root + mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + (my_tree_root + mask)*(recvcount*recvtype_extent + 16+12)),
                                        (n -
                                        (my_tree_root +
                                        mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                        curr_cnt += recently_received;
                        //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                        //decrypt the received messages

                        for(f=0; f<recently_received; ++f){
                            unsigned long count=0;
                            
                            in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12));
                            out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                            out = (char*)((char*) tmp_buf + out_index * recvcount * recvtype_extent);

                            //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                            if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                            NULL, 0)){

                                printf("Error in Naive+ decryption: allgather ML_RD (II) while %d tried to decrypt from %d to %d\n", rank, (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12), recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);
                                fflush(stdout);        
                            }
                        }// End for

                    }else{ // Not Naive+
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (n - (my_tree_root + mask)) * recvcount, recvtype);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + offset),
                                        (n -
                                        (my_tree_root +
                                        mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount)); 
                        curr_cnt += recently_received;
			//printf("%d  - %d recved (II) %d from %d at %d, curr_cnt=%d\n", recvcount, rank, recently_received, dst, offset, curr_cnt);

                        for(f=0; f<recently_received; ++f){
                            unsigned long count=0;
                            
                            in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent));
                            out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                            out = (char*)((char*) tmp_buf + out_index * recvcount * recvtype_extent);

                            mpi_errno = MPIR_Localcopy(
                                in, recvcount, recvtype,
                                out, recvcount, recvtype);
                            if (mpi_errno) {
                                MPIR_ERR_POP(mpi_errno);
                            }
                        }// End for


                    }
                    
                }
                tmp_mask >>= 1;
                k--;
            }//end while tmp_mask
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }
    //    if(recvcount==1048576)
	//    printf("%d finished Inter-Node (ML-RD-NB)\n", rank);

    /* Intra-Node RD*/
    

    mask = 0x1;
    i = 0;
    curr_cnt = 1;
    
    rank_index = comm_ptr->dev.ch.rank_list_index;
    
    while (mask <  p) {
        shared_mem_dst = shared_mem_rank ^ mask;
        dst_index = my_node * p + shared_mem_dst;
        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = shared_mem_dst >> i;
        dst_tree_root <<= i;

        my_tree_root = shared_mem_rank >> i;
        my_tree_root <<= i;

        if ( dst_index < (my_node + 1) * p && dst_index>= (my_node * p) ) {
            dst = comm_ptr->dev.ch.rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/


            /** #TODO: for loop to send all the msgs received in the inter-node step**/
            int previous_recv_count = 0;
            for(f=0; f< n; ++f){
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = ((my_tree_root + f * p) % comm_size) * recvcount * recvtype_extent;
                recv_offset = ((dst_tree_root + f * p) % comm_size) * recvcount * recvtype_extent;

                /*if(recvcount==1048576 && (rank == 41 || rank == 48 || rank == 55))
		  printf("%d is going to send (Intra-I) %d from %d to %d and receive %d at %d, send_req_idx=%d\n", rank, curr_cnt, (my_tree_root + f * p) % comm_size, dst, (p - dst_tree_root), (dst_tree_root + f * p) % comm_size, send_req_idx);*/
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, curr_cnt*recvcount, recvtype); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (p - dst_tree_root) * recvcount, recvtype); 
                /*mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) tmp_buf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) tmp_buf + recv_offset),
                                    (p -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);*/
		mpi_errno =
                    MPIC_Sendrecv(((char *) tmp_buf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) tmp_buf + recv_offset),
                                    (p -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
		
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }
                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                if(previous_recv_count==0){
                    previous_recv_count = recently_received;
                }else{
                    if(previous_recv_count != recently_received){
                        printf("ERROR in Multileader-RD-NB!\nReceived sizes are not equal!\n");
                    }
                }
                //printf("%d  - %d received (Intra-I) %d from %d and curr_cnt is now %d\n", recvcount, rank, recently_received, dst, curr_cnt);
            }//end for f
            curr_cnt += recently_received;
            
        }

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */
    
        /* --BEGIN EXPERIMENTAL-- */
        


        if (dst_tree_root + mask > p) {
            nprocs_completed = p - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                shared_mem_dst = shared_mem_rank ^ tmp_mask;
                dst_index = my_node * p + shared_mem_dst;

                tree_root = shared_mem_rank >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->dev.ch.rank_list[dst_index];
                
                if ((shared_mem_dst > shared_mem_rank) && (shared_mem_rank < tree_root + nprocs_completed)
                    && (shared_mem_dst >= tree_root + nprocs_completed)) {

                    for(f=0; f<n; ++f){
                        sbuf = (char*)((char*) tmp_buf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                    
                        //send
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, send, recently_received * recvcount, recvtype);
			/*if((rank==20 || rank ==27 || rank ==13) && send_req_idx>=29 && recvcount==1048576)
			  printf("%d is going to send %d from %d to %d, send_req_idx=%d\n", rank, recently_received, (my_tree_root + mask+ f*p)%comm_size, dst, send_req_idx);*/
                        mpi_errno =
                            MPIC_Send(sbuf,
                                    recently_received * recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG, comm_ptr, errflag);
			/*mpi_errno =
                            MPIC_Send_Plus(sbuf,
                                    recently_received * recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);*/
			
                        if (mpi_errno) {
                            /* for communication errors, just record the error but
                            continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            
                        }
                    }//end for f
                            

                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((shared_mem_dst < shared_mem_rank) &&
                            (shared_mem_dst < tree_root + nprocs_completed) &&
                            (shared_mem_rank >= tree_root + nprocs_completed)) {
                    int previous_recv_count = 0;
                    for(f=0; f<n; ++f){
                        rbuf = (char*)((char*) tmp_buf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd_nonblocked, recv, (p - (my_tree_root + mask)) * recvcount, recvtype);
			/*if((rank==48 && dst ==20) || (rank ==55 && dst==27) || (rank ==41 && dst == 13) && recvcount==1048576)
			  printf("%d is going to recv %d at %d from %d\n", rank, p - (my_tree_root + mask), (my_tree_root + mask+ f*p)%comm_size, dst);*/
                        mpi_errno =
                            MPIC_Recv(rbuf,
                                        (p - (my_tree_root + mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount));
                        if(previous_recv_count==0){
                            previous_recv_count = recently_received;
                        }else{
                            if(previous_recv_count != recently_received){
                                printf("ERROR in Multileader-RD-NB!\nReceived sizes are not equal!\n");
                            }
                        }
			/*if((rank==48 && dst ==20) || (rank ==55 && dst==27) || (rank ==41 && dst == 13) && recvcount==1048576)
			  printf("%d received (Intra-II) %d from %d at %d and curr_cnt is now %d\n", rank, recently_received, dst,(my_tree_root + mask+ f*p)%comm_size, curr_cnt);*/
                    }//end for f
                    curr_cnt += recently_received;

  
                }
                tmp_mask >>= 1;
                k--;

            }//end while tmp_mask

        }

        // /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    //printf("%d - %d Done!\n", recvcount, rank);
    //printf("%d finished Intra-Node (ML-RD)\n", rank);

    sbuf = (char*)tmp_buf;

    /* copy data for each rank from temp buffer to receive buffer */
    for (i = 0; i < comm_size; i++) {
        /* get next rank in list */
        int dstrank = comm_ptr->dev.ch.rank_list[i];

        /* compute position in receive buffer for this rank */
        void* rbuf = (void*)((char*)recvbuf + dstrank * recvcount * recvtype_extent);

        /* copy data to its correct place */
        mpi_errno = MPIR_Localcopy(sbuf, recvcount, recvtype,
                                    rbuf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* update pointer to next spot in temp buffer */
        sbuf += recvcount * recvtype_extent;
    }

    /* free the temporary buffer if we allocated one */
    tmp_buf = (void*)((char*)tmp_buf + recvtype_true_lb);
    MPIU_Free(tmp_buf);

    /*if(recvcount==1048576 && (rank==13 || rank==20 || rank==27))
      printf("%d Rotation finished | send_req_idx=%d\n", rank, send_req_idx);*/

    /*for(i=0; i<send_req_idx; ++i){
	/*if(recvcount==1048576 && (rank==13 || rank==20 || rank==27))
	  printf("%d gonna wait for %d out of %d\n", rank, i, send_req_idx-1);
        mpi_errno = MPIC_Wait((send_req_ptr[i]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[i]->status.MPI_ERROR;
        }
	 MPID_Request_release(send_req_ptr[i]);
	 }*/
    
    //printf("%d- %d finished\n", recvcount, rank);

    fn_fail:
        MPIR_TIMER_END(coll,allgather,2lvl_multileader_rd_nonblocked);

	    return (mpi_errno);
}
/*****************************************************/















/**************** Added by Mehran *****************/
/**
 * 
 * In this function, first, we perform an inter-node 
 * RD allgather where all the ranks exchange 
 * their data with their peers on other nodes. 
 * Then, we perform inter-node allgather where all
 * the ranks at each node, exchange all the data
 * that they have using RD algorithm.
 * 
 **/
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Multileader_RD_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Multileader_RD_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_multileader_rd);

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, j, k, f;
    char *in, *out, *sbuf, *rbuf;
    MPI_Status status;
    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    
    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int comm_size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Multileader_RD_MV2\n");
    int p = shmem_commptr->local_size; // number of ranks per node
    int n = (int) (comm_size / p); // number of nodes
    
    int mask, dst_tree_root, my_tree_root, send_offset, recv_offset, send_req_idx = 0,
        last_recv_cnt = 0, nprocs_completed, offset, tmp_mask, out_index, dst,
         dst_node, my_node, tree_root, rank_index, dst_index, curr_cnt=1, recently_received;

    /************* For MPIC_Sendrecv_Plus ************/
    MPID_Request *send_req_ptr[20*(comm_size-1)];
    int s;
    for(s=0; s<comm_size-1; ++s){
        (send_req_ptr[s])=NULL;
    }
    /*************************************************/

    /* get extent of receive type */
    MPI_Aint recvtype_extent, sendtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_multileader_rd, 1);

    /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcount, sendtype,
            rbuf,    recvcount, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }


    /* Find the local node id*/
    MPID_Node_id_t node_id;    
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    my_node = node_id;

    rank_index = comm_ptr->dev.ch.rank_list_index;
    //    printf("B - %d [%d] is in node %d and global blocked = %d, equal_local_sizes = %d\n", rank, rank_index, my_node, comm_ptr->dev.ch.is_global_block, comm_ptr->dev.ch.equal_local_sizes);
    
    if(security_approach==2){
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[rank_index] * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + my_node * (recvcount * recvtype_extent + 12 + 16));

        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long in_size=0;
        in_size = (unsigned long)(sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        //printf("%d (%d) is going to encrypt from %d to %d\n", rank, local_rank, rank * recvcount * recvtype_extent, rank * (recvcount * recvtype_extent + 12 + 16) );
        if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, in_size,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather MultiLeader-RD\n");
                fflush(stdout);
        }
    }else{
        /** 
         * Only for the Unencrypted version
         * Copy from recv_buf to ciphertext_recvbuf
         **/
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);
        void* crbuf = (void*)((char*) ciphertext_recvbuf + my_node * recvcount * recvtype_extent);
        mpi_errno = MPIR_Localcopy(
            rbuf, sendcount, sendtype,
            crbuf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    
    /* Inter-Node RD*/
    
    mask = 0x1;
    i = 0;
    
    while (mask < comm_size) {
        dst_node = my_node ^ mask;
        dst_index = rank_index + (dst_node - my_node) * p;
        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = dst_node >> i;
        dst_tree_root <<= i;

        my_tree_root = my_node >> i;
        my_tree_root <<= i;

        if (dst_index < comm_size) { 
            dst = comm_ptr->dev.ch.rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/

            if(security_approach==2){
                send_offset = my_tree_root * (recvcount * recvtype_extent +16 +12);
                recv_offset = dst_tree_root * (recvcount * recvtype_extent +16 +12);

                //printf("%d is going to send (Inter-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root);
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);

                    
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent+16+12));
                curr_cnt += recently_received;

                //printf("%d received %d (%d) and curr_cunt = %d\n", rank, last_recv_cnt, recently_received, curr_cnt);

                for(f=0; f<recently_received; ++f){
                    unsigned long count=0;
                    
                    in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)));
                    out_index = rank_index + (dst_tree_root+f-my_node) * p;
                    out = (char*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                    in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                    NULL, 0)){

                        printf("Error in Naive+ decryption: allgather ML_RD (I) while %d tried to decrypt from %d to %d\n", rank, recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)), recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }// End for





            }else{//if not Naive+
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = my_tree_root * recvcount * recvtype_extent;
                recv_offset = dst_tree_root * recvcount * recvtype_extent;

                // if(recvcount==16)
                //     printf("%d is going to send (Inter-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root);

                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*recvcount, recvtype); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - dst_tree_root) * recvcount, recvtype); 
                mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);

                    
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                curr_cnt += recently_received;

                // if(recvcount==16)
                //     printf("%d received %d (%d) and curr_cunt = %d\n", rank, last_recv_cnt, recently_received, curr_cnt);


                for(f=0; f<recently_received; ++f){
                    
                    in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * recvcount * recvtype_extent));
                    out_index = rank_index + (dst_tree_root+f-my_node) * p;
                    out = (char*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    // if(recvcount==16)
                    //     printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                    mpi_errno = MPIR_Localcopy(
                        in, recvcount, recvtype,
                        out, recvcount, recvtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }// End for
            }//end else
            

        }//end if dst_index < comm_size

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */

    
        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > n) {
            
            nprocs_completed = n - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                dst_node = my_node ^ tmp_mask;
                dst_index = rank + (dst_node - my_node) * p;

                tree_root = my_node >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->dev.ch.rank_list[dst_index];
                
                if ((dst_node > my_node) && (my_node < tree_root + nprocs_completed)
                    && (dst_node >= tree_root + nprocs_completed)) {
                        if(security_approach==2){
                            //Naive+
                            // if(comm_size - (my_tree_root+mask) < recently_received)
                            //     recently_received = comm_size - (my_tree_root+mask);
                            

                            sbuf = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask) * (recvcount * recvtype_extent + 16+12));
                            
                            //send
                            MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, recently_received * (recvcount*recvtype_extent + 16+12), MPI_CHAR); 
                            //printf("%d is going to send (II) %d from %d to %d\n", rank, recently_received, (my_tree_root + mask), dst);
                            mpi_errno =
                                MPIC_Send_Plus(sbuf,
                                        recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);

                            /*mpi_errno =
                                MPIC_Send(sbuf,
                                        recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, errflag);*/



                            if (mpi_errno) {
                                /* for communication errors, just record the error but
                                continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                
                            }
                            
                            //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                            
                        }//End Naive +
                        else{
                            MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, last_recv_cnt, recvtype); 
                            mpi_errno =
                                MPIC_Send_Plus(((char *) ciphertext_recvbuf + offset),
                                            recently_received*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);

                            /*mpi_errno =
                                MPIC_Send(((char *) ciphertext_recvbuf + offset),
                                            recently_received*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);*/
                            
                            /* last_recv_cnt was set in the previous
                            * receive. that's the amount of data to be
                            * sent now. */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }
                    
                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((dst_node < my_node) &&
                            (dst_node < tree_root + nprocs_completed) &&
                            (my_node >= tree_root + nprocs_completed)) {
                    if(security_approach==2){
                        
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (comm_size - (my_tree_root + mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + (my_tree_root + mask)*(recvcount*recvtype_extent + 16+12)),
                                        (n -
                                        (my_tree_root +
                                        mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                        curr_cnt += recently_received;
                        //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                        //decrypt the received messages

                        for(f=0; f<recently_received; ++f){
                            unsigned long count=0;
                            
                            in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12));
                            out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                            out = (char*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                            //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                            if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                            NULL, 0)){

                                printf("Error in Naive+ decryption: allgather ML_RD (II) while %d tried to decrypt from %d to %d\n", rank, (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12), recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);
                                fflush(stdout);        
                            }
                        }// End for

                    }else{ // Not Naive+
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - (my_tree_root + mask)) * recvcount, recvtype);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + offset),
                                        (n -
                                        (my_tree_root +
                                        mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount)); 
                        curr_cnt += recently_received;


                        for(f=0; f<recently_received; ++f){
                            unsigned long count=0;
                            
                            in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent));
                            out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                            out = (char*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[out_index] * recvcount * recvtype_extent);

                            mpi_errno = MPIR_Localcopy(
                                in, recvcount, recvtype,
                                out, recvcount, recvtype);
                            if (mpi_errno) {
                                MPIR_ERR_POP(mpi_errno);
                            }
                        }// End for


                    }
                    
                }
                tmp_mask >>= 1;
                k--;
            }//end while tmp_mask
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    //printf("%d finished Inter-Node (ML-RD)\n", rank);

    /* Intra-Node RD*/
    

    mask = 0x1;
    i = 0;
    curr_cnt = 1;
    

    
    rank_index = comm_ptr->dev.ch.rank_list_index;
    int shared_mem_rank, shared_mem_dst; //I'm here
    shared_mem_rank = shmem_commptr->rank;

    while (mask <  p) {
        shared_mem_dst = shared_mem_rank ^ mask;
        dst_index = my_node * p + shared_mem_dst;

        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = shared_mem_dst >> i;
        dst_tree_root <<= i;

        my_tree_root = shared_mem_rank >> i;
        my_tree_root <<= i;

        if ( dst_index < (my_node + 1) * p && dst_index>= (my_node * p) ) {
            dst = comm_ptr->dev.ch.rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/


            /** #TODO: for loop to send all the msgs received in the inter-node step**/
            int previous_recv_count=0;
            for(f=0; f< n; ++f){
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = comm_ptr->dev.ch.rank_list[((my_tree_root + f * p) % comm_size)] * recvcount * recvtype_extent;
                recv_offset = comm_ptr->dev.ch.rank_list[((dst_tree_root + f * p) % comm_size)] * recvcount * recvtype_extent;

                // if(recvcount==16)
                //     printf("%d is going to send (Intra-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, comm_ptr->dev.ch.rank_list[(my_tree_root + f * p) % comm_size], dst, (p - dst_tree_root), comm_ptr->dev.ch.rank_list[(dst_tree_root + f * p) % comm_size]);
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*recvcount, recvtype); 
                MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (p - dst_tree_root) * recvcount, recvtype); 
                mpi_errno =
                    MPIC_Sendrecv_Plus(((char *) recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) recvbuf + recv_offset),
                                    (p -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);

                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }
            

                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                if(previous_recv_count==0){
                    previous_recv_count = recently_received;
                }else{
                    if(previous_recv_count != recently_received){
                        printf("ERROR in Multileader-RD (Intra-I)!\nReceived sizes are not equal!\n");
                    }
                }
                //printf("%d  - %d received (Intra-I) %d from %d and curr_cnt is now %d\n", recvcount, rank, recently_received, dst, curr_cnt);
            }//end for f
            curr_cnt += recently_received;
            
	    
            // if(recvcount==16)
            //     printf("%d received (Intra-I) %d and curr_cnt is now %d\n",rank, recently_received, curr_cnt);
        }

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */
    
        /* --BEGIN EXPERIMENTAL-- */
        


        if (dst_tree_root + mask > p) {
            nprocs_completed = p - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                shared_mem_dst = shared_mem_rank ^ tmp_mask;
                dst_index = my_node * p + shared_mem_dst;

                tree_root = shared_mem_rank >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->dev.ch.rank_list[dst_index];
                
                if ((shared_mem_dst > shared_mem_rank) && (shared_mem_rank < tree_root + nprocs_completed)
                    && (shared_mem_dst >= tree_root + nprocs_completed)) {

                    for(f=0; f<n; ++f){
                        sbuf = (char*)((char*) recvbuf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                    
                        //send
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, recently_received * recvcount, recvtype); 
                        // printf("%d  -   %d is going to send %d from %d to %d\n", recvcount, rank, recently_received, (my_tree_root + mask+ f*p)%comm_size, dst);
                        mpi_errno =
                            MPIC_Send_Plus(sbuf,
                                    recently_received * recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);

                        /*mpi_errno =
                            MPIC_Send(sbuf,
                                    recently_received * recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG, comm_ptr, errflag);*/

                        
                        if (mpi_errno) {
                            /* for communication errors, just record the error but
                            continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            
                        }
                    }//end for f
                            

                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((shared_mem_dst < shared_mem_rank) &&
                            (shared_mem_dst < tree_root + nprocs_completed) &&
                            (shared_mem_rank >= tree_root + nprocs_completed)) {
                    int previous_recv_count=0;
                    for(f=0; f<n; ++f){
                        rbuf = (char*)((char*) recvbuf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                        MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (p - (my_tree_root + mask)) * recvcount, recvtype);
                        // printf("%d  -   %d is going to recv %d at %d from %d\n", recvcount, rank, p - (my_tree_root + mask), (my_tree_root + mask+ f*p)%comm_size, dst);
                        mpi_errno =
                            MPIC_Recv(rbuf,
                                        (p - (my_tree_root + mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount));
                        if(previous_recv_count==0){
                            previous_recv_count = recently_received;
                        }else{
                            if(previous_recv_count != recently_received){
                                printf("ERROR in Multileader-RD-NB!\nReceived sizes are not equal!\n");
                            }
                        }
                        //printf("%d  - %d received (Intra-I) %d from %d and curr_cnt is now %d\n", recvcount, rank, recently_received, dst, curr_cnt);
                    }//end for f
                    curr_cnt += recently_received;
            

  
                }
                tmp_mask >>= 1;
                k--;

            }//end while tmp_mask

        }

        // /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    //printf("%d finished Intra-Node (ML-RD)\n", rank);

    for(i=0; i<send_req_idx; ++i){
        mpi_errno = MPIC_Wait((send_req_ptr[i]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[i]->status.MPI_ERROR;
        }
	 MPID_Request_release(send_req_ptr[i]);
    }

    fn_fail:
        MPIR_TIMER_END(coll,allgather,2lvl_multileader_rd);

	    return (mpi_errno);
}
/*****************************************************/




/**************** Added by Mehran *****************/
/**
 * 
 * In this function, first, we perform an inter-node 
 * ring allgather where all the ranks exchange 
 * their data with their peers on other nodes. 
 * Then, we perform inter-node allgather where all
 * the ranks at each node, exchange all the data
 * that they have through a ring.
 * 
 **/
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Multileader_Ring_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Multileader_Ring_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_multileader_ring);

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, j;
    
    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    
    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Multileader_Ring_MV2\n");
    int p = shmem_commptr->local_size; // number of ranks per node
    int n = (int) (size / p); // number of nodes
    
    /************* For MPIC_Sendrecv_Plus ************/
    MPID_Request *send_req_ptr[size-1];
    int s;
    for(s=0; s<size-1; ++s){
        (send_req_ptr[s])=NULL;
    }
    /*************************************************/

    /* get extent of receive type */
    MPI_Aint recvtype_extent, sendtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_multileader_ring, 1);

    /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcount, sendtype,
            rbuf,    recvcount, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* Inter-Node Ring*/
    
    MPID_Node_id_t node_id, left_node_id, right_node_id;    
    MPID_Get_node_id(comm_ptr, rank, &node_id);

    int rank_index = comm_ptr->dev.ch.rank_list_index, send_req_idx=0;

    //If naive+ encryption is selected, each rank needs to encrypt its local data first

    if(security_approach==2){
        
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) recvbuf + comm_ptr->dev.ch.rank_list[rank_index] * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + node_id * (recvcount * recvtype_extent + 12 + 16));

        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long in_size=0;
        in_size = (unsigned long)(sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        //printf("%d (%d) is going to encrypt from %d to %d\n", rank, local_rank, rank * recvcount * recvtype_extent, rank * (recvcount * recvtype_extent + 12 + 16) );
        if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, in_size,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather MultiLeader\n");
                fflush(stdout);
        }

    }

    /* lookup our index in the rank list */
    
    //    printf("%d is %d\n", rank, comm_ptr->dev.ch.rank_list[rank_index]);
    /* compute the left and right neighbor ranks in the rank_list */
    int left_index  = (size + rank_index - p) % size;
    int right_index = (size + rank_index + p) % size;
    int left  = comm_ptr->dev.ch.rank_list[left_index];
    int right = comm_ptr->dev.ch.rank_list[right_index];

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
    int send_index = rank_index;
    int recv_index = left_index;
    

    for (i=1; i < n; ++i){
	
        int send_rank = comm_ptr->dev.ch.rank_list[send_index];
        int recv_rank = comm_ptr->dev.ch.rank_list[recv_index];
        /* compute position within buffer to send from and receive into */
        //if(recvcount==16)
        //  	    printf("%d is going to send from %d to %d and receive from %d at %d\n", rank, send_rank, right, left, recv_rank);

        if(security_approach == 2){
            MPID_Get_node_id(comm_ptr, send_rank, &right_node_id);
            MPID_Get_node_id(comm_ptr, recv_rank, &left_node_id);

            void* sbuf = (void*)((char*) ciphertext_recvbuf + right_node_id * (recvcount * recvtype_extent + 16 + 12));
            void* rbuf = (void*)((char*) ciphertext_recvbuf + left_node_id * (recvcount * recvtype_extent + 16 + 12));
        
            /* exchange data with our neighbors in the ring */
            MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR); 
            MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR); 

            mpi_errno = MPIC_Sendrecv_Plus(
            sbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
            rbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
            comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
            );

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            // Decrypt recently received data
            unsigned long count=0;
            
            if(!EVP_AEAD_CTX_open(ctx, (recvbuf+recv_rank*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))), 12,
                        (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))+12), (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                printf("Error in Naive+ decryption: allgather MultiLeader\n");
                fflush(stdout);        
            }

        }else{
            void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
            void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
        
            /* exchange data with our neighbors in the ring */
            MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, recvcount, recvtype); 
            MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, recvcount, recvtype); 
            mpi_errno = MPIC_Sendrecv_Plus(
            sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
            rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
            comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
            );
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        /* update index values to account for data we just received */
        send_index = recv_index;
        recv_index = (size + recv_index - p) % size;
    }//End for
    
    //printf("%d finished inter-node\n", rank);
    /* Intra-node Ring */

    right_index = (rank_index + 1);
    if(right_index >= size){
	    right_index -= p;
    }else{
        MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[right_index], &right_node_id);
        if(right_node_id != node_id){
            right_index -= p;
        }
    }
    right = comm_ptr->dev.ch.rank_list[right_index];
    
    left_index = (rank_index - 1);
    if(left_index<0){
	    left_index += p;
    }else{
        MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[left_index], &left_node_id);
        if(left_node_id != node_id){
            (left_index += p);
        }
    }
    left = comm_ptr->dev.ch.rank_list[left_index];

    send_index = rank_index;
    recv_index = left_index;

    for (i=1; i < p; ++i){
        
        for(j=0; j<n; ++j){
            
	    //        send_index = (size + rank_index - j*p) % size;
            //recv_index = (size + left_index - j*p) % size;

        int send_rank = comm_ptr->dev.ch.rank_list[(size + send_index - j*p) % size];
        int recv_rank = comm_ptr->dev.ch.rank_list[(size + recv_index - j*p) % size];

	    //if(recvcount==16)
	    //	    printf("%d is going to send from %d to %d and receive from %d at %d\n", rank, send_rank, right, left, recv_rank);
        void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
        void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
    
        /* exchange data with our neighbors in the ring */
        MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, recvcount, recvtype); 
        MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, recvcount, recvtype); 
        
        
        mpi_errno = MPIC_Sendrecv_Plus(
        sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
        rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
        );

            
        
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }//end for j
	    send_index = recv_index;
	    recv_index = recv_index - 1;
	    if(recv_index<0){
		    recv_index += p;
	    }else{
            MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[recv_index], &left_node_id);
            if(left_node_id != node_id){
                (recv_index += p);
            }
	    }
    }//End for i

    //printf("%d finished Intra-Node\n", rank);
    for(i=0; i<send_req_idx; ++i){
        mpi_errno = MPIC_Wait((send_req_ptr[i]), errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr[i]->status.MPI_ERROR;
        }
	 MPID_Request_release(send_req_ptr[i]);
    }

    fn_fail:
        MPIR_TIMER_END(coll,allgather,2lvl_multileader_ring);

	    return (mpi_errno);
}
/*****************************************************/




/* Execute an allgather by forwarding data through a ring of
 * processes.  This implementation uses the two-level data
 * structures to account for how procs are assigned to nodes
 * to ensure data is only sent into and out of each node once. */
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Ring_nonblocked_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Ring_nonblocked_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_ring_nonblocked);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i;
    /***********«Added by Mehran************/
    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    int local_rank = shmem_commptr->rank;
    int local_size = shmem_commptr->local_size;
    MPI_Status status;
    /***************************************/
    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Ring_nonblocked_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent, sendtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_ring_nonblocked, 1);

    /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcount, sendtype,
            rbuf,    recvcount, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /***************** Added by Mehran *****************/
    int send_idx=0;
    if(security_approach==2 && local_rank==local_size-1){
        
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));

        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long in_size=0;
        in_size = (unsigned long)(sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        //printf("%d (%d) is going to encrypt from %d to %d\n", rank, local_rank, rank * recvcount * recvtype_extent, rank * (recvcount * recvtype_extent + 12 + 16) );
        if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, in_size,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather Ring_NB\n");
                fflush(stdout);
        }

    }
    /***************************************************/

    /* Now, send left to right. */

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int left_index  = (size + rank_index - 1) % size;
    int right_index = (size + rank_index + 1) % size;
    int left  = comm_ptr->dev.ch.rank_list[left_index];
    int right = comm_ptr->dev.ch.rank_list[right_index];

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
    int send_index = rank_index;
    int recv_index = left_index;
    
    /***************** Added by Mehran *****************/
    if(security_approach==2){
        unsigned long count=0;
        //printf("Naive+ Ring(NB)\n");
        for (i = 1; i < size; i++) {
            /* compute ranks whose data we'll send and receive in this step */
            int send_rank = comm_ptr->dev.ch.rank_list[send_index];
            int recv_rank = comm_ptr->dev.ch.rank_list[recv_index];
            if(local_rank==0){
                //receive at ciphertext rcvbuffer for extended size
                void* rbuf = (void*)((char*) ciphertext_recvbuf + (send_idx+1)%2 * (recvcount * recvtype_extent + 12 + 16));
                //void* rbuf = (void*)((char*) ciphertext_recvbuf + recv_rank * (recvcount * recvtype_extent + 12 + 16));

                if(local_rank==local_size-1){
                    //send from ciphertext rcvbuffer for extended size
                    void* sbuf = (void*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));

                    /* exchange data with our neighbors in the ring */
                    MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, send, (sendcount * sendtype_extent + 12 + 16), MPI_CHAR); 
                    MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, recv, (recvcount * recvtype_extent + 12 + 16), MPI_CHAR); 
                    //printf("%d (%d) is going to send %d to %d, and receive %d from %d\n", rank, local_rank, (sendcount * sendtype_extent + 12 + 16), right, (recvcount * recvtype_extent + 12 + 16), left);
                    mpi_errno = MPIC_Sendrecv(
                    sbuf, (sendcount * sendtype_extent + 12 + 16), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                    rbuf, (recvcount * recvtype_extent + 12 + 16), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
                    comm_ptr, MPI_STATUS_IGNORE, errflag
                    );
                    
                }else{
                //send from rcvbuffer for non-extended size
                    void* sbuf = (void*)((char*) recvbuf + send_rank * (recvcount * recvtype_extent));

                    /* exchange data with our neighbors in the ring */
                    MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, send, (recvcount * recvtype_extent), MPI_CHAR); 
                    MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, recv, (recvcount * recvtype_extent + 12 + 16), MPI_CHAR); 

                    //printf("%d (%d) is going to send %d to %d, and receive %d from %d\n", rank, local_rank, (sendcount * sendtype_extent), right, (recvcount * recvtype_extent + 12 + 16), left);
                    //printf("%d (%d) is sending from %d and receiving from %d\n", rank, local_rank, send_rank * (recvcount * recvtype_extent), recv_rank * (recvcount * recvtype_extent + 12 + 16));
                    mpi_errno = MPIC_Sendrecv(
                    sbuf, (recvcount * recvtype_extent), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                    rbuf, (recvcount * recvtype_extent + 12 + 16), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
                    comm_ptr, MPI_STATUS_IGNORE, errflag
                    );

                }

                //decrypt to local rcvbuffer
                //printf("%d (%d) is going to decrypt from %d to %d\n", rank, local_rank, (recv_rank*(sendcount*sendtype_extent+16+12)), recv_rank*recvcount*recvtype_extent);
                if(!EVP_AEAD_CTX_open(ctx, (recvbuf+recv_rank*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        (ciphertext_recvbuf+(((send_idx+1)%2)*(sendcount*sendtype_extent+16+12))), 12,
                        (ciphertext_recvbuf+(((send_idx+1)%2)*(sendcount*sendtype_extent+16+12))+12), (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                    printf("Error in Naive+ decryption: allgather Ring_NB\n");
                fflush(stdout);        
                }
                send_idx = (send_idx+1)%2;
            }
            else if(local_rank==local_size-1){
                //send from ciphertext recvbuffer for extended size
                void* sbuf = (void*)((char*) ciphertext_recvbuf + send_idx * (recvcount * recvtype_extent + 12 + 16));

                //recv at recvbuffer for non-extended size
                void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);

                /* exchange data with our neighbors in the ring */
                MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, send, (sendcount * sendtype_extent + 12 + 16), MPI_CHAR); 
                MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, recv, (recvcount * recvtype_extent), MPI_CHAR); 
                
                //printf("%d (%d) is going to send %d to %d, and receive %d from %d\n", rank, local_rank, (sendcount * sendtype_extent + 12 + 16), right, (recvcount * recvtype_extent), left);

                mpi_errno = MPIC_Sendrecv(
                sbuf, (recvcount * recvtype_extent + 12 + 16), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                rbuf, (recvcount * recvtype_extent), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag
                );

                //encrypt to ciphertext recvdbuf
                unsigned long  ciphertext_len = 0;

                void* in = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
                void* out = (void*)((char*) ciphertext_recvbuf + ((send_idx+1)%2) * (recvcount * recvtype_extent + 12 + 16));

                RAND_bytes(out, 12); // 12 bytes of nonce
                unsigned long in_size=0;
                in_size = (unsigned long)(recvcount * recvtype_extent);
                unsigned long max_out_len = (unsigned long) (16 + in_size);
                //printf("%d (%d) is going to encrypt from %d to %d\n", rank, local_rank, recv_rank * recvcount * recvtype_extent, recv_rank * (recvcount * recvtype_extent + 12 + 16) );
                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                    &ciphertext_len, max_out_len,
                                    out, 12,
                                    in, in_size,
                                    NULL, 0))
                {
                        printf("Error in Naive+ encryption: allgather Ring_NB\n");
                        fflush(stdout);
                }
                send_idx = (send_idx+1)%2;

            }else{
                /* compute position within buffer to send from and receive into */
                void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
                void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
            
                /* exchange data with our neighbors in the ring */
                MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, send, recvcount, recvtype); 
                MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, recv, recvcount, recvtype); 
                //printf("%d (%d) is going to send %d to %d, and receive %d from %d\n", rank, local_rank, (sendcount * sendtype_extent), right, (recvcount * recvtype_extent), left);
                mpi_errno = MPIC_Sendrecv(
                sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
                rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag
                );
            }

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* update index values to account for data we just received */
            send_index = recv_index;
            recv_index = (size + recv_index - 1) % size;

        }//End for
        //End of Naive+ Ring_NB
        /***************************************************/

    }else{
        for (i = 1; i < size; i++) {
            /* compute ranks whose data we'll send and receive in this step */
            int send_rank = comm_ptr->dev.ch.rank_list[send_index];
            int recv_rank = comm_ptr->dev.ch.rank_list[recv_index];
            /* compute position within buffer to send from and receive into */
            void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
            void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
        
            /* exchange data with our neighbors in the ring */
            MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, send, recvcount, recvtype); 
            MPIR_PVAR_INC(allgather, 2lvl_ring_nonblocked, recv, recvcount, recvtype); 
            mpi_errno = MPIC_Sendrecv(
            sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
            rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
            comm_ptr, MPI_STATUS_IGNORE, errflag
            );
        
        
        
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* update index values to account for data we just received */
            send_index = recv_index;
            recv_index = (size + recv_index - 1) % size;
        }//End for
    }//End Else (Original implementation)

  fn_fail:
    MPIR_TIMER_END(coll,allgather,2lvl_ring_nonblocked);
    return (mpi_errno);
}

/* In this implementation, we "gather" data from all procs on a node
 * to their leader.  This is done with direct send/recv and we write
 * the data directly to the receive buffer on the leader process.
 * The leaders then execute an "allgather" by directly sending each
 * of these messages. Finally, we broadcast the final receive buffer
 * to the procs on the node. */
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Direct_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Direct_MV2(
    const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
          void *recvbuf, int recvcnt, MPI_Datatype recvtype,
    MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_direct);
    int i, j;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_direct, 1);

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    MPIU_CHKLMEM_DECL(2);

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Direct_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    int local_rank = shmem_commptr->rank;
    int local_size = shmem_commptr->local_size;

    /* get info about communicator across node leaders, allocate temp buffer */
    MPID_Comm* leader_commptr = NULL;
    int leader_rank = -1;
    int leader_size = 0;
    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        MPI_Comm leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_rank = leader_commptr->rank;
        leader_size = leader_commptr->local_size;
    }

    int gather_msgs    = 0;
    int allgather_msgs = 0;

    /* request/status object allocation is different for 
     * leader vs. non-leader */
    if (local_rank == 0) {
        gather_msgs    = local_size - 1; 
        allgather_msgs = (local_size * (leader_size - 1)) + (size - local_size);
    } else {
        /* if non-leader only send one msg in gather step */
        gather_msgs = 1;
    }

    /* now get the max of these two because the gather
     * completes before the allgather to save space */ 
    int max = allgather_msgs; 
    if (gather_msgs > max) {
        max = gather_msgs;
    }

    /* allocate memory for request objects */
    MPID_Request **reqarray = NULL;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        max * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    
    /* allocate memory for status objects */
    MPI_Status *starray = NULL;
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        max * sizeof (MPI_Status),
                        mpi_errno, "starray");

    /****************************
     * Gather data to leaders using direct send/recv
     ****************************/

    /* track number of requests */
    int reqs = 0;

    /* gather data to leaders on each node */
    int rank_index = comm_ptr->dev.ch.rank_list_index;
    if (local_rank == 0) {
        /* post receives & point i at leader rank in the rank_list */
        for (i = 1; i < local_size; i++) {
            /* get global rank of incoming data */
            int dstrank = comm_ptr->dev.ch.rank_list[rank_index + i];

            /* compute pointer in receive buffer for incoming data from this rank */
            void* rbuf = (void*)((char*) recvbuf + dstrank * recvcnt * recvtype_extent);

            /* post receive for data from this rank */
            MPIR_PVAR_INC(allgather, 2lvl_direct, recv, recvcnt, recvtype);
            mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
                i, MPIR_ALLGATHER_TAG, shmem_commptr, &reqarray[reqs++]
            );

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
       }

       /* copy our data to our receive buffer if needed */
       if (sendbuf != MPI_IN_PLACE) {
           /* compute location in receive buffer for our data */
           void* rbuf = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
           mpi_errno = MPIR_Localcopy(
               sendbuf, sendcnt, sendtype,
               rbuf, recvcnt, recvtype
           );
           if (mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
           }
       }
    } else {
        /* get parameters for sending data */
        const void* sbuf   = sendbuf;
        int scnt           = sendcnt;
        MPI_Datatype stype = sendtype;
        if (sendbuf == MPI_IN_PLACE) {
            /* use receive params if IN_PLACE */
            sbuf   = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
            scnt   = recvcnt;
            stype  = recvtype;
        }

        /* send data to the leader process */
        MPIR_PVAR_INC(allgather, 2lvl_direct, send, scnt, stype); 
        mpi_errno = MPIC_Isend(sbuf, scnt, stype,
           0, MPIR_ALLGATHER_TAG, shmem_commptr, &reqarray[reqs++], errflag
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
   }

    /* wait for all outstanding requests to complete */
    mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
    if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
           MPIR_ERR_POP(mpi_errno);
    }

    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno == MPI_ERR_IN_STATUS) {
        for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                mpi_errno = starray[i].MPI_ERROR;
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
    }

    /****************************
     * Execute direct-send allgather across leaders
     ****************************/

    /*If there is just one node, after gather itself,
     * root has all the data and it can do bcast*/

    /* Exchange the data between the node leaders */
    if (local_rank == 0) {
        /* initialize the active request counter */
        reqs = 0;

        /* post receives */
        for (i = 1; i < leader_size; i++) {
            /* compute source rank sending to us in this step */
            int src = leader_rank - i;
            if (src < 0) {
                src += leader_size;
            }

            /* receive all messages from each rank on src node */
            int recv_count = comm_ptr->dev.ch.node_sizes[src];
            int recv_displ = comm_ptr->dev.ch.node_disps[src];
            for (j = 0; j < recv_count; j++) {
                /* get rank of incoming data */
                int srcrank = comm_ptr->dev.ch.rank_list[recv_displ + j];

                /* get pointer to receive buffer for this rank */
                void* rbuf = (void*)((char*) recvbuf + srcrank * recvcnt * recvtype_extent);

                /* post receive */
                MPIR_PVAR_INC(allgather, 2lvl_direct, recv, recvcnt, recvtype);
                mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
                    src, MPIR_ALLGATHER_TAG, leader_commptr, &reqarray[reqs++]
                );
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }

        /* post sends */
        for (i = 1; i < leader_size; i++) {
            /* compute destination rank for this step */
            int dst = leader_rank + i;
            if (dst >= leader_size) {
                dst -= leader_size;
            }

            /* send all messages for this leader to each other leader rank */
            int send_count = comm_ptr->dev.ch.node_sizes[leader_rank];
            for (j = 0; j < send_count; j++) {
                /* get source rank for data we'll send */
                int dstrank = comm_ptr->dev.ch.rank_list[rank_index + j];

                /* get pointer into buffer for this rank */
                void* sbuf = (void*)((char*) recvbuf + dstrank * recvcnt * recvtype_extent);

                /* post send to this destination rank */
                MPIR_PVAR_INC(allgather, 2lvl_direct, send, sendcnt, sendtype); 
                mpi_errno = MPIC_Isend(sbuf, sendcnt, sendtype,
                    dst, MPIR_ALLGATHER_TAG, leader_commptr, &reqarray[reqs++], errflag
                );

                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }

        /* wait for all outstanding requests to complete */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
           MPIR_ERR_POP(mpi_errno);
        }

        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = starray[i].MPI_ERROR;
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
    }

    /****************************
     * Broadcast data from leaders to procs on the node
     ****************************/

    /* Bcast the entire data from node leaders to other ranks on node */
    mpi_errno = MPIR_Bcast_impl(recvbuf, recvcnt * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();

  fn_fail:
    MPIR_TIMER_END(coll,allgather,2lvl_direct);
    return (mpi_errno);
}

/* In this implementation, we "gather" data from all procs on a node
 * to their leader.  This is done with direct send/recv and we write
 * the data directly to the receive buffer on the leader process.
 * The leaders then execute an "allgather" by directly sending each
 * of these messages.  We use a ring algorithm to forward data through
 * leaders.  Finally, we broadcast the final receive buffer to the
 * procs on the node. */
#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Ring_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Ring_MV2(
    const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
          void *recvbuf, int recvcnt, MPI_Datatype recvtype,
    MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,2lvl_ring);
    int i, j;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgather_2lvl_ring, 1);

    if (recvcnt == 0) {
        return MPI_SUCCESS;
    }

    MPIU_CHKLMEM_DECL(2);

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    if(rank==0)
        printf("MPIR_2lvl_Allgather_Ring_MV2\n");
    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* get info about communicator for ranks on the same node */
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    int local_rank = shmem_commptr->rank;
    int local_size = shmem_commptr->local_size;

    /* get info about communicator across node leaders */
    MPID_Comm* leader_commptr = NULL;
    int leader_rank = -1;
    int leader_size = 0;
    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        MPI_Comm leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_rank = leader_commptr->rank;
        leader_size = leader_commptr->local_size;
    }

    int gather_msgs    = 0;
    int allgather_msgs = 0;
    int max_local_size = 0;
    int *node_sizes    = comm_ptr->dev.ch.node_sizes;

    /* request/status object allocation is different for 
     * leader vs. non-leader */
    if (local_rank == 0) {
        gather_msgs = local_size - 1; 
        for (i = 0; i < leader_size; i++) {
            if (node_sizes[i] > max_local_size) {
                max_local_size = node_sizes[i];
            } 
        }
        allgather_msgs = 2 * max_local_size;
    } else {
        /* if non-leader only send one msg in gather step */
        gather_msgs = 1;
    }

    /* now get the max of these two because the gather
     * completes before the allgather to save space */ 
    int max = allgather_msgs; 
    if (gather_msgs > max) {
        max = gather_msgs;
    }

    /* allocate array of request objects */
    MPID_Request **reqarray = NULL;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        max * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    /* allocate array of status objects */
    MPI_Status *starray = NULL;
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        max * sizeof (MPI_Status),
                        mpi_errno, "starray");

    /****************************
     * Gather data to leaders using direct send/recv
     ****************************/

    int reqs = 0;

    /* gather data to leaders on each node */
    int rank_index = comm_ptr->dev.ch.rank_list_index;
    if (local_rank == 0) {
        /* post receives for incoming data from procs on our node */
        for (i = 1; i < local_size; i++) {
            /* get global rank of this process */
            int srcrank = comm_ptr->dev.ch.rank_list[rank_index + i];

            /* compute pointer in receive buffer for incoming data from this rank */
            void* rbuf = (void*)((char*) recvbuf + srcrank * recvcnt * recvtype_extent);

            /* post receive for data from this rank on shared mem comm */
            MPIR_PVAR_INC(allgather, 2lvl_ring, recv, recvcnt, recvtype);
            mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
                i, MPIR_ALLGATHER_TAG, shmem_commptr, &reqarray[reqs++]
            );
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
       }

       /* copy our data to our receive buffer if needed */
       if (sendbuf != MPI_IN_PLACE) {
           /* compute location in receive buffer for our data */
           void* rbuf = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
           mpi_errno = MPIR_Localcopy(
               sendbuf, sendcnt, sendtype,
               rbuf, recvcnt, recvtype
           );
           if (mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
           }
       }
    } else {
        /* get parameters for sending data */
        const void* sbuf   = sendbuf;
        int scnt           = sendcnt;
        MPI_Datatype stype = sendtype;
        if (sendbuf == MPI_IN_PLACE) {
            /* use receive params if IN_PLACE */
            sbuf   = (void*)((char*) recvbuf + rank * recvcnt * recvtype_extent);
            scnt   = recvcnt;
            stype  = recvtype;
        }

        /* send data to leader of our node */
        MPIR_PVAR_INC(allgather, 2lvl_ring, send, scnt, stype); 
        mpi_errno = MPIC_Isend(sbuf, scnt, stype,
           0, MPIR_ALLGATHER_TAG, shmem_commptr, &reqarray[reqs++], errflag
        );
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
   }

   /* wait for all outstanding requests to complete */
   mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
   if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
          MPIR_ERR_POP(mpi_errno);
   }

   /* --BEGIN ERROR HANDLING-- */
   if (mpi_errno == MPI_ERR_IN_STATUS) {
       for (i = 0; i < reqs; i++) {
           if (starray[i].MPI_ERROR != MPI_SUCCESS) {
               mpi_errno = starray[i].MPI_ERROR;
               if (mpi_errno) {
                   /* for communication errors, just record the error but continue */
                   *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                   MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                   MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
               }
           }
       }
   }

   if (mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
   }

    /****************************
     * Execute ring-based allgather across leaders
     ****************************/

    /* Exchange the data between the node leaders */
    if (local_rank == 0 && leader_size > 1) {
        /* get our left and right ranks in our leader comm */
        int left  = (leader_size + leader_rank - 1) % leader_size;
        int right = (leader_size + leader_rank + 1) % leader_size;

        /* start by sending our own data and receiving data
         * from the leader to our left */
        int send_index = leader_rank;
        int recv_index = left;
        for (i = 0; i < leader_size; i++) {
            /* initalize our request counter */
            reqs = 0;

            /* post receives for data coming from the left */
            int recv_count = comm_ptr->dev.ch.node_sizes[recv_index];
            int recv_displ = comm_ptr->dev.ch.node_disps[recv_index];
            for (j = 0; j < recv_count; j++) {
                /* get source rank for this message */
                int srcrank = comm_ptr->dev.ch.rank_list[recv_displ + j];

                /* compute pointer in receive buffer for incoming data from this rank */
                void* rbuf = (void*)((char*) recvbuf + srcrank * recvcnt * recvtype_extent);

                /* post receive for data from this rank */
                MPIR_PVAR_INC(allgather, 2lvl_ring, recv, recvcnt, recvtype);
                mpi_errno = MPIC_Irecv(rbuf, recvcnt, recvtype,
                    left, MPIR_ALLGATHER_TAG, leader_commptr, &reqarray[reqs++]
                );
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

            /* TODO: consider placing a barrier here to ensure
             * receives are posted before sends, especially for large messages */
            //MPIR_Barrier_impl(comm_ptr);

            /* post sends for data we're sending to the right */
            int send_count = comm_ptr->dev.ch.node_sizes[send_index];
            int send_displ = comm_ptr->dev.ch.node_disps[send_index];
            for (j = 0; j < send_count; j++) {
                /* get source rank for this message */
                int dstrank = comm_ptr->dev.ch.rank_list[send_displ + j];

                /* compute pointer in receive buffer for outgoing data from this rank */
                void* sbuf = (void*)((char*) recvbuf + dstrank * recvcnt * recvtype_extent);

                /* post send for data from this rank */
                MPIR_PVAR_INC(allgather, 2lvl_ring, send, recvcnt, recvtype); 
                mpi_errno = MPIC_Isend(sbuf, recvcnt, recvtype,
                    right, MPIR_ALLGATHER_TAG, leader_commptr, &reqarray[reqs++], errflag
                );
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

            /* wait for all outstanding requests to complete */
            mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
            if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
                   MPIR_ERR_POP(mpi_errno);
            }

            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno == MPI_ERR_IN_STATUS) {
                for (i = 0; i < reqs; i++) {
                    if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                        mpi_errno = starray[i].MPI_ERROR;
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                    }
                }
            }

            /* update index values so in the next step, we send the data
             * that we just received */
            send_index = recv_index;
            recv_index = (leader_size + recv_index - 1) % leader_size;
        }
    }

    /****************************
     * Broadcast data from leaders to procs on the node
     ****************************/

    /* Bcast the entire data from node leaders to other ranks on node */
    mpi_errno = MPIR_Bcast_impl(recvbuf, recvcnt * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();

  fn_fail:
    MPIR_TIMER_END(coll,allgather,2lvl_ring);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_index_tuned_intra_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{

    int mpi_errno = MPI_SUCCESS;
    int nbytes = 0, comm_size, recvtype_size;
    int comm_size_index = 0;
    int inter_node_algo_index = 0;
    int local_size = 0;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int table_min_comm_size = 0;
    int table_max_comm_size = 0;
    int table_min_inter_size = 0;
    int table_max_inter_size = 0;
    int last_inter;
    int lp2ltn; // largest power of 2 less than n
    int lp2ltn_min;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr=NULL;

    /* Get the size of the communicator */
    comm_size = comm_ptr->local_size;

    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = recvtype_size * recvcount;

    int i, rank;
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
#ifdef _ENABLE_CUDA_
    int send_mem_type = 0;
    int recv_mem_type = 0;
    int snbytes = INT_MAX;
    MPI_Aint sendtype_extent;
    if (rdma_enable_cuda) {
        send_mem_type = is_device_buffer(sendbuf);
        recv_mem_type = is_device_buffer(recvbuf);
    }

    /*Handling Non-contig datatypes */
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type)) {
        cuda_coll_pack((void **)&sendbuf, &sendcount, &sendtype,
                       &recvbuf, &recvcount, &recvtype,
                       rank * recvcount * recvtype_extent, 1, comm_size);
    }

    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    if (sendbuf != MPI_IN_PLACE) {
        snbytes = sendtype_extent * sendcount;
    }
    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = recvtype_size * recvcount;

    if (rdma_enable_cuda && rdma_cuda_allgather_fgp &&
        send_mem_type && recv_mem_type &&
        snbytes >
        rdma_cuda_allgather_naive_limit / (FGP_SWITCH_FACTOR * comm_size) &&
        nbytes > rdma_cuda_allgather_naive_limit / (FGP_SWITCH_FACTOR * comm_size)) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIR_Allgather_cuda_intra_MV2(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcount, recvtype,
                                              comm_ptr, errflag);
        } else {
            mpi_errno =
                MPIR_Allgather_cuda_intra_MV2(recvbuf +
                                              rank * recvcount *
                                              recvtype_extent, recvcount,
                                              recvtype, recvbuf, recvcount,
                                              recvtype, comm_ptr, errflag);
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        goto fn_exit;
    } else if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
               rdma_cuda_use_naive && (nbytes <= rdma_cuda_allgather_naive_limit)) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = cuda_stage_alloc((void **)&sendbuf, sendcount * sendtype_extent,
                                         &recvbuf,
                                         recvcount * recvtype_extent *
                                         comm_size, send_mem_type, recv_mem_type, 0);
        } else {
            mpi_errno = cuda_stage_alloc((void **)&sendbuf, recvcount * recvtype_extent,
                                         &recvbuf,
                                         recvcount * recvtype_extent *
                                         comm_size, send_mem_type,
                                         recv_mem_type,
                                         rank * recvcount * recvtype_extent);
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */

    if (mv2_use_old_allgather == 1) {
	MPIR_Allgather_intra_MV2(sendbuf, sendcount, sendtype, recvbuf, recvcount,
				 recvtype, comm_ptr, errflag);
	goto fn_exit;
    }
    
    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        i = 0;
        if (mv2_allgather_indexed_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }
        if ((comm_ptr->dev.ch.allgather_comm_ok != 0) &&
                (comm_ptr->dev.ch.is_blocked == 0 &&
                (mv2_allgather_cyclic_algo_threshold <= nbytes ||
                 mv2_allgather_ring_algo_threshold <= nbytes))) {
                /* for large messages or nonblocked hostfiles, use ring-allgather algorithm. */
                mpi_errno = MPIR_2lvl_Allgather_Ring_nonblocked_MV2(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_ptr, errflag);
                goto fn_cuda_exit;
        }

        FIND_PPN_INDEX  (allgather, local_size,conf_index, partial_sub_ok)


    }

    if (partial_sub_ok != 1) {
        conf_index = mv2_allgather_indexed_num_ppn_conf/2;
    }
        
conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_allgather_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_allgather_indexed_thresholds_table[conf_index][mv2_size_allgather_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
	/* Comm size smaller than smallest configuration in table: use smallest available */
	comm_size_index = 0;
    }
    else if (comm_size > table_max_comm_size) {
	/* Comm size larger than largest configuration in table: use largest available */
	comm_size_index = mv2_size_allgather_indexed_tuning_table[conf_index] - 1;
    }
    else {
	/* Comm size in between smallest and largest configuration: find closest match */
    lp2ltn_min = pow(2, (int)log2(table_min_comm_size));
	if (comm_ptr->dev.ch.is_pof2) {
	    comm_size_index = log2( comm_size / lp2ltn_min );
	}
	else {
	    lp2ltn = pow(2, (int)log2(comm_size));
        comm_size_index = (lp2ltn < lp2ltn_min) ? 0 : log2( lp2ltn / lp2ltn_min );
	}
    }

    last_inter = mv2_allgather_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
    table_min_inter_size = mv2_allgather_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
    table_max_inter_size = mv2_allgather_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[last_inter].msg_sz;
    
    if (nbytes < table_min_inter_size) {
	/* Msg size smaller than smallest configuration in table: use smallest available */
	inter_node_algo_index = 0;
    }
    else if (nbytes > table_max_inter_size) {
	/* Msg size larger than largest configuration in table: use largest available */
	inter_node_algo_index = last_inter;
    }
    else {
	/* Msg size in between smallest and largest configuration: find closest match */
	if (pow(2, (int)log2(nbytes)) == nbytes) {
	    inter_node_algo_index = log2( nbytes / table_min_inter_size );
	}
	else {
	    lp2ltn = pow(2, (int)log2(nbytes));
	    inter_node_algo_index = (lp2ltn < table_min_inter_size) ? 0 : log2( lp2ltn / table_min_inter_size );
	}
    }

    /* Set inter-leader pt */
    MV2_Allgather_function =
                          mv2_allgather_indexed_thresholds_table[conf_index][comm_size_index].
	inter_leader[inter_node_algo_index].MV2_pt_Allgather_function;

    if(MV2_Allgather_function == &MPIR_Allgather_RD_Allgather_Comm_MV2) {
        if(comm_ptr->dev.ch.allgather_comm_ok == 1) {
            int sendtype_iscontig = 0, recvtype_iscontig = 0;
            void *tmp_recv_buf = NULL;
            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
            if (sendtype != MPI_DATATYPE_NULL && recvtype != MPI_DATATYPE_NULL) {
                MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
                MPIR_Datatype_iscontig(recvtype, &recvtype_iscontig);
            }

            MPID_Comm *allgather_comm_ptr;
            MPID_Comm_get_ptr(comm_ptr->dev.ch.allgather_comm, allgather_comm_ptr);

            /*creation of a temporary recvbuf */
            tmp_recv_buf = MPIU_Malloc(recvcount * comm_size * recvtype_extent);
            if (!tmp_recv_buf) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**nomem", 0);
                return mpi_errno;
            }
            /* Calling Allgather with temporary buffer and allgather communicator */
            if (sendbuf != MPI_IN_PLACE) {
                mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype,
                                                     tmp_recv_buf, recvcount,
                                                     recvtype, allgather_comm_ptr, errflag);
            } else {
                mpi_errno = MPIR_Allgather_RD_MV2(recvbuf + rank * recvcount *
                                                     recvtype_extent, recvcount,
                                                     recvtype, tmp_recv_buf,
                                                     recvcount, recvtype,
                                                     allgather_comm_ptr, errflag);
            }

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            /* Reordering data into recvbuf */
            if (sendtype_iscontig == 1 && recvtype_iscontig == 1
#if defined(_ENABLE_CUDA_)
                && rdma_enable_cuda == 0
#endif
            ){
                for (i = 0; i < comm_size; i++) {
                    MPIUI_Memcpy((void *) ((char *) recvbuf +
                                           (comm_ptr->dev.ch.allgather_new_ranks[i]) *
                                           nbytes),
                                           (char *) tmp_recv_buf + i * nbytes, nbytes);
                }
            } else {
                for (i = 0; i < comm_size; i++) {
                    mpi_errno = MPIR_Localcopy((void *) ((char *) tmp_recv_buf +
                                                i * recvcount *
                                                recvtype_extent),
                                                recvcount, recvtype,
                                                (void *) ((char *) recvbuf +
                                                (comm_ptr->dev.ch.allgather_new_ranks[i])
                                                * recvcount * recvtype_extent),
                                           recvcount, recvtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
            }
            MPIU_Free(tmp_recv_buf);
        } else {
            mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype,
                                                recvbuf, recvcount, recvtype,
                                                comm_ptr, errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        } 
    } else if(MV2_Allgather_function == &MPIR_Allgather_Bruck_MV2 
            || MV2_Allgather_function == &MPIR_Allgather_RD_MV2
            || MV2_Allgather_function == &MPIR_Allgather_Ring_MV2
            || MV2_Allgather_function == &MPIR_Allgather_Direct_MV2
            || MV2_Allgather_function == &MPIR_Allgather_DirectSpread_MV2) {
            mpi_errno = MV2_Allgather_function(sendbuf, sendcount, sendtype,
                                          recvbuf, recvcount, recvtype,
                                          comm_ptr, errflag);
    } else if ((comm_ptr->dev.ch.shmem_coll_ok == 1 &&
                comm_ptr->dev.ch.rank_list != NULL) &&
                sendbuf != MPI_IN_PLACE &&
            (MV2_Allgather_function == &MPIR_Allgather_gather_bcast_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_nonblocked_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_Ring_nonblocked_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_Multileader_Ring_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_Multileader_RD_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_Direct_MV2
            || MV2_Allgather_function == &MPIR_2lvl_Allgather_Ring_MV2)) {
	/***** Added by Mehran *****/
    if(MV2_Allgather_function == &MPIR_2lvl_Allgather_Multileader_Ring_MV2 && comm_ptr->dev.ch.equal_local_sizes!=1){
        MV2_Allgather_function = &MPIR_2lvl_Allgather_Ring_nonblocked_MV2;
    }
	if(MV2_Allgather_function == &MPIR_2lvl_Allgather_Multileader_RD_MV2){
        if(comm_ptr->dev.ch.equal_local_sizes==1){
            if(comm_ptr->dev.ch.is_global_block!=1){
                MV2_Allgather_function = &MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2;
            }
        }else{
            MV2_Allgather_function = &MPIR_2lvl_Allgather_Ring_nonblocked_MV2;
        } 
    } 
    /***************************/
    mpi_errno = MV2_Allgather_function(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcount, recvtype,
                                    comm_ptr, errflag);
    } else {
        mpi_errno = MPIR_Allgather_intra(sendbuf, sendcount, sendtype,
                                         recvbuf, recvcount, recvtype, comm_ptr, errflag);
    }

fn_cuda_exit:
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && ((send_mem_type == 1) || (recv_mem_type == 1)) &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_allgather_naive_limit)) {
        cuda_stage_free((void **)&sendbuf,
                        &recvbuf, recvcount * recvtype_extent * comm_size,
                        send_mem_type, recv_mem_type);
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
#ifdef _ENABLE_CUDA_
    /*Handling Non-Contig datatypes */
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type)) {
        cuda_coll_unpack(&recvcount, comm_size);
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    
    int mpi_errno = MPI_SUCCESS;
    int nbytes = 0, comm_size, recvtype_size;
    int range = 0;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int range_threshold = 0;
    int is_two_level = 0;
    int local_size = -1;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr=NULL;
    
    if (mv2_use_indexed_tuning || mv2_use_indexed_allgather_tuning) {
        if(comm_ptr->rank==0)
            printf("Flag 0, %d, %d \n", sendcount, recvcount);
	    return MPIR_Allgather_index_tuned_intra_MV2(sendbuf, sendcount,
                            sendtype, recvbuf, recvcount,
				            recvtype, comm_ptr, errflag);
    }

    /* Get the size of the communicator */
    comm_size = comm_ptr->local_size;

    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = recvtype_size * recvcount;

    int i, rank;
    MPI_Aint recvtype_extent;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
#ifdef _ENABLE_CUDA_
    int send_mem_type = 0;
    int recv_mem_type = 0;
    int snbytes = INT_MAX;
    MPI_Aint sendtype_extent;
    if (rdma_enable_cuda) {
        send_mem_type = is_device_buffer(sendbuf);
        recv_mem_type = is_device_buffer(recvbuf);
    }

    /*Handling Non-contig datatypes */
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type)) {
        cuda_coll_pack((void **)&sendbuf, &sendcount, &sendtype,
                       &recvbuf, &recvcount, &recvtype,
                       rank * recvcount * recvtype_extent, 1, comm_size);
    }

    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    if (sendbuf != MPI_IN_PLACE) {
        snbytes = sendtype_extent * sendcount;
    }
    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = recvtype_size * recvcount;

    if (rdma_enable_cuda && rdma_cuda_allgather_fgp &&
        send_mem_type && recv_mem_type &&
        snbytes >
        rdma_cuda_allgather_naive_limit / (FGP_SWITCH_FACTOR * comm_size) &&
        nbytes > rdma_cuda_allgather_naive_limit / (FGP_SWITCH_FACTOR * comm_size)) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIR_Allgather_cuda_intra_MV2(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcount, recvtype,
                                              comm_ptr, errflag);
        } else {
            mpi_errno =
                MPIR_Allgather_cuda_intra_MV2(recvbuf +
                                              rank * recvcount *
                                              recvtype_extent, recvcount,
                                              recvtype, recvbuf, recvcount,
                                              recvtype, comm_ptr, errflag);
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        goto fn_exit;
    } else if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
               rdma_cuda_use_naive && (nbytes <= rdma_cuda_allgather_naive_limit)) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = cuda_stage_alloc((void **)&sendbuf, sendcount * sendtype_extent,
                                         &recvbuf,
                                         recvcount * recvtype_extent *
                                         comm_size, send_mem_type, recv_mem_type, 0);
        } else {
            mpi_errno = cuda_stage_alloc((void **)&sendbuf, recvcount * recvtype_extent,
                                         &recvbuf,
                                         recvcount * recvtype_extent *
                                         comm_size, send_mem_type,
                                         recv_mem_type,
                                         rank * recvcount * recvtype_extent);
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */

    if (mv2_use_old_allgather == 1) {
        if(comm_ptr->rank==0)
            printf("Flag 1, %d, %d \n", sendcount, recvcount);
	MPIR_Allgather_intra_MV2(sendbuf, sendcount, sendtype, recvbuf, recvcount,
				 recvtype, comm_ptr, errflag);
	goto fn_exit;
    }
    
    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        i = 0;
        if (mv2_allgather_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }
        

        FIND_PPN_INDEX  (allgather, local_size,conf_index, partial_sub_ok)

        
    }

    if (partial_sub_ok != 1) {
        conf_index = mv2_allgather_num_ppn_conf/2;
    }

conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_allgather_tuning_table[conf_index] - 1)) &&
           (comm_size >
            mv2_allgather_thresholds_table[conf_index][range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold <
         (mv2_allgather_thresholds_table[conf_index][range].size_inter_table - 1))
           && (nbytes > mv2_allgather_thresholds_table[conf_index][range].inter_leader[range_threshold].max)
           && (mv2_allgather_thresholds_table[conf_index][range].inter_leader[range_threshold].max !=
               -1)) {
        range_threshold++;
    }

    /* Set inter-leader pt */
    MV2_Allgather_function =
                          mv2_allgather_thresholds_table[conf_index][range].inter_leader[range_threshold].
                          MV2_pt_Allgather_function;

    is_two_level =  mv2_allgather_thresholds_table[conf_index][range].two_level[range_threshold];

    /* intracommunicator */
    if(is_two_level ==1){
        if(comm_ptr->rank==0)
            printf("Flag 2, %d, %d \n", sendcount, recvcount);
        
        if(comm_ptr->dev.ch.shmem_coll_ok == 1){
            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
	   if (1 == comm_ptr->dev.ch.is_blocked) {
                mpi_errno = MPIR_2lvl_Allgather_MV2(sendbuf, sendcount, sendtype,
						    recvbuf, recvcount, recvtype,
						    comm_ptr, errflag);
	   }
	   else {
	       mpi_errno = MPIR_Allgather_intra(sendbuf, sendcount, sendtype,
						recvbuf, recvcount, recvtype,
						comm_ptr, errflag);
	   }
        } else {
            mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype,
                                                recvbuf, recvcount, recvtype,
                                                comm_ptr, errflag);
        }
    } else if(MV2_Allgather_function == &MPIR_Allgather_RD_Allgather_Comm_MV2){
        if(comm_ptr->rank==0)
            printf("Flag 3, %d, %d \n", sendcount, recvcount);
        if(comm_ptr->dev.ch.allgather_comm_ok == 1) {
            int sendtype_iscontig = 0, recvtype_iscontig = 0;
            void *tmp_recv_buf = NULL;
            MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
            if (sendtype != MPI_DATATYPE_NULL && recvtype != MPI_DATATYPE_NULL) {
                MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
                MPIR_Datatype_iscontig(recvtype, &recvtype_iscontig);
            }

            MPID_Comm *allgather_comm_ptr;
            MPID_Comm_get_ptr(comm_ptr->dev.ch.allgather_comm, allgather_comm_ptr);

            /*creation of a temporary recvbuf */
            tmp_recv_buf = MPIU_Malloc(recvcount * comm_size * recvtype_extent);
            if (!tmp_recv_buf) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**nomem", 0);
                return mpi_errno;
            }
            /* Calling Allgather with temporary buffer and allgather communicator */
            if (sendbuf != MPI_IN_PLACE) {
                mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype,
                                                     tmp_recv_buf, recvcount,
                                                     recvtype, allgather_comm_ptr, errflag);
            } else {
                mpi_errno = MPIR_Allgather_RD_MV2(recvbuf + rank * recvcount *
                                                     recvtype_extent, recvcount,
                                                     recvtype, tmp_recv_buf,
                                                     recvcount, recvtype,
                                                     allgather_comm_ptr, errflag);
            }

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            /* Reordering data into recvbuf */
            if (sendtype_iscontig == 1 && recvtype_iscontig == 1
#if defined(_ENABLE_CUDA_)
                && rdma_enable_cuda == 0
#endif
            ){
                for (i = 0; i < comm_size; i++) {
                    MPIUI_Memcpy((void *) ((char *) recvbuf +
                                           (comm_ptr->dev.ch.allgather_new_ranks[i]) *
                                           nbytes),
                                           (char *) tmp_recv_buf + i * nbytes, nbytes);
                }
            } else {
                for (i = 0; i < comm_size; i++) {
                    mpi_errno = MPIR_Localcopy((void *) ((char *) tmp_recv_buf +
                                                i * recvcount *
                                                recvtype_extent),
                                                recvcount, recvtype,
                                                (void *) ((char *) recvbuf +
                                                (comm_ptr->dev.ch.allgather_new_ranks[i])
                                                * recvcount * recvtype_extent),
                                           recvcount, recvtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
            }
            MPIU_Free(tmp_recv_buf);
        } else {
            mpi_errno = MPIR_Allgather_RD_MV2(sendbuf, sendcount, sendtype,
                                                recvbuf, recvcount, recvtype,
                                                comm_ptr, errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        } 
    } else if(MV2_Allgather_function == &MPIR_Allgather_Bruck_MV2 
            || MV2_Allgather_function == &MPIR_Allgather_RD_MV2
            || MV2_Allgather_function == &MPIR_Allgather_Ring_MV2) {
                if(comm_ptr->rank==0)
                    printf("Flag 4, %d, %d \n", sendcount, recvcount);
            mpi_errno = MV2_Allgather_function(sendbuf, sendcount, sendtype,
                                          recvbuf, recvcount, recvtype,
                                          comm_ptr, errflag);
    } else {
        if(comm_ptr->rank==0)
            printf("Flag 5, %d, %d \n", sendcount, recvcount);
        mpi_errno = MPIR_Allgather_intra(sendbuf, sendcount, sendtype,
                                         recvbuf, recvcount, recvtype, comm_ptr, errflag);
    }

#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_allgather_naive_limit)) {
        cuda_stage_free((void **)&sendbuf,
                        &recvbuf, recvcount * recvtype_extent * comm_size,
                        send_mem_type, recv_mem_type);
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
#ifdef _ENABLE_CUDA_
    /*Handling Non-Contig datatypes */
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type)) {
        cuda_coll_unpack(&recvcount, comm_size);
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
