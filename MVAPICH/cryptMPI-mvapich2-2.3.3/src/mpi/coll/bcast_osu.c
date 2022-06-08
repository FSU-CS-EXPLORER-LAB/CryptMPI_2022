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
#include <unistd.h>
#include "coll_shmem.h"
#include <unistd.h>
#include "common_tuning.h"
#include "bcast_tuning.h"


/**********Added by Cong *********/
#include "secure_allgather.h"
/********************/

#define INTRA_NODE_ROOT 0

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_binomial);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_scatter_doubling_allgather);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_scatter_ring_allgather);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_scatter_ring_allgather_shm);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_shmem);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_knomial_internode);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_knomial_intranode);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_mcast_internode);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_pipelined);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_shmem_zcpy);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_bcast_pipelined_zcpy);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_binomial);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_doubling_allgather);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_shm);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_shmem);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_internode);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_intranode);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_mcast_internode);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_pipelined);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_subcomm);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_binomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_for_bcast_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_doubling_allgather_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_shm_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_internode_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_intranode_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_mcast_internode_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_pipelined_zcpy_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_tune_inter_node_helper_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_inter_node_helper_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_binomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_for_bcast_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_doubling_allgather_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_shm_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_internode_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_intranode_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_mcast_internode_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_pipelined_zcpy_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_tune_inter_node_helper_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_inter_node_helper_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_binomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_for_bcast_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_doubling_allgather_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_shm_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_internode_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_intranode_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_mcast_internode_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_pipelined_zcpy_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_tune_inter_node_helper_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_inter_node_helper_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_binomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_for_bcast_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_doubling_allgather_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_scatter_ring_allgather_shm_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_internode_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_knomial_intranode_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_mcast_internode_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_pipelined_zcpy_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_tune_inter_node_helper_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_inter_node_helper_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_bcast_count_recv);

/* A binomial tree broadcast algorithm.  Good for short messages, 
   Cost = lgp.alpha + n.lgp.beta */
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_binomial_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int (*MV2_Bcast_function) (void *buffer, int count, MPI_Datatype datatype,
                           int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag) = NULL;

int (*MV2_Bcast_intra_node_function) (void *buffer, int count, MPI_Datatype datatype,
                                      int root, MPID_Comm * comm_ptr,
                                      MPIR_Errflag_t *errflag) = NULL;

int MPIR_Bcast_binomial_MV2(void *buffer,
                            int count,
                            MPI_Datatype datatype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{

    MPIR_TIMER_START(coll,bcast,binomial);

    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t nbytes = 0;
    int is_contig, is_homogeneous;
    MPI_Aint type_size;
    MPI_Aint position;
    void *tmp_buf = NULL;
    MPID_Datatype *dtp;
    MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_binomial, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    //printf("%d @ MPIR_Bcast_binomial_MV2, %d , %d\n", rank, count, comm_size);
    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }

    nbytes = (MPIDI_msg_sz_t) (count) * (type_size); 

    if (!is_contig || !is_homogeneous) {
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

        /* TODO: Pipeline the packing and communication */
        position = 0;
        if (rank == root) {
            mpi_errno = MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes,
                                       &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    /* Use short message algorithm, namely, binomial tree */

    /* Algorithm:
       This uses a fairly basic recursive subdivision algorithm.
       The root sends to the process comm_size/2 away; the receiver becomes
       a root for a subtree and applies the same process. 

       So that the new root can easily identify the size of its
       subtree, the (subtree) roots are all powers of two (relative
       to the root) If m = the first power of 2 such that 2^m >= the
       size of the communicator, then the subtree at root at 2^(m-k)
       has size 2^k (with special handling for subtrees that aren't
       a power of two in size).

       Do subdivision.  There are two phases:
       1. Wait for arrival of data.  Because of the power of two nature
       of the subtree roots, the source of this message is alwyas the
       process whose relative rank has the least significant 1 bit CLEARED.
       That is, process 4 (100) receives from process 0, process 7 (111) 
       from process 6 (110), etc.   
       2. Forward to my subtree

       Note that the process that is the tree root is handled automatically
       by this code, since it has no bits set.  */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;
            if (!is_contig || !is_homogeneous)
            {
                MPIR_PVAR_INC(bcast, binomial, recv, nbytes, MPI_BYTE);
                mpi_errno = MPIC_Recv(tmp_buf, nbytes, MPI_BYTE, src,
                                         MPIR_BCAST_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
            }
            else
            {
                MPIR_PVAR_INC(bcast, binomial, recv, count, datatype);
                mpi_errno = MPIC_Recv(buffer, count, datatype, src,
                                         MPIR_BCAST_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);

            }
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
       set from the LSB upto (but not including) mask.  Because of
       the "not including", we start by shifting mask back down one.

       We can easily change to a different algorithm at any power of two
       by changing the test (mask > 1) to (mask > block_size) 

       One such version would use non-blocking operations for the last 2-4
       steps (this also bounds the number of MPI_Requests that would
       be needed).  */

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;
            if (!is_contig || !is_homogeneous)
            {
                MPIR_PVAR_INC(bcast, binomial, send, nbytes, MPI_BYTE);
                mpi_errno = MPIC_Send(tmp_buf, nbytes, MPI_BYTE, dst,
                                         MPIR_BCAST_TAG, comm_ptr, errflag);
            }
            else
            {
                MPIR_PVAR_INC(bcast, binomial, send,  count, datatype);
                mpi_errno = MPIC_Send(buffer, count, datatype, dst,
                                         MPIR_BCAST_TAG, comm_ptr, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);

            }
        }
        mask >>= 1;
    }

    if (!is_contig || !is_homogeneous) {

	if (rank != root) {
            position = 0;
            mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                         count, datatype);
            if (mpi_errno){
                MPIR_ERR_POP(mpi_errno);

	    }

        }
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,bcast,binomial);  
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* FIXME it would be nice if we could refactor things to minimize
   duplication between this and MPIR_Scatter_intra and friends.  We can't use
   MPIR_Scatter_intra as is without inducing an extra copy in the noncontig case. */
/* There are additional arguments included here that are unused because we
   always assume that the noncontig case has been packed into a contig case by
   the caller for now.  Once we start handling noncontig data at the upper level
   we can start handling it here.
   
   At the moment this function always scatters a buffer of nbytes starting at
   tmp_buf address. */
#undef FUNCNAME
#define FUNCNAME scatter_for_bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int scatter_for_bcast_MV2(void *buffer ATTRIBUTE((unused)),
                                 int count ATTRIBUTE((unused)),
                                 MPI_Datatype datatype ATTRIBUTE((unused)),
                                 int root,
                                 MPID_Comm * comm_ptr,
                                 MPIDI_msg_sz_t nbytes,
                                 void *tmp_buf,
                                 int is_contig, int is_homogeneous, MPIR_Errflag_t *errflag)
{
    MPI_Status status;
    int rank, comm_size, src, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t scatter_size=0, curr_size=0, recv_size = 0, send_size=0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;
    //    printf("%d @ scatter_for_bcast_MV2\n", rank);
    
    /* use long message algorithm: binomial tree scatter followed by an
     * allgather */

    /* The scatter algorithm divides the buffer into nprocs pieces and
       scatters them among the processes. Root gets the first piece,
       root+1 gets the second piece, and so forth. Uses the same binomial
       tree algorithm as above. Ceiling division
       is used to compute the size of each piece. This means some
       processes may not get any data. For example if bufsize = 97 and
       nprocs = 16, ranks 15 and 16 will get 0 data. On each process, the
       scattered data is stored at the same offset in the buffer as it is
       on the root process. */

    scatter_size = (nbytes + comm_size - 1) / comm_size;    /* ceiling division */
    curr_size = (rank == root) ? nbytes : 0;    /* root starts with all the
                                                   data */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;
            recv_size = nbytes - relative_rank * scatter_size;
            /* recv_size is larger than what might actually be sent by the
               sender. We don't need compute the exact value because MPI
               allows you to post a larger recv. */
            if (recv_size <= 0) {
                curr_size = 0;  /* this process doesn't receive any data
                                   because of uneven division */
            } else {                
                MPIR_PVAR_INC(bcast, scatter_for_bcast, recv,  recv_size, MPI_BYTE);
                mpi_errno = MPIC_Recv(((char *) tmp_buf +
                                          relative_rank * scatter_size),
                                         recv_size, MPI_BYTE, src,
                                         MPIR_BCAST_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    curr_size = 0;
                } else
                    /* query actual size of data received */
                    MPIR_Get_elements_x_impl(&status, MPI_BYTE, (MPI_Count *) &curr_size);
            }
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
       set from the LSB upto (but not including) mask.  Because of
       the "not including", we start by shifting mask back down
       one. */

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            send_size = curr_size - scatter_size * mask;
            /* mask is also the size of this process's subtree */

            if (send_size > 0) {
                dst = rank + mask;
                if (dst >= comm_size)
                    dst -= comm_size;
                MPIR_PVAR_INC(bcast, scatter_for_bcast, send, send_size, MPI_BYTE);
                mpi_errno = MPIC_Send(((char *) tmp_buf +
                                          scatter_size * (relative_rank +
                                                          mask)), send_size,
                                         MPI_BYTE, dst, MPIR_BCAST_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                     * continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                curr_size -= send_size;
            }
        }
        mask >>= 1;
    }

    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;
}

/*
   Broadcast based on a scatter followed by an allgather.

   We first scatter the buffer using a binomial tree algorithm. This costs
   lgp.alpha + n.((p-1)/p).beta
   If the datatype is contiguous and the communicator is homogeneous,
   we treat the data as bytes and divide (scatter) it among processes
   by using ceiling division. For the noncontiguous or heterogeneous
   cases, we first pack the data into a temporary buffer by using
   MPI_Pack, scatter it as bytes, and unpack it after the allgather.

   For the allgather, we use a recursive doubling algorithm for 
   medium-size messages and power-of-two number of processes. This
   takes lgp steps. In each step pairs of processes exchange all the
   data they have (we take care of non-power-of-two situations). This
   costs approximately lgp.alpha + n.((p-1)/p).beta. (Approximately
   because it may be slightly more in the non-power-of-two case, but
   it's still a logarithmic algorithm.) Therefore, for long messages
   Total Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta
*/

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_scatter_doubling_allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_scatter_doubling_allgather_MV2(void *buffer,
                                              int count,
                                              MPI_Datatype datatype,
                                              int root,
                                              MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,scatter_doubling_allgather);
    MPI_Status status;
    int rank, comm_size, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t scatter_size, curr_size, recv_size = 0;
    MPIDI_msg_sz_t  nbytes = 0; 
    int j, k, i, tmp_mask, is_contig, is_homogeneous;
    MPI_Aint type_size;
    int relative_dst, dst_tree_root, my_tree_root, send_offset;
    int recv_offset, tree_root, nprocs_completed, offset;
    MPI_Aint position;
    MPIU_CHKLMEM_DECL(1);
    MPID_Datatype *dtp;
    MPI_Aint true_extent, true_lb;
    void *tmp_buf;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_scatter_doubling_allgather,
            1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;
    //    printf("%d @ MPIR_Bcast_scatter_doubling_allgather_MV2\n", rank);
    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }

    nbytes = (MPIDI_msg_sz_t) (count) * type_size;

    if (nbytes < comm_size && !comm_ptr->dev.ch.is_pof2) {
	mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count, datatype, root, comm_ptr,  errflag);
        goto fn_exit;
    }

    if (is_contig && is_homogeneous) {
        /* contiguous and homogeneous. no need to pack. */
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

        tmp_buf = (char *) buffer + true_lb;
    } else {
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

        /* TODO: Pipeline the packing and communication */
        position = 0;
        if (rank == root) {
            mpi_errno = MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes,
                                       &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    scatter_size = (nbytes + comm_size - 1) / comm_size;    /* ceiling division */
    curr_size = (rank == root) ? nbytes : 0;    /* root starts with all the
                                                   data */

    mpi_errno = scatter_for_bcast_MV2(buffer, count, datatype, root, comm_ptr,
                                      nbytes, tmp_buf, is_contig,
                                      is_homogeneous, errflag);
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    /* medium size allgather and pof2 comm_size. use recurive doubling. */

    mask = 0x1;
    i = 0;
    while (mask < comm_size) {
        relative_dst = relative_rank ^ mask;

        dst = (relative_dst + root) % comm_size;

        /* find offset into send and recv buffers.
           zero out the least significant "i" bits of relative_rank and
           relative_dst to find root of src and dst
           subtrees. Use ranks of roots as index to send from
           and recv into  buffer */

        dst_tree_root = relative_dst >> i;
        dst_tree_root <<= i;

        my_tree_root = relative_rank >> i;
        my_tree_root <<= i;

        send_offset = my_tree_root * scatter_size;
        recv_offset = dst_tree_root * scatter_size;

        if (relative_dst < comm_size) {
            MPIR_PVAR_INC(bcast, scatter_doubling_allgather, send, curr_size, MPI_BYTE);
            MPIR_PVAR_INC(bcast, scatter_doubling_allgather, recv, (nbytes - recv_offset < 0 ? 0 : nbytes - recv_offset), MPI_BYTE);			
            mpi_errno = MPIC_Sendrecv(((char *) tmp_buf + send_offset),
                                         curr_size, MPI_BYTE, dst,
                                         MPIR_BCAST_TAG,
                                         ((char *) tmp_buf + recv_offset),
                                         (nbytes - recv_offset <
                                          0 ? 0 : nbytes - recv_offset),
                                         MPI_BYTE, dst, MPIR_BCAST_TAG, comm_ptr,
                                         &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                recv_size = 0;
            } else
                MPIR_Get_elements_x_impl(&status, MPI_BYTE, (MPI_Count *) &recv_size);
            curr_size += recv_size;
        }

        /* if some processes in this process's subtree in this step
           did not have any destination process to communicate with
           because of non-power-of-two, we need to send them the
           data that they would normally have received from those
           processes. That is, the haves in this subtree must send to
           the havenots. We use a logarithmic recursive-halfing algorithm
           for this. */

        /* This part of the code will not currently be
           executed because we are not using recursive
           doubling for non power of two. Mark it as experimental
           so that it doesn't show up as red in the coverage tests. */

        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > comm_size) {
            nprocs_completed = comm_size - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
               subtree that have all the data. Send data to others
               in a tree fashion. First find root of current tree
               that is being divided into two. k is the number of
               least-significant bits in this process's rank that
               must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            offset = (scatter_size * (my_tree_root + mask)) % nbytes;
            tmp_mask = mask >> 1;

            while (tmp_mask) {
                relative_dst = relative_rank ^ tmp_mask;
                dst = (relative_dst + root) % comm_size;

                tree_root = relative_rank >> k;
                tree_root <<= k;
                /* send only if this proc has data and destination
                   doesn't have data. */

                if ((relative_dst > relative_rank) &&
                    (relative_rank < tree_root + nprocs_completed)
                    && (relative_dst >= tree_root + nprocs_completed)) {

                    MPIR_PVAR_INC(bcast, scatter_doubling_allgather, send, recv_size, MPI_BYTE);
                    mpi_errno = MPIC_Send(((char *) tmp_buf + offset),
                                             recv_size, MPI_BYTE, dst,
                                             MPIR_BCAST_TAG, comm_ptr, errflag);
                    /* recv_size was set in the previous
                       receive. that's the amount of data to be
                       sent now. */
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                /* recv only if this proc. doesn't have data and sender
                   has data */
                else if ((relative_dst < relative_rank) &&
                         (relative_dst < tree_root + nprocs_completed) &&
                         (relative_rank >= tree_root + nprocs_completed)) {
                    MPIR_PVAR_INC(bcast, scatter_doubling_allgather, recv, nbytes - offset, MPI_BYTE);
                    mpi_errno = MPIC_Recv(((char *) tmp_buf + offset),
                                             nbytes - offset,
                                             MPI_BYTE, dst, MPIR_BCAST_TAG,
                                             comm_ptr, &status, errflag);
                    /* nprocs_completed is also equal to the no. of processes
                       whose data we don't have */
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        recv_size = 0;
                    } else
                        MPIR_Get_elements_x_impl(&status, MPI_BYTE, (MPI_Count *) &recv_size);
                    curr_size += recv_size;
                }
                tmp_mask >>= 1;
                k--;
            }
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    if (!is_contig || !is_homogeneous) {
        if (rank != root) {
            position = 0;
            mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                         count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    MPIR_TIMER_END(coll,bcast,scatter_doubling_allgather);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
/********************* Added by Mehran *********************/


#undef FUNCNAME
#define FUNCNAME MPIR_Concurrent_Bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)


int MPIR_Concurrent_Bcast_MV2(void *buffer,
                                          int count,
                                          MPI_Datatype datatype,
                                          int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    //MPIR_TIMER_START(coll,bcast,scatter_ring_allgather);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t scatter_size=0;
    
    MPIU_CHKLMEM_DECL(3);

    //MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_scatter_ring_allgather, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (comm_size == 1)
        goto fn_exit;


    MPI_Comm shmem_comm, conc_comm;
    MPID_Comm *shmem_commptr=NULL, *conc_commptr = NULL;
    
    
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    conc_comm = comm_ptr->dev.ch.concurrent_comm;

    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(conc_comm, conc_commptr);

    int local_rank, local_size, conc_rank, conc_size;
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;


    
    if(rank == root){
        //scatter

        scatter_size = (count + local_size - 1) / local_size;    //ceiling division


        mpi_errno = MPIR_Scatter_impl(buffer, scatter_size, datatype,
                                  buffer, scatter_size, datatype, root,
                                  shmem_commptr, errflag);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }

        mpi_errno =  MPIR_Bcast_impl(buffer, scatter_size, datatype, 0, conc_commptr, errflag);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }


    }else{
        //compare node id with that of the root, if the same, call scatter
        MPID_Node_id_t my_node_id, root_node_id;
        
        MPID_Get_node_id(comm_ptr, rank, &my_node_id);
        MPID_Get_node_id(comm_ptr, root, &root_node_id);
	scatter_size = (count + local_size - 1) / local_size;
        if(my_node_id == root_node_id){

	      mpi_errno = MPIR_Scatter_impl(buffer, scatter_size, datatype,
                                    buffer+(local_rank*scatter_size), scatter_size, datatype, root,
                                    shmem_commptr, errflag);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
                goto fn_fail;
            }

        }//end if node_id

	    
        mpi_errno =  MPIR_Bcast_impl(buffer+(local_rank*scatter_size), scatter_size, datatype, 0, conc_commptr, errflag);

	    if (mpi_errno) {
		MPIR_ERR_POP(mpi_errno);
		goto fn_fail;
	    }
	    

    }//end else
    
    mpi_errno = MPIR_Allgather_impl(buffer+(local_rank*scatter_size), scatter_size, datatype,
                                                buffer, scatter_size, datatype,
                                                shmem_commptr, errflag);
    if (mpi_errno) {
	MPIR_ERR_POP(mpi_errno);
	goto fn_fail;
    }

    

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/****************************************************************/







/********************* Added by Cong *********************/
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_ML_Shmem_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

/*Shmem-ML (I): (m/l)-byte encryption + concurrent bcast*/

int MPIR_Bcast_ML_Shmem_MV2(void *buffer,
                                          int count,
                                          MPI_Datatype datatype,
                                          int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    ///
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME){
	char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[Bcast rank = %d host = %s count = %d  SA=%d] Func: MPIR_Bcast_ML_Shmem_MV2\n", comm_ptr->rank,hostname,count,se);fflush(stdout);}
#endif        
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t nbytes, scatter_size;
    MPI_Aint type_size;
 
    MPIU_CHKLMEM_DECL(3);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (comm_size == 1)
        goto fn_exit;


    MPID_Comm *shmem_commptr=NULL, *conc_commptr = NULL;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.concurrent_comm, conc_commptr);

    int local_rank, local_size;
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;
    

    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    scatter_size = (nbytes + local_size - 1) / local_size;    
    

    if(rank == root){
        /*Copy plaintext to the shared memory  buffer*/
        mpi_errno = MPIR_Localcopy((void*)((char*)buffer), count, datatype, 
                                    (void*)((char*)shmem_buffer), count, datatype);
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }

        if(security_approach == 3 ){ 
            /*Encrypts (m/l) to SHM cipher*/

                unsigned long ciphertext_len = 0;
                void* out;
                void* in;
                unsigned long in_size;
                if (scatter_size < 16){
                    in_size = scatter_size;
                }else{
                    in_size = (unsigned long)(scatter_size); 
                }
                //out = (void*)( large_send_buffer);
                out = (void*)( ciphertext_shmem_buffer);
                RAND_bytes(out, 12); // 12 bytes of nonce
                in = (void*)(shmem_buffer);
                unsigned long max_out_len = (16 + in_size);
            
                
            if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                    &ciphertext_len, max_out_len,
                                    out, 12,
                                    in, in_size,
                                    NULL, 0))
                {
                        printf("Error in  encryption: SHM-ML-2  \n");
                        fflush(stdout);
                }    
    
            /*Concurrent Bcast*/
    
           mpi_errno = MPIR_Bcast_impl(ciphertext_shmem_buffer, (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);

        }
        else{    

            mpi_errno = MPIR_Bcast_impl(shmem_buffer, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);
        }

    }//end if root
    else{
        //compare node id with that of the root
        MPID_Node_id_t node_id, root_node_id;
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        MPID_Get_node_id(comm_ptr, root, &root_node_id);


        if(node_id == root_node_id){
            
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);  /*Wait for  Copy*/

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
                goto fn_fail;
            }
            
            
            if (security_approach ==3 ){
                /*Encrypts (m/l) to shared cipher buffer*/
                unsigned long ciphertext_len = 0;
                void* out;
                void* in;
                unsigned long in_size;
                if (scatter_size < 16){
                    in_size = scatter_size;
                }else{
                    in_size = (unsigned long)(scatter_size); 
                }
                //out = (void*)( large_send_buffer);
                out = (void*)( ciphertext_shmem_buffer+local_rank*(scatter_size+28));
                RAND_bytes(out, 12); //nonce
                in = (void*)(shmem_buffer +local_rank*scatter_size);
                unsigned long max_out_len = (16 + in_size);
            
                
                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                    &ciphertext_len, max_out_len,
                                    out, 12,
                                    in, in_size,
                                    NULL, 0))
                {
                        printf("Error in  encryption: SHM-ML-1  \n");
                        fflush(stdout);
                }    


                //mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag); 

                /*Concurrent Bcast*/
        
                //mpi_errno = MPIR_Bcast_impl(large_send_buffer, scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                mpi_errno = MPIR_Bcast_impl(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), 
                                                (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);
                /*Local copy */
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);

            }
            else{
                //Unecrypted 
               /* mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE, 
                                    (void*)((char*)ciphertext_shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE);*/
                mpi_errno = MPIR_Bcast_impl(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);

                /*Local copy */
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);
            }

            if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
            }

        }//end if node_id
        else{
            
            if (security_approach ==3){
                //mpi_errno = MPIR_Bcast_impl(large_send_buffer, scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                mpi_errno = MPIR_Bcast_impl(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
                goto fn_fail;
                }
                /*Decrypt to the buffer*/
                void* out;
                void* in;
                unsigned long  decrypted_len;
                unsigned long  ciphertext_len = (scatter_size+16);

                //in = (void*)(large_send_buffer);
                in = (void*)(ciphertext_shmem_buffer+local_rank*(scatter_size+28));
                out = (void*)(shmem_buffer +local_rank*scatter_size);
                /*printf("dec starts, size=%d, rank=%d\n", ciphertext_len, rank);
                fflush(stdout);*/

                if(!EVP_AEAD_CTX_open(ctx, out, &decrypted_len, (ciphertext_len-16),
                        in, 12, in+12, (unsigned long )(ciphertext_len),
                        NULL, 0)){
                        printf("Error in SHM-ML-1 decryption:  while %d tried to decrypt\n", rank);
                        fflush(stdout);   
                    }
                //printf(" dec is done\n");
                
                mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag); /*Wait for decryption*/

                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);

            }else{
                 mpi_errno = MPIR_Bcast_impl(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag); 
                 /*mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
                 mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                         (void*)((char*)buffer), count, datatype);*/
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE, 
                                         (void*)((char*)buffer+local_rank*scatter_size), scatter_size, MPI_BYTE);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                int s=0;
                mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

                for (;s<local_size;s++){
                    if(s!=local_rank){
                        mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+s*scatter_size), scatter_size, MPI_BYTE, 
                                         (void*)((char*)buffer+s*scatter_size), scatter_size, MPI_BYTE);
                    }
                }
                
            }

        }//end of other nodes
       
    }//end all ranks
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
        goto fn_fail;
    }

    

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;


}





/****************************************************************/









/*
   Broadcast based on a scatter followed by an allgather.

   We first scatter the buffer using a binomial tree algorithm. This costs
   lgp.alpha + n.((p-1)/p).beta
   If the datatype is contiguous and the communicator is homogeneous,
   we treat the data as bytes and divide (scatter) it among processes
   by using ceiling division. For the noncontiguous or heterogeneous
   cases, we first pack the data into a temporary buffer by using
   MPI_Pack, scatter it as bytes, and unpack it after the allgather.

   We use a ring algorithm for the allgather, which takes p-1 steps.
   This may perform better than recursive doubling for long messages and
   medium-sized non-power-of-two messages.
   Total Cost = (lgp+p-1).alpha + 2.n.((p-1)/p).beta
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_scatter_ring_allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_scatter_ring_allgather_MV2(void *buffer,
                                          int count,
                                          MPI_Datatype datatype,
                                          int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,scatter_ring_allgather);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t nbytes=0, scatter_size=0;
    int j, i, is_contig, is_homogeneous;
    MPI_Aint type_size, position;
    MPIDI_msg_sz_t *recvcnts=NULL, *displs=NULL; 
    int left, right, jnext;
    void *tmp_buf;
    MPID_Datatype *dtp;
    MPI_Aint true_extent, true_lb;
    MPIU_CHKLMEM_DECL(3);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_scatter_ring_allgather, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    //    printf("%d @ MPIR_Bcast_scatter_ring_allgather_MV2\n", rank);
    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }

    nbytes = (MPIDI_msg_sz_t) (count) * type_size;

    if (is_contig && is_homogeneous) {
        /* contiguous and homogeneous. no need to pack. */
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

        tmp_buf = (char *) buffer + true_lb;
    } else {
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

        /* TODO: Pipeline the packing and communication */
        position = 0;
        if (rank == root) {
            mpi_errno = MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes,
                                       &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    scatter_size = (nbytes + comm_size - 1) / comm_size;    /* ceiling division */

    mpi_errno = scatter_for_bcast_MV2(buffer, count, datatype, root, comm_ptr,
                                      nbytes, tmp_buf, is_contig,
                                      is_homogeneous, errflag);
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    /* long-message allgather or medium-size but non-power-of-two. use ring
     * algorithm. */

    MPIU_CHKLMEM_MALLOC(recvcnts, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t), 
                        mpi_errno, "recvcnts");
    MPIU_CHKLMEM_MALLOC(displs, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t),
                        mpi_errno, "displs");

    for (i = 0; i < comm_size; i++) {
        recvcnts[i] = nbytes - i * scatter_size;
        if (recvcnts[i] > scatter_size)
            recvcnts[i] = scatter_size;
        if (recvcnts[i] < 0)
            recvcnts[i] = 0;
    }

    displs[0] = 0;
    for (i = 1; i < comm_size; i++)
        displs[i] = displs[i - 1] + recvcnts[i - 1];

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    j = rank;
    jnext = left;
    for (i = 1; i < comm_size; i++) {
        MPIR_PVAR_INC(bcast, scatter_ring_allgather, send, recvcnts[(j - root + comm_size) % comm_size], MPI_BYTE);
        MPIR_PVAR_INC(bcast, scatter_ring_allgather, recv, recvcnts[(jnext - root + comm_size) % comm_size], MPI_BYTE);
        mpi_errno =
            MPIC_Sendrecv((char *) tmp_buf +
                             displs[(j - root + comm_size) % comm_size],
                             recvcnts[(j - root + comm_size) % comm_size],
                             MPI_BYTE, right, MPIR_BCAST_TAG,
                             (char *) tmp_buf +
                             displs[(jnext - root + comm_size) % comm_size],
                             recvcnts[(jnext - root + comm_size) % comm_size],
                             MPI_BYTE, left,
                             MPIR_BCAST_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
    }

    if (!is_contig || !is_homogeneous) {
        if (rank != root) {
            position = 0;
            mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                         count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    MPIR_TIMER_END(coll,bcast,scatter_ring_allgather);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/*
   This function is based on MPIR_Bcast_scatter_ring_allgather_MV2(),
   we overlap shared memory bcast with the allgather phase
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_scatter_ring_allgather_shm_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_scatter_ring_allgather_shm_MV2(void *buffer,
                                              int count,
                                              MPI_Datatype datatype,
                                              int root,
                                              MPID_Comm * comm_ptr,
                                              MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,scatter_ring_allgather_shm);
    int rank, comm_size, local_rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIDI_msg_sz_t nbytes=0, scatter_size=0;
    int j, i, is_contig = 1, is_homogeneous = 1;
    MPI_Aint type_size;
    int left = -1, right = -1, jnext;
    MPIDI_msg_sz_t *recvcnts=NULL, *displs=NULL;
    MPI_Aint true_extent, true_lb;
    void * tmp_buf = NULL;
    MPIU_CHKLMEM_DECL(3);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    MPID_Request *request[2];
    MPI_Status status[2];
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    bcast_ring_allgather_shm_packet para_packet; 
    
    MPIDI_msg_sz_t  shmem_offset, shmem_nbytes;
    MPI_Comm leader_comm;
    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_scatter_ring_allgather_shm,
            1);
    local_rank = shmem_commptr->rank;
    rank = comm_ptr->rank;
    if (local_rank == 0) {
        comm_size = leader_commptr->local_size;
        rank = leader_commptr->rank;
    }
    //    printf("%d @ MPIR_Bcast_scatter_ring_allgather_shm_MV2\n", rank);
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    /* even though we always call this algorithm with contigious buffer, still,
     * the datatype might have some holes in the beginning. Therefore, true_lb
     * might be non zero */
    tmp_buf = buffer + true_lb;

    MPIR_Bcast_MV2(&comm_size, 1, MPI_INT, 0, shmem_commptr, errflag);
    if (comm_size == 1) {
        goto fn_exit;
    }

    if(local_rank==0) {

        MPID_Datatype_get_size_macro(datatype, type_size);
        nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    
        scatter_size = (nbytes + comm_size - 1) / comm_size;    /* ceiling division */

        /* long-message allgather or medium-size but non-power-of-two. use ring
         * algorithm. */

        MPIU_CHKLMEM_MALLOC(recvcnts, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t),
                            mpi_errno, "recvcnts");
        MPIU_CHKLMEM_MALLOC(displs, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t),
                            mpi_errno, "displs");

        for (i = 0; i < comm_size; i++) {
            recvcnts[i] = nbytes - i * scatter_size;
            if (recvcnts[i] > scatter_size) {
                recvcnts[i] = scatter_size;
            }
            if (recvcnts[i] < 0) {
                recvcnts[i] = 0;
            }
        }

        displs[0] = 0;
        for (i = 1; i < comm_size; i++) {
            displs[i] = displs[i - 1] + recvcnts[i - 1];
        }
        left = (comm_size + rank - 1) % comm_size;
        right = (rank + 1) % comm_size;

        j = rank;
        jnext = left;
     
        /* parameters are packed up and broadcasted within the node, 
         * therefore a leader pass the parameters to non-leaders
         */
        para_packet.j=j;
        para_packet.jnext=jnext;
        para_packet.root=root;
        para_packet.nbytes=nbytes;
        para_packet.scatter_size=scatter_size;

        MPIR_Bcast_MV2(&para_packet, sizeof(bcast_ring_allgather_shm_packet), 
                       MPI_BYTE, 0, shmem_commptr, errflag);

        mpi_errno = scatter_for_bcast_MV2(tmp_buf, count, datatype, root, leader_commptr,
                                      nbytes, tmp_buf, is_contig,
                                      is_homogeneous, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
       
        /* one chunk is moving along the allgather ring, node-leaders are involved*/
        MPIR_PVAR_INC(bcast, scatter_ring_allgather_shm, recv, recvcnts[(jnext - root + comm_size) % comm_size], MPI_BYTE);
        MPIC_Irecv((char *) tmp_buf +
                   displs[(jnext - root + comm_size) % comm_size],
                   recvcnts[(jnext - root + comm_size) % comm_size],
                   MPI_BYTE, left, MPIR_BCAST_TAG,
                   leader_commptr, &request[0]);

        MPIR_PVAR_INC(bcast, scatter_ring_allgather_shm, send, recvcnts[(j - root + comm_size) % comm_size], MPI_BYTE);
        MPIC_Isend((char *) tmp_buf +
                   displs[(j - root + comm_size) % comm_size], 
                   recvcnts[(j - root + comm_size) % comm_size], 
                   MPI_BYTE, right, MPIR_BCAST_TAG,
                   leader_commptr, &request[1], errflag);

        shmem_offset =  displs[(j - root + comm_size) % comm_size];
        shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];
   
        mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf + shmem_offset, shmem_nbytes, MPI_BYTE,
                                        INTRA_NODE_ROOT, shmem_commptr, errflag);

        mpi_errno = MPIC_Waitall(2, request, status, errflag);

        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        if (mpi_errno) {
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
        
        /* Leaders receive other chunks via allgather ring. When a leader is geting
         * ith chunk from another leader, it broadcast (i-1)th chunk to non-leaders
         * inside the node
        */
        for (i = 2; i < comm_size; i++) {
            MPIR_PVAR_INC(bcast, scatter_ring_allgather_shm, recv, recvcnts[(jnext - root + comm_size) % comm_size], MPI_BYTE);
            MPIC_Irecv((char *) tmp_buf +
                        displs[(jnext - root + comm_size) % comm_size], 
                        recvcnts[(jnext - root + comm_size) % comm_size], 
                        MPI_BYTE, left, MPIR_BCAST_TAG,
                        leader_commptr, &request[0]);

            MPIR_PVAR_INC(bcast, scatter_ring_allgather_shm, send, recvcnts[(j - root + comm_size) % comm_size], MPI_BYTE);
            MPIC_Isend((char *) tmp_buf +
                        displs[(j - root + comm_size) % comm_size], 
                        recvcnts[(j - root + comm_size) % comm_size], 
                        MPI_BYTE, right, MPIR_BCAST_TAG,
                        leader_commptr, &request[1], errflag);

           
            shmem_offset =  displs[(j - root + comm_size) % comm_size];
            shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];


            mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf + shmem_offset, shmem_nbytes, MPI_BYTE,
                                             INTRA_NODE_ROOT, shmem_commptr, errflag);



            mpi_errno = MPIC_Waitall(2, request, status, errflag);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            if (mpi_errno) {
                // for communication errors, just record the error but continue
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            j = jnext;
            jnext = (comm_size + jnext - 1) % comm_size;
        }

        shmem_offset =  displs[(j - root + comm_size) % comm_size];
        shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];
    }


    if(local_rank!=0) {
        MPIR_Bcast_MV2(&para_packet, sizeof(bcast_ring_allgather_shm_packet), 
                       MPI_BYTE, 0, shmem_commptr, errflag);
        j=para_packet.j;
        jnext=para_packet.jnext;
        root=para_packet.root;
        nbytes=para_packet.nbytes;
        scatter_size=para_packet.scatter_size;
        MPIU_CHKLMEM_MALLOC(recvcnts, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t),
                            mpi_errno, "recvcnts");
        MPIU_CHKLMEM_MALLOC(displs, MPIDI_msg_sz_t *, comm_size * sizeof (MPIDI_msg_sz_t),
                            mpi_errno, "displs");


        for (i = 0; i < comm_size; i++) {
            recvcnts[i] = nbytes - i * scatter_size;
            if (recvcnts[i] > scatter_size) {
                recvcnts[i] = scatter_size;
            }
            if (recvcnts[i] < 0) {
                recvcnts[i] = 0;
            }
        }

        displs[0] = 0;
        for (i = 1; i < comm_size; i++) {
            displs[i] = displs[i - 1] + recvcnts[i - 1];
        }
       
        /* Each node-leader has one chunk already in the right place, this chunk doesn't
         * require inter-node communication, we broadcast this chunk to non-leaders in 
         * the node
         */
        /* Non-leaders compute offset and count */
        shmem_offset =  displs[(j - root + comm_size) % comm_size];
        shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];

        mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf + shmem_offset, shmem_nbytes, MPI_BYTE,
                                        INTRA_NODE_ROOT, shmem_commptr, errflag);

        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;

        /* Leaders receive other chunks via allgather ring. When a leader is geting
         * ith chunk from another leader, it broadcast (i-1)th chunk to non-leaders
         * inside the node
         */
        for (i = 2; i < comm_size; i++) {

            /* Non-leaders compute offset and count */
            shmem_offset =  displs[(j - root + comm_size) % comm_size];
            shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];


            mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf + shmem_offset, shmem_nbytes, MPI_BYTE,
                                             INTRA_NODE_ROOT, shmem_commptr, errflag);


            j = jnext;
            jnext = (comm_size + jnext - 1) % comm_size;
        }

        /* Non-leaders compute offset and count */
        shmem_offset =  displs[(j - root + comm_size) % comm_size];
        shmem_nbytes =  recvcnts[(j - root + comm_size) % comm_size];

    } 

    mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf + shmem_offset, shmem_nbytes, MPI_BYTE,
                                         INTRA_NODE_ROOT, shmem_commptr, errflag);


    /* indicate that we have finished shared-memory bcast */
    comm_ptr->dev.ch.intra_node_done = 1;
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret) {
        mpi_errno = mpi_errno_ret;
    } else if (*errflag) {
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    }
    MPIR_TIMER_END(coll,bcast,scatter_ring_allgather_shm);
    return mpi_errno;
  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Shmem_Bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Shmem_Bcast_MV2(void *buffer,
                         int count,
                         MPI_Datatype datatype,
                         int root, MPID_Comm * shmem_comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,shmem);
    int mpi_errno = MPI_SUCCESS;
    int shmem_comm_rank; 
    MPI_Aint type_size;
    MPIDI_msg_sz_t nbytes;
    int local_rank, local_size;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_shmem, 1);
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    shmem_comm_rank = shmem_comm_ptr->dev.ch.shmem_comm_rank;
    void *shmem_buf = NULL;

    local_rank = shmem_comm_ptr->rank;
    local_size = shmem_comm_ptr->local_size;

    if (count == 0) {
        MPIR_TIMER_END(coll,bcast,shmem);
        return MPI_SUCCESS;
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);

    if (mv2_use_slot_shmem_coll && mv2_use_slot_shmem_bcast) {
        char *buf;
        int len; 
        MPIDI_msg_sz_t pos;
        MPI_Aint extent;
        MPI_Aint true_lb, true_extent;
        MPID_Datatype_get_extent_macro(datatype, extent);
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
        nbytes = (MPIDI_msg_sz_t) (count) * extent;
        for (pos = 0; pos < nbytes; pos += mv2_shm_slot_len) {
            buf = (char *) buffer + true_lb + pos;
            len = MIN(nbytes - pos, mv2_shm_slot_len);
            mv2_shm_bcast(shmem_comm_ptr->dev.ch.shmem_info, buf, len, 0);
        }
        MPIR_TIMER_END(coll,bcast,shmem);
        return MPI_SUCCESS;
    }

    if (local_rank == 0) {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(local_size, local_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(buffer, count, datatype, shmem_buf, nbytes, MPI_BYTE);
        MPIDI_CH3I_SHMEM_Bcast_Complete(local_size, local_rank, shmem_comm_rank);
    } else {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(local_size, local_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(shmem_buf, nbytes, MPI_BYTE, buffer, count, datatype);
        MPIDI_CH3I_SHMEM_Bcast_Complete(local_size, local_rank, shmem_comm_rank);
    }
    if (mpi_errno) {
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_POP(mpi_errno);
    }

  fn_fail :
    MPIR_TIMER_END(coll,bcast,shmem);
    return mpi_errno;
}




#undef FUNCNAME
#define FUNCNAME MPIR_Knomial_Bcast_inter_node_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Knomial_Bcast_inter_node_MV2(void *buffer,
                                      int count,
                                      MPI_Datatype datatype,
                                      int root, int knomial_factor, 
                                      MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{   
    MPIR_TIMER_START(coll,bcast,knomial_internode);
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = 0;
    int comm_size = 0, rank = 0;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPID_Request **reqarray = NULL;
    MPI_Status *starray = NULL;
    int src, dst, mask, relative_rank;
    int k;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_knomial_internode, 1);
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(leader_comm, leader_commptr);
    local_rank = shmem_commptr->rank;

    comm_size = leader_commptr->local_size;
    rank = leader_commptr->rank;
    //    printf("%d @ MPIR_Knomial_Bcast_inter_node_MV2\n", rank);
    MPIU_CHKLMEM_DECL(2);

    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        2 * knomial_factor * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        2 * knomial_factor * sizeof (MPI_Status),
                        mpi_errno, "starray");
    if (local_rank == 0) {
        /* inter-node k-nomial bcast  */
        if (comm_size > 1) {
            relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;
            mask = 0x1;

            while (mask < comm_size) {
                if (relative_rank % (knomial_factor * mask)) {
                    src = relative_rank / (knomial_factor * mask) *
                        (knomial_factor * mask) + root;
                    if (src >= comm_size) {
                        src -= comm_size;
                    }

                    MPIR_PVAR_INC(bcast, knomial_internode, recv, count, datatype);
                    mpi_errno = MPIC_Recv(buffer, count, datatype, src,
                                             MPIR_BCAST_TAG, leader_commptr,
                                             MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    break;
                }
                mask *= knomial_factor;
            }

            mask /= knomial_factor;

            while (mask > 0) {
                int reqs = 0;
                for (k = 1; k < knomial_factor; k++) {
                    if (relative_rank + mask * k < comm_size) {
                        dst = rank + mask * k;
                        if (dst >= comm_size) {
                            dst -= comm_size;
                        }
                        MPIR_PVAR_INC(bcast, knomial_internode, send, count, datatype);
                        mpi_errno = MPIC_Isend(buffer, count, datatype, dst,
                                                  MPIR_BCAST_TAG, leader_commptr,
                                                  &reqarray[reqs++], errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                    }
                }
                mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
                if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS)
                    MPIR_ERR_POP(mpi_errno);

                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno == MPI_ERR_IN_STATUS) {
                    int j;
                    for (j = 0; j < reqs; j++) {
                        if (starray[j].MPI_ERROR != MPI_SUCCESS) {
                            mpi_errno = starray[j].MPI_ERROR;
                            if (mpi_errno) {
                                /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }
                    }
                }
                mask /= knomial_factor;
            }
        }
        if (mv2_use_old_bcast == 0) {
            /* Start the shmem-bcast before we send the data across the network */
            mpi_errno = MV2_Bcast_intra_node_function(buffer, count, datatype,
                                                      INTRA_NODE_ROOT,
                                                      shmem_commptr, errflag);
        } else {
            MPI_Aint type_size = 0;
            MPID_Datatype_get_size_macro(datatype, type_size);
            MPIDI_msg_sz_t nbytes; 
 
            nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
            if (nbytes <= mv2_knomial_intra_node_threshold) {
                mpi_errno = MPIR_Shmem_Bcast_MV2(buffer, count, datatype,
                                                 INTRA_NODE_ROOT,
                                                 shmem_commptr, errflag);
            } else {
                mpi_errno =
                    MPIR_Knomial_Bcast_intra_node_MV2(buffer, count, datatype,
                                                      INTRA_NODE_ROOT,
                                                      shmem_commptr, errflag);
            }
        }
        comm_ptr->dev.ch.intra_node_done = 1;
    }
  fn_fail:

    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,bcast,knomial_internode);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Knomial_Bcast_intra_node_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Knomial_Bcast_intra_node_MV2(void *buffer,
                                      int count,
                                      MPI_Datatype datatype,
                                      int root, MPID_Comm * comm_ptr, 
                                      MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,knomial_intranode);
    int local_size = 0, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPID_Request **reqarray = NULL;
    MPI_Status *starray = NULL;
    int src, dst, mask, relative_rank;
    int k;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_knomial_intranode, 1);
    local_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    MPIU_CHKLMEM_DECL(2);
    //    printf("%d @ MPIR_Knomial_Bcast_intra_node_MV2\n", rank);
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                        2 * mv2_intra_node_knomial_factor * sizeof (MPID_Request*),
                        mpi_errno, "reqarray");

    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                        2 * mv2_intra_node_knomial_factor * sizeof (MPI_Status),
                        mpi_errno, "starray");

    /* intra-node k-nomial bcast  */
    if (local_size > 1) {
        relative_rank = (rank >= root) ? rank - root : rank - root + local_size;
        mask = 0x1;

        while (mask < local_size) {
            if (relative_rank % (mv2_intra_node_knomial_factor * mask)) {
                src = relative_rank / (mv2_intra_node_knomial_factor * mask) *
                    (mv2_intra_node_knomial_factor * mask) + root;
                if (src >= local_size) {
                    src -= local_size;
                }

                MPIR_PVAR_INC(bcast, knomial_intranode, recv, count, datatype);
                mpi_errno = MPIC_Recv(buffer, count, datatype, src,
                                         MPIR_BCAST_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                break;
            }
            mask *= mv2_intra_node_knomial_factor;
        }
        mask /= mv2_intra_node_knomial_factor;

        while (mask > 0) {
            int reqs = 0;
            for (k = 1; k < mv2_intra_node_knomial_factor; k++) {
                if (relative_rank + mask * k < local_size) {
                    dst = rank + mask * k;
                    if (dst >= local_size) {
                        dst -= local_size;
                    }
                    MPIR_PVAR_INC(bcast, knomial_intranode, send, count, datatype);
                    mpi_errno = MPIC_Isend(buffer, count, datatype, dst,
                                              MPIR_BCAST_TAG, comm_ptr,
                                              &reqarray[reqs++], errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
            mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
            if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS)
                MPIR_ERR_POP(mpi_errno);

            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno == MPI_ERR_IN_STATUS) {
                int j;
                for (j = 0; j < reqs; j++) {
                    if (starray[j].MPI_ERROR != MPI_SUCCESS) {
                        mpi_errno = starray[j].MPI_ERROR;
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                    }
                }
            }
            mask /= mv2_intra_node_knomial_factor;
        }
    }

  fn_fail:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,bcast,knomial_intranode);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Knomial_Bcast_inter_node_wrapper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Knomial_Bcast_inter_node_wrapper_MV2(void *buffer,
                                      int count,
                                      MPI_Datatype datatype,
                                      int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
   int mpi_errno = MPI_SUCCESS; 
   int knomial_factor=0; 
   if(MV2_Bcast_function == &MPIR_Pipelined_Bcast_MV2) { 
       knomial_factor = mv2_pipelined_knomial_factor; 
   } else { 
       knomial_factor = mv2_inter_node_knomial_factor; 
   } 
   mpi_errno = MPIR_Knomial_Bcast_inter_node_MV2(buffer, count, datatype, root, 
                                         knomial_factor, comm_ptr, errflag); 
   if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
   }

fn_fail:
   return mpi_errno; 
   

} 

#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#undef FUNCNAME
#define FUNCNAME
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Mcast_inter_node_MV2(void *buffer,
                              int count,
                              MPI_Datatype datatype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,mcast_internode);
    int mpi_errno = MPI_SUCCESS;
    int rank, comm_size;
    int extent; 
    MPIDI_msg_sz_t nbytes;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int leader_rank, leader_comm_rank, leader_of_root;
    bcast_info_t *bcast_info;
    void *buf;
    MPIDI_msg_sz_t len, pos;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_mcast_internode, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    bcast_info = (bcast_info_t *) comm_ptr->dev.ch.bcast_info;
    //    printf("%d @ MPIR_Mcast_inter_node_MV2\n", rank);
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);
    leader_comm_rank = comm_ptr->dev.ch.leader_rank[rank];
    leader_rank = comm_ptr->dev.ch.leader_map[rank];
    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    MPI_Aint true_lb, true_extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    
    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;
   
    MPID_Datatype_get_extent_macro(datatype, extent);
    nbytes = (MPIDI_msg_sz_t) (count) * (extent);
    PRINT_DEBUG(DEBUG_MCST_verbose > 3,
                "Calling mcast msg of size %ld fragment size %ld\n",
                nbytes, MAX_MCAST_FRAGMENT_SIZE);
    for (pos = 0; pos < nbytes; pos += MAX_MCAST_FRAGMENT_SIZE) {
        buf = (char *) buffer + true_lb + pos;
        len = MIN(nbytes - pos, MAX_MCAST_FRAGMENT_SIZE);

        if (leader_rank == leader_of_root) {
            if (mv2_use_mcast_pipeline_shm) {
                mpi_errno = MPIR_Shmem_Bcast_MV2((char *) buf, len,
                                                 MPI_BYTE, 0, shmem_commptr, errflag);
            }
        }

        if (leader_comm_rank >= 0) {

            if (IS_MCAST_WINDOW_FULL(bcast_info->win_head, bcast_info->win_tail)) {
                MPIR_Barrier_impl(leader_commptr, errflag);
                bcast_info->win_head++;
                mv2_mcast_flush_sendwin(&bcast_info->send_window);
                bcast_info->win_tail = bcast_info->win_head - 1;
                PRINT_DEBUG(DEBUG_MCST_verbose > 4,    
                            "sendwindow full. tail set to :%u\n", bcast_info->win_tail);
                MPIU_Assert(bcast_info->send_window.head == NULL);
            }

            if (rank == leader_of_root) {
                mv2_mcast_send((bcast_info_t *) comm_ptr->dev.ch.bcast_info, buf, len);
            } else {
                mv2_mcast_recv((bcast_info_t *) comm_ptr->dev.ch.bcast_info, buf, len,
                               leader_of_root);
            }
        }

        if (mv2_use_mcast_pipeline_shm && leader_rank != leader_of_root) {
            mpi_errno = MPIR_Shmem_Bcast_MV2((char *) buf, len,
                                             MPI_BYTE, 0, shmem_commptr, errflag);
        }

        bcast_info->win_head++;
    }

    if (mv2_use_mcast_pipeline_shm) {
        comm_ptr->dev.ch.intra_node_done = 1;
    }

  fn_exit:
    MPIR_TIMER_END(coll,bcast,mcast_internode);
    return mpi_errno;
}
#endif

#undef FUNCNAME
#define FUNCNAME MPIR_Pipelined_Bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Pipelined_Bcast_MV2(void *buffer,
                             int count,
                             MPI_Datatype datatype,
                             int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,pipelined);
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr = NULL;
    int local_rank = 0;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint type_size = 0; 
    MPIDI_msg_sz_t nbytes=0, rem_count = 0, bcast_segment_count = 0, bcast_curr_count = 0;
    MPI_Aint extent;
    MPI_Aint true_extent, true_lb;
    void * tmp_buf = NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_bcast_pipelined, 1);
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Datatype_get_extent_macro(datatype, extent);

    local_rank = shmem_commptr->rank;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * extent;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    /* even though we always call this algorithm with contigious buffer, still,
     * the datatype might have some holes in the beginning. Therefore, true_lb
     * might be non zero */
    tmp_buf = buffer + true_lb;

    rem_count = nbytes;
    bcast_segment_count = MIN(rem_count, bcast_segment_size);

    while (bcast_curr_count < nbytes) {
        comm_ptr->dev.ch.intra_node_done = 0;
        if (local_rank == 0) {
            mpi_errno = MPIR_Knomial_Bcast_inter_node_wrapper_MV2((char *) tmp_buf +
                                                          bcast_curr_count,
                                                          bcast_segment_count,
                                                          MPI_BYTE, root,
                                                          comm_ptr, errflag);
        }
        if (comm_ptr->dev.ch.intra_node_done != 1) {
            if (mv2_use_old_bcast == 0) {
                mpi_errno = MV2_Bcast_intra_node_function((char *) tmp_buf +
                                                 bcast_curr_count,
                                                 bcast_segment_count,
                                                 MPI_BYTE, INTRA_NODE_ROOT,
                                                 shmem_commptr, errflag);
            } else {
                if (bcast_segment_count * type_size <= mv2_knomial_intra_node_threshold) {
                   mpi_errno = MPIR_Shmem_Bcast_MV2((char *) tmp_buf +
                                                     bcast_curr_count,
                                                     bcast_segment_count,
                                                     MPI_BYTE, INTRA_NODE_ROOT,
                                                     shmem_commptr, errflag);
                } else {
                    mpi_errno = MPIR_Knomial_Bcast_intra_node_MV2((char *) tmp_buf +
                                                                  bcast_curr_count,
                                                                  bcast_segment_count,
                                                                  MPI_BYTE, INTRA_NODE_ROOT,
                                                                  shmem_commptr, errflag);
                }
            }
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        bcast_curr_count += bcast_segment_count;
        rem_count -= bcast_segment_count;
        bcast_segment_count = MIN(rem_count, bcast_segment_count);
    }

    comm_ptr->dev.ch.intra_node_done = 1;

  fn_fail:
    MPIR_TIMER_END(coll,bcast,pipelined);
    return mpi_errno;
}

int MPIR_Knomial_Bcast_inter_node_trace_MV2(int root, int mv2_bcast_knomial_factor, 
                 int *src, int *expected_send_count, 
                 int *expected_recv_count, int **dst_array, 
                 MPID_Comm *comm_ptr)
{ 
    int mask=0x1, k, local_size, dst, rank, relative_rank;
    int orig_mask=0x1;
    int recv_iter=0, send_iter=0;
    int *knomial_bcast_dst_array=NULL;
    rank      = comm_ptr->rank;
    local_size = comm_ptr->local_size;
    //    printf("%d @ MPIR_Knomial_Bcast_inter_node_trace_MV2\n", rank);
    relative_rank = (rank >= root) ? rank - root : rank - root + local_size;
 
    while (mask < local_size) {
        if (relative_rank % (mv2_bcast_knomial_factor * mask)) {
            *src = relative_rank / (mv2_bcast_knomial_factor * mask) *
                (mv2_bcast_knomial_factor * mask) + root;
            if (*src >= local_size) {
                *src -= local_size;
            }
            recv_iter++; 
            break;
        }
        mask *= mv2_bcast_knomial_factor;
    }
    mask /= mv2_bcast_knomial_factor;

    orig_mask = mask; 
    while (mask > 0) {
        for (k = 1; k < mv2_bcast_knomial_factor; k++) {
            if (relative_rank + mask * k < local_size) {
                send_iter++; 
            }
        }
        mask /= mv2_bcast_knomial_factor;
     } 

    /* Finally, fill up the dst array */
    if(send_iter > 0) {
        knomial_bcast_dst_array = MPIU_Malloc(sizeof(int)*send_iter);
    }

    mask = orig_mask;
    send_iter=0;
    while (mask > 0) {
        for(k=1;k<mv2_bcast_knomial_factor;k++) {
            if (relative_rank + mask*k < local_size) {
                dst = rank + mask*k;
                if (dst >= local_size) {
                    dst -= local_size;
                }
                knomial_bcast_dst_array[send_iter++] = dst;
            }
        }
        mask /= mv2_bcast_knomial_factor;
    }

    *expected_recv_count = recv_iter;
    *expected_send_count = send_iter;
    *dst_array = knomial_bcast_dst_array;
    return 0;
} 

#ifdef CHANNEL_MRAIL_GEN2
#undef FUNCNAME
#define FUNCNAME MPIR_Shmem_Bcast_Zcpy_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Shmem_Bcast_Zcpy_MV2(void *buffer,
                         int count,
                         MPI_Datatype datatype,
                         int root, 
                         int src, int expected_recv_count, 
                         int *dst_array, int expected_send_count,
                         int knomial_factor, 
                         MPID_Comm *comm_ptr, 
                         MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,bcast,shmem_zcpy);
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint type_size;  
    MPIDI_msg_sz_t nbytes;
    MPI_Comm shmem_comm; 
    MPID_Comm *shmem_commptr=NULL;

    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    shmem_comm = comm_ptr->dev.ch.shmem_comm; 
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);

    MPIU_Assert(mv2_enable_zcpy_bcast==1 && mv2_use_slot_shmem_coll==1);
    if ( count == 0) {
        MPIR_TIMER_END(coll,bcast,shmem_zcpy);
        return MPI_SUCCESS;
    }

    if (mv2_use_slot_shmem_coll && mv2_use_slot_shmem_bcast) {
        char *buf; 
        int len; 
        MPIDI_msg_sz_t pos;
        MPI_Aint extent;
        MPI_Aint true_lb, true_extent;
        MPID_Datatype_get_extent_macro(datatype, extent);
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
        nbytes = count * extent;
        for (pos = 0; pos < nbytes; pos += mv2_shm_slot_len) {
            buf = (char *) buffer + true_lb + pos;
            len = MIN(nbytes - pos, mv2_shm_slot_len);
            mpi_errno = mv2_shm_zcpy_bcast(shmem_commptr->dev.ch.shmem_info, buf, len, root, 
                              src, expected_recv_count, 
                              dst_array, expected_send_count, 
                              knomial_factor,                               
                              comm_ptr); 
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        MPIR_TIMER_END(coll,bcast,shmem_zcpy);
        return MPI_SUCCESS;
    } 

  fn_fail:
    MPIR_TIMER_END(coll,bcast,shmem_zcpy);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Pipelined_Bcast_Zcpy_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Pipelined_Bcast_Zcpy_MV2(void *buffer,
                             int count,
                             MPI_Datatype datatype,
                             int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{   
    
    MPIR_TIMER_START(coll,bcast,pipelined_zcpy);
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr = NULL;
    int local_rank = 0, rank=0;
    int mpi_errno = MPI_SUCCESS;
    int new_root=0;; 
    MPIDI_msg_sz_t nbytes=0;
    int leader_of_root=0, leader_root=0; 
    MPIDI_msg_sz_t   rem_count = 0, bcast_curr_count = 0;
    int bcast_segment_count = 0; 
    int src, expected_send_count=-1, expected_recv_count=-1; 
    int *dst_array = NULL; 
    MPI_Aint extent;
    MPI_Aint true_extent, true_lb;
    void * tmp_buf = NULL;
    static int fn_call=0; 
    MPID_Request *prev_request = NULL, *next_request = NULL; 
    MPI_Status prev_status, next_status; 


    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    /* even though we always call this algorithm with contigious buffer, still,
     * the datatype might have some holes in the beginning. Therefore, true_lb
     * might be non zero */
    tmp_buf = buffer + true_lb;

    rank       = comm_ptr->rank; 
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Datatype_get_extent_macro(datatype, extent);

   /*if (rank==0) 
        printf("%d @ MPIR_Pipelined_Bcast_Zcpy_MV2\n", rank);*/
    
    MPIU_Assert(mv2_enable_zcpy_bcast==1 && mv2_use_slot_shmem_coll==1);
    local_rank = shmem_commptr->rank;
    nbytes = count*extent; 
    rem_count = nbytes;
    bcast_segment_count = MIN(rem_count, bcast_segment_size);
 
    leader_of_root = comm_ptr->dev.ch.leader_map[new_root];
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];

    if (local_rank == 0) {
        MPID_Comm *leader_commptr=NULL; 
        shmem_info_t *shmem_info = NULL; 
        MPI_Comm leader_comm; 
        leader_comm = comm_ptr->dev.ch.leader_comm; 
        MPID_Comm_get_ptr(leader_comm, leader_commptr);             

        shmem_info = comm_ptr->dev.ch.shmem_info; 
        /* If the knomial_factor requested for this specific bcast 
         * is the same as the one that we have used before, the communication
         * tree is already setup and cached. No need to do it again */ 
        if((shmem_info)->bcast_knomial_factor != zcpy_knomial_factor) {
             MPIR_Knomial_Bcast_inter_node_trace_MV2(leader_root, zcpy_knomial_factor,   
                               &src, &expected_send_count, 
                               &expected_recv_count, &dst_array, leader_commptr); 
             (shmem_info)->bcast_exchange_rdma_keys = 1;
        }
    } 

    /* If root is not 0, send the data the rank0. This 
    * is because we are re-using the communication tree 
    * that we have already set up */
    if(rank == root && rank != 0) {
        MPIR_PVAR_INC(bcast, pipelined_zcpy, send, bcast_segment_count, MPI_BYTE);
        mpi_errno = MPIC_Isend( (char *) tmp_buf +
                               bcast_curr_count, bcast_segment_count, 
                               MPI_BYTE, new_root, 
                               MPIR_BCAST_TAG, comm_ptr, 
                               &prev_request, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    if(rank == 0 && rank != root) {
        MPIR_PVAR_INC(bcast, pipelined_zcpy, recv, bcast_segment_count, MPI_BYTE);
        mpi_errno = MPIC_Irecv((char *) tmp_buf + 
                              bcast_curr_count, bcast_segment_count, 
                              MPI_BYTE, root, MPIR_BCAST_TAG, comm_ptr, &prev_request);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    while (bcast_curr_count < nbytes) {
        comm_ptr->dev.ch.intra_node_done = 0;

        if(rank == root && rank != 0) {
            int bcast_next_segment_count=0; 
            bcast_next_segment_count = MIN(rem_count - bcast_segment_count, bcast_segment_count);
            if(bcast_curr_count + bcast_segment_count < nbytes) {
                MPIR_PVAR_INC(bcast, pipelined_zcpy, send, bcast_next_segment_count, MPI_BYTE);
                mpi_errno = MPIC_Isend( (char *) tmp_buf +
                                       bcast_curr_count + bcast_segment_count, 
                                       bcast_next_segment_count, 
                                       MPI_BYTE, new_root, 
                                       MPIR_BCAST_TAG, comm_ptr,
                                       &next_request, errflag);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            } 
        }

        if(rank == 0 && rank != root) {
            int bcast_next_segment_count=0; 
            bcast_next_segment_count = MIN(rem_count - bcast_segment_count, bcast_segment_count);
            if(bcast_curr_count + bcast_segment_count < nbytes) { 
                MPIR_PVAR_INC(bcast, pipelined_zcpy, recv, bcast_next_segment_count, MPI_BYTE);
                mpi_errno = MPIC_Irecv((char *) tmp_buf + 
                                      bcast_curr_count + bcast_segment_count, 
                                      bcast_next_segment_count, 
                                      MPI_BYTE, root, MPIR_BCAST_TAG, comm_ptr,
                                      &next_request);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            } 
        }

        if( (rank == root && rank != 0) || 
            (rank == 0 && rank != root)){ 
             mpi_errno = MPIC_Waitall(1, &prev_request, &prev_status, errflag); 
             prev_request = next_request; 
             prev_status  = next_status; 
        } 
 
        mpi_errno = MPIR_Shmem_Bcast_Zcpy_MV2((char *) tmp_buf +
                                             bcast_curr_count,
                                             bcast_segment_count,
                                             MPI_BYTE, leader_root,
                                             src, expected_recv_count, 
                                             dst_array, expected_send_count, 
                                             zcpy_knomial_factor, 
                                             comm_ptr, 
                                             errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        bcast_curr_count += bcast_segment_count;
        rem_count -= bcast_segment_count;
        bcast_segment_count = MIN(rem_count, bcast_segment_count);
    }

    comm_ptr->dev.ch.intra_node_done = 1;
    if(dst_array != NULL) { 
        MPIU_Free(dst_array); 
    }  
    fn_call++; 

  fn_fail:
    MPIR_TIMER_END(coll,bcast,pipelined_zcpy);
    return mpi_errno;
}
#endif /* CHANNEL_MRAIL_GEN2 */



#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_tune_inter_node_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Bcast_tune_inter_node_helper_MV2(void *buffer,
                                                 int count,
                                                 MPI_Datatype datatype,
                                                 int root,
                                                 MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint type_size; 
    MPIDI_msg_sz_t nbytes=0;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank, local_size, global_rank = -1;
    int leader_root, leader_of_root;

    rank = comm_ptr->rank;
    //    printf("%d @ MPIR_Bcast_tune_inter_node_helper_MV2\n", rank);
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    if ((local_rank == 0) && (local_size > 1)) {
        global_rank = leader_commptr->rank;
    }

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);

#ifdef CHANNEL_MRAIL_GEN2
    if(&MPIR_Pipelined_Bcast_Zcpy_MV2 == MV2_Bcast_function) { 
       /* We should not be reaching here, with bcast_fn set to the 
        * zcpy function. The bcast-zcpy runtime variable has been disabled. 
        * Just set MV2_Bcast_function to something else to handle this corner
        * case */
        MV2_Bcast_function = &MPIR_Pipelined_Bcast_MV2; 
    } 
#endif

    if (local_size > 1) {
        if ((local_rank == 0) && (root != rank) && (leader_root == global_rank)) {
            MPIR_PVAR_INC(bcast, tune_inter_node_helper, recv, count, datatype);
            mpi_errno = MPIC_Recv(buffer, count, datatype, root,
                                     MPIR_BCAST_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
        if ((local_rank != 0) && (root == rank)) {
            MPIR_PVAR_INC(bcast, tune_inter_node_helper, send, count, datatype);
            mpi_errno = MPIC_Send(buffer, count, datatype,
                                     leader_of_root, MPIR_BCAST_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }
#if defined(_MCST_SUPPORT_)
    if (MV2_SELECT_MCAST_BASED_BCAST(comm_ptr, nbytes)) {
        mpi_errno = MPIR_Mcast_inter_node_MV2(buffer, count, datatype, root, comm_ptr,
                                              errflag);
        if (mpi_errno == MPI_SUCCESS) {
            goto fn_exit;
        } else {
            goto fn_fail;
        }
    }
#endif

    if (local_rank == 0) {
        leader_comm = comm_ptr->dev.ch.leader_comm;
        root = leader_root;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        rank = leader_commptr->rank;
    }

    if (MV2_Bcast_function == &MPIR_Pipelined_Bcast_MV2) {
        mpi_errno = MPIR_Pipelined_Bcast_MV2(buffer, count, datatype,
                                             root, comm_ptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else if (MV2_Bcast_function == &MPIR_Bcast_scatter_ring_allgather_shm_MV2) {
        mpi_errno = MPIR_Bcast_scatter_ring_allgather_shm_MV2(buffer, count,
                                                              datatype, leader_root,
                                                              comm_ptr,
                                                              errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        if (local_rank == 0) {
            if (MV2_Bcast_function == &MPIR_Knomial_Bcast_inter_node_wrapper_MV2) {
                mpi_errno = MPIR_Knomial_Bcast_inter_node_wrapper_MV2(buffer, count,
                                                              datatype, root,
                                                              comm_ptr, errflag);
            } else {
                mpi_errno = MV2_Bcast_function(buffer, count, datatype,
                                               root, leader_commptr, errflag);
            }
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_inter_node_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Bcast_inter_node_helper_MV2(void *buffer,
                                            int count,
                                            MPI_Datatype datatype,
                                            int root,
                                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint type_size; 
    MPIDI_msg_sz_t nbytes=0;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank, local_size, global_rank = -1;
    int leader_root, leader_of_root;

    rank = comm_ptr->rank;
    //    printf("%d @ MPIR_Bcast_inter_node_helper_MV2\n", rank);
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    if ((local_rank == 0) && (local_size > 1)) {
        global_rank = leader_commptr->rank;
    }

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);

    if (local_size > 1) {
        if ((local_rank == 0) && (root != rank) && (leader_root == global_rank)) {
            MPIR_PVAR_INC(bcast, inter_node_helper, recv, count, datatype);
            mpi_errno = MPIC_Recv(buffer, count, datatype, root,
                                     MPIR_BCAST_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
        if ((local_rank != 0) && (root == rank)) {
            MPIR_PVAR_INC(bcast, inter_node_helper, send, count, datatype);
            mpi_errno = MPIC_Send(buffer, count, datatype,
                                     leader_of_root, MPIR_BCAST_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }
#if defined(_MCST_SUPPORT_)
    if (MV2_SELECT_MCAST_BASED_BCAST(comm_ptr, nbytes)) {
        mpi_errno = MPIR_Mcast_inter_node_MV2(buffer, count, datatype,
                                              root, comm_ptr, errflag);
        if (mpi_errno == MPI_SUCCESS) {
            goto fn_exit;
        }
    }
#endif

    if (mv2_use_pipelined_bcast == 1 && nbytes > bcast_segment_size) {
        mpi_errno = MPIR_Pipelined_Bcast_MV2(buffer, count, datatype,
                                             leader_root, comm_ptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        if (local_rank == 0) {
            leader_comm = comm_ptr->dev.ch.leader_comm;
            root = leader_root;
            MPID_Comm_get_ptr(leader_comm, leader_commptr);
            rank = leader_commptr->rank;
        }

        if (mv2_knomial_inter_leader_bcast == 1
            && nbytes <= mv2_knomial_inter_leader_threshold) {
            if (local_rank == 0) {
                mpi_errno = MPIR_Knomial_Bcast_inter_node_wrapper_MV2(buffer, count,
                                                              datatype, root,
                                                              comm_ptr, errflag);
            }
        } else {
            if (mv2_scatter_ring_inter_leader_bcast) {
                if (mv2_bcast_scatter_ring_overlap == 1) {
                    if (nbytes <= mv2_bcast_scatter_ring_overlap_msg_upperbound &&
                        comm_ptr->local_size >=
                        mv2_bcast_scatter_ring_overlap_cores_lowerbound) {

                        mpi_errno = MPIR_Bcast_scatter_ring_allgather_shm_MV2(buffer,
                                                                              count,
                                                                              datatype,
                                                                              leader_root,
                                                                              comm_ptr,
                                                                              errflag);
                    } else if (local_rank == 0) {

                        mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count,
                                                                          datatype,
                                                                          root,
                                                                          leader_commptr,
                                                                          errflag);
                    }
                } else if (local_rank == 0) {
                    mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count,
                                                                      datatype,
                                                                      root,
                                                                      leader_commptr,
                                                                      errflag);
                }

            } else if (local_rank == 0) {

                if (mv2_scatter_rd_inter_leader_bcast) {
                    mpi_errno =
                        MPIR_Bcast_scatter_doubling_allgather_MV2(buffer, count,
                                                                  datatype, root,
                                                                  leader_commptr,
                                                                  errflag);
                } else if (mv2_knomial_inter_leader_bcast) {
                    mpi_errno = MPIR_Knomial_Bcast_inter_node_wrapper_MV2(buffer, count,
                                                                  datatype, root,
                                                                  comm_ptr, errflag);
                } else {
                    mpi_errno = MPIR_Bcast_binomial_MV2(buffer, count,
                                                        datatype, root,
                                                        leader_commptr, errflag);
                }
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Bcast_intra_MV2(void *buffer,
                         int count,
                         MPI_Datatype datatype,
                         int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size, rank;
    int two_level_bcast = 1;
    MPIDI_msg_sz_t nbytes = 0; 
    int is_homogeneous, is_contig;
    MPI_Aint type_size, position;
    void *tmp_buf = NULL;
    MPID_Comm *shmem_commptr = NULL;
    MPI_Comm shmem_comm;
    MPID_Datatype *dtp;
    //    printf("%d - Hello from MPIR_Bcast_intra_MV2\n", count);
    MPIU_THREADPRIV_DECL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPIR_BCAST_INTRA_MV2);

    MPID_MPI_FUNC_ENTER(MPID_STATE_MPIR_BCAST_INTRA_MV2);
    MPIU_CHKLMEM_DECL(1);

    /* The various MPIR_Bcast_* impls use NMPI functions, so we bump the nest
       count here to avoid repeatedly calling incr/decr. */
    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    if (count == 0)
        goto fn_exit;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    //    printf("%d @ MPIR_Bcast_intra_MV2\n", rank);
    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    if (comm_size <= mv2_bcast_two_level_system_size) {
        if (nbytes > mv2_bcast_short_msg && nbytes < mv2_bcast_large_msg) {
            two_level_bcast = 1;
        } else {
            two_level_bcast = 0;
        }
    }

    if (comm_ptr->dev.ch.shmem_coll_ok == 1
        && mv2_enable_shmem_bcast == 1
        && (two_level_bcast == 1
#if defined(_MCST_SUPPORT_)
            || MV2_SELECT_MCAST_BASED_BCAST(comm_ptr, nbytes)
#endif
        )) {

        if (!is_contig || !is_homogeneous) {
            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

            /* TODO: Pipeline the packing and communication */
            position = 0;
            if (rank == root) {
                mpi_errno =
                    MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes, &position);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
        }

        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        if (!is_contig || !is_homogeneous) {
            mpi_errno =
                MPIR_Bcast_inter_node_helper_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                 root, comm_ptr, errflag);
        } else {
            mpi_errno =
                MPIR_Bcast_inter_node_helper_MV2(buffer, count, datatype, root,
                                                 comm_ptr, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        /* We are now done with the inter-node phase */
        if (comm_ptr->dev.ch.intra_node_done == 0) {
            if (nbytes <= mv2_knomial_intra_node_threshold) {
                if (!is_contig || !is_homogeneous) {
                    mpi_errno = MPIR_Shmem_Bcast_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                     root, shmem_commptr, errflag);
                } else {
                    mpi_errno = MPIR_Shmem_Bcast_MV2(buffer, count, datatype,
                                                     root, shmem_commptr, errflag);
                }
            } else {
                if (!is_contig || !is_homogeneous) {
                    mpi_errno =
                        MPIR_Knomial_Bcast_intra_node_MV2(tmp_buf, nbytes,
                                                          MPI_BYTE,
                                                          INTRA_NODE_ROOT,
                                                          shmem_commptr, errflag);
                } else {
                    mpi_errno =
                        MPIR_Knomial_Bcast_intra_node_MV2(buffer, count,
                                                          datatype,
                                                          INTRA_NODE_ROOT,
                                                          shmem_commptr, errflag);
                }
            }
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        if (!is_contig || !is_homogeneous) {
            /* Finishing up... */
            if (rank != root) {
                position = 0;
                mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                             count, datatype);
            }
        }
    } else {
        if (nbytes <= mv2_bcast_short_msg) {
            mpi_errno = MPIR_Bcast_binomial_MV2(buffer, count, datatype, root,
                                                comm_ptr, errflag);
        } else {
            if (mv2_scatter_rd_inter_leader_bcast) {
                mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count,
                                                                  datatype,
                                                                  root,
                                                                  comm_ptr, errflag);
            } else {
                mpi_errno =
                    MPIR_Bcast_scatter_doubling_allgather_MV2(buffer, count,
                                                              datatype, root,
                                                              comm_ptr, errflag);
            }
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

  fn_exit:
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPIR_BCAST_INTRA_MV2);
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;

  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Bcast_index_tuned_intra_MV2(void *buffer,
                              int count,
                              MPI_Datatype datatype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
      
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size, rank;
    int two_level_bcast = 1;
    MPIDI_msg_sz_t nbytes = 0;
    int comm_size_index = 0;
    int inter_node_algo_index = 0;
    int intra_node_algo_index = 0;
    int local_size = 0;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int table_min_comm_size = 0;
    int table_max_comm_size = 0;
    int table_min_inter_size = 0;
    int table_max_inter_size = 0;
    int table_min_intra_size = 0;
    int table_max_intra_size = 0;
    int last_inter;
    int last_intra;
    int lp2ltn; // largest power of 2 less than n
    int lp2ltn_min;
    int is_homogeneous, is_contig;
    MPI_Aint type_size, position;
    void *tmp_buf = NULL;
    MPID_Comm *shmem_commptr = NULL;
    MPI_Comm shmem_comm;
    MPID_Datatype *dtp;

    MPIU_THREADPRIV_DECL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPIR_BCAST_INDEX_TUNED_INTRA_MV2);

    MPID_MPI_FUNC_ENTER(MPID_STATE_MPIR_BCAST_INDEX_TUNED_INTRA_MV2);
    MPIU_CHKLMEM_DECL(1);

    /* The various MPIR_Bcast_* impls use NMPI functions, so we bump the nest
       count here to avoid repeatedly calling incr/decr. */
    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    if (count == 0)
        goto fn_exit;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    
    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        if (mv2_bcast_indexed_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }
        if (likely(mv2_enable_shmem_bcast && mv2_enable_skip_tuning_table_search && (nbytes <= mv2_coll_skip_table_threshold))) {
            /* for small messages, force shmem + zcpy pipeline */
#if defined CHANNEL_MRAIL_GEN2
            if (mv2_enable_zcpy_bcast == 1) {
                MV2_Bcast_function = &MPIR_Pipelined_Bcast_Zcpy_MV2;
            } else {
                MV2_Bcast_function = &MPIR_Pipelined_Bcast_MV2; 
            }
            MV2_Bcast_intra_node_function = &MPIR_Knomial_Bcast_intra_node_MV2;
#elif defined CHANNEL_PSM 
            MV2_Bcast_function = &MPIR_Bcast_binomial_MV2; 
            MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
#endif
            two_level_bcast = 1;
            zcpy_knomial_factor = 8;
            mv2_inter_node_knomial_factor = 8;
            bcast_segment_size = 8192;
            goto skip_tuning_tables;
        }

        FIND_PPN_INDEX  (bcast, local_size,conf_index, partial_sub_ok)
    }
 
    if (partial_sub_ok != 1) {
        conf_index = mv2_bcast_indexed_num_ppn_conf/2;
    }
        
conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_bcast_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_bcast_indexed_thresholds_table[conf_index][mv2_size_bcast_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
	/* Comm size smaller than smallest configuration in table: use smallest available */
	comm_size_index = 0;
    }
    else if (comm_size > table_max_comm_size) {
	/* Comm size larger than largest configuration in table: use largest available */
	comm_size_index = mv2_size_bcast_indexed_tuning_table[conf_index] - 1;
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

    last_inter = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
    table_min_inter_size = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
    table_max_inter_size = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[last_inter].msg_sz;
    last_intra = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].size_intra_table - 1;
    table_min_intra_size = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].intra_node[0].msg_sz;
    table_max_intra_size = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].intra_node[last_intra].msg_sz;
    
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
    
    if (nbytes < table_min_intra_size) {
	/* Msg size smaller than smallest configuration in table: use smallest available */
	intra_node_algo_index = 0;
    }
    else if (nbytes > table_max_intra_size) {
	/* Msg size larger than largest configuration in table: use largest available */
	intra_node_algo_index = last_intra;
    }
    else {
	/* Msg size in between smallest and largest configuration: find closest match */
	if (pow(2, (int)log2(nbytes)) == nbytes) {
	    intra_node_algo_index = log2(nbytes / table_min_intra_size );
	}
	else {
	    lp2ltn = pow(2, (int)log2(nbytes));
	    intra_node_algo_index = (lp2ltn < table_min_intra_size) ? 0 : log2(lp2ltn / table_min_intra_size );
	}
    }
        
    MV2_Bcast_function =
        mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[inter_node_algo_index].
        MV2_pt_Bcast_function;

    MV2_Bcast_intra_node_function =
        mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].
        intra_node[intra_node_algo_index].MV2_pt_Bcast_function;

    /***************** Added by Mehran  ***********************/

    // if(concurrent_comm == 1 && MV2_Bcast_intra_node_function == &Connnnnnnn){

    // }

    /*********************************************************/


    if (mv2_user_bcast_intra == NULL && 
            MV2_Bcast_intra_node_function == &MPIR_Knomial_Bcast_intra_node_MV2) {
            MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
    }

    if (mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[inter_node_algo_index].
        zcpy_pipelined_knomial_factor != -1) {
        zcpy_knomial_factor = 
            mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[inter_node_algo_index].
            zcpy_pipelined_knomial_factor;
    }

    if (mv2_pipelined_zcpy_knomial_factor != -1) {
        zcpy_knomial_factor = mv2_pipelined_zcpy_knomial_factor;
    }

    /* If we use previous shmem scheme, fall back to previous threshold for intra-node*/
    if (!mv2_use_slot_shmem_coll || !mv2_use_slot_shmem_bcast){
        /* not depending on intra node tuning table with old shmem design */
        if (nbytes <= mv2_knomial_intra_node_threshold){
            MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
        } else {
            MV2_Bcast_intra_node_function = &MPIR_Knomial_Bcast_intra_node_MV2;
        }
    } else if(MV2_Bcast_intra_node_function == NULL) {
        /* if tuning table do not have any intra selection, set func pointer to
        ** default one for mcast intra node */
        MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
    }

    /* Set value of pipeline segment size */
    bcast_segment_size = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].bcast_segment_size;
    
    /* Set value of inter node knomial factor */
    mv2_inter_node_knomial_factor = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].inter_node_knomial_factor;

    /* Set value of intra node knomial factor */
    mv2_intra_node_knomial_factor = mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].intra_node_knomial_factor;

    /* Check if we will use a two level algorithm or not */
    two_level_bcast =
        mv2_bcast_indexed_thresholds_table[conf_index][comm_size_index].is_two_level_bcast[inter_node_algo_index];

    if (MV2_Bcast_function == &MPIR_Knomial_Bcast_inter_node_wrapper_MV2 &&  
            two_level_bcast != 1) {
        /* knomial inter node wrapper algorithm relies on leader_comm, therefore, it
         * most be called only on leader_comm */
        MV2_Bcast_function = &MPIR_Bcast_binomial_MV2;
    }

skip_tuning_tables:
#if defined CHANNEL_MRAIL_GEN2
    if (mv2_bcast_indexed_table_ppn_conf[0] != -1 && (mv2_enable_zcpy_bcast == 0 
#if defined(_MCST_SUPPORT_)
        || MV2_SELECT_MCAST_BASED_BCAST(comm_ptr, nbytes) 
#endif
        )) {
        MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
        MV2_Bcast_function = &MPIR_Knomial_Bcast_inter_node_wrapper_MV2;
        two_level_bcast = 1;
    }
#endif

/*****************Add by Cong ************************************/
if(concurrent_comm == 1 && concurrent_bcast == 2 && comm_ptr->dev.ch.concurrent_comm != NULL){
    /*if(rank ==0 || rank ==18){
    printf("MPIR_Bcast_ML_Shmem_MV2\n");
    }*/

	mpi_errno = MPIR_Bcast_ML_Shmem_MV2(buffer, count,
                            datatype, root,
                            comm_ptr, errflag);


    }


/******************************************************************/


  else  if(concurrent_comm == 1 && concurrent_bcast == 1 && comm_ptr->dev.ch.concurrent_comm != NULL){
	mpi_errno = MPIR_Concurrent_Bcast_MV2(buffer, count,
                            datatype, root,
                            comm_ptr, errflag);

    }
    else if (comm_ptr->dev.ch.shmem_coll_ok != 1) {
        if(nbytes < MPICH_LARGE_MSG_COLLECTIVE_SIZE) { 
            mpi_errno = MPIR_Bcast_intra(buffer, count, datatype, root, 
                        comm_ptr, errflag);
        } else { 
            mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count, 
                            datatype, root, 
                            comm_ptr, errflag);
        } 
    } else if (mv2_enable_shmem_bcast == 1 && two_level_bcast == 1) {
        if (!is_contig || !is_homogeneous) {
            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

            /* TODO: Pipeline the packing and communication */
            position = 0;
            if (rank == root) {
                mpi_errno =
                    MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes, &position);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
        }
#ifdef _OSU_MVAPICH_
#ifdef CHANNEL_MRAIL_GEN2
        if ((mv2_enable_zcpy_bcast == 1) &&
              (&MPIR_Pipelined_Bcast_Zcpy_MV2 == MV2_Bcast_function)) {  
            if (!is_contig || !is_homogeneous) {
                mpi_errno = MPIR_Pipelined_Bcast_Zcpy_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                 root, comm_ptr, errflag);
            } else { 
                mpi_errno = MPIR_Pipelined_Bcast_Zcpy_MV2(buffer, count, datatype,
                                                 root, comm_ptr, errflag);
            } 
        } else 
#endif
#endif /* _OSU_MVAPICH_ */
        { 
            shmem_comm = comm_ptr->dev.ch.shmem_comm;
            MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
            if (!is_contig || !is_homogeneous) {
                mpi_errno =
                    MPIR_Bcast_tune_inter_node_helper_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                          root, comm_ptr, errflag);
            } else {
                mpi_errno =
                    MPIR_Bcast_tune_inter_node_helper_MV2(buffer, count, datatype, root,
                                                          comm_ptr, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* We are now done with the inter-node phase */
            if (comm_ptr->dev.ch.intra_node_done == 0) {

                if (!is_contig || !is_homogeneous) {
                    mpi_errno = MV2_Bcast_intra_node_function(tmp_buf, nbytes,
                                                              MPI_BYTE, INTRA_NODE_ROOT, shmem_commptr,
                                                              errflag);
                } else {
                    mpi_errno = MV2_Bcast_intra_node_function(buffer, count,
                                                              datatype, INTRA_NODE_ROOT, shmem_commptr,
                                                              errflag);

                }
            }
        } 
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        if (!is_contig || !is_homogeneous) {
            /* Finishing up... */
            if (rank != root) {
                position = 0;
                mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                             count, datatype);
            }
        }
    } else {
        /* We use Knomial for intra node */
        MV2_Bcast_intra_node_function = &MPIR_Knomial_Bcast_intra_node_MV2;
        if (mv2_enable_shmem_bcast == 0) {
            /* Fall back to non-tuned version */
            MPIR_Bcast_intra_MV2(buffer, count, datatype, root, comm_ptr, errflag);
        } else {
#ifdef CHANNEL_MRAIL_GEN2
            if ((&MPIR_Pipelined_Bcast_Zcpy_MV2 == MV2_Bcast_function) &&
                (mv2_enable_zcpy_bcast == 0)) {
                /* We should not be reaching here, with bcast_fn set to the 
                 * zcpy function. The bcast-zcpy runtime variable has been disabled. 
                 * Just set MV2_Bcast_function to something else to handle this corner
                 * case */
                MV2_Bcast_function = &MPIR_Bcast_binomial_MV2; 
            } 
#endif
            mpi_errno = MV2_Bcast_function(buffer, count, datatype, root,
                                           comm_ptr, errflag);

        }
    }

    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

  fn_exit:
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPIR_BCAST_INDEX_TUNED_INTRA_MV2);
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;

  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_tune_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Bcast_tune_intra_MV2(void *buffer,
                              int count,
                              MPI_Datatype datatype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{



    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size, rank;
    int two_level_bcast = 1;
    MPIDI_msg_sz_t nbytes = 0; 
    int range = 0;
    int range_threshold = 0;
    int range_threshold_intra = 0;
    int is_homogeneous, is_contig;
    MPI_Aint type_size, position;
    void *tmp_buf = NULL;
    MPID_Comm *shmem_commptr = NULL;
    MPI_Comm shmem_comm;
    MPID_Datatype *dtp;

    MPIU_THREADPRIV_DECL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPIR_BCAST_TUNE_INTRA_MV2);

    MPID_MPI_FUNC_ENTER(MPID_STATE_MPIR_BCAST_TUNE_INTRA_MV2);
    MPIU_CHKLMEM_DECL(1);

    /* The various MPIR_Bcast_* impls use NMPI functions, so we bump the nest
       count here to avoid repeatedly calling incr/decr. */
    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    if (count == 0)
        goto fn_exit;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

/******************Added by Cong (debug)********************************/
   /* printf("MPIR_Bcast_tune_intra_MV2, rank =%d, security_approach=%d\n",rank,security_approach);
    fflush(stdout);*/

/*******************************************************/
    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN)
        is_contig = 1;
    else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* MPI_Type_size() might not give the accurate size of the packed
     * datatype for heterogeneous systems (because of padding, encoding,
     * etc). On the other hand, MPI_Pack_size() can become very
     * expensive, depending on the implementation, especially for
     * heterogeneous systems. We want to use MPI_Type_size() wherever
     * possible, and MPI_Pack_size() in other places.
     */
    if (is_homogeneous) {
        MPID_Datatype_get_size_macro(datatype, type_size);
    } else {
        MPIR_Pack_size_impl(1, datatype, &type_size);
    }
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);

    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_bcast_tuning_table - 1)) &&
           (comm_size > mv2_bcast_thresholds_table[range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold < (mv2_bcast_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_bcast_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_bcast_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    /* Search for corresponding intra-node function */
    while ((range_threshold_intra <
            (mv2_bcast_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_bcast_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_bcast_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }

    MV2_Bcast_function =
        mv2_bcast_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Bcast_function;

    MV2_Bcast_intra_node_function =
        mv2_bcast_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Bcast_function;

    if (mv2_user_bcast_intra == NULL && 
            MV2_Bcast_intra_node_function == &MPIR_Knomial_Bcast_intra_node_MV2) {
            MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
    }

    if (mv2_bcast_thresholds_table[range].inter_leader[range_threshold].
        zcpy_pipelined_knomial_factor != -1) {
        zcpy_knomial_factor = 
            mv2_bcast_thresholds_table[range].inter_leader[range_threshold].
            zcpy_pipelined_knomial_factor;
    }

    if (mv2_pipelined_zcpy_knomial_factor != -1) {
        zcpy_knomial_factor = mv2_pipelined_zcpy_knomial_factor;
    }

    /* If we use previous shmem scheme, fall back to previous threshold for intra-node*/
    if (!mv2_use_slot_shmem_coll || !mv2_use_slot_shmem_bcast){
        /* not depending on intra node tuning table with old shmem design */
        if (nbytes <= mv2_knomial_intra_node_threshold){
            MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
        } else {
            MV2_Bcast_intra_node_function = &MPIR_Knomial_Bcast_intra_node_MV2;
        }
    } else if(MV2_Bcast_intra_node_function == NULL) {
        /* if tuning table do not have any intra selection, set func pointer to
        ** default one for mcast intra node */
        MV2_Bcast_intra_node_function = &MPIR_Shmem_Bcast_MV2;
    }

    /* Set value of pipeline segment size */
    bcast_segment_size = mv2_bcast_thresholds_table[range].bcast_segment_size;
    
    /* Set value of inter node knomial factor */
    mv2_inter_node_knomial_factor = mv2_bcast_thresholds_table[range].inter_node_knomial_factor;

    /* Set value of intra node knomial factor */
    mv2_intra_node_knomial_factor = mv2_bcast_thresholds_table[range].intra_node_knomial_factor;

    /* Check if we will use a two level algorithm or not */
    two_level_bcast =
        mv2_bcast_thresholds_table[range].is_two_level_bcast[range_threshold];




    /*****************Add by Cong ************************************/
    if(concurrent_comm == 1 && concurrent_bcast == 2 && comm_ptr->dev.ch.concurrent_comm != NULL){
        /*if(rank ==0 || rank ==18){
            printf("MPIR_Bcast_ML_Shmem_MV2\n");
        }*/

        mpi_errno = MPIR_Bcast_ML_Shmem_MV2(buffer, count,
                                datatype, root,
                                comm_ptr, errflag);

        }


/******************************************************************/


  else  if(concurrent_comm == 1 && concurrent_bcast == 1 && comm_ptr->dev.ch.concurrent_comm != NULL){
        mpi_errno = MPIR_Concurrent_Bcast_MV2(buffer, count,
                            datatype, root,
                            comm_ptr, errflag);

    }

    else if (comm_ptr->dev.ch.shmem_coll_ok != 1) {
        if(nbytes < MPICH_LARGE_MSG_COLLECTIVE_SIZE) { 
            mpi_errno = MPIR_Bcast_intra(buffer, count, datatype, root, 
                        comm_ptr, errflag);
        } else { 
            mpi_errno = MPIR_Bcast_scatter_ring_allgather_MV2(buffer, count, 
                            datatype, root, 
                            comm_ptr, errflag);
        } 
    } else if (mv2_enable_shmem_bcast == 1 && two_level_bcast == 1) {
        if (!is_contig || !is_homogeneous) {
            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

            /* TODO: Pipeline the packing and communication */
            position = 0;
            if (rank == root) {
                mpi_errno =
                    MPIR_Pack_impl(buffer, count, datatype, tmp_buf, nbytes, &position);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
        }
#ifdef CHANNEL_MRAIL_GEN2
        if ((mv2_enable_zcpy_bcast == 1) &&
              (&MPIR_Pipelined_Bcast_Zcpy_MV2 == MV2_Bcast_function)) {  
            if (!is_contig || !is_homogeneous) {
                mpi_errno = MPIR_Pipelined_Bcast_Zcpy_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                 root, comm_ptr, errflag);
            } else { 
                mpi_errno = MPIR_Pipelined_Bcast_Zcpy_MV2(buffer, count, datatype,
                                                 root, comm_ptr, errflag);
            } 
        } else 
#endif /* defined(CHANNEL_MRAIL_GEN2) */
        { 
            shmem_comm = comm_ptr->dev.ch.shmem_comm;
            MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
            /*Debug*/
           /* if(rank ==0 || rank ==18){
                    printf("MPIR_Bcast_tune_inter_node_helper_MV2\n");
            }*/


            if (!is_contig || !is_homogeneous) {
                mpi_errno =
                    MPIR_Bcast_tune_inter_node_helper_MV2(tmp_buf, nbytes, MPI_BYTE,
                                                          root, comm_ptr, errflag);
            } else {
                mpi_errno =
                    MPIR_Bcast_tune_inter_node_helper_MV2(buffer, count, datatype, root,
                                                          comm_ptr, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* We are now done with the inter-node phase */
            if (comm_ptr->dev.ch.intra_node_done == 0) {

                if (!is_contig || !is_homogeneous) {
                    mpi_errno = MV2_Bcast_intra_node_function(tmp_buf, nbytes,
                                                              MPI_BYTE, INTRA_NODE_ROOT, shmem_commptr,
                                                              errflag);
                } else {
                    mpi_errno = MV2_Bcast_intra_node_function(buffer, count,
                                                              datatype, INTRA_NODE_ROOT, shmem_commptr,
                                                              errflag);

                }
            }
        } 
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        if (!is_contig || !is_homogeneous) {
            /* Finishing up... */
            if (rank != root) {
                position = 0;
                mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, buffer,
                                             count, datatype);
            }
        }
    } else {
        /* We use Knomial for intra node */
        MV2_Bcast_intra_node_function = &MPIR_Knomial_Bcast_intra_node_MV2;
        if (mv2_enable_shmem_bcast == 0) {
            /* Fall back to non-tuned version */
            MPIR_Bcast_intra_MV2(buffer, count, datatype, root, comm_ptr, errflag);
        } else {
            mpi_errno = MV2_Bcast_function(buffer, count, datatype, root,
                                           comm_ptr, errflag);

        }
    }

    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

  fn_exit:
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPIR_BCAST_TUNE_INTRA_MV2);
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;

  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_MV2(void *buf, int count, MPI_Datatype datatype,
                   int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME){
	char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[Bcast rank = %d host = %s count = %d  SA=%d] Func: MPIR_Bcast_MV2\n", comm_ptr->rank,hostname,count,security_approach);fflush(stdout);}
#endif        

    int mpi_errno = MPI_SUCCESS;
    MPIR_T_PVAR_COMM_COUNTER_INC(MV2,mv2_coll_bcast_subcomm,1,comm_ptr);
#ifdef _ENABLE_CUDA_
    MPI_Aint datatype_extent;
    MPID_Datatype_get_extent_macro(datatype, datatype_extent);
    MPIDI_msg_sz_t nbytes = 0; 
    nbytes = (MPIDI_msg_sz_t) (count) * (datatype_extent);
    int mem_type = 0;
    int rank = comm_ptr->rank;
    //    printf("%d at F3\n", rank);
    if (rdma_enable_cuda) {
        mem_type = is_device_buffer(buf);
    }

    if (rdma_enable_cuda && mem_type &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_bcast_naive_limit)) {
        if (rank == root) {
            mpi_errno = cuda_stage_alloc(&buf, count * datatype_extent,
                                         NULL, 0, mem_type, 0, 0);
        } else {
            mpi_errno = cuda_stage_alloc(NULL, 0, &buf, count * datatype_extent, 0, 1, 0);
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */

  /*  int rank = comm_ptr->rank;

    printf("MPIR_Bcast_tune_intra_MV2, rank =%d, security_approach=%d\n",rank, security_approach);
    fflush(stdout);*/


    if (mv2_use_old_bcast == 0) {
        /* Use the new tuned bcast */
	if (mv2_use_indexed_tuning || mv2_use_indexed_bcast_tuning) {
	    mpi_errno = MPIR_Bcast_index_tuned_intra_MV2(buf, count, datatype,
						  root, comm_ptr, errflag);
	}
	else {
	    mpi_errno = MPIR_Bcast_tune_intra_MV2(buf, count, datatype,
						  root, comm_ptr, errflag);
	}
    } else {
        /* Use the previous tuned bcast */
        mpi_errno = MPIR_Bcast_intra_MV2(buf, count, datatype, root, comm_ptr, errflag);
    }
    comm_ptr->dev.ch.intra_node_done = 0;
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && mem_type &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_bcast_naive_limit)) {
        if (rank == root) {
            cuda_stage_free(&buf, NULL, 0, mem_type, 0);
        } else {
            cuda_stage_free(NULL, &buf, count * datatype_extent, 0, mem_type);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_ */
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
