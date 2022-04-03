/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
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

#include "mpiimpl.h"
#include "coll_shmem.h"
#include "common_tuning.h"
#include "reduce_tuning.h"

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_binomial);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_redscat_gather);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_shmem);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_knomial);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_zcpy);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_binomial);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_redscat_gather);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_shmem);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_knomial);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_zcpy);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_subcomm);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_binomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_redscat_gather_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_two_level_helper_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_knomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_zcpy_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_binomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_redscat_gather_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_two_level_helper_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_knomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_zcpy_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_binomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_redscat_gather_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_two_level_helper_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_knomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_zcpy_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_binomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_redscat_gather_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_two_level_helper_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_knomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_zcpy_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_count_recv);


int (*MV2_Reduce_function)(const void *sendbuf,
                           void *recvbuf,
                           int count,
                           MPI_Datatype datatype,
                           MPI_Op op,
                           int root,
                           MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)=NULL;

int (*MV2_Reduce_intra_function)(const void *sendbuf,
                                 void *recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 int root,
                                 MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)=NULL;

static int (*MPIR_Rank_list_mapper)(MPID_Comm *, int)=NULL;
/* This function implements a binomial tree reduce.

   Cost = lgp.alpha + n.lgp.beta + n.lgp.gamma
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_binomial_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_binomial_MV2(const void *sendbuf,
                                    void *recvbuf,
                                    int count,
                                    MPI_Datatype datatype,
                                    MPI_Op op,
                                    int root,
                                    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,reduce,binomial);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    int comm_size, rank, is_commutative;
    int root_rank_list_index, i;
    int mask, relrank, source, lroot;
    MPI_User_function *uop;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf = NULL, *tmp_rcv_buf = NULL;
    MPID_Op *op_ptr;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(2);
    MPIU_THREADPRIV_DECL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_binomial, 1);

    if (count == 0)
    { 
        MPIR_TIMER_END(coll,reduce,binomial);
        return MPI_SUCCESS;
    }

    comm_size = comm_ptr->local_size;

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_GET;
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    /* Create a temporary buffer */

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
            is_commutative = 0;
        else
            is_commutative = 1;

#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif
        if (op_ptr->language == MPID_LANG_C)
            uop = (MPI_User_function *) op_ptr->function.c_function;
        else
            uop = (MPI_User_function *) op_ptr->function.f77_function;
    }

    if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1 ||
            comm_ptr->dev.ch.is_blocked == 1) {
        rank = comm_ptr->rank;
        MPIR_Rank_list_mapper = &Bunch_Rank_list_mapper;
        root_rank_list_index = root;
    } else {
        /* my index on rank_list */
        rank = comm_ptr->dev.ch.rank_list_index;
        MPIU_Assert(rank >= 0);
        MPIR_Rank_list_mapper = &Cyclic_Rank_list_mapper;

        /* now find the root's index on rank_list */
        for (i = 0; i < comm_size; i++) {
            if (comm_ptr->dev.ch.rank_list[i] == root) 
                break;
        }
        root_rank_list_index = i;
    }

    /* I think this is the worse case, so we can avoid an assert() 
     * inside the for loop */
    /* should be buf+{this}? */
    MPIU_Ensure_Aint_fits_in_pointer(count * MPIR_MAX(extent, true_extent));

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)) * 2, mpi_errno,
                        "temporary buffer");

    /* If I'm not the root, then my recvbuf may not be valid, therefore
     * use a temporary buffer */
    if (rank != root_rank_list_index) {
        if (sendbuf == MPI_IN_PLACE) {
            tmp_rcv_buf  = recvbuf;
        }
        recvbuf = (void *) ((char *)(tmp_buf + count * (MPIR_MAX(extent, true_extent))) - true_lb);
        if (sendbuf == MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(tmp_rcv_buf, count, datatype, recvbuf,
                    count, datatype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    if ((sendbuf != MPI_IN_PLACE) && (sendbuf != recvbuf)) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                   count, datatype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* This code is from MPICH-1. */

    /* Here's the algorithm.  Relative to the root, look at the bit pattern in 
       my rank.  Starting from the right (lsb), if the bit is 1, send to 
       the node with that bit zero and exit; if the bit is 0, receive from the
       node with that bit set and combine (as long as that node is within the
       group)

       Note that by receiving with source selection, we guarentee that we get
       the same bits with the same input.  If we allowed the parent to receive 
       the children in any order, then timing differences could cause different
       results (roundoff error, over/underflows in some cases, etc).

       Because of the way these are ordered, if root is 0, then this is correct
       for both commutative and non-commutitive operations.  If root is not
       0, then for non-commutitive, we use a root of zero and then send
       the result to the root.  To see this, note that the ordering is
       mask = 1: (ab)(cd)(ef)(gh)            (odds send to evens)
       mask = 2: ((ab)(cd))((ef)(gh))        (3,6 send to 0,4)
       mask = 4: (((ab)(cd))((ef)(gh)))      (4 sends to 0)

       Comments on buffering.  
       If the datatype is not contiguous, we still need to pass contiguous 
       data to the user routine.  
       In this case, we should make a copy of the data in some format, 
       and send/operate on that.

       In general, we can't use MPI_PACK, because the alignment of that
       is rather vague, and the data may not be re-usable.  What we actually
       need is a "squeeze" operation that removes the skips.
     */
    mask = 0x1;
    if (is_commutative)
        lroot = root_rank_list_index;
    else
        lroot = 0;
    relrank = (rank - lroot + comm_size) % comm_size;

    while ( /*(mask & relrank) == 0 && */ mask < comm_size) {
        /* Receive */
        if ((mask & relrank) == 0) {
            source = (relrank | mask);
            if (source < comm_size) {
                source = (source + lroot) % comm_size;
                MPIR_PVAR_INC(reduce, binomial, recv, count, datatype);
                mpi_errno = MPIC_Recv(tmp_buf, count,
                        datatype,MPIR_Rank_list_mapper(comm_ptr, source),
                        MPIR_REDUCE_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* The sender is above us, so the received buffer must be
                   the second argument (in the noncommutative case). */
                if (is_commutative) {
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                        count, datatype, uop);
                    } else
#endif
                        (*uop) (tmp_buf, recvbuf, &count, &datatype);
                } else {
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                        count, datatype, uop);
                    } else
#endif
                        (*uop) (recvbuf, tmp_buf, &count, &datatype);
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                               recvbuf, count, datatype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
            }
        } else {
            /* I've received all that I'm going to.  Send my result to 
               my parent */
            source = ((relrank & (~mask)) + lroot) % comm_size;
            MPIR_PVAR_INC(reduce, binomial, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count, datatype,
                                     MPIR_Rank_list_mapper(comm_ptr, source),
                                     MPIR_REDUCE_TAG, comm_ptr, errflag);
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

    if (!is_commutative && (root_rank_list_index != 0)) {
        if (rank == 0) {
            MPIR_PVAR_INC(reduce, binomial, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count, datatype,
                    MPIR_Rank_list_mapper(comm_ptr, root_rank_list_index), MPIR_REDUCE_TAG,
                    comm_ptr, errflag);
        } else if (rank == root_rank_list_index) {
            MPIR_PVAR_INC(reduce, binomial, recv, count, datatype);
            mpi_errno = MPIC_Recv(recvbuf, count, datatype,
                    MPIR_Rank_list_mapper(comm_ptr, 0), MPIR_REDUCE_TAG,
                    comm_ptr, &status, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* FIXME does this need to be checked after each uop invocation for
       predefined operators? */
    /* --BEGIN ERROR HANDLING-- */
    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
        goto fn_fail;
    }
    /* --END ERROR HANDLING-- */

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,reduce,binomial);  
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* An implementation of Rabenseifner's reduce algorithm (see
   http://www.hlrs.de/organization/par/services/models/mpi/myreduce.html).

   This algorithm implements the reduce in two steps: first a
   reduce-scatter, followed by a gather to the root. A
   recursive-halving algorithm (beginning with processes that are
   distance 1 apart) is used for the reduce-scatter, and a binomial tree
   algorithm is used for the gather. The non-power-of-two case is
   handled by dropping to the nearest lower power-of-two: the first
   few odd-numbered processes send their data to their left neighbors
   (rank-1), and the reduce-scatter happens among the remaining
   power-of-two processes. If the root is one of the excluded
   processes, then after the reduce-scatter, rank 0 sends its result to
   the root and exits; the root now acts as rank 0 in the binomial tree
   algorithm for gather.

   For the power-of-two case, the cost for the reduce-scatter is 
   lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma. The cost for the
   gather to root is lgp.alpha + n.((p-1)/p).beta. Therefore, the
   total cost is:
   Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma

   For the non-power-of-two case, assuming the root is not one of the
   odd-numbered processes that get excluded in the reduce-scatter,
   Cost = (2.floor(lgp)+1).alpha + (2.((p-1)/p) + 1).n.beta + 
           n.(1+(p-1)/p).gamma
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_redscat_gather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_redscat_gather_MV2(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          int root,
                                          MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,reduce,redscat_gather);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size, rank, pof2, rem, newrank;
    int mask, *cnts, *disps, i, j, send_idx = 0;
    int recv_idx, last_idx = 0, newdst;
    int dst, send_cnt, recv_cnt, newroot, newdst_tree_root, newroot_tree_root;
    MPI_User_function *uop;
    int is_commutative;
    int root_rank_list_index;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPID_Op *op_ptr;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(4);
    MPIU_THREADPRIV_DECL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_redscat_gather, 1);

    comm_size = comm_ptr->local_size;

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_GET;
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    /* Create a temporary buffer */

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
            is_commutative = 0;
        else
            is_commutative = 1;

#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif
        if (op_ptr->language == MPID_LANG_C)
            uop = (MPI_User_function *) op_ptr->function.c_function;
        else
            uop = (MPI_User_function *) op_ptr->function.f77_function;
    }

    if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1 ||
            comm_ptr->dev.ch.is_blocked == 1) {
        rank = comm_ptr->rank;
        MPIR_Rank_list_mapper = &Bunch_Rank_list_mapper;
        root_rank_list_index = root;
    } else {
        /* my index on rank_list */
        rank = comm_ptr->dev.ch.rank_list_index;
        MPIU_Assert(rank >= 0);
        MPIR_Rank_list_mapper = &Cyclic_Rank_list_mapper;

        /* now find the root's index on rank_list */
        for (i = 0; i < comm_size; i++) {
            if (comm_ptr->dev.ch.rank_list[i] == root) 
                break;
        }
        root_rank_list_index = i;
    }

    /* I think this is the worse case, so we can avoid an assert() 
     * inside the for loop */
    /* should be buf+{this}? */
    MPIU_Ensure_Aint_fits_in_pointer(count * MPIR_MAX(extent, true_extent));

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");
    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* If I'm not the root, then my recvbuf may not be valid, therefore
       I have to allocate a temporary one */
    if (rank != root_rank_list_index && sendbuf != MPI_IN_PLACE) {
        MPIU_CHKLMEM_MALLOC(recvbuf, void *,
                            count * (MPIR_MAX(extent, true_extent)),
                            mpi_errno, "receive buffer");
        recvbuf = (void *) ((char *) recvbuf - true_lb);
    }

    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                   count, datatype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send their data to
       (rank-1). These odd-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. 

       Note that in MPI_Allreduce we have the even-numbered processes
       send data to odd-numbered processes. That is better for
       non-commutative operations because it doesn't require a
       buffer copy. However, for MPI_Reduce, the most common case
       is commutative operations with root=0. Therefore we want
       even-numbered processes to participate the computation for
       the root=0 case, in order to avoid an extra send-to-root
       communication after the reduce-scatter. In MPI_Allreduce it
       doesn't matter because all processes must get the result. */

    if (rank < 2 * rem) {
        if (rank % 2 != 0) {    /* odd */
            MPIR_PVAR_INC(reduce, redscat_gather, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count,
                                     datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_REDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
               process does not pariticipate in recursive
               doubling */
            newrank = -1;
        } else {                /* even */
            MPIR_PVAR_INC(reduce, redscat_gather, recv, count, datatype);
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                     datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_REDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. */
            /* This algorithm is used only for predefined ops
               and predefined ops are always commutative. */
#ifdef HAVE_CXX_BINDING
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                count, datatype, uop);
            } else
#endif
                (*uop) (tmp_buf, recvbuf, &count, &datatype);

            /* change the rank */
            newrank = rank / 2;
        }
    } else                      /* rank >= 2*rem */
        newrank = rank - rem;

    /* for the reduce-scatter, calculate the count that
       each process receives and the displacement within
       the buffer */

    /* We allocate these arrays on all processes, even if newrank=-1,
       because if root is one of the excluded processes, we will
       need them on the root later on below. */
    MPIU_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof (int), mpi_errno, "counts");
    MPIU_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof (int), mpi_errno,
                        "displacements");

    if (newrank != -1) {
        for (i = 0; i < (pof2 - 1); i++)
            cnts[i] = count / pof2;
        cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

        disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

/*                    printf("Rank %d, send_idx %d, recv_idx %d, send_cnt %d, recv_cnt %d, last_idx %d\n", newrank, send_idx, recv_idx,
                  send_cnt, recv_cnt, last_idx);
*/
            /* Send data from recvbuf. Recv into tmp_buf */
            MPIR_PVAR_INC(reduce, redscat_gather, send, send_cnt, datatype);
            MPIR_PVAR_INC(reduce, redscat_gather, recv, recv_cnt, datatype);
            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                         disps[send_idx] * extent,
                                         send_cnt, datatype,
                                         MPIR_Rank_list_mapper(comm_ptr, dst), 
                                         MPIR_REDUCE_TAG,
                                         (char *) tmp_buf +
                                         disps[recv_idx] * extent,
                                         recv_cnt, datatype, 
                                         MPIR_Rank_list_mapper(comm_ptr, dst),
                                         MPIR_REDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* tmp_buf contains data received in this step.
               recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
               and predefined ops are always commutative. */
#ifdef HAVE_CXX_BINDING
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) ((char *) tmp_buf +
                                                disps[recv_idx] * extent,
                                                (char *) recvbuf +
                                                disps[recv_idx] * extent,
                                                recv_cnt, datatype, uop);
            } else
#endif
                (*uop) ((char *) tmp_buf + disps[recv_idx] * extent,
                        (char *) recvbuf + disps[recv_idx] * extent,
                        &recv_cnt, &datatype);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
               because the value is needed in the gather
               step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }
    }

    /* now do the gather to root */

    /* Is root one of the processes that was excluded from the
       computation above? If so, send data from newrank=0 to
       the root and have root take on the role of newrank = 0 */

    if (root_rank_list_index < 2 * rem) {
        if (root_rank_list_index % 2 != 0) {
            if (rank == root_rank_list_index) { /* recv */
                /* initialize the arrays that weren't initialized */
                for (i = 0; i < (pof2 - 1); i++)
                    cnts[i] = count / pof2;
                cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

                disps[0] = 0;
                for (i = 1; i < pof2; i++)
                    disps[i] = disps[i - 1] + cnts[i - 1];

                MPIR_PVAR_INC(reduce, redscat_gather, recv, cnts[0], datatype);
                mpi_errno = MPIC_Recv(recvbuf, cnts[0], datatype,
                                         MPIR_Rank_list_mapper(comm_ptr, 0), 
                                         MPIR_REDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                newrank = 0;
                send_idx = 0;
                last_idx = 2;
            } else if (newrank == 0) {  /* send */
                MPIR_PVAR_INC(reduce, redscat_gather, send, cnts[0], datatype);
                mpi_errno = MPIC_Send(recvbuf, cnts[0], datatype,
                                         MPIR_Rank_list_mapper(comm_ptr, root_rank_list_index), 
                                         MPIR_REDUCE_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                newrank = -1;
            }
            newroot = 0;
        } else
            newroot = root_rank_list_index / 2;
    } else
        newroot = root_rank_list_index - rem;

    if (newrank != -1) {
        j = 0;
        mask = 0x1;
        while (mask < pof2) {
            mask <<= 1;
            j++;
        }
        mask >>= 1;
        j--;
        while (mask > 0) {
            newdst = newrank ^ mask;

            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 : newdst + rem;
            /* if root is playing the role of newdst=0, adjust for
               it */
            if ((newdst == 0) && (root_rank_list_index < 2 * rem) && (root_rank_list_index % 2 != 0))
                dst = root_rank_list_index;

            /* if the root of newdst's half of the tree is the
               same as the root of newroot's half of the tree, send to
               newdst and exit, else receive from newdst. */

            newdst_tree_root = newdst >> j;
            newdst_tree_root <<= j;

            newroot_tree_root = newroot >> j;
            newroot_tree_root <<= j;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            if (newdst_tree_root == newroot_tree_root) {
                /* send and exit */
                /* printf("Rank %d, send_idx %d, send_cnt %d, last_idx %d\n", newrank, send_idx, send_cnt, last_idx);
                   fflush(stdout); */
                /* Send data from recvbuf. Recv into tmp_buf */
                MPIR_PVAR_INC(reduce, redscat_gather, send, send_cnt, datatype);
                mpi_errno = MPIC_Send((char *) recvbuf +
                                         disps[send_idx] * extent,
                                         send_cnt, datatype,
                                         MPIR_Rank_list_mapper(comm_ptr, dst), 
                                         MPIR_REDUCE_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                break;
            } else {
                /* recv and continue */
                /* printf("Rank %d, recv_idx %d, recv_cnt %d, last_idx %d\n", newrank, recv_idx, recv_cnt, last_idx);
                   fflush(stdout); */
                MPIR_PVAR_INC(reduce, redscat_gather, recv, recv_cnt, datatype);
                mpi_errno = MPIC_Recv((char *) recvbuf +
                                         disps[recv_idx] * extent,
                                         recv_cnt, datatype, 
                                         MPIR_Rank_list_mapper(comm_ptr, dst),
                                         MPIR_REDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            j--;
        }
    }

    /* FIXME does this need to be checked after each uop invocation for
       predefined operators? */
    /* --BEGIN ERROR HANDLING-- */
    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
        goto fn_fail;
    }
    /* --END ERROR HANDLING-- */

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,reduce,redscat_gather);  
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Shmem_Reduce_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_shmem_MV2(const void *sendbuf,
                          void *recvbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,reduce,shmem);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, stride, local_rank, local_size, shmem_comm_rank;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    char *shmem_buf = NULL;
    void *local_buf = NULL;
    char *tmp_buf = NULL;
    int buf_allocated = 0;
    MPI_Aint true_lb, true_extent, extent;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_shmem, 1);

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);

    local_rank = comm_ptr->rank;
    local_size = comm_ptr->local_size;
    shmem_comm_rank = comm_ptr->dev.ch.shmem_comm_rank;

    if (sendbuf != MPI_IN_PLACE && local_rank == 0) {
	/* if local_rank == 0 and not root then the recvbuf may not be valid*/
	if (sendbuf == recvbuf || recvbuf == NULL) {
	    tmp_buf = recvbuf;
	    recvbuf = MPIU_Malloc(count * MPIR_MAX(extent, true_extent));
	    buf_allocated = 1;
	}
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                                   datatype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    if (local_size == 0) {
        /* Only one process. So, return */
        goto fn_exit;
    }

    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            } else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
            }
#if defined(HAVE_CXX_BINDING)
        }
#endif                          /* defined(HAVE_CXX_BINDING) */
    }

    if (local_rank == 0) {
        MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                          shmem_comm_rank, (void *) &shmem_buf);
        for (i = 1; i < local_size; i++) {
            local_buf = (char *) shmem_buf + stride * i;
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (local_buf, recvbuf,
                                                count, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (local_buf, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
        }
        MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size,
                                                local_rank, shmem_comm_rank);
    } else {
        MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                          shmem_comm_rank, (void *) &shmem_buf);
        local_buf = (char *) shmem_buf + stride * local_rank;
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype,
                                   local_buf, count, datatype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size, local_rank,
                                                shmem_comm_rank);
    }

    if (buf_allocated) {
	MPIU_Free(recvbuf);
	recvbuf = tmp_buf;
    }
  fn_exit:
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,reduce,shmem);  
    return mpi_errno;
  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_kinomial_trace
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_knomial_trace(int root, int mv2_reduce_knomial_factor,  
        MPID_Comm *comm_ptr, int *dst, int *expected_send_count,
        int *expected_recv_count, int **src_array)
{
    int mask=0x1, k, comm_size, src, rank, relative_rank, lroot=0;
    int orig_mask=0x1; 
    int recv_iter=0, send_iter=0;
    int *knomial_reduce_src_array=NULL;
    rank      = comm_ptr->rank;
    comm_size = comm_ptr->local_size;

    lroot = root;
    relative_rank = (rank - lroot + comm_size) % comm_size;

    /* First compute to whom we need to send data */ 
    while (mask < comm_size) {
        if (relative_rank % (mv2_reduce_knomial_factor*mask)) {
            *dst = relative_rank/(mv2_reduce_knomial_factor*mask)*
                (mv2_reduce_knomial_factor*mask)+root;
            if (*dst >= comm_size) {
                *dst -= comm_size;
            }
            send_iter++;
            break;
        }
        mask *= mv2_reduce_knomial_factor;
    }
    mask /= mv2_reduce_knomial_factor;

    /* Now compute how many children we have in the knomial-tree */ 
    orig_mask = mask; 
    while (mask > 0) {
        for(k=1;k<mv2_reduce_knomial_factor;k++) {
            if (relative_rank + mask*k < comm_size) {
                recv_iter++;
            }
        }
        mask /= mv2_reduce_knomial_factor;
    }

    /* Finally, fill up the src array */ 
    if(recv_iter > 0) { 
        knomial_reduce_src_array = MPIU_Malloc(sizeof(int)*recv_iter); 
    } 

    mask = orig_mask; 
    recv_iter=0; 
    while (mask > 0) {
        for(k=1;k<mv2_reduce_knomial_factor;k++) {
            if (relative_rank + mask*k < comm_size) {
                src = rank + mask*k;
                if (src >= comm_size) {
                    src -= comm_size;
                }
                knomial_reduce_src_array[recv_iter++] = src;
            }
        }
        mask /= mv2_reduce_knomial_factor;
    }

    *expected_recv_count = recv_iter;
    *expected_send_count = send_iter;
    *src_array = knomial_reduce_src_array; 
    return 0; 
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_kinomial_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_knomial_MV2 (
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        int mv2_reduce_knomial_factor,
        MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,reduce,knomial);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int rank, is_commutative;
    int src, k;
    MPI_User_function *uop;
    MPID_Request *send_request = NULL;
    int index=0;
    MPI_Aint true_lb, true_extent, extent;
    MPI_Status status; 
    int recv_iter=0, dst, expected_send_count, expected_recv_count;
    int *src_array=NULL;
    MPID_Op *op_ptr;
    void **tmp_buf=NULL;
    MPID_Request **requests=NULL;
    MPI_Request *mpi_reqs =NULL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(1);
    MPIU_THREADPRIV_DECL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_knomial, 1);

    if (count == 0) 
    {
      MPIR_TIMER_END(coll,reduce,knomial);
      return MPI_SUCCESS;
    }

    rank = comm_ptr->rank;

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_GET;
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    /* Create a temporary buffer */

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op%16 - 1];
    }
    else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
            is_commutative = 0;
        else
            is_commutative = 1;

#ifdef HAVE_CXX_BINDING            
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        }
        else
#endif
            if (op_ptr->language == MPID_LANG_C)
                uop = (MPI_User_function *) op_ptr->function.c_function;
            else
                uop = (MPI_User_function *) op_ptr->function.f77_function;
    }

    /* I think this is the worse case, so we can avoid an assert() 
     * inside the for loop */
    /* should be buf+{this}? */
    MPIU_Ensure_Aint_fits_in_pointer(count * MPIR_MAX(extent, true_extent));

    if (rank != root && sendbuf != MPI_IN_PLACE) {
        MPIU_CHKLMEM_MALLOC(recvbuf, void *,
                count*(MPIR_MAX(extent,true_extent)),
                mpi_errno, "receive buffer");
        recvbuf = (void *)((char*)recvbuf - true_lb);
    }

    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                count, datatype);
        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
    }



    MPIR_Reduce_knomial_trace(root, mv2_reduce_knomial_factor, comm_ptr, 
           &dst, &expected_send_count, &expected_recv_count, &src_array);

    if(expected_recv_count > 0 ) {
        tmp_buf  = MPIU_Malloc(sizeof(void *)*expected_recv_count);
        requests = (MPID_Request**) MPIU_Malloc(sizeof(MPID_Request*)*expected_recv_count);
        mpi_reqs = (MPI_Request*) MPIU_Malloc(sizeof(MPI_Request)*expected_recv_count);
        for(k=0; k < expected_recv_count; k++ ) {
            tmp_buf[k] = MPIU_Malloc(count*(MPIR_MAX(extent,true_extent)));
            tmp_buf[k] = (void *)((char*)tmp_buf[k] - true_lb);
        }

        while(recv_iter  < expected_recv_count) {
            src = src_array[expected_recv_count - (recv_iter+1)];

            MPIR_PVAR_INC(reduce, knomial, recv, count, datatype);
            mpi_errno = MPIC_Irecv (tmp_buf[recv_iter], count, datatype ,src,
                    MPIR_REDUCE_TAG, comm_ptr, &requests[recv_iter]);
            /* Convert the MPID_Request objects to MPI_Request objects */
            mpi_reqs[recv_iter] = requests[recv_iter]->handle;
            recv_iter++;

            if (mpi_errno) {
                /* for communication errors, just record the error but continue*/
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        recv_iter=0;
        while(recv_iter < expected_recv_count) {
            mpi_errno = PMPI_Waitany(expected_recv_count, mpi_reqs, &index,
                    &status);
            recv_iter++;
            if (mpi_errno) {
                /* for communication errors, just record the error but
                 * continue*/
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            if (is_commutative) {
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn)( tmp_buf[index], recvbuf,
                            count, datatype, uop );
                }
                else
#endif
                    (*uop)(tmp_buf[index], recvbuf, &count, &datatype);
            }
        }

        for(k=0; k < expected_recv_count; k++ ) {
            MPIU_Free(tmp_buf[k]);
        }
        MPIU_Free(tmp_buf);
        MPIU_Free(requests);
        MPIU_Free(mpi_reqs);
    }

    if(src_array != NULL) { 
        MPIU_Free(src_array);
    } 

    if(rank != root) {
        MPIR_PVAR_INC(reduce, knomial, send, count, datatype);
        mpi_errno = MPIC_Isend(recvbuf,count, datatype, dst,
                MPIR_REDUCE_TAG,comm_ptr,&send_request,
                errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue
             * */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        mpi_errno = MPIC_Waitall(1, &send_request, &status, errflag);
    }

    /* FIXME does this need to be checked after each uop invocation for
     * predefined operators? */
    /* --BEGIN ERROR HANDLING-- */
    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
        goto fn_fail;
    }
    /* --END ERROR HANDLING-- */

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,reduce,knomial);  
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_inter_knomial_wrapper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_inter_knomial_wrapper_MV2 (
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
   int mpi_errno = MPI_SUCCESS;
   mpi_errno = MPIR_Reduce_knomial_MV2 (sendbuf, recvbuf, count, datatype, op,
               root, mv2_reduce_inter_knomial_factor, comm_ptr, errflag);

   if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
   }

fn_fail:
   return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_intra_knomial_wrapper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_intra_knomial_wrapper_MV2 (
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
   int mpi_errno = MPI_SUCCESS;
   mpi_errno = MPIR_Reduce_knomial_MV2 (sendbuf, recvbuf, count, datatype, op,
               root, mv2_reduce_inter_knomial_factor, comm_ptr, errflag);

   if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
   }

fn_fail:
   return mpi_errno;
}

#ifdef CHANNEL_MRAIL_GEN2
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_Zcpy_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_Zcpy_MV2(const void *sendbuf,
                                     void *recvbuf,
                                     int count,
                                     MPI_Datatype datatype,
                                     MPI_Op op,
                                     int root,
                                     MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,reduce,zcpy);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int my_rank, local_rank;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    void *in_buf = NULL, *out_buf = NULL;
    MPI_Aint true_lb, true_extent, extent;
    int stride = 0;
    int dst, expected_send_count, expected_recv_count;
    int *src_array=NULL;
    int pseudo_root = 0;
    static int fn_call=0;  
    MPI_Status status; 

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_zcpy, 1);

    fn_call++; 

    my_rank = comm_ptr->rank;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;

    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);

    if (local_rank == 0) {
        int leader_of_psuedo_root, leader_psuedo_root; 
        shmem_info_t *shmem_info = NULL;
        leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
 
        leader_of_psuedo_root = comm_ptr->dev.ch.leader_map[pseudo_root];
        leader_psuedo_root = comm_ptr->dev.ch.leader_rank[leader_of_psuedo_root];

        shmem_info = comm_ptr->dev.ch.shmem_info;
        /* If the knomial_factor requested for this specific bcast 
         * is the same as the one that we have used before, the communication
         * tree is already setup and cached. No need to do it again
         */
        if((shmem_info)->reduce_knomial_factor != mv2_reduce_zcopy_inter_knomial_factor) {
             MPIR_Reduce_knomial_trace(leader_psuedo_root, 
                               mv2_reduce_zcopy_inter_knomial_factor,
                               leader_commptr, &dst, &expected_send_count,
                               &expected_recv_count, &src_array);
             (shmem_info)->reduce_exchange_rdma_keys = 1;
        }
    }

    if (sendbuf != MPI_IN_PLACE) {
        in_buf = (void *)sendbuf;
    } else {
        in_buf = recvbuf;
    } 
    
    mpi_errno = mv2_shm_zcpy_reduce(shmem_commptr->dev.ch.shmem_info, 
                                  in_buf, &out_buf, count, stride, 
                                  datatype, op,
                                  root,
                                  expected_recv_count, src_array, 
                                  expected_send_count, dst, 
                                  mv2_reduce_zcopy_inter_knomial_factor, 
                                  comm_ptr, 
                                  errflag);
    if (mpi_errno) {
        /* for communication errors, just record the error
         * but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    if(my_rank == 0 && root == my_rank) { 
         MPIU_Memcpy(recvbuf, out_buf, stride);
    } else {  
        /* Send the message to the root if the root is not rank0 */ 
        if ((my_rank == 0) && (root != my_rank)) { 
            MPIR_PVAR_INC(reduce, zcpy, send, count, datatype);
            mpi_errno = MPIC_Send(out_buf, count, datatype, root,
                                     MPIR_REDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error 
                 * but continue */
                fprintf(stderr,"%d send to %d failed, mpi_errno %d\n", comm_ptr->rank, root, mpi_errno); 
                
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        if ((my_rank != 0) && (root == my_rank)) {
            MPIR_PVAR_INC(reduce, zcpy, recv, count, datatype);
            mpi_errno = MPIC_Recv(recvbuf, count, datatype,
                                     pseudo_root,
                                     MPIR_REDUCE_TAG, comm_ptr,
                                     &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but
                 * continue */
                fprintf(stderr,"%d send to %d failed, mpi_errno %d\n", comm_ptr->rank, root, mpi_errno); 
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } 

    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,reduce,zcpy);  
    return mpi_errno;
}
#endif /* CHANNEL_MRAIL_GEN2 */




#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_two_level_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_two_level_helper_MV2(const void *sendbuf,
                                     void *recvbuf,
                                     int count,
                                     MPI_Datatype datatype,
                                     MPI_Op op,
                                     int root,
                                     MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int my_rank, total_size, local_rank, local_size;
    int leader_comm_rank = -1, leader_comm_size = 0;
    MPI_Comm shmem_comm, leader_comm;
    int leader_root, leader_of_root;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    void *in_buf = NULL, *out_buf = NULL, *tmp_buf = NULL;
    MPI_Aint true_lb, true_extent, extent;
    MPID_Op *op_ptr;
    int is_commutative = 0, stride = 0;
    int intra_node_root=0; 
    MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
    my_rank = comm_ptr->rank;
    total_size = comm_ptr->local_size;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;

    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
    } else {
        MPID_Op_get_ptr(op, op_ptr)
            if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);

    if (local_size == total_size) {
        /* First handle the case where there is only one node */
        if (comm_ptr->dev.ch.shmem_coll_ok == 1 &&
            stride <= mv2_coll_param.shmem_intra_reduce_msg &&
            mv2_enable_shmem_reduce && is_commutative == 1) {
            if (local_rank == 0 ) {
                MPIU_CHKLMEM_MALLOC(tmp_buf, void *, count *
                                    (MPIR_MAX(extent, true_extent)),
                                    mpi_errno, "receive buffer");
                tmp_buf = (void *) ((char *) tmp_buf - true_lb);
            }

            if (sendbuf != MPI_IN_PLACE) {
                in_buf = (void *)sendbuf;
            } else {
                in_buf = recvbuf;
            }

            if (local_rank == 0) { 
                 if( my_rank != root) {
                     out_buf = tmp_buf;
                 } else { 
                     out_buf = recvbuf; 
                     if(in_buf == out_buf) { 
                        in_buf = MPI_IN_PLACE; 
                        out_buf = recvbuf; 
                     } 
                 } 
            } else {
               if (my_rank != root) {
                   in_buf  = (void *)sendbuf;
                   out_buf = NULL;
               }
               else {
                   if (sendbuf !=  MPI_IN_PLACE) {
                       in_buf  = (void *)sendbuf;
                       out_buf = (void *)recvbuf;
                   }
                   else {
                       in_buf  = (void *)recvbuf;
                       out_buf = (void *)recvbuf;
                   }
               }
            }

	    if (count * (MPIR_MAX(extent, true_extent)) < mv2_g_shmem_coll_max_msg_size) {
		mpi_errno = MPIR_Reduce_shmem_MV2(in_buf, out_buf, count,
						  datatype, op,
						  0, shmem_commptr, errflag);
	    }
	    else {
		mpi_errno = MPIR_Reduce_intra_knomial_wrapper_MV2(in_buf, out_buf, count,
								  datatype, op,
								  0, shmem_commptr, errflag);
	    }
	    
            if (local_rank == 0 && root != my_rank) {
                MPIR_PVAR_INC(reduce, two_level_helper, send, count, datatype);
                mpi_errno = MPIC_Send(out_buf, count, datatype, root,
                                         MPIR_REDUCE_TAG, comm_ptr, errflag);
            }
            if ((local_rank != 0) && (root == my_rank)) {
                MPIR_PVAR_INC(reduce, two_level_helper, recv, count, datatype);
                mpi_errno = MPIC_Recv(recvbuf, count, datatype,
                                         leader_of_root, MPIR_REDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);
            }
        } else {
            if(mv2_use_knomial_reduce == 1) { 
                reduce_fn = &MPIR_Reduce_intra_knomial_wrapper_MV2; 
            } else { 
                reduce_fn = &MPIR_Reduce_binomial_MV2; 
            } 
            mpi_errno = reduce_fn(sendbuf, recvbuf, count,
                                  datatype, op,
                                  root, comm_ptr,
                                  errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but
             * continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        /* We are done */
        goto fn_exit;
    }
    
#ifdef CHANNEL_MRAIL_GEN2
    if(mv2_use_slot_shmem_coll &&
       mv2_enable_zcpy_reduce == 1 && 
       stride <= mv2_shm_slot_len && 
       comm_ptr->dev.ch.shmem_coll_ok == 1 &&
       mv2_enable_shmem_reduce && is_commutative == 1){ 
        mpi_errno = MPIR_Reduce_Zcpy_MV2(sendbuf, recvbuf, count,
                                         datatype, op,
                                         root, comm_ptr,
                                         errflag); 
        if (mpi_errno) {
            /* for communication errors, just record the error but
             * continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        /* We are done */
        goto fn_exit;
    } 
#endif /* CHANNEL_MRAIL_GEN2 */

    if (local_rank == 0) {
        leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_comm_rank = leader_commptr->rank;
        leader_comm_size = leader_commptr->local_size;
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, count *
                            (MPIR_MAX(extent, true_extent)),
                            mpi_errno, "receive buffer");
        tmp_buf = (void *) ((char *) tmp_buf - true_lb);
    }
    if (sendbuf != MPI_IN_PLACE) {
        in_buf = (void *)sendbuf;
    } else {
        in_buf = recvbuf;
    }
    if (local_rank == 0) {
        out_buf = tmp_buf;
    } else {
        out_buf = NULL;
    }


    if(local_size > 1) { 
        /* Lets do the intra-node reduce operations, if we have more than one
         * process in the node */

        /*Fix the input and outbuf buffers for the intra-node reduce.
         *Node leaders will have the reduced data in tmp_buf after 
         *this step*/
        if (MV2_Reduce_intra_function == & MPIR_Reduce_shmem_MV2)
        {
            if (comm_ptr->dev.ch.shmem_coll_ok == 1 &&
                mv2_enable_shmem_reduce && is_commutative == 1
		&& (count * (MPIR_MAX(extent, true_extent)) < mv2_g_shmem_coll_max_msg_size)) {
                    mpi_errno = MV2_Reduce_intra_function(in_buf, out_buf, count,
                                      datatype, op,
                                      intra_node_root, shmem_commptr, errflag);
            } else {
                    mpi_errno = MPIR_Reduce_intra_knomial_wrapper_MV2(in_buf, out_buf, count,
                                      datatype, op,
                                      intra_node_root, shmem_commptr, errflag);
            }
        } else {

            mpi_errno = MV2_Reduce_intra_function(in_buf, out_buf, count,
                                      datatype, op,
                                      intra_node_root, shmem_commptr, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but
             * continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    } else { 
        tmp_buf = in_buf; 
    } 

    /* Now work on the inter-leader phase. Data is in tmp_buf */
    if (local_rank == 0 && leader_comm_size > 1) {
        /*The leader of root will have the global reduced data in tmp_buf 
           or recv_buf
           at the end of the reduce */
        if (leader_comm_rank == leader_root) {
            if (my_rank == root) {
                /* I am the root of the leader-comm, and the 
                 * root of the reduce op. So, I will write the 
                 * final result directly into my recvbuf */
                if(tmp_buf != recvbuf) { 
                    in_buf = tmp_buf;
                    out_buf = recvbuf;
                } else { 
                    in_buf = MPI_IN_PLACE; 
                    out_buf = recvbuf; 
                } 
            } else {
                in_buf = MPI_IN_PLACE;
                out_buf = tmp_buf;
            }
        } else {
            in_buf = tmp_buf;
            out_buf = NULL;
        }

        /* inter-leader communication  */
        mpi_errno = MV2_Reduce_function(in_buf, out_buf, count,
                              datatype, op,
                              leader_root, leader_commptr,
                              errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error
             * but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    if (local_size > 1) {
        /* Send the message to the root if the leader is not the
         * root of the reduce operation. The reduced data is in tmp_buf */
        if ((local_rank == 0) && (root != my_rank)
            && (leader_root == leader_comm_rank)) {
            MPIR_PVAR_INC(reduce, two_level_helper, send, count, datatype);
            mpi_errno = MPIC_Send(tmp_buf, count, datatype, root,
                                     MPIR_REDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error 
                 * but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        if ((local_rank != 0) && (root == my_rank)) {
            MPIR_PVAR_INC(reduce, two_level_helper, recv, count, datatype);
            mpi_errno = MPIC_Recv(recvbuf, count, datatype,
                                     leader_of_root,
                                     MPIR_REDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but
                 * continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
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

/* This is the default implementation of reduce. The algorithm is:
   
   Algorithm: MPI_Reduce

   For long messages and for builtin ops and if count >= pof2 (where
   pof2 is the nearest power-of-two less than or equal to the number
   of processes), we use Rabenseifner's algorithm (see 
   http://www.hlrs.de/organization/par/services/models/mpi/myreduce.html ).
   This algorithm implements the reduce in two steps: first a
   reduce-scatter, followed by a gather to the root. A
   recursive-halving algorithm (beginning with processes that are
   distance 1 apart) is used for the reduce-scatter, and a binomial tree
   algorithm is used for the gather. The non-power-of-two case is
   handled by dropping to the nearest lower power-of-two: the first
   few odd-numbered processes send their data to their left neighbors
   (rank-1), and the reduce-scatter happens among the remaining
   power-of-two processes. If the root is one of the excluded
   processes, then after the reduce-scatter, rank 0 sends its result to
   the root and exits; the root now acts as rank 0 in the binomial tree
   algorithm for gather.

   For the power-of-two case, the cost for the reduce-scatter is 
   lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma. The cost for the
   gather to root is lgp.alpha + n.((p-1)/p).beta. Therefore, the
   total cost is:
   Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma

   For the non-power-of-two case, assuming the root is not one of the
   odd-numbered processes that get excluded in the reduce-scatter,
   Cost = (2.floor(lgp)+1).alpha + (2.((p-1)/p) + 1).n.beta + 
           n.(1+(p-1)/p).gamma

   For short messages, user-defined ops, and count < pof2, we use a
   binomial tree algorithm for both short and long messages. 

   Cost = lgp.alpha + n.lgp.beta + n.lgp.gamma

   We use the binomial tree algorithm in the case of user-defined ops
   because in this case derived datatypes are allowed, and the user
   could pass basic datatypes on one process and derived on another as
   long as the type maps are the same. Breaking up derived datatypes
   to do the reduce-scatter is tricky.

   FIXME: Per the MPI-2.1 standard this case is not possible.  We
   should be able to use the reduce-scatter/gather approach as long as
   count >= pof2.  [goodell@ 2009-01-21]

   Possible improvements: 

   End Algorithm: MPI_Reduce
*/

/* not declared static because a machine-specific function may call this one 
   in some cases */
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_index_tuned_intra_MV2(const void *sendbuf,
                    void *recvbuf,
                    int count,
                    MPI_Datatype datatype,
                    MPI_Op op, int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
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
    int is_commutative, pof2;
    MPID_Op *op_ptr;
    int comm_size = 0;
    int nbytes = 0;
    int sendtype_size;
    int is_two_level = 0;
    MPID_Comm *shmem_commptr = NULL;
    MPI_Comm shmem_comm;
    comm_size = comm_ptr->local_size;
    MPID_Datatype_get_size_macro(datatype, sendtype_size);
    nbytes = count * sendtype_size;

    if (count == 0)
        return MPI_SUCCESS;
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
    } else {
        MPID_Op_get_ptr(op, op_ptr)
            if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

#ifdef _ENABLE_CUDA_
    int rank = 0, stride = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;

    rank = comm_ptr->rank;

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(recvbuf);
       if ( sendbuf != MPI_IN_PLACE ){
         send_mem_type = is_device_buffer(sendbuf);
       }
    }
    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)sendbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
                            (void *)recvbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        recvbuf = recv_host_buf;
    }
#endif

    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        if (mv2_reduce_indexed_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }
        if (likely(mv2_enable_shmem_reduce && mv2_enable_skip_tuning_table_search && (nbytes <= mv2_coll_skip_table_threshold))) {
            /* for small messages, force shmem + binomial */
            MV2_Reduce_intra_function = MPIR_Reduce_shmem_MV2;
            MV2_Reduce_function = MPIR_Reduce_binomial_MV2;
            is_two_level = 1;
            goto skip_tuning_tables;
        }
         FIND_PPN_INDEX  (reduce, local_size,conf_index, partial_sub_ok)
    }

    if (partial_sub_ok != 1) {
        conf_index = mv2_reduce_indexed_num_ppn_conf/2;
    }
    
conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_reduce_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_reduce_indexed_thresholds_table[conf_index][mv2_size_reduce_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
	/* Comm size smaller than smallest configuration in table: use smallest available */
	comm_size_index = 0;
    }
    else if (comm_size > table_max_comm_size) {
	/* Comm size larger than largest configuration in table: use largest available */
	comm_size_index = mv2_size_reduce_indexed_tuning_table[conf_index] - 1;
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

    last_inter = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
    table_min_inter_size = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
    table_max_inter_size = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[last_inter].msg_sz;
    last_intra = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].size_intra_table - 1;
    table_min_intra_size = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].intra_node[0].msg_sz;
    table_max_intra_size = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].intra_node[last_intra].msg_sz;
    
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

    /* Set intra-node function pt for reduce_two_level */
    MV2_Reduce_intra_function = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].
	intra_node[intra_node_algo_index].MV2_pt_Reduce_function;
    /* Set inter-leader pt */
    MV2_Reduce_function = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].
	inter_leader[inter_node_algo_index].MV2_pt_Reduce_function;

    if(mv2_reduce_intra_knomial_factor < 0)
    {
        mv2_reduce_intra_knomial_factor = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].
	    intra_k_degree;
    }
    if(mv2_reduce_inter_knomial_factor < 0)
    {
        mv2_reduce_inter_knomial_factor = mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].
	    inter_k_degree;
    }
    if(mv2_reduce_indexed_thresholds_table[conf_index][comm_size_index].
       is_two_level_reduce[inter_node_algo_index] == 1) {
               is_two_level = 1;
    }
    
skip_tuning_tables:
#ifdef CHANNEL_MRAIL_GEN2
    if (MPIR_Reduce_Zcpy_MV2 == MV2_Reduce_function) {
       
       if(mv2_use_slot_shmem_coll &&
          mv2_enable_zcpy_reduce == 1 && 
          nbytes <= mv2_shm_slot_len &&
	  comm_ptr->dev.ch.shmem_coll_ok == 1 &&
          mv2_enable_shmem_reduce && is_commutative == 1) {
           //do nothing and continue to use zcpy
       } else {
           //fall back to trusty algorithm because it's invalid to
           //use zcpy without the initializations.
           MV2_Reduce_function = MPIR_Reduce_binomial_MV2;
       }
    } 
#endif /* CHANNEL_MRAIL_GEN2 */

    
    /* We call Reduce function */
    if(is_two_level == 1)
    {
        if (comm_ptr->dev.ch.shmem_coll_ok == 1
            && is_commutative == 1) {
            mpi_errno = MPIR_Reduce_two_level_helper_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        }
    } else if(MV2_Reduce_function == &MPIR_Reduce_inter_knomial_wrapper_MV2 ){
        if(is_commutative ==1)
        {
            mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        }
    } else if(MV2_Reduce_function == &MPIR_Reduce_redscat_gather_MV2){
        if ((HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) && (count >= pof2))
        {
            mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                            datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                            datatype, op, root, comm_ptr, errflag);
        }
    } else {
        mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                        datatype, op, root, comm_ptr, errflag);
    }

    if (mpi_errno) {
       /* for communication errors, just record the error but continue */

        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type && ( rank == root )){
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
                            (void *)recv_host_buf, 
                            stride, 
                            cudaMemcpyHostToDevice);
    }
    if(rdma_enable_cuda && recv_mem_type){
        if(recv_host_buf){
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if(rdma_enable_cuda && send_mem_type){
        if(send_host_buf){
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    fn_exit:
      return mpi_errno;

    fn_fail:
      goto fn_exit;
}

/* not declared static because a machine-specific function may call this one 
   in some cases */
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_MV2(const void *sendbuf,
                    void *recvbuf,
                    int count,
                    MPI_Datatype datatype,
                    MPI_Op op, int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int range = 0;
    int range_threshold = 0;
    int range_intra_threshold = 0;
    int is_commutative, pof2;
    MPID_Op *op_ptr;
    int comm_size = 0;
    int nbytes = 0;
    int sendtype_size;
    int is_two_level = 0;

    MPIR_T_PVAR_COMM_COUNTER_INC(MV2,mv2_coll_reduce_subcomm,1,comm_ptr);
 
    if (mv2_use_indexed_tuning || mv2_use_indexed_reduce_tuning) {
	MPIR_Reduce_index_tuned_intra_MV2(sendbuf, recvbuf, count, 
					  datatype, op, root, comm_ptr, errflag);
	goto fn_exit;
    }
    comm_size = comm_ptr->local_size;
    MPID_Datatype_get_size_macro(datatype, sendtype_size);
    nbytes = count * sendtype_size;

    if (count == 0)
        return MPI_SUCCESS;
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
    } else {
        MPID_Op_get_ptr(op, op_ptr)
            if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

#ifdef _ENABLE_CUDA_
    int rank = 0, stride = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;
    const char *temp_sendbuf = sendbuf;

    rank = comm_ptr->rank;

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(recvbuf);
       if ( sendbuf != MPI_IN_PLACE ){
         send_mem_type = is_device_buffer(sendbuf);
       }
    }
    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)sendbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
                            (void *)recvbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        recvbuf = recv_host_buf;
    }
#endif

    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_reduce_tuning_table - 1)) &&
           (comm_size > mv2_reduce_thresholds_table[range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold < (mv2_reduce_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_reduce_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_reduce_thresholds_table[range].inter_leader[range_threshold].max !=
               -1)) {
        range_threshold++;
    }

    /* Search for corresponding intra node function */
    while ((range_intra_threshold < (mv2_reduce_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_reduce_thresholds_table[range].intra_node[range_intra_threshold].max)
           && (mv2_reduce_thresholds_table[range].intra_node[range_intra_threshold].max !=
               -1)) {
        range_intra_threshold++;
    }

    /* Set intra-node function pt for reduce_two_level */
    MV2_Reduce_intra_function = 
                          mv2_reduce_thresholds_table[range].intra_node[range_intra_threshold].
                          MV2_pt_Reduce_function;
    /* Set inter-leader pt */
    MV2_Reduce_function =
                          mv2_reduce_thresholds_table[range].inter_leader[range_threshold].
                          MV2_pt_Reduce_function;

    if(mv2_reduce_intra_knomial_factor<0)
    {
        mv2_reduce_intra_knomial_factor = mv2_reduce_thresholds_table[range].intra_k_degree;
    }
    if(mv2_reduce_inter_knomial_factor<0)
    {
        mv2_reduce_inter_knomial_factor = mv2_reduce_thresholds_table[range].inter_k_degree;
    }
    if(mv2_reduce_thresholds_table[range].is_two_level_reduce[range_threshold] == 1){
               is_two_level = 1;
    }
    /* We call Reduce function */
    if(is_two_level == 1)
    {
        if (comm_ptr->dev.ch.shmem_coll_ok == 1
            && is_commutative == 1) {
            mpi_errno = MPIR_Reduce_two_level_helper_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        }
    } else if(MV2_Reduce_function == &MPIR_Reduce_inter_knomial_wrapper_MV2 ){
        if(is_commutative ==1)
        {
            mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                           datatype, op, root, comm_ptr, errflag);
        }
    } else if(MV2_Reduce_function == &MPIR_Reduce_redscat_gather_MV2){
        if ((HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) && (count >= pof2))
        {
            mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                            datatype, op, root, comm_ptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_binomial_MV2(sendbuf, recvbuf, count, 
                                            datatype, op, root, comm_ptr, errflag);
        }
    } else {
        mpi_errno = MV2_Reduce_function(sendbuf, recvbuf, count, 
                                        datatype, op, root, comm_ptr, errflag);
    }

    if (mpi_errno) {
       /* for communication errors, just record the error but continue */

        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type && ( rank == root )){
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
                            (void *)recv_host_buf, 
                            stride, 
                            cudaMemcpyHostToDevice);
    }
    if(rdma_enable_cuda && recv_mem_type){
        if(recv_host_buf){
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if(rdma_enable_cuda && send_mem_type){
        sendbuf = temp_sendbuf;
        if(send_host_buf){
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    fn_exit:
      return mpi_errno;

    fn_fail:
      goto fn_exit;
}
