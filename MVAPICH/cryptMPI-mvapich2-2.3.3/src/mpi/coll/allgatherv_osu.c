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
#include "coll_shmem.h"
#include "allgatherv_tuning.h"

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgatherv_rec_doubling);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgatherv_bruck);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgatherv_ring);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgatherv_ring_cyclic);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_rec_doubling);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bruck);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring_cyclic);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_rec_doubling_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bruck_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_rec_doubling_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bruck_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_rec_doubling_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bruck_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_rec_doubling_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bruck_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_ring_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgatherv_count_recv);


/* This is the default implementation of allgatherv. The algorithm is:

   Algorithm: MPI_Allgatherv

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

   For long messages or medium-size messages and non-power-of-two
   no. of processes, we use a ring algorithm. In the first step, each
   process i sends its contribution to process i+1 and receives
   the contribution from process i-1 (with wrap-around). From the
   second step onwards, each process i forwards to process i+1 the
   data it received from process i-1 in the previous step. This takes
   a total of p-1 steps.

   Cost = (p-1).alpha + n.((p-1)/p).beta

   Possible improvements:

   End Algorithm: MPI_Allgatherv
*/

int (*MV2_Allgatherv_function)(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               const int *recvcounts,
                               const int *displs,
                               MPI_Datatype recvtype,
                               MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
#undef FUNCNAME
#define FUNCNAME MPIR_Allgatherv_Rec_Doubling_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgatherv_Rec_Doubling_MV2(const void *sendbuf,
                                     int sendcount,
                                     MPI_Datatype sendtype,
                                     void *recvbuf,
                                     const int *recvcounts,
                                     const int *displs,
                                     MPI_Datatype recvtype,
                                     MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgatherv,rec_doubling);
    int comm_size, rank, j, i;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent, recvtype_true_extent, recvtype_true_lb;
    int curr_cnt, dst, total_count;
    void *tmp_buf;
    int mask, dst_tree_root, my_tree_root, is_homogeneous, position,
        send_offset, recv_offset, last_recv_cnt, nprocs_completed, k,
        offset, tmp_mask, tree_root;

#ifdef MPID_HAS_HETERO
    int tmp_buf_size, nbytes;
#endif
    MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgatherv_rec_doubling, 1);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    if (is_homogeneous) {
        /* need to receive contiguously into tmp_buf because
           displs could make the recvbuf noncontiguous */

        MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb,
                                       &recvtype_true_extent);

        MPIU_Ensure_Aint_fits_in_pointer(total_count *
                                         (MPIR_MAX
                                          (recvtype_true_extent,
                                           recvtype_extent)));
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                            total_count *
                            (MPIR_MAX
                             (recvtype_true_extent, recvtype_extent)),
                            mpi_errno, "tmp_buf");

        /* adjust for potential negative lower bound in datatype */
        tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

        /* copy local data into right location in tmp_buf */
        position = 0;
        for (i = 0; i < rank; i++)
            position += recvcounts[i];
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       ((char *) tmp_buf + position *
                                        recvtype_extent), recvcounts[rank],
                                       recvtype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        } else {
            /* if in_place specified, local data is found in recvbuf */
            mpi_errno = MPIR_Localcopy(((char *) recvbuf +
                                        displs[rank] * recvtype_extent),
                                       recvcounts[rank], recvtype,
                                       ((char *) tmp_buf + position *
                                        recvtype_extent), recvcounts[rank],
                                       recvtype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }

        curr_cnt = recvcounts[rank];

        mask = 0x1;
        i = 0;
        while (mask < comm_size) {
            dst = rank ^ mask;

            /* find offset into send and recv buffers. zero out
               the least significant "i" bits of rank and dst to
               find root of src and dst subtrees. Use ranks of
               roots as index to send from and recv into buffer */

            dst_tree_root = dst >> i;
            dst_tree_root <<= i;

            my_tree_root = rank >> i;
            my_tree_root <<= i;

            if (dst < comm_size) {
                send_offset = 0;
                for (j = 0; j < my_tree_root; j++)
                    send_offset += recvcounts[j];

                recv_offset = 0;
                for (j = 0; j < dst_tree_root; j++)
                    recv_offset += recvcounts[j];

                MPIR_PVAR_INC(allgatherv, rec_doubling, send, curr_cnt, recvtype);
                MPIR_PVAR_INC(allgatherv, rec_doubling, recv, total_count - recv_offset, recvtype);
                mpi_errno =
                    MPIC_Sendrecv(((char *) tmp_buf +
                                      send_offset * recvtype_extent),
                                     curr_cnt, recvtype, dst,
                                     MPIR_ALLGATHERV_TAG,
                                     ((char *) tmp_buf +
                                      recv_offset * recvtype_extent),
                                     total_count - recv_offset, recvtype,
                                     dst, MPIR_ALLGATHERV_TAG, comm_ptr,
                                     &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                } else
                    /* for convenience, recv is posted for a bigger amount
                       than will be sent */
                    MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                curr_cnt += last_recv_cnt;
            }

            /* if some processes in this process's subtree in this step
               did not have any destination process to communicate with
               because of non-power-of-two, we need to send them the
               data that they would normally have received from those
               processes. That is, the haves in this subtree must send to
               the havenots. We use a logarithmic
               recursive-halfing algorithm for this. */

            /* This part of the code will not currently be
               executed because we are not using recursive
               doubling for non power of two. Mark it as experimental
               so that it doesn't show up as red in the coverage
               tests. */

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

                tmp_mask = mask >> 1;

                while (tmp_mask) {
                    dst = rank ^ tmp_mask;

                    tree_root = rank >> k;
                    tree_root <<= k;

                    /* send only if this proc has data and destination
                       doesn't have data. at any step, multiple processes
                       can send if they have the data */
                    if ((dst > rank)
                        && (rank < tree_root + nprocs_completed)
                        && (dst >= tree_root + nprocs_completed)) {

                        offset = 0;
                        for (j = 0; j < (my_tree_root + mask); j++)
                            offset += recvcounts[j];
                        offset *= recvtype_extent;

                        MPIR_PVAR_INC(allgatherv, rec_doubling, send, last_recv_cnt, recvtype);
                        mpi_errno =
                            MPIC_Send(((char *) tmp_buf + offset),
                                         last_recv_cnt, recvtype, dst,
                                         MPIR_ALLGATHERV_TAG, comm_ptr, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                        /* last_recv_cnt was set in the previous
                           receive. that's the amount of data to be
                           sent now. */
                    }
                    /* recv only if this proc. doesn't have data and sender
                       has data */
                    else if ((dst < rank) &&
                             (dst < tree_root + nprocs_completed) &&
                             (rank >= tree_root + nprocs_completed)) {

                        offset = 0;
                        for (j = 0; j < (my_tree_root + mask); j++)
                            offset += recvcounts[j];

                        MPIR_PVAR_INC(allgatherv, rec_doubling, recv, total_count - offset, recvtype);
                        mpi_errno =
                            MPIC_Recv(((char *) tmp_buf +
                                          offset * recvtype_extent),
                                         total_count - offset, recvtype,
                                         dst, MPIR_ALLGATHERV_TAG, comm_ptr,
                                         &status, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        } else
                            /* for convenience, recv is posted for a
                               bigger amount than will be sent */
                            MPIR_Get_count_impl(&status, recvtype,
                                                &last_recv_cnt);
                        curr_cnt += last_recv_cnt;
                    }
                    tmp_mask >>= 1;
                    k--;
                }
            }
            /* --END EXPERIMENTAL-- */

            mask <<= 1;
            i++;
        }

        /* copy data from tmp_buf to recvbuf */
        position = 0;
        for (j = 0; j < comm_size; j++) {
            if ((sendbuf != MPI_IN_PLACE) || (j != rank)) {
                /* not necessary to copy if in_place and
                   j==rank. otherwise copy. */
                mpi_errno =
                    MPIR_Localcopy(((char *) tmp_buf +
                                    position * recvtype_extent),
                                   recvcounts[j], recvtype,
                                   ((char *) recvbuf +
                                    displs[j] * recvtype_extent),
                                   recvcounts[j], recvtype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
            position += recvcounts[j];
        }
    }
#ifdef MPID_HAS_HETERO
    else {
        /* heterogeneous. need to use temp. buffer. */
        MPIR_Pack_size_impl(total_count, recvtype, &tmp_buf_size);
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno,
                            "tmp_buf");

        /* calculate the value of nbytes, the number of bytes in packed
           representation corresponding to a single recvtype. Since
           MPI_Pack_size returns only an upper bound on
           the size, to get the real size we actually pack some data
           into tmp_buf and see by how much 'position' is incremented. */

        position = 0;
        mpi_errno =
            MPIR_Pack_impl(recvbuf, 1, recvtype, tmp_buf, tmp_buf_size,
                           &position);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        nbytes = position;

        /* pack local data into right location in tmp_buf */
        position = 0;
        for (i = 0; i < rank; i++)
            position += recvcounts[i];
        position *= nbytes;

        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIR_Pack_impl(sendbuf, sendcount, sendtype, tmp_buf,
                               tmp_buf_size, &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        } else {
            /* if in_place specified, local data is found in recvbuf */
            mpi_errno =
                MPIR_Pack_impl(((char *) recvbuf +
                                displs[rank] * recvtype_extent),
                               recvcounts[rank], recvtype, tmp_buf,
                               tmp_buf_size, &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }

        curr_cnt = recvcounts[rank] * nbytes;

        mask = 0x1;
        i = 0;
        while (mask < comm_size) {
            dst = rank ^ mask;

            /* find offset into send and recv buffers. zero out
               the least significant "i" bits of rank and dst to
               find root of src and dst subtrees. Use ranks of
               roots as index to send from and recv into buffer. */

            dst_tree_root = dst >> i;
            dst_tree_root <<= i;

            my_tree_root = rank >> i;
            my_tree_root <<= i;

            send_offset = 0;
            for (j = 0; j < my_tree_root; j++)
                send_offset += recvcounts[j];
            send_offset *= nbytes;

            recv_offset = 0;
            for (j = 0; j < dst_tree_root; j++)
                recv_offset += recvcounts[j];
            recv_offset *= nbytes;

            if (dst < comm_size) {
                MPIR_PVAR_INC(allgatherv, rec_doubling, send, curr_cnt, MPI_BYTE);
                MPIR_PVAR_INC(allgatherv, rec_doubling, recv, tmp_buf_size - recv_offset, MPI_BYTE);
                mpi_errno =
                    MPIC_Sendrecv(((char *) tmp_buf + send_offset),
                                     curr_cnt, MPI_BYTE, dst,
                                     MPIR_ALLGATHERV_TAG,
                                     ((char *) tmp_buf + recv_offset),
                                     tmp_buf_size - recv_offset, MPI_BYTE,
                                     dst, MPIR_ALLGATHERV_TAG, comm_ptr,
                                     &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                } else
                    /* for convenience, recv is posted for a bigger amount
                       than will be sent */
                    MPIR_Get_count_impl(&status, MPI_BYTE, &last_recv_cnt);
                curr_cnt += last_recv_cnt;
            }

            /* if some processes in this process's subtree in this step
               did not have any destination process to communicate with
               because of non-power-of-two, we need to send them the
               data that they would normally have received from those
               processes. That is, the haves in this subtree must send to
               the havenots. We use a logarithmic recursive-halfing algorithm
               for this. */

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

                offset = 0;
                for (j = 0; j < (my_tree_root + mask); j++)
                    offset += recvcounts[j];
                offset *= nbytes;
                tmp_mask = mask >> 1;

                while (tmp_mask) {
                    dst = rank ^ tmp_mask;

                    tree_root = rank >> k;
                    tree_root <<= k;

                    /* send only if this proc has data and destination
                       doesn't have data. at any step, multiple processes
                       can send if they have the data */
                    if ((dst > rank)
                        && (rank < tree_root + nprocs_completed)
                        && (dst >= tree_root + nprocs_completed)) {

                        MPIR_PVAR_INC(allgatherv, rec_doubling, send, last_recv_cnt, MPI_BYTE);
                        mpi_errno =
                            MPIC_Send(((char *) tmp_buf + offset),
                                         last_recv_cnt, MPI_BYTE, dst,
                                         MPIR_ALLGATHERV_TAG, comm_ptr, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                        /* last_recv_cnt was set in the previous
                           receive. that's the amount of data to be
                           sent now. */
                    }
                    /* recv only if this proc. doesn't have data and sender
                       has data */
                    else if ((dst < rank) &&
                             (dst < tree_root + nprocs_completed) &&
                             (rank >= tree_root + nprocs_completed)) {
                        MPIR_PVAR_INC(allgatherv, rec_doubling, recv, tmp_buf_size - offset, MPI_BYTE);
                        mpi_errno =
                            MPIC_Recv(((char *) tmp_buf + offset),
                                         tmp_buf_size - offset, MPI_BYTE,
                                         dst, MPIR_ALLGATHERV_TAG, comm_ptr,
                                         &status, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        } else
                            /* for convenience, recv is posted for a bigger amount
                               than will be sent */
                            MPIR_Get_count_impl(&status, MPI_BYTE,
                                                &last_recv_cnt);
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
        for (j = 0; j < comm_size; j++) {
            if ((sendbuf != MPI_IN_PLACE) || (j != rank)) {
                /* not necessary to unpack if in_place and
                   j==rank. otherwise unpack. */
                mpi_errno =
                    MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position,
                                     ((char *) recvbuf +
                                      displs[j] * recvtype_extent),
                                     recvcounts[j], recvtype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
        }
    }
#endif                          /* MPID_HAS_HETERO */
  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,allgatherv,rec_doubling);      
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgatherv_Bruck_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgatherv_Bruck_MV2(const void *sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              const int *recvcounts,
                              const int *displs,
                              MPI_Datatype recvtype,
                              MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgatherv,bruck);
    int comm_size, rank, j, i;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvbuf_extent, recvtype_extent, recvtype_true_extent,
        recvtype_true_lb;
    int curr_cnt, send_cnt, dst, total_count, pof2, src, rem;
    int recv_cnt;
    void *tmp_buf;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgatherv_bruck, 1);
    MPIU_CHKLMEM_DECL(1);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* allocate a temporary buffer of the same size as recvbuf. */
    /* get true extent of recvtype */
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb,
                                   &recvtype_true_extent);

    MPIU_Ensure_Aint_fits_in_pointer(total_count *
                                     MPIR_MAX(recvtype_true_extent,
                                              recvtype_extent));
    recvbuf_extent =
        total_count * (MPIR_MAX(recvtype_true_extent, recvtype_extent));

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *, recvbuf_extent, mpi_errno, "tmp_buf");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    /* copy local data to the top of tmp_buf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   tmp_buf, recvcounts[rank], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else {
        mpi_errno = MPIR_Localcopy(((char *) recvbuf +
                                    displs[rank] * recvtype_extent),
                                   recvcounts[rank], recvtype,
                                   tmp_buf, recvcounts[rank], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

    /* do the first \floor(\lg p) steps */

    curr_cnt = recvcounts[rank];
    pof2 = 1;
    while (pof2 <= comm_size / 2) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        MPIR_PVAR_INC(allgatherv, bruck, send, curr_cnt, recvtype);
        MPIR_PVAR_INC(allgatherv, bruck, recv, total_count - curr_cnt, recvtype);
        mpi_errno = MPIC_Sendrecv(tmp_buf, curr_cnt, recvtype, dst,
                                     MPIR_ALLGATHERV_TAG,
                                     ((char *) tmp_buf +
                                      curr_cnt * recvtype_extent),
                                     total_count - curr_cnt, recvtype, src,
                                     MPIR_ALLGATHERV_TAG, comm_ptr, &status,
                                     errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            recv_cnt = 0;
        } else
            MPIR_Get_count_impl(&status, recvtype, &recv_cnt);
        curr_cnt += recv_cnt;

        pof2 *= 2;
    }

    /* if comm_size is not a power of two, one more step is needed */

    rem = comm_size - pof2;
    if (rem) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        send_cnt = 0;
        for (i = 0; i < rem; i++)
            send_cnt += recvcounts[(rank + i) % comm_size];

        MPIR_PVAR_INC(allgatherv, bruck, send, send_cnt, recvtype);
        MPIR_PVAR_INC(allgatherv, bruck, recv, total_count - curr_cnt, recvtype);
        mpi_errno = MPIC_Sendrecv(tmp_buf, send_cnt, recvtype,
                                     dst, MPIR_ALLGATHERV_TAG,
                                     ((char *) tmp_buf +
                                      curr_cnt * recvtype_extent),
                                     total_count - curr_cnt, recvtype, src,
                                     MPIR_ALLGATHERV_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Rotate blocks in tmp_buf down by (rank) blocks and store
     * result in recvbuf. */

    send_cnt = 0;
    for (i = 0; i < (comm_size - rank); i++) {
        j = (rank + i) % comm_size;
        mpi_errno =
            MPIR_Localcopy((char *) tmp_buf + send_cnt * recvtype_extent,
                           recvcounts[j], recvtype,
                           (char *) recvbuf + displs[j] * recvtype_extent,
                           recvcounts[j], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        send_cnt += recvcounts[j];
    }

    for (i = 0; i < rank; i++) {
        mpi_errno =
            MPIR_Localcopy((char *) tmp_buf + send_cnt * recvtype_extent,
                           recvcounts[i], recvtype,
                           (char *) recvbuf + displs[i] * recvtype_extent,
                           recvcounts[i], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        send_cnt += recvcounts[i];
    }
  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,allgatherv,bruck);
    return mpi_errno;
  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Allgatherv_Ring_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgatherv_Ring_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcounts,
                             const int *displs,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgatherv,ring);
    int comm_size, rank, i, left, right, total_count;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;

    /* User has not forced algorthitm selection and non-block allocation is used */
    if (mv2_user_allgatherv_inter == NULL
        && comm_ptr->dev.ch.rank_list != NULL
        && comm_ptr->dev.ch.is_blocked != 1) {
        return MPIR_Allgatherv_Ring_Cyclic_MV2(
                sendbuf, sendcount, sendtype,
                recvbuf, recvcounts, displs, recvtype,
                comm_ptr, errflag);
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgatherv_ring, 1);
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    char *sbuf = NULL, *rbuf = NULL;
    int soffset, roffset;
    int torecv, tosend, min;
    int sendnow, recvnow;
    int sindex, rindex;

    if (sendbuf != MPI_IN_PLACE) {
        /* First, load the "local" version in the recvbuf. */
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    displs[rank] * recvtype_extent),
                                   recvcounts[rank], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }
#ifdef _ENABLE_CUDA_ 
    /* This synchronization is needed because MPIR_Localcopy calls cudamemcpy
     * on the default stream (0) but subsequent MPI_Isend/Irecv calls access
     * GPU buffers using non-default streams which don't wait for the initial
     * local copy to complete*/
    if (rdma_enable_cuda && cuda_initialized
        && rdma_cuda_nonblocking_streams) {
            CUDA_CHECK(cudaEventRecord(cuda_nbstream_sync_event, 0));
            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, cuda_nbstream_sync_event, 0));
    }
#endif

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    torecv = total_count - recvcounts[rank];
    tosend = total_count - recvcounts[right];

    min = recvcounts[0];
    for (i = 1; i < comm_size; i++)
        if (min > recvcounts[i])
            min = recvcounts[i];
    if (min * recvtype_extent < MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE)
        min = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE / recvtype_extent;
    /* Handle the case where the datatype extent is larger than
     * the pipeline size. */
    if (!min)
        min = 1;

    sindex = rank;
    rindex = left;
    soffset = 0;
    roffset = 0;
    while (tosend || torecv) {  /* While we have data to send or receive */
        sendnow =
            ((recvcounts[sindex] - soffset) >
             min) ? min : (recvcounts[sindex] - soffset);
        recvnow =
            ((recvcounts[rindex] - roffset) >
             min) ? min : (recvcounts[rindex] - roffset);
        sbuf =
            (char *) recvbuf + ((displs[sindex] + soffset) * recvtype_extent);
        rbuf =
            (char *) recvbuf + ((displs[rindex] + roffset) * recvtype_extent);

        /* Protect against wrap-around of indices */
        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        /* Communicate */
        if (!sendnow && !recvnow) {
            /* Don't do anything. This case is possible if two
             * consecutive processes contribute 0 bytes each. */
        } else if (!sendnow) {  /* If there's no data to send, just do a recv call */
            MPIR_PVAR_INC(allgatherv, ring, recv, recvnow, recvtype);
            mpi_errno =
                MPIC_Recv(rbuf, recvnow, recvtype, left,
                             MPIR_ALLGATHERV_TAG, comm_ptr, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            torecv -= recvnow;
        } else if (!recvnow) {  /* If there's no data to receive, just do a send call */
            MPIR_PVAR_INC(allgatherv, ring, send, sendnow, recvtype);
            mpi_errno =
                MPIC_Send(sbuf, sendnow, recvtype, right,
                             MPIR_ALLGATHERV_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            tosend -= sendnow;
        } else {                /* There's data to be sent and received */
            MPIR_PVAR_INC(allgatherv, ring, send, sendnow, recvtype);
            MPIR_PVAR_INC(allgatherv, ring, recv, recvnow, recvtype);
            mpi_errno =
                MPIC_Sendrecv(sbuf, sendnow, recvtype, right,
                                 MPIR_ALLGATHERV_TAG, rbuf, recvnow,
                                 recvtype, left, MPIR_ALLGATHERV_TAG, comm_ptr,
                                 &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }

        soffset += sendnow;
        roffset += recvnow;
        if (soffset == recvcounts[sindex]) {
            soffset = 0;
            sindex = (sindex + comm_size - 1) % comm_size;
        }
        if (roffset == recvcounts[rindex]) {
            roffset = 0;
            rindex = (rindex + comm_size - 1) % comm_size;
        }
    }
  fn_exit:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,allgatherv,ring);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Execute an allgatherv by forwarding data through a ring of
 * processes.  This implementation uses the two-level data
 * structures to account for how procs are assigned to nodes
 * to ensure data is only sent into and out of each node once. */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgatherv_Ring_Cyclic_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgatherv_Ring_Cyclic_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcounts,
                             const int *displs,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgatherv,ring_cyclic);
    int comm_size, rank, i, total_count;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Status status;
    MPI_Aint recvtype_extent;

    /* rank_list is not initialized until create_2level_comm is called */
    if (!comm_ptr->dev.ch.rank_list) {
        return MPIR_Allgatherv_Ring_MV2(
                sendbuf, sendcount, sendtype,
                recvbuf, recvcounts, displs, recvtype,
                comm_ptr, errflag);
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allgatherv_ring_cyclic, 1);
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    char *sbuf = NULL, *rbuf = NULL;
    int soffset, roffset;
    int torecv, tosend, min;
    int sendnow, recvnow;

    if (sendbuf != MPI_IN_PLACE) {
        /* First, load the "local" version in the recvbuf. */
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    displs[rank] * recvtype_extent),
                                   recvcounts[rank], recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }
#ifdef _ENABLE_CUDA_
    /* This synchronization is needed because MPIR_Localcopy calls cudamemcpy
     * on the default stream (0) but subsequent MPI_Isend/Irecv calls access
     * GPU buffers using non-default streams which don't wait for the initial
     * local copy to complete*/
    if (rdma_enable_cuda && cuda_initialized
        && rdma_cuda_nonblocking_streams) {
            CUDA_CHECK(cudaEventRecord(cuda_nbstream_sync_event, 0));
            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, cuda_nbstream_sync_event, 0));
    }
#endif

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int left_index  = (comm_size + rank_index - 1) % comm_size;
    int right_index = (comm_size + rank_index + 1) % comm_size;
    int left  = comm_ptr->dev.ch.rank_list[left_index];
    int right = comm_ptr->dev.ch.rank_list[right_index];

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
    int send_index = rank_index;
    int recv_index = left_index;

    torecv = total_count - recvcounts[rank];
    tosend = total_count - recvcounts[right];

    min = recvcounts[0];
    for (i = 1; i < comm_size; i++)
        if (min > recvcounts[i])
            min = recvcounts[i];
    if (min * recvtype_extent < MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE)
        min = MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE / recvtype_extent;
    /* Handle the case where the datatype extent is larger than
     * the pipeline size. */
    if (!min)
        min = 1;

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
    soffset = 0;
    roffset = 0;
    while (tosend || torecv) {  /* While we have data to send or receive */

        /* compute ranks whose data we'll send and receive in this step */
        int send_rank = comm_ptr->dev.ch.rank_list[send_index];
        int recv_rank = comm_ptr->dev.ch.rank_list[recv_index];

		sendnow =
            ((recvcounts[send_rank] - soffset) >
             min) ? min : (recvcounts[send_rank] - soffset);
        recvnow =
            ((recvcounts[recv_rank] - roffset) >
             min) ? min : (recvcounts[recv_rank] - roffset);
        sbuf =
            (char *) recvbuf + ((displs[send_rank] + soffset) * recvtype_extent);
        rbuf =
            (char *) recvbuf + ((displs[recv_rank] + roffset) * recvtype_extent);

        /* Protect against wrap-around of indices */
        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        /* Communicate */
        if (!sendnow && !recvnow) {
            /* Don't do anything. This case is possible if two
             * consecutive processes contribute 0 bytes each. */
        } else if (!sendnow) {  /* If there's no data to send, just do a recv call */
            mpi_errno =
                MPIC_Recv(rbuf, recvnow, recvtype, left,
                             MPIR_ALLGATHERV_TAG, comm_ptr, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            torecv -= recvnow;
        } else if (!recvnow) {  /* If there's no data to receive, just do a send call */
            mpi_errno =
                MPIC_Send(sbuf, sendnow, recvtype, right,
                             MPIR_ALLGATHERV_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            tosend -= sendnow;
        } else {                /* There's data to be sent and received */
            mpi_errno =
                MPIC_Sendrecv(sbuf, sendnow, recvtype, right,
                                 MPIR_ALLGATHERV_TAG, rbuf, recvnow,
                                 recvtype, left, MPIR_ALLGATHERV_TAG, comm_ptr,
                                 &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            tosend -= sendnow;
            torecv -= recvnow;
        }

        soffset += sendnow;
        roffset += recvnow;
        if (soffset == recvcounts[send_rank]) {
            soffset = 0;
            send_index = (send_index + comm_size - 1) % comm_size;
        }
        if (roffset == recvcounts[recv_rank]) {
            roffset = 0;
            recv_index = (recv_index + comm_size - 1) % comm_size;
        }
    }
  fn_exit:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag)
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    MPIR_TIMER_END(coll,allgatherv,ring_cyclic);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* MPIR_Allgatherv performs an allgatherv using point-to-point
   messages.  This is intended to be used by device-specific
   implementations of allgatherv.  In all other cases
   MPIR_Allgatherv_impl should be used. */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgatherv_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgatherv_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                        void *recvbuf, const int *recvcounts, const int *displs,
                        MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                        MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int range = 0, comm_size, total_count, recvtype_size, i;
    int range_threshold = 0;
    int nbytes = 0;

    comm_size = comm_ptr->local_size;
    total_count = 0;
    for (i = 0; i < comm_size; i++)
        total_count += recvcounts[i];

    if (total_count == 0)
        goto fn_exit;

    MPID_Datatype_get_size_macro(recvtype, recvtype_size);
    nbytes = total_count * recvtype_size;

    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_allgatherv_tuning_table - 1)) &&
           (comm_size > mv2_allgatherv_thresholds_table[range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold < (mv2_allgatherv_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               comm_size * mv2_allgatherv_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_allgatherv_thresholds_table[range].inter_leader[range_threshold].max !=
               -1)) {
        range_threshold++;
    }
    /* Set inter-leader pt */
    MV2_Allgatherv_function =
                          mv2_allgatherv_thresholds_table[range].inter_leader[range_threshold].
                          MV2_pt_Allgatherv_function;

    if (MV2_Allgatherv_function == &MPIR_Allgatherv_Rec_Doubling_MV2)
    {
        if(!(comm_size & (comm_size - 1)))
        {
            mpi_errno =
                MPIR_Allgatherv_Rec_Doubling_MV2(sendbuf, sendcount,
                                                 sendtype, recvbuf,
                                                 recvcounts, displs,
                                                 recvtype, comm_ptr, errflag);
        } else {
            mpi_errno =
                MPIR_Allgatherv_Bruck_MV2(sendbuf, sendcount,
                                          sendtype, recvbuf,
                                          recvcounts, displs,
                                          recvtype, comm_ptr, errflag);
        }
    } else {
        mpi_errno =
            MV2_Allgatherv_function(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcounts, displs,
                                    recvtype, comm_ptr, errflag);
    }

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
