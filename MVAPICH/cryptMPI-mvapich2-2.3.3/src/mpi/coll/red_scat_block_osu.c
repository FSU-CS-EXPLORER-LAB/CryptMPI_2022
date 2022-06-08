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
#include <unistd.h>
#include "coll_shmem.h"
#include "red_scat_block_tuning.h"
#include <unistd.h>

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
int (*MV2_Red_scat_block_function)(
    const void* sendbuf,
    void* recvbuf,
    int recvcount,
    MPI_Datatype datatype,
    MPI_Op op,
    MPID_Comm *comm_ptr,
    MPIR_Errflag_t *errflag);

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_block_ring_2lvl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_block_ring_2lvl_MV2(
    const void* sendbuf,
    void* recvbuf,
    int recvcount,
    MPI_Datatype datatype,
    MPI_Op op,
    MPID_Comm *comm_ptr,
    MPIR_Errflag_t *errflag)
{
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;

    if (comm_ptr->dev.ch.rank_list == NULL) {
        return MPIR_Reduce_scatter_block_ring_MV2(
                                        sendbuf, recvbuf, recvcount,
                                        datatype, op, comm_ptr, errflag);
    }

    MPIU_CHKLMEM_DECL(2);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* number of elements each rank will receive */
    size_t elem_per_rank = (size_t) recvcount;

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int index_left  = (comm_size + rank_index - 1) % comm_size;
    int index_right = (comm_size + rank_index + 1) % comm_size;
    int rank_left   = comm_ptr->dev.ch.rank_list[index_left];
    int rank_right  = comm_ptr->dev.ch.rank_list[index_right];

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
    while (nread < elem_per_rank) {
        /* compute number of elements left to process */
        size_t count = elem_per_rank - nread;
        if (count > elem_per_buffer) {
            count = elem_per_buffer;
        }

        /* execute a lap through the ring */
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute offset into input buffer to pull data for this chunk */
            int src_index = (rank_index + dist) % comm_size;
            int src_rank  = comm_ptr->dev.ch.rank_list[src_index];
            unsigned long elem_offset = elem_per_rank * (unsigned long) src_rank + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, count, datatype,
                    tmp_sendbuf, count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
                MPIC_Irecv(tmp_recvbuf, count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag);
                MPIC_Waitall(2, request, status, errflag);
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, count, datatype,
                        buf, count, datatype);
            }
        }

        nread += count;
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_block_ring_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_block_ring_MV2(
    const void* sendbuf,
    void* recvbuf,
    int recvcount,
    MPI_Datatype datatype,
    MPI_Op op,
    MPID_Comm *comm_ptr,
    MPIR_Errflag_t *errflag)
{
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;
    int rank          = comm_ptr->rank;
    MPIU_CHKLMEM_DECL(2);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* number of elements each rank will receive */
    size_t elem_per_rank = (size_t) recvcount;

    /* compute left and right neighbors */
    int rank_left  = (rank - 1 + comm_size) % comm_size;
    int rank_right = (rank + 1 + comm_size) % comm_size;

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
    while (nread < elem_per_rank) {
        /* compute number of elements left to process */
        size_t count = elem_per_rank - nread;
        if (count > elem_per_buffer) {
            count = elem_per_buffer;
        }

        /* execute a lap through the ring */
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute offset into input buffer to pull data for this chunk */
            int src_rank = (rank + dist) % comm_size;
            unsigned long elem_offset = elem_per_rank * (unsigned long) src_rank + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, count, datatype,
                    tmp_sendbuf, count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
                MPIC_Irecv(tmp_recvbuf, count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag);
                MPIC_Waitall(2, request, status, errflag);
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, count, datatype,
                        buf, count, datatype);
            }
        }

        nread += count;
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_block_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_block_MV2(const void *sendbuf, void *recvbuf, 
                                int recvcount, MPI_Datatype datatype,
                                MPI_Op op, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	MPID_Op *op_ptr = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int is_commutative = 0, nbytes = 0;

    MPID_THREADPRIV_DECL;

    /* set op_errno to 0. stored in perthread structure */
    MPID_THREADPRIV_GET;
    MPID_THREADPRIV_FIELD(op_errno) = 0;

    if (recvcount == 0) {
        goto fn_exit;
    }

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

    if (!is_commutative) {
        mpi_errno = MPIR_Reduce_scatter_block(
                                    sendbuf, recvbuf, recvcount,
                                    datatype, op, comm_ptr, errflag);
    } else {
        int range = 0;
        int range_threshold = 0;
	    int comm_size = comm_ptr->local_size;

        /* Search for the corresponding system size inside the tuning table */
        while ((range < (mv2_size_red_scat_block_tuning_table - 1)) &&
               (comm_size > mv2_red_scat_block_thresholds_table[range].numproc)) {
            range++;
        }
        /* Search for corresponding inter-leader function */
        while ((range_threshold < (mv2_red_scat_block_thresholds_table[range].size_inter_table - 1))
               && (nbytes >
                   mv2_red_scat_block_thresholds_table[range].inter_leader[range_threshold].max)
               && (mv2_red_scat_block_thresholds_table[range].inter_leader[range_threshold].max !=
                   -1)) {
            range_threshold++;
        }

        /* Set inter-leader pt */
        MV2_Red_scat_block_function =
                              mv2_red_scat_block_thresholds_table[range].inter_leader[range_threshold].
                              MV2_Red_scat_block_function;

        mpi_errno = MV2_Red_scat_block_function(
                                    sendbuf, recvbuf, recvcount,
                                    datatype, op, comm_ptr, errflag);
    }
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );

    if (MPID_THREADPRIV_FIELD(op_errno))
    mpi_errno = MPID_THREADPRIV_FIELD(op_errno);

    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#endif /*defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)*/
