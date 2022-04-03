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

#ifdef _ENABLE_CUDA_
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
#include "datatype.h"
#include "coll_shmem.h"
#include "unistd.h"

extern void *mv2_cuda_allgather_store_buf;
extern int mv2_cuda_allgather_store_buf_size;

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_intra_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_intra_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_intra_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_intra_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allgather_cuda_count_recv);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allgather_cuda);

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_cuda_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_cuda_intra_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_TIMER_START(coll,allgather,cuda);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent = 0;
    int j, i;
    int curr_cnt, dst, left, right, jnext;
    int mask, dst_tree_root, my_tree_root,
        send_offset, recv_offset;
    int comm_size_is_pof2;
    MPI_Status status;
    int page_size = 0;
    int result, max_size;
    MPID_Request *recv_req;
    MPID_Request *send_req;
    cudaError_t cudaerr;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0)) {
        MPIR_TIMER_END(coll,allgather,cuda);
        return MPI_SUCCESS;
    }

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    comm_size_is_pof2 = comm_ptr->dev.ch.is_pof2;

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*Creating Store Buffer*/
    page_size = getpagesize();

    max_size = mv2_cuda_allgather_store_buf_size < recvcount * comm_size * recvtype_extent ? 
            recvcount * comm_size * recvtype_extent : mv2_cuda_allgather_store_buf_size;

    if (mv2_cuda_allgather_store_buf_size < max_size || !mv2_cuda_allgather_store_buf){
        if(mv2_cuda_allgather_store_buf){
            ibv_cuda_unregister(mv2_cuda_allgather_store_buf);
            MPIU_Memalign_Free(mv2_cuda_allgather_store_buf);
        } 
        result = MPIU_Memalign(&mv2_cuda_allgather_store_buf, page_size, max_size);
        if ((result!=0) || (NULL == mv2_cuda_allgather_store_buf)) {
            mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                    FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                    "posix_memalign", strerror(errno));
            MPIR_ERR_POP (mpi_errno);
        }
        ibv_cuda_register(mv2_cuda_allgather_store_buf, max_size);
        mv2_cuda_allgather_store_buf_size = max_size;
    }

    if (recvcount*recvtype_extent > rdma_cuda_allgather_rd_limit*comm_size || 
            !comm_size_is_pof2) { // RING
            
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
        
    /* This synchronization is needed because MPIR_Localcopy calls cudamemcpy
     * on the default stream (0) but subsequent MPI_Isend/Irecv calls access
     * GPU buffers using non-default streams which don't wait for the initial
     * local copy to complete*/
        if (rdma_enable_cuda && cuda_initialized
            && rdma_cuda_nonblocking_streams) {
                CUDA_CHECK(cudaEventRecord(cuda_nbstream_sync_event, 0));
                CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, cuda_nbstream_sync_event, 0));
        }
        

        /*Starting the RING stages*/
        left  = (comm_size + rank - 1) % comm_size;
        right = (rank + 1) % comm_size;
        j     = rank;
        jnext = left;

        MPIR_PVAR_INC(allgather_cuda, intra, recv, recvcount*recvtype_extent, MPI_BYTE);
        mpi_errno = MPIC_Irecv( ((char *)mv2_cuda_allgather_store_buf + jnext*recvcount*recvtype_extent),
                                recvcount*recvtype_extent,
                                MPI_BYTE,
                                left,
                                MPIR_ALLGATHER_TAG,
                                comm_ptr,
                                &recv_req );
        MPIR_PVAR_INC(allgather_cuda, intra, send, recvcount*recvtype_extent, MPI_BYTE);
        mpi_errno = MPIC_Isend(((char *)recvbuf + j*recvcount*recvtype_extent),
                            recvcount*recvtype_extent,
                            MPI_BYTE,
                            right,
                            MPIR_ALLGATHER_TAG,
                            comm_ptr,
                            &send_req, errflag);
        mpi_errno = MPIC_Waitall(1, &recv_req, &status, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        MPIU_Memcpy_CUDA_Async((void *)((char *)recvbuf + jnext*recvcount*recvtype_extent),
                (void *)((char *)mv2_cuda_allgather_store_buf + jnext*recvcount*recvtype_extent),
                recvcount*recvtype_extent,
                cudaMemcpyHostToDevice,
                stream_h2d );

        mpi_errno = MPIC_Waitall(1, &send_req, &status, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        j       = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;

        /*Intermediate steps of communication*/
        for (i=2; i<comm_size-1; i++) {
            MPIR_PVAR_INC(allgather_cuda, intra, recv, recvcount, recvtype);
            mpi_errno = MPIC_Irecv( ((char *)mv2_cuda_allgather_store_buf + jnext*recvcount*recvtype_extent),
                                    recvcount,
                                    recvtype,
                                    left,
                                    MPIR_ALLGATHER_TAG,
                                    comm_ptr,
                                    &recv_req );
            MPIR_PVAR_INC(allgather_cuda, intra, send, recvcount, recvtype);
            mpi_errno = MPIC_Isend(((char *)mv2_cuda_allgather_store_buf + j*recvcount*recvtype_extent),
                                    recvcount,
                                    recvtype,
                                    right,
                                    MPIR_ALLGATHER_TAG,
                                    comm_ptr,
                                    &send_req, errflag);
            mpi_errno = MPIC_Waitall(1, &recv_req, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
                    
            MPIU_Memcpy_CUDA_Async((void *)((char *)recvbuf + jnext*recvcount*recvtype_extent),
                    (void *)((char *)mv2_cuda_allgather_store_buf + jnext*recvcount*recvtype_extent),
                    recvcount*recvtype_extent,
                    cudaMemcpyHostToDevice,
                    stream_h2d );

            mpi_errno = MPIC_Waitall(1, &send_req, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            j        = jnext;
            jnext = (comm_size + jnext - 1) % comm_size;
        }

        /*Last stage of communication - copy directly to device*/
        if ( i < comm_size ){
            MPIR_PVAR_INC(allgather_cuda, intra, recv, recvcount, recvtype);
            mpi_errno = MPIC_Irecv( ((char *)recvbuf + jnext*recvcount*recvtype_extent),
                    recvcount,
                    recvtype,
                    left,
                    MPIR_ALLGATHER_TAG,
                    comm_ptr,
                    &recv_req );
            MPIR_PVAR_INC(allgather_cuda, intra, send, recvcount, recvtype);
            mpi_errno = MPIC_Isend(((char *)mv2_cuda_allgather_store_buf + j*recvcount*recvtype_extent),
                    recvcount,
                    recvtype,
                    right,
                    MPIR_ALLGATHER_TAG,
                    comm_ptr,
                    &send_req, errflag);
            mpi_errno = MPIC_Waitall(1, &recv_req, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            mpi_errno = MPIC_Waitall(1, &send_req, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

        }

    } else { /*Recursive Doubling*/
            if (sendbuf != MPI_IN_PLACE) {
                mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                           ((char *) recvbuf +
                                            rank * recvcount * recvtype_extent),
                                           recvcount, recvtype);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
    /* This synchronization is needed because MPIR_Localcopy calls cudamemcpy
     * on the default stream (0) but subsequent MPI_Isend/Irecv calls access
     * GPU buffers using non-default streams which don't wait for the initial
     * local copy to complete*/
            if (rdma_enable_cuda && cuda_initialized
                && rdma_cuda_nonblocking_streams) {
                    CUDA_CHECK(cudaEventRecord(cuda_nbstream_sync_event, 0));
                    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, cuda_nbstream_sync_event, 
                              0));
            }
            
            curr_cnt = recvcount;
            
            mask = 0x1;
            i = 0;
            

            dst = rank ^ mask;
            dst_tree_root = dst >> i;
            dst_tree_root <<= i;
            
            my_tree_root = rank >> i;
            my_tree_root <<= i;

        /* F: saving an MPI_Aint into an int */
            send_offset = my_tree_root * recvcount * recvtype_extent;
            recv_offset = dst_tree_root * recvcount * recvtype_extent;
            
            if (dst < comm_size) {
                MPIU_Memcpy_CUDA((void*)((char *)mv2_cuda_allgather_store_buf + rank*recvcount*recvtype_extent), 
                                        (void*)((char *)recvbuf + rank*recvcount*recvtype_extent), 
                                        recvcount * recvtype_extent, 
                                        cudaMemcpyDeviceToHost);

                MPIR_PVAR_INC(allgather_cuda, intra, recv, (mask)*recvcount, recvtype);
                mpi_errno = MPIC_Irecv( ((char *)mv2_cuda_allgather_store_buf + recv_offset),
                                        (mask)*recvcount, 
                                        recvtype, 
                                        dst, 
                                        MPIR_ALLGATHER_TAG,
                                        comm_ptr,
                                        &recv_req );
                MPIR_PVAR_INC(allgather_cuda, intra, send, curr_cnt, recvtype);
                mpi_errno = MPIC_Isend(((char *)mv2_cuda_allgather_store_buf + send_offset),
                                        curr_cnt, 
                                        recvtype, 
                                        dst, 
                                        MPIR_ALLGATHER_TAG,
                                        comm_ptr,
                                        &send_req, errflag);

                mpi_errno = MPIC_Waitall(1, &recv_req, &status, errflag);
                if (mpi_errno) {
                           /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                MPIU_Memcpy_CUDA_Async((void*)((char *)recvbuf + recv_offset),
                                    (void*)((char *)mv2_cuda_allgather_store_buf + recv_offset),
                                    (mask)*recvcount*recvtype_extent,
                                    cudaMemcpyHostToDevice,
                                    stream_h2d );

                mpi_errno = MPIC_Waitall(1, &send_req, &status, errflag);
                if (mpi_errno) {
                           /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                curr_cnt += mask*recvcount; 
            }

            mask <<= 1;
            i++;

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

        /* FIXME: saving an MPI_Aint into an int */
                send_offset = my_tree_root * recvcount * recvtype_extent;
                recv_offset = dst_tree_root * recvcount * recvtype_extent;
                
                if (dst < comm_size) {
                    if (mask == comm_size/2) {
                        MPIR_PVAR_INC(allgather_cuda, intra, recv, (mask)*recvcount, recvtype);
                        mpi_errno = MPIC_Irecv( ((char *)recvbuf + recv_offset),
                                                (mask)*recvcount, 
                                                recvtype, 
                                                dst, 
                                                MPIR_ALLGATHER_TAG,
                                                comm_ptr,
                                                &recv_req );
                    } else {
                        MPIR_PVAR_INC(allgather_cuda, intra, recv, (mask)*recvcount, recvtype);
                        mpi_errno = MPIC_Irecv( ((char *)mv2_cuda_allgather_store_buf + recv_offset),
                                                (mask)*recvcount, 
                                                recvtype, 
                                                dst, 
                                                MPIR_ALLGATHER_TAG,
                                                comm_ptr,
                                                &recv_req );
                    }
                    MPIR_PVAR_INC(allgather_cuda, intra, send, curr_cnt, recvtype);                        
                    mpi_errno = MPIC_Isend(((char *)mv2_cuda_allgather_store_buf + send_offset),
                                            curr_cnt, 
                                            recvtype, 
                                            dst, 
                                            MPIR_ALLGATHER_TAG,
                                            comm_ptr,
                                            &send_req, errflag);
                    mpi_errno = MPIC_Waitall(1, &recv_req, &status, errflag);
                    if (mpi_errno) {
                               /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }

                    if (mask < comm_size/2) {
                        MPIU_Memcpy_CUDA_Async(((void*) ((char *)recvbuf + recv_offset)),
                                            (void *)((char *)mv2_cuda_allgather_store_buf + recv_offset),
                                            (mask)*recvcount*recvtype_extent,
                                            cudaMemcpyHostToDevice,
                                            stream_h2d );
                    }
                    mpi_errno = MPIC_Waitall(1, &send_req, &status, errflag);
                    if (mpi_errno) {
                               /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    
                    curr_cnt += mask*recvcount;
                }
                
                mask <<= 1;
                i++;
            }
    }

    /* wait for the receive copies into the device to complete */
    cudaerr = cudaEventRecord(cuda_nbstream_sync_event, stream_h2d);
    if (cudaerr != cudaSuccess) {
        mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME,
                __LINE__, MPI_ERR_OTHER, "**cudaEventRecord", 0);
        return mpi_errno;
    }
    cudaEventSynchronize(cuda_nbstream_sync_event);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_fail:
    MPIR_TIMER_END(coll,allgather,cuda);
    return (mpi_errno);
}
/* end:nested */
#endif /* #if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
#endif /*#ifdef(_ENABLE_CUDA_)*/
