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

#include "mpiimpl.h"
#include "coll_shmem.h"
#include "math.h"
#include "unistd.h"

#if defined(_ENABLE_CUDA_)

typedef enum _send_stat_ {
    COPY_COMPLETE,
    SEND_COMPLETE
} send_stat;

cudaEvent_t *send_events = NULL, *recv_event = NULL;
int send_events_count = 0;

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_alltoall_cuda);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_intra_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_intra_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_intra_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_intra_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoall_cuda_count_recv);

#undef FUNCNAME
#define FUNCNAME MPIR_Alltoall_CUDA_cleanup
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_CUDA_cleanup () 
{
    int mpi_errno = MPI_SUCCESS; 
    int i = 0;

    if (send_events) {
        for(i = 0; i < send_events_count; i++) {
            CUDA_CHECK(cudaEventDestroy(send_events[i]));
        }
        MPIU_Free(send_events); 
    }
    if (recv_event) {
        CUDA_CHECK(cudaEventDestroy(*recv_event));
        MPIU_Free(recv_event); 
    }

    return mpi_errno; 
}

#undef FUNCNAME
#define FUNCNAME MPIR_Alltoall_CUDA_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_CUDA_intra_MV2( 
    const void *sendbuf, 
    int sendcount, 
    MPI_Datatype sendtype, 
    void *recvbuf, 
    int recvcount, 
    MPI_Datatype recvtype, 
    MPID_Comm *comm_ptr, 
    MPIR_Errflag_t *errflag )
{
    MPIR_TIMER_START(coll,alltoall,cuda);
    cudaError_t cudaerr = cudaSuccess;
    int mpi_errno=MPI_SUCCESS;
    int dst, rank, comm_size;
    int i, j, procs_in_block, flag, sblock = 0, rblock = 0;
    int begin = 0, end = 0, bytes_copied = 0, disp = 0, avail = 0;
    int recvreq_complete = 0,  num_sbufs = 0, num_rbufs = 0;
    int sbufs_filled = 0, rbufs_filled = 0, bufs_send_initiated = 0, bufs_recvd = 0;
    send_stat *send_complete = NULL;
    MPI_Aint sendtype_extent, recvtype_extent;
    MPID_Request **sendreq = NULL, **recvreq = NULL;
    MPI_Request *request_ptrs = NULL;
    MPI_Status *sendstat = NULL, *recvstat = NULL;
    MPIDI_CH3U_COLL_SRBuf_element_t **send_buf = NULL, **recv_buf = NULL;

    /*get comm size and rank*/
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get extent of send and recv types */
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );

    /*calculate the block count based on the messages size and cuda block size 
      for pipelining*/
    sblock = (int) rdma_cuda_block_size/(sendcount*sendtype_extent);
    sblock = sblock < comm_size ? sblock : comm_size; 
    MPIU_Assert(sblock != 0 && sblock <= comm_size);

    rblock = (int) rdma_cuda_block_size/(recvcount*recvtype_extent);
    rblock = rblock < comm_size ? rblock : comm_size; 
    MPIU_Assert(rblock != 0 && rblock <= comm_size);

    /*allocate send buffers*/
    num_sbufs = ceil((double)comm_size/sblock); 
    send_buf = (MPIDI_CH3U_COLL_SRBuf_element_t **) 
                    MPIU_Malloc(sizeof(MPIDI_CH3U_COLL_SRBuf_element_t *)*num_sbufs);
    for (i = 0; i < num_sbufs; i++) {
        MPIDI_CH3U_COLL_SRBuf_alloc(send_buf[i]);
    }
    /*allocate recv buffers*/
    num_rbufs = ceil((double)comm_size/rblock); 
    recv_buf = (MPIDI_CH3U_COLL_SRBuf_element_t **) 
                    MPIU_Malloc(sizeof(MPIDI_CH3U_COLL_SRBuf_element_t *)*num_rbufs);
    for (i = 0; i < num_rbufs; i++) { 
        MPIDI_CH3U_COLL_SRBuf_alloc(recv_buf[i]);
    }

    /*create events to make copies in and out of the GPU asynchronous*/
    if (send_events_count < num_sbufs) {
        if (send_events) { 
            for(i = 0; i < send_events_count; i++) { 
                cudaEventDestroy(send_events[i]);
            }
            MPIU_Free(send_events);
        }
        send_events = (cudaEvent_t *) MPIU_Malloc(sizeof(cudaEvent_t) * num_sbufs); 
        for (i=0; i<num_sbufs; i++) { 
            cudaerr = cudaEventCreateWithFlags(&send_events[i], cudaEventDisableTiming);
            if(cudaerr != cudaSuccess) {
                mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME,
                        __LINE__, MPI_ERR_OTHER, "**nomem", 0);

                MPIR_TIMER_END(coll,alltoall,cuda);
                return mpi_errno;
            }
        }
        send_events_count = num_sbufs;
    }
    if (!recv_event) {
        recv_event = (cudaEvent_t *) MPIU_Malloc(sizeof(cudaEvent_t));
        cudaerr = cudaEventCreateWithFlags(recv_event, cudaEventDisableTiming);
        if(cudaerr != cudaSuccess) {
            mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME,
                    __LINE__, MPI_ERR_OTHER, "**nomem", 0);
            MPIR_TIMER_END(coll,alltoall,cuda);
            return mpi_errno;
        }
    }

    /*allocate send and receive requests, statuses and counters*/
    sendreq = (MPID_Request **) MPIU_Malloc(comm_size * sizeof(MPID_Request*));
    if (!sendreq) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }

    request_ptrs = (MPI_Request *) MPIU_Malloc(comm_size * sizeof(MPI_Request));
    if (!request_ptrs) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }

    recvreq = (MPID_Request **) MPIU_Malloc(comm_size * sizeof(MPID_Request*));
    if (!recvreq) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }

    sendstat = (MPI_Status *) MPIU_Malloc(comm_size * sizeof(MPI_Status));
    if (!sendstat) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }

    recvstat = (MPI_Status *) MPIU_Malloc(comm_size * sizeof(MPI_Status));
    if (!recvstat) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }

    send_complete = (send_stat *) MPIU_Malloc(sizeof(send_stat) * num_sbufs);
    if (NULL == send_complete) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                            FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0 );
        MPIR_TIMER_END(coll,alltoall,cuda);
        return mpi_errno;
    }
    MPIU_Memset(send_complete, 0, sizeof(send_stat)*num_sbufs);

    /*initiate asynchronous copies for all the data to be sent*/
    for (i = 0; i < comm_size; i+=sblock) {  

        if (i == 0) { 
            begin = (rank - sblock + 1 + comm_size) % comm_size;
            end = (rank + comm_size) % comm_size;
        } else { 
            avail = (comm_size - i); 
            if (avail < sblock) { 
                end = (begin - 1 + comm_size) % comm_size;  
                begin = (begin - avail + comm_size) % comm_size;
            } else {
                end = (begin - 1 + comm_size) % comm_size;  
                begin = (begin - sblock + comm_size) % comm_size;
            }
        }

        if (begin <= end) {
            MPIU_Memcpy_CUDA_Async(send_buf[sbufs_filled]->buf, 
                    ((char *) sendbuf
                     + begin*sendcount*sendtype_extent),
                    (end - begin + 1)*sendcount*sendtype_extent,
                    cudaMemcpyDeviceToHost, stream_d2h);
        } else {
            MPIU_Memcpy_CUDA_Async(send_buf[sbufs_filled]->buf, 
                    ((char *) sendbuf
                     + begin*sendcount*sendtype_extent),
                    (comm_size - begin)*sendcount*sendtype_extent,
                    cudaMemcpyDeviceToHost, stream_d2h);

            bytes_copied = (comm_size - begin)*sendcount*sendtype_extent; 

            MPIU_Memcpy_CUDA_Async(((char *)send_buf[sbufs_filled]->buf 
                        + bytes_copied),
                    sendbuf,
                    (end + 1)*sendcount*sendtype_extent,
                    cudaMemcpyDeviceToHost, stream_d2h);
        }

        cudaerr = cudaEventRecord(send_events[sbufs_filled], stream_d2h);
        if (cudaerr != cudaSuccess) {
            mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME,
                    __LINE__, MPI_ERR_OTHER, "**cudaEventRecord", 0);
            return mpi_errno;
        }
        sbufs_filled++;
    }

    /*issue all irecvs ahead*/
    for (i = 0; i < comm_size; i+=rblock) {

        procs_in_block = (comm_size - i) < rblock ? (comm_size - i) : rblock;

        for (j = 0; j < procs_in_block; j++) {
            dst = (rank + j + i) % comm_size;
            disp = j*recvcount*recvtype_extent;

            MPIR_PVAR_INC(alltoall_cuda, intra, recv, recvcount, recvtype);
            mpi_errno = MPIC_Irecv((char *) recv_buf[bufs_recvd]->buf +
                    disp,
                    recvcount, recvtype, dst,
                    MPIR_ALLTOALL_TAG, comm_ptr,
                    &recvreq[i+j]);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        bufs_recvd++;
    }

    /*now, wait on each event and issue the correspondings sends */
    while (rbufs_filled < num_rbufs || bufs_send_initiated < num_sbufs) { 

        if (bufs_send_initiated < num_sbufs) { 
            for (i=0; i<num_sbufs; i++) { 
                if (send_complete[i] > 0) continue;
                cudaerr = cudaErrorNotReady;
                cudaerr = cudaEventQuery(send_events[i]);
                if (cudaerr == cudaSuccess) {
                    send_complete[i] = COPY_COMPLETE; 
                    break;
                }
            }

            if ( i < num_sbufs && send_complete[i] == COPY_COMPLETE) {
                send_complete[i] = SEND_COMPLETE;
                procs_in_block = (comm_size - i*sblock) < sblock ? (comm_size - i*sblock) : sblock;
                for (j=0; j<procs_in_block; ++j ) {
                    dst = (rank - j - (i * sblock) + comm_size) % comm_size;
                    disp = (procs_in_block - j - 1)*sendcount*sendtype_extent;

                    MPIR_PVAR_INC(alltoall_cuda, intra, send, sendcount, sendtype);
                    mpi_errno = MPIC_Isend((char *) send_buf[i]->buf +
                        disp,
                        sendcount, sendtype, dst,
                        MPIR_ALLTOALL_TAG, comm_ptr,
                        &sendreq[i*sblock + j], errflag);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
                bufs_send_initiated++;
            } 
        }

        if (rbufs_filled < num_rbufs) { 
            flag = 0;
            request_ptrs[recvreq_complete] = recvreq[recvreq_complete]->handle;
            mpi_errno = MPIR_Test_impl(&request_ptrs[recvreq_complete], &flag, MPI_STATUS_IGNORE);
            if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
                MPIR_ERR_POP(mpi_errno);
            }

            if (flag) { 
                recvreq_complete++;
            }

            /*we initiate copy to the device of a set of receives have completed*/    
            if (flag && 
                (((recvreq_complete > 0) &&
                  (recvreq_complete % rblock) == 0) ||  
                 (recvreq_complete == comm_size))) { 

                if (recvreq_complete < comm_size) { 
                    procs_in_block = rblock; 
                } else { 
                    procs_in_block = (comm_size % rblock > 0) ? (comm_size % rblock) : rblock; 
                }

                begin = (rank + recvreq_complete - procs_in_block) % comm_size;
                end = (begin + procs_in_block - 1) % comm_size;

                if (begin <= end) {
                    MPIU_Memcpy_CUDA_Async(((char *) recvbuf
                                + begin*recvcount*recvtype_extent),
                            recv_buf[rbufs_filled]->buf, 
                            procs_in_block*recvcount*recvtype_extent,
                            cudaMemcpyHostToDevice, stream_h2d);
                } else {
                    MPIU_Memcpy_CUDA_Async(((char *) recvbuf 
                                + begin*recvcount*recvtype_extent),
                            recv_buf[rbufs_filled]->buf,  
                            (comm_size - begin)*recvcount*recvtype_extent,
                            cudaMemcpyHostToDevice, stream_h2d);

                    bytes_copied = (comm_size - begin)*recvcount*recvtype_extent;

                    MPIU_Memcpy_CUDA_Async(recvbuf,
                            ((char *) recv_buf[rbufs_filled]->buf  
                             + bytes_copied), 
                            (end + 1)*recvcount*recvtype_extent,
                            cudaMemcpyHostToDevice, stream_h2d);
                }
                rbufs_filled++;
            }
        }
    } 

    /* wait for ss sends and recvs to finish: */
    mpi_errno = MPIC_Waitall(comm_size, sendreq, sendstat, errflag);
    if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
          MPIR_ERR_POP(mpi_errno);
    }

    /* wait for the receive copies into the device to complete */
    cudaerr = cudaEventRecord(*recv_event, stream_h2d);
    if (cudaerr != cudaSuccess) {
        mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME,
                __LINE__, MPI_ERR_OTHER, "**cudaEventRecord", 0);
        return mpi_errno;
    }
    cudaEventSynchronize(*recv_event);

    MPIU_Free(send_complete);             
    MPIU_Free(request_ptrs);
    MPIU_Free(recvreq);
    MPIU_Free(sendreq);
    MPIU_Free(sendstat);
    MPIU_Free(recvstat);

    for (i = 0; i < num_sbufs; i++) {
        MPIDI_CH3U_COLL_SRBuf_free(send_buf[i]);
    }
    for (i = 0; i < num_rbufs; i++) {
        MPIDI_CH3U_COLL_SRBuf_free(recv_buf[i]);
    }
    MPIU_Free(recv_buf);
    MPIU_Free(send_buf);

 fn_fail:    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
    MPIR_TIMER_END(coll,alltoall,cuda);
    return (mpi_errno);
}
#endif /*#ifdef _ENABLE_CUDA_*/

/* end:nested */
