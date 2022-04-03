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

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scan_algo);

#undef FUNCNAME
#define FUNCNAME MPIR_Scan_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scan_MV2(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scan_algo, 1);

#ifdef _ENABLE_CUDA_
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;
    const char *temp_sendbuf = sendbuf;
    int stride = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);

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

    mpi_errno = MPIR_Scan(sendbuf, recvbuf, count, datatype,
			  op, comm_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type){
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
    
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
    
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
