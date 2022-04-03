/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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

/* added by abu naser */
#include "secure_allgather.h"
/* end of add */

/* -- Begin Profiling Symbol Block for routine MPI_Scatterv */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Scatterv = PMPI_Scatterv
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Scatterv  MPI_Scatterv
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Scatterv as PMPI_Scatterv
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
                 __attribute__((weak,alias("PMPI_Scatterv")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Scatterv
#define MPI_Scatterv PMPI_Scatterv

/* This is the default implementation of scatterv. The algorithm is:
   
   Algorithm: MPI_Scatterv

   Since the array of sendcounts is valid only on the root, we cannot
   do a tree algorithm without first communicating the sendcounts to
   other processes. Therefore, we simply use a linear algorithm for the
   scatter, which takes (p-1) steps versus lgp steps for the tree
   algorithm. The bandwidth requirement is the same for both algorithms.

   Cost = (p-1).alpha + n.((p-1)/p).beta

   Possible improvements: 

   End Algorithm: MPI_Scatterv
*/

/* not declared static because it is called in intercomm. reduce_scatter */
/* MPIR_Scatterv performs an scatterv using point-to-point messages.
   This is intended to be used by device-specific implementations of
   scatterv.  In all other cases MPIR_Scatterv_impl should be used. */
#undef FUNCNAME
#define FUNCNAME MPIR_Scatterv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int rank, comm_size, mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent;
    int      i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    rank = comm_ptr->rank;
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPID_INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPID_Datatype_get_extent_macro(sendtype, extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf + extent);

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **, comm_size * sizeof(MPID_Request *), mpi_errno, "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof(MPI_Status), mpi_errno, "starray");

        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcounts[i]) {
                if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
                    if (recvbuf != MPI_IN_PLACE) {
                        mpi_errno = MPIR_Localcopy(((char *)sendbuf+displs[rank]*extent), 
                                                   sendcounts[rank], sendtype,
                                                   recvbuf, recvcount, recvtype);
                        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    }
                }
                else {
                    mpi_errno = MPIC_Isend(((char *)sendbuf+displs[i]*extent),
                                              sendcounts[i], sendtype, i,
                                              MPIR_SCATTERV_TAG, comm_ptr, &reqarray[reqs++], errflag);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
            }
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) MPIR_ERR_POP(mpi_errno);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = starray[i].MPI_ERROR;
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
        /* --END ERROR HANDLING-- */
    }

    else if (root != MPI_PROC_NULL) { /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcount) {
            mpi_errno = MPIC_Recv(recvbuf,recvcount,recvtype,root,
                                     MPIR_SCATTERV_TAG,comm_ptr,MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }
    
    
fn_exit:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

/* added by abu naser */
#undef FUNCNAME
#define FUNCNAME MPIR_ENC_Scatterv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_ENC_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int rank, comm_size, mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent;
    int      i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    rank = comm_ptr->rank;
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPID_INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPID_Datatype_get_extent_macro(sendtype, extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf + extent);

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **, comm_size * sizeof(MPID_Request *), mpi_errno, "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof(MPI_Status), mpi_errno, "starray");

        reqs = 0;
        int dest = 0;
        int t;
        unsigned long max_out_len, ciphertext_sendbuf_len;

        for (i = 0; i < comm_size; i++) {
            if (sendcounts[i]) {
                if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
                    if (recvbuf != MPI_IN_PLACE) {
                        mpi_errno = MPIR_Localcopy(((char *)sendbuf+displs[rank]*extent), 
                                                   sendcounts[rank], sendtype,
                                                   recvbuf, recvcount, recvtype);
                        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    }
                }
                else {
                    t =  sendcounts[i] * extent;
                    unsigned int temp_data = (unsigned int)t;
                    ciphertext_sendbuf[dest] = (temp_data >> 24) & 0xFF;
                    ciphertext_sendbuf[dest+1] = (temp_data >> 16) & 0xFF;
                    ciphertext_sendbuf[dest+2] = (temp_data >> 8) & 0xFF;
                    ciphertext_sendbuf[dest+3] = temp_data & 0xFF;
                    RAND_bytes(ciphertext_sendbuf + dest + 4, 12);
                    max_out_len = t+16;

                    if (!EVP_AEAD_CTX_seal(global_ctx, (ciphertext_sendbuf + dest + 4 + 12),
                                           &ciphertext_sendbuf_len, max_out_len,
                                           (ciphertext_sendbuf + dest + 4), 12,
                                           (sendbuf+displs[i]*extent), t,
                                           NULL, 0))
                    {
                        printf("Error in encryption enc scatterv: [rank %d ] scatter direct dest %d\n",
                               rank, i);
                        fflush(stdout);
                    }
#if 0
                                else
                                {
                                    printf(COLOR_GREEN"Enc Scatterv:: [rank %d  dest %d] Encrypted data size %lu"COLOR_RESET"\n",
                                           rank,  i ,  ciphertext_sendbuf_len);
                                    fflush(stdout);
                                }
#endif

                    mpi_errno = MPIC_Isend(ciphertext_sendbuf+dest,
                                              (t+16+12+4), MPI_BYTE, i,
                                              MPIR_SCATTERV_TAG, comm_ptr, &reqarray[reqs++], errflag);
                      dest += t + 16 + 12 + 4;                          
                    /*mpi_errno = MPIC_Isend(((char *)sendbuf+displs[i]*extent),
                                              sendcounts[i], sendtype, i,
                                              MPIR_SCATTERV_TAG, comm_ptr, &reqarray[reqs++], errflag);*/
                    if (mpi_errno)
                        MPIR_ERR_POP(mpi_errno);
                }
            }
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) MPIR_ERR_POP(mpi_errno);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = starray[i].MPI_ERROR;
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
        /* --END ERROR HANDLING-- */
    }

    else if (root != MPI_PROC_NULL) { /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcount) {
            /*mpi_errno = MPIC_Recv(recvbuf,recvcount,recvtype,root,
                                     MPIR_SCATTERV_TAG,comm_ptr,MPI_STATUS_IGNORE, errflag);*/
             MPID_Datatype_get_extent_macro(recvtype, extent);
            int t = recvcount * extent;
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t+16+12+4), MPI_BYTE, root,
                                  MPIR_SCATTERV_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);

            unsigned int temp_data = ((unsigned char)ciphertext_recvbuf[3] << 0) | ((unsigned char)ciphertext_recvbuf[2] << 8) | ((unsigned char)ciphertext_recvbuf[1] << 16) | ((unsigned char)ciphertext_recvbuf[0] << 24);    
            t = (int) temp_data;
            unsigned long max_out_len = t;
            unsigned long count;

            if (!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf+4), 12,
                                   (ciphertext_recvbuf + 12+4), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error enc scatterv: scatter two-level-direct t %d rank %d temp_data %u\n",t,rank,temp_data);
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatterv two-level-direct:: [rank %d ] Decrypted data size %lu" COLOR_RESET "\n", rank, count);
                fflush(stdout);
            }
#endif

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }
    
    
fn_exit:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
/* end of add */

/* MPIR_Scatterv_impl should be called by any internal component that
   would otherwise call MPI_Scatterv.  This differs from MPIR_Scatterv
   in that this will call the coll_fns version if it exists.  This
   function replaces NMPI_Scatterv. */
#undef FUNCNAME
#define FUNCNAME MPIR_Scatterv_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatterv_impl(const void *sendbuf, const int *sendcounts, const int *displs,
                       MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;

#if defined(_ENABLE_CUDA_)
    int i, rank, comm_size;
    int sendbuf_on_device = 0, recvbuf_on_device = 0;
    int sendtype_extent = 0, recvtype_extent = 0, total_count = 0;
    int total_size = 0, total_msgs = 0, avg_size = 0;
    int *recv_displs;

    if (rdma_enable_cuda) {
        rank = comm_ptr->rank;
        if (comm_ptr->comm_kind == MPID_INTRACOMM) {
            comm_size = comm_ptr->local_size;
        } else {
            comm_size = comm_ptr->remote_size;
        }

        if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
            ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
            sendbuf_on_device = is_device_buffer(sendbuf);
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            for (i = 0; i < comm_size; i++) {
                total_count += sendcounts[i];
            }
            total_size = total_count * sendtype_extent;
            total_msgs = comm_size;
        }

        if (recvbuf != MPI_IN_PLACE) {
            recvbuf_on_device = is_device_buffer(recvbuf);
            MPID_Datatype_get_extent_macro(recvtype, recvtype_extent); 
            total_size += recvcount * recvtype_extent;
            total_msgs += 1;
        }

        avg_size = total_size / total_msgs;

        if ((sendbuf_on_device || recvbuf_on_device) &&
             rdma_cuda_use_naive &&
             avg_size <= rdma_cuda_scatterv_naive_limit) {

            recv_displs = (int *) MPIU_Malloc(sizeof(int));
            if (recv_displs == NULL) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }
            recv_displs[0] = 0;

            if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
                ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
                mpi_errno = cuda_stage_alloc_v ((void **)&sendbuf, (int *)sendcounts, sendtype,
                         (int **)&displs, comm_size,
                         &recvbuf, &recvcount, recvtype,
                         &recv_displs, 1,
                         sendbuf_on_device, recvbuf_on_device,
                         rank);
            } else {
                mpi_errno = cuda_stage_alloc_v (NULL, NULL, sendtype,
                         NULL, 0, 
                         &recvbuf, &recvcount, recvtype,
                         &recv_displs, 1, 
                         sendbuf_on_device, recvbuf_on_device,
                         rank);
            }
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }
#endif       
 
    if (comm_ptr->coll_fns != NULL && comm_ptr->coll_fns->Scatter != NULL) {
	/* --BEGIN USEREXTENSION-- */
	mpi_errno = comm_ptr->coll_fns->Scatterv(sendbuf, sendcounts, displs,
                                                 sendtype, recvbuf, recvcount,
                                                 recvtype, root, comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	/* --END USEREXTENSION-- */
    } else {
        mpi_errno = MPIR_Scatterv(sendbuf, sendcounts, displs, sendtype,
                                  recvbuf, recvcount, recvtype,
                                  root, comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda) {
        if ((sendbuf_on_device || recvbuf_on_device) &&
             rdma_cuda_use_naive &&
             avg_size <= rdma_cuda_scatterv_naive_limit) {

            if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
                ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
                cuda_stage_free_v ((void **)&sendbuf, (int *)sendcounts, sendtype,
                         (int **)&displs, comm_size,
                         &recvbuf, &recvcount, recvtype,
                         &recv_displs, 1,
                         sendbuf_on_device, recvbuf_on_device,
                         rank);
            } else {
                cuda_stage_free_v (NULL, NULL, sendtype,
                         NULL, 0,
                         &recvbuf, &recvcount, recvtype,
                         &recv_displs, 1,
                         sendbuf_on_device, recvbuf_on_device,
                         rank);
            }

            MPIU_Free(recv_displs);

        }
    }
#endif

 fn_exit:
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Scatterv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@

MPI_Scatterv - Scatters a buffer in parts to all processes in a communicator

Input Parameters:
+ sendbuf - address of send buffer (choice, significant only at 'root') 
. sendcounts - integer array (of length group size) 
specifying the number of elements to send to each processor  
. displs - integer array (of length group size). Entry 
 'i'  specifies the displacement (relative to sendbuf  from
which to take the outgoing data to process  'i' 
. sendtype - data type of send buffer elements (handle) 
. recvcount - number of elements in receive buffer (integer) 
. recvtype - data type of receive buffer elements (handle) 
. root - rank of sending process (integer) 
- comm - communicator (handle) 

Output Parameters:
. recvbuf - address of receive buffer (choice) 

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_COUNT
.N MPI_ERR_TYPE
.N MPI_ERR_BUFFER
@*/
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_SCATTERV);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_COLL_FUNC_ENTER(MPID_STATE_MPI_SCATTERV);

    /* Validate parameters, especially handles needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPIR_ERRTEST_COMM(comm, mpi_errno);
	}
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPID_Comm_get_ptr( comm, comm_ptr );

    /* Validate parameters and objects (post conversion) */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPID_Datatype *sendtype_ptr=NULL, *recvtype_ptr=NULL;
            int i, comm_size, rank;
	    
            MPID_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            if (mpi_errno != MPI_SUCCESS) goto fn_fail;

            if (comm_ptr->comm_kind == MPID_INTRACOMM) {
		MPIR_ERRTEST_INTRA_ROOT(comm_ptr, root, mpi_errno);
                rank = comm_ptr->rank;
                comm_size = comm_ptr->local_size;

                if (rank == root) {
                    for (i=0; i<comm_size; i++) {
                        MPIR_ERRTEST_COUNT(sendcounts[i], mpi_errno);
                        MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
                    }
                    if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                        MPID_Datatype_valid_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    for (i=0; i<comm_size; i++) {
                        if (sendcounts[i] > 0) {
                            MPIR_ERRTEST_USERBUFFER(sendbuf,sendcounts[i],sendtype,mpi_errno);
                            break;
                        }
                    }  
                    for (i=0; i<comm_size; i++) {
                        if (sendcounts[i] > 0) {
                            MPIR_ERRTEST_SENDBUF_INPLACE(sendbuf, sendcounts[i], mpi_errno);
                            break;
                        }
                    }
                    /* catch common aliasing cases */
                    if (recvbuf != MPI_IN_PLACE && sendtype == recvtype && sendcounts[comm_ptr->rank] != 0 && recvcount != 0) {
                        int sendtype_size;
                        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
                        MPIR_ERRTEST_ALIAS_COLL(recvbuf, (char*)sendbuf + displs[comm_ptr->rank]*sendtype_size, mpi_errno);
                    }
                }
                else 
                    MPIR_ERRTEST_RECVBUF_INPLACE(recvbuf, recvcount, mpi_errno);

                if (recvbuf != MPI_IN_PLACE) {
                    MPIR_ERRTEST_COUNT(recvcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
                    if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                        MPID_Datatype_valid_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_USERBUFFER(recvbuf,recvcount,recvtype,mpi_errno);
                }
            }

            if (comm_ptr->comm_kind == MPID_INTERCOMM) {
		MPIR_ERRTEST_INTER_ROOT(comm_ptr, root, mpi_errno);
                if (root == MPI_ROOT) {
                    comm_size = comm_ptr->remote_size;
                    for (i=0; i<comm_size; i++) {
                        MPIR_ERRTEST_COUNT(sendcounts[i], mpi_errno);
                        MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
                    }
                    if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                        MPID_Datatype_valid_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    for (i=0; i<comm_size; i++) {
                        if (sendcounts[i] > 0) {
                            MPIR_ERRTEST_SENDBUF_INPLACE(sendbuf, sendcounts[i], mpi_errno);
                            MPIR_ERRTEST_USERBUFFER(sendbuf,sendcounts[i],sendtype,mpi_errno);
                            break;
                        }
                    }
                }       
                else if (root != MPI_PROC_NULL) {
                    MPIR_ERRTEST_COUNT(recvcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
                    if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                        MPID_Datatype_valid_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_RECVBUF_INPLACE(recvbuf, recvcount, mpi_errno);
                    MPIR_ERRTEST_USERBUFFER(recvbuf,recvcount,recvtype,mpi_errno);
                }
            }

            if (mpi_errno != MPI_SUCCESS) goto fn_fail;
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPIR_Scatterv_impl(sendbuf, sendcounts, displs, sendtype,
                                   recvbuf, recvcount, recvtype,
                                   root, comm_ptr, &errflag);
    if (mpi_errno) goto fn_fail;

    /* ... end of body of routine ... */
    
  fn_exit:
    MPID_MPI_COLL_FUNC_EXIT(MPID_STATE_MPI_SCATTERV);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_scatterv",
	    "**mpi_scatterv %p %p %p %D %p %d %D %d %C", sendbuf, sendcounts, displs, sendtype,
	    recvbuf, recvcount, recvtype, root, comm);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
