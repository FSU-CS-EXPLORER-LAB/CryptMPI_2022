/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* This is the machine-independent implementation of scatterv. The algorithm is:

   Algorithm: Linear

   Since the array of sendcounts is valid only on the root, we cannot
   do a tree algorithm without first communicating the sendcounts to
   other processes. Therefore, we simply use a linear algorithm for the
   scatter, which takes (p-1) steps versus lgp steps for the tree
   algorithm. The bandwidth requirement is the same for both algorithms.

   Cost = (p-1).alpha + n.((p-1)/p).beta
*/


#undef FUNCNAME
#define FUNCNAME MPIR_ENC_Scatterv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_ENC_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                                 MPI_Datatype recvtype, int root, MPIR_Comm * comm_ptr,
                                 MPIR_Errflag_t * errflag)
{
    int rank, comm_size, mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent;
    int i, reqs;
    
    MPIR_Request **reqarray;
    MPI_Status *starray;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_CHKLMEM_DECL(2);

    rank = comm_ptr->rank;

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPIR_Datatype_get_extent_macro(sendtype, extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf + extent);

        MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof(MPIR_Request *),
                            mpi_errno, "reqarray", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof(MPI_Status), mpi_errno,
                            "starray", MPL_MEM_BUFFER);

        reqs = 0;
        int dest = 0;
        int t;
        unsigned long max_out_len, ciphertext_sendbuf_len;

        for (i = 0; i < comm_size; i++) {
            if (sendcounts[i]) {
                if ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (i == rank)) {
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
             MPIR_Datatype_get_extent_macro(recvtype, extent);
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
    MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}



#undef FUNCNAME
#define FUNCNAME MPIR_Scatterv_allcomm_linear
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatterv_allcomm_linear(const void *sendbuf, const int *sendcounts, const int *displs,
                                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                                 MPI_Datatype recvtype, int root, MPIR_Comm * comm_ptr,
                                 MPIR_Errflag_t * errflag)
{
    int rank, comm_size, mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent;
    int i, reqs;
    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(2);

    rank = comm_ptr->rank;

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPIR_Datatype_get_extent_macro(sendtype, extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf + extent);

        MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof(MPIR_Request *),
                            mpi_errno, "reqarray", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof(MPI_Status), mpi_errno,
                            "starray", MPL_MEM_BUFFER);

        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcounts[i]) {
                if ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (i == rank)) {
                    if (recvbuf != MPI_IN_PLACE) {
                        mpi_errno = MPIR_Localcopy(((char *) sendbuf + displs[rank] * extent),
                                                   sendcounts[rank], sendtype,
                                                   recvbuf, recvcount, recvtype);
                        if (mpi_errno)
                            MPIR_ERR_POP(mpi_errno);
                    }
                } else {
                    mpi_errno = MPIC_Isend(((char *) sendbuf + displs[i] * extent),
                                           sendcounts[i], sendtype, i,
                                           MPIR_SCATTERV_TAG, comm_ptr, &reqarray[reqs++], errflag);
                    if (mpi_errno)
                        MPIR_ERR_POP(mpi_errno);
                }
            }
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS)
            MPIR_ERR_POP(mpi_errno);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = starray[i].MPI_ERROR;
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag =
                            MPIX_ERR_PROC_FAILED ==
                            MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
        /* --END ERROR HANDLING-- */
    }

    else if (root != MPI_PROC_NULL) {   /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcount) {
            mpi_errno = MPIC_Recv(recvbuf, recvcount, recvtype, root,
                                  MPIR_SCATTERV_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }


  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
