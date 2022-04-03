/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/*
 * Algorithm: Recursive Doubling
 *
 * We use this algorithm in the case of user-defined ops because in this case
 * derived datatypes are allowed, and the user could pass basic datatypes on
 * one process and derived on another as long as the type maps are the same.
 * Breaking up derived datatypes to do the reduce-scatter is tricky.
 *
 * Cost = lgp.alpha + n.lgp.beta + n.lgp.gamma
 

#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_recursive_doubling
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_recursive_doubling(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    if (security_approach==2005 && init_phase == 0)
        return MPIR_Allreduce_intra_recursive_doubling_SEC(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_intra_recursive_doubling_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}
*/

#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_recursive_doubling_SEC
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_recursive_doubling_SEC(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_intra_recursive_doubling_SEC\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif      
    MPIR_CHKLMEM_DECL(3);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;    
    int mask, dst, is_commutative, pof2, newrank, rem, newdst;
    MPI_Aint true_extent, true_lb, extent;
    
    
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    //MPIR_CHKLMEM_DECL(2);



    is_commutative = MPIR_Op_is_commutative(op);

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

        // Mohsen

    int pos, max_out_len, enc_recv_sz;
    unsigned long ciphertext_sendbuf_len, dec_count;
    MPI_Status *starray;
    MPIR_Request **reqarray;

    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, 2  * sizeof(MPI_Status), mpi_errno,
                        "starray", MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, 2  * sizeof(MPIR_Request *), mpi_errno,
                        "reqarray", MPL_MEM_BUFFER);

    // End

    MPIR_Ensure_Aint_fits_in_pointer(count * MPL_MAX(extent, true_extent));
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->pof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even 
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);*/

            printf("Send mid rank = %d\n",rank);    fflush(stdout);                      
            
            RAND_bytes(large_send_buffer, NONCE_SIZE);
            max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                                    &ciphertext_sendbuf_len, max_out_len,
                                    (large_send_buffer), NONCE_SIZE,
                                    (char *)recvbuf, count*extent,
                                    NULL, 0))
                {
                    printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)
                        fprintf(stderr,COLOR_GREEN"RD_NP ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, dst);
                }
#endif
            
            mpi_errno = MPIC_Isend(large_send_buffer, count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, rank + 1,
											  MPIR_ALLREDUCE_TAG, comm_ptr,
											  &reqarray[1], errflag);
            
            mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);  

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag); */

            printf("Recv mid rank = %d\n",rank);fflush(stdout); 

            mpi_errno = MPIC_Irecv(large_recv_buffer, count*extent+NONCE_SIZE+ENC_MSG_TAG_SIZE, MPI_CHAR, rank - 1,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &reqarray[0]);                       
            
            mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
                                &dec_count, (count*extent),
                                (large_recv_buffer), NONCE_SIZE,
                                (large_recv_buffer + NONCE_SIZE), (unsigned long)((count*extent) + ENC_MSG_TAG_SIZE),
                                NULL, 0))
            {
                printf("RD_NP Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, dst);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
        else
        {
            if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"RD_NP MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
            fflush(stdout);
        }
#endif    
            
            
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
             * into tmp_buf 
            mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                      dst, MPIR_ALLREDUCE_TAG, tmp_buf,
                                      count, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);*/

            mpi_errno = MPIC_Irecv(large_recv_buffer, count*extent+NONCE_SIZE+ENC_MSG_TAG_SIZE, MPI_CHAR, dst,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &reqarray[0]);                          
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);
				
				RAND_bytes(large_send_buffer, NONCE_SIZE);
				max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
				if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
									   &ciphertext_sendbuf_len, max_out_len,
									   (large_send_buffer), NONCE_SIZE,
									   (char *)recvbuf, count*extent,
									   NULL, 0))
					{
						printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG)
							fprintf(stderr,COLOR_GREEN"RD_NP ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, dst);
					}
#endif
				
				mpi_errno = MPIC_Isend(large_send_buffer, count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, dst,
											  MPIR_ALLREDUCE_TAG, comm_ptr,
											  &reqarray[1], errflag);

				
				mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 

				 if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
									   &dec_count, (count*extent),
									   (large_recv_buffer), NONCE_SIZE,
									   (large_recv_buffer + NONCE_SIZE), (unsigned long)((count*extent) + ENC_MSG_TAG_SIZE),
									   NULL, 0))
					{
						printf("RD_NP Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, dst);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"RD_NP MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
					fflush(stdout);
				}
#endif          
				mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);  

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
                mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            } else {
                /* op is noncommutative and the order is not right */
                mpi_errno = MPIR_Reduce_local(recvbuf, tmp_buf, count, datatype, op);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                /* copy result back into recvbuf */
                mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype, recvbuf, count, datatype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
            mask <<= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2){   /* odd 
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);*/
            
            printf("Send Final mid rank = %d\n",rank);    fflush(stdout);                      
            
            RAND_bytes(large_send_buffer, NONCE_SIZE);
            max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                                    &ciphertext_sendbuf_len, max_out_len,
                                    (large_send_buffer), NONCE_SIZE,
                                    (char *)recvbuf, count*extent,
                                    NULL, 0))
                {
                    printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)
                        fprintf(stderr,COLOR_GREEN"RD_NP ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, dst);
                }
#endif
            
            mpi_errno = MPIC_Isend(large_send_buffer, count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, rank - 1,
											  MPIR_ALLREDUCE_TAG, comm_ptr,
											  &reqarray[1], errflag);
            
            mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);  
        
        } else {    /* even 
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);*/

            printf("Recv Final mid rank = %d\n",rank);fflush(stdout); 

            mpi_errno = MPIC_Irecv(large_recv_buffer, count*extent+NONCE_SIZE+ENC_MSG_TAG_SIZE, MPI_CHAR, rank + 1,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &reqarray[0]);                       
            
            mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                &dec_count, (count*extent),
                                (large_recv_buffer), NONCE_SIZE,
                                (large_recv_buffer + NONCE_SIZE), (unsigned long)((count*extent) + ENC_MSG_TAG_SIZE),
                                NULL, 0))
            {
                printf("RD_NP Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, dst);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
        else
        {
            if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"RD_NP MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
            fflush(stdout);
        }
#endif 
        
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}



#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_recursive_doubling
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_recursive_doubling(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_intra_recursive_doubling\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif        
    MPIR_CHKLMEM_DECL(1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank, rem, newdst;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    is_commutative = MPIR_Op_is_commutative(op);

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_Ensure_Aint_fits_in_pointer(count * MPL_MAX(extent, true_extent));
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->pof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
             * into tmp_buf */
            mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                      dst, MPIR_ALLREDUCE_TAG, tmp_buf,
                                      count, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
                mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            } else {
                /* op is noncommutative and the order is not right */
                mpi_errno = MPIR_Reduce_local(recvbuf, tmp_buf, count, datatype, op);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                /* copy result back into recvbuf */
                mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype, recvbuf, count, datatype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
            mask <<= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        else    /* even */
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
