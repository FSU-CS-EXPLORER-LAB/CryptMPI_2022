/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Algorithm: Bruck's
 *
 * This algorithm is from the IEEE TPDS Nov 97 paper by Jehoshua Bruck
 * et al.  It is a variant of the disemmination algorithm for barrier.
 * It takes ceiling(lg p) steps.
 *
 * Cost = lgp.alpha + n.((p-1)/p).beta
 * where n is total size of data gathered on each process.
 */

 
#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_Bruck_SEC
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_Bruck_SEC(const void *sendbuf,
                                int sendcount,
                                MPI_Datatype sendtype,
                                void *recvbuf,
                                int recvcount,
                                MPI_Datatype recvtype,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_Allgather_Bruck_SEC\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    int pof2, src, rem;
    void *tmp_buf = NULL;
    int curr_cnt, dst;
	int node_id, dst_node_id, src_node_id;

    MPIR_CHKLMEM_DECL(1);

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_Bruck_SEC\n");
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    /* allocate a temporary buffer of the same size as recvbuf. */

    /* get true extent of recvtype */
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    recvbuf_extent = recvcount * comm_size * (MPL_MAX(recvtype_true_extent, recvtype_extent));

    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, recvbuf_extent, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    /* copy local data to the top of tmp_buf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype, tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        mpi_errno = MPIR_Localcopy(((char *) recvbuf +
                                    rank * recvcount * recvtype_extent),
                                   recvcount, recvtype, tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* do the first \floor(\lg p) steps */
    /********************* Added by Mehran ***************/
    int last_encrypted_index = 0; //Nothing encrypted so far
    char *in, *out, *rbuf, *sbuf;
    int recently_received=0, s_msg_size, r_msg_size;
    if(security_approach==2005){
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        curr_cnt = 1;
    }else{
        curr_cnt = recvcount;
    }
    int i=0;
    unsigned long  ciphertext_len = 0, count=0, in_size=0;
    in_size = (unsigned long)(recvcount * recvtype_extent);
    unsigned long max_out_len = (unsigned long) (16 + in_size);
    /* do the first \floor(\lg p) steps */
    pof2 = 1;
    while (pof2 <= comm_size / 2) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        if(security_approach==2005){
            //Naive+
            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
            MPID_Get_node_id(comm_ptr, src, &src_node_id);
            if(node_id != dst_node_id){
                //encrypt from tmp_buf to large_recv_buffer and set the send buffer from large_recv_buffer and s_msg_size
                for(; last_encrypted_index<curr_cnt; ++last_encrypted_index){
                    in = (char*)((char*) tmp_buf + last_encrypted_index * recvcount * recvtype_extent);
                    out = (char*)((char*) large_recv_buffer + last_encrypted_index * (recvcount * recvtype_extent + 16+12));
                    // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in Naive+ encryption: allgather Bruck (Default-I)\n");
                        fflush(stdout);
                    }

                }

                sbuf = (char*) large_recv_buffer;
                s_msg_size = (recvcount * recvtype_extent + 16+12);

            }else{
                //set the send buffer from tmp_buf and s_msg_size
                sbuf = (char*) tmp_buf;
                s_msg_size = (recvcount * recvtype_extent);
            }
            if(node_id != src_node_id){
                //set the recvbuffer from large_recv_buffer and r_msg_size
                rbuf = (char*)((char*) large_recv_buffer + curr_cnt * (recvcount * recvtype_extent + 16+12));
                r_msg_size = (recvcount * recvtype_extent + 16+12);
            }else{
                //set the recv buffer from tmp_buf and r_msg_size
                rbuf = (char*)((char*) tmp_buf + curr_cnt * recvcount * recvtype_extent);
                r_msg_size = (recvcount * recvtype_extent);
            }

            mpi_errno = MPIC_Sendrecv(sbuf, curr_cnt * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, curr_cnt * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, errflag);

            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(sbuf, curr_cnt * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, curr_cnt * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }


            if(node_id != src_node_id){
                //decrypt recvd messages from large_recv_buffer to temp_buf
                int decryption_index = curr_cnt;
                for(; decryption_index<2*curr_cnt; ++decryption_index){
                    in = (char*)((char*) large_recv_buffer + decryption_index * (recvcount * recvtype_extent + 16+12));
                    out = (char*)((char*) tmp_buf + decryption_index * recvcount * recvtype_extent);
                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){

                        printf("Error in Naive+ decryption: allgather Bruck (Default-I) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }

            }
        //end naive+
        }else{

    
            pof2 = 1;
            while (pof2 <= comm_size / 2) {
                src = (rank + pof2) % comm_size;
                dst = (rank - pof2 + comm_size) % comm_size;

                mpi_errno = MPIC_Sendrecv(tmp_buf, curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) tmp_buf + curr_cnt * recvtype_extent),
                                        curr_cnt, recvtype,
                                        src, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
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
        curr_cnt *= 2;
        pof2 *= 2;
    }

    /* if comm_size is not a power of two, one more step is needed */

    rem = comm_size - pof2;
    if (rem) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;
        if(security_approach==2005){
            //Naive+
            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
            MPID_Get_node_id(comm_ptr, src, &src_node_id);
            if(node_id != dst_node_id){
                //encrypt from tmp_buf to large_recv_buffer and set the send buffer from large_recv_buffer and s_msg_size
                for(; last_encrypted_index<rem; ++last_encrypted_index){
                    in = (char*)((char*) tmp_buf + last_encrypted_index * recvcount * recvtype_extent);
                    out = (char*)((char*) large_recv_buffer + last_encrypted_index * (recvcount * recvtype_extent + 16+12));
                    // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in Naive+ encryption: allgather Bruck (Default-II)\n");
                        fflush(stdout);
                    }

                }

                sbuf = (char*) large_recv_buffer;
                s_msg_size = (recvcount * recvtype_extent + 16+12);

            }else{
                //set the send buffer from tmp_buf and s_msg_size
                sbuf = (char*) tmp_buf;
                s_msg_size = (recvcount * recvtype_extent);
            }
            if(node_id != src_node_id){
                //set the recvbuffer from large_recv_buffer and r_msg_size
                rbuf = (char*)((char*) large_recv_buffer + curr_cnt * (recvcount * recvtype_extent + 16+12));
                r_msg_size = (recvcount * recvtype_extent + 16+12);
            }else{
                //set the recv buffer from tmp_buf and r_msg_size
                rbuf = (char*)((char*) tmp_buf + curr_cnt * recvcount * recvtype_extent);
                r_msg_size = (recvcount * recvtype_extent);
            }

            mpi_errno = MPIC_Sendrecv(sbuf, rem * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, rem * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, errflag);

            /************* For MPIC_Sendrecv_Plus ************
            mpi_errno = MPIC_Sendrecv_Plus(sbuf, rem * s_msg_size, MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG,
                                        rbuf, rem * r_msg_size,
                                        MPI_CHAR, src, MPIR_ALLGATHER_TAG,
                                        comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i++]), errflag);
            *************************************************/


            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }


            if(node_id != src_node_id){
                //decrypt recvd messages from large_recv_buffer to temp_buf
                int decryption_index = curr_cnt;
                for(; decryption_index<curr_cnt+rem; ++decryption_index){
                    in = (char*)((char*) large_recv_buffer + decryption_index * (recvcount * recvtype_extent + 16+12));
                    out = (char*)((char*) tmp_buf + decryption_index * recvcount * recvtype_extent);
                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){

                        printf("Error in Naive+ decryption: allgather Bruck (Default-II) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                        fflush(stdout);        
                    }
                }

            }
        //end naive+
        }else{

            mpi_errno = MPIC_Sendrecv(tmp_buf, rem * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG,
                                    ((char *) tmp_buf + curr_cnt * recvtype_extent),
                                    rem * recvcount, recvtype,
                                    src, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
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

    /* Rotate blocks in tmp_buf down by (rank) blocks and store
     * result in recvbuf. */

    mpi_errno = MPIR_Localcopy(tmp_buf, (comm_size - rank) * recvcount,
                               recvtype, (char *) recvbuf + rank * recvcount * recvtype_extent,
                               (comm_size - rank) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (rank) {
        mpi_errno = MPIR_Localcopy((char *) tmp_buf +
                                   (comm_size - rank) * recvcount * recvtype_extent,
                                   rank * recvcount, recvtype, recvbuf, rank * recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
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
#define FUNCNAME MPIR_Allgather_intra_brucks
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_intra_brucks(const void *sendbuf,
                                int sendcount,
                                MPI_Datatype sendtype,
                                void *recvbuf,
                                int recvcount,
                                MPI_Datatype recvtype,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allgather_intra_brucks\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    int pof2, src, rem;
    void *tmp_buf = NULL;
    int curr_cnt, dst;

    MPIR_CHKLMEM_DECL(1);

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_intra_brucks\n");
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    /* allocate a temporary buffer of the same size as recvbuf. */

    /* get true extent of recvtype */
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    recvbuf_extent = recvcount * comm_size * (MPL_MAX(recvtype_true_extent, recvtype_extent));

    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, recvbuf_extent, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    /* copy local data to the top of tmp_buf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype, tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        mpi_errno = MPIR_Localcopy(((char *) recvbuf +
                                    rank * recvcount * recvtype_extent),
                                   recvcount, recvtype, tmp_buf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* do the first \floor(\lg p) steps */

    curr_cnt = recvcount;
    pof2 = 1;
    while (pof2 <= comm_size / 2) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        mpi_errno = MPIC_Sendrecv(tmp_buf, curr_cnt, recvtype, dst,
                                  MPIR_ALLGATHER_TAG,
                                  ((char *) tmp_buf + curr_cnt * recvtype_extent),
                                  curr_cnt, recvtype,
                                  src, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        curr_cnt *= 2;
        pof2 *= 2;
    }

    /* if comm_size is not a power of two, one more step is needed */

    rem = comm_size - pof2;
    if (rem) {
        src = (rank + pof2) % comm_size;
        dst = (rank - pof2 + comm_size) % comm_size;

        mpi_errno = MPIC_Sendrecv(tmp_buf, rem * recvcount, recvtype,
                                  dst, MPIR_ALLGATHER_TAG,
                                  ((char *) tmp_buf + curr_cnt * recvtype_extent),
                                  rem * recvcount, recvtype,
                                  src, MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Rotate blocks in tmp_buf down by (rank) blocks and store
     * result in recvbuf. */

    mpi_errno = MPIR_Localcopy(tmp_buf, (comm_size - rank) * recvcount,
                               recvtype, (char *) recvbuf + rank * recvcount * recvtype_extent,
                               (comm_size - rank) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (rank) {
        mpi_errno = MPIR_Localcopy((char *) tmp_buf +
                                   (comm_size - rank) * recvcount * recvtype_extent,
                                   rank * recvcount, recvtype, recvbuf, rank * recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
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
