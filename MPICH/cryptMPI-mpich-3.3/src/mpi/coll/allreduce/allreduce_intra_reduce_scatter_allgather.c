/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Algorithm: Rabenseifner's Algorithm
 *
 * Restrictions: Built-in ops only
 *
 * This algorithm is from http://www.hlrs.de/mpi/myreduce.html.
.
 * This algorithm implements the allreduce in two steps: first a
 * reduce-scatter, followed by an allgather. A recursive-halving algorithm
 * (beginning with processes that are distance 1 apart) is used for the
 * reduce-scatter, and a recursive doubling algorithm is used for the
 * allgather. The non-power-of-two case is handled by dropping to the nearest
 * lower power-of-two: the first few even-numbered processes send their data to
 * their right neighbors (rank+1), and the reduce-scatter and allgather happen
 * among the remaining power-of-two processes. At the end, the first few
 * even-numbered processes get the result from their right neighbors.
 *
 * For the power-of-two case, the cost for the reduce-scatter is:
 *
 * lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma.
 *
 * The cost for the allgather:
 *
 * lgp.alpha +.n.((p-1)/p).beta
 *
 * Therefore, the total cost is:
 *
 * Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma
 *
 * For the non-power-of-two case:
 *
 * Cost = (2.floor(lgp)+2).alpha + (2.((p-1)/p) + 2).n.beta + n.(1+(p-1)/p).gamma
 


#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_reduce_scatter_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_reduce_scatter_allgather(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    if (security_approach==2005 && init_phase == 0)
        return MPIR_Allreduce_intra_reduce_scatter_allgather_SEC_NodeAware(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_intra_reduce_scatter_allgather_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}
*/


#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_reduce_scatter_allgather_SEC_NodeAware
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_reduce_scatter_allgather_SEC_NodeAware(const void *sendbuf,
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
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_intra_reduce_scatter_allgather_SEC_NodeAware\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif      
    MPIR_CHKLMEM_DECL(4);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    long unsigned int ciphertext_sendbuf_len, max_out_len, dec_count;

    comm_size = comm_ptr->local_size;
    int my_rank = comm_ptr->rank;

    if (comm_ptr->is_blocked == 1)
        rank = my_rank;
    else
        rank = comm_rank_list[my_rank];

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

    // Mohsen

    // MPID_Node_id_t my_node_id, remote_node_id;
    int my_node_id, remote_node_id;

    MPID_Get_node_id(comm_ptr, my_rank, &my_node_id);

    unsigned int enc_chunk_size = -1;
    MPIR_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
	
	int context_id, *enc_disps;;
    context_id = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
        MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;

    // End
    
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
											  &send_req_ptr, errflag);
            
            // mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);  
            mpi_errno = MPIC_Wait(send_req_ptr, errflag);

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
										  &recv_req_ptr);                       
            
            // mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 
            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);

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

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN);
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {
        MPIR_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof(int), mpi_errno, "displacements", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(enc_disps, int *, pof2 * sizeof(int), mpi_errno, "enc_displacements", MPL_MEM_BUFFER); 

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
            enc_disps[0] = 0;
        for (i = 1; i < pof2; i++){
            disps[i] = disps[i - 1] + cnts[i - 1];
            enc_disps[i] = enc_disps[i-1] + (cnts[i-1]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        }

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

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

            /* Send data from recvbuf. Recv into tmp_buf 
            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);*/                    
            
            MPID_Get_node_id(comm_ptr, my_rank, &my_node_id);
            MPID_Get_node_id(comm_ptr,comm_rank_list_back[dst], &remote_node_id);
            
            // if (comm_ptr->is_blocked == 1)
               //  MPID_Get_node_id(comm_ptr,comm_rank_list_back[dst], &remote_node_id);
            // if (comm_ptr->node_comm != NULL) {


            // }

            
			
            if ((my_node_id == remote_node_id)){		
                                                
                mpi_errno = MPID_Irecv((char *) tmp_buf + disps[recv_idx] * extent, recv_cnt, datatype, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);					
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);						
                
                mpi_errno = MPID_Isend((char *) recvbuf + disps[send_idx] * extent, send_cnt, datatype, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);					
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                                                
            } else {

                enc_chunk_size = send_cnt*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;
                	
                MPID_Irecv(large_recv_buffer, enc_chunk_size, MPI_CHAR, comm_rank_list_back[dst],  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);
                                                                
                RAND_bytes(large_send_buffer, NONCE_SIZE);
                max_out_len = enc_chunk_size; 
                if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                                &ciphertext_sendbuf_len, max_out_len,
                                (&large_send_buffer[0]), NONCE_SIZE,
                                (char *) recvbuf + (disps[send_idx] * extent), recv_cnt*extent,
                                NULL, 0))
                {
                    printf("Error in encryption: RSA rank = %d\n",comm_ptr->rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"RSA : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, comm_rank_list_back[dst]);
                }
#endif
                
                MPID_Isend(large_send_buffer, ciphertext_sendbuf_len+12, MPI_CHAR, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);	
                
                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                
                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf + (disps[recv_idx] * extent),
                                &dec_count, max_out_len,
                                (&large_recv_buffer[0]), NONCE_SIZE,
                                large_recv_buffer + NONCE_SIZE, ciphertext_sendbuf_len,
                                NULL, 0))
                {
                    if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA_NP_RS : rank = %d source =%d  iter = %d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, comm_rank_list_back[dst],i,count,mask,recv_cnt);
                    //fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_BLUE"RSA_NP_RS : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, comm_rank_list_back[dst]);
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

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPIR_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            if (my_node_id != remote_node_id){
				
                mpi_errno = MPIC_Wait(send_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                //MPID_Request_release(send_req_ptr);
                //MPID_Request_release(recv_req_ptr);
            }

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        /* There is at least one inter-node communication  */
        int all_intra_node=0;
        int first_intra_node=0;
        
        // MPID_Get_node_id(comm_ptr, rank, &my_node_id);
        // MPID_Get_node_id(comm_ptr, dst, &remote_node_id);

        MPID_Get_node_id(comm_ptr, my_rank, &my_node_id);
        MPID_Get_node_id(comm_ptr,comm_rank_list_back[dst], &remote_node_id);
        
        if (my_node_id != remote_node_id) {

            RAND_bytes(large_recv_buffer + enc_disps[recv_idx] , NONCE_SIZE);
            max_out_len = recv_cnt * extent + ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer + (enc_disps[recv_idx] ) + NONCE_SIZE),
                            &ciphertext_sendbuf_len, max_out_len,
                            (&large_recv_buffer[enc_disps[recv_idx]]), NONCE_SIZE,
                            (char *) recvbuf + disps[recv_idx] * extent, recv_cnt * extent,
                            NULL, 0))
            {
                printf("Error in encryption: RSA rank = %d\n",comm_ptr->rank);
                fflush(stdout);
            }			
        } else {
            all_intra_node = 1;
            first_intra_node = 1;
        }

        int seg_count=0;
        int same_node = 0;
        int basic_seg_size = recv_cnt * extent;
        int basic_cipher_size = basic_seg_size + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        int step_power = 0;
        int last_recv_idx=0;

        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

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

                seg_count = recv_idx-send_idx;

            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
                
                seg_count = last_idx-send_idx;
            }

            enc_chunk_size = (unsigned int) ( (send_cnt*extent) + (seg_count*(12+16)));
            max_out_len = enc_chunk_size;  
                                        
            // MPID_Get_node_id(comm_ptr, rank, &my_node_id);
            // MPID_Get_node_id(comm_ptr, dst, &remote_node_id);

            MPID_Get_node_id(comm_ptr, my_rank, &my_node_id);
            MPID_Get_node_id(comm_ptr,comm_rank_list_back[dst], &remote_node_id);
            
            if ((my_node_id == remote_node_id)){
                
                if ((same_node == 0)  &&  (all_intra_node == 0)) {
                    
                    for (i=0; i<step_power; i++){

                            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent) +i*basic_seg_size),
                                            &dec_count, max_out_len,
                                            (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                                            large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
                                            NULL, 0))
                            {
                                if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG-First-Same_node) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, comm_rank_list_back[dst],i,recv_cnt/2);
                            }
#if ENABLE_SECURE_MPI_DEBUG
                            else
                            {
                                if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  RSA_NP (ALG-First-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, comm_rank_list_back[dst]);
                                fflush(stdout);
                            }
#endif 								
                    }						
                    same_node++;
                }

            /* mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag); */
            
                MPID_Irecv((char *) recvbuf + disps[recv_idx] * extent, recv_cnt, datatype, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);				
                                            
                MPID_Isend((char *) recvbuf + disps[send_idx] * extent, send_cnt, datatype, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);
                
                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                                            
            } else {

                MPID_Irecv(large_recv_buffer + (enc_disps[recv_idx]), enc_chunk_size, MPI_CHAR, comm_rank_list_back[dst],  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);					
                                        
                MPID_Isend(large_recv_buffer + (enc_disps[send_idx]), enc_chunk_size, MPI_CHAR, comm_rank_list_back[dst], MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

                if (step_power > 0)
                {						
                    for (i=0; i<step_power; i++){												
                
                        if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent)+i*basic_seg_size),
                                        &dec_count, max_out_len,
                                        (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                                        large_recv_buffer + (enc_disps[last_recv_idx]) + (i * basic_cipher_size + NONCE_SIZE), basic_cipher_size - NONCE_SIZE,
                                        NULL, 0))
                        {
                            if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, comm_rank_list_back[dst],i,count);
                        }
#if ENABLE_SECURE_MPI_DEBUG
                        else
                        {
                            if (ENABLE_SECURE_DEBUG) fprintf(COLOR_GREEN"RSA_NP (ALG-Not-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                            fflush(stdout);
                        }
#endif 								
                    }
                } 
                
                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
                
                // last_recv_idx=recv_idx;
                                    
            }
            
            
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            if ( step_power == 0) step_power=1;
			else step_power = step_power * 2;
				
			last_recv_idx=recv_idx;
            
            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }

        /* When All the communications were Inter-node ones */
			if ((same_node == 0) && (all_intra_node == 0)) {
					
				for (i=0; i<step_power; i++){
				
					if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent)+i*basic_seg_size),
								   &dec_count, max_out_len,
								   (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
								   large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
								   NULL, 0))
					{
						if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG-Last-out) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, dst,i,recv_cnt);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  RSA_NP (ALG-Last-Dec): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
						fflush(stdout);
					}
#endif 							
				}
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
											  &send_req_ptr, errflag);
            
            // mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);  
            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
        
        } else {    /* even 
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);*/

            printf("Recv Final mid rank = %d\n",rank);fflush(stdout); 

            mpi_errno = MPIC_Irecv(large_recv_buffer, count*extent+NONCE_SIZE+ENC_MSG_TAG_SIZE, MPI_CHAR, rank + 1,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &recv_req_ptr);                       
            
            // mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 
            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);

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
#define FUNCNAME MPIR_Allreduce_intra_reduce_scatter_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_reduce_scatter_allgather(const void *sendbuf,
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
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_intra_reduce_scatter_allgather\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif      
    MPIR_CHKLMEM_DECL(3);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

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

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN);
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {
        MPIR_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof(int), mpi_errno, "displacements",
                            MPL_MEM_BUFFER);

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

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

            /* Send data from recvbuf. Recv into tmp_buf */
            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
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

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPIR_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

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

            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
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
