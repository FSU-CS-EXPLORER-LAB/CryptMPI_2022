/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"


/*
#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_smp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_smp(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    if (security_approach==2005 && inter_allreduce_tuning>5 && init_phase == 0)
        return MPIR_Allreduce_intra_concurrent_SEC_NodeAware(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_intra_smp_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}
 */


#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_concurrent_SEC_NodeAware
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_concurrent_SEC_NodeAware(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                             MPIR_Errflag_t * errflag)
{
    char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){	
    printf("[allreduce CONCUR_INTER_METHOD =%d rank = %d host = %s count = %d  Method = %d] Func: MPIR_Allreduce_intra_concurrent_SEC_NodeAware\n",
    CONCUR_INTER_METHOD,comm_ptr->rank,hostname,count,inter_allreduce_tuning);fflush(stdout);}
#endif
    MPIR_CHKLMEM_DECL(4);       
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Datatype_get_extent_macro(datatype, extent);
    void * recv_chunk = NULL, * send_chunk = NULL, * comp_chunk = NULL, * tmp_buf;
    MPIR_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);    
    MPIR_Ensure_Aint_fits_in_pointer(count * MPL_MAX(extent, true_extent));
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, count * (MPL_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer", MPL_MEM_BUFFER);
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);   
    int context_id;
    context_id = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
        MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;

    int i;
    int type_size = extent;
    int size = comm_ptr->local_size;
    int rank =  comm_ptr->rank;
    //int rank_index = rank;
    int rank_index = comm_rank_list[rank];
    int shmem_rank = comm_ptr->node_comm->rank;
    int ppn = comm_ptr->node_comm->local_size; // number of ranks per node
    int node = (int) (size / ppn); // number of nodes
    leader_cnt = ppn;

    int chunk_size_inter  = (count * type_size) / (leader_cnt*node);
	int chunk_size_intra  = (count * type_size) / ppn;
    int chunk_size_leader  = (count * type_size) / leader_cnt;

    int chunk_count_inter = count / (leader_cnt*node);
    int chunk_count_intra = count / ppn;
    int chunk_count_leader = count / leader_cnt;

    // int shmem_rank = rank_index - (node_id * ppn);
	int inter_prefix = (shmem_rank * count * type_size) / leader_cnt;
    int max_out_len;
	unsigned long ciphertext_sendbuf_len, dec_count;	
	unsigned int chunk_enc_size = chunk_size_inter + ENC_MSG_TAG_SIZE + NONCE_SIZE; 	
	unsigned int recv_enc_chunk , send_enc_chunk; 
    int node_id, left_node_id, right_node_id;    

    MPID_Get_node_id(comm_ptr, rank, &node_id);

    int right_index = (shmem_rank + 1);
	if (right_index >= ppn) right_index = 0;
	
	int	right = right_index;
		
	int	left_index = (shmem_rank - 1);
	if (left_index < 0) left_index += ppn;
	
	int	left = left_index;
	
	/* if (rank==3) 
		for (i = 0; i < comm_ptr->remote_size; i++)
            fprintf(stderr,"&& comm_rank_list[%d] =%d ",i,comm_rank_list[i]); */

    // Phase 1:                

    /* on each node, each process copy sendbuf to shmem_buffer */
    

    mpi_errno = MPIR_Localcopy((void*)((char*) sendbuf), count, datatype, 
                               (void*)((char*) (shmem_buffer + shmem_rank * count * extent)), count, datatype);
            
    mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
        goto fn_fail;
    }

    recv_chunk  = sendbuf + (shmem_rank * chunk_size_leader);

    for (i = 0; i < ppn; i++) {

        if (i != shmem_rank) {
                        
                comp_chunk  = shmem_buffer + (i * count * type_size) + (shmem_rank * chunk_size_leader);      

                mpi_errno = MPIR_Reduce_local(comp_chunk, recv_chunk, chunk_count_leader, datatype, op);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);              
#if 0
#if defined(HAVE_CXX_BINDING)
    if (is_cxx_uop) {
        (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                        chunk_count_leader, datatype, uop);
    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
        (*uop) (comp_chunk, recv_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
#endif
        }
    }

    mpi_errno = MPIR_Localcopy((void*)((char*) (sendbuf + (shmem_rank*chunk_size_leader))), (count / leader_cnt), datatype, 
                                (void*)((char*) (tmp_buf + (shmem_rank*chunk_size_leader))), (count / leader_cnt), datatype);

    
#if 0
    /* on each node, do a reduce to the local root */
    if (comm_ptr->node_comm != NULL) {
        /* take care of the MPI_IN_PLACE case. For reduce,
         * MPI_IN_PLACE is specified only on the root;
         * for allreduce it is specified on all processes. */

        if ((sendbuf == MPI_IN_PLACE) && (comm_ptr->node_comm->rank != 0)) {
            /* IN_PLACE and not root of reduce. Data supplied to this
             * allreduce is in recvbuf. Pass that as the sendbuf to reduce. */

            mpi_errno =
                MPIR_Reduce(recvbuf, NULL, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        } else {
            mpi_errno =
                MPIR_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } else {
        /* only one process on the node. copy sendbuf to recvbuf */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }
#endif    

    
    // Phase 2:
    
    void * rep_chunk = NULL;

    rep_chunk = shmem_buffer;
	// if (CONCUR_AllGTHER_METHOD == 2) rep_chunk = shmem_buffer;   else rep_chunk = recvbuf;

    if (Print_Ring && SHOW_FINAL_RESULT && SHOW_FINAL_RESULT && count==8192) fprintf(stderr,"Ran_leader = %d  host = % s chunk_size_leader = %d\n",rank,hostname,chunk_size_leader); 

    if (CONCUR_INTER_METHOD == 1)  // RD-method
    {
        int mask = ppn;  //leader_cnt
        int dst, dst_index = 0;

        int pof2 = comm_ptr->pof2;

        chunk_enc_size = chunk_size_leader + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        while (mask < pof2) {
			
            dst_index = rank_index ^ mask;

            dst  = comm_rank_list[dst_index];
			
			// fprintf(stderr,"rank_index=%d  Rank=[%d %d]=dst   dst_index=%d\n",rank_index,rank,dst,dst_index);
			
            mpi_errno = MPID_Irecv(large_recv_buffer, chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);  
			
			RAND_bytes(large_send_buffer, NONCE_SIZE);
			
			max_out_len = chunk_size_leader+ENC_MSG_TAG_SIZE; 
			if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer+NONCE_SIZE),
										   &ciphertext_sendbuf_len, max_out_len,
										   (&large_send_buffer[0]), NONCE_SIZE,
										   (char *)send_chunk, chunk_size_leader,
										   NULL, 0))
						{
							printf("Error in encryption: Concur-Inter: RD  rank = %d\n",comm_ptr->rank);
							fflush(stdout);
						}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"Concur-Inter: RD : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dst);
					fflush(stdout);
				}
#endif

            
			mpi_errno = MPID_Isend(large_send_buffer, chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);                       
			//if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			//if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
			
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_leader,
							   (&large_recv_buffer[0]), NONCE_SIZE,
							   large_recv_buffer+(NONCE_SIZE), (chunk_size_leader + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Concur-Inter: RD  rank = %d source = %d  mask = %d  count = %d\n",comm_ptr->rank, dst,mask,count);
				fflush(stdout);
            }


            recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

            send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            mpi_errno = MPIR_Reduce_local(recv_chunk, send_chunk, chunk_count_leader, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);              

#if 0

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (recv_chunk, send_chunk,
                                                chunk_count_leader, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (recv_chunk, send_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
#endif

            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            
            mask <<= 1;

        }

    }
    
    else if (CONCUR_INTER_METHOD == 2)  // RSA-method
    {
		 
        
        // fprintf(stderr,"CONCUR_INTER_METHOD=%d\n",CONCUR_INTER_METHOD);
        
        int step, mask, dst, i, dst_index, send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps, *enc_disps;  
        int dst_offset = ppn;     
		
		// int pof2 = comm_ptr->dev.ch.gpof2;
        

        // MPIU_CHKLMEM_MALLOC(cnts, int *, node * sizeof (int), mpi_errno, "counts");
        // MPIU_CHKLMEM_MALLOC(disps, int *, node * sizeof (int), mpi_errno, "displacements");								
        // MPIU_CHKLMEM_MALLOC(enc_disps, int *, node * sizeof (int), mpi_errno, "enc_displacements");

        MPIR_CHKLMEM_MALLOC(cnts, int *, node * sizeof(int), mpi_errno, "counts", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(disps, int *, node * sizeof(int), mpi_errno, "displacements", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(enc_disps, int *, node * sizeof(int), mpi_errno, "enc_displacements", MPL_MEM_BUFFER); 
   
        for (i = 0; i < (node - 1); i++) {
                cnts[i] = chunk_count_leader / node;
            }
        cnts[node - 1] = chunk_count_leader - (chunk_count_leader / node) * (node - 1);

        disps[0] = 0;
        enc_disps[0] = 0;
        for (i = 1; i < node; i++) {
            disps[i] = disps[i - 1] + cnts[i - 1];
            enc_disps[i] = enc_disps[i-1] + (cnts[i-1]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        }

        mask = 0x1; //ppn;
        send_idx = recv_idx = 0; //pof2*(shmem_rank)/node;
        last_idx = node; //pof2*(shmem_rank+1)/node;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        if (Print_Ring && SHOW_FINAL_RESULT && count==8192 && 0) {

            void * print_chunk = tmp_buf + (shmem_rank * chunk_size_leader); 

            int tempp = step;
            step = extent;
            step = tempp;

        } 


        while (mask < node) { 

            dst_index = rank_index ^ (mask * dst_offset);

            dst  = comm_rank_list[dst_index];

            //dst  = rank ^ mask;

            send_cnt = recv_cnt = 0;
//            if (rank_index < dst) {
            if (rank < dst) {
                send_idx = recv_idx + node / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + node / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            chunk_enc_size = send_cnt*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;

			MPID_Irecv(large_recv_buffer, chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

            RAND_bytes(large_send_buffer, NONCE_SIZE);
            max_out_len = chunk_enc_size; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                            &ciphertext_sendbuf_len, max_out_len,
                            (&large_send_buffer[0]), NONCE_SIZE,
                            (char *) send_chunk + (disps[send_idx] * extent), recv_cnt*extent,
                            NULL, 0))
            {
                printf("Error in encryption: ConCur_Inter : RSA rank = %d source =%d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, dst,count,mask,recv_cnt);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
//#if 1

            else
            {
                if (ENABLE_SECURE_DEBUG) 
                //if (((count==1048576) || (count==1048576/2))  && mask==2)
					fprintf(stderr,COLOR_GREEN"ConCur_Inter : RSA ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dst);
            }
#endif

            MPID_Isend(large_send_buffer, ciphertext_sendbuf_len+12, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk + (disps[recv_idx] * extent),
                            &dec_count, max_out_len,
                            (&large_recv_buffer[0]), NONCE_SIZE,
                            large_recv_buffer + NONCE_SIZE, ciphertext_sendbuf_len,
                            NULL, 0))
            {
                printf("Err in Decryption : ConCur_Inter : RSA rank = %d source =%d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, dst,count,mask,recv_cnt);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
//#if 1
            else
            {
                if (ENABLE_SECURE_DEBUG) 
				//if (((count==1048576) || (count==1048576/2))  && mask==2)
			fprintf(stderr,COLOR_BLUE"ConCur_Inter : RSA : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                fflush(stdout);
            }
#endif 
         
            if (mask == node/2){

                recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

                send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            }   

            mpi_errno = MPIR_Reduce_local((char *) recv_chunk + disps[recv_idx] * extent,
                                          (char *) send_chunk + disps[recv_idx] * extent,
                                          recv_cnt, datatype, op);
            if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);                   
            
            /* (*uop) ((char *) recv_chunk + disps[recv_idx] * extent,
                    (char *) send_chunk + disps[recv_idx] * extent,
                    &recv_cnt, &datatype); */

            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
			send_idx = recv_idx;
            mask <<= 1;

            // MPID_Request_release(recv_req_ptr);

            if (mask < node)
                last_idx = recv_idx + node / mask;

        }


        RAND_bytes(large_recv_buffer + enc_disps[recv_idx] , NONCE_SIZE);
        max_out_len = recv_cnt * extent + ENC_MSG_TAG_SIZE; 
        if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer + (enc_disps[recv_idx] ) + NONCE_SIZE),
                        &ciphertext_sendbuf_len, max_out_len,
                        (&large_recv_buffer[enc_disps[recv_idx]]), NONCE_SIZE,
                        (char *) (send_chunk + disps[recv_idx] * extent), recv_cnt * extent,
                        NULL, 0))
        {
            printf("Err in only encryption : ConCur_Inter : RSA rank = %d source =%d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, dst,count,mask,recv_cnt);
                fflush(stdout);
        }


        mask >>= 1;

        int seg_count = 0;
        int step_power = 0;
		int last_recv_idx = 0;
        int basic_seg_size = recv_cnt * extent;
		int basic_cipher_size = basic_seg_size + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        while (mask > 0) {

                dst_index = rank_index ^ (mask * dst_offset);

                dst  = comm_rank_list[dst_index];
                
                /* dst_index = rank_index ^ (mask * dst_offset);
                dst  = comm_ptr->dev.ch.rank_list[dst_index];
                comm_ptr->dev.ch.rank_list[dst_index] = ppn * internode_table[node_root] + comm_ptr->intranode_table[node_rank];   */

                send_cnt = recv_cnt = 0;
				
				int node_half = node ;
				
                if (rank < dst) {
                    /* update last_idx except on first iteration */
                    if (mask != node_half / 2) {
                        last_idx = last_idx + node_half / (mask * 2);
                    }

                    recv_idx = send_idx + node_half / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = recv_idx-send_idx;
                } else {
                    recv_idx = send_idx - node_half / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = last_idx-send_idx;
                }

                chunk_enc_size = (send_cnt*extent) + (seg_count*(12+16));
                max_out_len = chunk_enc_size;

                MPID_Irecv(large_recv_buffer + (enc_disps[recv_idx]), chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

                MPID_Isend(large_recv_buffer + (enc_disps[send_idx]), chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

                if (step_power > 0)
                {
                    for (i=0; i<step_power; i++){

                        if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *) (send_chunk + ((disps[last_recv_idx]*extent)+i*basic_seg_size)),
                                        &dec_count, max_out_len,
                                        (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                                        large_recv_buffer + (enc_disps[last_recv_idx]) + (i * basic_cipher_size + NONCE_SIZE), basic_cipher_size - NONCE_SIZE,
                                        NULL, 0))
                        {
                            printf("Err in midle Decryption : ConCur_Inter : RSA rank = %d source =%d  count = %d  mask=%d  recv_cnt=%d  i=%d\n",comm_ptr->rank, dst,count,mask,recv_cnt,i);
                            fflush(stdout);
                        }
#if ENABLE_SECURE_MPI_DEBUG
                        else
                        {
                            if (ENABLE_SECURE_DEBUG) fprintf(COLOR_GREEN"ConCur_Inter : RSA (ALG-Not-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                            fflush(stdout);
                        }
#endif 					
                    }
                } 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
                			
			    last_recv_idx=recv_idx;

                if ( step_power == 0) step_power=1;
				else step_power = step_power * 2;
				                
                // MPID_Request_release(recv_req_ptr);
				
                if ( rank > dst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
			
        }

        for (i=0; i<step_power; i++){	

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *) (send_chunk + ((disps[last_recv_idx]*extent)+i*basic_seg_size)),
                            &dec_count, max_out_len,
                            (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                            large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
                            NULL, 0))
            {
                printf("Err in last Decryption : ConCur_Inter : RSA (ALG-Last-out) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, dst,i,recv_cnt);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  ConCur_Inter : RSA (ALG-Last-Dec): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                fflush(stdout);
            }
#endif 		

    
     
        }
    }

    else if (CONCUR_INTER_METHOD == 3)  // Ring-method
    {   
        int i = 0;

        // fprintf(stderr,"CONCUR_INTER_METHOD=%d\n",CONCUR_INTER_METHOD);
		
        void * recv_enc = NULL, * send_enc = NULL;

        void * rep_buffer = recvbuf;
        
		/* compute the left and right neighbor ranks in the rank_list */
		left_index  = (size + rank_index - ppn) % size;
		right_index = (size + rank_index + ppn) % size;
		int left_Inter  = comm_rank_list[left_index];
		int right_Inter = comm_rank_list[right_index];
        
        // int left_Inter  = comm_ptr->dev.ch.rank_list[left_index];
		// int right_Inter = comm_ptr->dev.ch.rank_list[right_index];
		

        for (i = 1; i < node; i++) {

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */
    
            if (i == 1)
                send_chunk  = tmp_buf + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;
                        
            else 
                send_chunk  = rep_buffer + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;

            comp_chunk  = tmp_buf + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;   		 			

            recv_enc_chunk = ((node_id - i + node) % node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-1) + node) % node)*chunk_enc_size + inter_prefix;
							   
			mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);  
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);	

			RAND_bytes(large_recv_buffer+send_enc_chunk, NONCE_SIZE);
			max_out_len = chunk_size_inter+ENC_MSG_TAG_SIZE; 
			if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer+(send_enc_chunk + NONCE_SIZE)),
										   &ciphertext_sendbuf_len, max_out_len,
										   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
										   (char *)send_chunk, chunk_size_inter,
										   NULL, 0))
						{
							printf("Error in encryption: Red-Scat step  rank = %d\n",comm_ptr->rank);
							fflush(stdout);
						}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"Ring: first allgather step : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, right_Inter);
					fflush(stdout);
				}
#endif

			mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);                       
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            if (i == (node -1))  rep_buffer = rep_chunk;

            recv_chunk  = rep_buffer + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;
			
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
			
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_inter,
							   (&large_recv_buffer[recv_enc_chunk]), NONCE_SIZE,
							   large_recv_buffer+(recv_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Red-Scat step: rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
				fflush(stdout);
            }

            mpi_errno = MPIR_Reduce_local(comp_chunk, recv_chunk, chunk_count_inter, datatype, op);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno); 

#if 0		
			/* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count_inter, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count_inter, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
#endif
        //    MPID_Request_release(recv_req_ptr);
		
		}
			
		i=1;
		
		recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk ;
			
		send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
		recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
		send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;
						   
		mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);  

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);	

		RAND_bytes(large_recv_buffer+send_enc_chunk, NONCE_SIZE);
		max_out_len = chunk_size_inter+ENC_MSG_TAG_SIZE; 
		if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer+(send_enc_chunk + NONCE_SIZE)),
									   &ciphertext_sendbuf_len, max_out_len,
									   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
									   (char *)send_chunk, chunk_size_inter,
									   NULL, 0))
					{
						printf("Error in encryption: first allgather step  rank = %d\n",comm_ptr->rank);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
			else
			{
				if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"first allgather step : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, right_Inter);
				fflush(stdout);
			}
#endif

		mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, MPIR_ALLREDUCE_TAG,
					   comm_ptr, context_id, &send_req_ptr);                       

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);

		
		mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
		if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

		//MPID_Request_release(send_req_ptr);
		// MPID_Request_release(recv_req_ptr);
		
        for (i = 2; i < node; i++) {		

			recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;
			 						
            mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            
            mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11, comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
						
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)send_chunk,
										   &dec_count, chunk_size_inter,
										   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
										   large_recv_buffer+(send_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
										   NULL, 0))
						{
							printf("Decryption error inside allgather step rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
							fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"First allgather step : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, left_Inter);
                fflush(stdout);
            }
#endif  

			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
				
            // MPID_Request_release(recv_req_ptr);			
        }
		
		/* Last received chunk Decryption */

		if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_inter,
							   (&large_recv_buffer[recv_enc_chunk]), NONCE_SIZE,
							   large_recv_buffer+(recv_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Last allgather step: rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
				fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"First allgather step : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, left_Inter);
                fflush(stdout);
            }
#endif 
		
    }
  	   
    
#if 0    
    /* now do an IN_PLACE allreduce among the local roots of all nodes */
    if (comm_ptr->node_roots_comm != NULL) {
        mpi_errno =
            MPIR_Allreduce(MPI_IN_PLACE, recvbuf, count, datatype, op, comm_ptr->node_roots_comm,
                           errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
#endif      

// Phase 3:

//    if (ppn > 1) 
    {
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
				goto fn_fail;
			}
 
        mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)recvbuf), count, datatype);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

#if 0
    /* now broadcast the result among local processes */
    if (comm_ptr->node_comm != NULL) {
        mpi_errno = MPIR_Bcast(recvbuf, count, datatype, 0, comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
#endif

// #if ALLREDUCE_PRINT_FUN    
#if 1
    //if (SHOW_FINAL_RESULT && count==8192) {
    if (SHOW_FINAL_RESULT) {
        int s , match=1;
        char hostname1[100];    
        gethostname(hostname1, MAX_HOSTNAME_LEN);
		int step = (count/8)*extent;  
        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf+extent))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,COLOR_RED"[%d] dismatch at %d th element ([0]=%.1f  [%d]=%.1f  [%d]=%.1f) count = %d   host=%s  [end]"COLOR_RESET"\n",
                rank,s, *((float *) (recvbuf)),s-1,*((float *) (recvbuf+(s-1)*extent)),s,*((float *) (recvbuf+s*extent)),count,hostname1);
                match =0;
                break;
            }

        }
        if (match==1) {            
			// fprintf(stderr,COLOR_GREEN"[%d] is match [value = [0]=%.1f [1]=%.1f] host=%s  count=%d  [end]"COLOR_RESET"\n",comm_ptr->rank,*((float *) (recvbuf)),*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,COLOR_CYAN"[%d] is not match  step:%d recvbuf = %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  count=%d   host=%s  [end]"COLOR_RESET"\n",rank,count/8 ,*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),*((float *) (recvbuf+8*step)),count,hostname1);
    }
#endif	

    goto fn_exit;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}





#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_smp_SEC
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_smp_SEC(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                             MPIR_Errflag_t * errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
    printf("[allreduce rank = %d host = %s count = %d  SA = %d] Func: MPIR_Allreduce_intra_smp_SEC\n",
    comm_ptr->rank,hostname,count, security_approach);fflush(stdout);}
#endif       
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    /* on each node, do a reduce to the local root */
    if (comm_ptr->node_comm != NULL) {
        /* take care of the MPI_IN_PLACE case. For reduce,
         * MPI_IN_PLACE is specified only on the root;
         * for allreduce it is specified on all processes. */

        if ((sendbuf == MPI_IN_PLACE) && (comm_ptr->node_comm->rank != 0)) {
            /* IN_PLACE and not root of reduce. Data supplied to this
             * allreduce is in recvbuf. Pass that as the sendbuf to reduce. */

            mpi_errno =
                MPIR_Reduce(recvbuf, NULL, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        } else {
            mpi_errno =
                MPIR_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } else {
        /* only one process on the node. copy sendbuf to recvbuf */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    /* now do an IN_PLACE allreduce among the local roots of all nodes */
    if (comm_ptr->node_roots_comm != NULL) {           
            mpi_errno = MPIR_Allreduce_intra_recursive_doubling_SEC(MPI_IN_PLACE, recvbuf, count, datatype, op, comm_ptr->node_roots_comm, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* now broadcast the result among local processes */
    if (comm_ptr->node_comm != NULL) {
        mpi_errno = MPIR_Bcast(recvbuf, count, datatype, 0, comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
    goto fn_exit;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}




#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_smp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_smp(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                             MPIR_Errflag_t * errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_intra_smp\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif       
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    /* on each node, do a reduce to the local root */
    if (comm_ptr->node_comm != NULL) {
        /* take care of the MPI_IN_PLACE case. For reduce,
         * MPI_IN_PLACE is specified only on the root;
         * for allreduce it is specified on all processes. */

        if ((sendbuf == MPI_IN_PLACE) && (comm_ptr->node_comm->rank != 0)) {
            /* IN_PLACE and not root of reduce. Data supplied to this
             * allreduce is in recvbuf. Pass that as the sendbuf to reduce. */

            mpi_errno =
                MPIR_Reduce(recvbuf, NULL, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        } else {
            mpi_errno =
                MPIR_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } else {
        /* only one process on the node. copy sendbuf to recvbuf */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    /* now do an IN_PLACE allreduce among the local roots of all nodes */
    if (comm_ptr->node_roots_comm != NULL) {            
        
        mpi_errno = MPIR_Allreduce(MPI_IN_PLACE, recvbuf, count, datatype, op, comm_ptr->node_roots_comm, errflag);

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* now broadcast the result among local processes */
    if (comm_ptr->node_comm != NULL) {
        mpi_errno = MPIR_Bcast(recvbuf, count, datatype, 0, comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
    goto fn_exit;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
