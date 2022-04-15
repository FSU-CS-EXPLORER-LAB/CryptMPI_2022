/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/*
 * Recursive Doubling Algorithm:
 *
 * Restrictions: power-of-two no. of processes
 *
 * Cost = lgp.alpha + n.((p-1)/p).beta
 *
 * TODO: On TCP, we may want to use recursive doubling instead of the
 * Bruck's algorithm in all cases because of the pairwise-exchange
 * property of recursive doubling (see Benson et al paper in Euro
 * PVM/MPI 2003).
 */





#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Multileader_RD
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Multileader_RD(const void *sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_2lvl_Allgather_Multileader_RD\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, j, k, f;
    char *in, *out, *sbuf, *rbuf;
    MPI_Status status;
    // unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;
    
    /* get info about communicator for ranks on the same node */
    MPIR_Comm* shmem_commptr;
    shmem_commptr = comm_ptr->node_comm;

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int comm_size = comm_ptr->local_size;
    // if(rank==0)
    //     printf("MPIR_2lvl_Allgather_Multileader_RD\n");
    int p = shmem_commptr->local_size; // number of ranks per node
    int n = (int) (comm_size / p); // number of nodes

    int mask, dst_tree_root, my_tree_root, send_offset, recv_offset, send_req_idx = 0,
    last_recv_cnt = 0, nprocs_completed, offset, tmp_mask, out_index, dst,
    dst_node, my_node, tree_root, rank_index, dst_index, curr_cnt=1, recently_received;

    /************* For MPIC_Sendrecv_Plus ************
    MPIR_Request *send_req_ptr[20*(comm_size-1)];
    int s;
    for(s=0; s<comm_size-1; ++s){
        (send_req_ptr[s])=NULL;
    }
    *************************************************/

    MPI_Aint recvtype_extent, sendtype_extent;
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);

     /* First, load the "local" version in the recvbuf. */
    if (sendbuf != MPI_IN_PLACE) {
        /* compute location in receive buffer for our data */
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);

        /* copy data from send buffer to receive buffer */
        mpi_errno = MPIR_Localcopy(
            sendbuf, sendcount, sendtype,
            rbuf,    recvcount, recvtype
        );
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* Inter-Node Ring*/
    
    int node_id, left_node_id, right_node_id;    
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    my_node = node_id;

    if (comm_ptr->comm_rank_list == NULL) host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes , &comm_ptr->comm_rank_list , &comm_ptr->comm_rank_list_back);

    rank_index = comm_ptr->comm_rank_list[rank];

    //If naive+ encryption is selected, each rank needs to encrypt its local data first

    
    if(security_approach==2005){
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) recvbuf + comm_ptr->comm_rank_list[rank_index] * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + my_node * (recvcount * recvtype_extent + 12 + 16));

        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long in_size=0;
        in_size = (unsigned long)(sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        //printf("%d (%d) is going to encrypt from %d to %d\n", rank, local_rank, rank * recvcount * recvtype_extent, rank * (recvcount * recvtype_extent + 12 + 16) );
        if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, in_size,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather MultiLeader-RD\n");
                fflush(stdout);
        }
    }else{
        /** 
         * Only for the Unencrypted version
         * Copy from recv_buf to ciphertext_recvbuf
         **/
        void* rbuf = (void*)((char*) recvbuf + rank * recvcount * recvtype_extent);
        void* crbuf = (void*)((char*) ciphertext_recvbuf + my_node * recvcount * recvtype_extent);
        mpi_errno = MPIR_Localcopy(
            rbuf, sendcount, sendtype,
            crbuf, recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    
    /* Inter-Node RD*/
    
    mask = 0x1;
    i = 0;
    
    while (mask < comm_size) {
        dst_node = my_node ^ mask;
        dst_index = rank_index + (dst_node - my_node) * p;
        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = dst_node >> i;
        dst_tree_root <<= i;

        my_tree_root = my_node >> i;
        my_tree_root <<= i;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] M-Ring 01 c=%d\n",init_rank,sendcount);

        if (dst_index < comm_size) { 
            dst = comm_ptr->comm_rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/

            if(security_approach==2005){
                send_offset = my_tree_root * (recvcount * recvtype_extent +16 +12);
                recv_offset = dst_tree_root * (recvcount * recvtype_extent +16 +12);

                //printf("%d is going to send (Inter-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root);
                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR); 
                // mpi_errno =
                //     MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                //                     curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR, dst,
                //                     MPIR_ALLGATHER_TAG,
                //                     ((char *) ciphertext_recvbuf + recv_offset),
                //                     (n -dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR,
                //                     dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] M-Ring 02 c=%d R=%d  dst=%d\n",init_rank,sendcount,rank_index,dst);
                mpi_errno =
                    MPIC_Sendrecv(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*(recvcount * recvtype_extent +16 +12), MPI_CHAR, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * (recvcount * recvtype_extent +16 +12), MPI_CHAR,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] M-Ring 03 c=%d R=%d  dst=%d\n",init_rank,sendcount,rank_index,dst);   
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent+16+12));
                curr_cnt += recently_received;

                //printf("%d received %d (%d) and curr_cunt = %d\n", rank, last_recv_cnt, recently_received, curr_cnt);

                if(overlap_decryption == 1){
                    for(f=0; f<recently_received; ++f){
                        unsigned long count=0;
                        
                        in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)));
                        out_index = rank_index + (dst_tree_root+f-my_node) * p;
                        out = (char*)((char*) recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                        //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                        if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                        in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                        NULL, 0)){

                            printf("Error in Naive+ decryption: allgather ML_RD (I) while %d tried to decrypt from %d to %d\n", rank, recv_offset + (f * (recvcount * recvtype_extent + 16 + 12)), recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);
                            fflush(stdout);        
                        }
                    }// End for
                }
                

            }else{//if not Naive+
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = my_tree_root * recvcount * recvtype_extent;
                recv_offset = dst_tree_root * recvcount * recvtype_extent;

                // if(recvcount==16)
                //     printf("%d is going to send (Inter-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, my_tree_root, dst, (n - dst_tree_root), dst_tree_root);

                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*recvcount, recvtype); 
                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - dst_tree_root) * recvcount, recvtype); 
                // mpi_errno =
                //     MPIC_Sendrecv_Plus(((char *) ciphertext_recvbuf + send_offset),
                //                     curr_cnt*recvcount, recvtype, dst,
                //                     MPIR_ALLGATHER_TAG,
                //                     ((char *) ciphertext_recvbuf + recv_offset),
                //                     (n -dst_tree_root) * recvcount, recvtype,
                //                     dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                mpi_errno =
                    MPIC_Sendrecv(((char *) ciphertext_recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) ciphertext_recvbuf + recv_offset),
                                    (n -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                    
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }

                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                curr_cnt += recently_received;

                // if(recvcount==16)
                //     printf("%d received %d (%d) and curr_cunt = %d\n", rank, last_recv_cnt, recently_received, curr_cnt);


                for(f=0; f<recently_received; ++f){
                    
                    in = (char*)((char*) ciphertext_recvbuf + recv_offset + (f * recvcount * recvtype_extent));
                    out_index = rank_index + (dst_tree_root+f-my_node) * p;
                    out = (char*)((char*) recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                    // if(recvcount==16)
                    //     printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                    mpi_errno = MPIR_Localcopy(
                        in, recvcount, recvtype,
                        out, recvcount, recvtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }// End for
            }//end else
            

        }//end if dst_index < comm_size

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */

    
        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > n) {
            
            nprocs_completed = n - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                dst_node = my_node ^ tmp_mask;
                dst_index = rank + (dst_node - my_node) * p;

                tree_root = my_node >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->comm_rank_list[dst_index];
                
                if ((dst_node > my_node) && (my_node < tree_root + nprocs_completed)
                    && (dst_node >= tree_root + nprocs_completed)) {
                        if(security_approach==2005){
                            //Naive+
                            // if(comm_size - (my_tree_root+mask) < recently_received)
                            //     recently_received = comm_size - (my_tree_root+mask);
                            

                            sbuf = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask) * (recvcount * recvtype_extent + 16+12));
                            
                            //send
                            // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, recently_received * (recvcount*recvtype_extent + 16+12), MPI_CHAR); 
                            //printf("%d is going to send (II) %d from %d to %d\n", rank, recently_received, (my_tree_root + mask), dst);
                            // mpi_errno =
                            //     MPIC_Send_Plus(sbuf,
                            //             recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                            //             MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                            mpi_errno =
                                MPIC_Send(sbuf,
                                        recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                            if (mpi_errno) {
                                /* for communication errors, just record the error but
                                continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                
                            }
                            
                            //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                            
                        }//End Naive +
                        else{
                            // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, last_recv_cnt, recvtype); 
                            // mpi_errno =
                            //     MPIC_Send_Plus(((char *) ciphertext_recvbuf + offset),
                            //                 recently_received*recvcount, recvtype, dst,
                            //                 MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);

                            mpi_errno =
                                MPIC_Send(((char *) ciphertext_recvbuf + offset),
                                            recently_received*recvcount, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                            
                            /* last_recv_cnt was set in the previous
                            * receive. that's the amount of data to be
                            * sent now. */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }
                    
                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((dst_node < my_node) &&
                            (dst_node < tree_root + nprocs_completed) &&
                            (my_node >= tree_root + nprocs_completed)) {
                    if(security_approach==2005){
                        
                        // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (comm_size - (my_tree_root + mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + (my_tree_root + mask)*(recvcount*recvtype_extent + 16+12)),
                                        (n -
                                        (my_tree_root +
                                        mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                        curr_cnt += recently_received;
                        //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                        if(overlap_decryption == 1){
                            //decrypt the received messages

                            for(f=0; f<recently_received; ++f){
                                unsigned long count=0;
                                
                                in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12));
                                out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                                out = (char*)((char*) recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                                //printf("%d is going to copy from %d to %d -> %d\n", rank , recv_offset + (f * recvcount * recvtype_extent), out_index, comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                                if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                                in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                                NULL, 0)){

                                    printf("Error in Naive+ decryption: allgather ML_RD (II) while %d tried to decrypt from %d to %d\n", rank, (my_tree_root + mask + f) * (recvcount * recvtype_extent + 16 + 12), recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);
                                    fflush(stdout);        
                                }
                            }// End for
                        }//end if overlap
                        

                    }else{ // Not Naive+
                        // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (n - (my_tree_root + mask)) * recvcount, recvtype);
                        mpi_errno =
                            MPIC_Recv(((char *) ciphertext_recvbuf + offset),
                                        (n -
                                        (my_tree_root +
                                        mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount)); 
                        curr_cnt += recently_received;


                        for(f=0; f<recently_received; ++f){
                            unsigned long count=0;
                            
                            in = (char*)((char*) ciphertext_recvbuf + (my_tree_root + mask + f) * (recvcount * recvtype_extent));
                            out_index = rank_index + (my_tree_root + mask + f - my_node) * p;
                            out = (char*)((char*) recvbuf + comm_ptr->comm_rank_list[out_index] * recvcount * recvtype_extent);

                            mpi_errno = MPIR_Localcopy(
                                in, recvcount, recvtype,
                                out, recvcount, recvtype);
                            if (mpi_errno) {
                                MPIR_ERR_POP(mpi_errno);
                            }
                        }// End for


                    }
                    
                }
                tmp_mask >>= 1;
                k--;
            }//end while tmp_mask
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    if(security_approach == 2005 && overlap_decryption == 0){
        
            // Decrypt recently received data
        for (i=0; i < n; ++i){
            if(i != my_node){
                int recv_rank = (rank%p) + i * p;

                unsigned long count=0;
                
                if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, (recvbuf+recv_rank*recvcount*recvtype_extent),
                            &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            (ciphertext_recvbuf+(i*(sendcount*sendtype_extent+16+12))), 12,
                            (ciphertext_recvbuf+(i*(sendcount*sendtype_extent+16+12))+12), (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){
                    printf("Error in Naive+ decryption: allgather MultiLeader\n");
                    fflush(stdout);        
                }
            
            }// end if my_node
        }//end for
    }//end if


    //printf("%d finished Inter-Node (ML-RD)\n", rank);

    /* Intra-Node RD*/
    

    mask = 0x1;
    i = 0;
    curr_cnt = 1;
    

    
    // rank_index = comm_ptr->dev.ch.rank_list_index;
    rank_index = comm_ptr->comm_rank_list[rank];
    int shared_mem_rank, shared_mem_dst; 
    shared_mem_rank = shmem_commptr->rank;

    while (mask <  p) {
        shared_mem_dst = shared_mem_rank ^ mask;
        dst_index = my_node * p + shared_mem_dst;

        
        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = shared_mem_dst >> i;
        dst_tree_root <<= i;

        my_tree_root = shared_mem_rank >> i;
        my_tree_root <<= i;

        if ( dst_index < (my_node + 1) * p && dst_index>= (my_node * p) ) {
            dst = comm_ptr->comm_rank_list[dst_index];
            /** #TODO: Check curr_cnt is not more than available space **/


            /** #TODO: for loop to send all the msgs received in the inter-node step**/
            int previous_recv_count=0;
            for(f=0; f< n; ++f){
                /* FIXME: saving an MPI_Aint into an int */
                send_offset = comm_ptr->comm_rank_list[((my_tree_root + f * p) % comm_size)] * recvcount * recvtype_extent;
                recv_offset = comm_ptr->comm_rank_list[((dst_tree_root + f * p) % comm_size)] * recvcount * recvtype_extent;

                // if(recvcount==16)
                //     printf("%d is going to send (Intra-I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt, comm_ptr->comm_rank_list[(my_tree_root + f * p) % comm_size], dst, (p - dst_tree_root), comm_ptr->comm_rank_list[(dst_tree_root + f * p) % comm_size]);
                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, curr_cnt*recvcount, recvtype); 
                // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (p - dst_tree_root) * recvcount, recvtype); 
                // mpi_errno =
                //     MPIC_Sendrecv_Plus(((char *) recvbuf + send_offset),
                //                     curr_cnt*recvcount, recvtype, dst,
                //                     MPIR_ALLGATHER_TAG,
                //                     ((char *) recvbuf + recv_offset),
                //                     (p -dst_tree_root) * recvcount, recvtype,
                //                     dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);

                mpi_errno =
                    MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                    curr_cnt*recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) recvbuf + recv_offset),
                                    (p -dst_tree_root) * recvcount, recvtype,
                                    dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);

                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }
            

                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/(recvcount));
                if(previous_recv_count==0){
                    previous_recv_count = recently_received;
                }else{
                    if(previous_recv_count != recently_received){
                        printf("ERROR in Multileader-RD (Intra-I)!\nReceived sizes are not equal!\n");
                    }
                }
                //printf("%d  - %d received (Intra-I) %d from %d and curr_cnt is now %d\n", recvcount, rank, recently_received, dst, curr_cnt);
            }//end for f
            curr_cnt += recently_received;
            
	    
            // if(recvcount==16)
            //     printf("%d received (Intra-I) %d and curr_cnt is now %d\n",rank, recently_received, curr_cnt);
        }

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */
    
        /* --BEGIN EXPERIMENTAL-- */
        


        if (dst_tree_root + mask > p) {
            nprocs_completed = p - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            while (tmp_mask) {
                shared_mem_dst = shared_mem_rank ^ tmp_mask;
                dst_index = my_node * p + shared_mem_dst;

                tree_root = shared_mem_rank >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                dst = comm_ptr->comm_rank_list[dst_index];
                
                if ((shared_mem_dst > shared_mem_rank) && (shared_mem_rank < tree_root + nprocs_completed)
                    && (shared_mem_dst >= tree_root + nprocs_completed)) {

                    for(f=0; f<n; ++f){
                        sbuf = (char*)((char*) recvbuf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                    
                        //send
                        // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, send, recently_received * recvcount, recvtype); 
                        // printf("%d  -   %d is going to send %d from %d to %d\n", recvcount, rank, recently_received, (my_tree_root + mask+ f*p)%comm_size, dst);
                        // mpi_errno =
                        //     MPIC_Send_Plus(sbuf,
                        //             recently_received * recvcount, recvtype, dst,
                        //             MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);

                        mpi_errno =
                            MPIC_Send(sbuf,
                                    recently_received * recvcount, recvtype, dst,
                                    MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                        
                        if (mpi_errno) {
                            /* for communication errors, just record the error but
                            continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            
                        }
                    }//end for f
                            

                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((shared_mem_dst < shared_mem_rank) &&
                            (shared_mem_dst < tree_root + nprocs_completed) &&
                            (shared_mem_rank >= tree_root + nprocs_completed)) {
                    int previous_recv_count=0;
                    for(f=0; f<n; ++f){
                        rbuf = (char*)((char*) recvbuf + ((my_tree_root + mask+ f*p)%comm_size) * (recvcount * recvtype_extent));
                        // MPIR_PVAR_INC(allgather, 2lvl_multileader_rd, recv, (p - (my_tree_root + mask)) * recvcount, recvtype);
                        // printf("%d  -   %d is going to recv %d at %d from %d\n", recvcount, rank, p - (my_tree_root + mask), (my_tree_root + mask+ f*p)%comm_size, dst);
                        mpi_errno =
                            MPIC_Recv(rbuf,
                                        (p - (my_tree_root + mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        recently_received = (int)(last_recv_cnt/(recvcount));
                        if(previous_recv_count==0){
                            previous_recv_count = recently_received;
                        }else{
                            if(previous_recv_count != recently_received){
                                printf("ERROR in Multileader-RD-NB!\nReceived sizes are not equal!\n");
                            }
                        }
                        //printf("%d  - %d received (Intra-I) %d from %d and curr_cnt is now %d\n", recvcount, rank, recently_received, dst, curr_cnt);
                    }//end for f
                    curr_cnt += recently_received;
            

  
                }
                tmp_mask >>= 1;
                k--;

            }//end while tmp_mask

        }

        // /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

    //printf("%d finished Intra-Node (ML-RD)\n", rank);

    // for(i=0; i<send_req_idx; ++i){
    //     mpi_errno = MPIC_Wait((send_req_ptr[i]), errflag);
    //     if (mpi_errno)
    //         MPIR_ERR_POP(mpi_errno);
        

    //     if (mpi_errno == MPI_SUCCESS) {
    //         mpi_errno = send_req_ptr[i]->status.MPI_ERROR;
    //     }
	// // MPID_Request_release(send_req_ptr[i]);
    // }



   fn_exit:
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;

  fn_fail:
    goto fn_exit;

}



#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_NaivePlus_RDB_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_NaivePlus_RDB_MV2(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d  Sec-AP = %d] Func: MPIR_Allgather_NaivePlus_RDB_MV2\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif    
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    int curr_cnt, dst;
    MPI_Status status;
    int mask, dst_tree_root, my_tree_root,
        send_offset, recv_offset, last_recv_cnt = 0, nprocs_completed, k,
        offset, tmp_mask, tree_root;
    int send_req_idx = 0;
    unsigned char *ciphertext_recvbuf =large_recv_buffer; 


    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_NaivePlus_RDB_MV2\n");
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 01 c=%d recvtype_extent=%d\n",init_rank,recvcount,recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf + (comm_size * recvcount * recvtype_extent));


    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                    ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent),
                                    recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    int node_id, dst_node_id;
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    int first_encrypted_index = -1, last_encrypted_index = -1; //Nothing encrypted so far
    char *in, *out, *rbuf, *sbuf;
    int recently_received=0;
    
    curr_cnt = 1;

    mask = 0x1;
    i = 0;

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 01 c=%d\n",init_rank,recvcount);

    
    while (mask < comm_size) {
        dst = rank ^ mask;

        /* find offset into send and recv buffers. zero out
            * the least significant "i" bits of rank and dst to
            * find root of src and dst subtrees. Use ranks of
            * roots as index to send from and recv into buffer */
    
        dst_tree_root = dst >> i;
        dst_tree_root <<= i;

        my_tree_root = rank >> i;
        my_tree_root <<= i;

        /* FIXME: saving an MPI_Aint into an int */
        send_offset = my_tree_root * recvcount * recvtype_extent;
        recv_offset = dst_tree_root * recvcount * recvtype_extent;

        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 02 c=%d\n",init_rank,recvcount);

        if (dst < comm_size) {
            if(comm_size - my_tree_root < curr_cnt)
                curr_cnt = comm_size - my_tree_root;
                
            int first_to_send = my_tree_root;
            int last_to_send = my_tree_root + (int) (curr_cnt);
            
            MPID_Get_node_id(comm_ptr, rank, &node_id);
            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 03 c=%d dst=%d (my_node = %d  dst_node = %d)\n",init_rank,recvcount,dst,node_id,dst_node_id);
            if(node_id != dst_node_id){
                
                //encrypt            
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 03 c=%d\n",init_rank,recvcount);
                
                unsigned long  ciphertext_len = 0, count=0, in_size=0;
                in_size = (unsigned long)(curr_cnt * recvcount * recvtype_extent);
                unsigned long max_out_len = (unsigned long) (16 + in_size);
                
                //printf("%d is going to encrypt [%d, %d] - [%d, %d]\n", rank, first_to_send, last_to_send, first_encrypted_index, last_encrypted_index);

                in = (char*)((char*) recvbuf + my_tree_root * recvcount * recvtype_extent);
                out = (char*)((char*) ciphertext_recvbuf + my_tree_root * (recvcount * recvtype_extent + 16+12));
                //printf("%d is going to encrypt %d\n", rank, enc_idx);
                RAND_bytes(out, 12);
                
                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12, in, in_size,
                            NULL, 0)){
                    printf("Error in Naive+ encryption: allgather RDB\n");
                    fflush(stdout);
                }

                //printf("now first and last encrypted indices for %d are %d and %d\n", rank , first_encrypted_index, last_encrypted_index);

                //send-recv
                
                sbuf = (char*)((char*) out);
                rbuf = (char*)((char*) ciphertext_recvbuf + dst_tree_root * (recvcount * recvtype_extent + 16+12));

                //send recv
                //printf("%d is going to send (I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt,   my_tree_root, dst, (comm_size - dst_tree_root), dst_tree_root);

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 04-in c=%d send = %d  recv = %d dst = %d  [curr_cnt=%d  comm_size=%d  my_tree_root=%d rec_ex=%d]\n", init_rank,recvcount,(curr_cnt * recvcount*recvtype_extent)+28,(comm_size - dst_tree_root) * (recvcount*recvtype_extent)+28,dst,curr_cnt,comm_size,my_tree_root,recvtype_extent);

                mpi_errno =
                    MPIC_Sendrecv(sbuf, (curr_cnt * recvcount*recvtype_extent) + 16+12, 
                            MPI_CHAR, dst, MPIR_ALLGATHER_TAG,
                            rbuf, (comm_size - dst_tree_root) * (recvcount*recvtype_extent) + 16+12, 
                            MPI_CHAR, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
            
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 04-out c=%d dst = %d (my_node = %d  dst_node = %d)\n",init_rank,recvcount,dst,node_id,dst_node_id);
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 05-1 c=%d\n",init_rank,recvcount);
                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 05-2 c=%d\n",init_rank,recvcount);
                recently_received = (int)((last_recv_cnt-(16+12))/(recvcount*recvtype_extent));
                curr_cnt += recently_received;
                //printf("%d received (I) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                //decrypt
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 05-3 c=%d\n",init_rank,recvcount);

                int decryption_index = dst_tree_root;
                
                //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                
                in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent + 16+12));
                out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 05-dec-in c=%d\n",init_rank,recvcount);

                //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recently_received * recvcount*recvtype_extent)+16),
                        in, 12, in+12, (unsigned long )((recently_received * recvcount*recvtype_extent)+16),
                        NULL, 0)){

                    printf("Error in Naive+ decryption: allgather RDB (I) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                    fflush(stdout);        
                }
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 05-dec-out c=%d\n",init_rank,recvcount);


            }else{
                //send-recv

                sbuf = (char*)((char*) recvbuf + my_tree_root * recvcount * recvtype_extent);
                rbuf = (char*)((char*) recvbuf + dst_tree_root * recvcount * recvtype_extent);

                //send recv
                // if (DEBUG_INIT_FILE) printf("%d is going to send (I) %d from %d to %d and receive %d at %d mask=%d\n", rank, curr_cnt,   my_tree_root, dst, (comm_size - dst_tree_root), dst_tree_root,mask);

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 06-in c=%d send = %d  recv = %d dst = %d [curr_cnt=%d  comm_size=%d  my_tree_root=%d rec_ex=%d]\n", init_rank,recvcount,(curr_cnt * recvcount*recvtype_extent),(comm_size - dst_tree_root) * (recvcount*recvtype_extent),dst,curr_cnt,comm_size,my_tree_root,recvtype_extent);
                  mpi_errno =
                    MPIC_Sendrecv(sbuf, (curr_cnt * recvcount) , 
                            recvtype, dst, MPIR_ALLGATHER_TAG,
                            rbuf, (comm_size - dst_tree_root) * (recvcount), 
                            recvtype, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 06-out c=%d dst = %d (my_node = %d  dst_node = %d)\n",init_rank,recvcount,dst,node_id,dst_node_id);
                if (mpi_errno) {
                    /* for communication errors, just record the error but
                    continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                }
                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                recently_received = (int)(last_recv_cnt/recvcount);
                curr_cnt += recently_received;

            }
    
        }

        /* if some processes in this process's subtree in this step
            * did not have any destination process to communicate with
            * because of non-power-of-two, we need to send them the
            * data that they would normally have received from those
            * processes. That is, the haves in this subtree must send to
            * the havenots. We use a logarithmic recursive-halfing algorithm
            * for this. */

        /* This part of the code will not currently be
            * executed because we are not using recursive
            * doubling for non power of two. Mark it as experimental
            * so that it doesn't show up as red in the coverage
            * tests. */
    
        /* --BEGIN EXPERIMENTAL-- */
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 07-1 c=%d\n",init_rank,recvcount);
        if (dst_tree_root + mask > comm_size) {
            
            nprocs_completed = comm_size - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
                * subtree that have all the data. Send data to others
                * in a tree fashion. First find root of current tree
                * that is being divided into two. k is the number of
                * least-significant bits in this process's rank that
                * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;
            unsigned long  ciphertext_len = 0, count=0, in_size=0;

            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 08 c=%d\n",init_rank,recvcount);
            
            while (tmp_mask) {
                dst = rank ^ tmp_mask;

                tree_root = rank >> k;
                tree_root <<= k;

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 09 c=%d\n",init_rank,recvcount);

                /* send only if this proc has data and destination
                    * doesn't have data. at any step, multiple processes
                    * can send if they have the data */
                if ((dst > rank) && (rank < tree_root + nprocs_completed)
                    && (dst >= tree_root + nprocs_completed)) {
                        if(security_approach==2005){
                            //Naive+
                            if(comm_size - (my_tree_root+mask) < recently_received)
                            recently_received = comm_size - (my_tree_root+mask);
                            MPID_Get_node_id(comm_ptr, rank, &node_id);
                            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                            if(node_id != dst_node_id){
                                //Inter Node
                                //if(comm_size - (my_tree_root+mask) < recently_received)
                                //recently_received = comm_size - (my_tree_root+mask);

                                int first_to_send = (my_tree_root + mask);
                                int last_to_send = (my_tree_root + mask) + recently_received;
                                //int enc_idx;
                                //printf("last_to_send (II) for %d is %d\n", rank, last_to_send);
                                //printf("%d is going to encrypt %d - %d\n", rank, last_encrypted_index, last_to_send);
                                
                                 
                                in = (char*)((char*) recvbuf + first_to_send * recvcount * recvtype_extent);
                                out = (char*)((char*) ciphertext_recvbuf + first_to_send * (recvcount * recvtype_extent) + 16+12);
                                
                                in_size = (unsigned long)(recently_received * recvcount * recvtype_extent);
                                unsigned long max_out_len = (unsigned long) (16 + in_size);

                                // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                                RAND_bytes(out, 12);
                                
                                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                            &ciphertext_len, max_out_len,
                                            out, 12, in, in_size,
                                            NULL, 0)){
                                    printf("Error in Naive+ encryption: allgather RD (Default)\n");
                                    fflush(stdout);
                                }


                                
                                
                                //send
                                // MPIR_PVAR_INC(allgather, rd, send, recently_received * (recvcount*recvtype_extent) + 16+12, MPI_CHAR); 
                                //printf("%d is going to send (II) %d from %d to %d\n", rank, recently_received, (my_tree_root + mask), dst);
                                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 10 c=%d\n",init_rank,recvcount);

                                mpi_errno =
                                    MPIC_Send(out,
                                            recently_received * (recvcount * recvtype_extent) + 16+12, MPI_CHAR, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                                
                                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 11 c=%d\n",init_rank,recvcount);
                                if (mpi_errno) {
                                    /* for communication errors, just record the error but
                                    continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    
                                }
                
                                //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                            }else{
                                //Intra Node
                                // MPIR_PVAR_INC(allgather, rd, send, recently_received*recvcount, recvtype); 
                                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 12 c=%d\n",init_rank,recvcount);
                                mpi_errno =
                                    MPIC_Send(((char *) recvbuf + offset),
                                                recently_received*recvcount, recvtype, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                                
                                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 13 c=%d\n",init_rank,recvcount);
                                /* recently_received was set in the previous
                                * receive. that's the amount of data to be
                                * sent now. */
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                }
                                
                            }


                        }//End Naive +
                        else{
                            // MPIR_PVAR_INC(allgather, rd, send, last_recv_cnt, recvtype); 
                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 14 c=%d\n",init_rank,recvcount);
                            mpi_errno =
                                MPIC_Send(((char *) recvbuf + offset),
                                            last_recv_cnt, recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 15 c=%d\n",init_rank,recvcount);

                            /* last_recv_cnt was set in the previous
                            * receive. that's the amount of data to be
                            * sent now. */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }
                    
                }//End send condition
                /* recv only if this proc. doesn't have data and sender
                    * has data */
                else if ((dst < rank) &&
                            (dst < tree_root + nprocs_completed) &&
                            (rank >= tree_root + nprocs_completed)) {
                    if(security_approach==2005){
                        MPID_Get_node_id(comm_ptr, rank, &node_id);
                        MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                        if(node_id != dst_node_id){
                            //Inter-Node

                            // MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * (recvcount*recvtype_extent) + 16+12, MPI_CHAR);
                            //printf("%d is going to recv (II) %d from %d @ %d\n", rank, (comm_size - (my_tree_root + mask)), dst, (my_tree_root + mask));
                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 16 c=%d\n",init_rank,recvcount);
                            mpi_errno =
                                MPIC_Recv(((char *) ciphertext_recvbuf + (my_tree_root + mask)*(recvcount*recvtype_extent) + 16+12),
                                            (comm_size -
                                            (my_tree_root +
                                            mask)) * (recvcount*recvtype_extent) + 16+12, MPI_CHAR,
                                            dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 17 c=%d\n",init_rank,recvcount);                
                            /* nprocs_completed is also equal to the
                            * no. of processes whose data we don't have */
                            //printf("%d @ flag1 \n", rank);
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                last_recv_cnt = 0;
                            }
                            //printf("%d @ flag2 \n", rank);
                            MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                            int recently_received = (int)((last_recv_cnt - (16+12))/(recvcount*recvtype_extent));
                            curr_cnt += recently_received;
                            //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                            //decrypt the received messages
                            int decryption_index = (my_tree_root + mask);
                            int last_to_decrypt = (my_tree_root + mask) + recently_received;
                            //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                            
                            in = (char*)((char*) ciphertext_recvbuf + decryption_index * (recvcount * recvtype_extent) + 16+12);
                            out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                            //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                            if(!EVP_AEAD_CTX_open(ctx, out, &count, (unsigned long )((recently_received * recvcount*recvtype_extent)+16),
                                    in, 12, in+12, (unsigned long )((recently_received * recvcount*recvtype_extent)+16),
                                    NULL, 0)){

                                printf("Error in Naive+ decryption: allgather RDB (default-II) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                                fflush(stdout);        
                            }
                            


                        }else{
                            //intra-node
                            // MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * recvcount, recvtype);
                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 18 c=%d\n",init_rank,recvcount);
                            mpi_errno =
                                MPIC_Recv(((char *) recvbuf + offset),
                                            (comm_size -
                                            (my_tree_root +
                                            mask)) * recvcount, recvtype,
                                            dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 19 c=%d\n",init_rank,recvcount);                
                            /* nprocs_completed is also equal to the
                            * no. of processes whose data we don't have */
                            if (mpi_errno) {
                                /* for communication errors, just record the error
                                but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                last_recv_cnt = 0;
                            }
                            MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                            curr_cnt += (int) (last_recv_cnt/recvcount);
                        }
                    }else{
                        // MPIR_PVAR_INC(allgather, rd, recv, (comm_size - (my_tree_root + mask)) * recvcount, recvtype);
                        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 20 c=%d\n",init_rank,recvcount);
                        mpi_errno =
                            MPIC_Recv(((char *) recvbuf + offset),
                                        (comm_size -
                                        (my_tree_root +
                                        mask)) * recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 21 c=%d\n",init_rank,recvcount);
                        if (mpi_errno) {
                            /* for communication errors, just record the error
                            but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        }
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        curr_cnt += last_recv_cnt;
                    }
                    
                }
                tmp_mask >>= 1;
                k--;
            }
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }
    
    
// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-Plus 22 c=%d\n",init_rank,recvcount);

  fn_exit:
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;

  fn_fail:
    goto fn_exit;
}





#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_RD_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_RD_MV2(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d Sec-AP = %d] Func: MPIR_Allgather_RD_MV2\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif    
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    MPI_Aint curr_cnt, last_recv_cnt = 0;
    int dst;
    MPI_Status status;
    int mask, dst_tree_root, my_tree_root,
        send_offset, recv_offset, nprocs_completed, k, offset, tmp_mask, tree_root;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_RD_MV2\n");
#ifdef HAVE_ERROR_CHECKING
    /* Currently this algorithm can only handle power-of-2 comm_size.
     * Non power-of-2 comm_size is still experimental */
    MPIR_Assert(!(comm_size & (comm_size - 1)));
#endif /* HAVE_ERROR_CHECKING */

    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    // if(rank==0)
    //     printf("CP0\n");
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent), recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    // if(rank==0)
    //     printf("CP1\n");
    int node_id, dst_node_id;
    int first_encrypted_index = -1, last_encrypted_index = -1; //Nothing encrypted so far
    char *in, *out, *rbuf, *sbuf;
    int recently_received=0;
    if(security_approach==2005){
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        curr_cnt = 1;
    }else{
        curr_cnt = recvcount;
    }

    // curr_cnt = recvcount;
    char* enc_map;
    // enc_map = MPIU_Malloc(comm_size);
    enc_map = MPL_malloc(comm_size, MPL_MEM_USER );
    /* --BEGIN ERROR HANDLING-- */
    if (!enc_map) {
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME,
                                __LINE__, MPI_ERR_OTHER, "**nomem", 0);
        
        return mpi_errno;
    }
    int cc;
    for(cc=0; cc< comm_size; ++cc){
        enc_map[cc] = 'U';
    }

    mask = 0x1;
    i = 0;
    while (mask < comm_size) {
        dst = rank ^ mask;

        /* find offset into send and recv buffers. zero out
         * the least significant "i" bits of rank and dst to
         * find root of src and dst subtrees. Use ranks of
         * roots as index to send from and recv into buffer */

        dst_tree_root = dst >> i;
        dst_tree_root <<= i;

        my_tree_root = rank >> i;
        my_tree_root <<= i;

        /* FIXME: saving an MPI_Aint into an int */
        send_offset = my_tree_root * recvcount * recvtype_extent;
        recv_offset = dst_tree_root * recvcount * recvtype_extent;
        unsigned long  ciphertext_len = 0, count=0, in_size=0, max_out_len=0;        

        if (dst < comm_size) {
            if(security_approach==2005){
                //Naive+        
                ciphertext_len = 0;
                count=0;                
                in_size = (unsigned long)(recvcount * recvtype_extent); 
                max_out_len = (unsigned long) (16 + in_size);             

                if(comm_size - my_tree_root < curr_cnt)
                    curr_cnt = comm_size - my_tree_root;
                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                if(node_id != dst_node_id){
                    //Inter-node
                    //printf("curr_cnt for %d is %d\n", rank, curr_cnt);
                    //encrypt all the unencrypted messages so far
                    /*if(last_encrypted_index == -1){
                        last_encrypted_index = my_tree_root;
                    }*/
                    //if(comm_size - my_tree_root < curr_cnt)
                    //curr_cnt = comm_size - my_tree_root;

                    int first_to_send = my_tree_root;
                    int last_to_send = my_tree_root + (int) (curr_cnt);
                    int enc_idx;
                    
                    //printf("last_to_send for %d is %d\n", rank, last_to_send);
                    //printf("%d is going to encrypt [%d, %d] - [%d, %d]\n", rank, first_to_send, last_to_send, first_encrypted_index, last_encrypted_index);
                    for(enc_idx = first_to_send; enc_idx<last_to_send; ++enc_idx){
                        bool already_encrypted = first_encrypted_index!= -1 && enc_idx >= first_encrypted_index && last_encrypted_index!= -1 && enc_idx <= last_encrypted_index;
                        if(!already_encrypted && enc_map[enc_idx] == 'U'){
                            in = (char*)((char*) recvbuf + enc_idx * recvcount * recvtype_extent);
                            out = (char*)((char*) large_recv_buffer + enc_idx * (recvcount * recvtype_extent + 16+12));
                            //printf("%d is going to encrypt %d\n", rank, enc_idx);
                            RAND_bytes(out, 12);
                            
                            if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                        &ciphertext_len, max_out_len,
                                        out, 12, in, in_size,
                                        NULL, 0)){
                                printf("Error in Naive+ encryption: allgather RD (Default) [%d] \n",rank);
                                fflush(stdout);
                            } else {
                                if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"[%d] Encryption allgather RD (Default) passed"COLOR_RESET"\n",rank);                               
                            }
            /*if (rank == 0){
                printf("Encrypted %d\n", enc_idx);
            }*/
            
                            enc_map[enc_idx] = 'E';
                        }//end if
                    }//end for
                    if(last_encrypted_index == -1 || last_to_send > last_encrypted_index){
                        last_encrypted_index = last_to_send-1;
                    }
                    if(first_encrypted_index == -1 || first_to_send < first_encrypted_index){
                        first_encrypted_index = first_to_send;
                    }
                    //--last_encrypted_index;
                    //printf("now first and last encrypted indices for %d are %d and %d\n", rank , first_encrypted_index, last_encrypted_index);
        /*	if(rank==0){
            int jjj;
            for (jjj=0; jjj<comm_size; ++jjj){
            printf("%c, ", enc_map[jjj]);
            }
            printf("\n\n");
            }*/
                    //set the send and recv buffers
                    
                    sbuf = (char*)((char*) large_recv_buffer + my_tree_root * (recvcount * recvtype_extent + 16+12));
                    rbuf = (char*)((char*) large_recv_buffer + dst_tree_root * (recvcount * recvtype_extent + 16+12));

                    //send recv
                    //printf("%d is going to send (I) %d from %d to %d and receive %d at %d\n", rank, curr_cnt,   my_tree_root, dst, (comm_size - dst_tree_root), dst_tree_root);
                    //changed (comm_size - dst_tree_root) to curr_cnt
                    
                    mpi_errno =
                        MPIC_Sendrecv(sbuf, curr_cnt * (recvcount*recvtype_extent + 16+12), 
                                MPI_CHAR, dst, MPIR_ALLGATHER_TAG,
                                rbuf, (comm_size - dst_tree_root) * (recvcount*recvtype_extent + 16+12), 
                                MPI_CHAR, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno =
                        MPIC_Sendrecv_Plus(sbuf, curr_cnt * (recvcount*recvtype_extent + 16+12), 
                                MPI_CHAR, dst, MPIR_ALLGATHER_TAG,
                                rbuf, (comm_size - dst_tree_root) * (recvcount*recvtype_extent + 16+12), 
                                MPI_CHAR, dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                    *************************************************/
                    if (mpi_errno) {
                        /* for communication errors, just record the error but
                        continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        last_recv_cnt = 0;
                    }
                    MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                    recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                    curr_cnt += recently_received;
                    //printf("%d received (I) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                    //decrypt the received messages
                    int decryption_index = dst_tree_root;
                    int last_to_decrypt = dst_tree_root + recently_received;
                    //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                    for(; decryption_index<last_to_decrypt; ++decryption_index){
            /*if(rank == 0){
            printf("set Flag for %d\n", decryption_index);
            }*/
            enc_map[decryption_index] = 'E';
                        in = (char*)((char*) large_recv_buffer + decryption_index * (recvcount * recvtype_extent + 16+12));
                        out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                        //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                        if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                NULL, 0)){

                            printf("Error in Naive+ decryption: allgather RD (default-I) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                            fflush(stdout);        
                        } else {
                                if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"[%d] decryption allgather RD (Default) passed"COLOR_RESET"\n",rank);
                            }
                    }
        /*if(rank==0){
                        int jjj;
                        for (jjj=0; jjj<comm_size; ++jjj){
                            printf("%c| ", enc_map[jjj]);
                        }
                        printf("\n\n");
            }*/



                }else{
                    //Intra-Node
                    //printf("curr_cnt for %d is %d (intra-node)\n", rank, curr_cnt);                  
                    mpi_errno = MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                        curr_cnt*recvcount, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) recvbuf + recv_offset),
                                        (comm_size - dst_tree_root)*recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno =
                        MPIC_Sendrecv_Plus(((char *) recvbuf + send_offset),
                                        curr_cnt*recvcount, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) recvbuf + recv_offset),
                                        (comm_size - dst_tree_root)*recvcount, recvtype,
                                        dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, &(send_req_ptr[send_req_idx++]), errflag);
                    *************************************************/
                    if (mpi_errno) {
                        /* for communication errors, just record the error but
                        continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        last_recv_cnt = 0;
                    }

                    MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                    recently_received = (int) (last_recv_cnt/recvcount);
                    curr_cnt += recently_received;
                }
            }//End Naive+
            else{
                mpi_errno = MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                        curr_cnt, recvtype, dst,
                                        MPIR_ALLGATHER_TAG,
                                        ((char *) recvbuf + recv_offset),
                                        (comm_size - dst_tree_root) * recvcount,
                                        recvtype, dst,
                                        MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    last_recv_cnt = 0;
                } else {
                    MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                }
                curr_cnt += last_recv_cnt;
            }
        }

        /* if some processes in this process's subtree in this step
         * did not have any destination process to communicate with
         * because of non-power-of-two, we need to send them the
         * data that they would normally have received from those
         * processes. That is, the haves in this subtree must send to
         * the havenots. We use a logarithmic recursive-halfing algorithm
         * for this. */

        /* This part of the code will not currently be
         * executed because we are not using recursive
         * doubling for non power of two. Mark it as experimental
         * so that it doesn't show up as red in the coverage
         * tests. */

        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > comm_size) {
            nprocs_completed = comm_size - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
             * subtree that have all the data. Send data to others
             * in a tree fashion. First find root of current tree
             * that is being divided into two. k is the number of
             * least-significant bits in this process's rank that
             * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;            
            in_size = (unsigned long)(recvcount * recvtype_extent);
            
            while (tmp_mask) {
                dst = rank ^ tmp_mask;

                tree_root = rank >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                 * doesn't have data. at any step, multiple processes
                 * can send if they have the data */
                if ((dst > rank) && (rank < tree_root + nprocs_completed)
                    && (dst >= tree_root + nprocs_completed)) {

                        if(security_approach==2005){
                            //Naive+
                            if(comm_size - (my_tree_root+mask) < recently_received)
                            recently_received = comm_size - (my_tree_root+mask);
                            MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                            if(node_id != dst_node_id){
                                //Inter Node
                                //if(comm_size - (my_tree_root+mask) < recently_received)
                                //recently_received = comm_size - (my_tree_root+mask);

                                int first_to_send = (my_tree_root + mask);
                                int last_to_send = (my_tree_root + mask) + recently_received;
                                int enc_idx;
                                //printf("last_to_send (II) for %d is %d\n", rank, last_to_send);
                                //printf("%d is going to encrypt %d - %d\n", rank, last_encrypted_index, last_to_send);
                                for(enc_idx = first_to_send; enc_idx<last_to_send; ++enc_idx){
                                    bool already_encrypted = first_encrypted_index!= -1 && enc_idx >= first_encrypted_index && last_encrypted_index!= -1 && enc_idx <= last_encrypted_index;
                                    if(!already_encrypted && enc_map[enc_idx]=='U'){
                                        in = (char*)((char*) recvbuf + enc_idx * recvcount * recvtype_extent);
                                        out = (char*)((char*) large_recv_buffer + enc_idx * (recvcount * recvtype_extent + 16+12));
                                        // printf("%d is going to encrypt %d\n", rank, last_encrypted_index);
                                        RAND_bytes(out, 12);
                                        
                                        if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                                    &ciphertext_len, max_out_len,
                                                    out, 12, in, in_size,
                                                    NULL, 0)){
                                            printf("Error in Naive+ encryption (non-power 2): allgather RD (Default) [%d] \n",rank);
                                            fflush(stdout);
                                        } else {
                                           if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"[%d] encryption allgather RD (Default) passed (non-power 2)"COLOR_RESET"\n",rank);
                                        }
                                        enc_map[enc_idx] = 'E';
                                    }//end if
                                }//end for
                                if(last_encrypted_index == -1 || last_to_send > last_encrypted_index){
                                    last_encrypted_index = last_to_send -1;
                                }
                                if(first_encrypted_index == -1 || first_to_send < first_encrypted_index){
                                    first_encrypted_index = first_to_send;
                                }

                                sbuf = (char*)((char*) large_recv_buffer + (my_tree_root + mask) * (recvcount * recvtype_extent + 16+12));                             

                                mpi_errno =
                                    MPIC_Send(sbuf,
                                            recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, errflag);

                                /************* For MPIC_Sendrecv_Plus ************
                                mpi_errno =
                                    MPIC_Send_Plus(sbuf,
                                            recently_received * (recvcount * recvtype_extent + 16+12), MPI_CHAR, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                                *************************************************/
                                if (mpi_errno) {
                                    /* for communication errors, just record the error but
                                    continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    
                                }
                
                                //printf("%d sent (II) %d to %d\n", rank, recently_received, dst);
                            }else{
                                //Intra Node                                
                                mpi_errno =
                                    MPIC_Send(((char *) recvbuf + offset),
                                                recently_received*recvcount, recvtype, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                                
                                /************* For MPIC_Sendrecv_Plus ************
                                mpi_errno =
                                    MPIC_Send_Plus(((char *) recvbuf + offset),
                                                recently_received*recvcount, recvtype, dst,
                                                MPIR_ALLGATHER_TAG, comm_ptr, &(send_req_ptr[send_req_idx++]), errflag);
                                *************************************************/

                                /* recently_received was set in the previous
                                * receive. that's the amount of data to be
                                * sent now. */
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                }
                                
                            }
                        }//End Naive +
                        else{
                            mpi_errno = MPIC_Send(((char *) recvbuf + offset),
                                                    last_recv_cnt,
                                                    recvtype, dst, MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                            /* last_recv_cnt was set in the previous
                                * receive. that's the amount of data to be
                                * sent now. */
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
                /* recv only if this proc. doesn't have data and sender
                 * has data */
                else if ((dst < rank) &&
                         (dst < tree_root + nprocs_completed) &&
                         (rank >= tree_root + nprocs_completed)) {
                             if(security_approach==2005){
                                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                                if(node_id != dst_node_id){
                                //Inter-Node

                                
				//printf("%d is going to recv (II) %d from %d @ %d\n", rank, (comm_size - (my_tree_root + mask)), dst, (my_tree_root + mask));
                                mpi_errno =
                                    MPIC_Recv(((char *) large_recv_buffer + (my_tree_root + mask)*(recvcount*recvtype_extent + 16+12)),
                                                (comm_size -
                                                (my_tree_root +
                                                mask)) * (recvcount*recvtype_extent + 16+12), MPI_CHAR,
                                                dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                                /* nprocs_completed is also equal to the
                                * no. of processes whose data we don't have */
				//printf("%d @ flag1 \n", rank);
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    last_recv_cnt = 0;
                                }
				//printf("%d @ flag2 \n", rank);
                                MPIR_Get_count_impl(&status, MPI_CHAR, &last_recv_cnt);
                                int recently_received = (int)(last_recv_cnt/(recvcount*recvtype_extent + 16+12));
                                curr_cnt += recently_received;
                                //printf("%d received (II) %d (or %d) from %d and curr_cnt is now %d\n", rank, recently_received, last_recv_cnt, dst, curr_cnt);

                                //decrypt the received messages
                                int decryption_index = (my_tree_root + mask);
                                int last_to_decrypt = (my_tree_root + mask) + recently_received;
                                //printf("%d is going to decrypt %d - %d\n", rank, decryption_index, last_to_decrypt);
                                for(; decryption_index<last_to_decrypt; ++decryption_index){
                                    enc_map[decryption_index] = 'E';
                                    in = (char*)((char*) large_recv_buffer + decryption_index * (recvcount * recvtype_extent + 16+12));
                                    out = (char*)((char*) recvbuf + decryption_index * recvcount * recvtype_extent);
                                    //printf("%d is going to decrypt %d from %d to %d\n", rank, decryption_index, decryption_index * (recvcount * recvtype_extent +16 +12), decryption_index * recvcount * recvtype_extent);
                                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, out, &count, (unsigned long )((recvcount*recvtype_extent)+16),
                                            in, 12, in+12, (unsigned long )((recvcount*recvtype_extent)+16),
                                            NULL, 0)){

                                        printf("Error in Naive+ decryption: allgather RD (default-II) while %d tried to decrypt from %d to %d\n", rank, decryption_index * (recvcount * recvtype_extent+16+12), decryption_index * recvcount * recvtype_extent);
                                        fflush(stdout);        
                                    } else {
                                           if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"[%d] Deccryption allgather RD (Default) passed (non-power 2)"COLOR_RESET"\n",rank);
                                        }
                                }


                            }else{
                                //intra-node
                                
                                mpi_errno =
                                    MPIC_Recv(((char *) recvbuf + offset),
                                                (comm_size -
                                                (my_tree_root +
                                                mask)) * recvcount, recvtype,
                                                dst, MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                                /* nprocs_completed is also equal to the
                                * no. of processes whose data we don't have */
                                if (mpi_errno) {
                                    /* for communication errors, just record the error
                                    but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    last_recv_cnt = 0;
                                }
                                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                                curr_cnt += (int) (last_recv_cnt/recvcount);
                            }
                        }else{
                        mpi_errno = MPIC_Recv(((char *) recvbuf + offset),
                                            (comm_size - (my_tree_root + mask)) * recvcount,
                                            recvtype, dst,
                                            MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        /* nprocs_completed is also equal to the
                        * no. of processes whose data we don't have */
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag =
                                MPIX_ERR_PROC_FAILED ==
                                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            last_recv_cnt = 0;
                        } else {
                            MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                        }
                        curr_cnt += last_recv_cnt;
                    }
                }
                tmp_mask >>= 1;
                k--;
            }
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }
    // if(rank==0)
    //     printf("CP2\n");
  fn_exit:
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;

  fn_fail:
    goto fn_exit;
}



#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_intra_recursive_doubling
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_intra_recursive_doubling(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d  SA=%d] Func: MPIR_Allgather_intra_recursive_doubling\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    MPI_Aint curr_cnt, last_recv_cnt = 0;
    int dst;
    MPI_Status status;
    int mask, dst_tree_root, my_tree_root,
        send_offset, recv_offset, nprocs_completed, k, offset, tmp_mask, tree_root;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_intra_recursive_doubling\n");
#ifdef HAVE_ERROR_CHECKING
    /* Currently this algorithm can only handle power-of-2 comm_size.
     * Non power-of-2 comm_size is still experimental */
    MPIR_Assert(!(comm_size & (comm_size - 1)));
#endif /* HAVE_ERROR_CHECKING */

    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 01 c = %d r = %d \n",recvcount,init_rank);

    // if ((!(rank == 0 && (security_approach == 1001 || security_approach == 2005 || security_approach == 2006 || security_approach == 2007))) || init_phase == 1){
    //     if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 02 c = %d r = %d \n",recvcount,init_rank);
    //     if (sendbuf != MPI_IN_PLACE) {
    //         if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 03 c = %d r = %d \n",recvcount,init_rank);        
    //         mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
    //                                 ((char *) recvbuf +
    //                                     rank * recvcount * recvtype_extent), recvcount, recvtype);
    //         if (mpi_errno) {
    //             MPIR_ERR_POP(mpi_errno);
    //         }
    //     }
    // }

    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent), recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }


    if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 04 c = %d r = %d \n",recvcount,init_rank);

    curr_cnt = recvcount;

    mask = 0x1;
    i = 0;
    while (mask < comm_size) {
        dst = rank ^ mask;

        /* find offset into send and recv buffers. zero out
         * the least significant "i" bits of rank and dst to
         * find root of src and dst subtrees. Use ranks of
         * roots as index to send from and recv into buffer */

        dst_tree_root = dst >> i;
        dst_tree_root <<= i;

        my_tree_root = rank >> i;
        my_tree_root <<= i;

        /* FIXME: saving an MPI_Aint into an int */
        send_offset = my_tree_root * recvcount * recvtype_extent;
        recv_offset = dst_tree_root * recvcount * recvtype_extent;
        // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 05 c = %d r = %d \n",recvcount,init_rank);
        if (dst < comm_size) {
            mpi_errno = MPIC_Sendrecv(((char *) recvbuf + send_offset),
                                      curr_cnt, recvtype, dst,
                                      MPIR_ALLGATHER_TAG,
                                      ((char *) recvbuf + recv_offset),
                                      (comm_size - dst_tree_root) * recvcount,
                                      recvtype, dst,
                                      MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
            // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 06 c = %d r = %d \n",recvcount,init_rank);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                last_recv_cnt = 0;
            } else {
                MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
            }
            curr_cnt += last_recv_cnt;
        }

        /* if some processes in this process's subtree in this step
         * did not have any destination process to communicate with
         * because of non-power-of-two, we need to send them the
         * data that they would normally have received from those
         * processes. That is, the haves in this subtree must send to
         * the havenots. We use a logarithmic recursive-halfing algorithm
         * for this. */

        /* This part of the code will not currently be
         * executed because we are not using recursive
         * doubling for non power of two. Mark it as experimental
         * so that it doesn't show up as red in the coverage
         * tests. */

        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > comm_size) {
            nprocs_completed = comm_size - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
             * subtree that have all the data. Send data to others
             * in a tree fashion. First find root of current tree
             * that is being divided into two. k is the number of
             * least-significant bits in this process's rank that
             * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            /* FIXME: saving an MPI_Aint into an int */
            offset = recvcount * (my_tree_root + mask) * recvtype_extent;
            tmp_mask = mask >> 1;

            while (tmp_mask) {
                dst = rank ^ tmp_mask;

                tree_root = rank >> k;
                tree_root <<= k;

                // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 07 c = %d r = %d \n",recvcount,init_rank);

                /* send only if this proc has data and destination
                 * doesn't have data. at any step, multiple processes
                 * can send if they have the data */
                if ((dst > rank) && (rank < tree_root + nprocs_completed)
                    && (dst >= tree_root + nprocs_completed)) {
                        // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 08 c = %d r = %d \n",recvcount,init_rank);
                    mpi_errno = MPIC_Send(((char *) recvbuf + offset),
                                          last_recv_cnt,
                                          recvtype, dst, MPIR_ALLGATHER_TAG, comm_ptr, errflag);
                    /* last_recv_cnt was set in the previous
                     * receive. that's the amount of data to be
                     * sent now. */
                    // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 09 c = %d r = %d \n",recvcount,init_rank);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag =
                            MPIX_ERR_PROC_FAILED ==
                            MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                /* recv only if this proc. doesn't have data and sender
                 * has data */
                else if ((dst < rank) &&
                         (dst < tree_root + nprocs_completed) &&
                         (rank >= tree_root + nprocs_completed)) {
                             // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 10 c = %d r = %d \n",recvcount,init_rank);
                    mpi_errno = MPIC_Recv(((char *) recvbuf + offset),
                                          (comm_size - (my_tree_root + mask)) * recvcount,
                                          recvtype, dst,
                                          MPIR_ALLGATHER_TAG, comm_ptr, &status, errflag);
                        // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 11 c = %d r = %d \n",recvcount,init_rank);
                    /* nprocs_completed is also equal to the
                     * no. of processes whose data we don't have */
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag =
                            MPIX_ERR_PROC_FAILED ==
                            MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        last_recv_cnt = 0;
                    } else {
                        MPIR_Get_count_impl(&status, recvtype, &last_recv_cnt);
                    }
                    curr_cnt += last_recv_cnt;
                }
                tmp_mask >>= 1;
                k--;
            }
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }
    // if (DEBUG_INIT_FILE)  printf("*** MPIR_Allgather_intra_recursive_doubling 12 c = %d r = %d \n",recvcount,init_rank);

  fn_exit:
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
