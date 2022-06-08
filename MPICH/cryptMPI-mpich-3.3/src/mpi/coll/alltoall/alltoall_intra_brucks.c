/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Algorithm: Bruck's Algorithm
 *
 * This algorithm is from the IEEE TPDS Nov 97 paper by Jehoshua Bruck et al.
 *
 * It is a store-and-forward algorithm that
 * takes lgp steps. Because of the extra communication, the bandwidth
 * requirement is (n/2).lgp.beta.
 *
 * Cost = lgp.alpha + (n/2).lgp.beta
 *
 * where n is the total amount of data a process needs to send to all
 * other processes.
 */


#undef FUNCNAME
#define FUNCNAME MPIR_Naive_Sec_Alltoall
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Naive_Sec_Alltoall(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if  ALLTOALL_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[AlltoAll rank=%d  host=%s  count=%d  security_approach=%d] Func: MPIR_Naive_Sec_Alltoall\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    
    int mpi_errno=MPI_SUCCESS;    
    int mpi_errno_ret = MPI_SUCCESS;
    int sendtype_sz, recvtype_sz;
    unsigned long  ciphertext_sendbuf_len = 0;
    sendtype_sz= recvtype_sz= 0;
    // int var;

    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_Datatype_get_extent_macro(recvtype, recvtype_sz);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_sz);


    int rank;
    rank = comm_ptr->rank;
    unsigned long count=0;
    unsigned long next, dest, src;
    unsigned int i;

    unsigned long t=0;
    t = (unsigned long)(sendtype_sz*sendcount);
    unsigned long   max_out_len = (unsigned long) (16 + (sendtype_sz*sendcount));

    for( i = 0; i < comm_ptr->local_size; i++){
        next =(unsigned long )(i*((recvcount*recvtype_sz) + 16+12));
        src =(unsigned long )(i*(recvcount*recvtype_sz));
        
        RAND_bytes(ciphertext_sendbuf+next, 12); // 12 bytes of nonce

        if(!EVP_AEAD_CTX_seal(ctx, ciphertext_sendbuf+next+12,
                         &ciphertext_sendbuf_len, max_out_len,
                         ciphertext_sendbuf+next, 12,
                         sendbuf+src,  t,
                        NULL, 0))
        {
            printf("Error in encryption: Naive Alltoall\n");
            fflush(stdout);
        }

    }//end for

    init_phase=1;

    mpi_errno = MPIR_Alltoall_impl(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR,
                                    ciphertext_recvbuf, ((recvcount*recvtype_sz) + 16+12), MPI_CHAR,
                                    comm_ptr, errflag);
    init_phase=0;
    
    for( i = 0; i < comm_ptr->local_size; i++){
        next =(unsigned long )(i*((recvcount*recvtype_sz) + 16+12));
        dest =(unsigned long )(i*(recvcount*recvtype_sz));
        

        if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                        &count, (unsigned long )((recvcount*recvtype_sz)+16),
                         (ciphertext_recvbuf+next), 12,
                        (ciphertext_recvbuf+next+12), (unsigned long )((recvcount*recvtype_sz)+16),
                        NULL, 0)){
                    printf("Decryption error Naive Alltoall rank=%d  i=%d  chunk=%lu  remain=%lu\n",rank,i,recvcount*recvtype_sz,COMMON_LARGE_SEND_BUFFER_SIZE-next);fflush(stdout);        
            }                               
       
    }
   
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
#define FUNCNAME MPIR_Alltoall_Conc_ShMem_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_Conc_ShMem_MV2(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLTOALL_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[AlltoAll rank=%d  host=%s  count=%d  security_approach=%d] Func: MPIR_Alltoall_Conc_ShMem_MV2\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    int          comm_size, i, j;
    MPI_Aint     sendtype_extent = 0, recvtype_extent = 0;
    int mpi_errno=MPI_SUCCESS;    
    int mpi_errno_ret = MPI_SUCCESS;
    int dst, rank, local_rank, local_size;
    int sendtype_size, nbytes;
    char *tmp_buf;
    int my_node_id, dst_node_id;
    MPI_Comm shmem_comm, conc_comm;
    MPIR_Comm *shmem_commptr=NULL, *conc_commptr = NULL;

    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_CHKLMEM_DECL(6);
  
    if (recvcount == 0) 
    {
      // MPIR_TIMER_END(coll,alltoall,bruck);
      return MPI_SUCCESS;
    }
    
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;


    shmem_commptr = comm_ptr->node_comm;
    conc_comm = comm_ptr->concurrent_comm;

    // conc_comm = comm_ptr->concurrent_comm;    
    

    if (conc_comm == NULL) {
        init_phase==1;
        mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
        init_phase==0;
        conc_comm = comm_ptr->concurrent_comm;    
    } 

    // MPIR_Comm_get_ptr(shmem_comm, shmem_commptr);    
    MPIR_Comm_get_ptr(conc_comm, conc_commptr);  
        
    local_size = shmem_commptr->local_size;
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 02 c=%d\n",init_rank,recvcount);  
    local_rank = shmem_commptr->rank;
    int n = (int) (comm_size/local_size);
    MPID_Get_node_id(comm_ptr, rank, &my_node_id);

    /* Get extent of send and recv types */
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 03 c=%d\n",init_rank,recvcount);  
        
    MPIR_CHKLMEM_MALLOC(tmp_buf, char *, (comm_size) * recvcount * recvtype_extent, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 04 c=%d is_blocked=%d\n",init_rank,recvcount,comm_ptr->is_blocked);  

    /**
     *  Here we sort the messages. Messages that should be sent out to other nodes, 
     * will be copied to the tmp_buf (sorted based on the node number and local rank).
     * Messages that should be sent to processes on the same node will be copied to 
     * the shmem buffer at the proper location such that there is only one copy needed 
     * at the end for each process
     **/

    //TODO: optimize for block

    
    int s, dst_local_rank=0;
    void *in, *out;
    
    if(comm_ptr->is_blocked!=1){
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 04-1 c=%d\n",init_rank,recvcount);  
        for(s=0; s<comm_size; ++s){
            MPID_Get_node_id(comm_ptr, s, &dst_node_id);
            
            for(i=0; i<local_size; ++i){
                
                if(comm_ptr->comm_rank_list[i + dst_node_id*local_size] == s){
                    dst_local_rank = i;
                }
            }
            
            out = (void*)((char*)tmp_buf + (dst_node_id * local_size + dst_local_rank)*sendcount * sendtype_extent);

            mpi_errno = MPIR_Localcopy(sendbuf + (s * sendcount * sendtype_extent), sendcount, sendtype, 
                                    out, sendcount, sendtype);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
                
        }
    }
    


    if(security_approach == 2002){
        /**
         * Here we use a naive approach for the concurrent alltoall.
         * Intra-node step is done.
         * For the inter-node step, first each process encrypts 
         * the messages it has to send to other nodes.
         **/


        //encryption

        int n = (int) (comm_size / local_size);
        unsigned long  ciphertext_len = 0, de_count=0, in_size=0;
        in_size = (unsigned long)(local_size * sendcount * sendtype_extent);
        unsigned long max_out_len = (unsigned long) (16 + in_size);
        for(i=0; i<n; ++i){
            out = (void*)((char*)ciphertext_sendbuf + i * (local_size * sendcount * sendtype_extent + 16 + 12));
            // //#TODO: Remove one extra encryption
            if(i == my_node_id){
                memset(out, 0, (local_size * sendcount * sendtype_extent + 16 + 12));
            }else{
                if(comm_ptr->is_blocked!=1){
                    in = (void*)((char*)tmp_buf + i * (local_size * sendcount * sendtype_extent));
                }else{
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 05-1 c=%d i=%d\n",init_rank,recvcount,i);  
                    in = (void*)((char*)sendbuf + i * (local_size * sendcount * sendtype_extent));
                }
                RAND_bytes(out, 12);
                        
                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                            &ciphertext_len, max_out_len,
                            out, 12, in, in_size,
                            NULL, 0)){
                    printf("Error in encryption: Concurrent ShMem Alltoall (1)\n");
                    fflush(stdout);
                }
            }
            
        }//end for

        

        // Concurrent alltoall
        // mpi_errno = MPIR_Alltoall_impl(ciphertext_sendbuf, local_size * sendcount * sendtype_extent +16 +12, MPI_CHAR,
        //                            ciphertext_recvbuf, local_size * recvcount * recvtype_extent + 16 + 12, MPI_CHAR, conc_commptr, errflag);MPIR_Alltoall_intra_brucks

        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 06 c=%d\n",init_rank,recvcount);  
        
        init_phase=1;
        mpi_errno = MPIR_Alltoall_inter_pairwise_exchange( ciphertext_sendbuf, local_size * sendcount * sendtype_extent + 16 + 12, MPI_CHAR,
                                                ciphertext_recvbuf, local_size * recvcount * recvtype_extent + 16 + 12, MPI_CHAR, conc_commptr, errflag);
        init_phase=0;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 07 c=%d\n",init_rank,recvcount);  
                           
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        
        //Decrypt and copy to shmem_buffer
        for(i=0; i<n; ++i){
            int dest =(unsigned long ) ((local_rank * comm_size) + (i*local_size)) * (recvcount * recvtype_extent);
            //TODO: Remove the one extra decryption
            if(i == my_node_id){
                if(comm_ptr->is_blocked!=1){
                    in = (void*)((char*)tmp_buf + i * (local_size * sendcount * sendtype_extent));
                }else{
                    in = (void*)((char*)sendbuf + i * (local_size * sendcount * sendtype_extent));
                }
                mpi_errno = MPIR_Localcopy(in, local_size * sendcount, recvtype, 
                                            shmem_buffer + dest, local_size * sendcount, recvtype);

                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }else{
                int next =(unsigned long )(i*(local_size * recvcount * recvtype_extent + 16+12));
                
                if(!EVP_AEAD_CTX_open(ctx, ((shmem_buffer+dest)),
                                &de_count, (unsigned long )((local_size * recvcount*recvtype_extent)),
                                (ciphertext_recvbuf+next), 12,
                                (ciphertext_recvbuf+next+12), (unsigned long )(local_size * recvcount*recvtype_extent+16),
                                NULL, 0)){
                    printf("Decryption error in Concurrent ShMem Alltoall (1) while %d tried to decrypt from %d to %d\n", rank, next, dest);
                    fflush(stdout);
                }
            }
            

        }//end for

        mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        } 



        // //Copy to user buffer
        for(i=0; i<n; ++i){
            
            int idx =  (local_rank + i * local_size) * (recvcount*recvtype_extent);
            // if(rank ==0){
            //     printf("i= %d, idx= %d\n", i, idx);
            // }
            for(j=0; j<local_size; ++j){
                // if(rank ==0){
                //     printf("j= %d, gonna copy from %d to %d\n", j, idx  + j * (comm_size*recvcount*recvtype_extent), comm_ptr->comm_rank_list[(i*local_size + j)]*(recvcount*recvtype_extent));
                // }
                
                in = ((char*)shmem_buffer + idx  + j * (comm_size*recvcount*recvtype_extent));
                out = ((char*)recvbuf + comm_ptr->comm_rank_list[(i*local_size + j)]*(recvcount*recvtype_extent));

                mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
                                            out, recvcount, recvtype);

                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }//end for j

        }//end for i


    }//end security approach == 2002

    else{
        // Concurrent alltoall
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 06 c=%d\n",init_rank,recvcount);  
        if(comm_ptr->is_blocked!=1){
            in = tmp_buf;
        }else{
            in = sendbuf;
        }

        
        if (security_approach == 2001){

            mpi_errno = MPIR_Alltoall_bruck_MV2(in, local_size * sendcount, sendtype,
                                     shmem_buffer+local_rank*(comm_size * recvcount*recvtype_extent), local_size * recvcount, recvtype, conc_commptr, errflag);
        } else {

            init_phase=1;

            mpi_errno = MPIR_Alltoall_impl(in, local_size * sendcount, sendtype,
                                        shmem_buffer+local_rank*(comm_size * recvcount*recvtype_extent), local_size * recvcount, recvtype, conc_commptr, errflag);
            init_phase=0;

        }
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] CHS 08 c=%d\n",init_rank,recvcount);  
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }


	//Debugging Here:
        /*int idx = local_rank;
        mpi_errno = MPIR_Localcopy(shmem_buffer + idx * comm_size * recvcount*recvtype_extent, comm_size * recvcount, recvtype, 
                                    recvbuf, comm_size * recvcount, recvtype);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
	    }*/

	mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
	 if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        } 
        //Copy to user buffer
        for(i=0; i<n; ++i){

            int idx =  (local_rank + i * local_size) * (recvcount*recvtype_extent);
            for(j=0; j<local_size; ++j){
		/*if(local_rank == 1 && recvcount*recvtype_extent == 8){
                    printf("%d wants to copy from %d to %d\n", rank, idx  + j * (comm_size*recvcount*recvtype_extent), comm_ptr->comm_rank_list[(i*local_size + j)]*(recvcount*recvtype_extent));
		    }*/
                in = ((char*)shmem_buffer + idx  + j * (comm_size*recvcount*recvtype_extent));
                out = ((char*)recvbuf + comm_ptr->comm_rank_list[(i*local_size + j)]*(recvcount*recvtype_extent));

                mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
                                            out, recvcount, recvtype);

                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }//end for j

        }//end for i
        
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
#define FUNCNAME MPIR_Alltoall_bruck_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_bruck_MV2(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLTOALL_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[AlltoAll rank = %d host = %s count = %d] Func: MPIR_Alltoall_bruck_MV2\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    int          comm_size, i, pof2;
    MPI_Aint     sendtype_extent, recvtype_extent;
    MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    int mpi_errno=MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int src, dst, rank;
    int block, *displs, count;
    MPI_Aint pack_size, position;
    MPI_Datatype newtype;
    void *tmp_buf;

    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_CHKLMEM_DECL(6);
    
    if (recvcount == 0) 
    {
      // MPIR_TIMER_END(coll,alltoall,bruck);
      return MPI_SUCCESS;
    }
    
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get extent of send and recv types */
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    
    
    /* check if multiple threads are calling this collective function */
    // MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );
    
    
    /* use the indexing algorithm by Jehoshua Bruck et al,
     * IEEE TPDS, Nov. 97 */
    
    /* allocate temporary buffer */
    MPIR_Pack_size_impl(recvcount*comm_size, recvtype, &pack_size);
    // tmp_buf = MPIU_Malloc(pack_size);
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, pack_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    
    /* Do Phase 1 of the algorithim. Shift the data blocks on process i
     * upwards by a distance of i blocks. Store the result in recvbuf. */
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 01 c=%d\n",init_rank,recvcount);  
    mpi_errno = MPIR_Localcopy((char *) sendbuf + rank*sendcount*sendtype_extent, (comm_size - rank)*sendcount, sendtype, 
                                        recvbuf,                                  (comm_size - rank)*recvcount, recvtype);
        // if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 02 c=%d\n",init_rank,recvcount);  
    mpi_errno = MPIR_Localcopy(sendbuf, rank*sendcount, sendtype,
                      (char *) recvbuf + (comm_size-rank)*recvcount*recvtype_extent, rank*recvcount, recvtype);
        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 03 c=%d\n",init_rank,recvcount);  
    /* Input data is now stored in recvbuf with datatype recvtype */
    
    /* Now do Phase 2, the communication phase. It takes
     ceiling(lg p) steps. In each step i, each process sends to rank+2^i
     and receives from rank-2^i, and exchanges all data blocks
     whose ith bit is 1. */
    
    /* allocate displacements array for indexed datatype used in
     communication */
    
    // displs = MPIU_Malloc(comm_size * sizeof(int));
     MPIR_CHKLMEM_MALLOC(displs, int *, comm_size * sizeof(int), mpi_errno, "displs",
                        MPL_MEM_BUFFER);

    
    pof2 = 1;
    while (pof2 < comm_size) {
        dst = (rank + pof2) % comm_size;
        src = (rank - pof2 + comm_size) % comm_size;
        
        /* Exchange all data blocks whose ith bit is 1 */
        /* Create an indexed datatype for the purpose */
        
        count = 0;
        for (block=1; block<comm_size; block++) {
            if (block & pof2) {
                displs[count] = block * recvcount;
                count++;
            }
        }

        /**************** Added by Mehran ***********************/

        int my_node_id, dst_node_id, src_node_id;
        //get src and dst nodes
        MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
        MPID_Get_node_id(comm_ptr, src, &src_node_id);
        MPID_Get_node_id(comm_ptr, rank, &my_node_id);


        if(security_approach == 2001){
            
	        int dest, next, i;
            unsigned long  ciphertext_len = 0, de_count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);

            if(my_node_id != dst_node_id && my_node_id != src_node_id){
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 04 c=%d\n",init_rank,recvcount);  
                    for(i=0; i< count; ++i){
                        mpi_errno = MPIR_Localcopy((char *) recvbuf +
                               displs[i]*recvtype_extent,
                               recvcount, recvtype, tmp_buf + i * recvcount*recvtype_extent,
                               recvcount, recvtype);
                        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
                    }   
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 05 c=%d\n",init_rank,recvcount);  

                    void* in = (void*)((char*) tmp_buf);
                    void* out = (void*)((char*) ciphertext_sendbuf);
                    in_size = (unsigned long)(count * recvcount * recvtype_extent);
                    unsigned long max_out_len = (unsigned long) (16 + in_size);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in O-Bruck encryption: alltoall Bruck (3)\n");
                        fflush(stdout);
                    }
                    //printf("%d @ C2 O-Bruck2\n", rank);

                    // MPIR_PVAR_INC(alltoall, bruck, send, (count*sendcount*sendtype_extent+16+12), MPI_CHAR);
                    // MPIR_PVAR_INC(alltoall, bruck, recv, (count*sendcount*sendtype_extent+16+12), MPI_CHAR);                
                    mpi_errno = MPIC_Sendrecv(ciphertext_sendbuf, (count*sendcount*sendtype_extent+16+12), MPI_CHAR, dst,
                                                MPIR_ALLTOALL_TAG, ciphertext_recvbuf, (count*recvcount*recvtype_extent+16+12), MPI_CHAR,
                                                src, MPIR_ALLTOALL_TAG, comm_ptr,
                                                MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    //printf("%d @ C3 O-Bruck2\n", rank);
                    if(!EVP_AEAD_CTX_open(ctx, tmp_buf,
                                &de_count, (unsigned long )(count * recvcount*recvtype_extent),
                                (ciphertext_recvbuf), 12,
                                (ciphertext_recvbuf+12), (unsigned long )(count * recvcount*recvtype_extent+16),
                                NULL, 0)){
                            printf("Decryption error in O-Bruck alltoall (3) rank=%d  dst=%d\n",rank,dst);fflush(stdout);        
                            
                    }
                    //printf("%d @ C4 O-Bruck2\n", rank);   
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 06 c=%d\n",init_rank,recvcount);   
                    for(i=0; i< count; ++i){
                        mpi_errno = MPIR_Localcopy((char *) tmp_buf +
                               i*recvcount*recvtype_extent,
                               recvcount*recvtype_extent, MPI_CHAR, recvbuf + displs[i] * recvtype_extent,
                               recvcount, recvtype);
                        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
                    }      
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 07 c=%d\n",init_rank,recvcount);  

                    pof2 *= 2;                

            } else if(my_node_id != dst_node_id){
                
                    mpi_errno = MPIR_Type_create_indexed_block_impl(count, recvcount,
                                                  displs, recvtype, &newtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                    
                    mpi_errno = MPIR_Type_commit_impl(&newtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                    
                    for(i=0; i< count; ++i){
                        mpi_errno = MPIR_Localcopy((char *) recvbuf +
                               displs[i]*recvtype_extent,
                               recvcount, recvtype, tmp_buf + i * recvcount*recvtype_extent,
                               recvcount, recvtype);
                        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
                    }   
                    //printf("%d @ C1 O-Bruck2\n", rank);

                    void* in = (void*)((char*) tmp_buf);
                    void* out = (void*)((char*) ciphertext_sendbuf);
                    in_size = (unsigned long)(count * recvcount * recvtype_extent);
                    unsigned long max_out_len = (unsigned long) (16 + in_size);
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in O-Bruck encryption: alltoall Bruck (4)\n");
                        fflush(stdout);
                    }
                    //printf("%d @ C2 O-Bruck2\n", rank);

                    // MPIR_PVAR_INC(alltoall, bruck, send, (count*sendcount*sendtype_extent+16+12), MPI_CHAR);
                    // MPIR_PVAR_INC(alltoall, bruck, recv, 1, newtype);                
                    mpi_errno = MPIC_Sendrecv(ciphertext_sendbuf, (count*sendcount*sendtype_extent+16+12), MPI_CHAR, dst,
                                                MPIR_ALLTOALL_TAG, recvbuf, 1, newtype,
                                                src, MPIR_ALLTOALL_TAG, comm_ptr,
                                                MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    
                    MPIR_Type_free_impl(&newtype);
                    
                    pof2 *= 2;

            }//end if my_node != dst_node
            else if(my_node_id != src_node_id){                
		    
                    mpi_errno = MPIR_Type_create_indexed_block_impl(count, recvcount,
                                                  displs, recvtype, &newtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                    
                    mpi_errno = MPIR_Type_commit_impl(&newtype);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                    
                    position = 0;
                    mpi_errno = MPIR_Pack_impl(recvbuf, 1, newtype, tmp_buf, 
                                            pack_size, &position);
                    if (mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                    
                    // MPIR_PVAR_INC(alltoall, bruck, send, position, MPI_PACKED);
                    // MPIR_PVAR_INC(alltoall, bruck, recv, (count*sendcount*sendtype_extent+16+12), MPI_CHAR);                
                    mpi_errno = MPIC_Sendrecv(tmp_buf, position, MPI_PACKED, dst,
                                                MPIR_ALLTOALL_TAG, ciphertext_recvbuf, (count*sendcount*sendtype_extent+16+12), MPI_CHAR,
                                                src, MPIR_ALLTOALL_TAG, comm_ptr,
                                                MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    MPIR_Type_free_impl(&newtype);
                    
                    //Decrypt
                    if(!EVP_AEAD_CTX_open(ctx, tmp_buf,
                                &de_count, (unsigned long )(count * recvcount*recvtype_extent),
                                (ciphertext_recvbuf), 12,
                                (ciphertext_recvbuf+12), (unsigned long )(count * recvcount*recvtype_extent+16),
                                NULL, 0)){
                            printf("Decryption error in O-Bruck alltoall (4) rank=%d  dst=%d\n",rank,dst);fflush(stdout);        
                    }
   
                    for(i=0; i< count; ++i){
                        mpi_errno = MPIR_Localcopy((char *) tmp_buf +
                               i*recvcount*recvtype_extent,
                               recvcount*recvtype_extent, MPI_CHAR, recvbuf + displs[i] * recvtype_extent,
                               recvcount, recvtype);
                        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
                    }    
                    pof2 *= 2;  
                    //if(recvcount==1024*2)
		    
                
            }//end if my_node != src_node
            else{
                //original code
                mpi_errno = MPIR_Type_create_indexed_block_impl(count, recvcount,
                                                  displs, recvtype, &newtype);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                
                mpi_errno = MPIR_Type_commit_impl(&newtype);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                
                position = 0;
                mpi_errno = MPIR_Pack_impl(recvbuf, 1, newtype, tmp_buf, 
                                        pack_size, &position);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                
                // MPIR_PVAR_INC(alltoall, bruck, send, position, MPI_PACKED);
                // MPIR_PVAR_INC(alltoall, bruck, recv, 1, newtype);                
                mpi_errno = MPIC_Sendrecv(tmp_buf, position, MPI_PACKED, dst,
                                            MPIR_ALLTOALL_TAG, recvbuf, 1, newtype,
                                            src, MPIR_ALLTOALL_TAG, comm_ptr,
                                            MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                
                MPIR_Type_free_impl(&newtype);
                
                pof2 *= 2;
            }//end else

            
            

        }//end if security_approach=2001

        /*******************************************************/
        else{

            mpi_errno = MPIR_Type_create_indexed_block_impl(count, recvcount,
                                                  displs, recvtype, &newtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            
            mpi_errno = MPIR_Type_commit_impl(&newtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            
            position = 0;
            mpi_errno = MPIR_Pack_impl(recvbuf, 1, newtype, tmp_buf, 
                                    pack_size, &position);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            
            // MPIR_PVAR_INC(alltoall, bruck, send, position, MPI_PACKED);
            // MPIR_PVAR_INC(alltoall, bruck, recv, 1, newtype);                
            mpi_errno = MPIC_Sendrecv(tmp_buf, position, MPI_PACKED, dst,
                                        MPIR_ALLTOALL_TAG, recvbuf, 1, newtype,
                                        src, MPIR_ALLTOALL_TAG, comm_ptr,
                                        MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            
            MPIR_Type_free_impl(&newtype);
            
            pof2 *= 2;

        }
        
    }
    
    /* Rotate blocks in recvbuf upwards by (rank + 1) blocks. Need
     * a temporary buffer of the same size as recvbuf. */
    
    /* get true extent of recvtype */
    // MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);
    // recvbuf_extent = recvcount * comm_size * (MPIR_MAX(recvtype_true_extent, recvtype_extent));
    // tmp_buf = MPIU_Malloc(recvbuf_extent);

    
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    recvbuf_extent = recvcount * comm_size * (MPL_MAX(recvtype_true_extent, recvtype_extent));
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, recvbuf_extent, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *)((char*)tmp_buf - recvtype_true_lb);

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 08 c=%d\n",init_rank,recvcount);  
    
    mpi_errno = MPIR_Localcopy((char *) recvbuf + 
                               (rank+1)*recvcount*recvtype_extent,
                               (comm_size - rank - 1)*recvcount, recvtype, tmp_buf,
                               (comm_size - rank - 1)*recvcount, recvtype);
        if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 09 c=%d\n",init_rank,recvcount);  
    mpi_errno = MPIR_Localcopy(recvbuf, (rank+1)*recvcount, recvtype,
                              (char *) tmp_buf + 
                              (comm_size-rank-1)*recvcount*recvtype_extent,
                              (rank+1)*recvcount, recvtype);
        if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck 10 c=%d\n",init_rank,recvcount);  
    
    /* Blocks are in the reverse order now (comm_size-1 to 0).
     * Reorder them to (0 to comm_size-1) and store them in recvbuf. */
    
    for (i=0; i<comm_size; i++)
        MPIR_Localcopy((char *) tmp_buf + i*recvcount*recvtype_extent,
                       recvcount, recvtype,
                       (char *) recvbuf + (comm_size-i-1)*recvcount*recvtype_extent,
                       recvcount, recvtype);
    
    void *tmp = (void*)(tmp_buf + recvtype_true_lb);
    
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;
  fn_fail:
    if (newtype != MPI_DATATYPE_NULL)
        MPIR_Type_free_impl(&newtype);
    goto fn_exit;
}




#undef FUNCNAME
#define FUNCNAME MPIR_Alltoall_Scatter_dest_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_Scatter_dest_MV2(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLTOALL_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[AlltoAll rank = %d host = %s count = %d security_approach=%d] Func: MPIR_Alltoall_Scatter_dest_MV2\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    int          comm_size, i, j;
    MPI_Aint     sendtype_extent = 0, recvtype_extent = 0;
    MPIR_Request **reqarray;
    MPI_Status *starray;
    int mpi_errno=MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int src, dst, rank;
    int block, *displs, count;

    int sendtype_size, nbytes;

    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_CHKLMEM_DECL(3);
    
    if (recvcount == 0) 
    {
      // MPIR_TIMER_END(coll,alltoall,bruck);
      return MPI_SUCCESS;
    }
    
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get extent of send and recv types */
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    // MPID_Datatype_get_size_macro(sendtype, sendtype_size);
    nbytes = sendtype_extent * sendcount;

    int ii, ss, bblock;

    /* 
    If throttle_factor is n, each process posts n pairs of isend/irecv
     in each iteration. 
    if (MPIDI_Process.my_pg->ch.num_local_processes >= comm_size) {
        bblock = mv2_coll_param.alltoall_intra_throttle_factor;
    } else {
        if (nbytes < MV2_ALLTOALL_LARGE_MSG) {
            bblock = mv2_coll_param.alltoall_throttle_factor;
        } else {
            bblock = mv2_coll_param.alltoall_large_msg_throttle_factor; 
        }
    }
    */

    bblock = MPIR_CVAR_ALLTOALL_THROTTLE;
    if (bblock == 0)
        bblock = comm_size;

    if (bblock >= comm_size) bblock = comm_size;

    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, 2*bblock*sizeof(MPIR_Request*), mpi_errno,
                        "reqarray", MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, 2*bblock*sizeof(MPI_Status), mpi_errno,
                        "starray", MPL_MEM_BUFFER);
    
    

    /***********************      Added by Mehran     ***************************/
    int dest, next;
    unsigned long  ciphertext_len = 0, de_count=0, in_size=0;
    in_size = (unsigned long)(recvcount * recvtype_extent);
    unsigned long max_out_len = (unsigned long) (16 + in_size);
    
    int  my_node_id, dst_node_id;
    //get src and dst nodes
    MPID_Get_node_id(comm_ptr, rank, &my_node_id);
    /***************************************************************************/

    for (ii=0; ii<comm_size; ii+=bblock) {
        ss = comm_size-ii < bblock ? comm_size-ii : bblock;
        /* do the communication -- post ss sends and receives: */
        for ( i=0; i<ss; i++ ) {
            dst = (rank+i+ii) % comm_size;
            /***********************      Added by Mehran     ***************************/
            if(security_approach == 2001){

                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                if(dst_node_id != my_node_id){
                    // MPIR_PVAR_INC(alltoall, sd, recv, recvcount*recvtype_extent + 16 + 12, MPI_CHAR);
                    mpi_errno = MPIC_Irecv((char *)ciphertext_recvbuf +
                                      dst*(recvcount*recvtype_extent+16+12),
                                      recvcount*recvtype_extent + 16 + 12, MPI_CHAR, dst,
                                      MPIR_ALLTOALL_TAG, comm_ptr,
                                      &reqarray[i]);
                }else{
                    // MPIR_PVAR_INC(alltoall, sd, recv, recvcount, recvtype);
                    mpi_errno = MPIC_Irecv((char *)recvbuf +
                                      dst*recvcount*recvtype_extent,
                                      recvcount, recvtype, dst,
                                      MPIR_ALLTOALL_TAG, comm_ptr,
                                      &reqarray[i]);
                }
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
            /***************************************************************************/
            else{
                // MPIR_PVAR_INC(alltoall, sd, recv, recvcount, recvtype);
                mpi_errno = MPIC_Irecv((char *)recvbuf +
                                      dst*recvcount*recvtype_extent,
                                      recvcount, recvtype, dst,
                                      MPIR_ALLTOALL_TAG, comm_ptr,
                                      &reqarray[i]);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
        }
        for ( i=0; i<ss; i++ ) {
            dst = (rank-i-ii+comm_size) % comm_size;
            /***********************      Added by Mehran     ***************************/
            if(security_approach == 2001){

                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                if(dst_node_id != my_node_id){
                    //encrypt here
                    void* in = (void*)((char*) sendbuf + dst*sendcount*sendtype_extent);
                    void* out = (void*)((char*) ciphertext_sendbuf + dst*(sendcount*sendtype_extent+16+12));
                    
                    RAND_bytes(out, 12);
                    
                    if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                &ciphertext_len, max_out_len,
                                out, 12, in, in_size,
                                NULL, 0)){
                        printf("Error in O-SD encryption: alltoall SD (1)\n");
                        fflush(stdout);
                    }

                    //send
                    // MPIR_PVAR_INC(alltoall, sd, send, sendcount * sendtype_extent + 16 + 12, MPI_CHAR);
                    mpi_errno = MPIC_Isend(out,
                                          sendcount * sendtype_extent + 16 + 12, MPI_CHAR, dst,
                                          MPIR_ALLTOALL_TAG, comm_ptr,
                                          &reqarray[i+ss], errflag);
                }else{
                    // MPIR_PVAR_INC(alltoall, sd, send, sendcount, sendtype);
                    mpi_errno = MPIC_Isend((char *)sendbuf +
                                          dst*sendcount*sendtype_extent,
                                          sendcount, sendtype, dst,
                                          MPIR_ALLTOALL_TAG, comm_ptr,
                                          &reqarray[i+ss], errflag);
                }
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
            /***************************************************************************/
            else{
                // MPIR_PVAR_INC(alltoall, sd, send, sendcount, sendtype);
                mpi_errno = MPIC_Isend((char *)sendbuf +
                                            dst*sendcount*sendtype_extent,
                                            sendcount, sendtype, dst,
                                            MPIR_ALLTOALL_TAG, comm_ptr,
                                            &reqarray[i+ss], errflag);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
            
        }
        
        /* ... then wait for them to finish: */
        mpi_errno = MPIC_Waitall(2*ss,reqarray,starray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
            MPIR_ERR_POP(mpi_errno);
        }
       
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
                for (j=0; j<2*ss; j++) {
                     if (starray[j].MPI_ERROR != MPI_SUCCESS) {
                         mpi_errno = starray[j].MPI_ERROR;
                     }
                }
        }

        /***********************      Added by Mehran     ***************************/
        //decrypt here
        for ( i=0; i<ss; i++ ) {
            dst = (rank+i+ii) % comm_size;
            
            if(security_approach == 2001){

                MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                if(dst_node_id != my_node_id){
                    next =(unsigned long )(dst*(recvcount * recvtype_extent + 16+12));
                    dest =(unsigned long )(dst*(recvcount * recvtype_extent));
                    if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                                    &de_count, (unsigned long )((recvcount*recvtype_extent)),
                                    (ciphertext_recvbuf+next), 12,
                                    (ciphertext_recvbuf+next+12), (unsigned long )(recvcount*recvtype_extent+16),
                                    NULL, 0)){
                                printf("Decryption error in O-SD alltoall (1) rank=%d  dest=%d\n",rank,dest);fflush(stdout);                                        
                    }
                }//end if inter node
            } //end if security_approach
        }//end for
        /***************************************************************************/

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
#define FUNCNAME MPIR_Alltoall_intra_brucks
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoall_intra_brucks(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLTOALL_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[AlltoAll rank = %d host = %s count = %d] Func: MPIR_Alltoall_intra_brucks\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif    
    int comm_size, i, pof2;
    MPI_Aint sendtype_extent, recvtype_extent;
    MPI_Aint recvtype_true_extent, recvbuf_extent, recvtype_true_lb;
    int mpi_errno = MPI_SUCCESS, src, dst, rank;
    int mpi_errno_ret = MPI_SUCCESS;
    int block, *displs, count;
    MPI_Aint pack_size, position;
    MPI_Datatype newtype = MPI_DATATYPE_NULL;
    void *tmp_buf;
    MPIR_CHKLMEM_DECL(6);

    if (recvcount == 0)
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(sendbuf != MPI_IN_PLACE);
#endif /* HAVE_ERROR_CHECKING */

    /* Get extent of send and recv types */
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);

    /* allocate temporary buffer */
    MPIR_Pack_size_impl(recvcount * comm_size, recvtype, &pack_size);
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, pack_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

    /* Do Phase 1 of the algorithim. Shift the data blocks on process i
     * upwards by a distance of i blocks. Store the result in recvbuf. */

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck-N 01 c=%d\n",init_rank,recvcount);  


    mpi_errno = MPIR_Localcopy((char *) sendbuf +
                               rank * sendcount * sendtype_extent,
                               (comm_size - rank) * sendcount, sendtype, recvbuf,
                               (comm_size - rank) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = MPIR_Localcopy(sendbuf, rank * sendcount, sendtype,
                               (char *) recvbuf +
                               (comm_size - rank) * recvcount * recvtype_extent,
                               rank * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* Input data is now stored in recvbuf with datatype recvtype */

    /* Now do Phase 2, the communication phase. It takes
     * ceiling(lg p) steps. In each step i, each process sends to rank+2^i
     * and receives from rank-2^i, and exchanges all data blocks
     * whose ith bit is 1. */

    /* allocate displacements array for indexed datatype used in
     * communication */

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Bruck-N 02 c=%d\n",init_rank,recvcount);  

    MPIR_CHKLMEM_MALLOC(displs, int *, comm_size * sizeof(int), mpi_errno, "displs",
                        MPL_MEM_BUFFER);

    pof2 = 1;
    while (pof2 < comm_size) {
        dst = (rank + pof2) % comm_size;
        src = (rank - pof2 + comm_size) % comm_size;

        /* Exchange all data blocks whose ith bit is 1 */
        /* Create an indexed datatype for the purpose */

        count = 0;
        for (block = 1; block < comm_size; block++) {
            if (block & pof2) {
                displs[count] = block * recvcount;
                count++;
            }
        }

        mpi_errno = MPIR_Type_create_indexed_block_impl(count, recvcount,
                                                        displs, recvtype, &newtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno = MPIR_Type_commit_impl(&newtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        position = 0;
        mpi_errno = MPIR_Pack_impl(recvbuf, 1, newtype, tmp_buf, pack_size, &position);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno = MPIC_Sendrecv(tmp_buf, position, MPI_PACKED, dst,
                                  MPIR_ALLTOALL_TAG, recvbuf, 1, newtype,
                                  src, MPIR_ALLTOALL_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        MPIR_Type_free_impl(&newtype);

        pof2 *= 2;
    }

    /* Rotate blocks in recvbuf upwards by (rank + 1) blocks. Need
     * a temporary buffer of the same size as recvbuf. */

    /* get true extent of recvtype */
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    recvbuf_extent = recvcount * comm_size * (MPL_MAX(recvtype_true_extent, recvtype_extent));
    MPIR_CHKLMEM_MALLOC(tmp_buf, void *, recvbuf_extent, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);
    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - recvtype_true_lb);

    mpi_errno = MPIR_Localcopy((char *) recvbuf + (rank + 1) * recvcount * recvtype_extent,
                               (comm_size - rank - 1) * recvcount, recvtype, tmp_buf,
                               (comm_size - rank - 1) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = MPIR_Localcopy(recvbuf, (rank + 1) * recvcount, recvtype,
                               (char *) tmp_buf + (comm_size - rank -
                                                   1) * recvcount * recvtype_extent,
                               (rank + 1) * recvcount, recvtype);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Blocks are in the reverse order now (comm_size-1 to 0).
     * Reorder them to (0 to comm_size-1) and store them in recvbuf. */

    for (i = 0; i < comm_size; i++) {
        mpi_errno = MPIR_Localcopy((char *) tmp_buf + i * recvcount * recvtype_extent,
                                   recvcount, recvtype,
                                   (char *) recvbuf + (comm_size - i -
                                                       1) * recvcount * recvtype_extent, recvcount,
                                   recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");

    return mpi_errno;
  fn_fail:
    if (newtype != MPI_DATATYPE_NULL)
        MPIR_Type_free_impl(&newtype);
    goto fn_exit;
}

