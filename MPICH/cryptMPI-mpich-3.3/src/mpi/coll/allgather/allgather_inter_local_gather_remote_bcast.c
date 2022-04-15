/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Intercommunicator Allgather
 *
 * Each group does a gather to local root with the local
 * intracommunicator, and then does an intercommunicator
 * broadcast.
 */




#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_SharedMem_Allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_SharedMem_Allgather_MV2(const void *sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather rank = %d host = %s count = %d  Sec-AP = %d SH_LEADERS = %d] Func: MPIR_2lvl_SharedMem_Allgather_MV2\n", comm_ptr->rank,hostname,recvcount,security_approach,shmem_leaders);fflush(stdout);
	}
#endif  	
    int rank, size;
    int local_rank, local_size, idx;
    int leader_comm_size = 0, leader_rank;
     
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_extent = 0;  
    // MPI_Comm shmem_comm, conc_comm;
    MPIR_Comm *shmem_commptr=NULL, *leader_commptr = NULL;
    int node_id;    
    rank = comm_ptr->rank;
    size = comm_ptr->local_size;
    // if(rank==0) printf("MPIR_2lvl_SharedMem_Allgather_MV2\n");
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    int my_node = node_id;
    
    if (recvcount == 0) {
        return MPI_SUCCESS;
    }

    // extract the rank,size information for the intra-node communicator
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] ALG-Conc 01\n",init_rank,recvcount);    
    shmem_commptr = comm_ptr->node_comm;
    //MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;    
    int p = local_size;
    int n = node_cnt;
    int workload = (int) (p / shmem_leaders);
    
    // mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
    // if(rank==0) printf("local_rank=%d, local_size=%d, p=%d, n=%d, workload=%d\n", local_rank, local_size, p, n, workload);
    // conc_comm = comm_ptr->concurrent_comm;    
    // MPIR_Comm_get_ptr(conc_comm, conc_commptr);
    if (local_rank == 0) {
        // Node leader. Extract the rank, size information for the leader communicator        
        leader_commptr = comm_ptr->node_roots_comm;
        leader_rank = comm_ptr->node_roots_comm->rank;
        leader_comm_size = comm_ptr->node_roots_comm->local_size;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] ALG-Conc 02-1\n",init_rank,recvcount);    
    }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] ALG-Conc 03\n",init_rank,recvcount);    
    //First, copy plaintext to the shmem_buf
    void* my_shmem_buffer = (void*)((char*)shmem_buffer + comm_ptr->shmem_idx * recvcount *recvtype_extent);
    void* my_ciphertext_shmem_buffer = (void*)((char*)ciphertext_shmem_buffer + comm_ptr->ctx_shmem_idx * (recvcount *recvtype_extent + 16 + 12));
    if(local_rank == 0) {
        if(sendbuf == MPI_IN_PLACE) {
            mpi_errno = MPIR_Gather_impl((void*)((char*)recvbuf + (rank * recvcount * recvtype_extent)), 
                                    recvcount , recvtype,
                                    (void*)((char*)my_shmem_buffer + ((my_node * p + local_rank) * recvcount * recvtype_extent)), 
                                    recvcount, recvtype,
                                    0, shmem_commptr, errflag);

        }else{
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcount,sendtype, 
                                    (void*)((char*)my_shmem_buffer + ((my_node * p + local_rank) * recvcount * recvtype_extent)), 
                                    recvcount, recvtype,
                                    0, shmem_commptr, errflag);
        }  
    } else {
        //Since in allgather all the processes could have its own data in place
        if(sendbuf == MPI_IN_PLACE) {
            mpi_errno = MPIR_Gather_impl((void*)((char*)recvbuf + (rank * recvcount * recvtype_extent)), 
                                        recvcount , recvtype, 
                                        NULL, recvcount, recvtype,
                                        0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcount,sendtype, 
                                        NULL, recvcount, recvtype,
                                        0, shmem_commptr, errflag);
        }
    }
    // mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
    // if(rank==0) {
    //     float* data = (float*) my_shmem_buffer;
    //     printf("%f, %f, %f\n", data[0], data[1], data[2]);
    //     // printf("check 1\n");
    // }
    //printf("%d @ check1\n", rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    
    if(security_approach == 2006 && shmem_leaders>1){
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
    }

    
    if(local_rank<shmem_leaders && security_approach==2006){
        unsigned long count=0;
        unsigned long next, dest;
        unsigned int i;
        //Step 1: Encryption

        unsigned long  ciphertext_len = 0;
        //void* out = (void*)((char*) my_ciphertext_shmem_buffer + local_rank * (workload * recvcount * recvtype_extent + 12 + 16));
        void* out = (void*)((char*) my_ciphertext_shmem_buffer + (my_node * p + local_rank * workload)* recvcount * recvtype_extent + (my_node * shmem_leaders + local_rank) * (16 + 12));
        RAND_bytes(out, 12); // 12 bytes of nonce

        unsigned long t=0;
        t = (unsigned long)(workload * recvcount * recvtype_extent);
        if(local_rank == shmem_leaders-1){
            t += (p - (shmem_leaders * workload)) * recvcount * recvtype_extent;
        }
        unsigned long   max_out_len = (unsigned long) (16 + t  + 12);
        
        if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,  
                            &ciphertext_len, max_out_len,
                            out, 12,
                            my_shmem_buffer + ((my_node * p + local_rank * workload)* recvcount * recvtype_extent),  t,
                            NULL, 0))
        {
                printf("Error in Naive+ encryption: allgather-shmem while %d tried to encrypt %d from %d to %d, count*size = %d\n", rank, max_out_len, (my_node * p + local_rank * workload)* recvcount * recvtype_extent, my_node * (workload * recvcount * recvtype_extent + 12 + 16), recvcount * recvtype_extent);
                fflush(stdout);
        }
        //if(recvcount * recvtype_extent == 524288)
	//  printf("%d encrypted %d from %d to %d\n", rank, ciphertext_len, ((my_node * p + local_rank * workload)* recvcount * recvtype_extent), (my_node * p + local_rank * workload)* recvcount * recvtype_extent + (my_node * shmem_leaders + local_rank) * (16 + 12));
    }
    // mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
    // if(rank==0) printf("check 2\n");

    if(security_approach == 2006 && shmem_leaders>1){
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
    }
    

    // Exchange the data between the node leaders
    if (local_rank == 0 && (leader_comm_size > 1)) {
        if(security_approach == 2006){
            //NAIVE PLUS
	        //printf("Naive+\n");
            //printf("%d @ check2\n", rank);
            //Step 2: Data exchange
            void* out = (void*)((char*) my_ciphertext_shmem_buffer + my_node * (p * recvcount * recvtype_extent + shmem_leaders *(12 + 16)));
            unsigned long max_out_len = (p * recvcount * recvtype_extent + shmem_leaders *(12 + 16));

            //mpi_errno = MPIR_Allgather_impl(out, (max_out_len), MPI_CHAR,
            //                                my_ciphertext_shmem_buffer, (max_out_len), MPI_CHAR, leader_commptr, errflag);
            

            mpi_errno = MPIR_Allgather_intra_recursive_doubling(MPI_IN_PLACE, (max_out_len), MPI_CHAR,
                                            my_ciphertext_shmem_buffer, (max_out_len), MPI_CHAR, leader_commptr, errflag);

            //printf("%d @ check3\n", rank);
            //Step3: Decryption
            //Working Here
            unsigned long count=0;
            int dest, next, i;
            for( i = 0; i < n; ++i){
                if(i != my_node){
                    int l_idx;
                    for(l_idx=0 ; l_idx<shmem_leaders ; ++l_idx){
                        if(((i+1)*(l_idx+1))%p == local_rank){
                            unsigned long t=0;
                            t = (unsigned long)(workload * recvcount * recvtype_extent);
                            if(l_idx == shmem_leaders-1){
                                t += (p - (shmem_leaders * workload)) * recvcount * recvtype_extent;
                            }
                            
                            int idx = i*p + l_idx * workload;
                            next =(unsigned long )(idx* recvcount * recvtype_extent + (i * shmem_leaders + l_idx)*(16+12));
                            dest =(unsigned long )(idx* recvcount * recvtype_extent);
                            //if(recvcnt * recvtype_extent == 524288)
			    //                printf("%d is going to decrypt %d from %d to %d\n", rank, t, next, dest);
                            if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((my_shmem_buffer+dest)),
                                            &count, t,
                                            (my_ciphertext_shmem_buffer+next), 12,
                                            (my_ciphertext_shmem_buffer+next+12), t+16,
                                            NULL, 0)){
                                    printf("Decryption error in Naive+ shmem_allgather I while %d tried to decrypt %d from %d to %d\n", rank, count, next, dest);fflush(stdout);        
                            }
                        }//end if local_rank
                        
                    }//end for j
                }//end if my_node                            
            }//end for i
            //printf("%d @ check4\n", rank);

            //End of NAIVE PLUS
        }else{   
            int alg_choice = mv2_user_allgather_inter;
            mv2_user_allgather_inter = 0;
            void* tmp_buf = MPL_malloc(sizeof(recvtype) * recvcount * p, MPL_MEM_USER);
            mpi_errno = MPIR_Localcopy((void*)((char*)my_shmem_buffer + (my_node*p * recvcount * recvtype_extent)), recvcount * p, recvtype, 
                                    (void*)((char*)tmp_buf), recvcount * p, recvtype);
            mpi_errno = MPIR_Allgather_impl(tmp_buf, 
                                                (recvcount*p),
                                                recvtype,
                                                my_shmem_buffer, (recvcount*p), recvtype,
                                                leader_commptr, errflag);  
            
            // mpi_errno = MPIR_Allgather_impl((void*)((char*)my_shmem_buffer + (my_node*p * recvcount * recvtype_extent)), 
            //                                     (recvcount*p),
            //                                     recvtype,
            //                                     my_shmem_buffer, (recvcount*p), recvtype,
            //                                     leader_commptr, errflag);          
            mv2_user_allgather_inter=alg_choice;
            MPL_free(tmp_buf);
        }

        
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    
    } 
    // mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
    // if(rank==0) printf("check 3\n");
    //printf("%d @ check6\n", rank);
    mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
        goto fn_fail;
    }
    //printf("%d @ check7\n", rank);
    if(security_approach==2006 && local_rank!=0){
        
        //help in decryption
        //#TODO: Decrypt from shared_memory and only for a portion
        int sendtype_sz, recvtype_sz;
        unsigned long count=0;
        sendtype_sz= recvtype_sz= 0;
        int var;
        var=MPI_Type_size(sendtype, &sendtype_sz);
        var=MPI_Type_size(recvtype, &recvtype_sz);
        unsigned long   max_out_len = (unsigned long) (16 + (p*sendtype_sz*sendcount));
        int i, next, dest;
        //printf("%d- local_rank:%d, leader_Comm_size: %d, rank/p:%d\n", rank, local_rank, leader_comm_size, (int)(rank/p));
        for( i = 0; i < n; ++i){
            if(i != my_node){
                int l_idx;
                for(l_idx=0 ; l_idx<shmem_leaders ; ++l_idx){
                    if(((i+1)*(l_idx+1))%p == local_rank){
                        unsigned long t=0;
                        t = (unsigned long)(workload * recvcount * recvtype_extent);
                        if(l_idx == shmem_leaders-1){
                            t += (p - (shmem_leaders * workload)) * recvcount * recvtype_extent;
                        }
                        
                        int idx = i*p + l_idx * workload;
                        next =(unsigned long )(idx* recvcount * recvtype_extent + (i * shmem_leaders + l_idx)*(16+12));
                        dest =(unsigned long )(idx* recvcount * recvtype_extent);
                        //if(recvcnt * recvtype_extent == 524288)
			//              printf("%d is going to decrypt (II) %d from %d to %d\n", rank, t, next, dest);
                        if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((my_shmem_buffer+dest)),
                                        &count, t,
                                        (my_ciphertext_shmem_buffer+next), 12,
                                        (my_ciphertext_shmem_buffer+next+12), t+16,
                                        NULL, 0)){
                                printf("Decryption error in Naive+ shmem_allgather II while %d tried to decrypt %d from %d to %d\n", rank, count, next, dest);fflush(stdout);        
                        }
                    }//end if local_rank
                    
                }//end for j
            }//end if my_node                            
        }//end for i
    }
    // mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
    // if(rank==0) printf("check 4\n");
    if(security_approach == 2006){
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
         if(comm_ptr->cyclic_hostfil != 1){
            //Blocked
            mpi_errno = MPIR_Localcopy((void*)((char*)my_shmem_buffer), recvcount * size, recvtype, 
                                    (void*)((char*)recvbuf), recvcount * size, recvtype);
        }else{
            //NonBlocked
            int s=0;
            for(; s<size; ++s){
                
                mpi_errno = MPIR_Localcopy((void*)((char*)my_shmem_buffer + s * recvcount  * recvtype_extent), recvcount , recvtype, 
                                        (void*)((char*)recvbuf + comm_ptr->comm_rank_list[s] * recvcount * recvtype_extent), recvcount, recvtype);
                
            }
        }
        
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }else{
                
         if(comm_ptr->cyclic_hostfil != 1){
            //Blocked
            // printf("\nBLOCKED!\n\n");
            mpi_errno = MPIR_Localcopy((void*)((char*)my_shmem_buffer), recvcount * size, recvtype, 
                                    (void*)((char*)recvbuf), recvcount * size, recvtype);
        
        }else{
            //NonBlocked
            // printf("\nNON BLOCKED!\n\n");
            int s=0;
            for(; s<size; ++s){
                
                mpi_errno = MPIR_Localcopy((void*)((char*)my_shmem_buffer + s * recvcount  * recvtype_extent), recvcount , recvtype, 
                                        (void*)((char*)recvbuf + comm_ptr->comm_rank_list[s] * recvcount * recvtype_extent), recvcount, recvtype);
                
            }

        }
    
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

  fn_fail:
    return (mpi_errno);
}




#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_nonblocked_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_nonblocked_MV2(const void *sendbuf, int sendcount,
                                                   MPI_Datatype sendtype, void *recvbuf,
                                                   int recvcount, MPI_Datatype recvtype,
                                                   MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather [%d] [%d]  host = %s Sec=%d count = %d] Func: MPIR_2lvl_Allgather_nonblocked_MV2\n", comm_ptr->rank,init_rank,hostname,security_approach,recvcount);fflush(stdout);
	}
#endif  

    int mpi_errno_ret = MPI_SUCCESS;
	int i;
    int mpi_errno = MPI_SUCCESS;

    if (recvcount == 0) {
        return MPI_SUCCESS;
    }

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    // if(rank==0) printf("MPIR_2lvl_Allgather_nonblocked_MV2\n");
    // int* node_sizes = NULL; // comm_ptr->node_comm->local_size;
    MPIR_CHKLMEM_DECL(4);

    /* get extent of receive type */
    MPI_Aint recvtype_extent;
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* get true extent of recvtype */
    MPI_Aint recvtype_true_lb, recvtype_true_extent;
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);

    /* get info about communicator for ranks on the same node */
    MPIR_Comm* shmem_commptr=comm_ptr->node_comm;
    // MPI_Comm shmem_comm = comm_ptr->node_comm;
    //MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    int local_rank = comm_ptr->node_comm->rank;
    int local_size = comm_ptr->node_comm->local_size;

    /* TODO: if ranks are in order, we can avoid the temp buffer and memcpy */
    int need_temp = 1;

    /* get info about communicator across node leaders, allocate temp buffer */
    MPIR_Comm* leader_commptr = NULL;
    int leader_rank = -1;
    int leader_size = 0;
    void* tmpbuf = recvbuf;
    
    // if (local_rank == 0) {
    if (comm_ptr->node_roots_comm != NULL) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        //MPI_Comm leader_comm = comm_ptr->dev.ch.leader_comm;
        //MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_commptr = comm_ptr->node_roots_comm;
        leader_rank = comm_ptr->node_roots_comm->rank;
        leader_size = comm_ptr->node_roots_comm->local_size;
        

        /* allocate a temporary buffer */
        if (need_temp) {
            // tmpbuf = MPIU_Malloc(size * recvcount * recvtype_extent);
            
            tmpbuf = MPL_malloc(size * recvcount * recvtype_extent, MPL_MEM_USER );
            // node_sizes = MPL_malloc(sizeof (int) * leader_size, MPL_MEM_USER );

            int i;

            // for (i=0; i<leader_size; i++)
                // node_sizes[i]=local_size;
            
            if (!tmpbuf) {
            
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                                 "**nomem", 0);
                return mpi_errno;
            }
            

            /* adjust for potential negative lower bound in datatype */
            tmpbuf = (void*)((char*) tmpbuf - recvtype_true_lb);
            
        }
    }

    

    MPI_Aint preceding_count = 0;
    
    for (i=0; i < leader_rank; i++) {
    
        preceding_count += node_sizes[i] * recvcount;
        
    }

    

    if (local_rank == 0) {
    //if (comm_ptr->node_roots_comm != NULL) {
    
    
        /* compute location to receive data from procs on our node */
        void* rbuf = (void*)((char*)tmpbuf + (preceding_count * recvtype_extent));

        /* gather data from procs on our node */
        if (sendbuf == MPI_IN_PLACE) {
            /* data is in our receive buffer indexed by our rank */
            void* sbuf = (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent));
            mpi_errno = MPIR_Gather(sbuf, recvcount, recvtype,
                                    rbuf, recvcount, recvtype,
                                    0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather(sendbuf, sendcount, sendtype,
                                    rbuf, recvcount, recvtype,
                                    0, shmem_commptr, errflag);
        }
        
    } else {
        /* send data to leader on our node */
        
        if (sendbuf == MPI_IN_PLACE) {
            /* data is in our receive buffer indexed by our rank */
            void* sbuf = (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent));
            mpi_errno = MPIR_Gather(sbuf, recvcount, recvtype,
                                    NULL, recvcount, recvtype,
                                    0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather(sendbuf, sendcount, sendtype,
                                    NULL,    recvcount, recvtype,
                                    0, shmem_commptr, errflag);
        }
        
    }
    
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
    }

    /* ----------------------------------------------
     * Execute allgather across leaders
     * ---------------------------------------------- */

    /*If there is just one node, after gather itself,
     * root has all the data and it can do bcast*/

    /* Exchange the data between the node leaders */

    // if (local_rank == 0 && (leader_size > 1)) {
    if ((comm_ptr->node_roots_comm != NULL)  && (leader_size > 1)) {
        if(security_approach == 2007){
            /************** NAIVE PLUS (NB)***************/
	        //printf("%d @ 0\n", rank);
            unsigned long count=0;
            unsigned long next, dest;
            unsigned int i;
			
			// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 06-0 c=%d\n",init_rank,rank,recvcount);

            /*Step 1: Encryption*/
            int sendtype_sz, recvtype_sz;
            unsigned long  large_send_buffer_len = 0;
            sendtype_sz= recvtype_sz= 0;
            // int var;
            // var=MPI_Type_size(sendtype, &sendtype_sz);
            // var=MPI_Type_size(recvtype, &recvtype_sz);
            MPIR_Datatype_get_extent_macro(sendtype, sendtype_sz);
            MPIR_Datatype_get_extent_macro(recvtype, recvtype_sz);
            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 06-1 c=%d\n",init_rank,rank,recvcount);

            RAND_bytes(large_send_buffer, 12); // 12 bytes of nonce

            unsigned long t=0;
            t = (unsigned long)(local_size*sendtype_sz*sendcount);
            unsigned long   max_out_len = (unsigned long) (16 + (local_size*sendtype_sz*sendcount));

            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 06-2 c=%d\n",init_rank,rank,recvcount);
	    
            if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer+12,
                                &large_send_buffer_len, max_out_len,
                                large_send_buffer, 12,
                                tmpbuf + (preceding_count * recvtype_extent),  t,
                                NULL, 0))
            {
                    printf("Error in Naive+ encryption: allgather\n");
                    fflush(stdout);
            }
	        //printf("%d @ 1\n", rank);
            /*Step 2: Data exchange*/

            /*When data in each socket is different*/
            // Mohsen: each node has diffrent number of process
            // if (comm_ptr->dev.ch.is_uniform != 1) {            
            if (comm_ptr->is_uniform != 1) {
				
                int *displs = NULL;
                int *recvcnts = NULL;
                // int *node_sizes;
                int i = 0;

                // node_sizes = comm_ptr->dev.ch.node_sizes;
                // node_sizes = comm_ptr->node_comm->local_size;

                // displs = MPIU_Malloc(sizeof (int) * leader_size);
                // recvcnts = MPIU_Malloc(sizeof (int) * leader_size);

                

                displs = MPL_malloc(sizeof (int) * leader_size, MPL_MEM_USER );
                recvcnts = MPL_malloc(sizeof (int) * leader_size, MPL_MEM_USER );

                

                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }

                
                recvcnts[0] = node_sizes[0] * recvcount * recvtype_extent + 12 + 16;
                
                displs[0] = 0;

                for (i = 1; i < leader_size; i++) {
                    
                    displs[i] = displs[i - 1] + (node_sizes[i - 1] * recvcount * recvtype_extent + 12 + 16);
                    recvcnts[i] = node_sizes[i] * recvcount * recvtype_extent + 12 + 16;
                }

                /* mpi_errno = MPIR_Allgatherv_impl(large_send_buffer, (max_out_len+12), MPI_CHAR, 
                                        large_recv_buffer, recvcnts, displs, MPI_CHAR,
                                        leader_commptr, errflag); */
				
                mpi_errno = MPIR_Allgatherv_impl(large_send_buffer, (max_out_len+12), MPI_CHAR, 
                                        large_recv_buffer, recvcnts, displs, MPI_CHAR,
                                        leader_commptr, errflag);
				
		    //MPIU_Free(displs);
                //MPIU_Free(recvcnts);
		    //printf("%d @ 2 (1)\n", rank);
                /*Step3: Decryption*/

                for( i = 0; i < leader_size; i++){
                    if(i!= leader_rank){
                    
                    next =(unsigned long )(displs[i]);
                    dest =(unsigned long )(node_sizes[i]*(recvcount*recvtype_sz));
                            
                    //printf("%d is going to decrypt from %d and put at %d, node size = %d\n", rank, next,dest, node_sizes[i] );
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((tmpbuf+dest)),
                                &count, (unsigned long )((node_sizes[i]*recvcount*recvtype_sz)),
                                (large_recv_buffer+next), 12,
                                (large_recv_buffer+next+12), (unsigned long )((node_sizes[i]*recvcount*recvtype_sz)+16), NULL, 0)){
                        printf("Decryption error in Naive+ allgather r = %d  dst = %d c = %d\n",comm_ptr->rank, i,recvcount);fflush(stdout);            
                                }                               
                    }
                }
		//printf("%d @ 3 (1)\n", rank);
		MPL_free(recvcnts);
		MPL_free(displs);
            } else {
		
                /* mpi_errno = MPIR_Allgather_impl(large_send_buffer, (max_out_len+12), MPI_CHAR,
                                                large_recv_buffer, (max_out_len+12), MPI_CHAR,
                                                leader_commptr, errflag); */

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 09 c=%d\n",init_rank,rank,recvcount);
				mpi_errno = MPIR_Allgather_intra_ring(large_send_buffer, (max_out_len+12), MPI_CHAR,
                                                large_recv_buffer, (max_out_len+12), MPI_CHAR,
                                                leader_commptr, errflag);

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 10 c=%d\n",init_rank,rank,recvcount);
				//printf("%d @ 2 (2)\n", rank);
                /*Step3: Decryption*/
                    for( i = 0; i < leader_size; i++){
                        if(i != leader_rank){
                        
                        next =(unsigned long )(i*(max_out_len+12));
                        dest =(unsigned long )(i*(local_size*sendtype_sz*sendcount));
                                

                        if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((tmpbuf+dest)),
                                    &count, (unsigned long )((local_size*recvcount*recvtype_sz)),
                                    (large_recv_buffer+next), 12,
                                    (large_recv_buffer+next+12), (unsigned long )((local_size*recvcount*recvtype_sz)+16), NULL, 0)){
                            printf("Decryption error in Naive+ allgather (NB) 2\n");fflush(stdout);        
                                    }                               
                        }
                    }
                //	printf("%d @ 3 (2)\n", rank);

            }

            /********* END OF NAIVE PLUS **********/
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* ----------------------------------------------
     * Unpack data into receive buffer in correct order
     * ---------------------------------------------- */

    /* ----------------------------------------------
     * TODO: can avoid temp buffer and mem copy when in block
     * ---------------------------------------------- */
    /* ----------------------------------------------
     * TODO: faster memcpy available?
     * ---------------------------------------------- */

    /* leaders copy data from temp buffer to receive buffer in correct order */
    if (local_rank == 0 && need_temp) {
        /* point to start of temp buffer */
        char* sbuf = (char*)tmpbuf;

        /* copy data for each rank from temp buffer to receive buffer */
        for (i = 0; i < size; i++) {
            /* get next rank in list */
            int dstrank = comm_ptr->comm_rank_list[i];

            /* compute position in receive buffer for this rank */
            void* rbuf = (void*)((char*)recvbuf + dstrank * recvcount * recvtype_extent);

            /* copy data to its correct place */
            mpi_errno = MPIR_Localcopy(sbuf, recvcount, recvtype,
                                       rbuf, recvcount, recvtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* update pointer to next spot in temp buffer */
            sbuf += recvcount * recvtype_extent;
        }
	
        /* free the temporary buffer if we allocated one */
        tmpbuf = (void*)((char*)tmpbuf + recvtype_true_lb);
        MPL_free(tmpbuf);
	

    }
    

    /* ----------------------------------------------
     * Broadcast receive buffer from leader to all procs on the node
     * ---------------------------------------------- */

    /* Bcast the entire data from node leaders to other ranks on node */
	// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 11 c=%d\n",init_rank,rank,recvcount);
    mpi_errno = MPIR_Bcast(recvbuf, recvcount * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
	// if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] NBlock 12 c=%d\n",init_rank,rank,recvcount);

    
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
#define FUNCNAME MPIR_Allgather_inter_local_gather_remote_bcast
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_inter_local_gather_remote_bcast(const void *sendbuf, int sendcount,
                                                   MPI_Datatype sendtype, void *recvbuf,
                                                   int recvcount, MPI_Datatype recvtype,
                                                   MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather [%d] [%d]  host = %s Sec=%d count = %d] Func: MPIR_Allgather_inter_local_gather_remote_bcast\n", comm_ptr->rank,init_rank,hostname,security_approach,recvcount);fflush(stdout);
	}
#endif  	
    int rank, local_size, remote_size, mpi_errno = MPI_SUCCESS, root;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint true_extent, true_lb = 0, extent, send_extent;
    void *tmp_buf = NULL;
    MPIR_Comm *newcomm_ptr = NULL;

    MPIR_CHKLMEM_DECL(1);

    local_size = comm_ptr->local_size;
    remote_size = comm_ptr->remote_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_inter_local_gather_remote_bcast\n");
    int my_node;
    MPID_Get_node_id(comm_ptr, rank, &my_node);
	
	// fprintf(stderr,"[%d %d] l_size = %d  r_size = %d\n",comm_ptr->rank,init_rank,local_size,remote_size);

    if ((rank == 0) && (sendcount != 0)) {
        /* In each group, rank 0 allocates temp. buffer for local
         * gather */
        // fprintf(stderr,"[%d] [%d] [%d] ALG-LGRB 00 \n",init_rank,rank,my_node);

        MPIR_Type_get_true_extent_impl(sendtype, &true_lb, &true_extent);

        MPIR_Datatype_get_extent_macro(sendtype, send_extent);
        extent = MPL_MAX(send_extent, true_extent);

        MPIR_Ensure_Aint_fits_in_pointer(extent * sendcount * local_size);
        MPIR_CHKLMEM_MALLOC(tmp_buf, void *, extent * sendcount * local_size, mpi_errno, "tmp_buf",
                            MPL_MEM_BUFFER);

        /* adjust for potential negative lower bound in datatype */
        tmp_buf = (void *) ((char *) tmp_buf - true_lb);
    }
	
	// fprintf(stderr,"[%d] [%d] ALG-LGRB 01\n",init_rank,recvcount);

    /* Get the local intracommunicator */
    if (!comm_ptr->local_comm)
        MPII_Setup_intercomm_localcomm(comm_ptr);

    newcomm_ptr = comm_ptr->local_comm;

    if (sendcount != 0) {
		// fprintf(stderr,"[%d] [%d] ALG-LGRB 02\n",init_rank,newcomm_ptr->local_size);
        mpi_errno = MPIR_Gather(sendbuf, sendcount, sendtype, tmp_buf, sendcount,
                                sendtype, 0, newcomm_ptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
	
	// fprintf(stderr,"[%d] [%d] ALG-LGRB 03\n",init_rank,recvcount);

    /* first broadcast from left to right group, then from right to
     * left group */
#if 0	 
    if (comm_ptr->is_low_group) {
        /* bcast to right */
			
        if (sendcount != 0) {
			// fprintf(stderr,"[%d] ALG-LGRB 04\n",init_rank);
            root = (rank == 0) ? MPI_ROOT : MPI_PROC_NULL;
            mpi_errno = MPIR_Bcast(tmp_buf, sendcount * local_size,
                                   sendtype, root, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        /* receive bcast from right */
        if (recvcount != 0) {
			// fprintf(stderr,"[%d] ALG-LGRB 05\n",init_rank);
            root = 0;
            mpi_errno = MPIR_Bcast(recvbuf, recvcount * remote_size,
                                   recvtype, root, comm_ptr, errflag);
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
        /* receive bcast from left */
        if (recvcount != 0) {			
            root = 0;
			// fprintf(stderr,"[%d] [%d] [%d]ALG-LGRB 06\n",init_rank,recvcount,recvcount * remote_size);
            mpi_errno = MPIR_Bcast(recvbuf, recvcount * remote_size,
            //mpi_errno = MPIR_Bcast_inter_remote_send_local_bcast(recvbuf, recvcount * remote_size,
                                   recvtype, root, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        /* bcast to left */
        if (sendcount != 0) {			
            root = (rank == 0) ? MPI_ROOT : MPI_PROC_NULL;
			// fprintf(stderr,"[%d] ALG-LGRB 07 root= %d size = %d\n",init_rank,root,sendcount * local_size);
            mpi_errno = MPIR_Bcast(tmp_buf, sendcount * local_size,
            //mpi_errno = MPIR_Bcast_inter_remote_send_local_bcast(tmp_buf, sendcount * local_size,
                                   sendtype, root, comm_ptr, errflag);
			// fprintf(stderr,"[%d] ALG-LGRB 08\n",init_rank);
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
#endif	
	
	mpi_errno = MPIR_Bcast(tmp_buf, sendcount * local_size, sendtype, 0, comm_ptr, errflag);
	
	 if (rank == 0){ 
	 // fprintf(stderr,"[%d] [%d] ALG-LGRB 08-1\n",init_rank,recvcount);
			mpi_errno = MPIR_Localcopy(tmp_buf, sendcount * local_size, sendtype,
										recvbuf, sendcount * local_size, sendtype);
								}
	
	// fprintf(stderr,"[%d] [%d] ALG-LGRB 09\n",init_rank,recvcount);

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