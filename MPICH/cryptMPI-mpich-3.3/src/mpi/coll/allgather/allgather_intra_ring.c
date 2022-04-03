/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */


#include "mpiimpl.h"
#include "secure_allgather.h"


#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_MV2(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d  Sec-AP = %d] Func: MPIR_2lvl_Allgather_MV2 (SH1-un-balance)\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
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
    // if(rank==0) printf("MPIR_2lvl_Allgather_MV2\n");
    MPIR_CHKLMEM_DECL(3);

    //MPID_Get_node_id(comm_ptr, rank, &node_id);
    // int my_node = node_id;
    
    if (recvcount == 0) {
        return MPI_SUCCESS;
    }

    // extract the rank,size information for the intra-node communicator
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] ALG-Conc 01\n",init_rank,recvcount);    
    
    shmem_commptr = comm_ptr->node_comm;
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 01 c=%d l_rank = %d  l-size = %d\n",init_rank,recvcount,local_rank,local_size);
    // int p = shmem_commptr->local_size; // number of ranks per node
    //int p = local_size;
    // int n = (int) (size / p); // number of nodes
    //int n = node_cnt;
    
    // conc_comm = comm_ptr->concurrent_comm;    
    // MPIR_Comm_get_ptr(conc_comm, conc_commptr);
    if (local_rank == 0) {
        // Node leader. Extract the rank, size information for the leader communicator        
        leader_commptr = comm_ptr->node_roots_comm;
        leader_rank = comm_ptr->node_roots_comm->rank;
        leader_comm_size = comm_ptr->node_roots_comm->local_size;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 01-1 c=%d led_rank = %d  led-size = %d\n",init_rank,recvcount,leader_rank,leader_comm_size);
    }

       if(local_rank == 0) {
           // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 02 c=%d\n",init_rank,recvcount);
        mpi_errno = MPIR_Gather_impl(sendbuf, sendcount,sendtype, 
                                    (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent)), 
                                     recvcount, recvtype,
                                     0, shmem_commptr, errflag);
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 03 c=%d\n",init_rank,recvcount);
    } else {
        /*Since in allgather all the processes could have 
         * its own data in place*/
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 04 c=%d\n",init_rank,recvcount);
        if(sendbuf == MPI_IN_PLACE) {
            mpi_errno = MPIR_Gather_impl((void*)((char*)recvbuf + (rank * recvcount * recvtype_extent)), 
                                         recvcount , recvtype, 
                                         recvbuf, recvcount, recvtype,
                                         0, shmem_commptr, errflag);
        } else {
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcount,sendtype, 
                                         recvbuf, recvcount, recvtype,
                                         0, shmem_commptr, errflag);
        }
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 05 c=%d\n",init_rank,recvcount);
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Exchange the data between the node leaders*/
    if (local_rank == 0 && (leader_comm_size > 1)) {
        if(security_approach == 2007){
            //NAIVE PLUS
	    //printf("Naive+\n");
            unsigned long count=0;
            unsigned long next, dest;
            unsigned int i;
            /*Step 1: Encryption*/


            int sendtype_sz, recvtype_sz;
            unsigned long  ciphertext_sendbuf_len = 0;
            sendtype_sz= recvtype_sz= 0;
            int var;
            MPIR_Datatype_get_extent_macro(sendtype, sendtype_sz);
            MPIR_Datatype_get_extent_macro(recvtype, recvtype_sz);

            int rank;
            rank = comm_ptr->rank;

            RAND_bytes(large_send_buffer, 12); // 12 bytes of nonce

            unsigned long t=0;
            t = (unsigned long)(local_size*sendtype_sz*sendcount);
            unsigned long   max_out_len = (unsigned long) (16 + (local_size*sendtype_sz*sendcount));

            if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer+12,
                                &ciphertext_sendbuf_len, max_out_len,
                                large_send_buffer, 12,
                                recvbuf + (rank * recvcount * recvtype_extent),  t,
                                NULL, 0))
            {
                    printf("Error in Naive+ encryption: allgather\n");
                    fflush(stdout);
            }
	    
            /*Step 2: Data exchange*/

            /*When data in each socket is different*/
            // Mohsen: each node has diffrent number of process
            //if (comm_ptr->dev.ch.is_uniform != 1) {
            
            if (comm_ptr->is_uniform != 1) {

                int *displs = NULL;
                int *recvcnts = NULL;
                // int *node_sizes;
                int i = 0;

                //node_sizes = comm_ptr->dev.ch.node_sizes;

                // node_sizes = MPL_malloc(sizeof (int) * node_cnt, MPL_MEM_USER );

                // for (i=0; i<node_cnt; i++)
                    // node_sizes[i]=local_size;

                // displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                // recvcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);

                displs = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                recvcnts = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
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

                for (i = 1; i < leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + (node_sizes[i - 1] * recvcount * recvtype_extent + 12 + 16);
                    recvcnts[i] = node_sizes[i] * recvcount * recvtype_extent + 12 + 16;
                    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 06 c=%d   disp[%d] = %d  recvct[%d] = %d\n",init_rank,recvcount,i,displs[i],i,recvcnts[i]);
                }

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 07 c=%d\n",init_rank,recvcount);


                mpi_errno = MPIR_Allgatherv_intra_recursive_doubling(large_send_buffer, (max_out_len+12), MPI_CHAR, 
                                        large_recv_buffer, recvcnts, displs, MPI_CHAR,
                                        leader_commptr, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 08 c=%d\n",init_rank,recvcount);
                //MPIU_Free(displs);
                //MPIU_Free(recvcnts);

                /*Step3: Decryption*/

                for( i = 0; i < leader_comm_size; i++){
                    next =(unsigned long )(displs[i]);
                    dest =(unsigned long )(node_sizes[i]*(recvcount*recvtype_sz));
                    

                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((recvbuf+dest)),
                                    &count, (unsigned long )((node_sizes[i]*recvcount*recvtype_sz)),
                                    (large_recv_buffer+next), 12,
                                    (large_recv_buffer+next+12), (unsigned long )((node_sizes[i]*recvcount*recvtype_sz)+16),
                                    NULL, 0)){
                                printf("Decryption error in Naive+ allgather r = %d  dst = %d c = %d\n",comm_ptr->rank, i,recvcount);fflush(stdout);        
                        }                               
                
                }

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 09 c=%d\n",init_rank,recvcount);


            } else {
                // init_rank =1;
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 10 c=%d\n",init_rank,recvcount);
                mpi_errno = MPIR_Allgather_intra_ring(large_send_buffer, (max_out_len+12), MPI_CHAR,
                                                large_recv_buffer, (max_out_len+12), MPI_CHAR,
                                                leader_commptr, errflag);


                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 11 c=%d\n",init_rank,recvcount);
                // init_rank =0;
                /*Step3: Decryption*/
                for( i = 0; i < leader_comm_size; i++){
                    next =(unsigned long )(i*(max_out_len+12));
                    dest =(unsigned long )(i*(local_size*sendtype_sz*sendcount));
                    

                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((recvbuf+dest)),
                                    &count, (unsigned long )((local_size*recvcount*recvtype_sz)),
                                    (large_recv_buffer+next), 12,
                                    (large_recv_buffer+next+12), (unsigned long )((local_size*recvcount*recvtype_sz)+16),
                                    NULL, 0)){
                                printf("Decryption error in Naive+ allgather r = %d  dst = %d c = %d\n",comm_ptr->rank, i,recvcount);fflush(stdout);        
                        }                               
                
                }
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 12 c=%d\n",init_rank,recvcount);

            }
            /**********************     End of NAIVE PLUS ********************/
        }else{
            /*When data in each socket is different*/
            // Mohsen: each node has diffrent number of process
            // if (comm_ptr->dev.ch.is_uniform != 1) {
            if (comm_ptr->is_uniform != 1) {

                int *displs = NULL;
                int *recvcnts = NULL;
                // int *node_sizes;
                int i = 0;

                //node_sizes = comm_ptr->dev.ch.node_sizes;

                // node_sizes = MPL_malloc(sizeof (int) * node_cnt, MPL_MEM_USER );

                // for (i=0; i<node_cnt; i++)
                    // node_sizes[i]=local_size;

                // displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                // recvcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);

                displs = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                recvcnts = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);
                    return mpi_errno;
                }
                recvcnts[0] = node_sizes[0] * recvcount;
                displs[0] = 0;

                for (i = 1; i < leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + node_sizes[i - 1] * recvcount;
                    recvcnts[i] = node_sizes[i] * recvcount;
                }


                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 13 c=%d\n",init_rank,recvcount);
                mpi_errno = MPIR_Allgatherv_intra_recursive_doubling(MPI_IN_PLACE,
                                        (recvcount*local_size),
                                        recvtype, 
                                        recvbuf, recvcnts,
                                        displs, recvtype,
                                        leader_commptr, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 14 c=%d\n",init_rank,recvcount);
                // MPIU_Free(displs);
                // MPIU_Free(recvcnts);
            } else {
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 15 c=%d\n",init_rank,recvcount);
                mpi_errno = MPIR_Allgather_intra_ring(MPI_IN_PLACE, 
                                                (recvcount*local_size),
                                                recvtype,
                                                recvbuf, (recvcount*local_size), recvtype,
                                                leader_commptr, errflag);

                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 16 c=%d\n",init_rank,recvcount);
            }
        }
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] RD-2lvl 17 c=%d\n",init_rank,recvcount);

    /*Bcast the entire data from node leaders to all other cores*/
    mpi_errno = MPIR_Bcast_impl (recvbuf, recvcount * size, recvtype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }



  fn_fail:
    return (mpi_errno);
}



#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather (Single-leader)\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
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
    // if(rank==0) printf("MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather\n");
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    int my_node = node_id;
    
    if (recvcount == 0) {
        return MPI_SUCCESS;
    }

    // extract the rank,size information for the intra-node communicator
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    
    // void* my_shmem_buffer = (void*)((char*)shmem_buffer + comm_ptr->shmem_idx * recvcount *recvtype_extent);
    // void* my_ciphertext_shmem_buffer = (void*)((char*)ciphertext_shmem_buffer + comm_ptr->ctx_shmem_idx * (recvcount *recvtype_extent + 16 + 12));

    // void* my_shmem_buffer = shmem_buffer;
    // void* my_ciphertext_shmem_buffer = ciphertext_shmem_buffer;

    shmem_commptr = comm_ptr->node_comm;
    //MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;
    
    int p = local_size;
    
    int n = node_cnt;
    
    
    if (local_rank == 0) {
        // Node leader. Extract the rank, size information for the leader communicator        
        leader_commptr = comm_ptr->node_roots_comm;
        leader_rank = comm_ptr->node_roots_comm->rank;
        leader_comm_size = comm_ptr->node_roots_comm->local_size;
    
    }
    
    //First, copy plaintext to the shmem_buf
    if(security_approach == 2006){

        unsigned long count=0;
        unsigned long next, dest;
        unsigned int i;

        unsigned long  ciphertext_len = 0;
        void* out = (void*)((char*) ciphertext_shmem_buffer + (my_node * p + local_rank) * (recvcount * recvtype_extent + 12 + 16));
        void* in;
        if(sendbuf == MPI_IN_PLACE) {
            in = (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent));
        }else{
            in = (void*)((char*)sendbuf);
        }  

        RAND_bytes(out, 12); // 12 bytes of nonce

        unsigned long t=0;
        t = (unsigned long)(recvcount * recvtype_extent);
        unsigned long   max_out_len = (unsigned long) (16 + t);
        if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,  
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, t,
                            NULL, 0))
        {
                printf("Error in Naive+ concurrent encryption: allgather-shmem\n");
                fflush(stdout);
        }
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
	    }
        
                //copy plaintext to the shmem_buf
        /*mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
	  (void*)((char*)my_shmem_buffer + (my_node * p + local_rank)), recvcount, recvtype);*/
	if(local_rank == 0){
           
	    mpi_errno = MPIR_Gather_impl(in, 
					 recvcount , recvtype,
					 (void*)((char*)shmem_buffer + (my_node * p * recvcount * recvtype_extent)), 
					 recvcount, recvtype,
					 0, shmem_commptr, errflag);

	} else {
	    //Since in allgather all the processes could have its own data in place
	    mpi_errno = MPIR_Gather_impl(in, 
                                         recvcount , recvtype, 
                                         NULL, recvcount, recvtype,
                                         0, shmem_commptr, errflag);
	}
	/*mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
	    }*/
	
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        // Exchange the data between the node leaders
        if (local_rank == 0 && (leader_comm_size > 1)) {
            
            unsigned long count=0;
            unsigned long next, dest;
            unsigned int i;
            
            mpi_errno = MPIR_Allgather_intra_recursive_doubling(MPI_IN_PLACE, p*(max_out_len+12), MPI_CHAR,
                                            ciphertext_shmem_buffer, p*(max_out_len+12), MPI_CHAR,
                                            leader_commptr, errflag);
            
            /* mpi_errno = MPIR_Allgather_impl(out, p*(max_out_len+12), MPI_CHAR,
                                            ciphertext_shmem_buffer, p*(max_out_len+12), MPI_CHAR,
                                            leader_commptr, errflag); */
            
            //Step3: Decryption
            
            for(i = 0; i < n; i+=1){
                if(i != my_node){
                    idx = i * p + local_rank;
                    next =(unsigned long )(idx*(max_out_len+12));
                    dest =(unsigned long )(idx*t);
                    
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((shmem_buffer+dest)),
                                    &count, t,
                                    (ciphertext_shmem_buffer+next), 12,
                                    (ciphertext_shmem_buffer+next+12), t+16,
                                    NULL, 0)){
                            printf("Decryption error in Naive+ shmem_allgather Concurrent Encryption while %d tried to decrypt %d from %d to %d\n", rank, count, next, dest);fflush(stdout);        
                    }
                     
                }                               
            }//end for
            //printf("%d @ check4\n", rank);

            //End of NAIVE PLUS
        

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        
        } 


        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }


        if(local_rank!=0){
        
            //help in decryption
            int sendtype_sz, recvtype_sz;
            unsigned long count=0;
            sendtype_sz= recvtype_sz= 0;
            int var;
            
            MPIR_Datatype_get_extent_macro(sendtype, sendtype_sz);
            MPIR_Datatype_get_extent_macro(recvtype, recvtype_sz);
            unsigned long   max_out_len = (unsigned long) (16 + (sendtype_sz*sendcount));
            int i, next, dest;
            //printf("%d- local_rank:%d, leader_Comm_size: %d, rank/p:%d\n", rank, local_rank, leader_comm_size, (int)(rank/p));
            for( i = 0; i < n; i+=1){
                //printf("%d- i:%d, local_rank:%d, leader_Comm_size: %d, rank/p:%d\n", rank, i, local_rank, leader_comm_size, (int)(rank/p));
                if(i != my_node){
                    idx = i * p + local_rank;
                    next =(unsigned long )(idx*(max_out_len+12));
                    dest =(unsigned long )(idx*(sendtype_sz*sendcount));
                    
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((shmem_buffer+dest)),
                                    &count, (unsigned long )((recvcount*recvtype_sz)),
                                    (ciphertext_shmem_buffer+next), 12,
                                    (ciphertext_shmem_buffer+next+12), (unsigned long )((recvcount*recvtype_sz)+16),
                                    NULL, 0)){
                            printf("Decryption error in Naive+ shmem_allgather Conncurrent Encryption II while %d tried to decrypt %d from %d to %d\n", rank, count, next, dest);fflush(stdout);        
                    }
                }
                                            
            }//end for
        }//end if

        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
        if(comm_ptr->is_blocked == 1){
            //Blocked
            mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), recvcount * size, recvtype, 
                                    (void*)((char*)recvbuf), recvcount * size, recvtype);

        }else{
            //NonBlocked
            int s=0;
            for(; s<size; ++s){
                
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer + s * recvcount  * recvtype_extent), recvcount , recvtype, 
                                        (void*)((char*)recvbuf + comm_ptr->comm_rank_list[s] * recvcount * recvtype_extent), recvcount, recvtype);
                
            }
        }
        
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

    }else{
        printf("SECURITY_APPROACH = %d should be set to 2 for MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather_MV2\n", security_approach);
        goto fn_fail;
    }


  fn_fail:
    return (mpi_errno);
}




#include "mpiimpl.h"
#include "secure_allgather.h"

#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_2lvl_Concurrent_Multileader_SharedMem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_2lvl_Concurrent_Multileader_SharedMem(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_Allgather_2lvl_Concurrent_Multileader_SharedMem\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	
    int rank, size;
    int local_rank, local_size, idx;
     
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_extent = 0;  
    MPI_Comm shmem_comm, conc_comm;
    MPIR_Comm *shmem_commptr=NULL, *conc_commptr = NULL;
    int node_id;    
    rank = comm_ptr->rank;
    size = comm_ptr->local_size;
    // if(rank==0) printf("MPIR_Allgather_2lvl_Concurrent_Multileader_SharedMem\n");
    MPID_Get_node_id(comm_ptr, rank, &node_id);
    int my_node = node_id;
    
    if (recvcount == 0) {
        return MPI_SUCCESS;
    }

    // extract the rank,size information for the intra-node communicator
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    
    // void* my_shmem_buffer = (void*)((char*)shmem_buffer + comm_ptr->shmem_idx * recvcount *recvtype_extent);
    // void* my_ciphertext_shmem_buffer = (void*)((char*)ciphertext_shmem_buffer + comm_ptr->ctx_shmem_idx * (recvcount *recvtype_extent + 16 + 12));

    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;    
    int p = local_size;
    int n = node_cnt;
    conc_comm = comm_ptr->concurrent_comm;    
    MPIR_Comm_get_ptr(conc_comm, conc_commptr);  
    
    if(security_approach == 2006){
        //First, copy plaintext to the shmem_buf
        void* in;
        if(sendbuf == MPI_IN_PLACE) {
            in = (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent));
        }else{
            in = (void*)((char*)sendbuf);
        }  
        
        mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
                                (void*)((char*)shmem_buffer + (my_node * p + local_rank)*(recvcount * recvtype_extent)),
                                recvcount, recvtype);

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        //Then, each process should encrypt its data
        unsigned long count=0;
        unsigned long next, dest;
        unsigned int i;
        unsigned long  ciphertext_len = 0;
        // void* out = (void*)((char*) ciphertext_recvbuf + (my_node * p + local_rank) * (recvcount * recvtype_extent + 12 + 16));
        void* out = (void*)((char*) ciphertext_shmem_buffer + (my_node * p + local_rank) * (recvcount * recvtype_extent + 12 + 16));
        RAND_bytes(out, 12); // 12 bytes of nonce
        unsigned long t=0;
        t = (unsigned long)(recvcount * recvtype_extent);
        unsigned long   max_out_len = (unsigned long) (16 + t);

        if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,  
                            &ciphertext_len, max_out_len,
                            out, 12,
                            in, t,
                            NULL, 0))
        {
                printf("Error in Naive+ concurrent encryption: allgather-shmem\n");
                fflush(stdout);
        }
        
        int alg_choice = mv2_user_allgather_inter;
        mv2_user_allgather_inter = 0;

        mpi_errno = MPIR_Allgather_impl(out, (max_out_len+12), MPI_CHAR,
                                            ciphertext_recvbuf, (max_out_len+12), MPI_CHAR,
                                            conc_commptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mv2_user_allgather_inter = alg_choice;

        // decrypt    
        max_out_len = (unsigned long) (16 + (recvtype_extent*sendcount));
        for(i = 0; i < n; i+=1){    
            //printf("%d- i:%d, local_rank:%d, leader_Comm_size: %d, rank/p:%d\n", rank, i, local_rank, leader_comm_size, (int)(rank/p));
            if(i != my_node){
                idx = i * p + local_rank;
                next =(unsigned long )(i*(max_out_len+12));
                dest =(unsigned long )(idx*(recvtype_extent*sendcount));
                
                if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((shmem_buffer+dest)),
                                &count, (unsigned long )((recvcount*recvtype_extent)),
                                (ciphertext_recvbuf+next), 12,
                                (ciphertext_recvbuf+next+12), (unsigned long )((recvcount*recvtype_extent)+16),
                                NULL, 0)){
                        printf("Decryption error in Concurrent Multileader shmem_allgather while %d tried to decrypt %d from %d to %d idx=%d (%d)\n", rank, count, next, dest,idx,i);fflush(stdout);        
                }
            }                      
        }//end for

        //barrier
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        } 

        //copy to user buffer
        if (comm_ptr->is_blocked==1){
            
            //Blocked
            mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), recvcount * size, recvtype, 
                                    (void*)((char*)recvbuf), recvcount * size, recvtype);
                
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }


        }else{
            //NonBlocked
            int s=0;
            for(; s<size; ++s){
                
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer + s * recvcount  * recvtype_extent), recvcount , recvtype, 
                                        (void*)((char*)recvbuf + comm_ptr->comm_rank_list[s] * recvcount * recvtype_extent), recvcount, recvtype);

                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }

            }
        }       

    }else{
        //For the unencrypted version, we first need to perform concurrent allgather
        void* in;
        if(sendbuf == MPI_IN_PLACE) {
            in = (void*)((char*)recvbuf + (rank * recvcount * recvtype_extent));
            mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
                                sendbuf, recvcount, recvtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        in = (void*)((char*)sendbuf);
        
        int alg_choice = mv2_user_allgather_inter;
        mv2_user_allgather_inter = 0;

        mpi_errno = MPIR_Allgather_impl(in, recvcount, recvtype,
                                            recvbuf, recvcount, recvtype,
                                            conc_commptr, errflag);
        mv2_user_allgather_inter=alg_choice;
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        //copy each message to the appropriate location in shmem_buffer
        int s=0;
        for(; s<n; ++s){
            in = (void*)((char*)recvbuf + (s * recvcount * recvtype_extent));
            void* out = (void*)((char*)shmem_buffer + ((s*p+local_rank) * recvcount * recvtype_extent));

            mpi_errno = MPIR_Localcopy(in, recvcount, recvtype, 
                                out, recvcount, recvtype);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }


        //barrier
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        } 


        //copy to user buffer
         if(comm_ptr->is_blocked==1){
            //Blocked
            mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), recvcount * size, recvtype, 
                                    (void*)((char*)recvbuf), recvcount * size, recvtype);
                
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }


        }else{
            //NonBlocked
            int s=0;
            for(; s<size; ++s){
                
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer + s * recvcount  * recvtype_extent), recvcount , recvtype, 
                                        (void*)((char*)recvbuf + comm_ptr->comm_rank_list[s] * recvcount * recvtype_extent), recvcount, recvtype);

                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
   
            }
        }
    }

  fn_fail:
    return (mpi_errno);
}





#undef FUNCNAME
#define FUNCNAME MPIR_2lvl_Allgather_Multileader_Ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_2lvl_Allgather_Multileader_Ring(const void *sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
        host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_2lvl_Allgather_Multileader_Ring\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i, j;
    // unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 01 c=%d\n",init_rank,sendcount);
    
    /* get info about communicator for ranks on the same node */
    MPIR_Comm* shmem_commptr;
    shmem_commptr = comm_ptr->node_comm;

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 02 c=%d\n",init_rank,sendcount);

    /* get our rank and the size of this communicator */
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    // if(rank==0)
    //     printf("MPIR_2lvl_Allgather_Multileader_Ring_MV2\n");
    int p = shmem_commptr->local_size; // number of ranks per node
    // int n = (int) (size / p); // number of nodes
    int max_node_id;
	mpi_errno = MPID_Get_max_node_id(comm_ptr, &max_node_id);
    int n = max_node_id +1;
    // ppn = (comm_ptr->local_size)/(node_cnt);



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

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 03 c=%d node_cnt=%d\n",init_rank,sendcount,n);
    
    int node_id, left_node_id, right_node_id;    
    MPID_Get_node_id(comm_ptr, rank, &node_id);

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 04 c=%d\n",init_rank,sendcount);

    if (comm_ptr->comm_rank_list == NULL) host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes , &comm_ptr->comm_rank_list , &comm_ptr->comm_rank_list_back);

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 05 c=%d\n",init_rank,sendcount);

    int rank_index = comm_ptr->comm_rank_list[rank], send_req_idx=0;

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 06 c=%d rank_index=%d\n",init_rank,sendcount,rank_index);

    //If naive+ encryption is selected, each rank needs to encrypt its local data first

    if(security_approach==2005){
        
        unsigned long  ciphertext_len = 0;
        //encrypt local data to ciphertext rcvbuffer
        void* in = (void*)((char*) recvbuf + comm_ptr->comm_rank_list[rank_index] * recvcount * recvtype_extent);
        void* out = (void*)((char*) ciphertext_recvbuf + node_id * (recvcount * recvtype_extent + 12 + 16));

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
                printf("Error in Naive+ encryption: allgather MultiLeader\n");
                fflush(stdout);
        }

    }

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 07 c=%d\n",init_rank,sendcount);

    /* lookup our index in the rank list */
    
    //    printf("%d is %d\n", rank, comm_ptr->comm_rank_list[rank_index]);
    /* compute the left and right neighbor ranks in the rank_list */
    int left_index  = (size + rank_index - p) % size;
    int right_index = (size + rank_index + p) % size;


    // Mohsen: for Cyclic Host this formula is wrong
    int left  = comm_ptr->comm_rank_list[left_index];
    int right = comm_ptr->comm_rank_list[right_index];

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] M-Ring 07 c=%d rank_index=%d  left_index=%d  left=%d  right_index=%d  right=%d\n",init_rank,sendcount,rank_index,left_index,left,right_index,right);

    left  = comm_ptr->comm_rank_list_back[left_index];
    right = comm_ptr->comm_rank_list_back[right_index];

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] M-Ring 08 c=%d rank_index=%d  left_index=%d  left=%d  right_index=%d  right=%d\n",init_rank,sendcount,rank_index,left_index,left,right_index,right);

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
    int send_index = rank_index;
    int recv_index = left_index;

    

    for (i=1; i < n; ++i){
	
        int send_rank = comm_ptr->comm_rank_list[send_index];
        int recv_rank = comm_ptr->comm_rank_list[recv_index];

        // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08 c=%d send_rank=%d  recv_rank=%d\n",init_rank,sendcount,send_rank,recv_rank);
        /* compute position within buffer to send from and receive into */
        //if(recvcount==16)
        //  	    printf("%d is going to send from %d to %d and receive from %d at %d\n", rank, send_rank, right, left, recv_rank);

        if(security_approach == 2005){
            MPID_Get_node_id(comm_ptr, send_rank, &right_node_id);
            MPID_Get_node_id(comm_ptr, recv_rank, &left_node_id);

            
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08-1 c=%d\n",init_rank,sendcount);
            void* sbuf = (void*)((char*) ciphertext_recvbuf + right_node_id * (recvcount * recvtype_extent + 16 + 12));
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08-2 c=%d\n",init_rank,sendcount);
            void* rbuf = (void*)((char*) ciphertext_recvbuf + left_node_id * (recvcount * recvtype_extent + 16 + 12));
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08-3 c=%d r=%d l=%d r=%d\n",init_rank,sendcount,rank,left_node_id,right_node_id);
        
            /* exchange data with our neighbors in the ring */
            // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR); 
            // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR); 

            // mpi_errno = MPIC_Sendrecv_Plus(
            // sbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
            // rbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
            // comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
            // );
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08-4 c=%d r=%d l=%d r=%d\n",init_rank,sendcount,rank,left,right);
            mpi_errno = MPIC_Sendrecv(
            sbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, right, MPIR_ALLGATHER_TAG,
            rbuf, (recvcount * recvtype_extent + 16 + 12), MPI_CHAR, left,  MPIR_ALLGATHER_TAG,
            comm_ptr, MPI_STATUS_IGNORE, errflag);
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 08-5 c=%d r=%d l=%d r=%d\n",init_rank,sendcount,rank,left_node_id,right_node_id);

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 09 c=%d\n",init_rank,sendcount);

            if(overlap_decryption == 1){
                // Decrypt recently received data
                unsigned long count=0;
                
                if(!EVP_AEAD_CTX_open(ctx, (recvbuf+recv_rank*recvcount*recvtype_extent),
                            &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))), 12,
                            (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))+12), (unsigned long )((recvcount*recvtype_extent)+16),
                            NULL, 0)){
                    printf("Error in Naive+ decryption: allgather MultiLeader\n");
                    fflush(stdout);        
                }
            }//end if overlap
            // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 10 c=%d\n",init_rank,sendcount);

        }else{
            void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
            void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
        
            /* exchange data with our neighbors in the ring */
            // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, recvcount, recvtype); 
            // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, recvcount, recvtype);
           
            // mpi_errno = MPIC_Sendrecv_Plus(
            // sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
            // rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
            // comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
            // );

            mpi_errno = MPIC_Sendrecv(
            sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
            rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
            comm_ptr, MPI_STATUS_IGNORE, errflag);
            
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
	    
        /* update index values to account for data we just received */
        send_index = recv_index;
        recv_index = (size + recv_index - p) % size;
    }//End for

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 11 c=%d\n",init_rank,sendcount);
       
    if(security_approach == 2005 && overlap_decryption == 0){
        recv_index = (size + rank_index - p) % size;
            // Decrypt recently received data
        for (i=1; i < n; ++i){
            // Mohsen:
            // int recv_rank = comm_ptr->comm_rank_list[recv_index];
            int recv_rank = comm_ptr->comm_rank_list_back[recv_index];
            MPID_Get_node_id(comm_ptr, recv_rank, &left_node_id);

            unsigned long count=0;
            
            if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, (recvbuf+recv_rank*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))), 12,
                        (ciphertext_recvbuf+(left_node_id*(sendcount*sendtype_extent+16+12))+12), (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                printf("Error in Naive+ decryption: allgather MultiLeader\n");
                fflush(stdout);        
            }
            recv_index = (size + recv_index - p) % size;
        }//end for
    }//end if

    //printf("%d finished inter-node\n", rank);
    /* Intra-node Ring */
    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 12 c=%d\n",init_rank,sendcount);

    right_index = (rank_index + 1);
    if(right_index >= size){
	    right_index -= p;
    }else{
        MPID_Get_node_id(comm_ptr, comm_ptr->comm_rank_list[right_index], &right_node_id);
        if(right_node_id != node_id){
            right_index -= p;
        }
    }
    right = comm_ptr->comm_rank_list[right_index];
    
    left_index = (rank_index - 1);
    if(left_index<0){
	    left_index += p;
    }else{
        MPID_Get_node_id(comm_ptr, comm_ptr->comm_rank_list[left_index], &left_node_id);
        if(left_node_id != node_id){
            (left_index += p);
        }
    }
    left = comm_ptr->comm_rank_list[left_index];

    send_index = rank_index;
    recv_index = left_index;

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 13 c=%d\n",init_rank,sendcount);
    
    for (i=1; i < p; ++i){
        for(j=0; j<n; ++j){

        // int send_rank = comm_ptr->comm_rank_list[(size + send_index - j*p) % size];
        // int recv_rank = comm_ptr->comm_rank_list[(size + recv_index - j*p) % size];

        int send_rank = comm_ptr->comm_rank_list_back[(size + send_index - j*p) % size];
        int recv_rank = comm_ptr->comm_rank_list_back[(size + recv_index - j*p) % size];

        void* sbuf = (void*)((char*) recvbuf + send_rank * recvcount * recvtype_extent);
        void* rbuf = (void*)((char*) recvbuf + recv_rank * recvcount * recvtype_extent);
    
        /* exchange data with our neighbors in the ring */
        // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, send, recvcount, recvtype); 
        // MPIR_PVAR_INC(allgather, 2lvl_multileader_ring, recv, recvcount, recvtype); 
        
        
        // mpi_errno = MPIC_Sendrecv_Plus(
        // sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
        // rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
        // comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[send_req_idx++]), errflag
        // );

        mpi_errno = MPIC_Sendrecv(
        sbuf, recvcount, recvtype, right, MPIR_ALLGATHER_TAG,
        rbuf, recvcount, recvtype, left,  MPIR_ALLGATHER_TAG,
        comm_ptr, MPI_STATUS_IGNORE, errflag);

            
        
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }//end for j
	    send_index = recv_index;
	    recv_index = recv_index - 1;
	    if(recv_index<0){
		    recv_index += p;
	    }else{
            MPID_Get_node_id(comm_ptr, comm_ptr->comm_rank_list[recv_index], &left_node_id);
            if(left_node_id != node_id){
                (recv_index += p);
            }
	    }
    }//End for i

    // if (DEBUG_INIT_FILE)fprintf(stderr,"[%d] M-Ring 14 c=%d\n",init_rank,sendcount);
    
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
#define FUNCNAME MPIR_Allgather_Ring_SEC
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_Ring_SEC(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_Allgather_Ring_SEC\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    int left, right, jnext;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_Ring_SEC\n");
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    /* First, load the "local" version in the recvbuf. */
    
	if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent), recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
	

    /*
     * Now, send left to right.  This fills in the receive area in
     * reverse order.
     */
    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    /******************** Added by Mehran ********************/
    int node_id, send_node_id, recv_node_id;
    int node_gateway=0, send_idx=0;
    if(security_approach==2005){
        
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        MPID_Get_node_id(comm_ptr, right, &send_node_id);
        MPID_Get_node_id(comm_ptr, left, &recv_node_id);
        if(node_id != send_node_id && node_id != recv_node_id){
            node_gateway=3;
        }else if(node_id != send_node_id){
            node_gateway=1;
        }else if(node_id != recv_node_id){
            node_gateway=2;
        }
        
    }
    /**********************************************************/

    j = rank;
    jnext = left;
    for (i = 1; i < comm_size; i++) {
        if(security_approach==2005){//Naive PLus (Default)
            unsigned long  ciphertext_len = 0, count=0, in_size=0;
            in_size = (unsigned long)(recvcount * recvtype_extent);
            unsigned long max_out_len = (unsigned long) (16 + in_size);
            char *in, *out, *sbuf, *rbuf;
            switch(node_gateway){
                case 1:
		    //printf("Case 1 in naive+ ring (default)\n");
                    //Encrypt before sending
                    in = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
                    out = (char*)((char*) large_recv_buffer + send_idx * (recvcount * recvtype_extent + 12 + 16));
                    RAND_bytes(out, 12); // 12 bytes of nonce

                    if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                        &ciphertext_len, max_out_len,
                                        out, 12,
                                        in, in_size,
                                        NULL, 0))
                    {
                        printf("Error in Naive+ encryption: allgather ring (Default)\n");
                        fflush(stdout);
                    }
                    
                    mpi_errno = MPIC_Sendrecv(out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    recvbuf + jnext * recvcount * recvtype_extent,
                                    recvcount, recvtype, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, errflag);
                    

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    recvbuf + jnext * recvcount * recvtype_extent,
                                    recvcount, recvtype, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/
                    send_idx = (send_idx+1)%2;
                    break;
                case 2:
                    //Decrypt after receiving
                    //printf("Case 2 in naive+ ring (default)\n");
                    sbuf = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
                    rbuf = (char*)((char*) large_recv_buffer + send_idx * (recvcount * recvtype_extent + 12 + 16));

                    mpi_errno = MPIC_Sendrecv(sbuf, recvcount* recvtype_extent,
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount * recvtype_extent + 16+12,
                                    MPI_CHAR, left, MPIR_ALLGATHER_TAG,
                                    comm_ptr, MPI_STATUS_IGNORE, errflag);

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(sbuf, recvcount,
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount * recvtype_extent + 16+12,
                                    MPI_CHAR, left, MPIR_ALLGATHER_TAG,
                                    comm_ptr, MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/


                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, (recvbuf+jnext*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        rbuf, 12, rbuf+12, (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
			                printf("Error in Naive+ decryption: allgather ring (default) I\n");
                            fflush(stdout);        
                        }
                    send_idx = (send_idx+1)%2;
                    break;
		case 3:
                    //Encrypt before sending and Decrypt after receiving
                    
                    //printf("Case 3 in naive+ ring (default)\n");
		    in = (char*)((char*) recvbuf + j * recvcount * recvtype_extent);
		    out = (char*)((char*) large_recv_buffer + send_idx * (recvcount * recvtype_extent + 12 + 16));
		    
		    rbuf = (char*)((char*) large_recv_buffer + (send_idx+1)%2 * (recvcount * recvtype_extent + 12 + 16));
		    
		    RAND_bytes(out, 12); // 12 bytes of nonce
                    
                    if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, out+12,
                                        &ciphertext_len, max_out_len,
                                        out, 12,
                                        in, in_size,
                                        NULL, 0))
                    {
                        printf("Error in Naive+ encryption: allgather ring (Default)\n");
                        fflush(stdout);
                    }
                    
                    mpi_errno = MPIC_Sendrecv((char *) out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount*recvtype_extent+16+12, MPI_CHAR, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, errflag);

                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus((char *) out, recvcount*recvtype_extent+16+12, 
                                    MPI_CHAR, right, MPIR_ALLGATHER_TAG,
                                    rbuf, recvcount*recvtype_extent+16+12, MPI_CHAR, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr,
                                    MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/

		    //printf("%d is going to decrypt from %d to %d\n", rank, jnext * (recvcount * recvtype_extent + 12 + 16), jnext*recvcount*recvtype_extent);
                    if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, (recvbuf+jnext*recvcount*recvtype_extent),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        rbuf, 12, rbuf+12, (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                            printf("Error in Naive+ decryption: allgather ring (default) II \n");
                            fflush(stdout);        
                        }

                        send_idx = (send_idx+1)%2;


                    break;
                default:
                    //No encryption/decryption
		    //printf("Default case in naive+ ring (default)\n");            
                    mpi_errno = MPIC_Sendrecv(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
                    /************* For MPIC_Sendrecv_Plus ************
                    mpi_errno = MPIC_Sendrecv_Plus(((char *) recvbuf +
                                        j * recvcount * recvtype_extent),
                                            recvcount, recvtype, right,
                                     MPIR_ALLGATHER_TAG,
                                     ((char *) recvbuf +
                                      jnext * recvcount * recvtype_extent),
                                     recvcount, recvtype, left,
                                     MPIR_ALLGATHER_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, &(send_req_ptr[i-1]), errflag);
                    **************************************************/

                    break;

            }
            
        }else{//Not Naive+
            mpi_errno = MPIC_Sendrecv(((char *) recvbuf +
                                    j * recvcount * recvtype_extent),
                                    recvcount, recvtype, right,
                                    MPIR_ALLGATHER_TAG,
                                    ((char *) recvbuf +
                                    jnext * recvcount * recvtype_extent),
                                    recvcount, recvtype, left,
                                    MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
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
#define FUNCNAME MPIR_Naive_Sec_Allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Naive_Sec_Allgather(const void *sendbuf,
                              int sendcount,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLGATHER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Allgather rank = %d host = %s count = %d  Sec-AP = %d] Func: MPIR_Naive_Sec_Allgather (Naive)\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif  	

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int sendtype_sz, recvtype_sz;
    unsigned long  ciphertext_sendbuf_len = 0;
    sendtype_sz= recvtype_sz= 0;
    int var;
    var=MPI_Type_size(sendtype, &sendtype_sz);
    var=MPI_Type_size(recvtype, &recvtype_sz);

    int rank;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Naive_Sec_Allgather\n");
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    // unsigned char *ciphertext_recvbuf =large_recv_buffer;

    RAND_bytes(ciphertext_sendbuf, 12); // 12 bytes of nonce

    unsigned long t=0;
    t = (unsigned long)(sendtype_sz*sendcount);
    unsigned long   max_out_len = (unsigned long) (16 + (sendtype_sz*sendcount));

    if(!EVP_AEAD_CTX_seal(global_coll_msg_ctx, ciphertext_sendbuf+12,
                         &ciphertext_sendbuf_len, max_out_len,
                         ciphertext_sendbuf, 12,
                         sendbuf,  t,
                        NULL, 0))
    {
              printf("Error in encryption: allgather\n");
              fflush(stdout);
    }

    if (comm_ptr->local_size == comm_ptr->pof2){
        mpi_errno = MPIR_Allgather_intra_recursive_doubling(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR,
                                                            ciphertext_recvbuf, ((recvcount*recvtype_sz) + 16+12), MPI_CHAR,
                                                            comm_ptr, errflag);
    } else if (comm_ptr->rank == 0) {
        mpi_errno = MPIR_Allgather_intra_ring(MPI_IN_PLACE, ciphertext_sendbuf_len+12, MPI_CHAR,
                                            ciphertext_recvbuf, ((recvcount*recvtype_sz) + 16+12), MPI_CHAR,
                                            comm_ptr, errflag);
    } else {
        mpi_errno = MPIR_Allgather_intra_ring(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR,
                                            ciphertext_recvbuf, ((recvcount*recvtype_sz) + 16+12), MPI_CHAR,
                                            comm_ptr, errflag);
    }     

    unsigned long count=0;
    unsigned long next, dest;
    unsigned int i;
    for( i = 0; i < comm_ptr->local_size; i++){
        next =(unsigned long )(i*((recvcount*recvtype_sz) + 16+12));
        dest =(unsigned long )(i*(recvcount*recvtype_sz));
        

        if(!EVP_AEAD_CTX_open(global_coll_msg_ctx, ((recvbuf+dest)),
                        &count, (unsigned long )((recvcount*recvtype_sz)+16),
                         (ciphertext_recvbuf+next), 12,
                        (ciphertext_recvbuf+next+12), (unsigned long )((recvcount*recvtype_sz)+16),
                        NULL, 0)){
                    printf("Decryption error: allgather\n");fflush(stdout);        
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



/* Algorithm: Ring
 *
 * In the first step, each process i sends its contribution to process
 * i+1 and receives the contribution from process i-1 (with
 * wrap-around).  From the second step onwards, each process i
 * forwards to process i+1 the data it received from process i-1 in
 * the previous step.  This takes a total of p-1 steps.
 *
 * Cost = (p-1).alpha + n.((p-1)/p).beta
 *
 * This algorithm is preferred to recursive doubling for long messages
 * because we find that this communication pattern (nearest neighbor)
 * performs twice as fast as recursive doubling for long messages (on
 * Myrinet and IBM SP).
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Allgather_intra_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allgather_intra_ring(const void *sendbuf,
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
		printf("[Allgather rank = %d host = %s count = %d] Func: MPIR_Allgather_intra_ring\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif  	
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_extent;
    int j, i;
    int left, right, jnext;

    if (((sendcount == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount == 0))
        return MPI_SUCCESS;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
    // if(rank==0) printf("MPIR_Allgather_intra_ring\n");
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_size * recvcount * recvtype_extent));

    /* First, load the "local" version in the recvbuf. */
    // if (init_rank != 0 || security_approach != 2005 || security_approach != 2006 || security_approach != 2007){
    // if (!(init_rank == 0 && (security_approach == 2005 || security_approach == 2006 || security_approach == 2007))){
	if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                   ((char *) recvbuf +
                                    rank * recvcount * recvtype_extent), recvcount, recvtype);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
	//}

    /*
     * Now, send left to right.  This fills in the receive area in
     * reverse order.
     */
    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    j = rank;
    jnext = left;
    for (i = 1; i < comm_size; i++) {
        mpi_errno = MPIC_Sendrecv(((char *) recvbuf +
                                   j * recvcount * recvtype_extent),
                                  recvcount, recvtype, right,
                                  MPIR_ALLGATHER_TAG,
                                  ((char *) recvbuf +
                                   jnext * recvcount * recvtype_extent),
                                  recvcount, recvtype, left,
                                  MPIR_ALLGATHER_TAG, comm_ptr, MPI_STATUS_IGNORE, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag =
                MPIX_ERR_PROC_FAILED ==
                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
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