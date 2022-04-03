/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"


#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_ML_Shmem_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_ML_Shmem_MV2(void *buffer, int count, MPI_Datatype datatype, int root,
                         MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
        printf("[Bcast rank = %d host = %s count = %d  Sec-AP = %d] Func: MPIR_Bcast_ML_Shmem_MV2\n", comm_ptr->rank,hostname,count,security_approach);fflush(stdout);
	}
#endif 	
// if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 01 \n",init_rank,count);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    unsigned long nbytes, scatter_size;
    MPI_Aint type_size;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (comm_size == 1)
        goto fn_exit;

    
    
    MPI_Comm conc_comm;
    MPIR_Comm *shmem_commptr, *conc_commptr = NULL;

    int local_size, local_rank, conc_rank, conc_size;
    MPIR_Datatype_get_extent_macro(datatype, type_size);    

    shmem_commptr = comm_ptr->node_comm;    

    conc_comm = comm_ptr->concurrent_comm;    
    

    if (conc_comm == NULL) {
        init_phase==1;
        mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
        init_phase==0;
        conc_comm = comm_ptr->concurrent_comm;    
    } 
    
    MPIR_Comm_get_ptr(conc_comm, conc_commptr);  

    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;

    /*if (comm_ptr->concurrent_comm == NULL){
        init_phase = 1;
        if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 01-6-1 \n",init_rank,count);
        mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
        if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 01-6-2 \n",init_rank,count);
        init_phase = 0;
    }*/
            
    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    rank = comm_ptr->rank;
    comm_size = comm_ptr->local_size;

    // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 01-8 \n",init_rank,count);

    nbytes = (unsigned long) (count) * (type_size);
    scatter_size = (nbytes + local_size - 1) / local_size; 
// if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 02 \n",init_rank,count);
    
    if(rank == root){
        /*Copy plaintext to the shared memory  buffer*/
        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 03-1 \n",init_rank,count);
        mpi_errno = MPIR_Localcopy((void*)((char*)buffer), count, datatype, 
                                    (void*)((char*)shmem_buffer), count, datatype);
        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 03-2 \n",init_rank,count);
        
        mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
        
        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 03-3 \n",init_rank,count);
        if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        if(security_approach == 333 ){ 
            /*Encrypts (m/l) to SHM cipher*/

                unsigned long ciphertext_len = 0;
                void* out;
                void* in;
                unsigned long in_size;
                if (scatter_size < 16){
                    in_size = scatter_size;
                }else{
                    in_size = (unsigned long)(scatter_size); 
                }
                //out = (void*)( large_send_buffer);
                out = (void*)( ciphertext_shmem_buffer);
                RAND_bytes(out, 12); // 12 bytes of nonce
                in = (void*)(shmem_buffer);
                unsigned long max_out_len = (16 + in_size);
            
                
            if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                    &ciphertext_len, max_out_len,
                                    out, 12,
                                    in, in_size,
                                    NULL, 0))
                {
                        printf("Error in  encryption: SHM-ML-2  \n");
                        fflush(stdout);
                }    
    
            /*Concurrent Bcast*/
    
           // mpi_errno = MPIR_Bcast_impl(ciphertext_shmem_buffer, (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);
           // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 03-4 \n",init_rank,count);
           mpi_errno = MPIR_Bcast_intra_binomial(ciphertext_shmem_buffer, (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);
           // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04 \n",init_rank,count);

        }
        else{    

            // mpi_errno = MPIR_Bcast_impl(shmem_buffer, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);
            // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-1 \n",init_rank,count);
            mpi_errno = MPIR_Bcast_intra_binomial(shmem_buffer, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);
            // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-2 \n",init_rank,count);
        }

    }//end if root
    else{
        //compare node id with that of the root
        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-2-1 \n",init_rank,count);
        int node_id, root_node_id;
        MPID_Get_node_id(comm_ptr, rank, &node_id);
        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-2-2 \n",init_rank,count);
        MPID_Get_node_id(comm_ptr, root, &root_node_id);

        // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-3 \n",init_rank,count);


        if(node_id == root_node_id){

            // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-4 \n",init_rank,count);
            
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);  /*Wait for  Copy*/

            // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-4-1 \n",init_rank,count);

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 04-5 \n",init_rank,count);
            
            if (security_approach == 333 ){
                /*Encrypts (m/l) to shared cipher buffer*/
                unsigned long ciphertext_len = 0;
                void* out;
                void* in;
                unsigned long in_size;
                if (scatter_size < 16){
                    in_size = scatter_size;
                }else{
                    in_size = (unsigned long)(scatter_size); 
                }
                //out = (void*)( large_send_buffer);
                out = (void*)( ciphertext_shmem_buffer+local_rank*(scatter_size+28));
                RAND_bytes(out, 12); //nonce
                in = (void*)(shmem_buffer +local_rank*scatter_size);
                unsigned long max_out_len = (16 + in_size);
            
                
                if(!EVP_AEAD_CTX_seal(ctx, out+12,
                                    &ciphertext_len, max_out_len,
                                    out, 12,
                                    in, in_size,
                                    NULL, 0))
                {
                        printf("Error in  encryption: SHM-ML-1  \n");
                        fflush(stdout);
                }    


                //mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag); 

                /*Concurrent Bcast*/
        
                //mpi_errno = MPIR_Bcast_impl(large_send_buffer, scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                // mpi_errno = MPIR_Bcast_impl(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 05 \n",init_rank,count);
                mpi_errno = MPIR_Bcast_intra_binomial(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), (scatter_size+28), MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 06 \n",init_rank,count);

                /*Local copy */
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);

            }
            else{
                //Unecrypted 
               /* mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE, 
                                    (void*)((char*)ciphertext_shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE);*/
                // mpi_errno = MPIR_Bcast_impl(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 07 \n",init_rank,count);
                mpi_errno = MPIR_Bcast_intra_binomial(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 08 \n",init_rank,count);

                /*Local copy */ 
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);
            }

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

        }//end if node_id
        else{
            
            if (security_approach == 333){ 
                //mpi_errno = MPIR_Bcast_impl(large_send_buffer, scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                // mpi_errno = MPIR_Bcast_impl(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 09 \n",init_rank,count);
                mpi_errno = MPIR_Bcast_intra_binomial(( ciphertext_shmem_buffer+local_rank*(scatter_size+28)), scatter_size+28, MPI_BYTE, 0, conc_commptr, errflag);
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 10 \n",init_rank,count);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                /*Decrypt to the buffer*/
                void* out;
                void* in;
                unsigned long  decrypted_len;
                unsigned long  ciphertext_len = (scatter_size+16);

                //in = (void*)(large_send_buffer);
                in = (void*)(ciphertext_shmem_buffer+local_rank*(scatter_size+28));
                out = (void*)(shmem_buffer +local_rank*scatter_size);
                /*printf("dec starts, size=%d, rank=%d\n", ciphertext_len, rank);
                fflush(stdout);*/

                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 10-1 \n",init_rank,count);
                if(!EVP_AEAD_CTX_open(ctx, out, &decrypted_len, (ciphertext_len-16),
                        in, 12, in+12, (unsigned long )(ciphertext_len),
                        NULL, 0)){
                        printf("Error in SHM-ML-1 decryption:  while %d tried to decrypt\n", rank);
                        fflush(stdout);   
                    }
                // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 10-2 \n",init_rank,count);
                //printf(" dec is done\n");
                
                mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag); /*Wait for decryption*/

                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                    (void*)((char*)buffer), count, datatype);

            }else{
                 // mpi_errno = MPIR_Bcast_impl(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag); 
                 // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 11 \n",init_rank,count);
                 mpi_errno = MPIR_Bcast_intra_binomial(shmem_buffer+local_rank*scatter_size, scatter_size, MPI_BYTE, 0, conc_commptr, errflag); 
                 // if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 12 \n",init_rank,count);
                 /*mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
                 mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
                                         (void*)((char*)buffer), count, datatype);*/
                mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+local_rank*scatter_size), scatter_size, MPI_BYTE, 
                                         (void*)((char*)buffer+local_rank*scatter_size), scatter_size, MPI_BYTE);
                if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                int s=0;
                mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

                for (;s<local_size;s++){
                    if(s!=local_rank){
                        mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer+s*scatter_size), scatter_size, MPI_BYTE, 
                                         (void*)((char*)buffer+s*scatter_size), scatter_size, MPI_BYTE);
                    }
                }
                
            }

        }//end of other nodes
       
    }//end all ranks
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag =
            MPIX_ERR_PROC_FAILED ==
            MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

// if (DEBUG_INIT_FILE)  printf("[%d] Bcast_ML_Shmem C=%d 13 \n",init_rank,count);
  fn_exit:
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
}






#undef FUNCNAME
#define FUNCNAME MPI_Naive_Sec_Bcast
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPI_Naive_Sec_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                         MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
        printf("[Bcast rank = %d host = %s count = %d  Sec-AP = %d] Func: MPI_Naive_Sec_Bcast\n", comm_ptr->rank,hostname,count,security_approach);fflush(stdout);
	}
#endif 	

    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    unsigned long ciphertext_len = 0;
    int  sendtype_sz=0;           
    
    MPIR_Datatype_get_extent_macro(datatype, sendtype_sz);  
    // MPI_Type_size(datatype, &sendtype_sz);         
    unsigned long   max_out_len = (unsigned long)(16 + (sendtype_sz*count));
    unsigned long decrypted_len=0;
		
	int rank = comm_ptr->rank;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    if (rank == root) {
        
        /* Set the nonce in send_ciphertext */
        RAND_bytes(ciphertext_sendbuf, 12); 

		if(!EVP_AEAD_CTX_seal(ctx, (ciphertext_sendbuf+12),
                            &ciphertext_len, max_out_len,
                            ciphertext_sendbuf, 12, 
                            buffer, (unsigned long)(count*sendtype_sz),
                             NULL, 0)){  
                    printf("Error in encryption: MPI_SEC_Bcast rank=0\n");		
                    fflush(stdout);
            }
            
            #if ENABLE_SECURE_MPI_DEBUG
            else{
                MPID_Comm *debug_comm_ptr = NULL;
                MPID_Comm_get_ptr( comm, debug_comm_ptr );
                printf("MPI_SEC_Bcast: ciphertext_len = %d Rank = %d\n",ciphertext_len, debug_comm_ptr->rank);
                fflush(stdout);
            }
            #endif              
		
		init_phase=1;
        mpi_errno = MPIR_Bcast_impl(ciphertext_sendbuf, ((sendtype_sz*count)+16+12), MPI_CHAR, root, comm_ptr, errflag);
        init_phase=0;
		
	}
	else if (rank != root) {	
	
		ciphertext_len = (unsigned long)((count*sendtype_sz)+16);
		
		init_phase=1;
        mpi_errno = MPIR_Bcast_impl(ciphertext_recvbuf, ((sendtype_sz*count)+16+12), MPI_CHAR, root, comm_ptr, errflag);
        init_phase=0;
		
		if(!EVP_AEAD_CTX_open(ctx, buffer, 
                            &decrypted_len, (ciphertext_len-16), 
                            ciphertext_recvbuf, 12, 
                            (ciphertext_recvbuf+12), ciphertext_len, 
                            NULL, 0)){
                printf("Decryption error: MPI_SEC_Bcast rank=%d\n",rank);	
                fflush(stdout);
            }
            
            #if ENABLE_SECURE_MPI_DEBUG
            else{
                MPID_Comm *debug_comm_ptr = NULL;
                MPID_Comm_get_ptr( comm, debug_comm_ptr );
                printf("MPI_SEC_Bcast: decrypted_len = %d Rank = %d\n",decrypted_len, debug_comm_ptr->rank);
                fflush(stdout);
            }
            #endif                		
		
	}


  fn_exit:
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
}





/* FIXME This function uses some heuristsics based off of some testing on a
 * cluster at Argonne.  We need a better system for detrmining and controlling
 * the cutoff points for these algorithms.  If I've done this right, you should
 * be able to make changes along these lines almost exclusively in this function
 * and some new functions. [goodell@ 2008/01/07] */
#undef FUNCNAME
#define FUNCNAME MPIR_Bcast_intra_smp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Bcast_intra_smp(void *buffer, int count, MPI_Datatype datatype, int root,
                         MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if BCAST_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		// printf("[Bcast [%d] [%d]  host = %s count = %d] Func: MPIR_Bcast_intra_smp\n", comm_ptr->rank,init_rank,hostname,count);fflush(stdout);
        printf("[Bcast rank = %d host = %s count = %d  Sec-AP = %d] Func: MPIR_Bcast_intra_smp\n", comm_ptr->rank,hostname,count,security_approach);fflush(stdout);
	}
#endif 	
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint type_size, nbytes = 0;
    MPI_Status status;
    MPI_Aint recvd_size;

#ifdef HAVE_ERROR_CHECKING
    if (!MPIR_CVAR_ENABLE_SMP_COLLECTIVES || !MPIR_CVAR_ENABLE_SMP_BCAST) {
        MPIR_Assert(0);
    }
    MPIR_Assert(MPIR_Comm_is_node_aware(comm_ptr));
#endif

    MPIR_Datatype_get_size_macro(datatype, type_size);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;   /* nothing to do */

    if ((nbytes < MPIR_CVAR_BCAST_SHORT_MSG_SIZE) ||
        (comm_ptr->local_size < MPIR_CVAR_BCAST_MIN_PROCS)) {
        /* send to intranode-rank 0 on the root's node */
        if (comm_ptr->node_comm != NULL && MPIR_Get_intranode_rank(comm_ptr, root) > 0) {       /* is not the node root (0) and is on our node (!-1) */
            if (root == comm_ptr->rank) {
                mpi_errno = MPIC_Send(buffer, count, datatype, 0,
                                      MPIR_BCAST_TAG, comm_ptr->node_comm, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            } else if (0 == comm_ptr->node_comm->rank) {
                mpi_errno =
                    MPIC_Recv(buffer, count, datatype, MPIR_Get_intranode_rank(comm_ptr, root),
                              MPIR_BCAST_TAG, comm_ptr->node_comm, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                /* check that we received as much as we expected */
                MPIR_Get_count_impl(&status, MPI_BYTE, &recvd_size);
                if (recvd_size != nbytes) {
                    if (*errflag == MPIR_ERR_NONE)
                        *errflag = MPIR_ERR_OTHER;
                    MPIR_ERR_SET2(mpi_errno, MPI_ERR_OTHER,
                                  "**collective_size_mismatch",
                                  "**collective_size_mismatch %d %d", recvd_size, nbytes);
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

        }

        /* perform the internode broadcast */
        if (comm_ptr->node_roots_comm != NULL) {
            mpi_errno = MPIR_Bcast(buffer, count, datatype,
                                   MPIR_Get_internode_rank(comm_ptr, root),
                                   comm_ptr->node_roots_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        /* perform the intranode broadcast on all except for the root's node */
        if (comm_ptr->node_comm != NULL) {
            mpi_errno = MPIR_Bcast(buffer, count, datatype, 0, comm_ptr->node_comm, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } else {    /* (nbytes > MPIR_CVAR_BCAST_SHORT_MSG_SIZE) && (comm_ptr->size >= MPIR_CVAR_BCAST_MIN_PROCS) */

        /* supposedly...
         * smp+doubling good for pof2
         * reg+ring better for non-pof2 */
        if (nbytes < MPIR_CVAR_BCAST_LONG_MSG_SIZE && MPL_is_pof2(comm_ptr->local_size, NULL)) {
            /* medium-sized msg and pof2 np */

            /* perform the intranode broadcast on the root's node */
            if (comm_ptr->node_comm != NULL && MPIR_Get_intranode_rank(comm_ptr, root) > 0) {   /* is not the node root (0) and is on our node (!-1) */
                /* FIXME binomial may not be the best algorithm for on-node
                 * bcast.  We need a more comprehensive system for selecting the
                 * right algorithms here. */
                mpi_errno = MPIR_Bcast(buffer, count, datatype,
                                       MPIR_Get_intranode_rank(comm_ptr, root),
                                       comm_ptr->node_comm, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

            /* perform the internode broadcast */
            if (comm_ptr->node_roots_comm != NULL) {
                mpi_errno = MPIR_Bcast(buffer, count, datatype,
                                       MPIR_Get_internode_rank(comm_ptr, root),
                                       comm_ptr->node_roots_comm, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }

            /* perform the intranode broadcast on all except for the root's node */
            if (comm_ptr->node_comm != NULL && MPIR_Get_intranode_rank(comm_ptr, root) <= 0) {  /* 0 if root was local root too, -1 if different node than root */
                /* FIXME binomial may not be the best algorithm for on-node
                 * bcast.  We need a more comprehensive system for selecting the
                 * right algorithms here. */
                mpi_errno = MPIR_Bcast(buffer, count, datatype, 0, comm_ptr->node_comm, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        } else {        /* large msg or non-pof2 */

            /* FIXME It would be good to have an SMP-aware version of this
             * algorithm that (at least approximately) minimized internode
             * communication. */
            mpi_errno =
                MPIR_Bcast_intra_scatter_ring_allgather(buffer, count, datatype, root, comm_ptr,
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
    }

  fn_exit:
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */
    return mpi_errno;
}
