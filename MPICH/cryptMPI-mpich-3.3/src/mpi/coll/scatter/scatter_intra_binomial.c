/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* This is the machine-independent implementation of scatter. The algorithm is:

   Algorithm: Binomial

   We use a binomial tree algorithm for both short and
   long messages. At nodes other than leaf nodes we need to allocate
   a temporary buffer to store the incoming message. If the root is
   not rank 0, we reorder the sendbuf in order of relative ranks by
   copying it into a temporary buffer, so that all the sends from the
   root are contiguous and in the right order.

   Cost = lgp.alpha + n.((p-1)/p).beta
   where n is the total size of the data to be scattered from the root.

   Possible improvements:

   End Algorithm: MPI_Scatter
*/





#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
    
if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif 
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }

#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct_no_shmem\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    // MPI_Aint extent = 0;
    int rank, comm_size; //, sendtype_size;
    //int relative_rank, nbytes;
    //MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(6);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-1 c=%d \n",init_rank,recvcount);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    MPI_Comm conc_comm;
    MPIR_Comm *shmem_commptr= NULL;
    int rank_node_id, root_node_id;
    
    int local_size, local_rank, conc_rank, conc_size;
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-2 c=%d \n",init_rank,recvcount);

    shmem_commptr = comm_ptr->node_comm;

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-3 c=%d \n",init_rank,recvcount);    
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;
    // int leader_size = local_size; 

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-5 c=%d \n",init_rank,recvcount);

    /* if (mpi_errno)
    {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
    if (mpi_errno)
    { 
        MPIR_ERR_POP(mpi_errno);
    } */
    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 01-6 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    // comm_size = comm_ptr->local_size;
    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray",MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);

    // if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 01 c=%d \n",init_rank,recvcount);

    reqs = 0;

    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);

    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        // MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        int next;
        for(i=1; i<local_size;i++)
        {
             next = i*each_rank_size*t;
             mpi_errno = MPIC_Isend((sendbuf + next),
                                           (t *each_rank_size),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
            //printf("**[intra-node] rank %d sending to rank %d\n",rank,i);fflush(stdout);
        }
                               
        //printf("root=%d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        // MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 01-1 c=%d \n",init_rank,recvcount);
        
        mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);   
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 01-2 c=%d \n",init_rank,recvcount);       
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 02 c=%d \n",init_rank,recvcount);

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {
            // MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIR_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                             sendtype_extent);

            /* divide encryption among ppn ranks */
            int each_rank_size = comm_size / leader_size; //local_size;
            int max_dest_ranks;

            if (local_rank == (leader_size - 1))
            {
                max_dest_ranks = comm_size;
            }
            else
            {
                max_dest_ranks = local_rank * each_rank_size + each_rank_size;
            }

            int next = 0;
            int dest = 0;
            int t = (sendtype_extent * sendcnt);
            unsigned long max_out_len = (unsigned long)(16 + (sendtype_extent * sendcnt));
            unsigned long ciphertext_sendbuf_len;
            int dest_node_id;
           

            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
                MPID_Get_node_id(comm_ptr, root, &root_node_id);
                if (rank == i && rank == root)
                {
                    memcpy(recvbuf, sendbuf+next, t);
                    //printf("@@ copy to own rank %d == i %d\n", rank, i);
                }
                else if (rank_node_id == dest_node_id && rank == root) // sending to same node
                {
                    mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    //printf("sending to same node rank %d ----> i %d\n", rank, i);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else if (rank_node_id == dest_node_id && rank != root) // sending to same node by non root
                {
                    mpi_errno = MPIC_Isend((ciphertext_recvbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    //printf("NON-ROOT sending to same node rank %d ----> i %d\n", rank, i);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else
                {
                    if(rank == root)
                        temp_buffer = sendbuf;
                    else
                        temp_buffer = ciphertext_recvbuf;    

                    RAND_bytes(ciphertext_sendbuf + dest, 12);
                    if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                           &ciphertext_sendbuf_len, max_out_len,
                                           ciphertext_sendbuf + dest, 12,
                                           temp_buffer + next, t,
                                           NULL, 0))
                    {
                        printf("Error in encryption: [rank %d rank_node_id %d] scatter direct dest %d dest_node_id %d\n",
                               rank, rank_node_id, i, dest_node_id);
                        fflush(stdout);
                    }
#if 0
                    else
                    {

                        printf(COLOR_GREEN "Scatter shm direct:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest %d dest_node_id %d] Encrypted data size %lu max_dest_ranks %d" COLOR_RESET "\n",
                               rank, local_rank, rank_node_id, root_node_id, i, dest_node_id, ciphertext_sendbuf_len, max_dest_ranks);
                        fflush(stdout);
                    }
#endif
                    //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
                    mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                           (t + 28),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag); 

                    dest += t + 16 + 12; // update after send
                   
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                 next += t; // get the encrytion location
            } // i
        }
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 03 c=%d \n",init_rank,recvcount);

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 03 t=%d c=%d \n",init_rank,t,recvcount);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int new_root = 0;
        //if(new_root >= local_size)
        //    new_root = local_size - 1;
        int each_rank_size = comm_size / leader_size;
        new_root = rank / each_rank_size;          
        int new_root_node_id;
        MPID_Get_node_id(comm_ptr, new_root, &new_root_node_id);

        /*if (rank == new_root)
        {
            printf("rank %d received from new_root\n", rank, new_root);
            fflush(stdout);
        }
        else if (rank_node_id == new_root_node_id)*/
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
        
        if (rank_node_id == new_root_node_id && rank != new_root)
        {
            //printf(COLOR_CYAN"[same node]rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Recv(recvbuf, t, MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }                      
            //printf("rank %d received from same node new_root %d\n",rank, new_root);fflush(stdout);                      
        }
        else if(rank_node_id != new_root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);

            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 03-3 c=%d \n",init_rank,recvcount);
            //if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 04 t=%d c=%d \n",init_rank,t,recvcount);
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 03-4 c=%d \n",init_rank,recvcount);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            //printf("[inter-node]rank %d received data from new_root %d\n", rank, new_root);
            if (!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf), 12,
                                   (ciphertext_recvbuf + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                fflush(stdout);
            }
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 03-5 c=%d \n",init_rank,recvcount);
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter shm recv:: [rank %d rank_node_id %d new_root %d] Decrypted data size %lu" COLOR_RESET "\n",
                       rank, rank_node_id, new_root, count);
                fflush(stdout);
            }
#endif
        }
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 04 c=%d \n",init_rank,recvcount);

    /* wait for finish sending */
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
         /* ... then wait for *all* of them to finish: */
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 04-1 c=%d \n",init_rank,recvcount);
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS)
                mpi_errno = starray[i].MPI_ERROR;
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            }
        }
        /* --END ERROR HANDLING-- */    
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-RR 05 c=%d \n",init_rank,recvcount);


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
#define FUNCNAME MPIR_Scatter_MV2_two_level_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_two_level_Direct(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);

if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif    
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }

#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_two_level_Direct\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    int comm_size, rank;
    int local_rank, local_size;
    int leader_comm_rank, leader_rank, leader_comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_size, sendtype_size, nbytes;
    void *tmp_buf = NULL;
    void *leader_scatter_buf = NULL;
    MPI_Status status;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_CHKLMEM_DECL(8);
    // MPIU_THREADPRIV_DECL;
    // MPIU_THREADPRIV_GET;
    int leader_root, leader_of_root = -1;
    MPI_Comm shmem_comm, leader_comm;
    MPIR_Comm *shmem_commptr, *leader_commptr = NULL;

    int sendcnt = sendcount;
    int recvcnt = recvcount;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcnt == 0))
        || ((rank != root) && (recvcnt == 0))) {

        // MPIR_TIMER_END(coll,scatter,two_level_direct);
        return MPI_SUCCESS;
    }

    shmem_commptr = comm_ptr->node_comm;
    //MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;    
    int p = local_size;
    int n = node_cnt;
    int workload = (int) (p / shmem_leaders);
    
    // conc_comm = comm_ptr->concurrent_comm;    
    // MPIR_Comm_get_ptr(conc_comm, conc_commptr);
    if (local_rank == 0) {
        // Node leader. Extract the rank, size information for the leader communicator        
        leader_commptr = comm_ptr->node_roots_comm;
        leader_rank = comm_ptr->node_roots_comm->rank;
        leader_comm_size = comm_ptr->node_roots_comm->local_size;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] [%d] ALG-Conc 02-1\n",init_rank,recvcount);    
    }

    if (local_size == comm_size) {
        /* purely intra-node scatter. Just use the direct algorithm and we are done */
        mpi_errno = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype,
                                            recvbuf, recvcnt, recvtype,
                                            root, comm_ptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Datatype_get_size_macro(sendtype, sendtype_size);

        if (rank == root) {
            nbytes = sendcnt * sendtype_size;
        } else {
            nbytes = recvcnt * recvtype_size;
        }

        if (local_rank == 0) {
            /* Node leader, allocate tmp_buffer */
            // tmp_buf = MPIU_Malloc(nbytes * local_size);
            tmp_buf = MPL_malloc(nbytes * local_size, MPL_MEM_USER );
            
        }

        // leader_of_root = comm_ptr->dev.ch.leader_map[root];
        /* leader_of_root is the global rank of the leader of the root */
        // leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
        /* leader_root is the rank of the leader of the root in leader_comm.
         * leader_root is to be used as the root of the inter-leader gather ops
         */

        if (root == 0){
            leader_of_root = 0;
            leader_root = 0;
        }
        
        /* MPID_Node_id_t rank_node_id, root_node_id, leader_of_root_node_id,leader_root_node_id,local_rank_node_id; 
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
        MPID_Get_node_id(comm_ptr, root, &root_node_id);
        MPID_Get_node_id(comm_ptr, leader_of_root, &leader_of_root_node_id);
        MPID_Get_node_id(comm_ptr, leader_root, &leader_root_node_id);
        MPID_Get_node_id(comm_ptr, local_rank, &local_rank_node_id);
        printf("[rank %d node_id %u] root %d[%u] local_rank %d[%u] leader_of_root %d[%u] leader_root %d[%u]\n",
        rank,rank_node_id,root,root_node_id, local_rank,local_rank_node_id,leader_of_root,leader_of_root_node_id,
        leader_root,leader_root_node_id); fflush(stdout);*/
        
        if ((local_rank == 0) && (root != rank)
            && (leader_of_root == rank)) {
            /* The root of the scatter operation is not the node leader. Recv
             * data from the node leader */
            //printf("[rank %d] root %d local_rank %d leader_of_root %d leader_root %d comm_size =%d\n", 
            //rank,root, local_rank,leader_of_root, leader_root,comm_size); fflush(stdout);
            /* MPID_Node_id_t rank_node_id;
             MPID_Get_node_id(comm_ptr, root, &rank_node_id);
            printf("[rank %d ] root %d[rank_node_id %u] local_rank %d leader_of_root %d comm_size =%d\n", 
            rank,root,rank_node_id, local_rank,leader_of_root,comm_size); fflush(stdout);*/

            // leader_scatter_buf = MPIU_Malloc(nbytes * comm_size);
            leader_scatter_buf = MPL_malloc(nbytes * comm_size, MPL_MEM_USER );
            // MPIR_PVAR_INC(scatter, two_level_direct, recv, nbytes * comm_size, MPI_BYTE);
            mpi_errno =
                MPIC_Recv(leader_scatter_buf, nbytes * comm_size, MPI_BYTE,
                             root, MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        if (rank == root && local_rank != 0) {
            /* The root of the scatter operation is not the node leader. Send
             * data to the node leader */
            /* MPID_Node_id_t root_node_id;
             MPID_Get_node_id(comm_ptr, root, &root_node_id);
            printf("[rank %d ] root %d[root_node_id %u] local_rank %d leader_of_root %d comm_size =%d\n", 
            rank,root,root_node_id, local_rank,leader_of_root,comm_size); fflush(stdout);*/
            
            // MPIR_PVAR_INC(scatter, two_level_direct, send, sendcnt * comm_size, sendtype);
            mpi_errno = MPIC_Send(sendbuf, sendcnt * comm_size, sendtype,
                                     leader_of_root, MPIR_SCATTER_TAG, comm_ptr,
                                     errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }

        if (leader_comm_size > 1 && local_rank == 0) {
            // if (comm_ptr->dev.ch.is_uniform != 1) {
            if (comm_ptr->is_uniform == 0){
                int *displs = NULL;
                int *sendcnts = NULL;
                // int *node_sizes;
                int i = 0;
                // node_sizes = comm_ptr->dev.ch.node_sizes;

                if (root != leader_of_root) {
                    if (leader_comm_rank == leader_root) {
                        // displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        // sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        displs = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                        sendcnts = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                        sendcnts[0] = node_sizes[0] * nbytes;
                        displs[0] = 0;

                        for (i = 1; i < leader_comm_size; i++) {
                            displs[i] =
                                displs[i - 1] + node_sizes[i - 1] * nbytes;
                            sendcnts[i] = node_sizes[i] * nbytes;
                        }
                    }
                    /* added by abu naser naive+ */
                    if (security_approach == 200 || (security_approach > 200 && security_approach <=211)) //&& (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1 )))
                    {
                        mpi_errno =
                            MPIR_ENC_Scatterv(leader_scatter_buf, sendcnts, displs,
                                              MPI_BYTE, tmp_buf, nbytes * local_size,
                                              MPI_BYTE, leader_root, leader_commptr,
                                              errflag);
                    }
                    else
                    {
                        mpi_errno =
                            MPIR_Scatterv(leader_scatter_buf, sendcnts, displs,
                                          MPI_BYTE, tmp_buf, nbytes * local_size,
                                          MPI_BYTE, leader_root, leader_commptr,
                                          errflag);
                    }
                } else {
                    if (leader_comm_rank == leader_root) {
                        // displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        // sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        displs = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                        sendcnts = MPL_malloc(sizeof (int) * leader_comm_size, MPL_MEM_USER );
                        sendcnts[0] = node_sizes[0] * sendcnt;
                        displs[0] = 0;

                        for (i = 1; i < leader_comm_size; i++) {
                            displs[i] =
                                displs[i - 1] + node_sizes[i - 1] * sendcnt;
                            sendcnts[i] = node_sizes[i] * sendcnt;
                        }
                    }

                    /* added by abu naser naive+ */
                     if (security_approach == 200 || (security_approach > 200 && security_approach <=211)) //&& (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1)))
                    {
                        mpi_errno = MPIR_ENC_Scatterv(sendbuf, sendcnts, displs,
                                                      sendtype, tmp_buf,
                                                      nbytes * local_size, MPI_BYTE,
                                                      leader_root, leader_commptr,
                                                      errflag);
                    }
                    else
                    {
                        mpi_errno = MPIR_Scatterv(sendbuf, sendcnts, displs,
                                                  sendtype, tmp_buf,
                                                  nbytes * local_size, MPI_BYTE,
                                                  leader_root, leader_commptr,
                                                  errflag);
                    }
                }
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                if (leader_comm_rank == leader_root) {
                    //MPIU_Free(displs);
                    // MPIU_Free(sendcnts);
                }
            } else {
                if (leader_of_root != root) {

                    /* added by abu naser naive+ */                      
                     if (security_approach == 200 || (security_approach > 200 && security_approach <=211))// && (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1)))
                    {
                        int i = 0;
                        int next = 0;
                        int dest = 0;
                        int t= (nbytes * local_size);
                        unsigned long   max_out_len = (unsigned long) (16 + (nbytes * local_size));
                        unsigned long ciphertext_sendbuf_len;
                        //printf("leader_comm_size=%d local_size=%d\n",leader_comm_size,local_size);fflush(stdout);
                        if (rank == leader_of_root)
                        {
                            for (i = 0; i < leader_comm_size; i++)
                            {
                                RAND_bytes(ciphertext_sendbuf + dest, 12);
                                if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                                       &ciphertext_sendbuf_len, max_out_len,
                                                       ciphertext_sendbuf + dest, 12,
                                                       leader_scatter_buf + next, t,
                                                       NULL, 0))
                                {
                                    printf("Error in encryption: scatter two-level-direct\n");
                                    fflush(stdout);
                                }
#if 0
                                else
                                {
                                    printf("Scatter two-level-direct:: [leader_of_root %d] Encrypted data size %lu dest leader rank %d\n",
                                           leader_of_root, ciphertext_sendbuf_len, i);
                                    fflush(stdout);
                                }
#endif
                                next += t;
                                dest += t + 16 + 12;
                            }
                        }

                        mpi_errno =
                            MPIR_Scatter_MV2_Direct_unencrypted(ciphertext_sendbuf,
                                                    ((nbytes * local_size) + 28), MPI_BYTE,
                                                    ciphertext_recvbuf, ((nbytes * local_size) + 28),
                                                    MPI_BYTE, leader_root,
                                                    leader_commptr, errflag);
                        
                        max_out_len = (unsigned long) (nbytes * local_size);
                        unsigned long count; 
                        
                        if(!EVP_AEAD_CTX_open(global_ctx, tmp_buf,
                            &count, max_out_len,
                            (ciphertext_recvbuf), 12,
                            (ciphertext_recvbuf+12), (unsigned long )(t+16),
                            NULL, 0))
                        {                            
                            printf("[rank=%d] Decryption error: scatter two-level-direct (leader_of_root != root)\n",comm_ptr->rank);fflush(stdout);        
                        }
#if 0            
                        else
                        {
                            printf("Scatter two-level-direct:: [rank %d] Decrypted data size %lu\n",rank, count);fflush(stdout);
                        }                             
#endif                        
                    }/* end of add */
                    else                   
                    { // unencrypted
                        mpi_errno =
                            MPIR_Scatter_MV2_Direct(leader_scatter_buf,
                                                nbytes * local_size, MPI_BYTE,
                                                tmp_buf, nbytes * local_size,
                                                MPI_BYTE, leader_root,
                                                leader_commptr, errflag);
                    }
                } else {

                     /* added by abu naser naive+ */    
                     if (security_approach == 200 || (security_approach > 200 && security_approach <=211))// && (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1)))               
                    {
                        int i = 0;
                        int next = 0;
                        int dest = 0;
                        int t= (sendtype_size * sendcnt * local_size);
                        unsigned long   max_out_len = (unsigned long) (16 + (sendtype_size * sendcnt * local_size));
                        unsigned long ciphertext_sendbuf_len;
                        //printf("leader_comm_size=%d local_size=%d\n",leader_comm_size,local_size);fflush(stdout);
                     
                        if (rank == leader_of_root)
                        {
                            for (i = 0; i < leader_comm_size; i++)
                            {
                                RAND_bytes(ciphertext_sendbuf + dest, 12);
                                if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                                       &ciphertext_sendbuf_len, max_out_len,
                                                       ciphertext_sendbuf + dest, 12,
                                                       sendbuf + next, t,
                                                       NULL, 0))
                                {
                                    printf("Error in encryption: scatter two-level-direct, root is node leader\n");
                                    fflush(stdout);
                                }
#if 0
                                else
                                {
                                    printf(COLOR_GREEN"Scatter two-level-direct:: [leader_of_root %d] Encrypted data size %lu dest leader rank %d, [root is node leader] next %d dest %d"COLOR_RESET"\n",
                                           leader_of_root, ciphertext_sendbuf_len, i, next, dest); fflush(stdout);
                                    
                                }
#endif
                                next += t;
                                dest += t + 16 + 12;
                            }
                        }

                        mpi_errno =
                            MPIR_Scatter_MV2_Direct_unencrypted(ciphertext_sendbuf,
                                                    ((sendtype_size * sendcnt  * local_size) + 28), MPI_BYTE,
                                                    ciphertext_recvbuf, ((nbytes * local_size) + 28),
                                                    MPI_BYTE, leader_root,
                                                    leader_commptr, errflag);
                        
                        max_out_len = (unsigned long) (nbytes * local_size);
                        unsigned long count; 
                        t = (nbytes * local_size);
                        
                        if(!EVP_AEAD_CTX_open(global_ctx, tmp_buf,
                            &count, max_out_len,
                            (ciphertext_recvbuf), 12,
                            (ciphertext_recvbuf+12), (unsigned long )(t+16),
                            NULL, 0))
                        {
                            printf("[rank=%d] Decryption error: scatter two-level-direct (leader_of_root == root)\n",comm_ptr->rank);fflush(stdout);        
                        }
#if 0            
                        else
                        {
                            printf(COLOR_BLUE"Scatter two-level-direct:: [rank %d leader_root %d] Decrypted data size %lu"COLOR_RESET"\n",rank, leader_root, count);fflush(stdout);
                        }                             
#endif                        

                    }/* end of add */
                    else
                    { // unencrypted
                    
                    mpi_errno =
                        MPIR_Scatter_MV2_Direct(sendbuf, sendcnt * local_size,
                                                sendtype, tmp_buf,
                                                nbytes * local_size, MPI_BYTE,
                                                leader_root, leader_commptr,
                                                errflag);
                    }

                }
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
        /* The leaders are now done with the inter-leader part. Scatter the data within the nodes */
        

        if (rank == root && recvbuf == MPI_IN_PLACE) {
            // mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
            mpi_errno = MPIR_Scatter_MV2_Direct_unencrypted(tmp_buf, nbytes, MPI_BYTE,
                                                (void *)sendbuf, sendcnt, sendtype,
                                                0, shmem_commptr, errflag);
        } else {
            // mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
            mpi_errno = MPIR_Scatter_MV2_Direct_unencrypted(tmp_buf, nbytes, MPI_BYTE,
                                                recvbuf, recvcnt, recvtype,
                                                0, shmem_commptr, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
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
#define FUNCNAME MPIR_Scatter_MV2_Direct_unencrypted
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_unencrypted(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct_unencrypted\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(2);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-1 c=%d \n",init_rank,recvcount);
    
    rank = comm_ptr->rank;

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIR_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         sendtype_extent);

        MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray",MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);
        
        
        /*
        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");
        */

        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcnt) {
            // if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
            if ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (i == rank)) {
                if (recvbuf != MPI_IN_PLACE) {
                mpi_errno =
                    MPIR_Localcopy(((char *) sendbuf +
                            rank * sendcnt * sendtype_extent),
                           sendcnt, sendtype, recvbuf, recvcnt,
                           recvtype);
                }
            } else {
                // MPIR_PVAR_INC(scatter, direct, send, sendcnt, sendtype);
                
                /*MPID_Node_id_t rank_node_id, dest_node_id;
                MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                printf("[rank %d rank_node_id %d] sending to %d dest_node_id %d\n", 
                rank,rank_node_id,i,dest_node_id ); fflush(stdout);*/
                
                mpi_errno =
                MPIC_Isend(((char *) sendbuf +
                           i * sendcnt * sendtype_extent), sendcnt,
                          sendtype, i, MPIR_SCATTER_TAG, comm_ptr,
                          &reqarray[reqs++], errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            }
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS)
                mpi_errno = starray[i].MPI_ERROR;
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            }
        }
        /* --END ERROR HANDLING-- */
    } else if (root != MPI_PROC_NULL) {   /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcnt) {
            // MPIR_PVAR_INC(scatter, direct, recv, recvcnt, recvtype);
            mpi_errno = MPIC_Recv(recvbuf, recvcnt, recvtype, root,
                                     MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                     errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
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
#define FUNCNAME MPIR_Scatter_MV2_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    int rank, comm_size; 
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(3);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-1 c=%d \n",init_rank,recvcount);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

      /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIR_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         sendtype_extent);

        MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray", MPL_MEM_BUFFER);
        MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);

        /* added by abu naser */
        int j = 0;
        int next = 0;
        int dest = 0;
        int t= (sendtype_extent * sendcnt);
        unsigned long   max_out_len = (unsigned long) (16 + (sendtype_extent * sendcnt));
        unsigned long ciphertext_sendbuf_len;
        int  rank_node_id;
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);   
        /* end of add */
        
        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcnt) {
            if ((comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) && (i == rank)) {
                if (recvbuf != MPI_IN_PLACE) {
                mpi_errno =
                    MPIR_Localcopy(((char *) sendbuf +
                            rank * sendcnt * sendtype_extent),
                           sendcnt, sendtype, recvbuf, recvcnt,
                           recvtype);
                }
            } else {
                // MPIR_PVAR_INC(scatter, direct, send, sendcnt, sendtype);
                    
                    /* added by abu naser naive+ */
                    
                    int dest_node_id;
                    MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                    //printf("MV2_Direct send:[root %d rank %d rank_node_id %d] i=%d dest_node_id %d\n",root,rank,rank_node_id,i,dest_node_id);fflush(stdout);

                    if((security_approach == 200 || (security_approach > 200 && security_approach <= 211)) && rank_node_id != dest_node_id)
                    {
                        //printf("leader_comm_size=%d local_size=%d\n",leader_comm_size,local_size);fflush(stdout);
                        
                                next = i * sendcnt * sendtype_extent; // get the encrytion location   
                                RAND_bytes(ciphertext_sendbuf + dest, 12);
                                if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                                       &ciphertext_sendbuf_len, max_out_len,
                                                       ciphertext_sendbuf + dest, 12,
                                                       sendbuf + next, t,
                                                       NULL, 0))
                                {
                                    printf("Error in encryption: [rank %d rank_node_id %d] scatter direct dest %d dest_node_id %d\n",
                                    rank,rank_node_id,i,dest_node_id);fflush(stdout);
                                }
#if 0
                                else
                                {
                                    printf(COLOR_GREEN"Scatter direct:: [rank %d rank_node_id %d dest %d dest_node_id %d] Encrypted data size %lu"COLOR_RESET"\n",
                                           rank, rank_node_id, i , dest_node_id, ciphertext_sendbuf_len);
                                    fflush(stdout);
                                }
#endif
                                mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                           (t+28),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);    
                                
                             
                                dest += t + 16 + 12; // update after send
                            
                        }
                        /* end of add */
                        else
                        
                        { // unencrypted
                            mpi_errno =
                                MPIC_Isend(((char *)sendbuf +
                                            i * sendcnt * sendtype_extent),
                                           sendcnt,
                                           sendtype, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                        }
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            }
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS)
                mpi_errno = starray[i].MPI_ERROR;
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            }
        }
        /* --END ERROR HANDLING-- */
    } else if (root != MPI_PROC_NULL) {   /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcnt) {
            // MPIR_PVAR_INC(scatter, direct, recv, recvcnt, recvtype);
            /* added by abu naser */
            
            int rank_node_id;
            MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
            int root_node_id;
            MPID_Get_node_id(comm_ptr, i, &root_node_id);
            //printf("MV2_Direct recv:[root %d rank %d rank_node_id %d] i=%d root_node_id %d\n",root,rank,rank_node_id,i,root_node_id);fflush(stdout);
            if((security_approach == 200 || (security_approach > 200 && security_approach <= 211)) && rank_node_id != root_node_id)
            {
                int j = 0;
                int next = 0;
                int dest = 0;
                MPI_Aint recvtype_extent;
                MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
                int t = (recvtype_extent * recvcnt);
                unsigned long max_out_len = (unsigned long) (recvtype_extent * recvcnt);
                unsigned long count;

                mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t+28), MPI_BYTE, root,
                                     MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                     errflag);

                 if(!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                            &count, max_out_len,
                            (ciphertext_recvbuf), 12,
                            (ciphertext_recvbuf+12), (unsigned long )(t+16),
                            NULL, 0))
                        {
                            printf("Decryption error:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n",rank,rank_node_id);fflush(stdout);        
                        }
#if 0            
                        else
                        {
                            printf(COLOR_BLUE"Scatter direct recv:: [rank %d rank_node_id %d root_node_id %d] Decrypted data size %lu"COLOR_RESET"\n",
                            rank, rank_node_id, root_node_id, count);fflush(stdout);
                        }                             
#endif                                             

            }
            /* end of add */
            else
            
            { // unencrypted
                mpi_errno = MPIC_Recv(recvbuf, recvcnt, recvtype, root,
                                     MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                     errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
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
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem_intra_RR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem_intra_RR(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
    
if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif 
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }

#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[allreduce rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct_no_shmem_intra_RR\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    // MPI_Aint extent = 0;
    int rank, comm_size; //, sendtype_size;
    //int relative_rank, nbytes;
    //MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(6);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-1 c=%d \n",init_rank,recvcount);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    MPI_Comm conc_comm;
    MPIR_Comm *shmem_commptr= NULL;
    int rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-2 c=%d \n",init_rank,recvcount);

    shmem_commptr = comm_ptr->node_comm;

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-3 c=%d \n",init_rank,recvcount);    
    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-5 c=%d \n",init_rank,recvcount);

    /* if (mpi_errno)
    {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
    if (mpi_errno)
    {
        MPIR_ERR_POP(mpi_errno);
    } */
    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 01-6 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    // comm_size = comm_ptr->local_size;
    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray",MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);

    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 01 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    
    
    reqs = 0;
    int j;
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        // MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        int next;
        /*for(i=1; i<local_size;i++)
        {
             next = i*each_rank_size*t;
             mpi_errno = MPIC_Isend((sendbuf + next),
                                           (t *each_rank_size),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
            //printf("**[intra-node] rank %d sending to rank %d\n",rank,i);fflush(stdout);
        }*/

        for(j=0;j<each_rank_size;j++)
        {
            for(i=1;i<local_size;i++)
            {
                next = i*each_rank_size*t + j*t;
                if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 02 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
                mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++], errflag);
                if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 03 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
                // printf("**[intra-node] rank %d sending to rank %d location: %d, (i*each_rank_size*t=%d) (j*t=%d)\n",rank,i,next,i*each_rank_size*t,j*t);fflush(stdout);

            }
        }
                               
        //printf("root=%d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        int next = 0;
        
        /*mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);*/  
        for(i=0;i<each_rank_size;i++)
        {
            if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 04 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf+next, t, MPI_BYTE, root,
                  i, comm_ptr,  &reqarray[reqs++]);
            if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 05 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
            next += t;      
        }                
    }

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;

    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);

    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06 c=%d [%d vs. %d]\n",init_rank,local_rank,conc_rank,recvcount,rank_node_id,root_node_id);
        if (local_rank < leader_size)
        {
            MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                             sendtype_extent);

            /* divide encryption among ppn ranks */
            int each_rank_size = comm_size / leader_size; //local_size;
            int max_dest_ranks;

            if (local_rank == (leader_size - 1))
            {
                max_dest_ranks = comm_size;
            }
            else
            {
                max_dest_ranks = local_rank * each_rank_size + each_rank_size;
            }

            int next = 0;
            int dest = 0;
            int t = (sendtype_extent * sendcnt);
            unsigned long max_out_len = (unsigned long)(16 + (sendtype_extent * sendcnt));
            unsigned long ciphertext_sendbuf_len;
            int dest_node_id;

            if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-0 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

            MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
           
            j = 0;
            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                if (rank == i && rank == root)
                {
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-1 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    memcpy(recvbuf, sendbuf+next, t);
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-2 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    //printf("@@ copy to own rank %d == i %d\n", rank, i);
                }
                else if (rank_node_id == dest_node_id && rank == root) // sending to same node
                {
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-3 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-4 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    //printf("sending to same node rank %d ----> i %d\n", rank, i);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else if (rank_node_id == dest_node_id && rank != root) // sending to same node by non root
                {
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-5 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                    j++;
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-6 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    mpi_errno = MPIC_Isend((ciphertext_recvbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-7 [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    //printf("NON-ROOT sending to same node rank %d ----> i %d\n", rank, i);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else
                {
                    if(rank == root)
                        temp_buffer = sendbuf;
                    else
                    {
                        if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-8 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                        mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                        if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-9 [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                        j++;
                       temp_buffer = ciphertext_recvbuf;    
                    }
                    RAND_bytes(ciphertext_sendbuf + dest, 12);
                    if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                           &ciphertext_sendbuf_len, max_out_len,
                                           ciphertext_sendbuf + dest, 12,
                                           temp_buffer + next, t,
                                           NULL, 0))
                    {
                        printf("Error in encryption: [rank %d rank_node_id %d] scatter direct dest %d dest_node_id %d\n",
                               rank, rank_node_id, i, dest_node_id);
                        fflush(stdout);
                    }
#if 0
                    else
                    {

                        printf(COLOR_GREEN "Scatter no-shm-intra-div direct:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest %d dest_node_id %d] Encrypted data size %lu max_dest_ranks %d" COLOR_RESET "\n",
                               rank, local_rank, rank_node_id, root_node_id, i, dest_node_id, ciphertext_sendbuf_len, max_dest_ranks);
                        fflush(stdout);
                    }
#endif
                    //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-10 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                           (t + 28),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);

                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 06-11 c=%d [r=%d j=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs,j);
                    dest += t + 16 + 12; // update after send
                   
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                 next += t; // get the encrytion location
            } // i
        }
    }

    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 07 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int new_root = 0;
        //if(new_root >= local_size)
        //    new_root = local_size - 1;
        int each_rank_size = comm_size / leader_size;
        new_root = rank / each_rank_size;          
        int new_root_node_id;
        MPID_Get_node_id(comm_ptr, new_root, &new_root_node_id);

        /*if (rank == new_root)
        {
            printf("rank %d received from new_root\n", rank, new_root);
            fflush(stdout);
        }
        else if (rank_node_id == new_root_node_id)*/
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);

        if (rank_node_id == new_root_node_id && rank != new_root)
        {
            //printf(COLOR_CYAN"[same node]rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Recv(recvbuf, t, MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 07-1 [r=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }                      
            //printf("rank %d received from same node new_root %d\n",rank, new_root);fflush(stdout);                      
        }
        else if(rank_node_id != new_root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 07-2 [r=%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            //printf("[inter-node]rank %d received data from new_root %d\n", rank, new_root);
            if (!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf), 12,
                                   (ciphertext_recvbuf + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter shm recv:: [rank %d rank_node_id %d new_root %d] Decrypted data size %lu" COLOR_RESET "\n",
                       rank, rank_node_id, new_root, count);
                fflush(stdout);
            }
#endif
        }
    }

    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 08 c=%d [%d]\n",init_rank,local_rank,conc_rank,recvcount,reqs);
    /* wait for finish sending */
    // if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    
    if(rank_node_id == root_node_id && local_size != comm_size && rank == root)
    {
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
         /* ... then wait for *all* of them to finish: */
        if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 08-0 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 08-1 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 08-2 c=%d [%d,%d]\n",init_rank,local_rank,conc_rank,recvcount,i,reqs);
            if (starray[i].MPI_ERROR != MPI_SUCCESS)
                mpi_errno = starray[i].MPI_ERROR;
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            }
        }
        /* --END ERROR HANDLING-- */    
    }
if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-RR 09 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

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
#define FUNCNAME MPIR_Scatter_MV2_Direct_HBcast
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_HBcast(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    MPI_Aint sendtype_extent, recvtype_extent;
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    int max_size_allow = recvcount * recvtype_extent;
    
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
    
if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1 || max_size_allow<(COMMON_LARGE_RECV_BUFFER_SIZE+12+16)){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d max_size_allow=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform,max_size_allow);fflush(stdout);
	}
#endif 

        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }    
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct_HBcast\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    // MPI_Aint extent = 0;
    int rank, comm_size; //, sendtype_size;
    //int relative_rank, nbytes;
    //MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    
    int i, reqs;
    unsigned int t;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    // MPIR_CHKLMEM_DECL(2);

    
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    
    int rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;


    unsigned long ciphertext_sendbuf_len, max_out_len, count;

    if (rank == root)
    {
        t = sendtype_extent * sendcnt * comm_ptr->local_size;
        max_out_len = t + 16;
        RAND_bytes(ciphertext_sendbuf, 12);
        if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                               &ciphertext_sendbuf_len, max_out_len,
                               ciphertext_sendbuf, 12,
                               sendbuf, t,
                               NULL, 0))
        {
            printf("Error in encryption HBcast: [rank %d rank_node_id %d] scatter direct\n",
                   rank, rank_node_id);
            fflush(stdout);
        }
#if 0
            else
            {

                printf(COLOR_GREEN "Scatter HBcast:: [rank %d  rank_node_id %d ] Encrypted data size %lu" COLOR_RESET "\n",
                       rank, rank_node_id, ciphertext_sendbuf_len);
                fflush(stdout);
            }
#endif
         mpi_errno = MPIR_Bcast_impl(ciphertext_sendbuf, (t+28), MPI_BYTE, root, comm_ptr, errflag);
         
         mpi_errno = MPIR_Localcopy((void*)((char*)sendbuf), sendcnt, sendtype, 
                                    (void*)((char*)recvbuf), sendcnt, sendtype);
    }
    else
    {
        t = recvtype_extent * recvcnt * comm_ptr->local_size;
        mpi_errno = MPIR_Bcast_impl(ciphertext_sendbuf, (t+28), MPI_BYTE, root, comm_ptr, errflag);
        
        if (!EVP_AEAD_CTX_open(global_ctx, ciphertext_recvbuf,
                                   &count, t,
                                   (ciphertext_sendbuf), 12,
                                   (ciphertext_sendbuf + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error HBcast:[rank %d rank_node_id %d t=%d] scatter direct recv\n", rank, rank_node_id,t);
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter HBcast:: [rank %d rank_node_id %d ] Decrypted data size %lu" COLOR_RESET "\n",
                       rank,rank_node_id, count);
                fflush(stdout);
            }  
#endif        

        t = recvtype_extent * recvcnt;   
        mpi_errno = MPIR_Localcopy((void*)((char*)(ciphertext_recvbuf+rank*t)), sendcnt, sendtype, 
                                    (void*)((char*)recvbuf), sendcnt, sendtype);
    }


  fn_exit:
    // MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}




#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_CHS
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_CHS(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
    
if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif 
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }    
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d] Func: MPIR_Scatter_MV2_Direct_CHS\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      
    MPI_Status status;
    // MPI_Aint extent = 0;
    int rank, comm_size; //, sendtype_size;
    //int relative_rank, nbytes;
    //MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPIR_Request **reqarray;
    MPI_Status *starray;
    MPIR_CHKLMEM_DECL(6);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-1 c=%d \n",init_rank,recvcount);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    MPI_Comm conc_comm;
    MPIR_Comm *shmem_commptr, *conc_commptr = NULL;
    int rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-2 c=%d \n",init_rank,recvcount);

    shmem_commptr = comm_ptr->node_comm;

     if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-3 c=%d \n",init_rank,recvcount);
    //conc_comm = comm_ptr->dev.ch.concurrent_comm;

    
    conc_comm = comm_ptr->concurrent_comm;    
    

    if (conc_comm == NULL) {
         fprintf(stderr,"[%d] NULL conc_comm c=%d\n",init_rank,sendcnt);
        mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
        conc_comm = comm_ptr->concurrent_comm;    
    } 
    

    MPIR_Comm_get_ptr(conc_comm, conc_commptr);  

    local_rank = comm_ptr->node_comm->rank;
    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-5 c=%d \n",init_rank,recvcount);
    local_size = comm_ptr->node_comm->local_size;

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Scatter-CHS 01-5 c=%d \n",init_rank,recvcount);

    /* if (mpi_errno)
    {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
    if (mpi_errno)
    {
        MPIR_ERR_POP(mpi_errno);
    } */
    // MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    // MPID_Comm_get_ptr(conc_comm, conc_commptr);

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 01-6 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    comm_size = comm_ptr->local_size;
    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray",MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);

 
    if (root == rank && local_size != comm_size)
    {
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 03 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        //comm_size = comm_ptr->remote_size;
        memcpy(shmem_buffer, sendbuf, sendcnt * sendtype_extent * comm_size);
        //printf("root=%d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 04 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

    int next = 0;
    int dest = 0;
    int dest_global_rank = 0;
    int j = 0;
    if (rank_node_id == root_node_id && local_size != comm_size)
    {
        mpi_errno = MPIR_Barrier_impl(shmem_commptr, errflag);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
        
        int t = (sendtype_extent * sendcnt);
        unsigned long max_out_len = (unsigned long)(16 + (sendtype_extent * sendcnt));
        unsigned long ciphertext_sendbuf_len;
       
        int each_rank_size = comm_size / local_size; // consider if equal now
        int max_ranks = local_rank * each_rank_size + each_rank_size;
       // printf("rank %d each_rank_size %d max_ranks %d\n", rank, each_rank_size, max_ranks);fflush(stdout);
        dest = 0;
        reqs = 0;
        //dest_global_rank = rank;
        dest_global_rank = rank + local_size;
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 05 c=%d \n",init_rank,local_rank,conc_rank,recvcount);

        /* don't send to own process, instead copy later */
        for (i = 1; i < conc_size ; i++)
        {
            next = dest_global_rank * sendcnt * sendtype_extent; // get the encrytion location

            RAND_bytes(large_send_buffer + dest, 12);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 06 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer + dest + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   large_send_buffer + dest, 12,
                                   shmem_buffer + next, t,
                                   NULL, 0))
            {
                printf("Error in encryption HS2-ML: [rank %d rank_node_id %d] scatter direct dest_global_rank %d conc_rank %d\n",
                       rank, rank_node_id, dest_global_rank, conc_rank);
                fflush(stdout);
            }
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 07 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
#if 0
            else
            {

                printf(COLOR_GREEN "Scatter HS2-ML:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest_global_rank %d dest %d] Encrypted data size %lu conc_rank %d" COLOR_RESET "\n",
                       rank, local_rank, rank_node_id, root_node_id, dest_global_rank, dest, ciphertext_sendbuf_len, conc_rank);
                fflush(stdout);
            }
#endif

               //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
               //printf(COLOR_MAGENTA"HS2-ML [rank %d conc_rank %d] sending to dest_rank %d dest conc rank %d"COLOR_RESET"\n",rank, conc_rank, dest_global_rank,i);fflush(stdout);
            mpi_errno = MPIC_Isend((large_send_buffer + dest),
                                    (t + 28),
                                    MPI_BYTE, i, MPIR_SCATTER_TAG, conc_commptr,
                                    &reqarray[reqs++], errflag);
           // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 08 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
           dest += (t + 28);
           dest_global_rank += local_size;

        }

        /* itner-node sending is done, now copy to own recv buffer */
        next = rank * t; 
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 09 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        memcpy(recvbuf, shmem_buffer+next,t);
        //printf("*****HS2-ML [rank %d conc_rank %d] copied data %d to own process next %d\n",rank,conc_rank,t,next);fflush(stdout);
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 10 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
    }

     /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 11 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int source = rank % local_size;

#if 0       
        if (rank_node_id == root_node_id && rank != root) // this case will not execute
        {
            printf(COLOR_CYAN"HS2-ML [same node]rank %d receiving from source %d"COLOR_RESET"\n",rank,source);fflush(stdout);
            mpi_errno = MPIC_Recv(recvbuf, t, MPI_BYTE, 0,
                                  MPIR_SCATTER_TAG, conc_commptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }                      
           // printf("rank %d received from same node new_root %d\n",rank, new_root);fflush(stdout);                      
        }
        else if(rank_node_id != root_node_id)
#endif        
        if(rank_node_id != root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"HS2-ML rank %d receiving from source %d "COLOR_RESET"\n",rank,source);fflush(stdout);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 12 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
            mpi_errno = MPIC_Recv(large_recv_buffer, (t + 28), MPI_BYTE, 0,
                                  MPIR_SCATTER_TAG, conc_commptr, MPI_STATUS_IGNORE,
                                  errflag);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 13 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
           // printf("[inter-node]rank %d received data from source %d\n", rank, source);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, recvbuf,
                                   &count, max_out_len,
                                   (large_recv_buffer), 12,
                                   (large_recv_buffer + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error HS2-ML:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                fflush(stdout);
            }
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 15 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter HS2-ML recv:: [rank %d rank_node_id %d source %d] Decrypted data size %lu" COLOR_RESET "\n",
                       rank, rank_node_id, source, count);
                fflush(stdout);
            }
#endif
        }
    }
    else
    { // un-uniform not handled
        printf("HS2-ML un-uniform case not handled yet\n");fflush(stdout);
    }

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size)
    {
         /* ... then wait for *all* of them to finish: */
         // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 16 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] [%d] [%d] Scatter-CHS 17 c=%d \n",init_rank,local_rank,conc_rank,recvcount);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
            if (starray[i].MPI_ERROR != MPI_SUCCESS)
                mpi_errno = starray[i].MPI_ERROR;
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            }
        }
        /* --END ERROR HANDLING-- */    
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
#define FUNCNAME MPIR_Naive_Sec_Scatter
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Naive_Sec_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
/*    
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);
    
if (comm_ptr->is_blocked != 1 || root !=0 || comm_ptr->is_uniform != 1){
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d  root=%d  is_blocked=%d  is_uniform=%d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif 
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }
*/
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[allreduce rank=%d host=%s count=%d] Func: MPIR_Naive_Sec_Scatter\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif      


    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int sendtype_sz, recvtype_sz;
    unsigned long  ciphertext_sendbuf_len = 0;
    sendtype_sz= recvtype_sz= 0;
    int var;
    unsigned long count;
    unsigned char *ciphertext_sendbuf =large_send_buffer;
    unsigned char *ciphertext_recvbuf =large_recv_buffer;

    MPIR_Datatype_get_size_macro(sendtype, sendtype_sz);
    MPIR_Datatype_get_size_macro(recvtype, recvtype_sz);

    int rank,i,next,dest;
    
    rank = comm_ptr->rank;


    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Scatter-Naive sc=%d rc=%d\n",init_rank,sendcount,recvcount);
    
    if(rank == root)
    {
        next = 0;
        dest = 0;
        unsigned long t= (unsigned long)(sendtype_sz*sendcount);
        unsigned long   max_out_len = (unsigned long) (16 + (sendtype_sz*sendcount));
        for(i= 0; i< comm_ptr->local_size; i++)
        {
              RAND_bytes(ciphertext_sendbuf+dest, 12);
               if(!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf+dest+12,
                         &ciphertext_sendbuf_len, max_out_len,
                         ciphertext_sendbuf+dest, 12,
                         sendbuf+next,  t,
                        NULL, 0))
                {
                    printf("Error in encryption: scatter\n");
                    fflush(stdout);
                }
#if 0                
                else
                {
                    printf("Scatter:: [root %d] Encrypted data size %lu dest rank %d\n",root, ciphertext_sendbuf_len, i);fflush(stdout);
                }
#endif            
            next += t;
            dest += t+16+12;    
        }
        //init_phase = 1;

        //if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Scatter-Naive sc=%d rc=%d\n",init_rank,(sendtype_sz*sendcount)+16+12,(recvcount*recvtype_sz)+16+12);MPIR_Scatter_impl

        mpi_errno = MPIR_Scatter_intra_binomial(ciphertext_sendbuf, ((sendtype_sz*sendcount)+16+12), MPI_CHAR,
                                  ciphertext_recvbuf, ((recvcount*recvtype_sz)+16+12), MPI_CHAR, root,
                                  comm_ptr, &errflag);
        //init_phase = 0;

        if(!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                        &count, (unsigned long )((recvcount*recvtype_sz)),
                         (ciphertext_recvbuf), 12,
                        (ciphertext_recvbuf+12), (unsigned long )((recvcount*recvtype_sz)+16),
                        NULL, 0))
            {
                    printf("Decryption error: scatter\n");fflush(stdout);        
            }
#if 0            
            else
            {
                printf("Scatter:: [rank %d] Decrypted data size %lu\n",rank, count);fflush(stdout);
            }                             
#endif

    }
    else
    {
        // init_phase = 1;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Scatter-Naive sc=%d rc=%d\n",init_rank,(sendtype_sz*sendcount)+16+12,(recvcount*recvtype_sz)+16+12);
        mpi_errno = MPIR_Scatter_intra_binomial(ciphertext_sendbuf, ((sendtype_sz*sendcount)+16+12), MPI_CHAR,
                                  ciphertext_recvbuf, ((recvcount*recvtype_sz)+16+12), MPI_CHAR, root,
                                  comm_ptr, &errflag);
        // init_phase = 0;
         if(!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                        &count, (unsigned long )((recvcount*recvtype_sz)),
                         (ciphertext_recvbuf), 12,
                        (ciphertext_recvbuf+12), (unsigned long )((recvcount*recvtype_sz)+16),
                        NULL, 0))
            {
                    printf("Decryption error: scatter\n");fflush(stdout);        
            }
#if 0            
            else
            {
                printf("Scatter:: [rank %d] Decrypted data size %lu\n",rank, count);fflush(stdout);
            }                             
#endif                          

    }

  fn_exit:
    // MPIR_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}





/* not declared static because a machine-specific function may call this one in some cases */
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_intra_binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_intra_binomial(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if SCATTER_PRINT_FUN
   if (PRINT_FUN_NAME){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Scatter rank=%d host=%s count=%d SA=%d] Func: MPIR_Scatter_intra_binomial\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif     
    MPI_Status status;
    MPI_Aint extent = 0;
    int rank, comm_size, sendtype_size;
    int relative_rank, nbytes;
    MPI_Aint curr_cnt, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
    int tmp_buf_size = 0;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPIR_CHKLMEM_DECL(4);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcount == 0)) || ((rank != root) && (recvcount == 0)))
        return MPI_SUCCESS;

    if (rank == root)
        MPIR_Datatype_get_extent_macro(sendtype, extent);

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;


    if (rank == root) {
        /* We separate the two cases (root and non-root) because
         * in the event of recvbuf=MPI_IN_PLACE on the root,
         * recvcount and recvtype are not valid */
        MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         extent * sendcount * comm_size);

        nbytes = sendtype_size * sendcount;
    } else {
        MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Ensure_Aint_fits_in_pointer(extent * recvcount * comm_size);
        nbytes = recvtype_size * recvcount;
    }

    curr_cnt = 0;

    /* all even nodes other than root need a temporary buffer to
     * receive data of max size (nbytes*comm_size)/2 */
    if (relative_rank && !(relative_rank % 2)) {
        tmp_buf_size = (nbytes * comm_size) / 2;
        MPIR_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);
    }

    /* if the root is not rank 0, we reorder the sendbuf in order of
     * relative ranks and copy it into a temporary buffer, so that
     * all the sends from the root are contiguous and in the right
     * order. */
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 01 c=%d\n",init_rank,recvcount);  
    if (rank == root) {
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 02 c=%d\n",init_rank,recvcount);  
        if (root != 0) {
            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 03 c=%d\n",init_rank,recvcount);  
            tmp_buf_size = nbytes * comm_size;
            MPIR_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf",
                                MPL_MEM_BUFFER);

            if (recvbuf != MPI_IN_PLACE)
                mpi_errno = MPIR_Localcopy(((char *) sendbuf + extent * sendcount * rank),
                                           sendcount * (comm_size - rank), sendtype, tmp_buf,
                                           nbytes * (comm_size - rank), MPI_BYTE);
            else
                mpi_errno = MPIR_Localcopy(((char *) sendbuf + extent * sendcount * (rank + 1)),
                                           sendcount * (comm_size - rank - 1),
                                           sendtype, (char *) tmp_buf + nbytes,
                                           nbytes * (comm_size - rank - 1), MPI_BYTE);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIR_Localcopy(sendbuf, sendcount * rank, sendtype,
                                       ((char *) tmp_buf + nbytes * (comm_size - rank)),
                                       nbytes * rank, MPI_BYTE);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            curr_cnt = nbytes * comm_size;
        } else {
            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 04 c=%d\n",init_rank,recvcount);  
            curr_cnt = sendcount * comm_size;
        }
    }

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05 c=%d\n",init_rank,recvcount);  

    /* root has all the data; others have zero so far */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            /* The leaf nodes receive directly into recvbuf because
             * they don't have to forward data to anyone. Others
             * receive data into a temporary buffer. */
            if (relative_rank % 2) {
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-1 c=%d mask=%d \n",init_rank,recvcount,mask);  
                mpi_errno = MPIC_Recv(recvbuf, recvcount, recvtype,
                                      src, MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-2 c=%d mask=%d \n",init_rank,recvcount,mask);  
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            } else {
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-3 c=%d mask=%d \n",init_rank,recvcount,mask);  
                mpi_errno = MPIC_Recv(tmp_buf, tmp_buf_size, MPI_BYTE, src,
                                      MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-4 c=%d mask=%d \n",init_rank,recvcount,mask);  
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    curr_cnt = 0;
                } else
                    /* the recv size is larger than what may be sent in
                     * some cases. query amount of data actually received */
                    MPIR_Get_count_impl(&status, MPI_BYTE, &curr_cnt);
            }
            break;
            // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-5 c=%d mask=%d \n",init_rank,recvcount,mask);  
        }
        mask <<= 1;
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 05-6 c=%d mask=%d \n",init_rank,recvcount,mask);  
    }

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06 c=%d\n",init_rank,recvcount);

    /* This process is responsible for all processes that have bits
     * set from the LSB upto (but not including) mask.  Because of
     * the "not including", we start by shifting mask back down
     * one. */

    mask >>= 1;
    while (mask > 0) {
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-1 c=%d mask=%d \n",init_rank,recvcount,mask);  
        if (relative_rank + mask < comm_size) {
            dst = rank + mask;
            if (dst >= comm_size)
                dst -= comm_size;

            if ((rank == root) && (root == 0)) {
                send_subtree_cnt = curr_cnt - sendcount * mask;
                /* mask is also the size of this process's subtree */
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-2 c=%d mask=%d \n",init_rank,recvcount,mask);  
                mpi_errno = MPIC_Send(((char *) sendbuf +
                                       extent * sendcount * mask),
                                      send_subtree_cnt,
                                      sendtype, dst, MPIR_SCATTER_TAG, comm_ptr, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-3 c=%d mask=%d \n",init_rank,recvcount,mask);  
            } else {
                /* non-zero root and others */
                send_subtree_cnt = curr_cnt - nbytes * mask;
                /* mask is also the size of this process's subtree */
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-4 c=%d mask=%d \n",init_rank,recvcount,mask);  
                mpi_errno = MPIC_Send(((char *) tmp_buf + nbytes * mask),
                                      send_subtree_cnt,
                                      MPI_BYTE, dst, MPIR_SCATTER_TAG, comm_ptr, errflag);
                // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-5 c=%d mask=%d \n",init_rank,recvcount,mask);  
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag =
                    MPIX_ERR_PROC_FAILED ==
                    MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            curr_cnt -= send_subtree_cnt;
        }
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 06-6 c=%d mask=%d \n",init_rank,recvcount,mask);  
        mask >>= 1;
    }

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 07 c=%d\n",init_rank,recvcount);

    if ((rank == root) && (root == 0) && (recvbuf != MPI_IN_PLACE)) {
        /* for root=0, put root's data in recvbuf if not MPI_IN_PLACE */
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else if (!(relative_rank % 2) && (recvbuf != MPI_IN_PLACE)) {
        /* for non-zero root and non-leaf nodes, copy from tmp_buf
         * into recvbuf */
        mpi_errno = MPIR_Localcopy(tmp_buf, nbytes, MPI_BYTE, recvbuf, recvcount, recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 08 c=%d\n",init_rank,recvcount);

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 09 c=%d\n",init_rank,recvcount);
    if (mpi_errno_ret){
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 10 c=%d\n",init_rank,recvcount);
        mpi_errno = mpi_errno_ret;
        }
    else if (*errflag != MPIR_ERR_NONE){
        // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 11 c=%d\n",init_rank,recvcount);
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    }
    // if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Satter-binomial 12 c=%d\n",init_rank,recvcount);
    mpi_errno = MPI_SUCCESS;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


