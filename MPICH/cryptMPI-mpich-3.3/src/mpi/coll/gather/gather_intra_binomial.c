/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"
unsigned char bio_temp_buffer[8388608*8]; 

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_GATHER_VSMALL_MSG_SIZE
      category    : COLLECTIVE
      type        : int
      default     : 1024
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        use a temporary buffer for intracommunicator MPI_Gather if the send
        buffer size is < this value (in bytes)
        (See also: MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE)

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* Algorithm: MPI_Gather
 *
 * We use a binomial tree algorithm for both short and long messages. At nodes
 * other than leaf nodes we need to allocate a temporary buffer to store the
 * incoming message. If the root is not rank 0, for very small messages, we
 * pack it into a temporary contiguous buffer and reorder it to be placed in
 * the right order. For small (but not very small) messages, we use a derived
 * datatype to unpack the incoming data into non-contiguous buffers in the
 * right order.
 *
 * Cost = lgp.alpha + n.((p-1)/p).beta where n is the total size of the data
 * gathered at the root.
 */


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_intra
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_intra(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if GATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Gather rank = %d host = %s count = %d] Func: MPIR_Gather_intra\n", comm_ptr->rank,hostname,recvcount);fflush(stdout);
	}
#endif

    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int relative_rank, is_homogeneous;
    int mask, src, dst, relative_src;
    MPI_Aint curr_cnt = 0, nbytes, sendtype_size, recvtype_size;
    int recvblks;
    int missing;
    MPI_Aint tmp_buf_size;
    void *tmp_buf = NULL;
    MPI_Status status;
    MPI_Aint extent = 0; /* Datatype extent */
    int blocks[2];
    int displs[2];
    MPI_Aint struct_displs[2];
    MPI_Datatype types[2], tmp_type;
    int copy_offset = 0, copy_blks = 0;

    int sendcnt = sendcount;
    int recvcnt = recvcount;    

    // void *ciphertext_recvbuf;
    // void *ciphertext_sendbuf;
    
    

    MPIR_CHKLMEM_DECL(4); 

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 01 c=%d \n",init_rank,recvcount);


    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcount == 0)) ||
        ((rank != root) && (sendcount == 0)))
        return MPI_SUCCESS;

    is_homogeneous = 1;

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;
    MPIR_Datatype_get_extent_macro(recvtype, extent);

    if (rank == root)
    {
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                         (extent * recvcount * comm_size));
    }


    if (((extent * recvcnt * comm_size) > COMMON_LARGE_SEND_BUFFER_SIZE) && 1){

        if (init_rank==0) printf("CryptMPI: Unable to work for message size %d and more. It needs more memory than permitted in this cluster. Change Gather mode (MV2_INTER_GATHER_TUNING). \n",recvcnt);
        return MPI_SUCCESS;

    }

    /* added by abu naser */
    int rank_node_id, root_node_id, dst_node_id, src_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id); 
    /* end of add */

    MPI_Aint true_lb, true_extent;    
    

    MPIR_Type_get_true_extent_impl(recvtype, &true_lb, &true_extent);    
    MPIR_Ensure_Aint_fits_in_pointer(recvcnt * MPL_MAX(extent, true_extent));

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 02 c=%d \n",init_rank,recvcount);

    if (is_homogeneous)
    {

        /* communicator is homogeneous. no need to pack buffer. */

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 03 c=%d \n",init_rank,recvcount);

        if (rank == root)
        {
            MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
            nbytes = recvtype_size * recvcount;
        }
        else
        {
            MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
            nbytes = sendtype_size * sendcount;
        }

        /* Find the number of missing nodes in my sub-tree compared to
	 * a balanced tree */
        for (mask = 1; mask < comm_size; mask <<= 1)
            ;
        --mask;
        while (relative_rank & mask)
            mask >>= 1;
        missing = (relative_rank | mask) - comm_size + 1;
        if (missing < 0)
            missing = 0;
        tmp_buf_size = (mask - missing);

        /* If the message is smaller than the threshold, we will copy
	 * our message in there too */
        if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
            tmp_buf_size++;

        tmp_buf_size *= nbytes;

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 04 c=%d \n",init_rank,recvcount);

        /* For zero-ranked root, we don't need any temporary buffer */
        if ((rank == root) && (!root || (nbytes >= MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)))
            tmp_buf_size = 0;

        if (tmp_buf_size)
        {
            MPIR_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);
        }

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 05 c=%d \n",init_rank,recvcount);

        if (rank == root)
        {
            if (sendbuf != MPI_IN_PLACE)
            {
                mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                           ((char *)recvbuf + extent * recvcount * rank), recvcount, recvtype);
                if (mpi_errno)
                {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
        }
        else if (tmp_buf_size && (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE))
        {
            /* copy from sendbuf into tmp_buf */
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       tmp_buf, nbytes, MPI_BYTE);
            if (mpi_errno)
            {
                MPIR_ERR_POP(mpi_errno);
            }
        }
        curr_cnt = nbytes;

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 06 c=%d \n",init_rank,recvcount);

        mask = 0x1;
        while (mask < comm_size)
        {
            if ((mask & relative_rank) == 0)
            {
                //  void *ciphertext_recvbuf;
                // unsigned char ciphertext_recvbuf[64 * 1024 *1024];
                
                unsigned char *ciphertext_recvbuf =large_recv_buffer;

                // MPIR_CHKLMEM_MALLOC(ciphertext_recvbuf, void *, recvcnt * (MPL_MAX(extent, true_extent)), mpi_errno, "ciphertext_recvbuf", MPL_MEM_BUFFER);
                // MPIR_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);

                src = relative_rank | mask;
                if (src < comm_size)
                {
                    src = (src + root) % comm_size;

                    if (rank == root)
                    {
                        recvblks = mask;
                        if ((2 * recvblks) > comm_size)
                            recvblks = comm_size - recvblks;

                        if ((rank + mask + recvblks == comm_size) ||
                            (((rank + mask) % comm_size) <
                             ((rank + mask + recvblks) % comm_size)))
                        {
                            /* If the data contiguously fits into the
			     * receive buffer, place it directly. This
			     * should cover the case where the root is
			     * rank 0. */
                            //printf("1 . rank %d root %d src %d mask %d recvblks %d recvcount %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, recvcount, MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                           // printf("1. rank %d src %d recvblks %d\n",rank, src,recvblks);fflush(stdout);
                           // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 07 c=%d \n",init_rank,recvcount);

                            if (security_approach == 302 || (security_approach > 302 && security_approach <= 305))
                            {
                                MPID_Get_node_id(comm_ptr, src, &src_node_id);
                                if (rank_node_id == src_node_id)
                                {
                                     mpi_errno = MPIC_Recv(((char *)recvbuf +
                                                       (((rank + mask) % comm_size) * (MPI_Aint)recvcount * extent)),
                                                      (MPI_Aint)recvblks * recvcount, recvtype, src,
                                                      MPIR_GATHER_TAG, comm_ptr,
                                                      &status, errflag);
                                }
                                else
                                {
                                    // receive encrypted data and unpack it
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 08 c=%d \n",init_rank,recvcount);
                                    int total_data = recvblks * recvcount * recvtype_size;
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                                          (total_data+28), MPI_BYTE, src,
                                                          MPIR_GATHER_TAG, comm_ptr,
                                                          &status, errflag);

                                    
                                    unsigned long count, max_out_len;
                                    int next;
                                    max_out_len = total_data;
                                    next =  (((rank + mask) % comm_size) * (MPI_Aint)recvcount * extent);
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 08-1 c=%d next=%d\n",init_rank,recvcount,next);
                                    // (char *) recvbuf+ 2 = 'a';
                                    //RAND_bytes(recvbuf, 12);
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 08-2 c=%d next=%d\n",init_rank,recvcount,next);
                                    //RAND_bytes(recvbuf, next);
                                     if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 09 c=%d src=%d next=%d total_data=%d\n",init_rank,recvcount,src,next,total_data);
                                    if (!EVP_AEAD_CTX_open(global_ctx, recvbuf+next,
                                                           &count, max_out_len,
                                                           (ciphertext_recvbuf), 12,
                                                           (ciphertext_recvbuf + 12), (unsigned long)(total_data + 16),
                                                           NULL, 0))
                                    {
                                        printf(COLOR_RED"1. Decryption error:[rank %d src %d total_data %d rank_node_id %d src_node_id %d] gather recv"COLOR_RESET"\n", 
                                        rank, src, total_data, rank_node_id, src_node_id);
                                        fflush(stdout);
                                    }
                                    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 09-1 c=%d \n",init_rank,recvcount);
#if 0
                                    else
                                    {
                                        printf(COLOR_BLUE "gather bio root recv:: [rank %d rank_node_id %d] Decrypted data size %lu" COLOR_RESET "\n",
                                        rank, rank_node_id, count);
                                        fflush(stdout);
                                    }
#endif
                                }
                            }
                            else
                            {
                                mpi_errno = MPIC_Recv(((char *)recvbuf +
                                                       (((rank + mask) % comm_size) * (MPI_Aint)recvcount * extent)),
                                                      (MPI_Aint)recvblks * recvcount, recvtype, src,
                                                      MPIR_GATHER_TAG, comm_ptr,
                                                      &status, errflag);
                            }
                            if (mpi_errno)
                            {
                                /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                        }
                        else if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
                        {
                            if (security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                            {
                                MPID_Get_node_id(comm_ptr, src, &src_node_id);
                                if (rank_node_id == src_node_id)
                                {
                                    MPIR_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
                                    //printf("2 . rank %d root %d src %d mask %d recvblks %d nbytes %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, nbytes,MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                                    // printf("2. rank %d src %d recvblks %d\n",rank, src, recvblks);fflush(stdout);
                                    mpi_errno = MPIC_Recv(tmp_buf, (int)(recvblks * nbytes),
                                                          MPI_BYTE, src, MPIR_GATHER_TAG,
                                                          comm_ptr, &status, errflag);
                                }
                                else
                                {
                                    MPIR_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
                                    int total_data = recvblks * nbytes;
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 11 c=%d \n",init_rank,recvcount);
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf, (total_data+28),
                                                          MPI_BYTE, src, MPIR_GATHER_TAG,
                                                          comm_ptr, &status, errflag);
                                    
                                    unsigned long count, max_out_len;
                                    int next;
                                    max_out_len = total_data;

                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 12 c=%d \n",init_rank,recvcount);
                                   
                                    if (!EVP_AEAD_CTX_open(global_ctx, tmp_buf,
                                                           &count, max_out_len,
                                                           (ciphertext_recvbuf), 12,
                                                           (ciphertext_recvbuf + 12), (unsigned long)(total_data + 16),
                                                           NULL, 0))
                                    {
                                        printf(COLOR_RED"2. Decryption error:[rank %d src %d total_data %d rank_node_id %d src_node_id %d] gather recv"COLOR_RESET"\n", 
                                        rank, src, total_data,rank_node_id,src_node_id);
                                        fflush(stdout);
                                    }
#if 0
                                    else
                                    {
                                        printf(COLOR_BLUE "2. gather bio root recv:: [rank %d rank_node_id %d] Decrypted data size %lu" COLOR_RESET "\n",
                                        rank, rank_node_id, count);
                                        fflush(stdout);
                                    }
#endif                      
                                }
                            }
                            else
                            { // unencrypted
                                /* small transfer size case. cast ok */
                                MPIR_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
                                //printf("2 . rank %d root %d src %d mask %d recvblks %d nbytes %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, nbytes,MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                                // printf("2. rank %d src %d recvblks %d\n",rank, src, recvblks);fflush(stdout);
                                mpi_errno = MPIC_Recv(tmp_buf, (int)(recvblks * nbytes),
                                                      MPI_BYTE, src, MPIR_GATHER_TAG,
                                                      comm_ptr, &status, errflag);
                            }

                            if (mpi_errno)
                            {
                                /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                            copy_offset = rank + mask;
                            copy_blks = recvblks;
                        }
                        else
                        {
                            if (security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                            {
                                MPID_Get_node_id(comm_ptr, src, &src_node_id);
                                if (rank_node_id == src_node_id)
                                {
                                    blocks[0] = recvcount * (comm_size - root - mask);
                                    displs[0] = recvcount * (root + mask);
                                    blocks[1] = (recvcount * recvblks) - blocks[0];
                                    displs[1] = 0;
                                    // printf("3. rank %d root %d src %d mask %d recvblks %d nbytes %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, nbytes,MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                                    //printf(COLOR_RED"3. rank %d src %d  blocks[0]=%d   blocks[1]=%d"COLOR_RESET"\n",rank, src,  blocks[0], blocks[1]);fflush(stdout);
                                    mpi_errno = MPIR_Type_indexed_impl(2, blocks, displs, recvtype, &tmp_type);
                                    if (mpi_errno)
                                        MPIR_ERR_POP(mpi_errno);

                                    mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                                    if (mpi_errno)
                                        MPIR_ERR_POP(mpi_errno);

                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 13 c=%d \n",init_rank,recvcount);

                                    mpi_errno = MPIC_Recv(recvbuf, 1, tmp_type, src,
                                                          MPIR_GATHER_TAG, comm_ptr, &status, errflag);
                                    
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 14 c=%d \n",init_rank,recvcount);
                                    if (mpi_errno)
                                    {
                                        /* for communication errors, just record the error but continue */
                                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                    }

                                    MPIR_Type_free_impl(&tmp_type);
                                }
                                else
                                {
                                    // receive encrypted data and unpack it
                                    int total_data = (recvcount * (comm_size - root - mask)) *recvtype_size + (recvcount * recvblks)*recvtype_size;
                                    int temp_data = (recvcount * (comm_size - root - mask)) * recvtype_size;
                                    total_data = total_data - temp_data;
                                    //printf("3. total data %d\n",total_data);fflush(stdout);

                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 15 c=%d \n",init_rank,recvcount);
                                    
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                                          (total_data+28), MPI_BYTE, src,
                                                          MPIR_GATHER_TAG, comm_ptr,
                                                          &status, errflag);

                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 16 c=%d \n",init_rank,recvcount);
                                    unsigned long count, max_out_len;
                                    int next;
                                    max_out_len = total_data;
                                   
                                    if (!EVP_AEAD_CTX_open(global_ctx, bio_temp_buffer,
                                                           &count, max_out_len,
                                                           (ciphertext_recvbuf), 12,
                                                           (ciphertext_recvbuf + 12), (unsigned long)(total_data + 16),
                                                           NULL, 0))
                                    {
                                        printf(COLOR_RED"3. Decryption error:[rank %d src %d total_data %d rank_node_id %d src_node_id %d] gather recv"COLOR_RESET"\n",
                                         rank, src, total_data, rank_node_id,src_node_id);
                                        fflush(stdout);
                                    }
#if 0
                                    else
                                    {
                                        printf(COLOR_BLUE "3. gather bio root recv:: [rank %d rank_node_id %d] Decrypted data size %lu" COLOR_RESET "\n",
                                        rank, rank_node_id, count);
                                        fflush(stdout);
                                    }
#endif
                                    temp_data = (recvcount * (comm_size - root - mask)) * recvtype_size;
                                    int temp_data_2 = temp_data;
                                    
                                    next = (recvcount * (root + mask)) * recvtype_size;
                                    memcpy(recvbuf+next,bio_temp_buffer, temp_data);
                                    
                                    temp_data = ((recvcount * recvblks) * recvtype_size) - ((recvcount * (comm_size - root - mask)) * recvtype_size);
                                    next = 0;
                                   
                                     memcpy(recvbuf+next,bio_temp_buffer+temp_data_2, temp_data);
                                }
                            }
                            else
                            { // unencrypted
                                blocks[0] = recvcount * (comm_size - root - mask);
                                displs[0] = recvcount * (root + mask);
                                blocks[1] = (recvcount * recvblks) - blocks[0];
                                displs[1] = 0;
                                // printf("3. rank %d root %d src %d mask %d recvblks %d nbytes %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, nbytes,MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                                //printf(COLOR_RED"3. rank %d src %d  blocks[0]=%d   blocks[1]=%d"COLOR_RESET"\n",rank, src,  blocks[0], blocks[1]);fflush(stdout);
                                mpi_errno = MPIR_Type_indexed_impl(2, blocks, displs, recvtype, &tmp_type);
                                if (mpi_errno)
                                    MPIR_ERR_POP(mpi_errno);

                                mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                                if (mpi_errno)
                                    MPIR_ERR_POP(mpi_errno);

                                mpi_errno = MPIC_Recv(recvbuf, 1, tmp_type, src,
                                                      MPIR_GATHER_TAG, comm_ptr, &status, errflag);
                                if (mpi_errno)
                                {
                                    /* for communication errors, just record the error but continue */
                                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                                }

                                MPIR_Type_free_impl(&tmp_type);
                            }
                        }
                    }
                    else /* Intermediate nodes store in temporary buffer */
                    {
                        MPI_Aint offset;

                        /* Estimate the amount of data that is going to come in */
                        recvblks = mask;
                        relative_src = ((src - root) < 0) ? (src - root + comm_size) : (src - root);
                        if (relative_src + mask > comm_size)
                            recvblks -= (relative_src + mask - comm_size);

                        if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
                            offset = mask * nbytes;
                        else
                            offset = (mask - 1) * nbytes;

                        if (security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                        {
                            MPID_Get_node_id(comm_ptr, src, &src_node_id);
                            if (rank_node_id == src_node_id)
                            {
                                 mpi_errno = MPIC_Recv(((char *)tmp_buf + offset),
                                              recvblks * nbytes, MPI_BYTE, src,
                                              MPIR_GATHER_TAG, comm_ptr,
                                              &status, errflag);
                            }
                            else
                            {
                                int total_data = recvblks * nbytes; 
                                unsigned long count, max_out_len;
                                int next;
                                max_out_len = total_data;

                                // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 17 c=%d \n",init_rank,recvcount);
                                
                                mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                              (total_data + 28), MPI_BYTE, src,
                                              MPIR_GATHER_TAG, comm_ptr,
                                              &status, errflag);

                                // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 18 c=%d \n",init_rank,recvcount);
                                if (!EVP_AEAD_CTX_open(global_ctx, ((char *)tmp_buf + offset),
                                                           &count, max_out_len,
                                                           (ciphertext_recvbuf), 12,
                                                           (ciphertext_recvbuf + 12), (unsigned long)(total_data + 16),
                                                           NULL, 0))
                                    {
                                        printf(COLOR_RED"5. Decryption error:[rank %d src %d total_data %d rank_node_id %d src_node_id %d] gather recv"COLOR_RESET"\n", 
                                        rank, src, total_data, rank_node_id, src_node_id);
                                        fflush(stdout);
                                    }
                                    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 18-1 c=%d \n",init_rank,recvcount);
#if 0
                                    else
                                    {
                                        printf(COLOR_BLUE "inter-node. gather bio root recv:: [rank %d rank_node_id %d] Decrypted data size %lu" COLOR_RESET "\n",
                                        rank, rank_node_id, count);
                                        fflush(stdout);
                                    }
#endif              

                            }
                        }    
                        else
                        {    
                         //printf("4.[inter-nodes] rank %d root %d src %d mask %d recvblks %d nbytes %d\n", rank, root, src, mask, recvblks, nbytes);fflush(stdout);    
                          // printf("4. rank %d src %d recvblks %d\n",rank, src, recvblks);fflush(stdout);
                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 19 c=%d \n",init_rank,recvcount);
                        mpi_errno = MPIC_Recv(((char *)tmp_buf + offset),
                                              recvblks * nbytes, MPI_BYTE, src,
                                              MPIR_GATHER_TAG, comm_ptr,
                                              &status, errflag);
                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 20 c=%d \n",init_rank,recvcount);
                        }
                        if (mpi_errno)
                        {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                        curr_cnt += (recvblks * nbytes);
                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 20-1 c=%d \n",init_rank,recvcount);
                    }
                }
            }
            else
            {
                // unsigned char ciphertext_sendbuf[64 * 1024 *1024];
                unsigned char *ciphertext_sendbuf =large_send_buffer;
                // void *ciphertext_sendbuf;
                // MPIR_CHKLMEM_MALLOC(ciphertext_sendbuf, void *, recvcnt * (MPL_MAX(extent, true_extent)), mpi_errno, "ciphertext_sendbuf", MPL_MEM_BUFFER);
                dst = relative_rank ^ mask;
                dst = (dst + root) % comm_size;

                if (!tmp_buf_size)
                {
                    /* leaf nodes send directly from sendbuf */
                    // printf("5. rank %d root %d dst %d mask %d relative_rank %d sendcount %d \n", rank, root, dst, mask, relative_rank, sendcount);fflush(stdout);
                    if(security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                    {
                        MPID_Get_node_id(comm_ptr, dst, &dst_node_id); 

                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 21 c=%d \n",init_rank,recvcount);

                        if(rank_node_id == dst_node_id)
                        {
                            //ciphertext_sendbuf[0]='U';  
                             mpi_errno = MPIC_Send(sendbuf, sendcount, sendtype, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);    
                        }
                        else
                        {
                            unsigned long ciphertext_sendbuf_len, max_out_len;
                            max_out_len = sendtype_size*sendcount+16;
                            //ciphertext_sendbuf[0]='E';
                            RAND_bytes(ciphertext_sendbuf, 12);
                            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                               &ciphertext_sendbuf_len, max_out_len,
                                               ciphertext_sendbuf, 12,
                                               sendbuf, sendtype_size*sendcount,
                                               NULL, 0))
                            {
                                printf("Error in encryption: gather binomial\n");
                                fflush(stdout);
                            }
#if 0
                            else
                            {
                                printf("Gather binomial:: [rank %d] Encrypted data size %lu dst rank %d\n",
                                   rank, ciphertext_sendbuf_len, dst);
                            fflush(stdout);
                            }
#endif                            
                           //printf("--> rank %d sending to dst %d data %lu\n",rank,dst,ciphertext_sendbuf_len+12);fflush(stdout);

                           // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 22 c=%d \n",init_rank,recvcount);
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (sendtype_size*sendcount+28), MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);

                           // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 23 c=%d \n",init_rank,recvcount);
                        }
                        
                    }
                    else
                    { // unencrypted
                        mpi_errno = MPIC_Send(sendbuf, sendcount, sendtype, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);
                    }
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
                {
                   // printf("6. rank %d root %d dst %d mask %d relative_rank %d curr_cnt %d \n", rank, root, dst, mask, relative_rank, curr_cnt);fflush(stdout);
                    if(security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                    {
                        MPID_Get_node_id(comm_ptr, dst, &dst_node_id); 

                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 24 c=%d \n",init_rank,recvcount);

                        if(rank_node_id == dst_node_id)
                        {
                              mpi_errno = MPIC_Send(tmp_buf, curr_cnt, MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);   
                        }
                        else
                        {
                            unsigned long ciphertext_sendbuf_len, max_out_len;
                            max_out_len = curr_cnt+16;
                            //ciphertext_sendbuf[0]='E';
                            RAND_bytes(ciphertext_sendbuf, 12);
                            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                               &ciphertext_sendbuf_len, max_out_len,
                                               ciphertext_sendbuf, 12,
                                               tmp_buf, curr_cnt,
                                               NULL, 0))
                            {
                                printf("Error in encryption: gather binomial\n");
                                fflush(stdout);
                            }
#if 0
                            else
                            {
                                printf("Gather binomial < 1K:: [rank %d] Encrypted data size %lu dst rank %d\n",
                                   rank, ciphertext_sendbuf_len, dst);
                            fflush(stdout);
                            }
#endif                            
                           // printf("--> rank %d sending to dst %d data %lu\n",rank,dst,ciphertext_sendbuf_len+12);fflush(stdout);
                           // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 25 c=%d \n",init_rank,recvcount);
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (curr_cnt+28), MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);
                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 26 c=%d \n",init_rank,recvcount);
                        }
                        
                    }
                    else
                    { // unencrypted
                        mpi_errno = MPIC_Send(tmp_buf, curr_cnt, MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);
                    }
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                else
                {
                    if (security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                    {
                        MPID_Get_node_id(comm_ptr, dst, &dst_node_id);
                        if (rank_node_id == dst_node_id)
                        {
                            blocks[0] = sendcount;
                            struct_displs[0] = MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf;
                            types[0] = sendtype;
                            /* check for overflow.  work around int limits if needed*/
                            if (curr_cnt - nbytes != (int)(curr_cnt - nbytes))
                            {
                                blocks[1] = 1;
                                MPIR_Type_contiguous_x_impl(curr_cnt - nbytes,
                                                            MPI_BYTE, &(types[1]));
                            }
                            else
                            {
                                MPIR_Assign_trunc(blocks[1], curr_cnt - nbytes, int);
                                types[1] = MPI_BYTE;
                            }
                            struct_displs[1] = MPIR_VOID_PTR_CAST_TO_MPI_AINT tmp_buf;
                            mpi_errno = MPIR_Type_create_struct_impl(2, blocks, struct_displs, types, &tmp_type);
                            if (mpi_errno)
                                MPIR_ERR_POP(mpi_errno);

                            mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                            if (mpi_errno)
                                MPIR_ERR_POP(mpi_errno);
                            // printf(COLOR_YELLOW"7. rank %d root %d dst %d mask %d relative_rank %d 1 "COLOR_RESET"\n", rank, root, dst, mask, relative_rank);fflush(stdout);
                            mpi_errno = MPIC_Send(MPI_BOTTOM, 1, tmp_type, dst,
                                                  MPIR_GATHER_TAG, comm_ptr, errflag);
                            if (mpi_errno)
                            {
                                /* for communication errors, just record the error but continue */
                                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                            }
                            MPIR_Type_free_impl(&tmp_type);
                            if (types[1] != MPI_BYTE)
                                MPIR_Type_free_impl(&types[1]);
                        }
                        else
                        {
                            unsigned long ciphertext_sendbuf_len, max_out_len;

                            //ciphertext_sendbuf[0]='E';
                            int next = sendcount * sendtype_size;
                            memcpy(bio_temp_buffer, sendbuf, sendcount * sendtype_size);
                            memcpy(bio_temp_buffer + next, tmp_buf, curr_cnt - nbytes);
                            int total_data = sendcount * sendtype_size + curr_cnt - nbytes;
                            max_out_len = total_data + 16;
                            /*printf("  @@ rank %d dst %d total_data %d curr_cnt-nbytes %d curr_cnt %d nbytes %d\n",
                                   rank, dst, total_data, curr_cnt - nbytes, curr_cnt, nbytes);
                            fflush(stdout);*/
                            RAND_bytes(ciphertext_sendbuf, 12);
                            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                                   &ciphertext_sendbuf_len, max_out_len,
                                                   ciphertext_sendbuf, 12,
                                                   bio_temp_buffer, total_data,
                                                   NULL, 0))
                            {
                                printf("Error in encryption: gather binomial sendbuf+tempbuf\n");
                                fflush(stdout);
                            }
#if 0
                            else
                            {
                                printf("Gather binomial sendbuf+tmpbuf:: [rank %d] Encrypted data size %lu dst rank %d\n",
                                       rank, ciphertext_sendbuf_len, dst);
                                fflush(stdout);
                            }
#endif
                            //printf("--> rank %d sending to dst %d data %lu\n", rank, dst, ciphertext_sendbuf_len + 12);
                            //fflush(stdout);
                            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 27 c=%d \n",init_rank,recvcount);
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (total_data + 28), MPI_BYTE, dst,
                                                  MPIR_GATHER_TAG, comm_ptr, errflag);
                        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 28 c=%d \n",init_rank,recvcount);
                        }
                    }
                    else
                    { // unencrypted
                        blocks[0] = sendcount;
                        struct_displs[0] = MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf;
                        types[0] = sendtype;
                        /* check for overflow.  work around int limits if needed*/
                        if (curr_cnt - nbytes != (int)(curr_cnt - nbytes))
                        {
                            blocks[1] = 1;
                            MPIR_Type_contiguous_x_impl(curr_cnt - nbytes,
                                                        MPI_BYTE, &(types[1]));
                        }
                        else
                        {
                            MPIR_Assign_trunc(blocks[1], curr_cnt - nbytes, int);
                            types[1] = MPI_BYTE;
                        }
                        struct_displs[1] = MPIR_VOID_PTR_CAST_TO_MPI_AINT tmp_buf;
                        mpi_errno = MPIR_Type_create_struct_impl(2, blocks, struct_displs, types, &tmp_type);
                        if (mpi_errno)
                            MPIR_ERR_POP(mpi_errno);

                        mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                        if (mpi_errno)
                            MPIR_ERR_POP(mpi_errno);
                        // printf(COLOR_YELLOW"7. rank %d root %d dst %d mask %d relative_rank %d 1 "COLOR_RESET"\n", rank, root, dst, mask, relative_rank);fflush(stdout);
                        mpi_errno = MPIC_Send(MPI_BOTTOM, 1, tmp_type, dst,
                                              MPIR_GATHER_TAG, comm_ptr, errflag);
                        if (mpi_errno)
                        {
                            /* for communication errors, just record the error but continue */
                            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                        MPIR_Type_free_impl(&tmp_type);
                        if (types[1] != MPI_BYTE)
                            MPIR_Type_free_impl(&types[1]);
                    }
                }

                break;
            }
            mask <<= 1;
        }

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 29-0 c=%d \n",init_rank,recvcount);

        if ((rank == root) && root && (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE) && copy_blks)
        {
            /* reorder and copy from tmp_buf into recvbuf */
            mpi_errno = MPIR_Localcopy(tmp_buf,
                                       nbytes * (comm_size - copy_offset), MPI_BYTE,
                                       ((char *)recvbuf + extent * recvcount * copy_offset),
                                       recvcount * (comm_size - copy_offset), recvtype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            mpi_errno = MPIR_Localcopy((char *)tmp_buf + nbytes * (comm_size - copy_offset),
                                       nbytes * (copy_blks - comm_size + copy_offset), MPI_BYTE,
                                       recvbuf,
                                       recvcount * (copy_blks - comm_size + copy_offset), recvtype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }
// if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_intra 29 c=%d \n",init_rank,recvcount);

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
#define FUNCNAME MPIR_Gather_MV2_Direct_CHS
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_CHS(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{

// host_checking_comm(comm_ptr->handle);
if (comm_ptr->is_blocked==NULL || comm_ptr->is_uniform==NULL)
    host_checking_comm(comm_ptr->handle, &comm_ptr->node_sizes,NULL,NULL);

if (comm_ptr->is_blocked != 1 || root!=0 || comm_ptr->is_uniform != 1){
#if GATHER_PRINT_FUN
   if (PRINT_FUN_NAME ){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Gather rank = %d host = %s count = %d  root = %d  is_blocked = %d  is_uniform = %d] Redirect to...\n", comm_ptr->rank,hostname,recvcount,root,comm_ptr->is_blocked,comm_ptr->is_uniform);fflush(stdout);
	}
#endif
        int ret = MPIR_Gather_intra(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, root,comm_ptr, errflag);
        return ret;
    }

#if GATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Gather rank = %d host = %s count = %d SA=%d] Func: MPIR_Gather_MV2_Direct_CHS\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    MPI_Status status;
    int rank, comm_size; 
    int recvtype_size = 0;
    int tmp_buf_size = 0;
    int sendcnt = sendcount;
    int recvcnt = recvcount;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    unsigned char *ciphertext_sendbuf = large_send_buffer;
    unsigned char *ciphertext_recvbuf = large_recv_buffer;
    void *tmp_buf = NULL;

    MPIR_CHKLMEM_DECL(3); 
    MPIR_Request **reqarray;
    MPI_Status *starray;
    //// if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] MPIR_Gather_MV2_Direct_CHS 01 c=%d \n",init_rank,recvcount);
    //MPIR_Request reqarray[200][200];
    //if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] MPIR_Gather_MV2_Direct_CHS 01-1 c=%d \n",init_rank,recvcount);
    //MPI_Status starray[200];
    //if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] MPIR_Gather_MV2_Direct_CHS 01-2 c=%d \n",init_rank,recvcount);

    MPI_Comm conc_comm;
    MPIR_Comm *shmem_commptr, *conc_commptr = NULL;
    int rank_node_id, root_node_id;

    int local_size, local_rank, conc_rank, conc_size;
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 01 c=%d \n",init_rank,recvcount);    

    shmem_commptr = comm_ptr->node_comm;

    // conc_comm = comm_ptr->concurrent_comm;   

         if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather-CHS 01-3 c=%d \n",init_rank,recvcount);
    //conc_comm = comm_ptr->dev.ch.concurrent_comm;
    conc_comm = comm_ptr->concurrent_comm;    
    

    // if (conc_comm == NULL) {
    //      fprintf(stderr,"[%d] NULL conc_comm c=%d\n",init_rank,sendcnt);
    //     mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
    //     if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather-CHS 01-4 c=%d \n",init_rank,recvcount);
    //     conc_comm = comm_ptr->concurrent_comm;    
    // } 
    if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather-CHS 01-5 c=%d \n",init_rank,recvcount);


    MPIR_Comm_get_ptr(conc_comm, conc_commptr);  

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 01-1 c=%d \n",init_rank,recvcount);    

    local_rank = comm_ptr->node_comm->rank;
    local_size = comm_ptr->node_comm->local_size;

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    rank = comm_ptr->rank;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);

    comm_size = comm_ptr->local_size;

    MPIR_CHKLMEM_MALLOC(reqarray, MPIR_Request **, comm_size * sizeof (MPIR_Request *), mpi_errno, "reqarray",MPL_MEM_BUFFER);
    MPIR_CHKLMEM_MALLOC(starray, MPI_Status *, comm_size * sizeof (MPI_Status), mpi_errno, "starray", MPL_MEM_BUFFER);

    int next = 0;
    int dest = 0;
    int src_global_rank = 0;
    int j = 0;
    
    //if (rank_node_id == root_node_id && local_size != comm_size)
    if (rank_node_id == root_node_id)
    {

        int t = (recvtype_extent * recvcount);
        unsigned long max_out_len = (unsigned long)(t);
        unsigned long count;

        //int each_rank_size = comm_size / local_size; // consider if equal now
        //int max_ranks = local_rank * each_rank_size + each_rank_size;
        // printf("rank %d each_rank_size %d max_ranks %d\n", rank, each_rank_size, max_ranks);fflush(stdout);
        dest = 0;
        reqs = 1;

       // if(rank == 0 && count_ite >= 16)
        //    printf("1. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);
        /* receiver side */
        next = 0;
         src_global_rank = rank + local_size;
         // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 02 c=%d \n",init_rank,recvcount);    
        for (i = 1; i < conc_size; i++)
        {
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 02-1 c=%d i=%d\n",init_rank,recvcount,i);    
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t + 28, MPI_BYTE, i, MPIR_GATHER_TAG, conc_commptr, &reqarray[reqs++]);
            // mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t + 28, MPI_BYTE, src_global_rank, MPIR_GATHER_TAG, comm_ptr, &reqarray[reqs++]);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 03-1 c=%d i=%d\n",init_rank,recvcount,i);    
            
            next += (t + 28);
             src_global_rank += local_size;
        }
        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 03 c=%d \n",init_rank,recvcount);    
       // if(rank == 0 && count_ite >= 16)
         //  printf("2. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);

        /* receiver side */
        next = 0;
        src_global_rank = rank + local_size;
        for (i = 1; i < conc_size; i++)
        {
            dest = src_global_rank * t; // get the decryption location

            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 04 c=%d i=%d\n",init_rank,recvcount,i);    

            mpi_errno = MPIC_Waitall(1, &reqarray[i], starray, errflag);
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 04-1 c=%d i=%d\n",init_rank,recvcount,i);    
            if (!EVP_AEAD_CTX_open(global_ctx, shmem_buffer + dest,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf + next), 12,
                                   (ciphertext_recvbuf + next + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error gather CHS:[rank %d rank_node_id %d root_node_id %d] gather direct recv i %d\n", rank, rank_node_id,root_node_id,i);
                fflush(stdout);
            }

            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 05 c=%d i=%d\n",init_rank,recvcount,i);    
            next += (t + 28);
            src_global_rank += local_size;
        }

        // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 06 c=%d \n",init_rank,recvcount);  

        /* itner-node recving is done, now copy own data */
        next = rank * t;
        memcpy(shmem_buffer + next, sendbuf, t);

        /* wait for all local ranks to finish decryption */
        mpi_errno = MPIR_Barrier_impl(shmem_commptr, errflag); 

        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
        //printf("rank %d now copy data\n",rank);fflush(stdout);
        /* now copy data to recvbuf */
        if(rank == root)
        {
           memcpy(recvbuf, shmem_buffer, recvcount * recvtype_extent * comm_size);

        }    
    }
    //printf("we are here\n");fflush(stdout);
    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 07 c=%d \n",init_rank,recvcount);  

    /* sender side*/
    if (local_size != comm_size)
    {
       
        int t = (sendtype_extent * sendcount);
        unsigned long max_out_len = (unsigned long)(sendtype_extent * sendcount+16);
        unsigned long ciphertext_sendbuf_len;
        int global_dest_rank = rank % local_size;

        // if(rank == 0 && count_ite >= 16)
         //   printf("3. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);
        if(rank_node_id != root_node_id)
        {
           
             RAND_bytes(ciphertext_sendbuf, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf, 12,
                                   sendbuf, t,
                                   NULL, 0))
            {
                printf("Error in encryption Gather CHS: [rank %d rank_node_id %d] gather  global_dest_rank %d conc_rank %d t %d\n",
                       rank, rank_node_id, global_dest_rank, conc_rank, t);
                fflush(stdout);
            }
#if 0
            else
            {

                printf(COLOR_GREEN "Gather CHS:: [rank %d local_rank %d rank_node_id %d root_node_id %d global_dest_rank %d] Encrypted data size %lu conc_rank %d" COLOR_RESET "\n",
                       rank, local_rank, rank_node_id, root_node_id, global_dest_rank,  ciphertext_sendbuf_len, conc_rank);
                fflush(stdout);
            }
#endif

               // printf(COLOR_MAGENTA"CHS gather [rank %d conc_rank %d] sending to dest_rank %d dest conc rank %d"COLOR_RESET"\n",rank, conc_rank, global_dest_rank,0);fflush(stdout);            
            // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 08 c=%d \n",init_rank,recvcount);  
            mpi_errno = MPIC_Send(ciphertext_sendbuf,
                                    (t + 28),
                                    MPI_BYTE, 0, MPIR_GATHER_TAG, conc_commptr,
                                    errflag);
                   // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 09 c=%d \n",init_rank,recvcount);  
                   /*mpi_errno = MPIC_Send(ciphertext_sendbuf,
                                           (t + 28),
                                           MPI_BYTE, global_dest_rank, MPIR_GATHER_TAG, comm_ptr,
                                            errflag);*/                        
                                          
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
          
        }
    }
    else
    { // un-uniform not handled
        printf("CHS gather un-uniform case not handled yet\n");fflush(stdout);
    }

    // if (DEBUG_INIT_FILE)  fprintf(stderr,"[%d] Gather_CHS 10 c=%d \n",init_rank,recvcount);  


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
#define FUNCNAME MPIR_Gather_intra_binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_intra_binomial(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if GATHER_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
		char hostname[100];
		gethostname(hostname, MAX_HOSTNAME_LEN);
		printf("[Gather rank = %d host = %s count = %d SA=%d] Func: MPIR_Gather_intra_binomial\n", comm_ptr->rank,hostname,recvcount,security_approach);fflush(stdout);
	}
#endif      
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int relative_rank;
    int mask, src, dst, relative_src;
    MPI_Aint curr_cnt = 0, nbytes, sendtype_size, recvtype_size;
    int recvblks;
    int missing;
    MPI_Aint tmp_buf_size;
    void *tmp_buf = NULL;
    MPI_Status status;
    MPI_Aint extent = 0;        /* Datatype extent */
    int blocks[2];
    int displs[2];
    MPI_Aint struct_displs[2];
    MPI_Datatype types[2], tmp_type;
    int copy_offset = 0, copy_blks = 0;
    MPIR_CHKLMEM_DECL(1);


    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcount == 0)) || ((rank != root) && (sendcount == 0)))
        return MPI_SUCCESS;

    /* Use binomial tree algorithm. */

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    if (rank == root) {
        MPIR_Datatype_get_extent_macro(recvtype, extent);
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                         (extent * recvcount * comm_size));
    }

    if (rank == root) {
        MPIR_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvtype_size * recvcount;
    } else {
        MPIR_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendtype_size * sendcount;
    }

    /* Find the number of missing nodes in my sub-tree compared to
     * a balanced tree */
    for (mask = 1; mask < comm_size; mask <<= 1);
    --mask;
    while (relative_rank & mask)
        mask >>= 1;
    missing = (relative_rank | mask) - comm_size + 1;
    if (missing < 0)
        missing = 0;
    tmp_buf_size = (mask - missing);

    /* If the message is smaller than the threshold, we will copy
     * our message in there too */
    if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
        tmp_buf_size++;

    tmp_buf_size *= nbytes;

    /* For zero-ranked root, we don't need any temporary buffer */
    if ((rank == root) && (!root || (nbytes >= MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)))
        tmp_buf_size = 0;

    if (tmp_buf_size) {
        MPIR_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf", MPL_MEM_BUFFER);
    }

    if (rank == root) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype,
                                       ((char *) recvbuf + extent * recvcount * rank),
                                       recvcount, recvtype);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    } else if (tmp_buf_size && (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)) {
        /* copy from sendbuf into tmp_buf */
        mpi_errno = MPIR_Localcopy(sendbuf, sendcount, sendtype, tmp_buf, nbytes, MPI_BYTE);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    curr_cnt = nbytes;

    mask = 0x1;
    while (mask < comm_size) {
        if ((mask & relative_rank) == 0) {
            src = relative_rank | mask;
            if (src < comm_size) {
                src = (src + root) % comm_size;

                if (rank == root) {
                    recvblks = mask;
                    if ((2 * recvblks) > comm_size)
                        recvblks = comm_size - recvblks;

                    if ((rank + mask + recvblks == comm_size) ||
                        (((rank + mask) % comm_size) < ((rank + mask + recvblks) % comm_size))) {
                        /* If the data contiguously fits into the
                         * receive buffer, place it directly. This
                         * should cover the case where the root is
                         * rank 0. */
                        mpi_errno = MPIC_Recv(((char *) recvbuf +
                                               (((rank +
                                                  mask) % comm_size) * (MPI_Aint) recvcount *
                                                extent)), (MPI_Aint) recvblks * recvcount,
                                              recvtype, src, MPIR_GATHER_TAG, comm_ptr, &status,
                                              errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag =
                                MPIX_ERR_PROC_FAILED ==
                                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED :
                                MPIR_ERR_OTHER;
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                    } else if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE) {
                        /* small transfer size case. cast ok */
                        MPIR_Assert(recvblks * nbytes == (int) (recvblks * nbytes));
                        mpi_errno = MPIC_Recv(tmp_buf, (int) (recvblks * nbytes),
                                              MPI_BYTE, src, MPIR_GATHER_TAG,
                                              comm_ptr, &status, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag =
                                MPIX_ERR_PROC_FAILED ==
                                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED :
                                MPIR_ERR_OTHER;
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }
                        copy_offset = rank + mask;
                        copy_blks = recvblks;
                    } else {
                        blocks[0] = recvcount * (comm_size - root - mask);
                        displs[0] = recvcount * (root + mask);
                        blocks[1] = (recvcount * recvblks) - blocks[0];
                        displs[1] = 0;

                        mpi_errno = MPIR_Type_indexed_impl(2, blocks, displs, recvtype, &tmp_type);
                        if (mpi_errno)
                            MPIR_ERR_POP(mpi_errno);

                        mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                        if (mpi_errno)
                            MPIR_ERR_POP(mpi_errno);

                        mpi_errno = MPIC_Recv(recvbuf, 1, tmp_type, src,
                                              MPIR_GATHER_TAG, comm_ptr, &status, errflag);
                        if (mpi_errno) {
                            /* for communication errors, just record the error but continue */
                            *errflag =
                                MPIX_ERR_PROC_FAILED ==
                                MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED :
                                MPIR_ERR_OTHER;
                            MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        }

                        MPIR_Type_free_impl(&tmp_type);
                    }
                } else {        /* Intermediate nodes store in temporary buffer */

                    MPI_Aint offset;

                    /* Estimate the amount of data that is going to come in */
                    recvblks = mask;
                    relative_src = ((src - root) < 0) ? (src - root + comm_size) : (src - root);
                    if (relative_src + mask > comm_size)
                        recvblks -= (relative_src + mask - comm_size);

                    if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)
                        offset = mask * nbytes;
                    else
                        offset = (mask - 1) * nbytes;
                    mpi_errno = MPIC_Recv(((char *) tmp_buf + offset),
                                          recvblks * nbytes, MPI_BYTE, src,
                                          MPIR_GATHER_TAG, comm_ptr, &status, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag =
                            MPIX_ERR_PROC_FAILED ==
                            MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                    curr_cnt += (recvblks * nbytes);
                }
            }
        } else {
            dst = relative_rank ^ mask;
            dst = (dst + root) % comm_size;

            if (!tmp_buf_size) {
                /* leaf nodes send directly from sendbuf */
                mpi_errno = MPIC_Send(sendbuf, sendcount, sendtype, dst,
                                      MPIR_GATHER_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            } else if (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE) {
                mpi_errno = MPIC_Send(tmp_buf, curr_cnt, MPI_BYTE, dst,
                                      MPIR_GATHER_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            } else {
                blocks[0] = sendcount;
                struct_displs[0] = MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf;
                types[0] = sendtype;
                /* check for overflow.  work around int limits if needed */
                if (curr_cnt - nbytes != (int) (curr_cnt - nbytes)) {
                    blocks[1] = 1;
                    MPIR_Type_contiguous_x_impl(curr_cnt - nbytes, MPI_BYTE, &(types[1]));
                } else {
                    MPIR_Assign_trunc(blocks[1], curr_cnt - nbytes, int);
                    types[1] = MPI_BYTE;
                }
                struct_displs[1] = MPIR_VOID_PTR_CAST_TO_MPI_AINT tmp_buf;
                mpi_errno =
                    MPIR_Type_create_struct_impl(2, blocks, struct_displs, types, &tmp_type);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                mpi_errno = MPIR_Type_commit_impl(&tmp_type);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                mpi_errno = MPIC_Send(MPI_BOTTOM, 1, tmp_type, dst,
                                      MPIR_GATHER_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag =
                        MPIX_ERR_PROC_FAILED ==
                        MPIR_ERR_GET_CLASS(mpi_errno) ? MPIR_ERR_PROC_FAILED : MPIR_ERR_OTHER;
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                MPIR_Type_free_impl(&tmp_type);
                if (types[1] != MPI_BYTE)
                    MPIR_Type_free_impl(&types[1]);
            }

            break;
        }
        mask <<= 1;
    }

    if ((rank == root) && root && (nbytes < MPIR_CVAR_GATHER_VSMALL_MSG_SIZE) && copy_blks) {
        /* reorder and copy from tmp_buf into recvbuf */
        mpi_errno = MPIR_Localcopy(tmp_buf,
                                   nbytes * (comm_size - copy_offset), MPI_BYTE,
                                   ((char *) recvbuf + extent * recvcount * copy_offset),
                                   recvcount * (comm_size - copy_offset), recvtype);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        mpi_errno = MPIR_Localcopy((char *) tmp_buf + nbytes * (comm_size - copy_offset),
                                   nbytes * (copy_blks - comm_size + copy_offset), MPI_BYTE,
                                   recvbuf,
                                   recvcount * (copy_blks - comm_size + copy_offset), recvtype);
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
    goto fn_exit;
}
