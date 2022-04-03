/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include "mpiimpl.h"
#ifdef _OSU_MVAPICH_
#   include "coll_shmem.h"
#endif /* _OSU_MVAPICH_ */

/* added by abu naser */
#include "secure_allgather.h"
unsigned char bio_temp_buffer[8388608*8]; // 8MB*8 temp buffer
/* end of add */

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

    - name        : MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE
      category    : COLLECTIVE
      type        : int
      default     : 2048
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        use the short message algorithm for intercommunicator MPI_Gather if the
        send buffer size is < this value (in bytes)
        (See also: MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* -- Begin Profiling Symbol Block for routine MPI_Gather */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Gather = PMPI_Gather
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Gather  MPI_Gather
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Gather as PMPI_Gather
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
               int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
               __attribute__((weak,alias("PMPI_Gather")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Gather
#define MPI_Gather PMPI_Gather
/* This is the default implementation of gather. The algorithm is:
   
   Algorithm: MPI_Gather

   We use a binomial tree algorithm for both short and long
   messages. At nodes other than leaf nodes we need to allocate a
   temporary buffer to store the incoming message. If the root is not
   rank 0, for very small messages, we pack it into a temporary
   contiguous buffer and reorder it to be placed in the right
   order. For small (but not very small) messages, we use a derived
   datatype to unpack the incoming data into non-contiguous buffers in
   the right order. In the heterogeneous case we first pack the
   buffers by using MPI_Pack and then do the gather.

   Cost = lgp.alpha + n.((p-1)/p).beta
   where n is the total size of the data gathered at the root.

   Possible improvements: 

   End Algorithm: MPI_Gather
*/

/* not declared static because it is called in intercomm. allgather */

#undef FUNCNAME
#define FUNCNAME MPIR_Gather_intra
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_intra(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather rank = %d host = %s] Func: MPIR_Gather_intra\n", comm_ptr->rank, hostname);
    fflush(stdout);
#endif
#if 0
     if (comm_ptr->dev.ch.is_uniform == 1 && security_approach > 302 && security_approach <= 305)
    {
        if (security_approach == 303) // call shared memory implementation
        {
            int ret = MPIR_Gather_MV2_Direct_CHS(sendbuf, sendcount, sendtype,
                                                 recvbuf, recvcount, recvtype,
                                                 root, comm_ptr, errflag);
            return ret;
        }
        else if (security_approach == 305) // call shared memory implementation
        {
            int ret = MPIR_Gather_MV2_Direct_CHS_UNENCRYPTED(sendbuf, sendcount, sendtype,
                                                             recvbuf, recvcount, recvtype,
                                                             root, comm_ptr, errflag);
            return ret;
        }
         else if(security_approach == 304) // call shared memory implementation 
        {
        int ret = MPIR_Gather_MV2_Direct_CNoSM_RR(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype,
                                      root, comm_ptr, errflag);
        return ret;
        } 
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
    MPIU_CHKLMEM_DECL(1);

#ifdef MPID_HAS_HETERO
    int position, recv_size;
#endif

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcount == 0)) ||
        ((rank != root) && (sendcount == 0)))
        return MPI_SUCCESS;

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* Use binomial tree algorithm. */

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    if (rank == root)
    {
        MPID_Datatype_get_extent_macro(recvtype, extent);
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                         (extent * recvcount * comm_size));
    }

    /* added by abu naser */
     MPID_Node_id_t rank_node_id, root_node_id, dst_node_id, src_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id); 
    /* end of add */

    if (is_homogeneous)
    {

        /* communicator is homogeneous. no need to pack buffer. */

        if (rank == root)
        {
            MPID_Datatype_get_size_macro(recvtype, recvtype_size);
            nbytes = recvtype_size * recvcount;
        }
        else
        {
            MPID_Datatype_get_size_macro(sendtype, sendtype_size);
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

        /* For zero-ranked root, we don't need any temporary buffer */
        if ((rank == root) && (!root || (nbytes >= MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)))
            tmp_buf_size = 0;

        if (tmp_buf_size)
        {
            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");
        }

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

        mask = 0x1;
        while (mask < comm_size)
        {
            if ((mask & relative_rank) == 0)
            {
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
                                    int total_data = recvblks * recvcount * recvtype_size;
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                                          (total_data+28), MPI_BYTE, src,
                                                          MPIR_GATHER_TAG, comm_ptr,
                                                          &status, errflag);

                                    unsigned long count, max_out_len;
                                    int next;
                                    max_out_len = total_data;
                                    next =  (((rank + mask) % comm_size) * (MPI_Aint)recvcount * extent);
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
                                    MPIU_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
                                    //printf("2 . rank %d root %d src %d mask %d recvblks %d nbytes %d MPIR_CVAR_GATHER_VSMALL_MSG_SIZE %d\n", rank, root, src, mask, recvblks, nbytes,MPIR_CVAR_GATHER_VSMALL_MSG_SIZE);fflush(stdout);
                                    // printf("2. rank %d src %d recvblks %d\n",rank, src, recvblks);fflush(stdout);
                                    mpi_errno = MPIC_Recv(tmp_buf, (int)(recvblks * nbytes),
                                                          MPI_BYTE, src, MPIR_GATHER_TAG,
                                                          comm_ptr, &status, errflag);
                                }
                                else
                                {
                                    MPIU_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
                                    int total_data = recvblks * nbytes;
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf, (total_data+28),
                                                          MPI_BYTE, src, MPIR_GATHER_TAG,
                                                          comm_ptr, &status, errflag);
                                    
                                    unsigned long count, max_out_len;
                                    int next;
                                    max_out_len = total_data;
                                   
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
                                MPIU_Assert(recvblks * nbytes == (int)(recvblks * nbytes));
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
                                else
                                {
                                    // receive encrypted data and unpack it
                                    int total_data = (recvcount * (comm_size - root - mask)) *recvtype_size + (recvcount * recvblks)*recvtype_size;
                                    int temp_data = (recvcount * (comm_size - root - mask)) * recvtype_size;
                                    total_data = total_data - temp_data;
                                    //printf("3. total data %d\n",total_data);fflush(stdout);
                                    
                                    mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                                          (total_data+28), MPI_BYTE, src,
                                                          MPIR_GATHER_TAG, comm_ptr,
                                                          &status, errflag);

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
                                
                                mpi_errno = MPIC_Recv(ciphertext_recvbuf,
                                              (total_data + 28), MPI_BYTE, src,
                                              MPIR_GATHER_TAG, comm_ptr,
                                              &status, errflag);

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
                        mpi_errno = MPIC_Recv(((char *)tmp_buf + offset),
                                              recvblks * nbytes, MPI_BYTE, src,
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
                        curr_cnt += (recvblks * nbytes);
                    }
                }
            }
            else
            {
                dst = relative_rank ^ mask;
                dst = (dst + root) % comm_size;

                if (!tmp_buf_size)
                {
                    /* leaf nodes send directly from sendbuf */
                    // printf("5. rank %d root %d dst %d mask %d relative_rank %d sendcount %d \n", rank, root, dst, mask, relative_rank, sendcount);fflush(stdout);
                    if(security_approach == 302  || (security_approach > 302 && security_approach <= 305))
                    {
                        MPID_Get_node_id(comm_ptr, dst, &dst_node_id); 

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
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (sendtype_size*sendcount+28), MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);
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
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (curr_cnt+28), MPI_BYTE, dst,
                                          MPIR_GATHER_TAG, comm_ptr, errflag);
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
                            struct_displs[0] = MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf;
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
                                MPIU_Assign_trunc(blocks[1], curr_cnt - nbytes, int);
                                types[1] = MPI_BYTE;
                            }
                            struct_displs[1] = MPIU_VOID_PTR_CAST_TO_MPI_AINT tmp_buf;
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
                            mpi_errno = MPIC_Send(ciphertext_sendbuf, (total_data + 28), MPI_BYTE, dst,
                                                  MPIR_GATHER_TAG, comm_ptr, errflag);
                        }
                    }
                    else
                    { // unencrypted
                        blocks[0] = sendcount;
                        struct_displs[0] = MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf;
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
                            MPIU_Assign_trunc(blocks[1], curr_cnt - nbytes, int);
                            types[1] = MPI_BYTE;
                        }
                        struct_displs[1] = MPIU_VOID_PTR_CAST_TO_MPI_AINT tmp_buf;
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

#ifdef MPID_HAS_HETERO
    else
    { /* communicator is heterogeneous. pack data into tmp_buf. */
        if (rank == root)
            MPIR_Pack_size_impl(recvcount * comm_size, recvtype, &tmp_buf_size);
        else
            MPIR_Pack_size_impl(sendcount * (comm_size / 2), sendtype, &tmp_buf_size);

        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");

        position = 0;
        if (sendbuf != MPI_IN_PLACE)
        {
            mpi_errno = MPIR_Pack_impl(sendbuf, sendcount, sendtype, tmp_buf,
                                       tmp_buf_size, &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            nbytes = position;
        }
        else
        {
            /* do a dummy pack just to calculate nbytes */
            mpi_errno = MPIR_Pack_impl(recvbuf, 1, recvtype, tmp_buf,
                                       tmp_buf_size, &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            nbytes = position * recvcount;
        }

        curr_cnt = nbytes;

        mask = 0x1;
        while (mask < comm_size)
        {
            if ((mask & relative_rank) == 0)
            {
                src = relative_rank | mask;
                if (src < comm_size)
                {
                    src = (src + root) % comm_size;
                    mpi_errno = MPIC_Recv(((char *)tmp_buf + curr_cnt),
                                          tmp_buf_size - curr_cnt, MPI_BYTE, src,
                                          MPIR_GATHER_TAG, comm_ptr,
                                          &status, errflag);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                        recv_size = 0;
                    }
                    else
                        /* the recv size is larger than what may be sent in
                           some cases. query amount of data actually received */
                        MPIR_Get_count_impl(&status, MPI_BYTE, &recv_size);
                    curr_cnt += recv_size;
                }
            }
            else
            {
                dst = relative_rank ^ mask;
                dst = (dst + root) % comm_size;
                mpi_errno = MPIC_Send(tmp_buf, curr_cnt, MPI_BYTE, dst,
                                      MPIR_GATHER_TAG, comm_ptr, errflag);
                if (mpi_errno)
                {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                break;
            }
            mask <<= 1;
        }

        if (rank == root)
        {
            /* reorder and copy from tmp_buf into recvbuf */
            if (sendbuf != MPI_IN_PLACE)
            {
                position = 0;
                mpi_errno = MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position,
                                             ((char *)recvbuf + extent * recvcount * rank),
                                             recvcount * (comm_size - rank), recvtype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
            else
            {
                position = nbytes;
                mpi_errno = MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position,
                                             ((char *)recvbuf + extent * recvcount * (rank + 1)),
                                             recvcount * (comm_size - rank - 1), recvtype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
            if (root != 0)
            {
                mpi_errno = MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position, recvbuf,
                                             recvcount * rank, recvtype);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);
            }
        }
    }
#endif /* MPID_HAS_HETERO */

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
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
#define FUNCNAME MPIR_Gather_inter
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_inter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather rank = %d host = %s] Func: MPIR_Gather_inter\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
/*  Intercommunicator gather.
    For short messages, remote group does a local intracommunicator
    gather to rank 0. Rank 0 then sends data to root.

    Cost: (lgp+1).alpha + n.((p-1)/p).beta + n.beta
   
    For long messages, we use linear gather to avoid the extra n.beta.

    Cost: p.alpha + n.beta
*/

    int rank, local_size, remote_size, mpi_errno=MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i;
    MPI_Aint nbytes, sendtype_size, recvtype_size;
    MPI_Status status;
    MPI_Aint extent, true_extent, true_lb = 0;
    void *tmp_buf=NULL;
    MPID_Comm *newcomm_ptr = NULL;
    MPIU_CHKLMEM_DECL(1);

    if (root == MPI_PROC_NULL)
    {
        /* local processes other than root do nothing */
        return MPI_SUCCESS;
    }

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER( comm_ptr );

    remote_size = comm_ptr->remote_size; 
    local_size = comm_ptr->local_size; 

    if (root == MPI_ROOT)
    {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvtype_size * recvcount * remote_size;
    }
    else
    {
        /* remote side */
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendtype_size * sendcount * local_size;
    }

    if (nbytes < MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE)
    {
        if (root == MPI_ROOT)
	{
            /* root receives data from rank 0 on remote group */
            mpi_errno = MPIC_Recv(recvbuf, recvcount*remote_size,
                                     recvtype, 0, MPIR_GATHER_TAG, comm_ptr,
                                     &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
        else
	{
            /* remote group. Rank 0 allocates temporary buffer, does
               local intracommunicator gather, and then sends the data
               to root. */
            
            rank = comm_ptr->rank;
            
            if (rank == 0)
	    {
                MPIR_Type_get_true_extent_impl(sendtype, &true_lb, &true_extent);
                MPID_Datatype_get_extent_macro(sendtype, extent);
 
		MPIU_Ensure_Aint_fits_in_pointer(sendcount*local_size*
						 (MPIR_MAX(extent, true_extent)));
                MPIU_CHKLMEM_MALLOC(tmp_buf, void *, sendcount*local_size*(MPIR_MAX(extent,true_extent)), mpi_errno, "tmp_buf");
                /* adjust for potential negative lower bound in datatype */
                tmp_buf = (void *)((char*)tmp_buf - true_lb);
            }
            
            /* all processes in remote group form new intracommunicator */
            if (!comm_ptr->local_comm) {
                mpi_errno = MPIR_Setup_intercomm_localcomm( comm_ptr );
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }

            newcomm_ptr = comm_ptr->local_comm;

            /* now do the a local gather on this intracommunicator */
            mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype,
                                         tmp_buf, sendcount, sendtype, 0,
                                         newcomm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            
            if (rank == 0)
	    {
                mpi_errno = MPIC_Send(tmp_buf, sendcount*local_size,
                                         sendtype, root,
                                         MPIR_GATHER_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
    }
    else
    {
        /* long message. use linear algorithm. */
        if (root == MPI_ROOT)
	{
            MPID_Datatype_get_extent_macro(recvtype, extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
					     (recvcount*remote_size*extent));

            for (i=0; i<remote_size; i++)
	    {
                mpi_errno = MPIC_Recv(((char *)recvbuf+recvcount*i*extent),
                                         recvcount, recvtype, i,
                                         MPIR_GATHER_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        }
        else
	{
            mpi_errno = MPIC_Send(sendbuf,sendcount,sendtype,root,
                                     MPIR_GATHER_TAG,comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }

 fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}


/* MPIR_Gather performs an gather using point-to-point messages.  This
   is intended to be used by device-specific implementations of
   gather.  In all other cases MPIR_Gather_impl should be used. */
#undef FUNCNAME
#define FUNCNAME MPIR_Gather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
        
    if (comm_ptr->comm_kind == MPID_INTRACOMM) {
        /* intracommunicator */
        mpi_errno = MPIR_Gather_intra(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype, root,
                                      comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    } else {
        /* intercommunicator */
        mpi_errno = MPIR_Gather_inter(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype, root,
                                      comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

 fn_exit:
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

/* MPIR_Gather_impl should be called by any internal component that
   would otherwise call MPI_Gather.  This differs from MPIR_Gather in
   that this will call the coll_fns version if it exists.  This
   function replaces NMPI_Gather. */
#undef FUNCNAME
#define FUNCNAME MPIR_Gather_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_impl(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->coll_fns != NULL && comm_ptr->coll_fns->Gather != NULL) {
	/* --BEGIN USEREXTENSION-- */
	mpi_errno = comm_ptr->coll_fns->Gather(sendbuf, sendcount,
                                               sendtype, recvbuf, recvcount,
                                               recvtype, root, comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	/* --END USEREXTENSION-- */
    } else {
        mpi_errno = MPIR_Gather(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, root,
                                comm_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

 fn_exit:
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}
#endif

/* added by abu naser */
#undef FUNCNAME
#define FUNCNAME MPIR_Naive_Sec_Gather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Naive_Sec_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype,
                          int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int sendtype_sz, recvtype_sz;
    unsigned long ciphertext_sendbuf_len = 0;
    sendtype_sz = recvtype_sz = 0;
    int var;
    unsigned long count;
    var = MPI_Type_size(sendtype, &sendtype_sz);
    var = MPI_Type_size(recvtype, &recvtype_sz);
    int t = sendcount * sendtype_sz;
    unsigned long max_out_len = t + 16;
    int rank, i, next, dest;

    rank = comm_ptr->rank;

    if (rank == root)
    {
      /*  RAND_bytes(ciphertext_sendbuf, 12);
        if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                               &ciphertext_sendbuf_len, max_out_len,
                               ciphertext_sendbuf, 12,
                               sendbuf, t,
                               NULL, 0))
        {
            printf("Error in encryption: naive gather\n");
            fflush(stdout);
        }
#if 1
        else
        {
            printf(COLOR_GREEN "Gather:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
            fflush(stdout);
        }
#endif
    */
        //mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_recvbuf, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
          mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
           
       /* dest = 0;
        for (i = 0; i < comm_ptr->local_size; i++)
        {
            next = i * (recvcount * recvtype_sz + 28);
            if (!EVP_AEAD_CTX_open(global_ctx, recvbuf + dest,
                                   &count, (unsigned long)((recvcount * recvtype_sz)),
                                   (ciphertext_recvbuf + next), 12,
                                   (ciphertext_recvbuf + next + 12), (unsigned long)((recvcount * recvtype_sz) + 16),
                                   NULL, 0))
            {
                printf("Decryption error: naive gather\n");
                fflush(stdout);
            }
#if 1
            else
            {
                printf(COLOR_BLUE "Gather::[rank %d] Decrypted data size %lu received from source %d" COLOR_RESET "\n", rank, count, i);
                fflush(stdout);
            }
#endif
            dest += recvcount * recvtype_sz;
        }*/
    }
    else
    {
        /*
        RAND_bytes(ciphertext_sendbuf, 12);
        if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                               &ciphertext_sendbuf_len, max_out_len,
                               ciphertext_sendbuf, 12,
                               sendbuf, t,
                               NULL, 0))
        {
            printf("Error in encryption: naive gather\n");
            fflush(stdout);
        }
#if 1
        else
        {
            printf(COLOR_GREEN "Gather:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
            fflush(stdout);
        }
#endif
*/
         mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
         
        //mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_recvbuf, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
    }

    return mpi_errno;
}
/* end of add */


#undef FUNCNAME
#define FUNCNAME MPI_Gather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@

MPI_Gather - Gathers together values from a group of processes
 
Input Parameters:
+ sendbuf - starting address of send buffer (choice) 
. sendcount - number of elements in send buffer (integer) 
. sendtype - data type of send buffer elements (handle) 
. recvcount - number of elements for any single receive (integer, 
significant only at root) 
. recvtype - data type of recv buffer elements 
(significant only at root) (handle) 
. root - rank of receiving process (integer) 
- comm - communicator (handle) 

Output Parameters:
. recvbuf - address of receive buffer (choice, significant only at 'root') 

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_COUNT
.N MPI_ERR_TYPE
.N MPI_ERR_BUFFER
@*/
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_GATHER);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_COLL_FUNC_ENTER(MPID_STATE_MPI_GATHER);

    /* Validate parameters, especially handles needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPIR_ERRTEST_COMM(comm, mpi_errno);
	}
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPID_Comm_get_ptr( comm, comm_ptr );

    /* Validate parameters and objects (post conversion) */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPID_Datatype *sendtype_ptr=NULL, *recvtype_ptr=NULL;
	    int rank;

            MPID_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            if (mpi_errno != MPI_SUCCESS) goto fn_fail;

	    if (comm_ptr->comm_kind == MPID_INTRACOMM) {
		MPIR_ERRTEST_INTRA_ROOT(comm_ptr, root, mpi_errno);

                if (sendbuf != MPI_IN_PLACE) {
                    MPIR_ERRTEST_COUNT(sendcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
                    if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                        MPID_Datatype_valid_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_USERBUFFER(sendbuf,sendcount,sendtype,mpi_errno);
                }
                
                rank = comm_ptr->rank;
                if (rank == root) {
                    MPIR_ERRTEST_COUNT(recvcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
                    if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                        MPID_Datatype_valid_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_RECVBUF_INPLACE(recvbuf, recvcount, mpi_errno);
                    MPIR_ERRTEST_USERBUFFER(recvbuf,recvcount,recvtype,mpi_errno);

                    /* catch common aliasing cases */
                    if (recvbuf != MPI_IN_PLACE && sendtype == recvtype && sendcount == recvcount && sendcount != 0) {
                        MPI_Aint recvtype_size;
                        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
                        MPIR_ERRTEST_ALIAS_COLL(sendbuf, ((char *)recvbuf) + comm_ptr->rank*recvcount*recvtype_size,mpi_errno);
                    }
                }
                else
                    MPIR_ERRTEST_SENDBUF_INPLACE(sendbuf, sendcount, mpi_errno);
            }

	    if (comm_ptr->comm_kind == MPID_INTERCOMM) {
		MPIR_ERRTEST_INTER_ROOT(comm_ptr, root, mpi_errno);

                if (root == MPI_ROOT) {
                    MPIR_ERRTEST_COUNT(recvcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
                    if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                        MPID_Datatype_valid_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( recvtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_RECVBUF_INPLACE(recvbuf, recvcount, mpi_errno);
                    MPIR_ERRTEST_USERBUFFER(recvbuf,recvcount,recvtype,mpi_errno);
                }
                
                else if (root != MPI_PROC_NULL) {
                    MPIR_ERRTEST_COUNT(sendcount, mpi_errno);
                    MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
                    if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                        MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                        MPID_Datatype_valid_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                        MPID_Datatype_committed_ptr( sendtype_ptr, mpi_errno );
                        if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                    }
                    MPIR_ERRTEST_SENDBUF_INPLACE(sendbuf, sendcount, mpi_errno);
                    MPIR_ERRTEST_USERBUFFER(sendbuf,sendcount,sendtype,mpi_errno);
                }
            }
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */
    /* Added by abu naser */
    //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
    
     MPI_Comm shmem_comm;
    //MPID_Comm *shmem_commptr;

        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        /*MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        MPID_Comm_valid_ptr( shmem_commptr, mpi_errno, FALSE );
         if (mpi_errno != MPI_SUCCESS)
        {
            printf("Invalid share comm rank %d\n",comm_ptr->rank);fflush(stdout);
        }*/
    if (security_approach == 301)
    {
        //NAIVE
        //mpi_errno = MPIR_Naive_Sec_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
        int sendtype_sz, recvtype_sz;
        unsigned long ciphertext_sendbuf_len = 0;
        sendtype_sz = recvtype_sz = 0;
        int var;
        unsigned long count;
        var = MPI_Type_size(sendtype, &sendtype_sz);
        var = MPI_Type_size(recvtype, &recvtype_sz);
        int t = sendcount * sendtype_sz;
        unsigned long max_out_len = t + 16;
        int rank, i, next, dest;

        rank = comm_ptr->rank;

        if (rank == root)
        {
            RAND_bytes(ciphertext_sendbuf, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf, 12,
                                   sendbuf, t,
                                   NULL, 0))
            {
                printf("Error in encryption: naive gather\n");
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_GREEN "Gather naive:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
                fflush(stdout);
            }
#endif

            mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_recvbuf, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
            //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);

            dest = 0;
            for (i = 0; i < comm_ptr->local_size; i++)
            {
                next = i * (recvcount * recvtype_sz + 28);
                if (!EVP_AEAD_CTX_open(global_ctx, recvbuf + dest,
                                       &count, (unsigned long)((recvcount * recvtype_sz)),
                                       (ciphertext_recvbuf + next), 12,
                                       (ciphertext_recvbuf + next + 12), (unsigned long)((recvcount * recvtype_sz) + 16),
                                       NULL, 0))
                {
                    printf("Decryption error: naive gather\n");
                    fflush(stdout);
                }
#if 0
                else
                {
                    printf(COLOR_BLUE "Gather naive::[rank %d] Decrypted data size %lu received from source %d" COLOR_RESET "\n", rank, count, i);
                    fflush(stdout);
                }
#endif
                dest += recvcount * recvtype_sz;
            }
        }
        else
        {

            RAND_bytes(ciphertext_sendbuf, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf, 12,
                                   sendbuf, t,
                                   NULL, 0))
            {
                printf("Error in encryption: naive gather\n");
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_GREEN "Gather naive:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
                fflush(stdout);
            }
#endif

            //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);

            mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_recvbuf, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
        }
        //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount*sendtype_sz, MPI_BYTE, recvbuf, recvcount*recvtype_sz, MPI_BYTE, root, comm_ptr, &errflag);
        //printf("done calling sendcnt %d recvcnt %d\n", sendcount * sendtype_sz, recvcount * recvtype_sz);fflush(stdout);
    }
    else if(security_approach == 310)
    {
        //NAIVE
        //mpi_errno = MPIR_Naive_Sec_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
        int sendtype_sz, recvtype_sz;
        unsigned long ciphertext_sendbuf_len = 0;
        sendtype_sz = recvtype_sz = 0;
        int var;
        unsigned long count;
        var = MPI_Type_size(sendtype, &sendtype_sz);
        var = MPI_Type_size(recvtype, &recvtype_sz);
        int t = sendcount * sendtype_sz;
        unsigned long max_out_len = t + 16;
        int rank, i, next, dest;
        MPID_Node_id_t rank_node_id, root_node_id;
        int local_rank, local_size;
         MPIR_Errflag_t *eflag;
        
        rank = comm_ptr->rank;
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
        MPID_Get_node_id(comm_ptr, root, &root_node_id);

       
       
         mpi_errno = PMPI_Comm_rank(shmem_comm, &local_rank);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
        }

        int decryption_tasks = comm_ptr->local_size / local_size; // assume for now they are divisible

        printf("rank %d comm_ptr->dev.ch.shmem_coll_ok %d local_rank %d local_size %d\n",rank,comm_ptr->dev.ch.shmem_coll_ok,local_rank,local_size);
        
       // if (rank == root)
       if(rank_node_id == root_node_id)
        {
            RAND_bytes(ciphertext_sendbuf, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf, 12,
                                   sendbuf, t,
                                   NULL, 0))
            {
                printf("Error in encryption: naive gather\n");
                fflush(stdout);
            }
#if 1
            else
            {
                printf(COLOR_GREEN "Gather 310:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
                fflush(stdout);
            }
#endif

            mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_shmem_buffer, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
            //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
            
            mpi_errno = MPI_Barrier(shmem_comm);
             // mpi_errno = MPIR_Barrier_impl(shmem_commptr, eflag);  
            dest = 0;
            for (i = local_rank*decryption_tasks; i < (local_rank*decryption_tasks+decryption_tasks); i++)
            {
                next = i * (recvcount * recvtype_sz + 28);
                if (!EVP_AEAD_CTX_open(global_ctx, shmem_buffer + dest,
                                       &count, (unsigned long)((recvcount * recvtype_sz)),
                                       (ciphertext_shmem_buffer + next), 12,
                                       (ciphertext_shmem_buffer + next + 12), (unsigned long)((recvcount * recvtype_sz) + 16),
                                       NULL, 0))
                {
                    printf("Decryption error: 310: naive gather\n");
                    fflush(stdout);
                }
#if 1
                else
                {
                    printf(COLOR_BLUE "Gather 310::[rank %d] Decrypted data size %lu received from source %d" COLOR_RESET "\n", rank, count, i);
                    fflush(stdout);
                }
#endif
                dest += recvcount * recvtype_sz;
            }

             //printf("root wait for all processes\n");fflush(stdout);
             mpi_errno = MPI_Barrier(shmem_comm); // wait for all the leaders to finish
              //mpi_errno = MPIR_Barrier_impl(shmem_commptr, eflag);  

             /*if (mpi_errno)
             {
                 MPIR_ERR_POP(mpi_errno);
                 goto fn_fail;
             }*/
            if(root == rank)
            {
                printf("-->root wait done for all processes\n");fflush(stdout);
                memcpy(recvbuf,shmem_buffer, comm_ptr->local_size*recvcount*recvtype_sz);
            }
        }
        else
        {

            RAND_bytes(ciphertext_sendbuf, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf, 12,
                                   sendbuf, t,
                                   NULL, 0))
            {
                printf("Error in encryption: naive gather\n");
                fflush(stdout);
            }
#if 1
            else
            {
                printf(COLOR_GREEN "Gather:: [rank %d] Encrypted data size %lu" COLOR_RESET "\n", rank, ciphertext_sendbuf_len);
                fflush(stdout);
            }
#endif

            //mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);

            mpi_errno = MPIR_Gather_impl(ciphertext_sendbuf, (t + 28), MPI_BYTE, ciphertext_recvbuf, (recvcount * recvtype_sz + 28), MPI_BYTE, root, comm_ptr, &errflag);
        
        }
    }// 310 
    else
    {
        mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm_ptr, &errflag);
    }
    /* end of add */  
    if (mpi_errno) goto fn_fail;
#ifdef _OSU_MVAPICH_
    if (mv2_use_osu_collectives) {
        if(comm_ptr->dev.ch.allgather_comm_ok == 0) {
            mpi_errno = mv2_increment_allgather_coll_counter(comm_ptr);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

#endif /* _OSU_MVAPICH_ */

        
    /* ... end of body of routine ... */
    
  fn_exit:
    MPID_MPI_COLL_FUNC_EXIT(MPID_STATE_MPI_GATHER);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_gather",
	    "**mpi_gather %p %d %D %p %d %D %d %C", sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/*******************************/
/* added by abu naser           */
/*******************************/

/*
*
* use shared memory with Multi-leader
*
*/


/*
*
* no share memory, multi-leader, and round-robin
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct_CNoSM_RR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_CNoSM_RR(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2_Direct_CNoSM_RR\n",comm_ptr->rank,hostname);fflush(stdout);
#endif 
    if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root != 0)
    {
        //printf("call MPIR_Gather_intra\n");fflush(stdout);
         int ret = MPIR_Gather_intra(sendbuf, sendcount, sendtype,
                                                             recvbuf, recvcount, recvtype,
                                                             root, comm_ptr, errflag);
            return ret;
    }       

    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, j, reqs, temp;
    unsigned long count;
    int tag_root =0;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id, source_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank, dest;
    int st_indx = 0;
    /* extract the rank,size information for the intra-node
     * communicator */
     //if(rank_node_id == root_node_id )
    //{
        //printf("rank %d root %d rank_node_id %d root_node_id %d\n",rank,root, rank_node_id,root_node_id);fflush(stdout);
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        mpi_errno = PMPI_Comm_rank(shmem_comm, &local_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    //}
    
    comm_size = comm_ptr->local_size;
     MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    
    reqs = 0;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
#if 0    
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        int t = (recvtype_extent * recvcount);
        int each_rank_size = comm_size / local_size;
        int next;
       
        for(j=0;j<each_rank_size;j++)
        {
            for(i=1;i<local_size;i++)
            {
                next = i*each_rank_size*t + j*t;
               /* mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);*/
                  /* instead of MPIR_GATHER_TAG use j as tag */                         
                  mpi_errno = MPIC_Irecv((recvbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++]);                           
                printf("**[intra-node] rank %d receiving from rank %d for %d tag %d location: %d, (i*each_rank_size*t = %d) (j*t = %d)\n",
                rank,i,(i*each_rank_size + j),j,next,i*each_rank_size*t,j*t);fflush(stdout);

            }
        }
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }
#endif    
    int t = (recvtype_extent * recvcount);
    int each_rank_size = comm_size / local_size;
    int next = 0;
    
    if(rank_node_id == root_node_id)
    { 
       
        /* Receive from local ranks */
        if (rank == root || ((local_rank * each_rank_size + each_rank_size) <= local_size)) // intra node recv
        {
              next = 0;
            for (i = local_rank * each_rank_size; i < (local_rank * each_rank_size + each_rank_size); i++)
            {
                if(rank==root && i==root)
                    memcpy(recvbuf+next,sendbuf,t); // copy own data
                else if (rank == root) // recive directly to recvbuf from other rank
                    mpi_errno = MPIC_Irecv(recvbuf + next, t, MPI_BYTE, i,
                                           MPIR_GATHER_TAG, comm_ptr, &reqarray[reqs++]);
                else // other same node leader recv to temp buff e.g. rank 1
                    mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t, MPI_BYTE, i,
                                           MPIR_GATHER_TAG, comm_ptr, &reqarray[reqs++]);
                //printf("@@ [rank %d] receiving from rank %d  data %d next %d\n",rank, i,t,next);fflush(stdout);    
                next += t;
            }
            if(rank != root)
            {
                /* send own data to leader */
                int dest = rank / each_rank_size;
                mpi_errno = MPIC_Isend(sendbuf, t, MPI_BYTE, dest,
                                  MPIR_GATHER_TAG, comm_ptr,&reqarray[reqs++], errflag);
                //printf("** [rank %d] sending to rank %d  data %d\n",rank, dest,t);fflush(stdout);                  
                
            }
          
        }
        else
        {
            /* receiving from remote ranks */
            for (i = local_rank * each_rank_size; i < (local_rank * each_rank_size + each_rank_size); i++)
            {
                mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, (t + 28), MPI_BYTE, i,
                                       MPIR_GATHER_TAG, comm_ptr, &reqarray[reqs++]);
                //printf("## [rank %d] receiving from rank %d  data %d next %d\n",rank, i,t,next);fflush(stdout);                       
                next += (t + 28);
            }

            /* send own data to leader */
                int dest = rank / each_rank_size;
                mpi_errno = MPIC_Isend(sendbuf, t, MPI_BYTE, dest,
                                  MPIR_GATHER_TAG, comm_ptr,&reqarray[reqs++], errflag);
                //printf("** [rank %d] sending to rank %d  data %d\n",rank, dest,t);fflush(stdout); 
        }
    }

    if(rank == root)
    {
        /* post receive for other leaders */
         for(j=0;j<each_rank_size;j++)
        {
            for(i=1;i<local_size;i++)
            {
                next = i*each_rank_size*t + j*t;                       
                  mpi_errno = MPIC_Irecv((recvbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++]);                           
                //printf("**[intra-node] [rank %d] receiving from rank %d for %d tag %d location: %d, (i*each_rank_size*t = %d) (j*t = %d) reqs %d\n",
                //rank,i,(i*each_rank_size + j),j,next,i*each_rank_size*t,j*t, reqs);fflush(stdout);

            }
        }
    }

    /* other leaders receive from the remote ranks */
     if(rank_node_id == root_node_id && rank != root)
    {
        j = 0;
        temp = 0;
        next = 0;
        dest = 0;
        tag_root =0;
        for (i = local_rank * each_rank_size; i < (local_rank * each_rank_size + each_rank_size); i++)
            {
                MPID_Get_node_id(comm_ptr, i, &source_node_id);              
                 if (rank_node_id == source_node_id ) // other leader receive data from own node
                {
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], &starray[j], errflag);
                    j++;
                    /* now forward to root */
                    mpi_errno = MPIC_Isend((ciphertext_recvbuf+temp),
                                           t,
                                           MPI_BYTE, root, tag_root, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    temp += t;                       
                    //printf("NON-ROOT sending to same node root rank %d ----> root %d i(source) %d tag %d\n", rank, root,i,tag_root);
                    tag_root++;
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
                   
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], &starray[j], errflag);
                    j++;
                      
                    if (!EVP_AEAD_CTX_open(global_ctx, ciphertext_sendbuf+next,
                                           &count, t,
                                           (ciphertext_recvbuf+dest), 12,
                                           (ciphertext_recvbuf + dest+ 12), (unsigned long)(t + 16),
                                           NULL, 0))
                    {
                        printf("Decryption error:[rank %d rank_node_id %d root_node_id %d] gather direct recv i %d\n",
                         rank, rank_node_id, root_node_id, i);
                        fflush(stdout);
                    }
#if 0
                    else
                    {
                        printf(COLOR_BLUE "gather noshm-rr recv:: [rank %d rank_node_id %d] Decrypted data size %lu i %d" COLOR_RESET "\n",
                               rank, rank_node_id, count,i);
                        fflush(stdout);
                    }
#endif
                    //printf("** after decryption rank %d sending to root %d of dest-rank %d data %d tag %d reqs %d j %d\n",rank, root, i, t,tag_root,reqs,j);fflush(stdout);
                    mpi_errno = MPIC_Isend((ciphertext_sendbuf + next),
                                           t,
                                           MPI_BYTE, root, tag_root, comm_ptr,
                                           &reqarray[reqs++], errflag);

                    dest += t + 16 + 12; // update after send
                    next += t; // get the encrytion location
                    tag_root++;
                   
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
                 
            } // i

            if( j > 0)
            {
               //printf("before: rank %d st_indx %d j %d reqs %d\n",rank, st_indx, j, reqs);fflush(stdout); 
               st_indx = j;
               //reqs = reqs - j;
               //printf("after: rank %d st_indx %d j %d reqs %d\n",rank, st_indx, j, reqs);fflush(stdout);
            }
    }
   


    /* sender side*/
    if (local_size != comm_size)
    {
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        t = (sendtype_extent * sendcount);
        unsigned long max_out_len = (unsigned long)(t+16);
        unsigned long ciphertext_sendbuf_len;
        int new_root = 0;
        
        int each_rank_size = comm_size / local_size;
        new_root = rank / each_rank_size;          
        MPID_Node_id_t new_root_node_id;
        MPID_Get_node_id(comm_ptr, new_root, &new_root_node_id);
       
        if(rank_node_id != new_root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"rank %d sending to  new_root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
                    RAND_bytes(ciphertext_sendbuf, 12);
                    if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                           &ciphertext_sendbuf_len, max_out_len,
                                           ciphertext_sendbuf, 12,
                                           sendbuf, t,
                                           NULL, 0))
                    {
                        printf("Error in encryption: [rank %d rank_node_id %d] gather direct new_root %d new_root_node_id %d\n",
                               rank, rank_node_id, new_root, new_root_node_id);
                        fflush(stdout);
                    }
#if 0
                    else
                    {

                        printf(COLOR_GREEN "gather no-shm-rr direct:: [rank %d local_rank %d rank_node_id %d root_node_id %d new_root %d new_root_node_id %d] Encrypted data size %lu" COLOR_RESET "\n",
                               rank, local_rank, rank_node_id, root_node_id, new_root, new_root_node_id, ciphertext_sendbuf_len);
                        fflush(stdout);
                    }
#endif
            
            mpi_errno = MPIC_Send(ciphertext_sendbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_GATHER_TAG, comm_ptr, errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
           
        }
    }

    /* wait for finish sending */
    if(rank_node_id == root_node_id )
    {
        //printf("--->> rank %d waiting to finish send reqs-st_indx %d st_indx %d\n",rank, reqs-st_indx, st_indx);fflush(stdout);
         /* ... then wait for *all* of them to finish: */
        if(rank != root)
            mpi_errno = MPIC_Waitall(reqs-st_indx, &reqarray[st_indx], starray, errflag);
        else  
            mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);  
       /*for(i=0; i< reqs; i++)
       {
           mpi_errno = MPIC_Waitall(1, &reqarray[i], &starray[i], errflag);
           printf("req i %d is done\n",i);fflush(stdout);
       }*/
        /* --BEGIN ERROR HANDLING-- */
#if 0        
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < (reqs-st_indx); i++) {
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
#endif        
        /* --END ERROR HANDLING-- */ 
        // printf(COLOR_YELLOW"DONE:: rank %d root %d rank_node_id %d root_node_id %d"COLOR_RESET"\n", rank, root,rank_node_id,root_node_id);fflush(stdout); 
    }

   

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#if 0
#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct_CNoSM_RR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_CNoSM_RR(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2_Direct_CNoSM_RR\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       

    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);


    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
    int st_indx = 0;
    /* extract the rank,size information for the intra-node
     * communicator */
     //if(rank_node_id == root_node_id )
    //{
        //printf("rank %d root %d rank_node_id %d root_node_id %d\n",rank,root, rank_node_id,root_node_id);fflush(stdout);
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        mpi_errno = PMPI_Comm_rank(shmem_comm, &local_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(shmem_comm, &local_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    //}
    
    comm_size = comm_ptr->local_size;
     MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    
    reqs = 0;
    int j;
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        int t = (recvtype_extent * recvcount);
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
               /* mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);*/
                  /* instead of MPIR_GATHER_TAG use j as tag */                         
                  mpi_errno = MPIC_Irecv((recvbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++]);                           
                printf("**[intra-node] rank %d receiving from rank %d for %d tag %d location: %d, (i*each_rank_size*t = %d) (j*t = %d)\n",
                rank,i,(i*each_rank_size + j),j,next,i*each_rank_size*t,j*t);fflush(stdout);

            }
        }
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from remote process
       
        int t = (recvtype_extent * recvcount);
        int each_rank_size = comm_size / local_size;
        int next = 0;
        
        /* Receive from remote ranks and local ranks */  
        for(i=local_rank * each_rank_size;i<(local_rank * each_rank_size+each_rank_size);i++)
        {
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf+next, (t+28), MPI_BYTE, i,
                  MPIR_GATHER_TAG, comm_ptr,  &reqarray[reqs++]);
                 
            next += (t+28);      
        }  

                     
    }


   

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    int tag_root =0;
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {

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
            int t = (recvtype_extent * recvcount);
            unsigned long max_out_len = (unsigned long)(16 + t);
            unsigned long count;
            MPID_Node_id_t source_node_id;
           
            j = 0;
            int temp = 0;
            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &source_node_id);
                if (rank == i && rank == root)
                {
                    memcpy(recvbuf, sendbuf+next, t);
                    printf("@@ copy to own rank %d == i %d\n", rank, i);
                }
                else if (rank_node_id == source_node_id && rank == root) // receiving from same node
                {
                    mpi_errno = MPIC_Irecv((recvbuf + next),
                                           t,
                                           MPI_BYTE, i, MPIR_GATHER_TAG, comm_ptr,
                                           &reqarray[reqs++]);
                    printf("receiving from the same node rank %d ----> i %d next %d\n", rank, i, next);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
#if 1                
                else if (rank_node_id == source_node_id && rank != root) // other leader receive data from own node
                {
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                    j++;
                    /* now forward to root */
                    mpi_errno = MPIC_Isend((ciphertext_recvbuf+temp),
                                           t,
                                           MPI_BYTE, root, MPIR_GATHER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);
                    temp += t + 28;                       
                    printf("NON-ROOT sending to same node root rank %d ----> root %d\n", rank, root);
                    if (mpi_errno)
                    {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
#endif                
                else
                {
                   // if(rank == root)
                    //    temp_buffer = sendbuf;
                    //else
                    //{
                        mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                        j++;
                      // temp_buffer = ciphertext_recvbuf;    
                    //}

                    if (!EVP_AEAD_CTX_open(global_ctx, ciphertext_sendbuf+next,
                                           &count, max_out_len,
                                           (ciphertext_recvbuf+dest), 12,
                                           (ciphertext_recvbuf + dest+ 12), (unsigned long)(t + 16),
                                           NULL, 0))
                    {
                        printf("Decryption error:[rank %d rank_node_id %d root_node_id %d] gather direct recv i %d\n",
                         rank, rank_node_id, root_node_id, i);
                        fflush(stdout);
                    }
#if 1
                    else
                    {
                        printf(COLOR_BLUE "gather noshm-rr recv:: [rank %d rank_node_id %d] Decrypted data size %lu i %d" COLOR_RESET "\n",
                               rank, rank_node_id, count,i);
                        fflush(stdout);
                    }
#endif
                    printf("** after decryption rank %d sending to root %d of dest-rank %d data %d tag %d reqs %d j %d\n",rank, root, i, t,tag_root,reqs,j);fflush(stdout);
                    mpi_errno = MPIC_Isend((ciphertext_sendbuf + next),
                                           (t),
                                           MPI_BYTE, root, tag_root, comm_ptr,
                                           &reqarray[reqs++], errflag);

                    dest += t + 16 + 12; // update after send
                    tag_root++;
                   
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

            if(rank != root && j > 0)
            {
               printf("before: rank %d st_indx %d j %d reqs %d\n",rank, st_indx, j, reqs);fflush(stdout); 
               st_indx = j;
               //reqs = reqs - j;
               printf("after: rank %d st_indx %d j %d reqs %d\n",rank, st_indx, j, reqs);fflush(stdout);
            }
        }
    }

    /* sender side*/
    if (local_size != comm_size)
    {
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcount);
        unsigned long max_out_len = (unsigned long)(t+16);
        unsigned long ciphertext_sendbuf_len;
        int new_root = 0;
        //if(new_root >= local_size)
        //    new_root = local_size - 1;
        int each_rank_size = comm_size / leader_size;
        new_root = rank / each_rank_size;          
        MPID_Node_id_t new_root_node_id;
        MPID_Get_node_id(comm_ptr, new_root, &new_root_node_id);

        /*if (rank == new_root)
        {
            printf("rank %d received from new_root\n", rank, new_root);
            fflush(stdout);
        }*/
         if (rank_node_id == new_root_node_id && root != rank )
        {
            
            t = (sendtype_extent * sendcount);
            
             printf(COLOR_CYAN"[same node]rank %d sending to new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Isend(sendbuf, t, MPI_BYTE, new_root,
                                  MPIR_GATHER_TAG, comm_ptr,&reqarray[reqs++], errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }                
        }
        
        else if(rank_node_id != new_root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            printf(COLOR_YELLOW"rank %d sending to  new_root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
                    RAND_bytes(ciphertext_sendbuf, 12);
                    if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + 12,
                                           &ciphertext_sendbuf_len, max_out_len,
                                           ciphertext_sendbuf, 12,
                                           sendbuf, t,
                                           NULL, 0))
                    {
                        printf("Error in encryption: [rank %d rank_node_id %d] gather direct new_root %d new_root_node_id %d\n",
                               rank, rank_node_id, new_root, new_root_node_id);
                        fflush(stdout);
                    }
#if 1
                    else
                    {

                        printf(COLOR_GREEN "gather no-shm-rr direct:: [rank %d local_rank %d rank_node_id %d root_node_id %d new_root %d new_root_node_id %d] Encrypted data size %lu" COLOR_RESET "\n",
                               rank, local_rank, rank_node_id, root_node_id, new_root, new_root_node_id, ciphertext_sendbuf_len);
                        fflush(stdout);
                    }
#endif
            
            mpi_errno = MPIC_Send(ciphertext_sendbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_GATHER_TAG, comm_ptr, errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
           
        }
    }

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
        printf("--->> rank %d waiting to finish send reqs-st_indx %d st_indx %d\n",rank, reqs-st_indx, st_indx);fflush(stdout);
         /* ... then wait for *all* of them to finish: */
        if(rank != root)
            mpi_errno = MPIC_Waitall(reqs-st_indx, &reqarray[st_indx], starray, errflag);
        else    
       for(i=0; i< reqs; i++)
       {
           mpi_errno = MPIC_Waitall(1, &reqarray[i], &starray[i], errflag);
           printf("req i %d is done\n",i);fflush(stdout);
       }
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < (reqs-st_indx); i++) {
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
         printf(COLOR_YELLOW"DONE:: rank %d root %d rank_node_id %d root_node_id %d"COLOR_RESET"\n", rank, root,rank_node_id,root_node_id);fflush(stdout); 
    }

   

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif



/* end of add */