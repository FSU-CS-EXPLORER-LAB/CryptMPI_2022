/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"
#include "coll_shmem.h"
#include "common_tuning.h"
#include "bcast_tuning.h"
#include "scatter_tuning.h"

/* added by abu naser */
#include "secure_allgather.h"
/* end of add */

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_mcast);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_binomial);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_direct);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_direct_blk);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_two_level_binomial);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_scatter_two_level_direct);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_mcast);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_binomial);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_blk);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_binomial);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_direct);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_mcast_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_mcast_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_binomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_binomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_blk_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_blk_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_binomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_binomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_inter_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_inter_bytes_recv);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_mcast_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_mcast_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_binomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_binomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_blk_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_direct_blk_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_binomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_binomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_two_level_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_inter_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_inter_count_recv);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_scatter_count_recv);

int (*MV2_Scatter_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)=NULL;

int (*MV2_Scatter_intra_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)=NULL;

/* This is the default implementation of scatter. The algorithm is:
   
   Algorithm: MPI_Scatter

   We use a binomial tree algorithm for both short and
   long messages. At nodes other than leaf nodes we need to allocate
   a temporary buffer to store the incoming message. If the root is
   not rank 0, we reorder the sendbuf in order of relative ranks by 
   copying it into a temporary buffer, so that all the sends from the
   root are contiguous and in the right order. In the heterogeneous
   case, we first pack the buffer by using MPI_Pack and then do the
   scatter. 

   Cost = lgp.alpha + n.((p-1)/p).beta
   where n is the total size of the data to be scattered from the root.

   Possible improvements: 

   End Algorithm: MPI_Scatter
*/

/* begin:nested */
/* not declared static because a machine-specific function may call this one in some cases */

#if defined(_MCST_SUPPORT_)
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_mcst_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_mcst_MV2(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_mcst_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
    MPIR_TIMER_START(coll,scatter,mcast); 
    int mpi_errno=MPI_SUCCESS; 
    int mpi_errno_ret=MPI_SUCCESS; 
    int rank=comm_ptr->rank, local_rank; 
    int comm_size = comm_ptr->local_size; 
    int sendtype_size, recvtype_size; 
    MPI_Aint sendtype_extent=0, recvtype_extent=0;
    MPI_Aint nbytes, copy_offset; 
    int tmp_buf_size=0, intra_node_root=0; 
    int leader_of_root, sendtype_contig; 
    void *mcast_scatter_buf=NULL; 
    void *in_buf=NULL; 
    int in_count=0;
    MPI_Datatype in_type; 
    MPI_Comm shmem_comm; 
    MPID_Comm *shmem_commptr; 
    MPI_Status status; 
    MPID_Datatype *dtp;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_mcast, 1);

    MPIU_CHKLMEM_DECL(1);
        
    if (rank == root) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        nbytes = sendcnt * sendtype_size;
        if (HANDLE_GET_KIND(sendtype) == HANDLE_KIND_BUILTIN) { 
            sendtype_contig = 1;
        } else {
            MPID_Datatype_get_ptr(sendtype, dtp);
            sendtype_contig = dtp->is_contig;
        }
        
        if( sendtype_contig == 0) { 
            tmp_buf_size = sendtype_extent * comm_size; 
        } 
    } else {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
        tmp_buf_size = recvcnt * recvtype_extent * comm_size; 
    } 
    
    /* Allocate tmp buffer space if required */ 
    if(tmp_buf_size > 0) { 
          MPIU_CHKLMEM_MALLOC(mcast_scatter_buf, void *, tmp_buf_size, 
                            mpi_errno, "temporary buffer");
    } 

    if( (rank == root && sendtype_contig == 0) ||
        (rank != root)) { 
        in_buf   = mcast_scatter_buf; 
        in_count = nbytes * comm_size; 
        in_type  = MPI_BYTE; 
    } else { 
        in_buf   = (void *)sendbuf; 
        in_count = sendcnt * comm_size; 
        in_type  = sendtype; 
    } 

   
    
    /* Before we do the 2-level mcast, we need to see if the root 
     * is also the node-level leader. If not, we need to transfer the
     * data from the root to its leader */  

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank; 

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    /* leader_of_root is the global rank of the leader of the root */

    if ((local_rank == 0) && (root != rank)
        && (leader_of_root == rank)) {
        /* The root of the scatter operation is not the node leader. Recv
         * data from the node leader, as bytes, so that the data is ready 
         * for the mcast */ 
        MPIR_PVAR_INC(scatter, mcast, recv, nbytes * comm_size, MPI_BYTE);
        mpi_errno =
            MPIC_Recv(mcast_scatter_buf, nbytes * comm_size, MPI_BYTE,
                         root, MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue
             */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    if (rank == root && local_rank != 0) {
        /* The root of the scatter operation is not the node leader. Send
         * data to the node leader */
        MPIR_PVAR_INC(scatter, mcast, send, sendcnt * comm_size, sendtype);
        mpi_errno = MPIC_Send(sendbuf, sendcnt * comm_size, sendtype,
                                 leader_of_root, MPIR_SCATTER_TAG, comm_ptr,
                                 errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue
             */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    if(rank == root && local_rank == 0 && sendtype_contig == 0) {
      /* I am the root and the node-leader. My data is non-contig and I need to 
       * pack before calling mcst-bcast  */
        mpi_errno = MPIR_Localcopy((char *)sendbuf, sendcnt*comm_size, sendtype, 
                                  mcast_scatter_buf, nbytes*comm_size, MPI_BYTE);   
    } 

    mpi_errno = MPIR_Mcast_inter_node_MV2(in_buf, in_count, in_type, 
                                     leader_of_root, comm_ptr, errflag);

    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    if(comm_ptr->dev.ch.intra_node_done == 0) { 
        mpi_errno = MPIR_Shmem_Bcast_MV2(in_buf, in_count, in_type,
                                         intra_node_root, shmem_commptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    } 

 
    if(rank == root) {  
       if( recvbuf != MPI_IN_PLACE)  { 
          copy_offset =  sendtype_extent * sendcnt * rank; 
          mpi_errno = MPIR_Localcopy((char *)sendbuf + copy_offset, sendcnt, sendtype, 
                                  recvbuf, recvcnt, recvtype); 
       } 
    } else { 
       copy_offset =  recvtype_extent * recvcnt * rank; 
       mpi_errno = MPIR_Localcopy((char *) mcast_scatter_buf + copy_offset, nbytes, MPI_BYTE, 
                                  recvbuf, recvcnt, recvtype); 
    } 

    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno) {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                 FCNAME, __LINE__, MPI_ERR_OTHER,
                                 "**fail", 0);
        return mpi_errno;
    }
     

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,mcast);
    return mpi_errno;
  fn_fail:
    goto fn_exit;

} 
#endif /* #if defined(_MCST_SUPPORT_) */ 


#undef FUNCNAME
#define FUNCNAME MPIR_mcst_wrap_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_mcst_wrap_MV2(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Binomial(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    if (sendcnt == 32)
    {
    printf("[scatter osu rank = %d root = %d host = %s] Func: MPIR_Scatter_MV2_Binomial\n",comm_ptr->rank,root, hostname);fflush(stdout);
    }
#endif      
    MPIR_TIMER_START(coll,scatter,binomial);
    MPI_Status status;
    MPI_Aint extent = 0;
    int rank, comm_size, is_homogeneous, sendtype_size;
    int curr_cnt, relative_rank;
    MPI_Aint nbytes, send_subtree_cnt;
    int mask, recvtype_size = 0, src, dst;
#ifdef MPID_HAS_HETERO
    int position;
#endif                          /* MPID_HAS_HETERO */
    int tmp_buf_size = 0;
    void *tmp_buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_binomial, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcnt == 0))
        || ((rank != root) && (recvcnt == 0))) {

        MPIR_TIMER_END(coll,scatter,binomial);
        return MPI_SUCCESS;
    }

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero) {
        is_homogeneous = 0;
    }
#endif                          /* MPID_HAS_HETERO */

/* Use binomial tree algorithm */

    if (rank == root) {
        MPID_Datatype_get_extent_macro(sendtype, extent);
    }

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    if (is_homogeneous) {
        /* communicator is homogeneous */
        if (rank == root) {
            /* We separate the two cases (root and non-root) because
               in the event of recvbuf=MPI_IN_PLACE on the root,
               recvcnt and recvtype are not valid */
            MPID_Datatype_get_size_macro(sendtype, sendtype_size);
            nbytes = sendtype_size * sendcnt;
        } else {
            MPID_Datatype_get_size_macro(recvtype, recvtype_size);
            nbytes = recvtype_size * recvcnt;
        }

        curr_cnt = 0;

        /* all even nodes other than root need a temporary buffer to
           receive data of max size (nbytes*comm_size)/2 */
        if (relative_rank && !(relative_rank % 2)) {
            tmp_buf_size = (nbytes * comm_size) / 2;
            tmp_buf = MPIU_Malloc(tmp_buf_size);
            /* --BEGIN ERROR HANDLING-- */
            if (!tmp_buf) {
                mpi_errno =
                    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**nomem", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */
        }

        /* if the root is not rank 0, we reorder the sendbuf in order of
           relative ranks and copy it into a temporary buffer, so that
           all the sends from the root are contiguous and in the right
           order. */
        if (rank == root) {
            if (root != 0) {
                tmp_buf_size = nbytes * comm_size;
                tmp_buf = MPIU_Malloc(tmp_buf_size);
                /* --BEGIN ERROR HANDLING-- */
                if (!tmp_buf) {
                    mpi_errno =
                        MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                             FCNAME, __LINE__, MPI_ERR_OTHER,
                                             "**nomem", 0);
                    return mpi_errno;
                }
                /* --END ERROR HANDLING-- */

                if (recvbuf != MPI_IN_PLACE) {
                    mpi_errno =
                        MPIR_Localcopy(((char *) sendbuf +
                                        extent * sendcnt * rank),
                                       sendcnt * (comm_size - rank), sendtype,
                                       tmp_buf, nbytes * (comm_size - rank),
                                       MPI_BYTE);
                } else {
                    mpi_errno =
                        MPIR_Localcopy(((char *) sendbuf +
                                        extent * sendcnt * (rank + 1)),
                                       sendcnt * (comm_size - rank - 1),
                                       sendtype, (char *) tmp_buf + nbytes,
                                       nbytes * (comm_size - rank - 1),
                                       MPI_BYTE);
                }
                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno) {
                    mpi_errno =
                        MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                             FCNAME, __LINE__, MPI_ERR_OTHER,
                                             "**fail", 0);
                    return mpi_errno;
                }
                /* --END ERROR HANDLING-- */

                mpi_errno = MPIR_Localcopy(sendbuf, sendcnt * rank, sendtype,
                                           ((char *) tmp_buf +
                                            nbytes * (comm_size - rank)),
                                           nbytes * rank, MPI_BYTE);
                /* --BEGIN ERROR HANDLING-- */
                if (mpi_errno) {
                    mpi_errno =
                        MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                             FCNAME, __LINE__, MPI_ERR_OTHER,
                                             "**fail", 0);
                    return mpi_errno;
                }
                /* --END ERROR HANDLING-- */

                curr_cnt = nbytes * comm_size;
            } else {
                curr_cnt = sendcnt * comm_size;
            }
        }

        /* root has all the data; others have zero so far */

        mask = 0x1;
        while (mask < comm_size) {
            if (relative_rank & mask) {
                src = rank - mask;
                if (src < 0)
                    src += comm_size;

                /* The leaf nodes receive directly into recvbuf because
                   they don't have to forward data to anyone. Others
                   receive data into a temporary buffer. */
                if (relative_rank % 2) {
                    MPIR_PVAR_INC(scatter, binomial, recv, recvcnt, recvtype);
                    mpi_errno = MPIC_Recv(recvbuf, recvcnt, recvtype,
                                             src, MPIR_SCATTER_TAG, comm_ptr,
                                             &status, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                } else {
                    MPIR_PVAR_INC(scatter, binomial, recv, tmp_buf_size, MPI_BYTE);
                    mpi_errno =
                        MPIC_Recv(tmp_buf, tmp_buf_size, MPI_BYTE, src,
                                     MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }

                    /* the recv size is larger than what may be sent in
                       some cases. query amount of data actually received */
                    MPIR_Get_count_impl(&status, MPI_BYTE, &curr_cnt);
                }
                break;
            }
            mask <<= 1;
        }

        /* This process is responsible for all processes that have bits
           set from the LSB upto (but not including) mask.  Because of
           the "not including", we start by shifting mask back down
           one. */

        mask >>= 1;
        while (mask > 0) {
            if (relative_rank + mask < comm_size) {
                dst = rank + mask;
                if (dst >= comm_size)
                    dst -= comm_size;

                if ((rank == root) && (root == 0)) {
                    send_subtree_cnt = curr_cnt - sendcnt * mask;
                    /* mask is also the size of this process's subtree */
                    MPIR_PVAR_INC(scatter, binomial, send, send_subtree_cnt, sendtype);
                    mpi_errno = MPIC_Send(((char *) sendbuf +
                                              extent * sendcnt * mask),
                                             send_subtree_cnt,
                                             sendtype, dst,
                                             MPIR_SCATTER_TAG, comm_ptr, errflag);
                } else {
                    /* non-zero root and others */
                    send_subtree_cnt = curr_cnt - nbytes * mask;
                    /* mask is also the size of this process's subtree */
                    MPIR_PVAR_INC(scatter, binomial, send, send_subtree_cnt, MPI_BYTE);
                    mpi_errno = MPIC_Send(((char *) tmp_buf + nbytes * mask),
                                             send_subtree_cnt,
                                             MPI_BYTE, dst,
                                             MPIR_SCATTER_TAG, comm_ptr, errflag);
                }
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                curr_cnt -= send_subtree_cnt;
            }
            mask >>= 1;
        }

        if ((rank == root) && (root == 0) && (recvbuf != MPI_IN_PLACE)) {
            /* for root=0, put root's data in recvbuf if not MPI_IN_PLACE */
            mpi_errno = MPIR_Localcopy(sendbuf, sendcnt, sendtype,
                                       recvbuf, recvcnt, recvtype);
            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno) {
                mpi_errno =
                    MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**fail", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */
        } else if (!(relative_rank % 2) && (recvbuf != MPI_IN_PLACE)) {
            /* for non-zero root and non-leaf nodes, copy from tmp_buf
               into recvbuf */
            mpi_errno = MPIR_Localcopy(tmp_buf, nbytes, MPI_BYTE,
                                       recvbuf, recvcnt, recvtype);
            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno) {
                mpi_errno =
                    MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**fail", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */
        }

        if (tmp_buf != NULL)
            MPIU_Free(tmp_buf);
    }
#ifdef MPID_HAS_HETERO
    else {                      /* communicator is heterogeneous */
        if (rank == root) {
            MPIR_Pack_size_impl(sendcnt * comm_size, sendtype, &tmp_buf_size);
            tmp_buf = MPIU_Malloc(tmp_buf_size);
            /* --BEGIN ERROR HANDLING-- */
            if (!tmp_buf) {
                mpi_errno =
                    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**nomem", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */

            /* calculate the value of nbytes, the number of bytes in packed
               representation that each process receives. We can't
               accurately calculate that from tmp_buf_size because
               MPI_Pack_size returns an upper bound on the amount of memory
               required. (For example, for a single integer, MPICH-1 returns
               pack_size=12.) Therefore, we actually pack some data into
               tmp_buf and see by how much 'position' is incremented. */

            position = 0;
            MPIR_Pack_impl(sendbuf, 1, sendtype, tmp_buf, tmp_buf_size,
                           &position);
            nbytes = position * sendcnt;

            curr_cnt = nbytes * comm_size;

            if (root == 0) {
                if (recvbuf != MPI_IN_PLACE) {
                    position = 0;
                    MPIR_Pack_impl(sendbuf, sendcnt * comm_size, sendtype,
                                   tmp_buf, tmp_buf_size, &position);
                } else {
                    position = nbytes;
                    MPIR_Pack_impl(((char *) sendbuf + extent * sendcnt),
                                   sendcnt * (comm_size - 1), sendtype, tmp_buf,
                                   tmp_buf_size, &position);
                }
            } else {
                if (recvbuf != MPI_IN_PLACE) {
                    position = 0;
                    MPIR_Pack_impl(((char *) sendbuf + extent * sendcnt * rank),
                                   sendcnt * (comm_size - rank), sendtype,
                                   tmp_buf, tmp_buf_size, &position);
                } else {
                    position = nbytes;
                    MPIR_Pack_impl(((char *) sendbuf +
                                    extent * sendcnt * (rank + 1)),
                                   sendcnt * (comm_size - rank - 1), sendtype,
                                   tmp_buf, tmp_buf_size, &position);
                }
                MPIR_Pack_impl(sendbuf, sendcnt * rank, sendtype, tmp_buf,
                               tmp_buf_size, &position);
            }
        } else {
            MPIR_Pack_impl_size(recvcnt * (comm_size / 2), recvtype,
                                &tmp_buf_size);
            tmp_buf = MPIU_Malloc(tmp_buf_size);
            /* --BEGIN ERROR HANDLING-- */
            if (!tmp_buf) {
                mpi_errno =
                    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**nomem", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */

            /* calculate nbytes */
            position = 0;
            MPIR_Pack_impl(recvbuf, 1, recvtype, tmp_buf, tmp_buf_size,
                           &position);
            nbytes = position * recvcnt;

            curr_cnt = 0;
        }

        mask = 0x1;
        while (mask < comm_size) {
            if (relative_rank & mask) {
                src = rank - mask;
                if (src < 0)
                    src += comm_size;

                MPIR_PVAR_INC(scatter, binomial, recv, tmp_buf_size, MPI_BYTE);
                mpi_errno = MPIC_Recv(tmp_buf, tmp_buf_size, MPI_BYTE, src,
                                         MPIR_SCATTER_TAG, comm_ptr, &status,
                                         errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                /* the recv size is larger than what may be sent in
                   some cases. query amount of data actually received */
                MPIR_Get_count_impl(&status, MPI_BYTE, &curr_cnt);
                break;
            }
            mask <<= 1;
        }

        /* This process is responsible for all processes that have bits
           set from the LSB upto (but not including) mask.  Because of
           the "not including", we start by shifting mask back down
           one. */

        mask >>= 1;
        while (mask > 0) {
            if (relative_rank + mask < comm_size) {
                dst = rank + mask;
                if (dst >= comm_size)
                    dst -= comm_size;

                send_subtree_cnt = curr_cnt - nbytes * mask;
                /* mask is also the size of this process's subtree */
                MPIR_PVAR_INC(scatter, binomial, send, send_subtree_cnt, MPI_BYTE);
                mpi_errno = MPIC_Send(((char *) tmp_buf + nbytes * mask),
                                         send_subtree_cnt, MPI_BYTE, dst,
                                         MPIR_SCATTER_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                curr_cnt -= send_subtree_cnt;
            }
            mask >>= 1;
        }

        /* copy local data into recvbuf */
        position = 0;
        if (recvbuf != MPI_IN_PLACE)
            MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position, recvbuf, recvcnt,
                             recvtype);
        MPIU_Free(tmp_buf);
    }
#endif                          /* MPID_HAS_HETERO */

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    MPIR_TIMER_END(coll,scatter,binomial);
    return (mpi_errno);
}

/* Added by abu naser */
/***********************/
/***********************/

/*
*
* use shared memory with Multi-leader + pipeline
*
*/
#if 1
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_HS2_ML_Pipeline
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_HS2_ML_Pipeline(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_HS2_ML_Pipeline\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    MPI_Comm shmem_comm, conc_comm;
    MPID_Comm *shmem_commptr, *conc_commptr = NULL;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    conc_comm = comm_ptr->dev.ch.concurrent_comm;

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
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(conc_comm, conc_commptr);

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    int pipeline_size = 64000; //global_pipeline_size; // define the pipeline size here
    int segments_no;
    unsigned int index_seg;
    int t = (sendtype_extent * sendcnt);
    if (t > pipeline_size)
    {
        segments_no = 1;
        segments_no += (t - pipeline_size - 1) / pipeline_size + 1;
    }
    else
    {
        segments_no = 1;
        pipeline_size = t;
    }

    comm_size = comm_ptr->local_size;
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);
    if(comm_size > segments_no)
    {  
        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");
    }
    else
    {
        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    segments_no * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    segments_no * sizeof (MPI_Status), mpi_errno,
                    "starray");
    }

    if (root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        memcpy(shmem_buffer, sendbuf, sendcnt * sendtype_extent * comm_size);
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }

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
        
        unsigned long max_out_len = (unsigned long)(16 + (sendtype_extent * sendcnt));
        unsigned long ciphertext_sendbuf_len;
       
        int each_rank_size = comm_size / local_size; // consider if equal now
        int max_ranks = local_rank * each_rank_size + each_rank_size;
       // printf("rank %d each_rank_size %d max_ranks %d\n", rank, each_rank_size, max_ranks);fflush(stdout);
        dest = 0;
        reqs = 0;
        //dest_global_rank = rank;
        dest_global_rank = rank + local_size;

        /* don't send to own process, instead copy later */
        for (i = 1; i < conc_size ; i++)
        {
            next = dest_global_rank * sendcnt * sendtype_extent; // get the encrytion location
            for (j = 0; j < segments_no; j++)
            {
                index_seg = (unsigned int)j;
                ciphertext_sendbuf[dest] = (index_seg >> 24) & 0xFF;
                ciphertext_sendbuf[dest + 1] = (index_seg >> 16) & 0xFF;
                ciphertext_sendbuf[dest + 2] = (index_seg >> 8) & 0xFF;
                ciphertext_sendbuf[dest + 3] = index_seg & 0xFF;
                RAND_bytes(ciphertext_sendbuf + dest + 4, 12);
                
                if (!EVP_AEAD_CTX_seal(global_ctx, (ciphertext_sendbuf + dest + 4 + 12),
                                       &ciphertext_sendbuf_len, max_out_len,
                                       (ciphertext_sendbuf + dest + 4), 12,
                                       (shmem_buffer + next +j*pipeline_size), pipeline_size,
                                       NULL, 0))
                {
                    printf("Error in encryption HS2-ML-Pipeline: [rank %d rank_node_id %d] scatter direct dest_global_rank %d conc_rank %d\n",
                           rank, rank_node_id, dest_global_rank, conc_rank);
                    fflush(stdout);
                }
#if 0
                else
                {

                    printf(COLOR_GREEN "Scatter HS2-ML-Pipeline:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest_global_rank %d dest %d] Encrypted data size %lu conc_rank %d" COLOR_RESET "\n",
                           rank, local_rank, rank_node_id, root_node_id, dest_global_rank, dest, ciphertext_sendbuf_len, conc_rank);
                    fflush(stdout);
                }
#endif

                //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
                //printf(COLOR_MAGENTA "HS2-ML Pipeline [rank %d conc_rank %d] sending to dest_rank %d dest-conc-rank %d j=%d dest %d next %d global_pipeline_size %d" COLOR_RESET "\n", rank, conc_rank, dest_global_rank, i, j,dest,next,global_pipeline_size);
                //fflush(stdout);
                mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                       (pipeline_size + 28 + 4),
                                       MPI_BYTE, i, MPIR_SCATTER_TAG, conc_commptr,
                                       &reqarray[reqs++], errflag);
                dest += (pipeline_size + 28 + 4);
            }
            dest_global_rank += local_size;

        }

        /* itner-node sending is done, now copy to own recv buffer */
        next = rank * t; 
        memcpy(recvbuf, shmem_buffer+next,t);
        //printf("*****HS2-ML [rank %d conc_rank %d] copied data %d to own process next %d\n",rank,conc_rank,t,next);fflush(stdout);
        
    }

     /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
         t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int source = rank % 16;
      
        if(rank_node_id != root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"HS2-ML rank %d receiving from source %d "COLOR_RESET"\n",rank,source);fflush(stdout);
            /*mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, 0,
                                  MPIR_SCATTER_TAG, conc_commptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
              /*  *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }*/
            //printf("[inter-node]rank %d received data from new_root %d\n", rank, new_root);
             if(t > pipeline_size)
            {
                segments_no = 1;
                segments_no += (t - pipeline_size-1) / pipeline_size + 1;
            }
            else
            {
                segments_no = 1;
                pipeline_size = t;

            }

            int next = 0;
            reqs = 0;
            for (j = 0; j < segments_no; j++)
            {
                mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, pipeline_size + 32, MPI_BYTE, 0,
                                       MPIR_SCATTER_TAG, conc_commptr, &reqarray[reqs++]);
                next += pipeline_size + 32;
            }

            next = 0;
            for (j = 0; j < segments_no; j++)
            {
                mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                index_seg = ((unsigned char)ciphertext_recvbuf[next+3] << 0) | ((unsigned char)ciphertext_recvbuf[next+2] << 8) | ((unsigned char)ciphertext_recvbuf[next+2] << 16) | ((unsigned char)ciphertext_recvbuf[next] << 24);
                next +=4;
                //printf("[inter-node]rank %d received data %d from source %d index_seg %d next %d  j=%d\n", rank, pipeline_size,source, index_seg,next,j);fflush(stdout);
                if (!EVP_AEAD_CTX_open(global_ctx, (recvbuf+index_seg*pipeline_size),
                                       &count, max_out_len,
                                       (ciphertext_recvbuf+next), 12,
                                       (ciphertext_recvbuf + next+ 12), (unsigned long)(pipeline_size + 16),
                                       NULL, 0))
                {
                    printf("Decryption error HS2-ML-pipeline:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                    fflush(stdout);
                }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter HS2-ML-pipeline recv:: [rank %d rank_node_id %d source %d] Decrypted data size %lu j %d index_seg %d" COLOR_RESET "\n",
                       rank, rank_node_id, source, count, j,index_seg);
                fflush(stdout);
            }
#endif
                next += pipeline_size + 28;
            }

   
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
    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif

/*
*
* use shared memory with Multi-leader: unencrypted version
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_HS2_ML
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_HS2_ML_UNENC(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_HS2_ML_UNENC\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
     if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1))
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    }     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm, conc_comm;
    MPID_Comm *shmem_commptr, *conc_commptr = NULL;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    conc_comm = comm_ptr->dev.ch.concurrent_comm;

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
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(conc_comm, conc_commptr);

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    comm_size = comm_ptr->local_size;
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);
       MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    if (root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        memcpy(shmem_buffer, sendbuf, sendcnt * sendtype_extent * comm_size);
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }

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

        /* don't send to own process, instead copy later */
        for (i = 1; i < conc_size ; i++)
        {
            next = dest_global_rank * sendcnt * sendtype_extent; // get the encrytion location

               //printf(COLOR_MAGENTA"HS2-ML [rank %d conc_rank %d] sending to dest_rank %d dest conc rank %d"COLOR_RESET"\n",rank, conc_rank, dest_global_rank,i);fflush(stdout);
                    mpi_errno = MPIC_Isend((shmem_buffer + next),
                                           t,
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, conc_commptr,
                                           &reqarray[reqs++], errflag);
            dest_global_rank += local_size;

        }

        /* itner-node sending is done, now copy to own recv buffer */
        next = rank * t; 
        memcpy(recvbuf, shmem_buffer+next,t);
        //printf("*****HS2-ML [rank %d conc_rank %d] copied data %d to own process next %d\n",rank,conc_rank,t,next);fflush(stdout);
        
    }

     /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int source = rank % local_size;

        if(rank_node_id != root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"HS2-ML-UNENC rank %d receiving from source %d "COLOR_RESET"\n",rank,source);fflush(stdout);
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

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/*
*
* use shared memory with Multi-leader CHS
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_CHS
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_CHS(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_CHS\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
     if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0)
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    }     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm, conc_comm;
    MPID_Comm *shmem_commptr, *conc_commptr = NULL;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank, conc_rank, conc_size;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    conc_comm = comm_ptr->dev.ch.concurrent_comm;

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
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    MPID_Comm_get_ptr(conc_comm, conc_commptr);

    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    comm_size = comm_ptr->local_size;
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);
       MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    if (root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        memcpy(shmem_buffer, sendbuf, sendcnt * sendtype_extent * comm_size);
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }

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

        /* don't send to own process, instead copy later */
        for (i = 1; i < conc_size ; i++)
        {
            next = dest_global_rank * sendcnt * sendtype_extent; // get the encrytion location

            RAND_bytes(ciphertext_sendbuf + dest, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_sendbuf + dest, 12,
                                   shmem_buffer + next, t,
                                   NULL, 0))
            {
                printf("Error in encryption HS2-ML: [rank %d rank_node_id %d] scatter direct dest_global_rank %d conc_rank %d\n",
                       rank, rank_node_id, dest_global_rank, conc_rank);
                fflush(stdout);
            }
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
                    mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                           (t + 28),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, conc_commptr,
                                           &reqarray[reqs++], errflag);
           dest += (t + 28);
           dest_global_rank += local_size;

        }

        /* itner-node sending is done, now copy to own recv buffer */
        next = rank * t; 
        memcpy(recvbuf, shmem_buffer+next,t);
        //printf("*****HS2-ML [rank %d conc_rank %d] copied data %d to own process next %d\n",rank,conc_rank,t,next);fflush(stdout);
        
    }

     /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
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
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, 0,
                                  MPIR_SCATTER_TAG, conc_commptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (mpi_errno)
            {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
           // printf("[inter-node]rank %d received data from source %d\n", rank, source);
            if (!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf), 12,
                                   (ciphertext_recvbuf + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error HS2-ML:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                fflush(stdout);
            }
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

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/***************/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_HBcast
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_HBcast(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_HBcast\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
 if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0)
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    }     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs, t;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPID_Node_id_t rank_node_id, root_node_id, leader_root_node_id, leader_of_root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
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
                printf("Decryption error HBcast:[rank %d rank_node_id %d] scatter direct recv\n", rank, rank_node_id);
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


   

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/***************/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_HS2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_HS2(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_HS2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    int leader_comm_rank, leader_comm_size;
    int leader_root, leader_of_root = -1;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr, *leader_commptr = NULL;
    MPID_Node_id_t rank_node_id, root_node_id, leader_root_node_id, leader_of_root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    int local_size, local_rank;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
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
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);

   

    if (local_rank == 0)
    {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        leader_comm = comm_ptr->dev.ch.leader_comm;
        mpi_errno = PMPI_Comm_rank(leader_comm, &leader_comm_rank);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(leader_comm, &leader_comm_size);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
        }
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        
    }

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    /* leader_of_root is the global rank of the leader of the root */
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];

   
    

    //printf("rank %d root %d rank_node_id %d root_node_id %d\n",rank,root, rank_node_id,root_node_id);fflush(stdout);

    comm_size = comm_ptr->local_size;
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);

    if (root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        memcpy(shmem_buffer, sendbuf, sendcnt * sendtype_extent * comm_size);
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);
    }

    int next = 0;
    int dest = 0;
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

        for (i = local_rank * each_rank_size; i < max_ranks; i++)
        {
            next = i * sendcnt * sendtype_extent; // get the encrytion location
            dest = next + i * 28;

            RAND_bytes(ciphertext_shmem_buffer + dest, 12);
            if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_shmem_buffer + dest + 12,
                                   &ciphertext_sendbuf_len, max_out_len,
                                   ciphertext_shmem_buffer + dest, 12,
                                   shmem_buffer + next, t,
                                   NULL, 0))
            {
                printf("Error in encryption: [rank %d rank_node_id %d] scatter direct dest %d dest %d\n",
                       rank, rank_node_id, i, dest);
                fflush(stdout);
            }
#if 0
            else
            {

                printf(COLOR_GREEN "Scatter H2:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest %d dest %d] Encrypted data size %lu max_ranks %d" COLOR_RESET "\n",
                       rank, local_rank, rank_node_id, root_node_id, i, dest, ciphertext_sendbuf_len, max_ranks);
                fflush(stdout);
            }
#endif

        }
        /* syc to finish encryption */
        mpi_errno = MPIR_Barrier_impl(shmem_commptr, errflag);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }
    }

    /* scatter among leaders */
    if (leader_comm_size > 1 && local_rank == 0)
    {
        if (comm_ptr->dev.ch.is_uniform != 1)
        {
            printf("Un-uniform not implemented yet\n");
            fflush(stdout);
        }
        else // when uniform
        {
            if (leader_of_root != root)
            {
                printf("Not implemented yet leader_of_root != root\n");
                fflush(stdout);
            }
            else
            {
               MPID_Get_node_id(comm_ptr, leader_root, &leader_root_node_id);
                MPID_Get_node_id(comm_ptr, leader_of_root, &leader_of_root_node_id);
                //printf("rank %d leader_comm_rank %d leader_root %d leader_of_root %d leader_root_node_id %d leader_of_root_node_id %d\n",
                //rank, leader_comm_rank, leader_root,leader_of_root,leader_root_node_id,leader_of_root_node_id);fflush(stdout);
                mpi_errno =
                    MPIR_Scatter_MV2_Direct_unencrypted(ciphertext_shmem_buffer,
                                                        ((sendtype_extent * sendcnt * local_size) + (local_size * 28)), MPI_BYTE,
                                                        shmem_buffer, ((recvtype_extent * recvcnt * local_size) + local_size * 28),
                                                        MPI_BYTE, leader_root,
                                                        leader_commptr, errflag);
            }
        }
    }

     mpi_errno = MPIR_Barrier_impl(comm_ptr, errflag);
        if (mpi_errno)
        {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }

     /* decrypt and copy data to receive buffer */
     dest = local_rank*recvtype_extent * recvcnt+ local_rank*28;
     unsigned long max_out_len = recvtype_extent * recvcnt;
     unsigned long count;
      if (!EVP_AEAD_CTX_open(global_ctx, recvbuf,
                                   &count, max_out_len,
                                   (shmem_buffer+dest), 12,
                                   (shmem_buffer + dest + 12), (unsigned long)(recvtype_extent * recvcnt + 16),
                                   NULL, 0))
            {
                printf("Decryption error:[rank %d local_rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, local_rank, rank_node_id,root_node_id);
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter H2 recv:: [rank %d local_rank %d rank_node_id %d ] Decrypted data size %lu" COLOR_RESET "\n",
                       rank, local_rank,rank_node_id, count);
                fflush(stdout);
            }  
#endif             

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/*
*
* no share memory, multi-leader, and round robin unencrypted
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0)
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    } 
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm;
    //MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
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
       // MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
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
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
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
                mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++], errflag);
                //printf("**[intra-node] rank %d sending to rank %d location: %d, (i*each_rank_size*t = %d) (j*t = %d)\n",rank,i,next,i*each_rank_size*t,j*t);fflush(stdout);

            }
        }
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        int next = 0;
        
        /*mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);*/  
        for(i=0;i<each_rank_size;i++)
        {
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf+next, t, MPI_BYTE, root,
                  i, comm_ptr,  &reqarray[reqs++]);
            next += t;      
        }                
    }

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
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
            MPID_Node_id_t dest_node_id;
           
            j = 0;
            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
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
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                    j++;
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
                    {
                        mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                        j++;
                       temp_buffer = ciphertext_recvbuf;    
                    }
                    //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
                    mpi_errno = MPIC_Isend((temp_buffer + dest),
                                           (t),
                                           MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                           &reqarray[reqs++], errflag);

                    dest += t; // update after send
                   
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

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
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
        }
        else if (rank_node_id == new_root_node_id)*/
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
            //printf(COLOR_YELLOW"CRR-UNE:rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Recv(recvbuf, (t), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
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
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
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
    }

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/*
*
* no share memory, multi-leader, and round robin
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem_intra_RR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem_intra_RR(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_no_shmem_intra_RR\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
    if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0)
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    }     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm;
    //MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
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
       // MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
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
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
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
                mpi_errno = MPIC_Isend((sendbuf + next),
                                           t,
                                           MPI_BYTE, i, j, comm_ptr,
                                           &reqarray[reqs++], errflag);
                //printf("**[intra-node] rank %d sending to rank %d location: %d, (i*each_rank_size*t = %d) (j*t = %d)\n",rank,i,next,i*each_rank_size*t,j*t);fflush(stdout);

            }
        }
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        int next = 0;
        
        /*mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);*/  
        for(i=0;i<each_rank_size;i++)
        {
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf+next, t, MPI_BYTE, root,
                  i, comm_ptr,  &reqarray[reqs++]);
            next += t;      
        }                
    }

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
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
            MPID_Node_id_t dest_node_id;
           
            j = 0;
            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
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
                    mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                    j++;
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
                    {
                        mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
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

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
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
        }
        else if (rank_node_id == new_root_node_id)*/
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
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
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

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
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
    }

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/*
*
* no share memory and multi-leader and pipeline for inter-comm
*
*/
#if 1
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem_pipeline
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem_pipeline(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_no_shmem_pipeline\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
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
    
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
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
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        
        mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);          
    }

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    int pipeline_size = 64000; //global_pipeline_size;
    unsigned int index_seg;
    int segments_no;
    int j;
    
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
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
            MPID_Node_id_t dest_node_id;

            if(t > pipeline_size)
            {
                segments_no = 1;
                segments_no += (t - pipeline_size-1) / pipeline_size + 1;
            }
            else
            {
                segments_no = 1;
                pipeline_size = t;

            }
           

            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
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

                    for (j = 0; j < segments_no; j++)
                    {
                        index_seg = (unsigned int) j;
                        ciphertext_sendbuf[dest] = (index_seg >> 24) & 0xFF;
                        ciphertext_sendbuf[dest+1] = (index_seg >> 16) & 0xFF;
                        ciphertext_sendbuf[dest+2] = (index_seg >> 8) & 0xFF;
                        ciphertext_sendbuf[dest+3] = index_seg & 0xFF;
                        
                        RAND_bytes(ciphertext_sendbuf + dest + 4, 12);
                        
                        if (!EVP_AEAD_CTX_seal(global_ctx, (ciphertext_sendbuf + dest + 4 + 12),
                                               &ciphertext_sendbuf_len, max_out_len,
                                               (ciphertext_sendbuf + dest + 4), 12,
                                               (temp_buffer + next+j*pipeline_size), pipeline_size,
                                               NULL, 0))
                        {
                            printf("Error in encryption: [rank %d rank_node_id %d] scatter direct dest %d dest_node_id %d\n",
                                   rank, rank_node_id, i, dest_node_id);
                            fflush(stdout);
                        }
#if 0
                    else
                    {

                        printf(COLOR_GREEN "Scatter no-shm-pipeline direct:: [rank %d local_rank %d rank_node_id %d root_node_id %d dest %d dest_node_id %d j=%d] Encrypted data size %lu dest %d" COLOR_RESET "\n",
                               rank, local_rank, rank_node_id, root_node_id, i, dest_node_id, j, ciphertext_sendbuf_len, dest);
                        fflush(stdout);
                    }
#endif
                        //  MPIR_PVAR_INC(scatter, direct, send, (t+28), MPI_BYTE);
                        mpi_errno = MPIC_Isend((ciphertext_sendbuf + dest),
                                               (pipeline_size + 28 + 4),
                                               MPI_BYTE, i, MPIR_SCATTER_TAG, comm_ptr,
                                               &reqarray[reqs++], errflag);

                        dest += pipeline_size + 16 + 12 + 4; // update after send
                    }

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

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
        int new_root = 0;
        //if(new_root >= local_size)
        //    new_root = local_size - 1;
        int each_rank_size = comm_size / leader_size;
        new_root = rank / each_rank_size;          
        MPID_Node_id_t new_root_node_id;
        MPID_Get_node_id(comm_ptr, new_root, &new_root_node_id);


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
           /* mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);*/
            
            if(t > pipeline_size)
            {
                segments_no = 1;
                segments_no += (t - pipeline_size-1) / pipeline_size + 1;
            }
            else
            {
                segments_no = 1;
                pipeline_size = t;

            }

            int next = 0;
            reqs = 0;
            for (j = 0; j < segments_no; j++)
            {
                mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, pipeline_size + 32, MPI_BYTE, new_root,
                                       MPIR_SCATTER_TAG, comm_ptr, &reqarray[reqs++]);
                next += pipeline_size + 32;
            }

            next = 0;
            for (j = 0; j < segments_no; j++)
            {
                mpi_errno = MPIC_Waitall(1, &reqarray[j], starray, errflag);
                index_seg = ((unsigned char)ciphertext_recvbuf[next+3] << 0) | ((unsigned char)ciphertext_recvbuf[next+2] << 8) | ((unsigned char)ciphertext_recvbuf[next+2] << 16) | ((unsigned char)ciphertext_recvbuf[next] << 24);
                next +=4;
                //printf("[inter-node]rank %d received data %d from new_root %d index_seg %d next %d\n", rank, pipeline_size,new_root, index_seg,next);fflush(stdout);
                if (!EVP_AEAD_CTX_open(global_ctx, (recvbuf+index_seg*pipeline_size),
                                       &count, max_out_len,
                                       (ciphertext_recvbuf+next), 12,
                                       (ciphertext_recvbuf + next+ 12), (unsigned long)(pipeline_size + 16),
                                       NULL, 0))
                {
                    printf("Decryption error noshm-pipeline:[rank %d rank_node_id %d root_node_id %d] scatter direct recv\n", rank, rank_node_id);
                    fflush(stdout);
                }
#if 0
            else
            {
                printf(COLOR_BLUE "Scatter no-shm-pipeline recv:: [rank %d rank_node_id %d new_root %d] Decrypted data size %lu j %d index_seg %d" COLOR_RESET "\n",
                       rank, rank_node_id, new_root, count, j,index_seg);
                fflush(stdout);
            }
#endif
                next += pipeline_size + 28;
            }
        }
    }

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
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
    }

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif
/*
*
* no share memory and multi-leader (204)
*
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_no_shmem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_no_shmem(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_no_shmem\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
     if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0 )
    {
        //printf("call MPIR_Scatter_MV2_Direct\n");fflush(stdout);
        int ret = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm_ptr, errflag);
        return ret;
    }       
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm;
    //MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
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
       // MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    //}
    
    comm_size = comm_ptr->local_size;
     MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    
    reqs = 0;
    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
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
                               
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }
    else if(root != rank && rank_node_id == root_node_id)
    { // receive data from root
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        int t = (sendtype_extent * sendcnt);
        int each_rank_size = comm_size / local_size;
        
        mpi_errno = MPIC_Recv(ciphertext_recvbuf, t*each_rank_size, MPI_BYTE, root, MPIR_SCATTER_TAG,
                comm_ptr, MPI_STATUS_IGNORE, errflag);          
    }

    /* multi-leader senders */
    int leader_size = local_size; 
    unsigned char * temp_buffer;
    if(rank_node_id == root_node_id && local_size != comm_size)
    {

        if (local_rank < leader_size)
        {
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
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
            MPID_Node_id_t dest_node_id;
           

            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
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

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
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
        }
        else if (rank_node_id == new_root_node_id)*/
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
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
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

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
       // printf("--->> rank %d waiting to finish send\n",rank);fflush(stdout);
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
    }

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}



/*
* use a shared memory to divide the encryption among ppn processes in the root node
*/
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_shmem
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_shmem(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_shmem\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Node_id_t rank_node_id, root_node_id;
    MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);  
    int local_size, local_rank;
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
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);

    if(root == rank && local_size != comm_size)
    {
        //comm_size = comm_ptr->remote_size;
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        memcpy(shmem_buffer, sendbuf, sendcnt*sendtype_extent*comm_size);                           
        //printf("root = %d rank %d comm_ptr->comm_kind %d comm_size*sendcnt*sendtype_extent %d\n",root, rank, comm_ptr->comm_kind,comm_size*sendcnt*sendtype_extent);fflush(stdout);

    }

     MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

    /* multi-leader senders */
    int leader_size = local_size; 
    /*leader_size = local_size / 2;
    if(leader_size > 8)
        leader_size = 8;*/
    if(rank_node_id == root_node_id && local_size != comm_size)
    {
        mpi_errno = MPIR_Barrier_impl(shmem_commptr, errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
            goto fn_fail;
        }

        if (local_rank < leader_size)
        {
            MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
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
            MPID_Node_id_t dest_node_id;
            reqs = 0;

            for (i = local_rank * each_rank_size; i < max_dest_ranks; i++)
            {
                next = i * sendcnt * sendtype_extent; // get the encrytion location
                MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                if (rank == i)
                {
                    memcpy(recvbuf, shmem_buffer + next, t);
                    //printf("copy to own rank %d == i %d\n", rank, i);
                }
                else if (rank_node_id == dest_node_id) // sending to same node
                {
                    mpi_errno = MPIC_Isend((shmem_buffer + next),
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
                else
                {

                    RAND_bytes(ciphertext_sendbuf + dest, 12);
                    if (!EVP_AEAD_CTX_seal(global_ctx, ciphertext_sendbuf + dest + 12,
                                           &ciphertext_sendbuf_len, max_out_len,
                                           ciphertext_sendbuf + dest, 12,
                                           shmem_buffer + next, t,
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
            }
        }
    }

    /* receiver side*/
    if (local_size != comm_size)
    {
        MPI_Aint recvtype_extent;
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        int t = (recvtype_extent * recvcnt);
        unsigned long max_out_len = (unsigned long)(recvtype_extent * recvcnt);
        unsigned long count;
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
        }
        else if (rank_node_id == new_root_node_id)*/
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
           // printf("rank %d received from same node new_root %d\n",rank, new_root);fflush(stdout);                      
        }
        else if(rank_node_id != new_root_node_id)
        {
           // printf("rank %d rank_node_id %d posting receive from new_root %d new_root_node_id %d\n",rank,rank_node_id,new_root,new_root_node_id);fflush(stdout);
            //printf(COLOR_YELLOW"rank %d receiving from new root %d"COLOR_RESET"\n",rank, new_root);fflush(stdout);
            mpi_errno = MPIC_Recv(ciphertext_recvbuf, (t + 28), MPI_BYTE, new_root,
                                  MPIR_SCATTER_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
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

    /* wait for finish sending */
    if(rank_node_id == root_node_id && local_size != comm_size && local_rank < leader_size)
    {
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
    }

    //printf("rank %d root %d rank_node_id %d root_node_id %d comm_ptr->comm_kind %d\n", rank, root,rank_node_id,root_node_id,comm_ptr->comm_kind);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/* end of add */

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
    
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPID_INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         sendtype_extent);

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

        /* added by abu naser */
        int j = 0;
        int next = 0;
        int dest = 0;
        int t= (sendtype_extent * sendcnt);
        unsigned long   max_out_len = (unsigned long) (16 + (sendtype_extent * sendcnt));
        unsigned long ciphertext_sendbuf_len;
        MPID_Node_id_t rank_node_id;
        MPID_Get_node_id(comm_ptr, rank, &rank_node_id);   
        /* end of add */
        
        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcnt) {
            if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
                if (recvbuf != MPI_IN_PLACE) {
                mpi_errno =
                    MPIR_Localcopy(((char *) sendbuf +
                            rank * sendcnt * sendtype_extent),
                           sendcnt, sendtype, recvbuf, recvcnt,
                           recvtype);
                }
            } else {
                MPIR_PVAR_INC(scatter, direct, send, sendcnt, sendtype);
                    
                    /* added by abu naser naive+ */
                    
                    MPID_Node_id_t dest_node_id;
                    MPID_Get_node_id(comm_ptr, i, &dest_node_id);
                    //printf("MV2_Direct send:[root %d rank %d rank_node_id %d] i = %d dest_node_id %d\n",root,rank,rank_node_id,i,dest_node_id);fflush(stdout);

                    if((security_approach == 200 || (security_approach > 200 && security_approach <= 211)) && rank_node_id != dest_node_id)
                    {
                        //printf("leader_comm_size = %d local_size = %d\n",leader_comm_size,local_size);fflush(stdout);
                        
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
            MPIR_PVAR_INC(scatter, direct, recv, recvcnt, recvtype);
            /* added by abu naser */
            
             MPID_Node_id_t rank_node_id;
            MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
            MPID_Node_id_t root_node_id;
            MPID_Get_node_id(comm_ptr, i, &root_node_id);
            //printf("MV2_Direct recv:[root %d rank %d rank_node_id %d] i = %d root_node_id %d\n",root,rank,rank_node_id,i,root_node_id);fflush(stdout);
            if((security_approach == 200 || (security_approach > 200 && security_approach <= 211)) && rank_node_id != root_node_id)
            {
                int j = 0;
                int next = 0;
                int dest = 0;
                MPI_Aint recvtype_extent;
                MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
                int t = (recvtype_extent * recvcnt);
                unsigned long   max_out_len = (unsigned long) (recvtype_extent * recvcnt);
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

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#if 1
/* added by abu naser */
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_unencrypted
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_unencrypted(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_unencrypted\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
    MPIR_TIMER_START(coll,scatter,direct);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPID_INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         sendtype_extent);

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");

        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (sendcnt) {
            if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
                if (recvbuf != MPI_IN_PLACE) {
                mpi_errno =
                    MPIR_Localcopy(((char *) sendbuf +
                            rank * sendcnt * sendtype_extent),
                           sendcnt, sendtype, recvbuf, recvcnt,
                           recvtype);
                }
            } else {
                MPIR_PVAR_INC(scatter, direct, send, sendcnt, sendtype);
                
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
            MPIR_PVAR_INC(scatter, direct, recv, recvcnt, recvtype);
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

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,scatter,direct);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
/* end of add */
#endif

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_Direct_Blk
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_Direct_Blk(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_Direct_Blk\n",comm_ptr->rank,hostname);fflush(stdout);
#endif         
    MPIR_TIMER_START(coll,scatter,direct_blk);
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent;
    int i;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_direct_blk, 1);

    rank = comm_ptr->rank;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* If I'm the root, then scatter */
    if (((comm_ptr->comm_kind == MPID_INTRACOMM) && (root == rank)) ||
        ((comm_ptr->comm_kind == MPID_INTERCOMM) && (root == MPI_ROOT))) {
        if (comm_ptr->comm_kind == MPID_INTRACOMM)
            comm_size = comm_ptr->local_size;
        else
            comm_size = comm_ptr->remote_size;

        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        /* We need a check to ensure extent will fit in a
         * pointer. That needs extent * (max count) but we can't get
         * that without looping over the input data. This is at least
         * a minimal sanity check. Maybe add a global var since we do
         * loop over sendcount[] in MPI_Scatterv before calling
         * this? */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         sendtype_extent);

        for (i = 0; i < comm_size; i++) {
            if (sendcnt) {
                if ((comm_ptr->comm_kind == MPID_INTRACOMM) && (i == rank)) {
                    if (recvbuf != MPI_IN_PLACE) {
                    mpi_errno =
                        MPIR_Localcopy(((char *) sendbuf +
                                rank * sendcnt * sendtype_extent),
                               sendcnt, sendtype, recvbuf, recvcnt,
                               recvtype);
                        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    }
                } else {
                    MPIR_PVAR_INC(scatter, direct_blk, send, sendcnt, sendtype);
                    mpi_errno =
                    MPIC_Send(((char *) sendbuf +
                               i * sendcnt * sendtype_extent), sendcnt,
                              sendtype, i, MPIR_SCATTER_TAG, comm_ptr,
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
    } else if (root != MPI_PROC_NULL) {   /* non-root nodes, and in the intercomm. case, non-root nodes on remote side */
        if (recvcnt) {
            MPIR_PVAR_INC(scatter, direct_blk, recv, recvcnt, recvtype);
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

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    MPIR_TIMER_END(coll,scatter,direct_blk);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_two_level_Binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_two_level_Binomial(const void *sendbuf,
                                        int sendcnt,
                                        MPI_Datatype sendtype,
                                        void *recvbuf,
                                        int recvcnt,
                                        MPI_Datatype recvtype,
                                        int root, MPID_Comm * comm_ptr,
                                        MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_two_level_Binomial\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
    MPIR_TIMER_START(coll,scatter,two_level_binomial);
    int comm_size, rank;
    int local_rank, local_size;
    int leader_comm_rank, leader_comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_size, sendtype_size, nbytes;
    void *tmp_buf = NULL;
    void *leader_scatter_buf = NULL;
    MPI_Status status;
    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    int leader_root = -1, leader_of_root = -1;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr, *leader_commptr = NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_two_level_binomial, 1);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcnt == 0))
        || ((rank != root) && (recvcnt == 0))) {

        MPIR_TIMER_END(coll,scatter,two_level_binomial);
        return MPI_SUCCESS;
    }
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    /* extract the rank,size information for the intra-node
     * communicator */
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

    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        leader_comm = comm_ptr->dev.ch.leader_comm;
        mpi_errno = PMPI_Comm_rank(leader_comm, &leader_comm_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(leader_comm, &leader_comm_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
    }

    if (local_size == comm_size) {
        /* purely intra-node scatter. Just use the direct algorithm and we are done */
        mpi_errno = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype,
                                            recvbuf, recvcnt, recvtype,
                                            root, comm_ptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    } else {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);

        if (rank == root) {
            nbytes = sendcnt * sendtype_size;
        } else {
            nbytes = recvcnt * recvtype_size;
        }

        if (local_rank == 0) {
            /* Node leader, allocate tmp_buffer */
            tmp_buf = MPIU_Malloc(nbytes * local_size);
        }

        leader_of_root = comm_ptr->dev.ch.leader_map[root];
        /* leader_of_root is the global rank of the leader of the root */
        leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
        /* leader_root is the rank of the leader of the root in leader_comm.
         * leader_root is to be used as the root of the inter-leader gather ops
         */

        if ((local_rank == 0) && (root != rank)
            && (leader_of_root == rank)) {
            /* The root of the scatter operation is not the node leader. Recv
             * data from the node leader */
            leader_scatter_buf = MPIU_Malloc(nbytes * comm_size);
            MPIR_PVAR_INC(scatter, two_level_binomial, recv, nbytes * comm_size, MPI_BYTE);
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
            MPIR_PVAR_INC(scatter, two_level_binomial, send, sendcnt * comm_size, sendtype);
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
            if (comm_ptr->dev.ch.is_uniform != 1) {
                int *displs = NULL;
                int *sendcnts = NULL;
                int *node_sizes;
                int i = 0;
                node_sizes = comm_ptr->dev.ch.node_sizes;

                if (root != leader_of_root) {
                    if (leader_comm_rank == leader_root) {
                        displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts[0] = node_sizes[0] * nbytes;
                        displs[0] = 0;

                        for (i = 1; i < leader_comm_size; i++) {
                            displs[i] =
                                displs[i - 1] + node_sizes[i - 1] * nbytes;
                            sendcnts[i] = node_sizes[i] * nbytes;
                        }
                    }
                    mpi_errno =
                        MPIR_Scatterv(leader_scatter_buf, sendcnts, displs,
                                      MPI_BYTE, tmp_buf, nbytes * local_size,
                                      MPI_BYTE, leader_root, leader_commptr,
                                      errflag);
                } else {
                    if (leader_comm_rank == leader_root) {
                        displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts[0] = node_sizes[0] * sendcnt;
                        displs[0] = 0;

                        for (i = 1; i < leader_comm_size; i++) {
                            displs[i] =
                                displs[i - 1] + node_sizes[i - 1] * sendcnt;
                            sendcnts[i] = node_sizes[i] * sendcnt;
                        }
                    }
                    mpi_errno = MPIR_Scatterv(sendbuf, sendcnts, displs,
                                              sendtype, tmp_buf,
                                              nbytes * local_size, MPI_BYTE,
                                              leader_root, leader_commptr,
                                              errflag);
                }
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                if (leader_comm_rank == leader_root) {
                    MPIU_Free(displs);
                    MPIU_Free(sendcnts);
                }
            } else {
                if (leader_of_root != root) {
                    mpi_errno =
                        MPIR_Scatter_MV2_Binomial(leader_scatter_buf,
                                                  nbytes * local_size, MPI_BYTE,
                                                  tmp_buf, nbytes * local_size,
                                                  MPI_BYTE, leader_root,
                                                  leader_commptr, errflag);
                } else {
                    mpi_errno =
                        MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt * local_size,
                                                  sendtype, tmp_buf,
                                                  nbytes * local_size, MPI_BYTE,
                                                  leader_root, leader_commptr,
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
        /* The leaders are now done with the inter-leader part. Scatter the data within the nodes */

        if (rank == root && recvbuf == MPI_IN_PLACE) {
            mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
                                                (void *)sendbuf, sendcnt, sendtype,
                                                0, shmem_commptr, errflag);
        } else {
            mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
                                                recvbuf, recvcnt, recvtype,
                                                0, shmem_commptr, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

  fn_fail:
    /* check if multiple threads are calling this collective function */
    if (comm_size != local_size && local_rank == 0) {
        MPIU_Free(tmp_buf);
        if (leader_of_root == rank && root != rank) {
            MPIU_Free(leader_scatter_buf);
        }
    }
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    MPIR_TIMER_END(coll,scatter,two_level_binomial);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2_two_level_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2_two_level_Direct(const void *sendbuf,
                                      int sendcnt,
                                      MPI_Datatype sendtype,
                                      void *recvbuf,
                                      int recvcnt,
                                      MPI_Datatype recvtype,
                                      int root, MPID_Comm * comm_ptr,
                                      MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2_two_level_Direct\n",comm_ptr->rank,hostname);fflush(stdout);
#endif
    
    MPIR_TIMER_START(coll,scatter,two_level_direct);
    int comm_size, rank;
    int local_rank, local_size;
    int leader_comm_rank, leader_comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_size, sendtype_size, nbytes;
    void *tmp_buf = NULL;
    void *leader_scatter_buf = NULL;
    MPI_Status status;
    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    int leader_root, leader_of_root = -1;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr, *leader_commptr = NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_scatter_two_level_direct, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcnt == 0))
        || ((rank != root) && (recvcnt == 0))) {

        MPIR_TIMER_END(coll,scatter,two_level_direct);
        return MPI_SUCCESS;
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    /* extract the rank,size information for the intra-node
     * communicator */
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

    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        leader_comm = comm_ptr->dev.ch.leader_comm;
        mpi_errno = PMPI_Comm_rank(leader_comm, &leader_comm_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(leader_comm, &leader_comm_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
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
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);

        if (rank == root) {
            nbytes = sendcnt * sendtype_size;
        } else {
            nbytes = recvcnt * recvtype_size;
        }

        if (local_rank == 0) {
            /* Node leader, allocate tmp_buffer */
            tmp_buf = MPIU_Malloc(nbytes * local_size);
        }

        leader_of_root = comm_ptr->dev.ch.leader_map[root];
        /* leader_of_root is the global rank of the leader of the root */
        leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
        /* leader_root is the rank of the leader of the root in leader_comm.
         * leader_root is to be used as the root of the inter-leader gather ops
         */
        
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

            leader_scatter_buf = MPIU_Malloc(nbytes * comm_size);
            MPIR_PVAR_INC(scatter, two_level_direct, recv, nbytes * comm_size, MPI_BYTE);
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
            
            MPIR_PVAR_INC(scatter, two_level_direct, send, sendcnt * comm_size, sendtype);
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
            if (comm_ptr->dev.ch.is_uniform != 1) {
                int *displs = NULL;
                int *sendcnts = NULL;
                int *node_sizes;
                int i = 0;
                node_sizes = comm_ptr->dev.ch.node_sizes;

                if (root != leader_of_root) {
                    if (leader_comm_rank == leader_root) {
                        displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
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
                        displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                        sendcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
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
                    MPIU_Free(displs);
                    MPIU_Free(sendcnts);
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
                        //printf("leader_comm_size = %d local_size = %d\n",leader_comm_size,local_size);fflush(stdout);
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
                            printf("Decryption error: scatter two-level-direct\n");fflush(stdout);        
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
                        //printf("leader_comm_size = %d local_size = %d\n",leader_comm_size,local_size);fflush(stdout);
                     
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
                            printf("Decryption error: scatter two-level-direct\n");fflush(stdout);        
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
            mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
                                                (void *)sendbuf, sendcnt, sendtype,
                                                0, shmem_commptr, errflag);
        } else {
            mpi_errno = MV2_Scatter_intra_function(tmp_buf, nbytes, MPI_BYTE,
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

  fn_fail:
    /* check if multiple threads are calling this collective function */
    if (comm_size != local_size && local_rank == 0) {
        MPIU_Free(tmp_buf);
        if (leader_of_root == rank && root != rank) {
            MPIU_Free(leader_scatter_buf);
        }
    }
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    MPIR_TIMER_END(coll,scatter,two_level_direct);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_index_tuned_intra_MV2(const void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_index_tuned_intra_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int rank, comm_size;
    MPI_Aint nbytes;
    MPI_Aint recvtype_size, sendtype_size;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int local_size = -1;
    int comm_size_index = 0;
    int inter_node_algo_index = 0;
    int intra_node_algo_index = 0;
    int table_min_comm_size = 0;
    int table_max_comm_size = 0;
    int table_min_inter_size = 0;
    int table_max_inter_size = 0;
    int table_min_intra_size = 0;
    int table_max_intra_size = 0;
    int last_inter;
    int last_intra;
    int lp2ltn; // largest power of 2 less than n
    int lp2ltn_min;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr=NULL;
    MPIU_THREADPRIV_DECL;

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    mpi_errno = PMPI_Comm_size(comm_ptr->handle, &comm_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    MPIU_THREADPRIV_GET;

    if (rank == root) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcnt * sendtype_size;
    } else {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
    }

    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        if (mv2_scatter_indexed_table_ppn_conf[0] == -1) {
            /* Indicating user defined tuning */
            conf_index = 0;
            goto conf_check_end;
        }
        FIND_PPN_INDEX  (scatter, local_size,conf_index, partial_sub_ok)
    }
    
    if (partial_sub_ok != 1) {
        conf_index = mv2_scatter_indexed_num_ppn_conf/2;
    }

conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_scatter_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_scatter_indexed_thresholds_table[conf_index][mv2_size_scatter_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
	/* Comm size smaller than smallest configuration in table: use smallest available */
	comm_size_index = 0;
    }
    else if (comm_size > table_max_comm_size) {
	/* Comm size larger than largest configuration in table: use largest available */
	comm_size_index = mv2_size_scatter_indexed_tuning_table[conf_index] - 1;
    }
    else {
	/* Comm size in between smallest and largest configuration: find closest match */
    lp2ltn_min = pow(2, (int)log2(table_min_comm_size));
	if (comm_ptr->dev.ch.is_pof2) {
	    comm_size_index = log2( comm_size / lp2ltn_min );
	}
	else {
	    lp2ltn = pow(2, (int)log2(comm_size));
	    comm_size_index = (lp2ltn < lp2ltn_min) ? 0 : log2( lp2ltn / lp2ltn_min );
	}
    }

    last_inter = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
    table_min_inter_size = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
    table_max_inter_size = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[last_inter].msg_sz;
    last_intra = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].size_intra_table - 1;
    table_min_intra_size = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].intra_node[0].msg_sz;
    table_max_intra_size = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].intra_node[last_intra].msg_sz;
    
    if (nbytes < table_min_inter_size) {
	/* Msg size smaller than smallest configuration in table: use smallest available */
	inter_node_algo_index = 0;
    }
    else if (nbytes > table_max_inter_size) {
	/* Msg size larger than largest configuration in table: use largest available */
	inter_node_algo_index = last_inter;
    }
    else {
	/* Msg size in between smallest and largest configuration: find closest match */
	if (pow(2, (int)log2(nbytes)) == nbytes) {
	    inter_node_algo_index = log2( nbytes / table_min_inter_size );
	}
	else {
	    lp2ltn = pow(2, (int)log2(nbytes));
	    inter_node_algo_index = (lp2ltn < table_min_inter_size) ? 0 : log2( lp2ltn / table_min_inter_size );
	}
    }
    
    if (nbytes < table_min_intra_size) {
	/* Msg size smaller than smallest configuration in table: use smallest available */
	intra_node_algo_index = 0;
    }
    else if (nbytes > table_max_intra_size) {
	/* Msg size larger than largest configuration in table: use largest available */
	intra_node_algo_index = last_intra;
    }
    else {
	/* Msg size in between smallest and largest configuration: find closest match */
	if (pow(2, (int)log2(nbytes)) == nbytes) {
	    intra_node_algo_index = log2(nbytes / table_min_intra_size );
	}
	else {
	    lp2ltn = pow(2, (int)log2(nbytes));
	    intra_node_algo_index = (lp2ltn < table_min_intra_size) ? 0 : log2(lp2ltn / table_min_intra_size );
	}
    }

    MV2_Scatter_function = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[inter_node_algo_index]
                            .MV2_pt_Scatter_function;

    if(MV2_Scatter_function == &MPIR_Scatter_mcst_wrap_MV2) { 
#if defined(_MCST_SUPPORT_)
        if(comm_ptr->dev.ch.is_mcast_ok == 1 
           && mv2_use_mcast_scatter == 1 
           && comm_ptr->dev.ch.shmem_coll_ok == 1) {
            MV2_Scatter_function = &MPIR_Scatter_mcst_MV2; 
        } else
#endif /*#if defined(_MCST_SUPPORT_) */
        {
            if(mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[inter_node_algo_index + 1].
               MV2_pt_Scatter_function != NULL) { 
                  MV2_Scatter_function = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index].
		      inter_leader[inter_node_algo_index + 1].MV2_pt_Scatter_function;
            } else { 
                  /* Fallback! */ 
                  MV2_Scatter_function = &MPIR_Scatter_MV2_Binomial; 
            }  
        } 
    } 
 
    if( (MV2_Scatter_function == &MPIR_Scatter_MV2_two_level_Direct) || 
        (MV2_Scatter_function == &MPIR_Scatter_MV2_two_level_Binomial)) { 
         if( comm_ptr->dev.ch.shmem_coll_ok == 1 && 
             comm_ptr->dev.ch.is_global_block == 1 ) {
             MV2_Scatter_intra_function = mv2_scatter_indexed_thresholds_table[conf_index][comm_size_index]
		 .intra_node[intra_node_algo_index].MV2_pt_Scatter_function;

             mpi_errno =
                   MV2_Scatter_function(sendbuf, sendcnt, sendtype,
                                        recvbuf, recvcnt, recvtype, root,
                                        comm_ptr, errflag);
         } else {
             mpi_errno = MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt, sendtype,
                                        recvbuf, recvcnt, recvtype, root,
                                        comm_ptr, errflag);

         }
    } else { 
         mpi_errno = MV2_Scatter_function(sendbuf, sendcnt, sendtype,
                                    recvbuf, recvcnt, recvtype, root,
                                    comm_ptr, errflag);
    } 


    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

  fn_fail:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    return (mpi_errno);

}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_tune_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_tune_intra_MV2(const void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_tune_intra_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    int range = 0, range_threshold = 0, range_threshold_intra = 0;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int rank, comm_size;
    MPI_Aint nbytes;
    MPI_Aint recvtype_size, sendtype_size;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int local_size = -1;
    int i;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr=NULL;
    MPIU_THREADPRIV_DECL;

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    mpi_errno = PMPI_Comm_size(comm_ptr->handle, &comm_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    MPIU_THREADPRIV_GET;

    if (rank == root) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcnt * sendtype_size;
    } else {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
    }

    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        i = 0;
        if (mv2_scatter_table_ppn_conf[0] == -1) {
            /* Indicating user defined tuning */
            conf_index = 0;
            goto conf_check_end;
        }
        do {
            if (local_size == mv2_scatter_table_ppn_conf[i]) {
                conf_index = i;
                partial_sub_ok = 1;
                break;
            }
            i++;
        } while(i < mv2_scatter_num_ppn_conf);
    }
    
    if (partial_sub_ok != 1) {
        conf_index = mv2_scatter_num_ppn_conf/2;
    }

conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_scatter_tuning_table[conf_index] - 1)) &&
           (comm_size > mv2_scatter_thresholds_table[conf_index][range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold < (mv2_scatter_thresholds_table[conf_index][range].size_inter_table - 1))
           && (nbytes >
           mv2_scatter_thresholds_table[conf_index][range].inter_leader[range_threshold].max)
           && (mv2_scatter_thresholds_table[conf_index][range].inter_leader[range_threshold].max != -1)) {
           range_threshold++;
    }

    /* Search for corresponding intra-node function */
    while ((range_threshold_intra <
           (mv2_scatter_thresholds_table[conf_index][range].size_intra_table - 1))
            && (nbytes >
                mv2_scatter_thresholds_table[conf_index][range].intra_node[range_threshold_intra].max)
            && (mv2_scatter_thresholds_table[conf_index][range].intra_node[range_threshold_intra].max !=
            -1)) {
            range_threshold_intra++;
    }

    MV2_Scatter_function = mv2_scatter_thresholds_table[conf_index][range].inter_leader[range_threshold]
                            .MV2_pt_Scatter_function;

    if(MV2_Scatter_function == &MPIR_Scatter_mcst_wrap_MV2) { 
#if defined(_MCST_SUPPORT_)
        if(comm_ptr->dev.ch.is_mcast_ok == 1 
           && mv2_use_mcast_scatter == 1 
           && comm_ptr->dev.ch.shmem_coll_ok == 1) {
            MV2_Scatter_function = &MPIR_Scatter_mcst_MV2; 
        } else
#endif /*#if defined(_MCST_SUPPORT_) */
        {
            if(mv2_scatter_thresholds_table[conf_index][range].inter_leader[range_threshold + 1].
               MV2_pt_Scatter_function != NULL) { 
                  MV2_Scatter_function = mv2_scatter_thresholds_table[conf_index][range].inter_leader[range_threshold + 1]
                                                                          .MV2_pt_Scatter_function;
            } else { 
                  /* Fallback! */ 
                  MV2_Scatter_function = &MPIR_Scatter_MV2_Binomial; 
            }  
        } 
    } 
 
    if( (MV2_Scatter_function == &MPIR_Scatter_MV2_two_level_Direct) || 
        (MV2_Scatter_function == &MPIR_Scatter_MV2_two_level_Binomial)) { 
         if( comm_ptr->dev.ch.shmem_coll_ok == 1 && 
             comm_ptr->dev.ch.is_global_block == 1 ) {
             MV2_Scatter_intra_function = mv2_scatter_thresholds_table[conf_index][range].intra_node[range_threshold_intra]
                                .MV2_pt_Scatter_function;

             mpi_errno =
                   MV2_Scatter_function(sendbuf, sendcnt, sendtype,
                                        recvbuf, recvcnt, recvtype, root,
                                        comm_ptr, errflag);
         } else {
             mpi_errno = MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt, sendtype,
                                        recvbuf, recvcnt, recvtype, root,
                                        comm_ptr, errflag);

         }
    } else { 
         mpi_errno = MV2_Scatter_function(sendbuf, sendcnt, sendtype,
                                    recvbuf, recvcnt, recvtype, root,
                                    comm_ptr, errflag);
    } 


    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

  fn_fail:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    return (mpi_errno);

}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_intra_MV2(const void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_intra_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif      
    int range = 0;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int rank, comm_size;
    MPI_Aint nbytes;
    MPI_Aint recvtype_size, sendtype_size;
    MPIU_THREADPRIV_DECL;

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    mpi_errno = PMPI_Comm_size(comm_ptr->handle, &comm_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    MPIU_THREADPRIV_GET;

    if (rank == root) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcnt * sendtype_size;
    } else {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
    }

    while ((range < mv2_size_mv2_scatter_mv2_tuning_table)
           && (comm_size > mv2_scatter_mv2_tuning_table[range].numproc)) {
        range++;
    }
#if defined(_MCST_SUPPORT_)
    if(comm_ptr->dev.ch.is_mcast_ok == 1
       && mv2_use_mcast_scatter == 1
       && nbytes <= mv2_mcast_scatter_msg_size
       && comm_size >= mv2_mcast_scatter_small_sys_size
       && comm_size <= mv2_mcast_scatter_large_sys_size ){
         mpi_errno = MPIR_Scatter_mcst_MV2(sendbuf, sendcnt, sendtype,
                                         recvbuf, recvcnt, recvtype, root,
                                         comm_ptr, errflag);
    } else 
#endif /*#if defined(_MCST_SUPPORT_) */ 
    { 
        if (mv2_use_two_level_scatter == 1 || mv2_use_direct_scatter == 1) {
            if (range < mv2_size_mv2_scatter_mv2_tuning_table) {
                if (nbytes < mv2_scatter_mv2_tuning_table[range].small) {
                    mpi_errno =
                        MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt, sendtype,
                                                  recvbuf, recvcnt, recvtype, root,
                                                  comm_ptr, errflag);
                } else if (nbytes > mv2_scatter_mv2_tuning_table[range].small
                           && nbytes < mv2_scatter_mv2_tuning_table[range].medium
                           && comm_ptr->dev.ch.shmem_coll_ok == 1
                           && mv2_use_two_level_scatter == 1) {
                    mpi_errno =
                        MPIR_Scatter_MV2_two_level_Direct(sendbuf, sendcnt,
                                                          sendtype, recvbuf,
                                                          recvcnt, recvtype, root,
                                                          comm_ptr, errflag);

                } else {
                    mpi_errno = MPIR_Scatter_MV2_Direct(sendbuf, sendcnt, sendtype,
                                                        recvbuf, recvcnt, recvtype,
                                                        root, comm_ptr, errflag);
                }
            } else if (comm_size > mv2_scatter_mv2_tuning_table[range - 1].numproc
                       && comm_ptr->dev.ch.shmem_coll_ok == 1
                       && mv2_use_two_level_scatter == 1) {
                mpi_errno =
                    MPIR_Scatter_MV2_two_level_Binomial(sendbuf, sendcnt, sendtype,
                                                        recvbuf, recvcnt, recvtype,
                                                        root, comm_ptr, errflag);
            } else {
                mpi_errno = MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt, sendtype,
                                                      recvbuf, recvcnt, recvtype,
                                                      root, comm_ptr, errflag);
            }
        } else {
            mpi_errno = MPIR_Scatter_MV2_Binomial(sendbuf, sendcnt, sendtype,
                                                  recvbuf, recvcnt, recvtype,
                                                  root, comm_ptr, errflag);
        }
    } 

    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

  fn_fail:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    return (mpi_errno);

}




/* begin:nested */
/* not declared static because a machine-specific function may call this one in some cases */
#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_inter_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_inter_MV2(void *sendbuf,
                           int sendcnt,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcnt,
                           MPI_Datatype recvtype,
                           int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_inter_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
/*  Intercommunicator scatter.
    For short messages, root sends to rank 0 in remote group. rank 0
    does local intracommunicator scatter (binomial tree). 
    Cost: (lgp+1).alpha + n.((p-1)/p).beta + n.beta
   
    For long messages, we use linear scatter to avoid the extra n.beta.
    Cost: p.alpha + n.beta
*/

    int rank, local_size, remote_size, mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i;
    MPI_Aint nbytes, sendtype_size, recvtype_size;
    MPI_Status status;
    MPI_Aint extent, true_extent, true_lb = 0;
    void *tmp_buf = NULL;
    MPID_Comm *newcomm_ptr = NULL;

    if (root == MPI_PROC_NULL) {
        /* local processes other than root do nothing */
        return MPI_SUCCESS;
    }

    remote_size = comm_ptr->remote_size;
    local_size = comm_ptr->local_size;

    if (root == MPI_ROOT) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendtype_size * sendcnt * remote_size;
    } else {
        /* remote side */
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvtype_size * recvcnt * local_size;
    }

    if (nbytes < MPIR_SCATTER_SHORT_MSG) {
        if (root == MPI_ROOT) {
            /* root sends all data to rank 0 on remote group and returns */
            MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
            MPIR_PVAR_INC(scatter, inter, send, sendcnt * remote_size, sendtype);
            mpi_errno = MPIC_Send(sendbuf, sendcnt * remote_size,
                                     sendtype, 0, MPIR_SCATTER_TAG, comm_ptr,
                                     errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            goto fn_exit;
        } else {
            /* remote group. rank 0 receives data from root. need to
               allocate temporary buffer to store this data. */

            rank = comm_ptr->rank;

            if (rank == 0) {
                MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                                               &true_extent);
                MPID_Datatype_get_extent_macro(recvtype, extent);
                tmp_buf =
                    MPIU_Malloc(recvcnt * local_size *
                                (MPIR_MAX(extent, true_extent)));
                /* --BEGIN ERROR HANDLING-- */
                if (!tmp_buf) {
                    mpi_errno =
                        MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                             FCNAME, __LINE__, MPI_ERR_OTHER,
                                             "**nomem", 0);
                    return mpi_errno;
                }
                /* --END ERROR HANDLING-- */
                /* adjust for potential negative lower bound in datatype */
                tmp_buf = (void *) ((char *) tmp_buf - true_lb);

                MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
                MPIR_PVAR_INC(scatter, inter, recv, recvcnt * local_size, recvtype);
                mpi_errno = MPIC_Recv(tmp_buf, recvcnt * local_size,
                                         recvtype, root,
                                         MPIR_SCATTER_TAG, comm_ptr, &status,
                                         errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

            }

            /* Get the local intracommunicator */
            if (!comm_ptr->local_comm)
                MPIR_Setup_intercomm_localcomm(comm_ptr);

            newcomm_ptr = comm_ptr->local_comm;

            /* now do the usual scatter on this intracommunicator */
            mpi_errno = MPIR_Scatter_MV2(tmp_buf, recvcnt, recvtype,
                                         recvbuf, recvcnt, recvtype, 0,
                                         newcomm_ptr, errflag);
            if (rank == 0) {
                void *tmp = (void*)(tmp_buf + true_lb);
                MPIU_Free(tmp);
            }
        }
    } else {
        /* long message. use linear algorithm. */
        MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
        if (root == MPI_ROOT) {
            MPID_Datatype_get_extent_macro(sendtype, extent);
            for (i = 0; i < remote_size; i++) {
                MPIR_PVAR_INC(scatter, inter, send, sendcnt, sendtype);
                mpi_errno =
                    MPIC_Send(((char *) sendbuf + sendcnt * i * extent),
                                 sendcnt, sendtype, i, MPIR_SCATTER_TAG, comm_ptr,
                                 errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
        } else {
            MPIR_PVAR_INC(scatter, inter, recv, recvcnt, recvtype);
            mpi_errno = MPIC_Recv(recvbuf, recvcnt, recvtype, root,
                                     MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
        MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    }

  fn_exit:

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Scatter_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Scatter_MV2(const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                     void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                     int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if SCATTER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[scatter osu rank = %d host = %s] Func: MPIR_Scatter_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif      
    int mpi_errno = MPI_SUCCESS;

#ifdef _ENABLE_CUDA_
   MPI_Aint sendtype_extent, recvtype_extent;
   MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
   MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
   MPI_Aint nbytes = recvtype_extent * recvcnt;
   int send_mem_type = 0;
   int recv_mem_type = 0;
   int comm_size = comm_ptr->local_size;
   int rank = comm_ptr->rank;
   if (rdma_enable_cuda) {
       send_mem_type = is_device_buffer(sendbuf);
       recv_mem_type = is_device_buffer(recvbuf);
   }

   if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
       rdma_cuda_use_naive && (nbytes <= rdma_cuda_scatter_naive_limit*comm_size)) {
       if (sendbuf != MPI_IN_PLACE) {
            if (rank == root) {
                mpi_errno = cuda_stage_alloc ((void **)&sendbuf, sendcnt*sendtype_extent*comm_size,
                          NULL, 0, 
                          send_mem_type, 0, 
                          0);
            } else {
                mpi_errno = cuda_stage_alloc (NULL, 0,
                          &recvbuf, recvcnt*recvtype_extent, 
                          0, recv_mem_type, 
                          0);
            }
       } else {
            mpi_errno = cuda_stage_alloc ((void **)&sendbuf, recvcnt*recvtype_extent*comm_size,
                      &recvbuf, recvcnt*recvtype_extent, 
                      0, recv_mem_type, 
                      rank*recvcnt*recvtype_extent);
       }
       if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
       }
   }
#endif /*#ifdef _ENABLE_CUDA_*/    

    if (mv2_use_old_scatter == 1 ) {
        mpi_errno = MPIR_Scatter_intra_MV2(sendbuf, sendcnt, sendtype,
                                       recvbuf, recvcnt, recvtype, root,
                                       comm_ptr, errflag);
    } else {
	if (mv2_use_indexed_tuning || mv2_use_indexed_scatter_tuning) {
	    mpi_errno = MPIR_Scatter_index_tuned_intra_MV2(sendbuf, sendcnt, sendtype,
						    recvbuf, recvcnt, recvtype, root,
						    comm_ptr, errflag);
	}
	else {
	    mpi_errno = MPIR_Scatter_tune_intra_MV2(sendbuf, sendcnt, sendtype,
						    recvbuf, recvcnt, recvtype, root,
						    comm_ptr, errflag);
	}
    }

#ifdef _ENABLE_CUDA_ 
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_scatter_naive_limit*comm_size)){
        if (rank == root) {
            cuda_stage_free ((void **)&sendbuf, 
                        &recvbuf, 0,
                        send_mem_type, recv_mem_type);
        } else {
            cuda_stage_free (NULL, 
                        &recvbuf, recvcnt*recvtype_extent,
                        send_mem_type, recv_mem_type);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_*/     
    comm_ptr->dev.ch.intra_node_done = 0;
        if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:

    goto fn_exit;
}
