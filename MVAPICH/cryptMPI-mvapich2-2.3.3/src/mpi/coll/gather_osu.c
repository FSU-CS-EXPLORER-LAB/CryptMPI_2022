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
#include "datatype.h"
#include "common_tuning.h"
#include "gather_tuning.h"

/* added by abu naser */
#include "secure_allgather.h"
int count_ite = 0;
/* end of add */

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_pt2pt);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_direct);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_direct_blk);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_two_level_direct);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_limic_scheme_pt_pt);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_limic_scheme_pt_linear);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_limic_scheme_linear_pt);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_limic_scheme_linear_linear);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_limic_scheme_single_leader);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_gather_intra_node_limic);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_pt2pt);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_blk);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_two_level_direct);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_limic_scheme_pt_pt);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_limic_scheme_pt_linear);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_limic_scheme_linear_pt);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_limic_scheme_linear_linear);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_limic_scheme_single_leader);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_intra_node_limic);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_blk_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_two_level_direct_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_blk_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_two_level_direct_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_blk_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_two_level_direct_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_direct_blk_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_two_level_direct_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_gather_count_recv);

static void *tmp_buf=NULL;
#define TEMP_BUF_HAS_DATA (1)
#define TEMP_BUF_HAS_NO_DATA (0)

#if defined(_SMP_LIMIC_)
int num_scheme;
extern int use_limic_gather;
extern int limic_fd;
extern int MPIR_Limic_Gather_OSU( void *sendbuf, int sendbytes, void *recvbuf, 
                                  int recvbytes, MPID_Comm *shmem_comm_ptr );
#endif /*#if defined(_SMP_LIMIC_)*/

typedef int (*MV2_Gather_function_ptr) (const void *sendbuf,
                                        int sendcnt,
                                        MPI_Datatype sendtype,
                                        void *recvbuf,
                                        int recvcnt,
                                        MPI_Datatype recvtype,
                                        int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

MV2_Gather_function_ptr MV2_Gather_inter_leader_function = NULL;
MV2_Gather_function_ptr MV2_Gather_intra_node_function = NULL;

#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct(const void *sendbuf,
                                  int sendcnt,
                                  MPI_Datatype sendtype,
                                  void *recvbuf,
                                  int recvcnt,
                                  MPI_Datatype recvtype,
                                  int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2_Direct\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,gather,direct);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent = 0;        /* Datatype extent */
    int reqs = 0, i = 0;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_DECL(2);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_direct, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
        ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,direct);
        return MPI_SUCCESS;
    }

    if (root == rank) {
        comm_size = comm_ptr->local_size;

        MPID_Datatype_get_extent_macro(recvtype, extent);
        /* each node can make sure it is not going to overflow aint */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT
                                         recvbuf +
                                         (extent * recvcnt * comm_size));

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                            comm_size * sizeof (MPID_Request*),
                            mpi_errno, "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                            comm_size * sizeof (MPI_Status),
                            mpi_errno, "starray");

        reqs = 0;
        for (i = 0; i < comm_size; i++) {
            if (i == rank) {
                if (sendbuf != MPI_IN_PLACE) {
                    mpi_errno = MPIR_Localcopy(sendbuf, sendcnt, sendtype,
                                               ((char *) recvbuf +
                                                rank * recvcnt * extent),
                                               recvcnt, recvtype);
                }
            } else {
                MPIR_PVAR_INC(gather, direct, recv, recvcnt, recvtype);
                mpi_errno = MPIC_Irecv(((char *) recvbuf +
                                           i * recvcnt * extent),
                                          recvcnt, recvtype, i,
                                          MPIR_GATHER_TAG, comm_ptr,
                                          &reqarray[reqs++]);

            }
            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno) {
                mpi_errno = MPIR_Err_create_code(mpi_errno,
                                                 MPIR_ERR_RECOVERABLE,
                                                 FCNAME,
                                                 __LINE__, MPI_ERR_OTHER,
                                                 "**fail", 0);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */
        }
        /* ... then wait for *all* of them to finish: */
        mpi_errno = MPIC_Waitall(reqs, reqarray, starray, errflag);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (i = 0; i < reqs; i++) {
                if (starray[i].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = starray[i].MPI_ERROR;
                    if (mpi_errno) {
                        /* for communication errors, just record 
                           the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
        /* --END ERROR HANDLING-- */
    } else if (root != rank) {    /* non-root nodes proceses */
        if (sendcnt) {
            comm_size = comm_ptr->local_size;
            if (sendbuf != MPI_IN_PLACE) {
                MPIR_PVAR_INC(gather, direct, send, sendcnt, sendtype);
                mpi_errno = MPIC_Send(sendbuf, sendcnt, sendtype, root,
                                         MPIR_GATHER_TAG, comm_ptr, errflag);
            } else {
                MPIR_PVAR_INC(gather, direct, send, sendcnt, sendtype);
                mpi_errno = MPIC_Send(recvbuf, sendcnt, sendtype, root,
                                         MPIR_GATHER_TAG, comm_ptr, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }
  fn_fail:
    /* check if multiple threads are calling this collective function */
    MPIU_CHKLMEM_FREEALL();

    MPIR_TIMER_END(coll,gather,direct);
    return (mpi_errno);
}


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct_Blk
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_Blk(const void *sendbuf,
                                  int sendcnt,
                                  MPI_Datatype sendtype,
                                  void *recvbuf,
                                  int recvcnt,
                                  MPI_Datatype recvtype,
                                  int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2_Direct_Blk\n",comm_ptr->rank,hostname);fflush(stdout);
#endif         
    MPIR_TIMER_START(coll,gather,direct_blk);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint extent = 0;        /* Datatype extent */
    int i = 0;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_direct_blk, 1);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
        ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,direct_blk);
        return MPI_SUCCESS;
    }

    if (root == rank) {
        comm_size = comm_ptr->local_size;

        MPID_Datatype_get_extent_macro(recvtype, extent);
        /* each node can make sure it is not going to overflow aint */
        MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT
                                         recvbuf +
                                         (extent * recvcnt * comm_size));

        for (i = 0; i < comm_size; i++) {
            if (i == rank) {
                if (sendbuf != MPI_IN_PLACE) {
                    mpi_errno = MPIR_Localcopy(sendbuf, sendcnt, sendtype,
                                               ((char *) recvbuf +
                                                rank * recvcnt * extent),
                                               recvcnt, recvtype);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
            } else {
                MPIR_PVAR_INC(gather, direct_blk, recv, recvcnt, recvtype);
                mpi_errno = MPIC_Recv(((char *) recvbuf +
                                           i * recvcnt * extent),
                                          recvcnt, recvtype, i,
                                          MPIR_GATHER_TAG, comm_ptr,
                                          MPI_STATUS_IGNORE, errflag); 

            }
            /* --BEGIN ERROR HANDLING-- */
            if (mpi_errno) {
                mpi_errno = MPIR_Err_create_code(mpi_errno,
                                                 MPIR_ERR_RECOVERABLE,
                                                 FCNAME,
                                                 __LINE__, MPI_ERR_OTHER,
                                                 "**fail", 0);

                MPIR_TIMER_END(coll,gather,direct_blk);
                return mpi_errno;
            }
            /* --END ERROR HANDLING-- */
        }
    } else if (root != rank) {    /* non-root nodes proceses */
        if (sendcnt) {
            comm_size = comm_ptr->local_size;
            if (sendbuf != MPI_IN_PLACE) {
                MPIR_PVAR_INC(gather, direct_blk, send, sendcnt, sendtype);
                mpi_errno = MPIC_Send(sendbuf, sendcnt, sendtype, root,
                                         MPIR_GATHER_TAG, comm_ptr, errflag);
            } else {
                MPIR_PVAR_INC(gather, direct_blk, send, sendcnt, sendtype);
                mpi_errno = MPIC_Send(recvbuf, sendcnt, sendtype, root,
                                         MPIR_GATHER_TAG, comm_ptr, errflag);
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
    MPIR_TIMER_END(coll,gather,direct_blk);
    return mpi_errno;
 fn_fail:
    goto fn_exit;


}

/* sendbuf           - (in) sender's buffer
 * sendcnt           - (in) sender's element count
 * sendtype          - (in) sender's data type
 * recvbuf           - (in) receiver's buffer
 * recvcnt           - (in) receiver's element count
 * recvtype          - (in) receiver's data type
 * root              - (in)root for the gather operation
 * rank              - (in) global rank(rank in the global comm)
 * tmp_buf           - (out/in) tmp_buf into which intra node
 *                     data is gathered
 * is_data_avail     - (in) based on this, tmp_buf acts
 *                     as in/out parameter.
 *                     1 - tmp_buf acts as in parameter
 *                     0 - tmp_buf acts as out parameter
 * comm_ptr          - (in) pointer to the communicator
 *                     (shmem_comm or intra_sock_comm or
 *                     inter-sock_leader_comm)
 * intra_node_fn_ptr - (in) Function ptr to choose the
 *                      intra node gather function  
 * errflag           - (out) to record errors
 */
int MPIR_pt_pt_intra_gather(const void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                            int root, int rank, 
                            void *tmp_buf, int nbytes,
                            int is_data_avail,
                            MPID_Comm *comm_ptr,  
                            MV2_Gather_function_ptr intra_node_fn_ptr,
                            MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_pt_pt_intra_gather\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
    MPIR_TIMER_START(coll,gather,pt2pt);
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_pt2pt, 1);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                                       &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                                       &recvtype_true_extent);
    }
    
    /* Special case, when tmp_buf itself has data */
    if (rank == root && sendbuf == MPI_IN_PLACE && is_data_avail) {
         
         mpi_errno = intra_node_fn_ptr(MPI_IN_PLACE,
                                       sendcnt, sendtype, tmp_buf, nbytes,
                                       MPI_BYTE, 0, comm_ptr, errflag);

    } else if (rank == root && sendbuf == MPI_IN_PLACE) {
         mpi_errno = intra_node_fn_ptr(recvbuf +
                                       rank * recvcnt * recvtype_extent,
                                       recvcnt, recvtype, tmp_buf, nbytes,
                                       MPI_BYTE, 0, comm_ptr, errflag);
    } else {
        mpi_errno = intra_node_fn_ptr(sendbuf, sendcnt, sendtype,
                                      tmp_buf, nbytes, MPI_BYTE,
                                      0, comm_ptr, errflag);
    }

    MPIR_TIMER_END(coll,gather,pt2pt);
    return mpi_errno;

}


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_two_level_Direct
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_two_level_Direct(const void *sendbuf,
                                            int sendcnt,
                                            MPI_Datatype sendtype,
                                            void *recvbuf,
                                            int recvcnt,
                                            MPI_Datatype recvtype,
                                            int root,
                                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2_two_level_Direct\n",comm_ptr->rank,hostname);fflush(stdout);
#endif          
    MPIR_TIMER_START(coll,gather,two_level_direct);
    void *leader_gather_buf = NULL;
    int comm_size, rank;
    int local_rank, local_size;
    int leader_comm_rank = -1, leader_comm_size = 0;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, sendtype_size = 0, nbytes=0;
    int leader_root, leader_of_root;
    MPI_Status status;
    MPI_Aint sendtype_extent = 0, recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    MPI_Comm shmem_comm, leader_comm;
    MPID_Comm *shmem_commptr, *leader_commptr = NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_two_level_direct, 1);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
        ((rank != root) && (sendcnt == 0))) {
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                                       &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                                       &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    if (local_rank == 0) {
        /* Node leader. Extract the rank, size information for the leader
         * communicator */
        leader_comm = comm_ptr->dev.ch.leader_comm;
        MPID_Comm_get_ptr(leader_comm, leader_commptr);
        leader_comm_rank = leader_commptr->rank;
        leader_comm_size = leader_commptr->local_size;
    }

    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

#if defined(_SMP_LIMIC_)
     if((g_use_limic2_coll) && (shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
         && (use_limic_gather)
         &&((num_scheme == USE_GATHER_PT_PT_BINOMIAL) 
            || (num_scheme == USE_GATHER_PT_PT_DIRECT)
            ||(num_scheme == USE_GATHER_PT_LINEAR_BINOMIAL) 
            || (num_scheme == USE_GATHER_PT_LINEAR_DIRECT)
            || (num_scheme == USE_GATHER_LINEAR_PT_BINOMIAL)
            || (num_scheme == USE_GATHER_LINEAR_PT_DIRECT)
            || (num_scheme == USE_GATHER_LINEAR_LINEAR)
            || (num_scheme == USE_GATHER_SINGLE_LEADER))) {
            
            mpi_errno = MV2_Gather_intra_node_function(sendbuf, sendcnt, sendtype,
                                                    recvbuf, recvcnt,recvtype, 
                                                    root, comm_ptr, errflag);
     } else

#endif/*#if defined(_SMP_LIMIC_)*/    
    {
        if (local_rank == 0) {
            /* Node leader, allocate tmp_buffer */
            if (rank == root) {
                tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * local_size);
            } else {
                tmp_buf = MPIU_Malloc(sendcnt * MPIR_MAX(sendtype_extent,
                            sendtype_true_extent) *
                        local_size);
            }
            if (tmp_buf == NULL) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                        MPIR_ERR_RECOVERABLE,
                        FCNAME, __LINE__, MPI_ERR_OTHER,
                        "**nomem", 0);
                return mpi_errno;
            }
        }
         /*while testing mpich2 gather test, we see that
         * which basically splits the comm, and we come to
         * a point, where use_intra_sock_comm == 0, but if the 
         * intra node function is MPIR_Intra_node_LIMIC_Gather_MV2,
         * it would use the intra sock comm. In such cases, we 
         * fallback to binomial as a default case.*/
#if defined(_SMP_LIMIC_)         
        if(*MV2_Gather_intra_node_function == MPIR_Intra_node_LIMIC_Gather_MV2) {

            mpi_errno  = MPIR_pt_pt_intra_gather(sendbuf,sendcnt, sendtype,
                                                 recvbuf, recvcnt, recvtype,
                                                 root, rank, 
                                                 tmp_buf, nbytes, 
                                                 TEMP_BUF_HAS_NO_DATA,
                                                 shmem_commptr,
                                                 MPIR_Gather_intra,
                                                 errflag);
        } else
#endif
        {
            /*We are gathering the data into tmp_buf and the output
             * will be of MPI_BYTE datatype. Since the tmp_buf has no
             * local data, we pass is_data_avail = TEMP_BUF_HAS_NO_DATA*/
           // printf("1 . rank %d root %d nbytes %d sendcnt %d recvcnt %d\n",rank, root, nbytes, sendcnt, recvcnt);fflush(stdout);
            mpi_errno  = MPIR_pt_pt_intra_gather(sendbuf,sendcnt, sendtype,
                                                 recvbuf, recvcnt, recvtype,
                                                 root, rank, 
                                                 tmp_buf, nbytes, 
                                                 TEMP_BUF_HAS_NO_DATA,
                                                 shmem_commptr,
                                                 MV2_Gather_intra_node_function,
                                                 errflag);
        }
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    leader_of_root = comm_ptr->dev.ch.leader_map[root];
    leader_root = comm_ptr->dev.ch.leader_rank[leader_of_root];
    /* leader_root is the rank of the leader of the root in leader_comm. 
     * leader_root is to be used as the root of the inter-leader gather ops 
     */
    if (comm_ptr->dev.ch.is_uniform != 1) {
        if (local_rank == 0) {
            int *displs = NULL;
            int *recvcnts = NULL;
            int *node_sizes;
            int i = 0;
            /* Node leaders have all the data. But, different nodes can have
             * different number of processes. Do a Gather first to get the 
             * buffer lengths at each leader, followed by a Gatherv to move
             * the actual data */

            if (leader_comm_rank == leader_root && root != leader_of_root) {
                /* The root of the Gather operation is not a node-level 
                 * leader and this process's rank in the leader_comm 
                 * is the same as leader_root */
                if(rank == root) { 
                    leader_gather_buf = MPIU_Malloc(recvcnt *
                                                MPIR_MAX(recvtype_extent,
                                                recvtype_true_extent) *
                                                comm_size);
                } else { 
                    leader_gather_buf = MPIU_Malloc(sendcnt *
                                                MPIR_MAX(sendtype_extent,
                                                sendtype_true_extent) *
                                                comm_size);
                } 
                if (leader_gather_buf == NULL) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                                                     MPIR_ERR_RECOVERABLE,
                                                     FCNAME, __LINE__,
                                                     MPI_ERR_OTHER,
                                                     "**nomem", 0);

                    MPIR_TIMER_END(coll,gather,two_level_direct);
                    return mpi_errno;
                }
            }

            node_sizes = comm_ptr->dev.ch.node_sizes;

            if (leader_comm_rank == leader_root) {
                displs = MPIU_Malloc(sizeof (int) * leader_comm_size);
                recvcnts = MPIU_Malloc(sizeof (int) * leader_comm_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                                                     MPIR_ERR_RECOVERABLE,
                                                     FCNAME, __LINE__,
                                                     MPI_ERR_OTHER,
                                                     "**nomem", 0);

                    MPIR_TIMER_END(coll,gather,two_level_direct);
                    return mpi_errno;
                }
            }

            if (root == leader_of_root) {
                /* The root of the gather operation is also the node 
                 * leader. Receive into recvbuf and we are done */
                if (leader_comm_rank == leader_root) {
                    recvcnts[0] = node_sizes[0] * recvcnt;
                    displs[0] = 0;

                    for (i = 1; i < leader_comm_size; i++) {
                        displs[i] = displs[i - 1] + node_sizes[i - 1] * recvcnt;
                        recvcnts[i] = node_sizes[i] * recvcnt;
                    }
                } 
                mpi_errno = MPIR_Gatherv(tmp_buf,
                                         local_size * nbytes,
                                         MPI_BYTE, recvbuf, recvcnts,
                                         displs, recvtype,
                                         leader_root, leader_commptr, errflag);
            } else {
                /* The root of the gather operation is not the node leader. 
                 * Receive into leader_gather_buf and then send 
                 * to the root */
                if (leader_comm_rank == leader_root) {
                    recvcnts[0] = node_sizes[0] * nbytes;
                    displs[0] = 0;

                    for (i = 1; i < leader_comm_size; i++) {
                        displs[i] = displs[i - 1] + node_sizes[i - 1] * nbytes;
                        recvcnts[i] = node_sizes[i] * nbytes;
                    }
                } 
                mpi_errno = MPIR_Gatherv(tmp_buf, local_size * nbytes,
                                         MPI_BYTE, leader_gather_buf,
                                         recvcnts, displs, MPI_BYTE,
                                         leader_root, leader_commptr, errflag);
            }
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            if (leader_comm_rank == leader_root) {
                MPIU_Free(displs);
                MPIU_Free(recvcnts);
            }
        }
    } else {
        /* All nodes have the same number of processes. 
         * Just do one Gather to get all 
         * the data at the leader of the root process */
        if (local_rank == 0) {
            if (leader_comm_rank == leader_root && root != leader_of_root) {
                /* The root of the Gather operation is not a node-level leader
                 */
                leader_gather_buf = MPIU_Malloc(nbytes * comm_size);
                if (leader_gather_buf == NULL) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                                                     MPIR_ERR_RECOVERABLE,
                                                     FCNAME, __LINE__,
                                                     MPI_ERR_OTHER,
                                                     "**nomem", 0);
                    
                    MPIR_TIMER_END(coll,gather,two_level_direct);
                    return mpi_errno;
                }
            }
            if (root == leader_of_root) {
                if (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) {
                    /* cyclic process mapping */

                    MPIU_Assert (comm_ptr->dev.ch.rank_list != NULL); 

                    void * reorder_tmp_buf = NULL;
                    if (leader_comm_rank == leader_root) {
                        reorder_tmp_buf = MPIU_Malloc(nbytes * comm_size);
                    }

                    mpi_errno = MPIR_Gather_MV2_Direct(tmp_buf,
                                                       nbytes * local_size,
                                                       MPI_BYTE, reorder_tmp_buf,
                                                       recvcnt * local_size,
                                                       recvtype, leader_root,
                                                       leader_commptr, errflag);

                    if (leader_comm_rank == leader_root) {
                        /* now reorder and place the results in recvbuf */
                        void *src = NULL, *dst = NULL; int i;
                        for (i = 0; i < comm_size; i++) {
                            src = reorder_tmp_buf + i * nbytes;
                            dst = recvbuf + comm_ptr->dev.ch.rank_list[i]*nbytes;
                            if (dst == NULL) PRINT_ERROR("recvbuf=%p comm_ptr->dev.ch.rank_list[%d]=%d\n", 
                                    recvbuf, i, comm_ptr->dev.ch.rank_list[i]);
                            MPIU_Memcpy(dst, src, nbytes);
                        }
                        MPIU_Free(reorder_tmp_buf);
                    }

                } else {
                    /* blocked (bunch) process mapping */
                    // printf("2. rank %d leader_root %d root %d nbytes %d recvcnt %d local_size %d\n",rank, leader_root, root, nbytes, recvcnt,local_size);fflush(stdout);
                    mpi_errno = MPIR_Gather_MV2_Direct(tmp_buf,
                                                       nbytes * local_size,
                                                       MPI_BYTE, recvbuf,
                                                       recvcnt * local_size,
                                                       recvtype, leader_root,
                                                       leader_commptr, errflag);
                }
            } else { // other non root, local rank 0
                // printf("3. rank %d leader_root %d root %d nbytes %d local_size %d\n",rank, leader_root, root, nbytes,local_size);fflush(stdout);
                mpi_errno = MPIR_Gather_MV2_Direct(tmp_buf, nbytes * local_size,
                                                   MPI_BYTE, leader_gather_buf,
                                                   nbytes * local_size,
                                                   MPI_BYTE, leader_root,
                                                   leader_commptr, errflag);
            }
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }
    if ((local_rank == 0) && (root != rank)
        && (leader_of_root == rank)) {
        MPIR_PVAR_INC(gather, two_level_direct, send, nbytes * comm_size, MPI_BYTE);
        // printf("4. rank %d leader_root %d root %d leader_of_root %d nbytes %d recvcnt %d local_size %d\n",rank, leader_root, root, leader_of_root, nbytes, recvcnt,local_size);fflush(stdout);
        mpi_errno = MPIC_Send(leader_gather_buf,
                                 nbytes * comm_size, MPI_BYTE,
                                 root, MPIR_GATHER_TAG, comm_ptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error 
               but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    if (rank == root && local_rank != 0) {
        /* The root of the gather operation is not the node leader. Receive
         y* data from the node leader */
        MPIR_PVAR_INC(gather, two_level_direct, recv, recvcnt * comm_size, recvtype);
        if (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) {
            /* cyclic process mapping */
            MPIU_Assert (comm_ptr->dev.ch.rank_list != NULL); 
            void * reorder_tmp_buf = MPIU_Malloc(nbytes * comm_size);
            mpi_errno = MPIC_Recv(reorder_tmp_buf, recvcnt * comm_size, recvtype,
                                     leader_of_root, MPIR_GATHER_TAG, comm_ptr,
                                     &status, errflag);
            /* now reorder and place the results in recvbuf */
            void *src = NULL, *dst = NULL; int i;
            for (i = 0; i < comm_size; i++) {
                src = reorder_tmp_buf + i * nbytes;
                dst = recvbuf + comm_ptr->dev.ch.rank_list[i]*nbytes;
                if (dst == NULL) PRINT_ERROR("recvbuf=%p comm_ptr->dev.ch.rank_list[%d]=%d\n", 
                        recvbuf, i, comm_ptr->dev.ch.rank_list[i]);
                MPIU_Memcpy(dst, src, nbytes);
            }
            MPIU_Free(reorder_tmp_buf);
        } else { 
            /* blocked (bunch) process mapping */
            mpi_errno = MPIC_Recv(recvbuf, recvcnt * comm_size, recvtype,
                                     leader_of_root, MPIR_GATHER_TAG, comm_ptr,
                                     &status, errflag);

        }
        if (mpi_errno) {
            /* for communication errors, just record the error but 
               continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

  fn_fail:
    /* check if multiple threads are calling this collective function */
    if (local_rank == 0 ) {
        if (tmp_buf != NULL) {
            MPIU_Free(tmp_buf);
        }
        if (leader_gather_buf != NULL) {
            MPIU_Free(leader_gather_buf);
        }
    }

    MPIR_TIMER_END(coll,gather,two_level_direct);
    return (mpi_errno);
}

#if defined(_SMP_LIMIC_)
#undef FUNCNAME
#define FUNCNAME MPIR_Limic_Gather_Scheme_PT_PT
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Limic_Gather_Scheme_PT_PT(
                                         const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                         void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                         int root, MPID_Comm * comm_ptr, 
                                         MV2_Gather_function_ptr intra_node_fn_ptr, 
                                         MPIR_Errflag_t *errflag) 
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Limic_Gather_Scheme_PT_PT\n",comm_ptr->rank,hostname);fflush(stdout);
#endif     
    MPIR_TIMER_START(coll,gather,limic_scheme_pt_pt);
    void *intra_tmp_buf = NULL;
    int rank;
    int local_size;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, sendtype_size = 0, nbytes=0;
    int sendtype_iscontig;
    int intra_sock_rank=0, intra_sock_comm_size=0;
    int intra_node_leader_rank=0, intra_node_leader_comm_size=0;
    MPI_Aint sendtype_extent = 0, recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Comm *intra_sock_commptr = NULL, *intra_node_leader_commptr=NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_limic_scheme_pt_pt, 1);
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
            ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,limic_scheme_pt_pt);
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_size = shmem_commptr->local_size;


    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

    if(shmem_commptr->dev.ch.use_intra_sock_comm == 1) { 
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm, intra_node_leader_commptr);

        intra_sock_rank = intra_sock_commptr->rank;
        intra_sock_comm_size = intra_sock_commptr->local_size;
        if(intra_sock_rank == 0) { 
            intra_node_leader_rank = intra_node_leader_commptr->rank;
            intra_node_leader_comm_size = intra_node_leader_commptr->local_size;
        }
    }
    if (intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            /* Node leaders, allocate large buffers which is used to gather
             * data for the entire node. The same buffer is used for inter-node
             * gather as well. This saves us a memcpy operation*/
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * local_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(sendcnt * MPIR_MAX(sendtype_extent,
                            sendtype_true_extent) * local_size);
            }
        } else {

            /* Socket leader, allocate tmp_buffer */
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * intra_sock_comm_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(sendcnt * MPIR_MAX(sendtype_extent,
                            sendtype_true_extent) * intra_sock_comm_size);
            }
        }
        if (intra_tmp_buf == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                    MPIR_ERR_RECOVERABLE,
                    FCNAME, __LINE__, MPI_ERR_OTHER,
                    "**nomem", 0);

            MPIR_TIMER_END(coll,gather,limic_scheme_pt_pt);
            return mpi_errno;
        }

    }

    /*Intra socket gather*/
    /*We are gathering the data into intra_tmp_buf and the output
     * will be of MPI_BYTE datatype. Since the tmp_buf has no
     * local data, we pass is_data_avail = TEMP_BUF_HAS_NO_DATA*/
    mpi_errno  = MPIR_pt_pt_intra_gather(sendbuf, sendcnt, sendtype,
                                         recvbuf, recvcnt, recvtype,
                                         root, rank, 
                                         intra_tmp_buf, nbytes,
                                         TEMP_BUF_HAS_NO_DATA,
                                         intra_sock_commptr, 
                                         intra_node_fn_ptr,
                                         errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /*Inter socket gather*/
    if(intra_sock_rank == 0) {
        /*When data in each socket is different*/
        if (shmem_commptr->dev.ch.is_socket_uniform != 1) {

            int *displs = NULL;
            int *recvcnts = NULL;
            int *socket_sizes;
            int i = 0;
            socket_sizes = shmem_commptr->dev.ch.socket_size;

            if (intra_node_leader_rank == 0) {
                tmp_buf = intra_tmp_buf;

                displs = MPIU_Malloc(sizeof (int) * intra_node_leader_comm_size);
                recvcnts = MPIU_Malloc(sizeof (int) * intra_node_leader_comm_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);

                    MPIR_TIMER_END(coll,gather,limic_scheme_pt_pt);
                    return mpi_errno;
                }

                recvcnts[0] = socket_sizes[0] * nbytes;
                displs[0] = 0;

                for (i = 1; i < intra_node_leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + socket_sizes[i - 1] * nbytes;
                    recvcnts[i] = socket_sizes[i] * nbytes;
                }

                mpi_errno = MPIR_Gatherv(MPI_IN_PLACE,
                                         intra_sock_comm_size * nbytes,
                                         MPI_BYTE, tmp_buf, recvcnts,
                                         displs, MPI_BYTE,
                                         0, intra_node_leader_commptr,
                                         errflag);

                /*Free the displacement and recvcnts buffer*/
                MPIU_Free(displs);
                MPIU_Free(recvcnts);
            } else {
                mpi_errno = MPIR_Gatherv(intra_tmp_buf,
                                         intra_sock_comm_size * nbytes,
                                         MPI_BYTE, tmp_buf, recvcnts,
                                         displs, MPI_BYTE,
                                         0, intra_node_leader_commptr, 
                                         errflag);

            }

        } else {

            if (intra_node_leader_rank == 0) {
                tmp_buf = intra_tmp_buf;

                /*We have now completed the intra_sock gather and all the 
                 * socket level leaders have data in their tmp_buf. So we 
                 * set sendbuf = MPI_IN_PLACE and also explicity set the
                 * is_data_avail= TEMP_BUF_HAS_DATA*/
                mpi_errno  = MPIR_pt_pt_intra_gather(MPI_IN_PLACE, 
                                                     (nbytes*intra_sock_comm_size), 
                                                     MPI_BYTE,
                                                     recvbuf, recvcnt, recvtype,
                                                     root, rank, 
                                                     tmp_buf,  
                                                     (nbytes*intra_sock_comm_size),
                                                     TEMP_BUF_HAS_DATA, 
                                                     intra_node_leader_commptr,              
                                                     intra_node_fn_ptr, 
                                                     errflag);
            } else {

                /*After the intra_sock gather, all the node level leaders
                 * have the data in intra_tmp_buf(sendbuf) and this is gathered into
                 * tmp_buf. Since the tmp_buf(in non-root processes) does not have 
                 * the data in tmp_buf is_data_avail = TEMP_BUF_HAS_NO_DATA*/
                mpi_errno  = MPIR_pt_pt_intra_gather(intra_tmp_buf, 
                                                    (nbytes*intra_sock_comm_size), 
                                                    MPI_BYTE,
                                                    recvbuf, recvcnt, recvtype,
                                                    root, rank,
                                                    tmp_buf,
                                                    (nbytes*intra_sock_comm_size),
                                                    TEMP_BUF_HAS_NO_DATA,
                                                    intra_node_leader_commptr,
                                                    intra_node_fn_ptr,
                                                    errflag);
            }

        }

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
fn_fail:
    /*Free the intra socket leader buffers*/
    if (intra_sock_rank == 0) {
        if ((intra_node_leader_rank != 0) && (intra_tmp_buf != NULL)) {
            MPIU_Free(intra_tmp_buf);
        }
    }

    MPIR_TIMER_END(coll,gather,limic_scheme_pt_pt);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Limic_Gather_Scheme_PT_LINEAR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Limic_Gather_Scheme_PT_LINEAR(
                                         const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                         void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                         int root, MPID_Comm * comm_ptr, 
                                         MV2_Gather_function_ptr intra_node_fn_ptr, 
                                         MPIR_Errflag_t *errflag) 
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Limic_Gather_Scheme_PT_LINEAR\n",comm_ptr->rank,hostname);fflush(stdout);
#endif            
    MPIR_TIMER_START(coll,gather,limic_scheme_pt_linear);
    void *intra_tmp_buf = NULL;
    void *local_sendbuf=NULL;
    int rank;
    int local_rank, local_size;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, nbytes=0;
    int sendtype_iscontig;
    int intra_sock_rank=0, intra_sock_comm_size=0;
    int intra_node_leader_rank=0, intra_node_leader_comm_size=0;
    MPI_Aint send_nbytes=0;
    MPI_Aint recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Comm *intra_sock_commptr = NULL, *intra_node_leader_commptr=NULL;
    MPI_Aint position = 0;
    MPI_Aint sendtype_size = 0;
    MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_limic_scheme_pt_linear, 1);
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
            ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,limic_scheme_pt_linear);
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;


    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

    if(shmem_commptr->dev.ch.use_intra_sock_comm == 1) { 
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm, intra_node_leader_commptr);

        intra_sock_rank = intra_sock_commptr->rank;
        intra_sock_comm_size = intra_sock_commptr->local_size;
        if(intra_sock_rank == 0) { 
            intra_node_leader_rank = intra_node_leader_commptr->rank;
            intra_node_leader_comm_size = intra_node_leader_commptr->local_size;
        }    
    }
    /*Pack data for non-contiguous buffer*/
    if ((!sendtype_iscontig) && (sendbuf != MPI_IN_PLACE)) {
        MPIR_Pack_size_impl(1, sendtype, &sendtype_size);
        send_nbytes= sendcnt * sendtype_size;
        MPIU_CHKLMEM_MALLOC(local_sendbuf, void *, send_nbytes, mpi_errno, "local_sendbuf");
        MPIR_Pack_impl(sendbuf, sendcnt, sendtype, local_sendbuf, send_nbytes, &position);
    } else {
        local_sendbuf = (void *)sendbuf;
        send_nbytes=nbytes;
    }


    if (intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            /* Node leaders, allocate large buffers which is used to gather
             * data for the entire node. The same buffer is used for inter-node
             * gather as well. This saves us a memcpy operation*/
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * local_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(send_nbytes * local_size);
            }

        } else {

            /* Socket leader, allocate tmp_buffer */
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * intra_sock_comm_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(send_nbytes * intra_sock_comm_size);
            }

        }

        if (intra_tmp_buf == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                    MPIR_ERR_RECOVERABLE,
                    FCNAME, __LINE__, MPI_ERR_OTHER,
                    "**nomem", 0);

            MPIR_TIMER_END(coll,gather,limic_scheme_pt_linear);
            return mpi_errno;
        }

        /*Local copy of buffer*/
        if(sendbuf != MPI_IN_PLACE) {
            MPIU_Memcpy(intra_tmp_buf, local_sendbuf, send_nbytes);
        } else {
            MPIR_Localcopy(((char *) recvbuf +rank * recvcnt * recvtype_extent),
                           recvcnt, recvtype,
                           intra_tmp_buf, send_nbytes, MPI_BYTE);
       }
    }

    if(local_rank !=0 && sendbuf == MPI_IN_PLACE) {
        mpi_errno = MPIR_Limic_Gather_OSU(intra_tmp_buf, 
                                          (intra_sock_comm_size * send_nbytes), 
                                          (recvbuf + (rank*nbytes)), nbytes,
                                          intra_sock_commptr );
    } else {
        mpi_errno = MPIR_Limic_Gather_OSU(intra_tmp_buf, 
                                          (intra_sock_comm_size * send_nbytes), 
                                          local_sendbuf, send_nbytes, 
                                          intra_sock_commptr );
    }
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /*Inter socket gather*/
    if(intra_sock_rank == 0) {
        /*When data in each socket is different*/
        if (shmem_commptr->dev.ch.is_socket_uniform != 1) {

            int *displs = NULL;
            int *recvcnts = NULL;
            int *socket_sizes;
            int i = 0;
            socket_sizes = shmem_commptr->dev.ch.socket_size;

            if (intra_node_leader_rank == 0) {
                tmp_buf = intra_tmp_buf;

                displs = MPIU_Malloc(sizeof (int) * intra_node_leader_comm_size);
                recvcnts = MPIU_Malloc(sizeof (int) * intra_node_leader_comm_size);
                if (!displs || !recvcnts) {
                    mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                            MPIR_ERR_RECOVERABLE,
                            FCNAME, __LINE__,
                            MPI_ERR_OTHER,
                            "**nomem", 0);

                    MPIR_TIMER_END(coll,gather,limic_scheme_pt_linear);
                    return mpi_errno;
                }

                recvcnts[0] = socket_sizes[0] * nbytes;
                displs[0] = 0;

                for (i = 1; i < intra_node_leader_comm_size; i++) {
                    displs[i] = displs[i - 1] + socket_sizes[i - 1] * nbytes;
                    recvcnts[i] = socket_sizes[i] * nbytes;
                }


                mpi_errno = MPIR_Gatherv(MPI_IN_PLACE,
                                         intra_sock_comm_size * nbytes,
                                         MPI_BYTE, tmp_buf, recvcnts,
                                         displs, MPI_BYTE,
                                         0, intra_node_leader_commptr, 
                                         errflag);

                /*Free the displacement and recvcnts buffer*/
                MPIU_Free(displs);
                MPIU_Free(recvcnts);

            } else {
                mpi_errno = MPIR_Gatherv(intra_tmp_buf,
                                         intra_sock_comm_size * nbytes,
                                         MPI_BYTE, tmp_buf, recvcnts,
                                         displs, MPI_BYTE,
                                         0, intra_node_leader_commptr, 
                                         errflag);

            }
        } else {

            if (intra_node_leader_rank == 0) {
                tmp_buf = intra_tmp_buf;

                /*We have now completed the intra_sock gather and all the 
                 * socket level leaders have data in their tmp_buf. So we 
                 * set sendbuf = MPI_IN_PLACE and also explicity set the
                 * is_data_avail= TEMP_BUF_HAS_DATA*/
                mpi_errno  = MPIR_pt_pt_intra_gather(MPI_IN_PLACE, 
                                                     (send_nbytes*intra_sock_comm_size), 
                                                     MPI_BYTE,
                                                     recvbuf, recvcnt, recvtype,
                                                     root, rank, 
                                                     tmp_buf, 
                                                     (send_nbytes*intra_sock_comm_size),
                                                     TEMP_BUF_HAS_DATA, 
                                                     intra_node_leader_commptr,
                                                     intra_node_fn_ptr,
                                                     errflag);
            } else {

                /*After the intra_sock gather, all the node level leaders
                 * have the data in intra_tmp_buf(sendbuf) and this is gathered into
                 * tmp_buf. Since the tmp_buf(in non-root processes) does not have 
                 * the data in tmp_buf is_data_avail = TEMP_BUF_HAS_NO_DATA*/
                mpi_errno  = MPIR_pt_pt_intra_gather(intra_tmp_buf, 
                                                     (send_nbytes*intra_sock_comm_size),
                                                     MPI_BYTE,
                                                     recvbuf, recvcnt, recvtype,
                                                     root, rank, 
                                                     tmp_buf, 
                                                     (send_nbytes*intra_sock_comm_size),
                                                     TEMP_BUF_HAS_NO_DATA, 
                                                     intra_node_leader_commptr,
                                                     intra_node_fn_ptr,
                                                     errflag);
            }
        }

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
fn_fail:
    /*Free the intra socket leader buffers*/
    if (intra_sock_rank == 0) { 
        if ((intra_node_leader_rank != 0) && (intra_tmp_buf != NULL)) {
            MPIU_Free(intra_tmp_buf);
        }
    }
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,gather,limic_scheme_pt_linear);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Limic_Gather_Scheme_LINEAR_PT
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Limic_Gather_Scheme_LINEAR_PT(
                                         const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                         void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                         int root, MPID_Comm * comm_ptr, 
                                         MV2_Gather_function_ptr intra_node_fn_ptr, 
                                         MPIR_Errflag_t *errflag) 
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Limic_Gather_Scheme_LINEAR_PT\n",comm_ptr->rank,hostname);fflush(stdout);
#endif       
    MPIR_TIMER_START(coll,gather,limic_scheme_linear_pt);
    void *intra_tmp_buf = NULL;
    int rank;
    int local_size;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, sendtype_size = 0, nbytes=0;
    int sendtype_iscontig;
    int intra_sock_rank=0, intra_sock_comm_size=0;
    int intra_node_leader_rank=0;
    MPI_Aint sendtype_extent = 0, recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Comm *intra_sock_commptr = NULL, *intra_node_leader_commptr=NULL;

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_limic_scheme_linear_pt, 1);
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
            ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,limic_scheme_linear_pt);
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
        MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_size = shmem_commptr->local_size;


    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

    if(shmem_commptr->dev.ch.use_intra_sock_comm == 1) { 
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm, intra_node_leader_commptr);

        intra_sock_rank = intra_sock_commptr->rank;
        intra_sock_comm_size = intra_sock_commptr->local_size;
        if(intra_sock_rank == 0) { 
            intra_node_leader_rank = intra_node_leader_commptr->rank;
        }    
    }

    if (intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            /* Node leaders, allocate large buffers which is used to gather
             * data for the entire node. The same buffer is used for inter-node
             * gather as well. This saves us a memcpy operation*/
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * local_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(sendcnt * MPIR_MAX(sendtype_extent,
                            sendtype_true_extent) * local_size);
            } 
        } else {

            /* Socket leader, allocate tmp_buffer */
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * intra_sock_comm_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(sendcnt * MPIR_MAX(sendtype_extent,
                            sendtype_true_extent) * intra_sock_comm_size);
            }
        }
        if (intra_tmp_buf == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                    MPIR_ERR_RECOVERABLE,
                    FCNAME, __LINE__, MPI_ERR_OTHER,
                    "**nomem", 0);

            MPIR_TIMER_END(coll,gather,limic_scheme_linear_pt);
            return mpi_errno;
        }
    }

    /*Intra socket gather*/
    /*We are gathering the data into intra_tmp_buf and the output
     * will be of MPI_BYTE datatype. Since the tmp_buf has no
     * local data, we pass is_data_avail = TEMP_BUF_HAS_NO_DATA*/
    mpi_errno  = MPIR_pt_pt_intra_gather(sendbuf, sendcnt, sendtype,
                                         recvbuf, recvcnt, recvtype,
                                         root, rank, 
                                         intra_tmp_buf, nbytes, 
                                         TEMP_BUF_HAS_NO_DATA,
                                         intra_sock_commptr,
                                         intra_node_fn_ptr,
                                         errflag);

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /*Inter socket gather*/
    if(intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            tmp_buf = intra_tmp_buf;
        }
        mpi_errno = MPIR_Limic_Gather_OSU(tmp_buf, (local_size * nbytes), 
                                          intra_tmp_buf, 
                                          (intra_sock_comm_size * nbytes), 
                                          intra_node_leader_commptr);
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
fn_fail:
    /*Free the intra socket leader buffers*/
    if (intra_sock_rank == 0) { 
        if ((intra_node_leader_rank != 0) && (intra_tmp_buf != NULL)) {
            MPIU_Free(intra_tmp_buf);
        }
    }

    MPIR_TIMER_END(coll,gather,limic_scheme_linear_pt);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Limic_Gather_Scheme_LINEAR_LINEAR
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Limic_Gather_Scheme_LINEAR_LINEAR(
                                         const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                         void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                         int root, MPID_Comm * comm_ptr, 
                                         MPIR_Errflag_t *errflag) 
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Limic_Gather_Scheme_LINEAR_LINEAR\n",comm_ptr->rank,hostname);fflush(stdout);
#endif      
    MPIR_TIMER_START(coll,gather,limic_scheme_linear_linear);
    void *intra_tmp_buf = NULL;
    void *local_sendbuf=NULL;
    int rank;
    int local_rank, local_size;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, nbytes=0;
    int sendtype_iscontig;
    int intra_sock_rank=0, intra_sock_comm_size=0;
    int intra_node_leader_rank=0;
    MPI_Aint send_nbytes=0;
    MPI_Aint recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPID_Comm *intra_sock_commptr = NULL, *intra_node_leader_commptr=NULL;
    MPI_Aint sendtype_size = 0;
    MPI_Aint position = 0;
    MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_limic_scheme_linear_linear, 1);
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
            ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,limic_scheme_linear_linear);
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;


    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

    if(shmem_commptr->dev.ch.use_intra_sock_comm == 1) { 
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
        MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm, intra_node_leader_commptr);

        intra_sock_rank = intra_sock_commptr->rank;
        intra_sock_comm_size = intra_sock_commptr->local_size;
        if(intra_sock_rank == 0) { 
            intra_node_leader_rank = intra_node_leader_commptr->rank;
        }    
    }
    
    /*Pack data for non-contiguous buffer*/
    if ((!sendtype_iscontig) && (sendbuf != MPI_IN_PLACE)) {

        MPIR_Pack_size_impl(1, sendtype, &sendtype_size);
        send_nbytes= sendcnt * sendtype_size;
        MPIU_CHKLMEM_MALLOC(local_sendbuf, void *, send_nbytes, mpi_errno, "local_sendbuf");
        MPIR_Pack_impl(sendbuf, sendcnt, sendtype, local_sendbuf, send_nbytes, &position);
    }
    else {
        local_sendbuf = (void *)sendbuf;
        send_nbytes = nbytes;
    }

    if (intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            /* Node leaders, allocate large buffers which is used to gather
             * data for the entire node. The same buffer is used for inter-node
             * gather as well. This saves us a memcpy operation*/
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * local_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(send_nbytes * local_size);
            }

        } else {

            /* Socket leader, allocate tmp_buffer */
            if (rank == root) {
                intra_tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                            recvtype_true_extent) * intra_sock_comm_size);
            } else {
                intra_tmp_buf = MPIU_Malloc(send_nbytes * intra_sock_comm_size);
            }

        }
        if (intra_tmp_buf == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                    MPIR_ERR_RECOVERABLE,
                    FCNAME, __LINE__, MPI_ERR_OTHER,
                    "**nomem", 0);

            MPIR_TIMER_END(coll,gather,limic_scheme_linear_linear);
            return mpi_errno;
        }

        /*Local copy of buffer*/
        if(sendbuf != MPI_IN_PLACE) {
            MPIU_Memcpy(intra_tmp_buf, local_sendbuf, send_nbytes);
        } else {
            MPIR_Localcopy(((char *) recvbuf +rank * recvcnt * recvtype_extent),
                           recvcnt, recvtype,
                           intra_tmp_buf, send_nbytes, MPI_BYTE);
        }
    }


    if(local_rank !=0 && sendbuf == MPI_IN_PLACE) {
        mpi_errno = MPIR_Limic_Gather_OSU(intra_tmp_buf,  
                                         (intra_sock_comm_size * send_nbytes), 
                                         (recvbuf + (rank*nbytes)), nbytes,
                                         intra_sock_commptr);
    } else {
        mpi_errno = MPIR_Limic_Gather_OSU(intra_tmp_buf, 
                                          (intra_sock_comm_size * send_nbytes), 
                                          local_sendbuf, send_nbytes, 
                                          intra_sock_commptr );
    }
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    
    /*Inter socket gather*/
    if(intra_sock_rank == 0) {
        if (intra_node_leader_rank == 0) {
            tmp_buf = intra_tmp_buf;
        }
        mpi_errno = MPIR_Limic_Gather_OSU(tmp_buf, (local_size * send_nbytes), 
                                          intra_tmp_buf, 
                                          (intra_sock_comm_size * send_nbytes), 
                                          intra_node_leader_commptr );
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
fn_fail:
    /*Free the intra socket leader buffers*/
    if (intra_sock_rank == 0) { 
        if ((intra_node_leader_rank != 0) && (intra_tmp_buf != NULL)) {
            MPIU_Free(intra_tmp_buf);
        }
    }
    
    MPIU_CHKLMEM_FREEALL(); 
    MPIR_TIMER_END(coll,gather,limic_scheme_linear_linear);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Limic_Gather_Scheme_SINGLE_LEADER
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Limic_Gather_Scheme_SINGLE_LEADER(
                                         const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                         void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                         int root, MPID_Comm * comm_ptr, 
                                         MPIR_Errflag_t *errflag) 
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Limic_Gather_Scheme_SINGLE_LEADER\n",comm_ptr->rank,hostname);fflush(stdout);
#endif      
    MPIR_TIMER_START(coll,gather,limic_scheme_single_leader);
    void *local_sendbuf=NULL;
    int rank;
    int local_rank, local_size;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint recvtype_size = 0, nbytes=0;
    int sendtype_iscontig;
    MPI_Aint send_nbytes=0; 
    MPI_Aint recvtype_extent = 0;  /* Datatype extent */
    MPI_Aint true_lb, sendtype_true_extent, recvtype_true_extent;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPI_Aint sendtype_size = 0;
    MPI_Aint position = 0;
    MPIU_CHKLMEM_DECL(1); 

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_limic_scheme_single_leader, 1);
    rank = comm_ptr->rank;

    if (((rank == root) && (recvcnt == 0)) ||
            ((rank != root) && (sendcnt == 0))) {

        MPIR_TIMER_END(coll,gather,limic_scheme_single_leader);
        return MPI_SUCCESS;
    }

    if (sendtype != MPI_DATATYPE_NULL) {
        MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        MPIR_Type_get_true_extent_impl(sendtype, &true_lb,
                &sendtype_true_extent);
    }
    if (recvtype != MPI_DATATYPE_NULL) {
        MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        MPIR_Type_get_true_extent_impl(recvtype, &true_lb,
                &recvtype_true_extent);
    }

    /* extract the rank,size information for the intra-node
     * communicator */
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;


    if (rank == root) {
        nbytes = recvcnt * recvtype_size;

    } else {
        nbytes = sendcnt * sendtype_size;
    }

    /*Pack data for non-contiguous buffer*/
    if ((!sendtype_iscontig) && (sendbuf != MPI_IN_PLACE)) {

        MPIR_Pack_size_impl(1, sendtype, &sendtype_size);
        send_nbytes= sendcnt * sendtype_size;
        MPIU_CHKLMEM_MALLOC(local_sendbuf, void *, send_nbytes, mpi_errno, "local_sendbuf");
        MPIR_Pack_impl(sendbuf, sendcnt, sendtype, local_sendbuf, send_nbytes, &position);
    }
    else {
        local_sendbuf = (void *)sendbuf;
        send_nbytes = nbytes; 
    }

    if (local_rank == 0) {
        /* Node leader, allocate tmp_buffer */
        if (rank == root) {
            tmp_buf = MPIU_Malloc(recvcnt * MPIR_MAX(recvtype_extent,
                        recvtype_true_extent) * local_size);
        } else {
            tmp_buf = MPIU_Malloc( send_nbytes * local_size);
        }
        if (tmp_buf == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                    MPIR_ERR_RECOVERABLE,
                    FCNAME, __LINE__, MPI_ERR_OTHER,
                    "**nomem", 0);
            MPIR_TIMER_END(coll,gather,limic_scheme_single_leader);
            return mpi_errno;
        }

        /*Local copy of buffer*/
        if(sendbuf != MPI_IN_PLACE) {
            MPIU_Memcpy(tmp_buf, local_sendbuf, send_nbytes);
        } else {
            MPIR_Localcopy(((char *) recvbuf +rank * recvcnt * recvtype_extent),
                           recvcnt, recvtype,
                           tmp_buf, send_nbytes, MPI_BYTE);
        } 
    } 

    if(local_rank !=0 && sendbuf == MPI_IN_PLACE) {
        mpi_errno = MPIR_Limic_Gather_OSU(tmp_buf, (local_size * send_nbytes), 
                                          (recvbuf + (rank*nbytes)), 
                                          nbytes, shmem_commptr );
    } else {   
        mpi_errno = MPIR_Limic_Gather_OSU(tmp_buf, (local_size * send_nbytes), 
                                          local_sendbuf, nbytes, 
                                          shmem_commptr );
    }

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

fn_fail:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,gather,limic_scheme_single_leader);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Intra_node_LIMIC_Gather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Intra_node_LIMIC_Gather_MV2(
                                     const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                     void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                     int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Intra_node_LIMIC_Gather_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
#endif        
    MPIR_TIMER_START(coll,gather,intra_node_limic);
    int mpi_errno = MPI_SUCCESS;
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_gather_intra_node_limic, 1);
    /* extract the rank,size information for the intra-node
     * communicator */
	shmem_comm = comm_ptr->dev.ch.shmem_comm;
	MPID_Comm_get_ptr(shmem_comm, shmem_commptr);

    /*This case uses the PT-PT scheme with binomial
     * algorithm */
    if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
            && (num_scheme ==  USE_GATHER_PT_PT_BINOMIAL)) {

        mpi_errno = MPIR_Limic_Gather_Scheme_PT_PT(sendbuf, sendcnt, sendtype,
                                                   recvbuf, recvcnt, recvtype,
                                                   root, comm_ptr, 
                                                   MPIR_Gather_intra,
                                                   errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } 
    /*This case uses the PT-PT scheme with DIRECT
     * algorithm */
    else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
            && (num_scheme == USE_GATHER_PT_PT_DIRECT)) {

        mpi_errno = MPIR_Limic_Gather_Scheme_PT_PT(sendbuf, sendcnt, sendtype,
                                                   recvbuf, recvcnt, recvtype,
                                                   root, comm_ptr, 
                                                   MPIR_Gather_MV2_Direct,
                                                   errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } 
    /*This case uses the PT-LINEAR scheme with binomial
     * algorithm */
    else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
            && (num_scheme == USE_GATHER_PT_LINEAR_BINOMIAL)) {
        
        mpi_errno = MPIR_Limic_Gather_Scheme_PT_LINEAR(sendbuf, sendcnt, sendtype,
                                                       recvbuf, recvcnt, recvtype,
                                                       root, comm_ptr, 
                                                       MPIR_Gather_intra,
                                                       errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

    } 
    /*This case uses the PT-LINEAR scheme with DIRECT
     * algorithm */
    else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
            && (num_scheme == USE_GATHER_PT_LINEAR_DIRECT)) {
        
        mpi_errno = MPIR_Limic_Gather_Scheme_PT_LINEAR(sendbuf, sendcnt, sendtype,
                                                       recvbuf, recvcnt, recvtype,
                                                       root, comm_ptr, 
                                                       MPIR_Gather_MV2_Direct,
                                                       errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

    } 
    /*This case uses the LINEAR-PT scheme with binomial
     * algorithm */
    else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
              && (num_scheme == USE_GATHER_LINEAR_PT_BINOMIAL)) {
        
        mpi_errno = MPIR_Limic_Gather_Scheme_LINEAR_PT(sendbuf, sendcnt, sendtype,
                                                       recvbuf, recvcnt, recvtype,
                                                       root, comm_ptr, 
                                                       MPIR_Gather_intra,
                                                       errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

    } 
    /*This case uses the LINEAR-PT scheme with DIRECT
     * algorithm */
    else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
              && (num_scheme == USE_GATHER_LINEAR_PT_DIRECT)) {
        
        mpi_errno = MPIR_Limic_Gather_Scheme_LINEAR_PT(sendbuf, sendcnt, sendtype,
                                                       recvbuf, recvcnt, recvtype,
                                                       root, comm_ptr, 
                                                       MPIR_Gather_MV2_Direct,
                                                       errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

    } else if((shmem_commptr->dev.ch.use_intra_sock_comm == 1) 
             && (num_scheme == USE_GATHER_LINEAR_LINEAR)) {

        mpi_errno = MPIR_Limic_Gather_Scheme_LINEAR_LINEAR(sendbuf, sendcnt, sendtype,
                                                       recvbuf, recvcnt, recvtype,
                                                       root, comm_ptr,
                                                       errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
      
    } else if(((comm_ptr->dev.ch.shmem_coll_ok == 1) || 
              (shmem_commptr->dev.ch.use_intra_sock_comm == 1))
             && (num_scheme == USE_GATHER_SINGLE_LEADER)) {

        mpi_errno = MPIR_Limic_Gather_Scheme_SINGLE_LEADER(sendbuf, sendcnt, sendtype,
                                                           recvbuf, recvcnt, recvtype,
                                                           root, comm_ptr,
                                                           errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } else {
        /*This is a invalid case, if we are in LIMIC Gather
         * the code flow should be in one of the if-else case*/
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                                         MPIR_ERR_RECOVERABLE,
                                         FCNAME, __LINE__, MPI_ERR_OTHER,
                                         "**badcase", 0);
    }


  fn_fail:
    MPIR_TIMER_END(coll,gather,intra_node_limic);
    return (mpi_errno);
}

#endif    /*#if defined(_SMP_LIMIC_) */


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_index_tuned_intra_MV2(const void *sendbuf,
                    int sendcnt,
                    MPI_Datatype sendtype,
                    void *recvbuf,
                    int recvcnt,
                    MPI_Datatype recvtype,
                    int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag) {
#if GATHER_PRINT_FUN
if (PRINT_FUN_NAME){
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather rank = %d host = %s count=%d  security_approach=%d] Func: MPIR_Gather_index_tuned_intra_MV2\n", comm_ptr->rank, hostname,recvcnt,security_approach);
}
#endif      
    int comm_size_index = 0;
    int inter_node_algo_index = 0;
    int intra_node_algo_index = 0;
    int local_size = 0;
    int partial_sub_ok = 0;
    int conf_index = 0;
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
    MPID_Comm *shmem_commptr = NULL;
    MPI_Comm shmem_comm;
    int mpi_errno = MPI_SUCCESS;
    int comm_size = 0;
    MPI_Aint nbytes = 0;
    MPI_Aint recvtype_size, sendtype_size;
    int rank = -1;
    MPIU_THREADPRIV_DECL;

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    MPIU_THREADPRIV_GET;
    if (rank == root) {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
    } else {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcnt * sendtype_size;
    }
    
    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        if (mv2_gather_indexed_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }

        FIND_PPN_INDEX  (gather, local_size,conf_index, partial_sub_ok)

    }

    if (partial_sub_ok != 1) {
        conf_index = mv2_gather_indexed_num_ppn_conf/2;
    }
    
conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_gather_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_gather_indexed_thresholds_table[conf_index][mv2_size_gather_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
	/* Comm size smaller than smallest configuration in table: use smallest available */
	comm_size_index = 0;
    }
    else if (comm_size > table_max_comm_size) {
	/* Comm size larger than largest configuration in table: use largest available */
	comm_size_index = mv2_size_gather_indexed_tuning_table[conf_index] - 1;
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

    last_inter = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
    table_min_inter_size = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
    table_max_inter_size = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[last_inter].msg_sz;
    last_intra = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].size_intra_table - 1;
    table_min_intra_size = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].intra_node[0].msg_sz;
    table_max_intra_size = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].intra_node[last_intra].msg_sz;
    
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
    
#ifdef _ENABLE_CUDA_
   MPI_Aint sendtype_extent;
   MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
   int recvtype_extent = 0;
   MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
   int send_mem_type = 0;
   int recv_mem_type = 0;
   if (rdma_enable_cuda) {
       send_mem_type = is_device_buffer(sendbuf);
       recv_mem_type = is_device_buffer(recvbuf);
   }
   if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
       rdma_cuda_use_naive && (nbytes <= rdma_cuda_gather_naive_limit/comm_size)) {
       if (sendbuf != MPI_IN_PLACE) {
            if (rank == root) {
                mpi_errno = cuda_stage_alloc (NULL, 0,
                          &recvbuf, recvcnt*recvtype_extent*comm_size, 
                          0, recv_mem_type, 
                          0);
            } else {
                mpi_errno = cuda_stage_alloc ((void **)&sendbuf, sendcnt*sendtype_extent,
                          NULL, 0, 
                          send_mem_type, 0, 
                          0);
            }
       } else {
            mpi_errno = cuda_stage_alloc ((void **)&sendbuf, recvcnt*recvtype_extent,
                      &recvbuf, recvcnt*recvtype_extent*comm_size, 
                      0, recv_mem_type, 
                      rank*recvcnt*recvtype_extent);
       }
       if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
       }
   }


    /* Use Direct algorithm in cuda configuration */
    if (rdma_enable_cuda && (((nbytes > rdma_cuda_gather_naive_limit/comm_size) &&
        rdma_cuda_use_naive) || !rdma_cuda_use_naive)) {
        mpi_errno = MPIR_Gather_MV2_Direct(sendbuf, sendcnt,
                                           sendtype, recvbuf, recvcnt, recvtype,
                                           root, comm_ptr, errflag);
    } else
#endif /*_ENABLE_CUDA_*/

    if (comm_ptr->dev.ch.rank_list != NULL &&
            mv2_use_direct_gather == 1 && mv2_use_two_level_gather == 1 &&
            comm_ptr->dev.ch.shmem_coll_ok == 1) {
        /* Set intra-node function pt for gather_two_level */
        MV2_Gather_intra_node_function = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].
	    intra_node[intra_node_algo_index].MV2_pt_Gather_function;
        /* Set inter-leader pt */
        MV2_Gather_inter_leader_function = mv2_gather_indexed_thresholds_table[conf_index][comm_size_index].
	    inter_leader[inter_node_algo_index].MV2_pt_Gather_function;
        /* We call Gather function */
        mpi_errno =
            MV2_Gather_inter_leader_function(sendbuf, sendcnt, sendtype, recvbuf, recvcnt,
                                             recvtype, root, comm_ptr, errflag);

    } else {
       // printf("calling from here \n");fflush(stdout);
        mpi_errno = MPIR_Gather_intra(sendbuf, sendcnt, sendtype,
                                      recvbuf, recvcnt, recvtype,
                                      root, comm_ptr, errflag);
    }

#ifdef _ENABLE_CUDA_ 
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_gather_naive_limit/comm_size)){
        if (rank == root) {
            cuda_stage_free (NULL, 
                        &recvbuf, recvcnt*recvtype_extent*comm_size,
                        0, recv_mem_type);
        } else {
            cuda_stage_free ((void **)&sendbuf, 
                        NULL, 0,
                        send_mem_type, 0);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_*/     

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2(const void *sendbuf,
                    int sendcnt,
                    MPI_Datatype sendtype,
                    void *recvbuf,
                    int recvcnt,
                    MPI_Datatype recvtype,
                    int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
if (PRINT_FUN_NAME ){
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather osu rank = %d host = %s] Func: MPIR_Gather_MV2\n",comm_ptr->rank,hostname);fflush(stdout);
}
#endif     
    int mpi_errno = MPI_SUCCESS;
    int range = 0;
    int range_threshold = 0;
    int range_intra_threshold = 0;
    MPI_Aint nbytes = 0;
    int comm_size = 0;
    MPI_Aint recvtype_size, sendtype_size;
    int rank = -1;
    MPIU_THREADPRIV_DECL;

    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);
    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    MPIU_THREADPRIV_GET;
    if (rank == root) {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcnt * recvtype_size;
    } else {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcnt * sendtype_size;
    }
    
    if (mv2_use_indexed_tuning || mv2_use_indexed_gather_tuning) {
	mpi_errno =  MPIR_Gather_index_tuned_intra_MV2(sendbuf, sendcnt,
                                           sendtype, recvbuf, recvcnt, recvtype,
                                           root, comm_ptr, errflag);
	goto fn_exit;
    }
    /* Search for the corresponding system size inside the tuning table */
    while ((range < (mv2_size_gather_tuning_table - 1)) &&
           (comm_size > mv2_gather_thresholds_table[range].numproc)) {
        range++;
    }
    /* Search for corresponding inter-leader function */
    while ((range_threshold < (mv2_gather_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_gather_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_gather_thresholds_table[range].inter_leader[range_threshold].max !=
               -1)) {
        range_threshold++;
    }

    /* Search for corresponding intra node function */
    while ((range_intra_threshold < (mv2_gather_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_gather_thresholds_table[range].intra_node[range_intra_threshold].max)
           && (mv2_gather_thresholds_table[range].intra_node[range_intra_threshold].max !=
               -1)) {
        range_intra_threshold++;
    }
#if defined(_SMP_LIMIC_)   
    int range_limic_scheme = 0;
    if (use_limic_gather){
        /* Search for corresponding limic-scheme function */
        while ((range_limic_scheme < (mv2_gather_thresholds_table[range].nb_limic_scheme - 1))
                && (nbytes >
                    mv2_gather_thresholds_table[range].limic_gather_scheme[range_limic_scheme].max)
                && (mv2_gather_thresholds_table[range].limic_gather_scheme[range_limic_scheme].max !=
                    -1)) {
            range_limic_scheme++;
        }
        num_scheme =   mv2_gather_thresholds_table[range].limic_gather_scheme[range_limic_scheme].scheme;
    }
#endif /*#if defined(_SMP_LIMIC_)*/
#ifdef _ENABLE_CUDA_
   MPI_Aint sendtype_extent;
   MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
   int recvtype_extent = 0;
   MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
   int send_mem_type = 0;
   int recv_mem_type = 0;
   if (rdma_enable_cuda) {
       send_mem_type = is_device_buffer(sendbuf);
       recv_mem_type = is_device_buffer(recvbuf);
   }
   if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
       rdma_cuda_use_naive && (nbytes <= rdma_cuda_gather_naive_limit/comm_size)) {
       if (sendbuf != MPI_IN_PLACE) {
            if (rank == root) {
                mpi_errno = cuda_stage_alloc (NULL, 0,
                          &recvbuf, recvcnt*recvtype_extent*comm_size, 
                          0, recv_mem_type, 
                          0);
            } else {
                mpi_errno = cuda_stage_alloc ((void **)&sendbuf, sendcnt*sendtype_extent,
                          NULL, 0, 
                          send_mem_type, 0, 
                          0);
            }
       } else {
            mpi_errno = cuda_stage_alloc ((void **)&sendbuf, recvcnt*recvtype_extent,
                      &recvbuf, recvcnt*recvtype_extent*comm_size, 
                      0, recv_mem_type, 
                      rank*recvcnt*recvtype_extent);
       }
       if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
       }
   }


    /* Use Direct algorithm in cuda configuration */
    if (rdma_enable_cuda && (((nbytes > rdma_cuda_gather_naive_limit/comm_size) &&
        rdma_cuda_use_naive) || !rdma_cuda_use_naive)) {
        mpi_errno = MPIR_Gather_MV2_Direct(sendbuf, sendcnt,
                                           sendtype, recvbuf, recvcnt, recvtype,
                                           root, comm_ptr, errflag);
    } else
#endif /*_ENABLE_CUDA_*/

    if (comm_ptr->dev.ch.is_global_block == 1 && mv2_use_direct_gather == 1 &&
            mv2_use_two_level_gather == 1 && comm_ptr->dev.ch.shmem_coll_ok == 1) {
        /* Set intra-node function pt for gather_two_level */
        MV2_Gather_intra_node_function = 
                              mv2_gather_thresholds_table[range].intra_node[range_intra_threshold].
                              MV2_pt_Gather_function;
        /* Set inter-leader pt */
        MV2_Gather_inter_leader_function =
                              mv2_gather_thresholds_table[range].inter_leader[range_threshold].
                              MV2_pt_Gather_function;
        /* We call Gather function */
        mpi_errno =
            MV2_Gather_inter_leader_function(sendbuf, sendcnt, sendtype, recvbuf, recvcnt,
                                             recvtype, root, comm_ptr, errflag);

    } else {
        mpi_errno = MPIR_Gather_intra(sendbuf, sendcnt, sendtype,
                                      recvbuf, recvcnt, recvtype,
                                      root, comm_ptr, errflag);
    }

#ifdef _ENABLE_CUDA_ 
    if (rdma_enable_cuda && (send_mem_type || recv_mem_type) &&
        rdma_cuda_use_naive && (nbytes <= rdma_cuda_gather_naive_limit/comm_size)){
        if (rank == root) {
            cuda_stage_free (NULL, 
                        &recvbuf, recvcnt*recvtype_extent*comm_size,
                        0, recv_mem_type);
        } else {
            cuda_stage_free ((void **)&sendbuf, 
                        NULL, 0,
                        send_mem_type, 0);
        }
    }
#endif                          /*#ifdef _ENABLE_CUDA_*/     

    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* end:nested */


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct_CHS
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_CHS(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather rank = %d host = %s count=%d  security_approach=%d] Func: MPIR_Gather_MV2_Direct_CHS\n", comm_ptr->rank, hostname,recvcount,security_approach);
}
#endif 
   if (comm_ptr->dev.ch.is_uniform != 1  || (comm_ptr->dev.ch.is_global_block != 1 || comm_ptr->dev.ch.is_blocked != 1) || root!=0)
    {
        //printf("call MPIR_Gather_intra\n");fflush(stdout);
         int ret = MPIR_Gather_intra(sendbuf, sendcount, sendtype,
                                                             recvbuf, recvcount, recvtype,
                                                             root, comm_ptr, errflag);
            return ret;
    } 
    //count_ite++;
    /*if(comm_ptr->rank == 0)
            printf("0. rank %d count_ite %d  comm_ptr->local_size %d comm_ptr->dev.ch.allgather_comm_ok %d comm_ptr->dev.ch.shmem_coll_ok %d\n",
            comm_ptr->rank, count_ite, comm_ptr->local_size,comm_ptr->dev.ch.allgather_comm_ok,comm_ptr->dev.ch.shmem_coll_ok);fflush(stdout);
    */
    int rank, comm_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint sendtype_extent, recvtype_extent;
    int i, reqs;
    MPID_Request **reqarray;
    MPI_Status *starray;
    
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr;
    MPI_Comm conc_comm;
    MPID_Comm *conc_commptr = NULL;
    MPID_Node_id_t rank_node_id, root_node_id;
   
    int local_size, local_rank, conc_rank, conc_size;
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

#if 0
    if(comm_ptr->dev.ch.shmem_coll_ok == 0)
    {
        //printf("rank %d count_ite %d\n",comm_ptr->rank, count_ite);fflush(stdout);
         /* If comm_ptr does not have leader/shmem sub-communicators,
                 * create them now */ 
                mpi_errno = create_2level_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
                if(mpi_errno) {
                   MPIR_ERR_POP(mpi_errno);
                }
                //printf("rank %d count_ite %d comm_ptr->dev.ch.shmem_coll_ok %d\n",
                //comm_ptr->rank, count_ite,comm_ptr->dev.ch.shmem_coll_ok);fflush(stdout);

    }
#endif    

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 01 \n",rank);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    //printf("Done get_ptr: rank %d count_ite %d\n",comm_ptr->rank, count_ite);fflush(stdout);
    //conc_size = 8;

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 02 \n",rank);
    conc_comm = comm_ptr->dev.ch.concurrent_comm;
    MPID_Comm_get_ptr(conc_comm, conc_commptr);
    conc_rank = conc_commptr->rank;
    conc_size = conc_commptr->local_size;

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 03 \n",rank);
     
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
    //printf("Done get_rank_size: rank %d count_ite %d\n",comm_ptr->rank, count_ite);fflush(stdout);
   
    MPIU_CHKLMEM_DECL(2);
    
if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 04 \n",rank);
    rank = comm_ptr->rank;
     MPID_Get_node_id(comm_ptr, rank, &rank_node_id);
    MPID_Get_node_id(comm_ptr, root, &root_node_id);
    //printf("Done get_nodeid: rank %d count_ite %d\n",comm_ptr->rank, count_ite);fflush(stdout);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /*
    * copy the data in shared mem
    * encrypt the data
    */
   
   
if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 05 \n",rank);
    comm_size = comm_ptr->local_size;
    //printf(" comm_ptr->local_size %d comm_ptr->remote_size %d local_size %d\n",comm_ptr->local_size,comm_ptr->remote_size,local_size);fflush(stdout);
       MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **,
                    comm_size * sizeof (MPID_Request*), mpi_errno,
                    "reqarray");
        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *,
                    comm_size * sizeof (MPI_Status), mpi_errno,
                    "starray");             
                    

if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 06 \n",rank);
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
        for (i = 1; i < conc_size; i++)
        {
            if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 07 \n",rank);
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t + 28, MPI_BYTE, i, MPIR_GATHER_TAG, conc_commptr, &reqarray[reqs++]);
            if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 08 \n",rank);
            // mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t + 28, MPI_BYTE, src_global_rank, MPIR_GATHER_TAG, comm_ptr, &reqarray[reqs++]);
            next += (t + 28);
             src_global_rank += local_size;
        }
       // if(rank == 0 && count_ite >= 16)
         //  printf("2. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);

        /* receiver side */
        next = 0;
        src_global_rank = rank + local_size;
        for (i = 1; i < conc_size; i++)
        {
            dest = src_global_rank * t; // get the decryption location
            if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 09 \n",rank);
            mpi_errno = MPIC_Waitall(1, &reqarray[i], starray, errflag);
            if (!EVP_AEAD_CTX_open(global_ctx, shmem_buffer + dest,
                                   &count, max_out_len,
                                   (ciphertext_recvbuf + next), 12,
                                   (ciphertext_recvbuf + next + 12), (unsigned long)(t + 16),
                                   NULL, 0))
            {
                printf("Decryption error gather CHS:[rank %d rank_node_id %d root_node_id %d] gather direct recv i %d\n", rank, rank_node_id,root_node_id,i);
                fflush(stdout);
            }
#if 0
            else
            {
                printf(COLOR_BLUE "Gather CHS recv:: [rank %d rank_node_id %d source %d] Decrypted data size %lu dest %d" COLOR_RESET "\n",
                       rank, rank_node_id, src_global_rank, count, dest);
                fflush(stdout);
            }
#endif
if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 10 \n",rank);
            next += (t + 28);
            src_global_rank += local_size;
        }

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
        if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 11 \n",rank);
    }
    //printf("we are here\n");fflush(stdout);

    /* sender side*/
    if (local_size != comm_size)
    {
       
        int t = (sendtype_extent * sendcount);
        unsigned long max_out_len = (unsigned long)(sendtype_extent * sendcount+16);
        unsigned long ciphertext_sendbuf_len;
        int global_dest_rank = rank % local_size;
        if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 12 \n",rank);

        // if(rank == 0 && count_ite >= 16)
         //   printf("3. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);
        if(rank_node_id != root_node_id)
        {
           
             RAND_bytes(ciphertext_sendbuf, 12);
             if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 13 \n",rank);
             
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
if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 14 \n",rank);
               // printf(COLOR_MAGENTA"CHS gather [rank %d conc_rank %d] sending to dest_rank %d dest conc rank %d"COLOR_RESET"\n",rank, conc_rank, global_dest_rank,0);fflush(stdout);
                    mpi_errno = MPIC_Send(ciphertext_sendbuf,
                                           (t + 28),
                                           MPI_BYTE, 0, MPIR_GATHER_TAG, conc_commptr,
                                            errflag);
                   /*mpi_errno = MPIC_Send(ciphertext_sendbuf,
                                           (t + 28),
                                           MPI_BYTE, global_dest_rank, MPIR_GATHER_TAG, comm_ptr,
                                            errflag);*/                        
                                          
if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 15 \n",rank);           
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
     // if(rank == 0 && count_ite >= 16)
       //     printf("4. rank %d count_ite %d\n",rank, count_ite);fflush(stdout);

     /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (DEBUG_INIT_FILE) fprintf(stderr,"[%d] Ga-CHS 15 \n",rank);
   
    
    //printf(COLOR_YELLOW"@@ rank %d root %d rank_node_id %d root_node_id %d"COLOR_RESET"\n", rank, root,rank_node_id,root_node_id);fflush(stdout);

   

  fn_exit:
    MPIU_CHKLMEM_FREEALL(); 
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIR_Gather_MV2_Direct_CHS_UNENCRYPTED
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Gather_MV2_Direct_CHS_UNENCRYPTED(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPID_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag)
{
#if GATHER_PRINT_FUN
if (PRINT_FUN_NAME){
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[gather rank = %d host = %s count=%d  security_approach=%d] Func: MPIR_Gather_MV2_Direct_CHS_UNENCRYPTED\n", comm_ptr->rank, hostname,recvcount,security_approach);
}
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


    int next = 0;
    int dest = 0;
    int src_global_rank = 0;
    int j = 0;
    if (rank_node_id == root_node_id && local_size != comm_size)
    {

        int t = (recvtype_extent * recvcount);
        unsigned long max_out_len = (unsigned long)(t);
        unsigned long count;

        int each_rank_size = comm_size / local_size; // consider if equal now
        int max_ranks = local_rank * each_rank_size + each_rank_size;
        // printf("rank %d each_rank_size %d max_ranks %d\n", rank, each_rank_size, max_ranks);fflush(stdout);
        dest = 0;
        reqs = 1;

        /* receiver side */
        next = 0;
        for (i = 1; i < conc_size; i++)
        {
            mpi_errno = MPIC_Irecv(ciphertext_recvbuf + next, t, MPI_BYTE, i, MPIR_GATHER_TAG, conc_commptr, &reqarray[reqs++]);
            next += (t);
        }

        /* receiver side */
        next = 0;
        src_global_rank = rank + local_size;
        for (i = 1; i < conc_size; i++)
        {
            dest = src_global_rank * t; // get the recv location
            mpi_errno = MPIC_Waitall(1, &reqarray[i], starray, errflag);
            memcpy(shmem_buffer + dest, ciphertext_recvbuf + next, t);
            next += (t);
            src_global_rank += local_size;
        }

        /* itner-node recving is done, now copy own data */
        next = rank * t;
        memcpy(shmem_buffer + next, sendbuf, t);

        /* wait for all local ranks to finish */
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

    /* sender side*/
    if (local_size != comm_size)
    {
       
        int t = (sendtype_extent * sendcount);
        unsigned long max_out_len = (unsigned long)(sendtype_extent * sendcount+16);
        unsigned long ciphertext_sendbuf_len;
        int global_dest_rank = rank % local_size;

       
        if(rank_node_id != root_node_id)
        {
           
            

               // printf(COLOR_MAGENTA"CHS gather [rank %d conc_rank %d] sending to dest_rank %d dest conc rank %d"COLOR_RESET"\n",rank, conc_rank, global_dest_rank,0);fflush(stdout);
                    mpi_errno = MPIC_Send(sendbuf,
                                           (t),
                                           MPI_BYTE, 0, MPIR_GATHER_TAG, conc_commptr,
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
        printf("CHS gather un-uniform case not handled yet\n");fflush(stdout);
    }


     /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    
    //printf(COLOR_YELLOW"@@ rank %d root %d rank_node_id %d root_node_id %d"COLOR_RESET"\n", rank, root,rank_node_id,root_node_id);fflush(stdout);

   

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

