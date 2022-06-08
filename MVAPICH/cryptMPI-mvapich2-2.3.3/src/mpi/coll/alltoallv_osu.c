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
#include "alltoallv_tuning.h"

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_pw);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_intra_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_intra_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_intra_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_intra_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_alltoallv_count_recv);

int (*MV2_Alltoallv_function) (const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)=NULL;

/* This is the default implementation of alltoallv. The algorithm is:
   
   Algorithm: MPI_Alltoallv

   Since each process sends/receives different amounts of data to
   every other process, we don't know the total message size for all
   processes without additional communication. Therefore we simply use
   the "middle of the road" isend/irecv algorithm that works
   reasonably well in all cases.

   We post all irecvs and isends and then do a waitall. We scatter the
   order of sources and destinations among the processes, so that all
   processes don't try to send/recv to/from the same process at the
   same time. 

   Possible improvements: 

   End Algorithm: MPI_Alltoallv
*/

/* begin:nested */
/* not declared static because a machine-specific function may call this one in some cases */
#undef FUNCNAME
#define FUNCNAME MPIR_Alltoallv_intra_scatter_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Alltoallv_intra_scatter_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int comm_size, i, j ;
    MPI_Aint send_extent, recv_extent;
    int mpi_errno = MPI_SUCCESS;
    int src, dst, rank;
    int ii, ss, bblock;
    int sreq_cnt, rreq_cnt;
    size_t sent_bytes = 0; 
    size_t max_bytes = mv2_alltoallv_intermediate_wait_threshold;
    MPI_Status *sstarray = NULL;
    MPID_Request **sreqarray = NULL;
    MPI_Status *rstarray = NULL;
    MPID_Request **rreqarray = NULL;
    int total_count = 0;
    const void *sendbuf_tmp = NULL;
    MPI_Aint true_extent, true_lb;
    const int *sendcnt_tmp = NULL;
    const int *sdispls_tmp = NULL;
    MPI_Datatype sendtype_tmp;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get extent of send and recv types */
    MPID_Datatype_get_extent_macro(recvtype, recv_extent);
    MPID_Datatype_get_extent_macro(recvtype, send_extent);



    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* If MPI_IN_PLACE is passed to the send buffer, copy the receive buffer to
     * sendbuf_tmp and then run as normal */
    if (sendbuf == MPI_IN_PLACE) {


        MPIR_Type_get_true_extent_impl(recvtype, &true_lb, &true_extent);

        total_count = rdispls[comm_size - 1] + recvcnts[comm_size - 1];

        MPIU_Ensure_Aint_fits_in_pointer(total_count * (MPIR_MAX(recv_extent,
                        true_extent)));

        sendbuf_tmp = (char *)MPIU_Malloc(total_count*(MPIR_MAX(recv_extent, true_extent)));

        /* adjust for potential negative lower bound in datatype */
        sendbuf_tmp = (void *)((char*)sendbuf_tmp - true_lb);
       
        mpi_errno = MPIR_Localcopy(((char *) recvbuf), total_count, recvtype,
              ((char *) sendbuf_tmp), total_count, recvtype);

        /* in th MPI_IN_PLACE case, all the followings for sender could be NULL,
         * use the receiver info instead of sender info  */
        sendcnt_tmp = recvcnts;
        sdispls_tmp = rdispls;
        sendtype_tmp = recvtype;

        MPID_Datatype_get_extent_macro(sendtype_tmp, send_extent);

    } else {

        sendbuf_tmp = sendbuf;
        sendcnt_tmp = sendcnts;
        sdispls_tmp = sdispls;
        sendtype_tmp = sendtype;

        MPID_Datatype_get_extent_macro(sendtype, send_extent);
    }

    MPIU_CHKLMEM_DECL(4);
    bblock = mv2_coll_param.alltoall_throttle_factor;

    if (bblock >= comm_size) {
        bblock = comm_size;
    }
    /* If throttle_factor is n, each process posts n pairs of isend/irecv
     * in each iteration. */

    /* FIXME: This should use the memory macros (there are storage
     * leaks here if there is an error, for example) */
    MPIU_CHKLMEM_MALLOC(sreqarray, MPID_Request **, bblock*sizeof(MPID_Request*),
            mpi_errno, "sreqarray");

    MPIU_CHKLMEM_MALLOC(sstarray, MPI_Status *, bblock*sizeof(MPI_Status),
            mpi_errno, "sstarray");

    MPIU_CHKLMEM_MALLOC(rreqarray, MPID_Request **, bblock*sizeof(MPID_Request*),
            mpi_errno, "rreqarray");

    MPIU_CHKLMEM_MALLOC(rstarray, MPI_Status *, bblock*sizeof(MPI_Status),
            mpi_errno, "rstarray");

    mpi_errno = MPIR_Localcopy(((char *) sendbuf_tmp + sdispls_tmp[rank] * send_extent),
                                sendcnt_tmp[rank], sendtype_tmp,
                                ((char *) recvbuf + rdispls[rank] * recv_extent),
                                recvcnts[rank], recvtype);
    if (mpi_errno) {
        mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                FCNAME, __LINE__, MPI_ERR_OTHER,
                "**fail", 0);
        return mpi_errno;
    }

    /* Do the exchanges */
    for (i = 0, ii=0; ii<comm_size; ii+=bblock) {
        sreq_cnt = rreq_cnt = 0;
        ss = comm_size-ii < bblock ? comm_size-ii : bblock;
        /* do the communication -- post ss receives: */
        for ( i=0; i<ss; i++ ) {
            src = (rank + i + ii) % comm_size;
            MPIR_PVAR_INC(alltoallv, intra, recv, recvcnts[src], recvtype);
            mpi_errno = MPIC_Irecv(((char *) recvbuf +
                                    rdispls[src] * recv_extent),
                                    recvcnts[src], recvtype, src,
                                    MPIR_ALLTOALLV_TAG, comm_ptr,
                                    &rreqarray[rreq_cnt]);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            rreq_cnt++;
        }
        /* do the communication -- post ss sends : */
        for ( i=0; i<ss; i++ ) {
            dst = (rank - i - ii + comm_size) % comm_size;
            MPIR_PVAR_INC(alltoallv, intra, send, sendcnt_tmp[dst], sendtype_tmp);
            mpi_errno = MPIC_Isend(((char *) sendbuf_tmp +
                                    sdispls_tmp[dst] * send_extent),
                                    sendcnt_tmp[dst], sendtype_tmp, dst,
                                    MPIR_ALLTOALLV_TAG, comm_ptr,
                                    &sreqarray[sreq_cnt], errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            sreq_cnt++;

            /* Throttle sends dynamically if pushing large amount of data */
            sent_bytes += send_extent * sendcnt_tmp[dst];
            if (max_bytes && sent_bytes >= max_bytes) {
                mpi_errno = MPIC_Waitall(sreq_cnt,sreqarray,sstarray, errflag);
                if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
                    MPIR_ERR_POP(mpi_errno);
                }
                sreq_cnt = 0;
            }
        }

        /* wait for recv to complete then wait for remainging sends*/
        mpi_errno = MPIC_Waitall(rreq_cnt,rreqarray,rstarray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
            MPIR_ERR_POP(mpi_errno);
        }

        mpi_errno = MPIC_Waitall(sreq_cnt,sreqarray,sstarray, errflag);
        if (mpi_errno && mpi_errno != MPI_ERR_IN_STATUS) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno == MPI_ERR_IN_STATUS) {
            for (j=0; j<rreq_cnt; j++) {
                if (rstarray[j].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = rstarray[j].MPI_ERROR;
                }
            }
            for (j=0; j<sreq_cnt; j++) {
                if (sstarray[j].MPI_ERROR != MPI_SUCCESS) {
                    mpi_errno = sstarray[j].MPI_ERROR;
                }
            }
        }
    }

    /* --END ERROR HANDLING-- */
    MPIU_CHKLMEM_FREEALL();
    if (sendbuf == MPI_IN_PLACE) {
        MPIU_Free(sendbuf_tmp);
    }

fn_fail:
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    return (mpi_errno);
}

#undef FUNCNAME
#define FUNCNAME MPIR_Alltoallv_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Alltoallv_intra_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_alltoallv_pw, 1);
    int comm_size, i, j;
    MPI_Aint send_extent, recv_extent;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int dst, rank;

    int pof2, src;
    MPI_Status status;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get extent of send and recv types */
    MPID_Datatype_get_extent_macro(sendtype, send_extent);
    MPID_Datatype_get_extent_macro(recvtype, recv_extent);

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    if (sendbuf == MPI_IN_PLACE) {
        /* We use pair-wise sendrecv_replace in order to conserve memory usage,
         * which is keeping with the spirit of the MPI-2.2 Standard.  But
         * because of this approach all processes must agree on the global
         * schedule of sendrecv_replace operations to avoid deadlock.
         *
         * Note that this is not an especially efficient algorithm in terms of
         * time and there will be multiple repeated malloc/free's rather than
         * maintaining a single buffer across the whole loop.  Something like
         * MADRE is probably the best solution for the MPI_IN_PLACE scenario. */
        for (i = 0; i < comm_size; ++i) {
            /* start inner loop at i to avoid re-exchanging data */
            for (j = i; j < comm_size; ++j) {
                if (rank == i) {
                    /* also covers the (rank == i && rank == j) case */
                    MPIR_PVAR_INC(alltoallv, intra, send, recvcnts[j], recvtype);
                    MPIR_PVAR_INC(alltoallv, intra, recv, recvcnts[j], recvtype);
                    mpi_errno =
                        MPIC_Sendrecv_replace(((char *) recvbuf +
                                                  rdispls[j] * recv_extent),
                                                 recvcnts[j], recvtype, j,
                                                 MPIR_ALLTOALL_TAG, j,
                                                 MPIR_ALLTOALL_TAG, comm_ptr,
                                                 &status, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but
                         * continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                } else if (rank == j) {
                    /* same as above with i/j args reversed */
                    MPIR_PVAR_INC(alltoallv, intra, send, recvcnts[j], recvtype);
                    MPIR_PVAR_INC(alltoallv, intra, recv, recvcnts[j], recvtype);
                    mpi_errno =
                        MPIC_Sendrecv_replace(((char *) recvbuf +
                                                  rdispls[i] * recv_extent),
                                                 recvcnts[i], recvtype, i,
                                                 MPIR_ALLTOALL_TAG, i,
                                                 MPIR_ALLTOALL_TAG, comm_ptr,
                                                 &status, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but
                         * continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }
                }
            }
        }
    } else if (mv2_use_scatter_dest_alltoallv) {
        mpi_errno = MPIR_Alltoallv_intra_scatter_MV2(sendbuf,
                             sendcnts,
                             sdispls,
                             sendtype,
                             recvbuf,
                             recvcnts,
                             rdispls,
                             recvtype,
                             comm_ptr, errflag);
    } else {
        mpi_errno = MPIR_Localcopy(((char *) sendbuf +
                                    sdispls[rank] * send_extent),
                                   sendcnts[rank], sendtype,
                                   ((char *) recvbuf +
                                    rdispls[rank] * recv_extent),
                                   recvcnts[rank], recvtype);

        if (mpi_errno) {
            mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                             FCNAME, __LINE__, MPI_ERR_OTHER,
                                             "**fail", 0);
            return mpi_errno;
        }

        /* Is comm_size a power-of-two? */
        pof2 = comm_ptr->dev.ch.is_pof2;

        /* Do the pairwise exchanges */
        for (i = 1; i < comm_size; i++) {
            if (pof2 == 1) {
                /* use exclusive-or algorithm */
                src = dst = rank ^ i;
            } else {
                src = (rank - i + comm_size) % comm_size;
                dst = (rank + i) % comm_size;
            }
            MPIR_PVAR_INC(alltoallv, intra, send, sendcnts[dst], sendtype);
            MPIR_PVAR_INC(alltoallv, intra, recv, recvcnts[src], recvtype);
            mpi_errno = MPIC_Sendrecv(((char *) sendbuf +
                                          sdispls[dst] * send_extent),
                                         sendcnts[dst], sendtype, dst,
                                         MPIR_ALLTOALL_TAG,
                                         ((char *) recvbuf +
                                          rdispls[src] * recv_extent),
                                         recvcnts[src], recvtype, src,
                                         MPIR_ALLTOALL_TAG, comm_ptr, &status,
                                         errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but
                 * continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

        }
    }

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);
    return (mpi_errno);
}

/* end:nested */

#undef FUNCNAME
#define FUNCNAME MPIR_Alltoallv_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoallv_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;

    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoallv_tuning) {
        mpi_errno = MPIR_Alltoallv_index_tuned_intra_MV2(sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm_ptr, errflag);
    }
    else {
        mpi_errno = MPIR_Alltoallv_intra_MV2(sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm_ptr, errflag);
    }

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Alltoallv_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Alltoallv_index_tuned_intra_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno=MPI_SUCCESS;
    int partial_sub_ok = 0;
    int conf_index = 0;
    int local_size = -1;
    int comm_size_index = 0;
    int inter_node_algo_index = 0;
    int table_min_comm_size = 0;
    int table_max_comm_size = 0;
    int lp2ltn; // largest power of 2 less than n
    MPI_Comm shmem_comm;
    MPID_Comm *shmem_commptr=NULL;
    int comm_size = comm_ptr->local_size;

    if (mv2_alltoallv_indexed_table_ppn_conf[0] == -1) {
        /* Indicating user defined tuning */
        conf_index = 0;
        goto conf_check_end;
    }

    if (likely(mv2_enable_skip_tuning_table_search)) {
        /* Force scatter dest algo */
        MV2_Alltoallv_function = MPIR_Alltoallv_intra_scatter_MV2;
        goto skip_tuning_tables;
    }

    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        FIND_PPN_INDEX  (alltoallv, local_size,conf_index, partial_sub_ok)
    }
    
    if (partial_sub_ok != 1) {
        conf_index = mv2_alltoallv_indexed_num_ppn_conf/2;
    }
    
conf_check_end:

    /* Search for the corresponding system size inside the tuning table */
    /*
     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
     */
    table_min_comm_size = mv2_alltoallv_indexed_thresholds_table[conf_index][0].numproc;
    table_max_comm_size =
	mv2_alltoallv_indexed_thresholds_table[conf_index][mv2_size_alltoallv_indexed_tuning_table[conf_index] - 1].numproc;
    
    if (comm_size < table_min_comm_size) {
        /* Comm size smaller than smallest configuration in table: use smallest available */
        comm_size_index = 0;
    } else if (comm_size > table_max_comm_size) {
        /* Comm size larger than largest configuration in table: use largest available */
        comm_size_index = mv2_size_alltoallv_indexed_tuning_table[conf_index] - 1;
    } else {
        /* Comm size in between smallest and largest configuration: find closest match */
        if (comm_ptr->dev.ch.is_pof2) {
            comm_size_index = log2( comm_size / table_min_comm_size );
        } else {
            lp2ltn = pow(2, (int)log2(comm_size));
            comm_size_index = (lp2ltn < table_min_comm_size) ? 0 : log2( lp2ltn / table_min_comm_size );
        }
    }

    inter_node_algo_index = 0;

    MV2_Alltoallv_function = mv2_alltoallv_indexed_thresholds_table[conf_index][comm_size_index].algo_table[inter_node_algo_index]
        .MV2_pt_Alltoallv_function;

skip_tuning_tables:
    mpi_errno = MV2_Alltoallv_function(sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm_ptr, errflag );
    return (mpi_errno);
}
