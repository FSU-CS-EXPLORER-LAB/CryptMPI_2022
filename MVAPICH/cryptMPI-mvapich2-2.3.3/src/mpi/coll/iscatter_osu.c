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
#include "collutil.h"

#include "iscatter_tuning.h"

int (*MV2_Iscatter_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

int (*MV2_Iscatter_intra_node_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_binomial_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_binomial_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_binomial_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_binomial_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iscatter_count_recv);

/* This function is the same as MPIR_Iscatter_intra but has been
   renamed for convenience */

struct shared_state {
    int sendcount;
    int curr_count;
    MPI_Aint send_subtree_count;
    int nbytes;
    MPI_Status status;
};

static int get_count(MPID_Comm *comm, int tag, void *state)
{
    struct shared_state *ss = state;
    MPIR_Get_count_impl(&ss->status, MPI_BYTE, &ss->curr_count);
    return MPI_SUCCESS;
}

static int calc_send_count_root(MPID_Comm *comm, int tag, void *state, void *state2)
{
    struct shared_state *ss = state;
    int mask = (int)(size_t)state2;
    ss->send_subtree_count = ss->curr_count - ss->sendcount * mask;
    return MPI_SUCCESS;
}

static int calc_send_count_non_root(MPID_Comm *comm, int tag, void *state, void *state2)
{
    struct shared_state *ss = state;
    int mask = (int)(size_t)state2;
    ss->send_subtree_count = ss->curr_count - ss->nbytes * mask;
    return MPI_SUCCESS;
}

static int calc_curr_count(MPID_Comm *comm, int tag, void *state)
{
    struct shared_state *ss = state;
    ss->curr_count -= ss->send_subtree_count;
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iscatter_binomial
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iscatter_binomial(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                        void *recvbuf, int recvcount, MPI_Datatype recvtype,
                        int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint extent = 0;
    int rank, comm_size, is_homogeneous, sendtype_size;
    int relative_rank;
    int mask, recvtype_size=0, src, dst;
    int tmp_buf_size = 0;
    void *tmp_buf = NULL;
    struct shared_state *ss = NULL;
    MPIR_SCHED_CHKPMEM_DECL(4);

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (((rank == root) && (sendcount == 0)) || ((rank != root) && (recvcount == 0)))
        goto fn_exit;

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif

/* Use binomial tree algorithm */

    MPIR_SCHED_CHKPMEM_MALLOC(ss, struct shared_state *, sizeof(struct shared_state), mpi_errno, "shared_state");
    ss->sendcount = sendcount;

    if (rank == root)
        MPID_Datatype_get_extent_macro(sendtype, extent);

    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    if (is_homogeneous) {
        /* communicator is homogeneous */
        if (rank == root) {
            /* We separate the two cases (root and non-root) because
               in the event of recvbuf=MPI_IN_PLACE on the root,
               recvcount and recvtype are not valid */
            MPID_Datatype_get_size_macro(sendtype, sendtype_size);
            MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                             extent*sendcount*comm_size);

            ss->nbytes = sendtype_size * sendcount;
        }
        else {
            MPID_Datatype_get_size_macro(recvtype, recvtype_size);
            MPIU_Ensure_Aint_fits_in_pointer(extent*recvcount*comm_size);
            ss->nbytes = recvtype_size * recvcount;
        }

        ss->curr_count = 0;

        /* all even nodes other than root need a temporary buffer to
           receive data of max size (ss->nbytes*comm_size)/2 */
        if (relative_rank && !(relative_rank % 2)) {
            tmp_buf_size = (ss->nbytes*comm_size)/2;
            MPIR_SCHED_CHKPMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");
        }

        /* if the root is not rank 0, we reorder the sendbuf in order of
           relative ranks and copy it into a temporary buffer, so that
           all the sends from the root are contiguous and in the right
           order. */
        if (rank == root) {
            if (root != 0) {
                tmp_buf_size = ss->nbytes*comm_size;
                MPIR_SCHED_CHKPMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");

                if (recvbuf != MPI_IN_PLACE)
                    mpi_errno = MPID_Sched_copy(((char *) sendbuf + extent*sendcount*rank),
                                                sendcount*(comm_size-rank), sendtype,
                                                tmp_buf, ss->nbytes*(comm_size-rank), MPI_BYTE, s);
                else
                    mpi_errno = MPID_Sched_copy(((char *) sendbuf + extent*sendcount*(rank+1)),
                                                sendcount*(comm_size-rank-1), sendtype,
                                                ((char *)tmp_buf + ss->nbytes),
                                                ss->nbytes*(comm_size-rank-1), MPI_BYTE, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                mpi_errno = MPID_Sched_copy(sendbuf, sendcount*rank, sendtype,
                                            ((char *) tmp_buf + ss->nbytes*(comm_size-rank)),
                                            ss->nbytes*rank, MPI_BYTE, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                MPID_SCHED_BARRIER(s);
                ss->curr_count = ss->nbytes*comm_size;
            }
            else
                ss->curr_count = sendcount*comm_size;
        }

        /* root has all the data; others have zero so far */

        mask = 0x1;
        while (mask < comm_size) {
            if (relative_rank & mask) {
                src = rank - mask;
                if (src < 0) src += comm_size;

                /* The leaf nodes receive directly into recvbuf because
                   they don't have to forward data to anyone. Others
                   receive data into a temporary buffer. */
                if (relative_rank % 2) {
                    mpi_errno = MPID_Sched_recv(recvbuf, recvcount, recvtype, src, comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);
                }
                else {

                    /* the recv size is larger than what may be sent in
                       some cases. query amount of data actually received */
                    mpi_errno = MPID_Sched_recv_status(tmp_buf, tmp_buf_size, MPI_BYTE, src, comm_ptr, &ss->status, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);
                    mpi_errno = MPID_Sched_cb(&get_count, ss, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);
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
                if (dst >= comm_size) dst -= comm_size;

                if ((rank == root) && (root == 0))
                {
#if 0
                    /* FIXME how can this be right? shouldn't (sendcount*mask)
                     * be the amount sent and curr_cnt be reduced by that?  Or
                     * is it always true the (curr_cnt/2==sendcount*mask)? */
                    send_subtree_cnt = curr_cnt - sendcount * mask;
#endif
                    mpi_errno = MPID_Sched_cb2(&calc_send_count_root, ss, ((void *)(size_t)mask), s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);

                    /* mask is also the size of this process's subtree */
                    mpi_errno = MPID_Sched_send_defer(((char *)sendbuf + extent*sendcount*mask),
                                                      &ss->send_subtree_count, sendtype, dst,
                                                      comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);
                }
                else
                {
                    /* non-zero root and others */
                    mpi_errno = MPID_Sched_cb2(&calc_send_count_non_root, ss, ((void *)(size_t)mask), s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);

                    /* mask is also the size of this process's subtree */
                    mpi_errno = MPID_Sched_send_defer(((char *)tmp_buf + ss->nbytes*mask),
                                                      &ss->send_subtree_count, MPI_BYTE, dst,
                                                      comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPID_SCHED_BARRIER(s);
                }
                mpi_errno = MPID_Sched_cb(&calc_curr_count, ss, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                MPID_SCHED_BARRIER(s);
            }
            mask >>= 1;
        }

        if ((rank == root) && (root == 0) && (recvbuf != MPI_IN_PLACE)) {
            /* for root=0, put root's data in recvbuf if not MPI_IN_PLACE */
            mpi_errno = MPID_Sched_copy(sendbuf, sendcount, sendtype,
                                        recvbuf, recvcount, recvtype, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            MPID_SCHED_BARRIER(s);
        }
        else if (!(relative_rank % 2) && (recvbuf != MPI_IN_PLACE)) {
            /* for non-zero root and non-leaf nodes, copy from tmp_buf
               into recvbuf */
            mpi_errno = MPID_Sched_copy(tmp_buf, ss->nbytes, MPI_BYTE,
                                        recvbuf, recvcount, recvtype, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            MPID_SCHED_BARRIER(s);
        }

    }
#ifdef MPID_HAS_HETERO
    else { /* communicator is heterogeneous */
        int position;
        MPIU_Assertp(FALSE); /* hetero case not yet implemented */

        if (rank == root) {
            MPIR_Pack_size_impl(sendcount*comm_size, sendtype, &tmp_buf_size);

            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");

          /* calculate the value of nbytes, the number of bytes in packed
             representation that each process receives. We can't
             accurately calculate that from tmp_buf_size because
             MPI_Pack_size returns an upper bound on the amount of memory
             required. (For example, for a single integer, MPICH-1 returns
             pack_size=12.) Therefore, we actually pack some data into
             tmp_buf and see by how much 'position' is incremented. */

            position = 0;
            mpi_errno = MPIR_Pack_impl(sendbuf, 1, sendtype, tmp_buf, tmp_buf_size, &position);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            nbytes = position*sendcount;

            curr_cnt = nbytes*comm_size;

            if (root == 0) {
                if (recvbuf != MPI_IN_PLACE) {
                    position = 0;
                    mpi_errno = MPIR_Pack_impl(sendbuf, sendcount*comm_size, sendtype, tmp_buf,
                                               tmp_buf_size, &position);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
                else {
                    position = nbytes;
                    mpi_errno = MPIR_Pack_impl(((char *) sendbuf + extent*sendcount),
                                               sendcount*(comm_size-1), sendtype, tmp_buf,
                                               tmp_buf_size, &position);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
            }
            else {
                if (recvbuf != MPI_IN_PLACE) {
                    position = 0;
                    mpi_errno = MPIR_Pack_impl(((char *) sendbuf + extent*sendcount*rank),
                                               sendcount*(comm_size-rank), sendtype, tmp_buf,
                                               tmp_buf_size, &position);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
                else {
                    position = nbytes;
                    mpi_errno = MPIR_Pack_impl(((char *) sendbuf + extent*sendcount*(rank+1)),
                                               sendcount*(comm_size-rank-1), sendtype, tmp_buf,
                                               tmp_buf_size, &position);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
                mpi_errno = MPIR_Pack_impl(sendbuf, sendcount*rank, sendtype, tmp_buf,
                                           tmp_buf_size, &position);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            }
        }
        else {
            MPIR_Pack_size_impl(recvcount*(comm_size/2), recvtype, &tmp_buf_size);
            MPIU_CHKLMEM_MALLOC(tmp_buf, void *, tmp_buf_size, mpi_errno, "tmp_buf");

            /* calculate nbytes */
            position = 0;
            mpi_errno = MPIR_Pack_impl(recvbuf, 1, recvtype, tmp_buf, tmp_buf_size, &position);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            nbytes = position*recvcount;

            curr_cnt = 0;
        }

        mask = 0x1;
        while (mask < comm_size) {
            if (relative_rank & mask) {
                src = rank - mask;
                if (src < 0) src += comm_size;

                MPIR_PVAR_INC(iscatter, binomial, recv, tmp_buf_size, MPI_BYTE);
                mpi_errno = MPIC_Recv(tmp_buf, tmp_buf_size, MPI_BYTE, src,
                                         MPIR_SCATTER_TAG, comm_ptr, &status, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    curr_cnt = 0;
                } else
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
                if (dst >= comm_size) dst -= comm_size;

                send_subtree_cnt = curr_cnt - nbytes * mask;
                /* mask is also the size of this process's subtree */
                MPIR_PVAR_INC(iscatter, binomial, send, send_subtree_cnt, MPI_BYTE);
                mpi_errno = MPIC_Send(((char *)tmp_buf + nbytes*mask),
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
        if (recvbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Unpack_impl(tmp_buf, tmp_buf_size, &position, recvbuf,
                                         recvcount, recvtype);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }
    }
#endif /* MPID_HAS_HETERO */


    MPIR_SCHED_CHKPMEM_COMMIT(s);
 fn_exit:
    return mpi_errno;
 fn_fail:
    MPIR_SCHED_CHKPMEM_REAP(s);
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iscatter_tune_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Iscatter_tune_helper_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int is_homogeneous ATTRIBUTE((unused));

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous);

    mpi_errno = MV2_Iscatter_function(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, root, comm_ptr, s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iscatter_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iscatter_intra_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous ATTRIBUTE((unused));
    MPI_Aint recvtype_size, sendtype_size, nbytes;
    
    int rank = comm_ptr->rank;
    int two_level_iscatter = 1;
    int range = 0;
    int range_threshold = 0;
    int range_threshold_intra = 0;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
    comm_size = comm_ptr->local_size;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous); /* we don't handle the hetero case right now */
    if (rank == root) {
        MPID_Datatype_get_size_macro(sendtype, sendtype_size);
        nbytes = sendcount * sendtype_size;
    } else {
        MPID_Datatype_get_size_macro(recvtype, recvtype_size);
        nbytes = recvcount * recvtype_size;
    }

    // Search for some parameters regardless of whether subsequent selected
    // algorithm is 2-level or not
    
    // Search for the corresponding system size inside the tuning table
    while ((range < (mv2_size_iscatter_tuning_table - 1)) &&
           (comm_size > mv2_iscatter_thresholds_table[range].numproc)) {
        range++;
    }
    
    // Search for corresponding inter-leader function
    while ((range_threshold < (mv2_iscatter_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_iscatter_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_iscatter_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    // Search for corresponding intra-node function
    
    // Commenting this for the time being as none of
    // the algorithms are 2-level
    /*
    while ((range_threshold_intra <
            (mv2_iscatter_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_iscatter_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_iscatter_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }
    */

    MV2_Iscatter_function =
        mv2_iscatter_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Iscatter_function;

    MV2_Iscatter_intra_node_function =
        mv2_iscatter_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Iscatter_function;

    /* There are currently no two-level nb-scatter functions hence
       setting to 0 by default */
    two_level_iscatter = 
        mv2_iscatter_thresholds_table[range].is_two_level_iscatter[range_threshold]; 
    if (1 != two_level_iscatter) {
        mpi_errno = MPIR_Iscatter_tune_helper_MV2(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, root, comm_ptr, s);
    }
    else {
        /* Code path should not enter this with the current algorithms*/
    }

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iscatter_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iscatter_MV2(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPID_INTRACOMM) {    
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
        mpi_errno = MPIR_Iscatter_intra_MV2(sendbuf, sendcount, sendtype, recvbuf,
                        recvcount, recvtype, root, comm_ptr, s);
#else
        mpi_errno = MPIR_Iscatter_intra(sendbuf, sendcount, sendtype, recvbuf,
                          recvcount, recvtype, root, comm_ptr, s);
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
    }
    else {
        mpi_errno = MPIR_Iscatter_inter(sendbuf, sendcount, sendtype, recvbuf,
                                       recvcount, recvtype, root, comm_ptr, s);
    }

    return mpi_errno;
}
