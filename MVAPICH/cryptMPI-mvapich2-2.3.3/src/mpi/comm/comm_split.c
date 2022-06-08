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

#include "mpidimpl.h"
#include "mpicomm.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) || defined(CHANNEL_NEMESIS_IB)
#include "coll_shmem.h"
#endif /* defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) || defined(CHANNEL_NEMESIS_IB) */

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

categories:
    - name        : COMMUNICATOR
      description : cvars that control communicator construction and operation

cvars:
    - name        : MPIR_CVAR_COMM_SPLIT_USE_QSORT
      category    : COMMUNICATOR
      type        : boolean
      default     : true
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Use qsort(3) in the implementation of MPI_Comm_split instead of bubble sort.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* -- Begin Profiling Symbol Block for routine MPI_Comm_split */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Comm_split = PMPI_Comm_split
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Comm_split  MPI_Comm_split
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Comm_split as PMPI_Comm_split
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) __attribute__((weak,alias("PMPI_Comm_split")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Comm_split
#define MPI_Comm_split PMPI_Comm_split

/***************************************
 * comm_split using parallel (bitonic) sort,
 * useful for large process counts,
 * currently only good for intracomms.
 ***************************************/

/* Executes an MPI_Comm_split operation using bitonic sort, a pt2pt
 * exchange to find color boundaries and left and right group
 * neighbors, a double inclusive scan to compute new rank and group
 * size, and a recv from ANY_SOURCE with a barrier to return the
 * output group as a chain (left/right/rank/size).
 *
 * Based on "Exascale Algorithms for Generalized MPI_Comm_split",
 * EuroMPI 2011, Adam Moody, Dong H. Ahn, and Bronis R. de Supinkski
 *
 * Also see "A New Vision for Coarray Fortran",
 * The Third Conference on Partitioned Global Address Space
 * Programming Models 2009, John Mellor-Crummey, Laksono Adhianto,
 * Guohua Jin, and William N. Scherer III. */

/* TODO: define these tags for comm_split in src/include/mpiimpl.h
 * along with other tags used for various collectives
 *
 * #define MPIR_ALLTOALLW_TAG            25
 * #define MPIR_SPLIT_A_TAG              26
 * #define MPIR_SPLIT_B_TAG              27
 * #define MPIR_TOPO_A_TAG               28
 * #define MPIR_TOPO_B_TAG               29
 *
 * */
/* TODO: right now we use two tags plus a barrier, this may be overkill
 * and we may be able to get by with one tag and the barrier */
static int tag0 = 0; // MPIR_SPLIT_A_TAG
static int tag1 = 2; // MPIR_SPLIT_B_TAG

#define CKRSIZE (3)
enum ckr_fields {
    CKR_COLOR = 0,
    CKR_KEY   = 1,
    CKR_RANK  = 2,
};

/* compares a (color,key,rank) integer tuple, first by color,
 * then key, then rank */
static int cmp_three_ints(const int a[], const int b[])
{
    /* compare color values first */
    if (a[CKR_COLOR] != b[CKR_COLOR]) {
        if (a[CKR_COLOR] > b[CKR_COLOR]) {
            return 1;
        }
        return -1;
    }

    /* then compare key values */
    if (a[CKR_KEY] != b[CKR_KEY]) {
        if (a[CKR_KEY] > b[CKR_KEY]) {
            return 1;
        }
        return -1;
    }

    /* finally compare ranks */
    if (a[CKR_RANK] != b[CKR_RANK]) {
        if (a[CKR_RANK] > b[CKR_RANK]) {
            return 1;
        }
        return -1;
    }

    /* all three are equal if we make it here */
    return 0;
}

/***************************************
 * Bitonic sort CKR_COLOR/KEY/RANK tuples
 ***************************************/

static int sort_bitonic_merge(
    int value[CKRSIZE],
    int start,
    int num,
    int direction,
    MPID_Comm *comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    int scratch[CKRSIZE];
    if (num > 1) {
        /* get our rank in the communicator */
        int rank = comm_ptr->rank;

        /* determine largest power of two that is smaller than num */
        int count = 1;
        while (count < num) {
            count <<= 1;
        }
        count >>= 1;

        /* divide range into two chunks, execute bitonic half-clean
         * step, then recursively merge each half */
        MPI_Status status[2];
        if (rank < start + count) {
            int dst_rank = rank + count;
            if (dst_rank < start + num) {
              /* exchange data with our partner rank */
              mpi_errno = MPIC_Sendrecv(
                  value,   CKRSIZE, MPI_INT, dst_rank, tag0,
                  scratch, CKRSIZE, MPI_INT, dst_rank, tag0,
                  comm_ptr, status, &errflag
              );
              /* TODO: process error! */

              /* select the appropriate value,
               * depedning on the sort direction */
              int cmp = cmp_three_ints(scratch, value);
              if ((direction && cmp < 0) || (!direction && cmp > 0)) {
                  memcpy(value, scratch, CKRSIZE * sizeof(int));
              }
            }

            /* recursively merge our half */
            sort_bitonic_merge(
                value, start, count, direction, comm_ptr
            );
            /* TODO: process error! */
        } else {
            int dst_rank = rank - count;
            if (dst_rank >= start) {
              mpi_errno = MPIC_Sendrecv(
                  value,   CKRSIZE, MPI_INT, dst_rank, tag0,
                  scratch, CKRSIZE, MPI_INT, dst_rank, tag0,
                  comm_ptr, status, &errflag
              );
              /* TODO: process error! */

              /* select the appropriate value,
               * depedning on the sort direction */
              int cmp = cmp_three_ints(scratch, value);
              if ((direction && cmp > 0) || (!direction && cmp < 0)) {
                  memcpy(value, scratch, CKRSIZE * sizeof(int));
              }
            }

            /* recursively merge our half */
            int new_start = start + count;
            int new_num   = num - count;
            sort_bitonic_merge(
                value, new_start, new_num, direction, comm_ptr
            );
            /* TODO: process error! */
        }
    }

    return mpi_errno;
}

static int sort_bitonic_sort(
    int value[CKRSIZE],
    int start,
    int num,
    int direction,
    MPID_Comm *comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    if (num > 1) {
        /* get our rank in our group */
        int rank = comm_ptr->rank;

        /* recursively divide and sort each half */
        int mid = (num >> 1);
        if (rank < start + mid) {
            sort_bitonic_sort(
                value, start, mid, !direction, comm_ptr
            );
            /* TODO: process error! */
        } else {
            int new_start = start + mid;
            int new_num   = num - mid;
            sort_bitonic_sort(
                value, new_start, new_num, direction, comm_ptr
            );
            /* TODO: process error! */
        }

        /* merge the two sorted halves */
        sort_bitonic_merge(
            value, start, num, direction, comm_ptr
        );
        /* TODO: process error! */
    }

    return mpi_errno;
}

/* globally sort (color,key,rank) items across processes in group,
 * each process provides its tuple as item on input,
 * on output item is overwritten with a new item
 * such that if rank_i < rank_j, item_i < item_j for all i and j */
static int sort_bitonic(int item[CKRSIZE], MPID_Comm *comm_ptr)
{
    /* conduct the bitonic sort on our values */
    int ranks = comm_ptr->local_size;
    int rc = sort_bitonic_sort(item, 0, ranks, 1, comm_ptr);
    /* TODO: process error! */
    return rc;
}

#define CHAINSIZE (4)
enum chain_fields {
    CHAIN_LEFT  = 0, /* address of proc whose rank is one less */
    CHAIN_RIGHT = 1, /* address of proc whose rank is one more */
    CHAIN_RANK  = 2, /* our rank within new group */
    CHAIN_SIZE  = 3, /* size of new group */
};

#define SCANSIZE (3)
enum scan_fields {
    SCAN_FLAG  = 0, /* set flag=1 when we should stop accumulating */
    SCAN_COUNT = 1, /* running count being accumulated */
    SCAN_NEXT  = 2, /* rank of next process to talk to */
};

/* assumes that color/key/rank tuples have been globally sorted
 * across ranks, computes corresponding group information in form
 * of chain (left,right,rank,size) and passes that back to
 * originating rank:
 *   1) determines group boundaries and left and right neighbors
 *      via pt2pt msgs to compare colors
 *   2) executes left-to-right and right-to-left (double) inclusive
 *      segmented scan to compute number of ranks to left and right
 *      sides
 *   3) sends chain info (left/right/rank/size)
 *      back to originating rank via isend/irecv ANY_SOURCE
 *      followed by a barrier */
static int split_sorted(
    const int val[CKRSIZE],
    int *outranks,
    int *outrank,
    int *outleft,
    int *outright,
    MPID_Comm *comm_ptr)
{
    int k;
    MPID_Request *request[4];
    MPI_Status  status[4];
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    /* we will fill in four integer values (left, right, rank, size)
     * representing the chain data structure for the the globally
     * ordered color/key/rank tuple that we hold, which we'll then
     * send back to the rank that contributed our item */
    int send_group_ints[CHAINSIZE];

    /* get our rank, number of ranks, and ranks of processes
     * that are one less (left) and one more (right) than our own */
    int rank  = comm_ptr->rank;
    int ranks = comm_ptr->local_size;

    int left_rank = rank - 1;
    if (left_rank < 0) {
        left_rank = MPI_PROC_NULL;
    }

    int right_rank = rank + 1;
    if (right_rank >= ranks) {
        right_rank = MPI_PROC_NULL;
    }

    /* exchange data with left and right neighbors to find
     * boundaries of group */
    k = 0;
    int left_buf[CKRSIZE];
    int right_buf[CKRSIZE];
    if (left_rank != MPI_PROC_NULL) {
        mpi_errno = MPIC_Isend(
            (void*)val, CKRSIZE, MPI_INT, left_rank,
            tag0, comm_ptr, &request[k], &errflag
        );
        k++;
        /* TODO: process error! */

        mpi_errno = MPIC_Irecv(
            left_buf, CKRSIZE, MPI_INT, left_rank,
            tag0, comm_ptr, &request[k]
        );
        k++;
        /* TODO: process error! */
    }
    if (right_rank != MPI_PROC_NULL) {
        mpi_errno = MPIC_Isend(
            (void*)val, CKRSIZE, MPI_INT, right_rank,
            tag0, comm_ptr, &request[k], &errflag
        );
        k++;
        /* TODO: process error! */

        mpi_errno = MPIC_Irecv(
            right_buf, CKRSIZE, MPI_INT, right_rank,
            tag0, comm_ptr, &request[k]
        );
        k++;
        /* TODO: process error! */
    }
    if (k > 0) {
        mpi_errno = MPIC_Waitall(k, request, status, &errflag);
        /* TODO: process error! */
    }

    /* if we have a left neighbor, and if his color value matches ours,
     * then our element is part of his group, otherwise we are the
     * first rank of a new group */
    int first_in_group = 0;
    if (left_rank != MPI_PROC_NULL &&
        left_buf[CKR_COLOR] == val[CKR_COLOR])
    {
        /* record the rank of the item from our left neighbor */
        send_group_ints[CHAIN_LEFT] = left_buf[CKR_RANK];
    } else {
        first_in_group = 1;
        send_group_ints[CHAIN_LEFT] = MPI_PROC_NULL;
    }

    /* if we have a right neighbor, and if his color value matches ours,
     * then our element is part of his group, otherwise we are the
     * last rank of our group */
    int last_in_group = 0;
    if (right_rank != MPI_PROC_NULL &&
        right_buf[CKR_COLOR] == val[CKR_COLOR])
    {
        /* record the rank of the item from our right neighbor */
        send_group_ints[CHAIN_RIGHT] = right_buf[CKR_RANK];
    } else {
        last_in_group = 1;
        send_group_ints[CHAIN_RIGHT] = MPI_PROC_NULL;
    }

    /* prepare buffers for our scan operations: flag, count, next */
    int send_left_ints[SCANSIZE]  = {0,1,MPI_PROC_NULL};
    int send_right_ints[SCANSIZE] = {0,1,MPI_PROC_NULL};
    int recv_left_ints[SCANSIZE]  = {0,0,MPI_PROC_NULL};
    int recv_right_ints[SCANSIZE] = {0,0,MPI_PROC_NULL};
    if (first_in_group) {
        left_rank = MPI_PROC_NULL;
        send_right_ints[SCAN_FLAG] = 1;
    }
    if (last_in_group) {
        right_rank = MPI_PROC_NULL;
        send_left_ints[SCAN_FLAG] = 1;
    }

    /* execute inclusive scan in both directions to count number of
     * ranks in our group to our left and right sides */
    while (left_rank != MPI_PROC_NULL || right_rank != MPI_PROC_NULL) {
        /* select our left and right partners for this iteration */
        k = 0;

        /* send and receive data with left partner */
        if (left_rank != MPI_PROC_NULL) {
            mpi_errno = MPIC_Irecv(
                recv_left_ints, SCANSIZE, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* send the rank of our right neighbor to our left,
             * since it will be his right neighbor in the next step */
            send_left_ints[SCAN_NEXT] = right_rank;
            mpi_errno = MPIC_Isend(
                send_left_ints, SCANSIZE, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */
        }

        /* send and receive data with right partner */
        if (right_rank != MPI_PROC_NULL) {
            mpi_errno = MPIC_Irecv(
                recv_right_ints, SCANSIZE, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* send the rank of our left neighbor to our right,
             * since it will be his left neighbor in the next step */
            send_right_ints[SCAN_NEXT] = left_rank;
            mpi_errno = MPIC_Isend(
                send_right_ints, SCANSIZE, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */
        }

        /* wait for communication to finsih */
        if (k > 0) {
            mpi_errno = MPIC_Waitall(k, request, status, &errflag);
            /* TODO: process error! */
        }

        /* reduce data from left partner */
        if (left_rank != MPI_PROC_NULL) {
            /* continue accumulating the count in our right-going data
             * if our flag has not already been set */
            if (send_right_ints[SCAN_FLAG] != 1) {
                send_right_ints[SCAN_FLAG]   = recv_left_ints[SCAN_FLAG];
                send_right_ints[SCAN_COUNT] += recv_left_ints[SCAN_COUNT];
            }

            /* get the next rank on our left */
            left_rank = recv_left_ints[SCAN_NEXT];
        }

        /* reduce data from right partner */
        if (right_rank != MPI_PROC_NULL) {
            /* continue accumulating the count in our left-going data
             * if our flag has not already been set */
            if (send_left_ints[SCAN_FLAG] != 1) {
                send_left_ints[SCAN_FLAG]   = recv_right_ints[SCAN_FLAG];
                send_left_ints[SCAN_COUNT] += recv_right_ints[SCAN_COUNT];
            }

            /* get the next rank on our right */
            right_rank = recv_right_ints[SCAN_NEXT];
        }
    }

    /* Now we can set our rank and the number of ranks in our group.
     * At this point, our right-going count is the number of ranks to
     * left including ourself, and the left-going count is the number
     * of ranks to right including ourself.
     * Our rank is the number of ranks to our left (right-going count
     * minus 1), and the group size is the sum of right-going and
     * left-going counts minus 1 so we don't double counts ourself. */
    send_group_ints[CHAIN_RANK] = send_right_ints[SCAN_COUNT] - 1;
    send_group_ints[CHAIN_SIZE] = send_right_ints[SCAN_COUNT] + send_left_ints[SCAN_COUNT] - 1;

    /* TODO: note we can avoid the any_source recv using
     * another sort */
    /* send group info back to originating rank,
     * receive our own from someone else
     * (don't know who so use an ANY_SOURCE recv) */
    int recv_group_ints[CHAINSIZE];
    mpi_errno = MPIC_Isend(
        send_group_ints, CHAINSIZE, MPI_INT, val[CKR_RANK],
        tag1, comm_ptr, &request[0], &errflag
    );
    /* TODO: process error! */
    mpi_errno = MPIC_Irecv(
        recv_group_ints, CHAINSIZE, MPI_INT, MPI_ANY_SOURCE,
        tag1, comm_ptr, &request[1]
    );
    /* TODO: process error! */
    mpi_errno = MPIC_Waitall(2, request, status, &errflag);
    /* TODO: process error! */

    /* execute barrier to ensure that everyone is done with
     * their above ANY_SOURCE recv */
    mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
    /* TODO: process error! */

    /* fill in return parameters */
    *outleft  = recv_group_ints[CHAIN_LEFT];
    *outright = recv_group_ints[CHAIN_RIGHT];
    *outrank  = recv_group_ints[CHAIN_RANK];
    *outranks = recv_group_ints[CHAIN_SIZE];

    return mpi_errno;
}

/* issues an allgather operation over processes in specified group */
#undef FUNCNAME
#define FUNCNAME chain_allgather_int
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int chain_allgather_int(
    int sendint,
    int recvbuf[],
    int ranks,
    int rank,
    int left,
    int right,
    MPID_Comm *comm_ptr)
{
    int left_rank, right_rank, next_left, next_right, count;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = FALSE;

    /* copy our own data into the receive buffer */
    recvbuf[rank] = sendint;

    /* execute the allgather operation */
    MPID_Request *request[8];
    MPI_Status status[8];
    left_rank  = left;
    right_rank = right;
    count = 1;
    while (left_rank != MPI_PROC_NULL || right_rank != MPI_PROC_NULL) {
        int k = 0;

        /* if we have a left partner, send him all data we know about
         * from on rank on to the right */
        if (left_rank != MPI_PROC_NULL) {
            /* receive rank of next left neighbor from current left neighbor */
            mpi_errno = MPIC_Irecv(
                &next_left, 1, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* determine offset in receive buffer and incoming count */
            int left_start = rank + 1 - 2 * count;
            int left_count = count;
            if (left_start < 0) {
                left_start = 0;
                left_count = rank + 1 - count;
            }

            /* issue receive for data from left partner */
            mpi_errno = MPIC_Irecv(
                recvbuf + left_start, left_count, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* send the rank of our right neighbor */
            mpi_errno = MPIC_Isend(
                &right_rank, 1, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */

            /* compute the number of elements we'll be sending left */
            int left_send_count = count;
            if (rank + left_send_count > ranks) {
                left_send_count = ranks - rank;
            }

            /* send our data to our left neighbor */
            mpi_errno = MPIC_Isend(
                recvbuf + rank, left_send_count, MPI_INT, left_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */
        }

        /* if we have a right partner, send him all data we know about
         * from on rank on to the left */
        if (right_rank != MPI_PROC_NULL) {
            /* receive rank of next right neighbor from current right neighbor */
            mpi_errno = MPIC_Irecv(
                &next_right, 1, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* determine offset in receive buffer and incoming count */
            int right_start = rank + count;
            int right_count = count;
            if (right_start + count > ranks) {
                right_count = ranks - right_start;
            }

            /* issue receive for data from right partner */
            mpi_errno = MPIC_Irecv(
                recvbuf + right_start, right_count, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k]
            );
            k++;
            /* TODO: process error! */

            /* send the rank of our left neighbor to our right neighbor */
            mpi_errno = MPIC_Isend(
                &left_rank, 1, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */

            /* compute the number of elements we'll be sending right */
            int right_send_start = rank + 1 - count;
            int right_send_count = count;
            if (right_send_start < 0) {
                right_send_start = 0;
                right_send_count = rank + 1;
            }

            /* send the data */
            mpi_errno = MPIC_Isend(
                recvbuf + right_send_start, right_send_count, MPI_INT, right_rank,
                tag0, comm_ptr, &request[k], &errflag
            );
            k++;
            /* TODO: process error! */
        }

        /* wait for communication to complete */
        if (k > 0) {
            mpi_errno = MPIC_Waitall(k, request, status, &errflag);
            /* TODO: process error! */
        }

        /* get next rank to our left */
        if (left_rank != MPI_PROC_NULL) {
            left_rank = next_left;
        }

        /* get next rank to our right */
        if (right_rank != MPI_PROC_NULL) {
            right_rank = next_right;
        }

        /* go on to next iteration */
        count <<= 1;
    }

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Comm_split_intra_bitonic
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_split_intra_bitonic(MPID_Comm *comm_ptr, int color, int key, MPID_Comm **newcomm_ptr)
{
    int i;
    int mpi_errno = MPI_SUCCESS;
    MPIU_CHKLMEM_DECL(1);

    int rank = comm_ptr->rank;
	
    /* TODO: for small groups, faster to do an allgather and
     * local sort */

    /* prepare item for sorting, tuple of (color,key,rank) */
    int item[CKRSIZE];
    item[CKR_COLOR] = color;
    item[CKR_KEY]   = key;
    item[CKR_RANK]  = rank;

    /* sort our values using bitonic sort algorithm -- 
     * O(log^2 N) communication */
    sort_bitonic(item, comm_ptr);
    /* TODO: process error! */

    /* now split our sorted values by comparing our value with our
     * left and right neighbors to determine group boundaries --
     * O(log N) communication */
    int newranks, newrank, newleft, newright;
    split_sorted(
        item, &newranks, &newrank, &newleft, &newright, comm_ptr
    );
    /* TODO: process error! */

    /* TRUE iff *newcomm should be populated */
    int in_newcomm = (color != MPI_UNDEFINED && newranks > 0);

    /* Collectively create a new context id.  The same context id will
       be used by each (disjoint) collections of processes.  The
       processes whose color is MPI_UNDEFINED will not influence the
       resulting context id (by passing ignore_id==TRUE). */
    /* In the multi-threaded case, MPIR_Get_contextid assumes that the
       calling routine already holds the single criticial section */
    MPIU_Context_id_t new_context_id;
    mpi_errno = MPIR_Get_contextid_sparse(comm_ptr, &new_context_id, !in_newcomm);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIU_Assert(new_context_id != 0);

    *newcomm_ptr = NULL;

    /* Now, create the new communicator structure if necessary */
    if (in_newcomm) {
        mpi_errno = MPIR_Comm_create( newcomm_ptr );
        if (mpi_errno) goto fn_fail;
        
        /* set INTRA Communicator fields */
        (*newcomm_ptr)->recvcontext_id = new_context_id;
        (*newcomm_ptr)->rank           = newrank;
        (*newcomm_ptr)->local_size     = newranks;
        (*newcomm_ptr)->comm_kind      = comm_ptr->comm_kind;
        (*newcomm_ptr)->context_id     = (*newcomm_ptr)->recvcontext_id;
        (*newcomm_ptr)->remote_size    = newranks;
        MPIDI_VCRT_Create( newranks, &(*newcomm_ptr)->dev.vcrt );
        MPIDI_VCRT_Get_ptr( (*newcomm_ptr)->dev.vcrt, &(*newcomm_ptr)->dev.vcr );
        
        /* Allocate memory to hold list of rank ids */
        int* members = NULL;
        MPIU_CHKLMEM_MALLOC(members,int*,newranks*sizeof(int),mpi_errno,
            "members_table");

        /* gather members of group -- O(N) communication,
         * but if max group size is small, N is small */
        chain_allgather_int(
            rank, members,
            newranks, newrank, newleft, newright, comm_ptr
        );
        /* TODO: process error! */

        /* duplicate each vcr in new comm */
        for (i=0; i < newranks; i++) {
            int orig_rank = members[i];
            MPIDI_VCR_Dup(comm_ptr->dev.vcrt->vcr_table[orig_rank],
                            &(*newcomm_ptr)->dev.vcrt->vcr_table[i]);
        }

        /* Inherit the error handler (if any) */
        MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
        (*newcomm_ptr)->errhandler = comm_ptr->errhandler;
        if (comm_ptr->errhandler) {
            MPIR_Errhandler_add_ref( comm_ptr->errhandler );
        }
        MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);

        mpi_errno = MPIR_Comm_commit(*newcomm_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    
 fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

/***************************************
 * comm_split using allgather and local sort
 ***************************************/

typedef struct splittype {
    int color, key;
} splittype;

/* Same as splittype but with an additional field to stabilize the qsort.  We
 * could just use one combined type, but using separate types simplifies the
 * allgather step. */
typedef struct sorttype {
    int color, key;
    int orig_idx;
} sorttype;

#if defined(HAVE_QSORT)
static int sorttype_compare(const void *v1, const void *v2) {
    const sorttype *s1 = v1;
    const sorttype *s2 = v2;

    if (s1->key > s2->key)
        return 1;
    if (s1->key < s2->key)
        return -1;

    /* (s1->key == s2->key), maintain original order */
    if (s1->orig_idx > s2->orig_idx)
        return 1;
    else if (s1->orig_idx < s2->orig_idx)
        return -1;

    /* --BEGIN ERROR HANDLING-- */
    return 0; /* should never happen */
    /* --END ERROR HANDLING-- */
}
#endif

/* Sort the entries in keytable into increasing order by key.  A stable
   sort should be used incase the key values are not unique. */
static void MPIU_Sort_inttable( sorttype *keytable, int size )
{
    sorttype tmp;
    int i, j;

#if defined(HAVE_QSORT)
    /* temporary switch for profiling performance differences */
    if (MPIR_CVAR_COMM_SPLIT_USE_QSORT)
    {
        /* qsort isn't a stable sort, so we have to enforce stability by keeping
         * track of the original indices */
        for (i = 0; i < size; ++i)
            keytable[i].orig_idx = i;
        qsort(keytable, size, sizeof(sorttype), &sorttype_compare);
    }
    else
#endif
    {
        /* --BEGIN USEREXTENSION-- */
        /* fall through to insertion sort if qsort is unavailable/disabled */
        for (i = 1; i < size; ++i) {
            tmp = keytable[i];
            j = i - 1;
            while (1) {
                if (keytable[j].key > tmp.key) {
                    keytable[j+1] = keytable[j];
                    j = j - 1;
                    if (j < 0)
                        break;
                }
                else {
                    break;
                }
            }
            keytable[j+1] = tmp;
        }
        /* --END USEREXTENSION-- */
    }
}

#undef FUNCNAME
#define FUNCNAME MPIR_Comm_split_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_split_allgather(MPID_Comm *comm_ptr, int color, int key, MPID_Comm **newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *local_comm_ptr;
    splittype *table, *remotetable=0;
    sorttype *keytable, *remotekeytable=0;
    int rank, size, remote_size, i, new_size, new_remote_size,
	first_entry = 0, first_remote_entry = 0, *last_ptr;
    int in_newcomm; /* TRUE iff *newcomm should be populated */
    MPIU_Context_id_t   new_context_id, remote_context_id;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPIR_Comm_map_t *mapper;
    MPIU_CHKLMEM_DECL(4);

    rank        = comm_ptr->rank;
    size        = comm_ptr->local_size;
    remote_size = comm_ptr->remote_size;
	
    /* Step 1: Find out what color and keys all of the processes have */
    MPIU_CHKLMEM_MALLOC(table,splittype*,size*sizeof(splittype),mpi_errno,
			"table");
    table[rank].color = color;
    table[rank].key   = key;

    /* Get the communicator to use in collectives on the local group of 
       processes */
    if (comm_ptr->comm_kind == MPID_INTERCOMM) {
	if (!comm_ptr->local_comm) {
	    MPIR_Setup_intercomm_localcomm( comm_ptr );
	}
	local_comm_ptr = comm_ptr->local_comm;
    }
    else {
	local_comm_ptr = comm_ptr;
    }
    /* Gather information on the local group of processes */
    mpi_errno = MPIR_Allgather_impl( MPI_IN_PLACE, 2, MPI_INT, table, 2, MPI_INT, local_comm_ptr, &errflag );
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    /* Step 2: How many processes have our same color? */
    new_size = 0;
    if (color != MPI_UNDEFINED) {
	/* Also replace the color value with the index of the *next* value
	   in this set.  The integer first_entry is the index of the 
	   first element */
	last_ptr = &first_entry;
	for (i=0; i<size; i++) {
	    /* Replace color with the index in table of the next item
	       of the same color.  We use this to efficiently populate 
	       the keyval table */
	    if (table[i].color == color) {
		new_size++;
		*last_ptr = i;
		last_ptr  = &table[i].color;
	    }
	}
    }
    /* We don't need to set the last value to -1 because we loop through
       the list for only the known size of the group */

    /* If we're an intercomm, we need to do the same thing for the remote
       table, as we need to know the size of the remote group of the
       same color before deciding to create the communicator */
    if (comm_ptr->comm_kind == MPID_INTERCOMM) {
	splittype mypair;
	/* For the remote group, the situation is more complicated.
	   We need to find the size of our "partner" group in the
	   remote comm.  The easiest way (in terms of code) is for
	   every process to essentially repeat the operation for the
	   local group - perform an (intercommunicator) all gather
	   of the color and rank information for the remote group.
	*/
	MPIU_CHKLMEM_MALLOC(remotetable,splittype*,
			    remote_size*sizeof(splittype),mpi_errno,
			    "remotetable");
	/* This is an intercommunicator allgather */
	
	/* We must use a local splittype because we've already modified the
	   entries in table to indicate the location of the next rank of the
	   same color */
	mypair.color = color;
	mypair.key   = key;
	mpi_errno = MPIR_Allgather_impl( &mypair, 2, MPI_INT, remotetable, 2, MPI_INT,
                                         comm_ptr, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
        
	/* Each process can now match its color with the entries in the table */
	new_remote_size = 0;
	last_ptr = &first_remote_entry;
	for (i=0; i<remote_size; i++) {
	    /* Replace color with the index in table of the next item
	       of the same color.  We use this to efficiently populate 
	       the keyval table */
	    if (remotetable[i].color == color) {
		new_remote_size++;
		*last_ptr = i;
		last_ptr  = &remotetable[i].color;
	    }
	}
	/* Note that it might find that there a now processes in the remote
	   group with the same color.  In that case, COMM_SPLIT will
	   return a null communicator */
    }
    else {
	/* Set the size of the remote group to the size of our group.
	   This simplifies the test below for intercomms with an empty remote
	   group (must create comm_null) */
	new_remote_size = new_size;
    }

    in_newcomm = (color != MPI_UNDEFINED && new_remote_size > 0);

    /* Step 3: Create the communicator */
    /* Collectively create a new context id.  The same context id will
       be used by each (disjoint) collections of processes.  The
       processes whose color is MPI_UNDEFINED will not influence the
       resulting context id (by passing ignore_id==TRUE). */
    /* In the multi-threaded case, MPIR_Get_contextid_sparse assumes that the
       calling routine already holds the single criticial section */
    mpi_errno = MPIR_Get_contextid_sparse(local_comm_ptr, &new_context_id, !in_newcomm);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIU_Assert(new_context_id != 0);

    /* In the intercomm case, we need to exchange the context ids */
    if (comm_ptr->comm_kind == MPID_INTERCOMM) {
	if (comm_ptr->rank == 0) {
	    mpi_errno = MPIC_Sendrecv( &new_context_id, 1, MPIU_CONTEXT_ID_T_DATATYPE, 0, 0,
				       &remote_context_id, 1, MPIU_CONTEXT_ID_T_DATATYPE, 
				       0, 0, comm_ptr, MPI_STATUS_IGNORE, &errflag );
	    if (mpi_errno) { MPIR_ERR_POP( mpi_errno ); }
	    mpi_errno = MPIR_Bcast_impl( &remote_context_id, 1, MPIU_CONTEXT_ID_T_DATATYPE, 0, local_comm_ptr, &errflag );
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
	}
	else {
	    /* Broadcast to the other members of the local group */
	    mpi_errno = MPIR_Bcast_impl( &remote_context_id, 1, MPIU_CONTEXT_ID_T_DATATYPE, 0, local_comm_ptr, &errflag );
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
	}
    }

    *newcomm_ptr = NULL;

    /* Now, create the new communicator structure if necessary */
    if (in_newcomm) {
    
	mpi_errno = MPIR_Comm_create( newcomm_ptr );
	if (mpi_errno) goto fn_fail;

	(*newcomm_ptr)->recvcontext_id = new_context_id;
	(*newcomm_ptr)->local_size	    = new_size;
	(*newcomm_ptr)->comm_kind	    = comm_ptr->comm_kind;
	/* Other fields depend on whether this is an intercomm or intracomm */

	/* Step 4: Order the processes by their key values.  Sort the
	   list that is stored in table.  To simplify the sort, we 
	   extract the table into a smaller array and sort that.
	   Also, store in the "color" entry the rank in the input communicator
	   of the entry. */
	MPIU_CHKLMEM_MALLOC(keytable,sorttype*,new_size*sizeof(sorttype),
			    mpi_errno,"keytable");
	for (i=0; i<new_size; i++) {
	    keytable[i].key   = table[first_entry].key;
	    keytable[i].color = first_entry;
	    first_entry	      = table[first_entry].color;
	}

	/* sort key table.  The "color" entry is the rank of the corresponding
	   process in the input communicator */
	MPIU_Sort_inttable( keytable, new_size );

	if (comm_ptr->comm_kind == MPID_INTERCOMM) {
	    MPIU_CHKLMEM_MALLOC(remotekeytable,sorttype*,
				new_remote_size*sizeof(sorttype),
				mpi_errno,"remote keytable");
	    for (i=0; i<new_remote_size; i++) {
		remotekeytable[i].key   = remotetable[first_remote_entry].key;
		remotekeytable[i].color = first_remote_entry;
		first_remote_entry	= remotetable[first_remote_entry].color;
	    }

	    /* sort key table.  The "color" entry is the rank of the
	       corresponding process in the input communicator */
	    MPIU_Sort_inttable( remotekeytable, new_remote_size );

            MPIR_Comm_map_irregular(*newcomm_ptr, comm_ptr, NULL,
                                    new_size, MPIR_COMM_MAP_DIR_L2L,
                                    &mapper);

            for (i = 0; i < new_size; i++) {
                mapper->src_mapping[i] = keytable[i].color;
		if (keytable[i].color == comm_ptr->rank)
		    (*newcomm_ptr)->rank = i;
            }

	    /* For the remote group, the situation is more complicated.
	       We need to find the size of our "partner" group in the
	       remote comm.  The easiest way (in terms of code) is for
	       every process to essentially repeat the operation for the
	       local group - perform an (intercommunicator) all gather
	       of the color and rank information for the remote group.
	     */
	    /* We apply the same sorting algorithm to the entries that we've
	       found to get the correct order of the entries.

	       Note that if new_remote_size is 0 (no matching processes with
	       the same color in the remote group), then MPI_COMM_SPLIT
	       is required to return MPI_COMM_NULL instead of an intercomm 
	       with an empty remote group. */

            MPIR_Comm_map_irregular(*newcomm_ptr, comm_ptr, NULL,
                                    new_remote_size,
                                    MPIR_COMM_MAP_DIR_R2R, &mapper);

            for (i = 0; i < new_remote_size; i++)
                mapper->src_mapping[i] = remotekeytable[i].color;

	    (*newcomm_ptr)->context_id     = remote_context_id;
	    (*newcomm_ptr)->remote_size    = new_remote_size;
	    (*newcomm_ptr)->local_comm     = 0;
	    (*newcomm_ptr)->is_low_group   = comm_ptr->is_low_group;

	}
	else {
	    /* INTRA Communicator */
	    (*newcomm_ptr)->context_id     = (*newcomm_ptr)->recvcontext_id;
	    (*newcomm_ptr)->remote_size    = new_size;

            MPIR_Comm_map_irregular(*newcomm_ptr, comm_ptr, NULL,
                                    new_size, MPIR_COMM_MAP_DIR_L2L,
                                    &mapper);

            for (i = 0; i < new_size; i++) {
                mapper->src_mapping[i] = keytable[i].color;
		if (keytable[i].color == comm_ptr->rank)
		    (*newcomm_ptr)->rank = i;
            }
	}

	/* Inherit the error handler (if any) */
        MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));
	(*newcomm_ptr)->errhandler = comm_ptr->errhandler;
	if (comm_ptr->errhandler) {
	    MPIR_Errhandler_add_ref( comm_ptr->errhandler );
	}
        MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));

        mpi_errno = MPIR_Comm_commit(*newcomm_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    
 fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

/***************************************
 * comm_split driver to pick appropriate algorithm
 ***************************************/

#undef FUNCNAME
#define FUNCNAME MPIR_Comm_split_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_split_impl(MPID_Comm *comm_ptr, int color, int key, MPID_Comm **newcomm_ptr)
{
    int rc;
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) || defined(CHANNEL_NEMESIS_IB)
    if (comm_ptr->comm_kind == MPID_INTERCOMM) {
        rc = MPIR_Comm_split_allgather(comm_ptr, color, key, newcomm_ptr);

        /* TODO: bitonic sort algorithm for intercomm split:
         *   each process stores local group as chain
         *   and stores address of remote process having same local rank (if any)
         *   0) create single sort chain by appending high group to low group
         *   1) create groupid/color/key/rank tuple
         *   2) sort by groupid, then color, then key, then rank,
         *      pt2pt with left/right to split into groups with same (groupid,color),
         *      scan within group to define chain for new local group
         *      send back to original rank, paying attention to which group its in
         *   3) create color/newrank/groupid/rank tuple (newrank = rank in new local group)
         *   4) sort by color, then newrank, then groupid
         *      pt2pt with left/right to split into groups with same (color,newrank)
         *      record address of remote process having same color and newrank (if any)
         *      send back to original rank paying attention to its group */
    } else {
        /* TODO: for small input comms, better to call allgather/qsort vs bitonic */
        int ranks = comm_ptr->local_size;
        if (mv2_use_bitonic_comm_split && (ranks > mv2_bitonic_comm_split_threshold)) {
            rc = MPIR_Comm_split_intra_bitonic(comm_ptr, color, key, newcomm_ptr);
        } else {
            rc = MPIR_Comm_split_allgather(comm_ptr, color, key, newcomm_ptr);
        }
    }
#else
    rc = MPIR_Comm_split_allgather(comm_ptr, color, key, newcomm_ptr);
#endif /* defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) || defined(CHANNEL_NEMESIS_IB) */
    return rc;
}

#endif /* #ifndef MPICH_MPI_FROM_PMPI */




#undef FUNCNAME
#define FUNCNAME MPI_Comm_split
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@

MPI_Comm_split - Creates new communicators based on colors and keys

Input Parameters:
+ comm - communicator (handle) 
. color - control of subset assignment (nonnegative integer).  Processes 
  with the same color are in the same new communicator 
- key - control of rank assignment (integer)

Output Parameters:
. newcomm - new communicator (handle) 

Notes:
  The 'color' must be non-negative or 'MPI_UNDEFINED'.

.N ThreadSafe

.N Fortran

Algorithm:
.vb
  1. Use MPI_Allgather to get the color and key from each process
  2. Count the number of processes with the same color; create a 
     communicator with that many processes.  If this process has
     'MPI_UNDEFINED' as the color, create a process with a single member.
  3. Use key to order the ranks
.ve
 
.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_EXHAUSTED

.seealso: MPI_Comm_free
@*/
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL, *newcomm_ptr;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_COMM_SPLIT);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_FUNC_ENTER(MPID_STATE_MPI_COMM_SPLIT);

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
    
    /* Get handles to MPI objects. */
    MPID_Comm_get_ptr( comm, comm_ptr );
    
    /* Validate parameters and objects (post conversion) */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            /* Validate comm_ptr */
            MPID_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
	    /* If comm_ptr is not valid, it will be reset to null */
            if (mpi_errno) goto fn_fail;
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */
    
    mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, &newcomm_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    if (newcomm_ptr)
        MPID_OBJ_PUBLISH_HANDLE(*newcomm, newcomm_ptr->handle);
    else
        *newcomm = MPI_COMM_NULL;

    /* ... end of body of routine ... */

  fn_exit:
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPI_COMM_SPLIT);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;
    
  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, 
	    "**mpi_comm_split",
	    "**mpi_comm_split %C %d %d %p", comm, color, key, newcomm);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

