/* -*- Mode: C; c-basic-offset:4 ; -*- */
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
#include <unistd.h>
#include "coll_shmem.h"
#include "red_scat_tuning.h"
#include <unistd.h>

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_noncomm);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_basic);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_rec_halving);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_pairwise);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_non_comm);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_ring);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_reduce_scatter_ring_2lvl);


MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_noncomm);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_basic);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_rec_halving);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_pairwise);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_non_comm);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_noncomm_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_basic_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_rec_halving_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_pairwise_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_non_comm_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_noncomm_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_basic_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_rec_halving_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_pairwise_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_non_comm_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_noncomm_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_basic_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_rec_halving_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_pairwise_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_non_comm_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_noncomm_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_basic_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_rec_halving_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_pairwise_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_non_comm_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_reduce_scatter_count_recv);
                               
int (*MV2_Red_scat_function)(const void *sendbuf,
                             void *recvbuf,
                             const int *recvcnts,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

/* Implements the "mirror permutation" of "bits" bits of an integer "x".

   positions 76543210, bits==3 yields 76543012.

   This function could/should be moved to a common utility location for use in
   other collectives as well. */
ATTRIBUTE((const))				/* tells the compiler that this func only depends on its args
								   and may be optimized much more aggressively, similar to "pure" */
static inline int mirror_permutation(unsigned int x, int bits)
{
	/* a mask for the high order bits that should be copied as-is */
	int high_mask = ~((0x1 << bits) - 1);
	int retval = x & high_mask;
	int i;

	for (i = 0; i < bits; ++i) {
		unsigned int bitval = (x & (0x1 << i)) >> i;	/* 0x1 or 0x0 */
		retval |= bitval << ((bits - i) - 1);
	}

	return retval;
}

/* FIXME should we be checking the op_errno here? */
#ifdef HAVE_CXX_BINDING
/* NOTE: assumes 'uop' is the operator function pointer and
   that 'is_cxx_uop' is is a boolean indicating the obvious */
#define call_uop(in_, inout_, count_, datatype_)                                     \
do {                                                                                 \
    if (is_cxx_uop) {                                                                \
        (*MPIR_Process.cxx_call_op_fn)((in_), (inout_), (count_), (datatype_), uop); \
    }                                                                                \
    else {                                                                           \
        (*uop)((in_), (inout_), &(count_), &(datatype_));                            \
    }                                                                                \
} while (0)

#else
#define call_uop(in_, inout_, count_, datatype_)      \
    (*uop)((in_), (inout_), &(count_), &(datatype_))
#endif

/* Implements the reduce-scatter butterfly algorithm described in J. L. Traff's
 * "An Improved Algorithm for (Non-commutative) Reduce-Scatter with an Application"
 * from EuroPVM/MPI 2005.  This function currently only implements support for
 * the power-of-2, block-regular case (all receive counts are equal). */
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_noncomm_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Reduce_scatter_noncomm_MV2(const void *sendbuf,
										   void *recvbuf,
										   const int *recvcnts,
										   MPI_Datatype datatype,
										   MPI_Op op,
										   MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
	MPIR_TIMER_START(coll,reduce_scatter,noncomm);
	int mpi_errno = MPI_SUCCESS;
	int mpi_errno_ret = MPI_SUCCESS;
	int comm_size = comm_ptr->local_size;
	int rank = comm_ptr->rank;
	int pof2;
	int log2_comm_size;
	int i, k;
	int recv_offset, send_offset;
	int block_size, total_count, size;
	MPI_Aint true_extent, true_lb;
	int buf0_was_inout;
	void *tmp_buf0;
	void *tmp_buf1;
	void *result_ptr;
	MPI_User_function *uop;
	MPID_Op *op_ptr;
#ifdef HAVE_CXX_BINDING
	int is_cxx_uop = 0;
#endif
	MPIU_CHKLMEM_DECL(3);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_scatter_noncomm, 1);

	MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

	if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
		/* get the function by indexing into the op table */
		uop = MPIR_Op_table[op % 16 - 1];
	} else {
		MPID_Op_get_ptr(op, op_ptr);
#ifdef HAVE_CXX_BINDING
		if (op_ptr->language == MPID_LANG_CXX) {
			uop = (MPI_User_function *) op_ptr->function.c_function;
			is_cxx_uop = 1;
		} else
#endif
		if (op_ptr->language == MPID_LANG_C)
			uop = (MPI_User_function *) op_ptr->function.c_function;
		else
			uop = (MPI_User_function *) op_ptr->function.f77_function;
	}

	pof2 = 1;
	log2_comm_size = 0;
	while (pof2 < comm_size) {
		pof2 <<= 1;
		++log2_comm_size;
	}

	/* begin error checking */
	MPIU_Assert(pof2 == comm_size);	/* FIXME this version only works for power of 2 procs */

	for (i = 0; i < (comm_size - 1); ++i) {
		MPIU_Assert(recvcnts[i] == recvcnts[i + 1]);
	}
	/* end error checking */

	/* size of a block (count of datatype per block, NOT bytes per block) */
	block_size = recvcnts[0];
	total_count = block_size * comm_size;

	MPIU_CHKLMEM_MALLOC(tmp_buf0, void *, true_extent * total_count, mpi_errno,
						"tmp_buf0");
	MPIU_CHKLMEM_MALLOC(tmp_buf1, void *, true_extent * total_count, mpi_errno,
						"tmp_buf1");
	/* adjust for potential negative lower bound in datatype */
	tmp_buf0 = (void *) ((char *) tmp_buf0 - true_lb);
	tmp_buf1 = (void *) ((char *) tmp_buf1 - true_lb);

	/* Copy our send data to tmp_buf0.  We do this one block at a time and
	   permute the blocks as we go according to the mirror permutation. */
	for (i = 0; i < comm_size; ++i) {
		mpi_errno =
			MPIR_Localcopy((char *) (sendbuf ==
									 MPI_IN_PLACE ? recvbuf : sendbuf) +
						   (i * true_extent * block_size), block_size, datatype,
						   (char *) tmp_buf0 +
						   (mirror_permutation(i, log2_comm_size) *
							true_extent * block_size), block_size, datatype);
		if (mpi_errno)
			MPIR_ERR_POP(mpi_errno);
	}
	buf0_was_inout = 1;

	send_offset = 0;
	recv_offset = 0;
	size = total_count;
	for (k = 0; k < log2_comm_size; ++k) {
		/* use a double-buffering scheme to avoid local copies */
		char *incoming_data = (buf0_was_inout ? tmp_buf1 : tmp_buf0);
		char *outgoing_data = (buf0_was_inout ? tmp_buf0 : tmp_buf1);
		int peer = rank ^ (0x1 << k);
		size /= 2;

		if (rank > peer) {
			/* we have the higher rank: send top half, recv bottom half */
			recv_offset += size;
		} else {
			/* we have the lower rank: recv top half, send bottom half */
			send_offset += size;
		}

		MPIR_PVAR_INC(reduce_scatter, noncomm, send, size, datatype);
		MPIR_PVAR_INC(reduce_scatter, noncomm, recv, size, datatype);
		mpi_errno = MPIC_Sendrecv(outgoing_data + send_offset * true_extent,
									 size, datatype, peer,
									 MPIR_REDUCE_SCATTER_TAG,
									 incoming_data + recv_offset * true_extent,
									 size, datatype, peer,
									 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
									 MPI_STATUS_IGNORE, errflag);
		if (mpi_errno) {
			/* for communication errors, just record the error but continue */
			*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
			MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
			MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
		}
		/* always perform the reduction at recv_offset, the data at send_offset
		   is now our peer's responsibility */
		if (rank > peer) {
			/* higher ranked value so need to call op(received_data, my_data) */
			call_uop(incoming_data + recv_offset * true_extent,
					 outgoing_data + recv_offset * true_extent, size, datatype);
			buf0_was_inout = buf0_was_inout;
		} else {
			/* lower ranked value so need to call op(my_data, received_data) */
			call_uop(outgoing_data + recv_offset * true_extent,
					 incoming_data + recv_offset * true_extent, size, datatype);
			buf0_was_inout = !buf0_was_inout;
		}

		/* the next round of send/recv needs to happen within the block (of size
		   "size") that we just received and reduced */
		send_offset = recv_offset;
	}

	MPIU_Assert(size == recvcnts[rank]);

	/* copy the reduced data to the recvbuf */
	result_ptr =
		(char *) (buf0_was_inout ? tmp_buf0 : tmp_buf1) +
		recv_offset * true_extent;
	mpi_errno =
		MPIR_Localcopy(result_ptr, size, datatype, recvbuf, size, datatype);
  fn_exit:
	MPIU_CHKLMEM_FREEALL();
	if (mpi_errno_ret)
		mpi_errno = mpi_errno_ret;
	else if (*errflag)
		MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
	MPIR_TIMER_END(coll,reduce_scatter,noncomm);
	return mpi_errno;
  fn_fail:
	goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_Scatter_Basic_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_Scatter_Basic_MV2(const void *sendbuf,
                                  void *recvbuf,
                                  const int *recvcnts,
                                  MPI_Datatype datatype,
                                  MPI_Op op,
                                  MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
	MPIR_TIMER_START(coll,reduce_scatter,basic);
	int mpi_errno = MPI_SUCCESS, i;
	int rank, size;
	int total_count;
	int *displs = NULL;
	char *tmprbuf = NULL;
	char *tmprbuf_free = NULL;
	int root = 0, use_scatterv = 0;
	MPIU_CHKLMEM_DECL(1);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_scatter_basic, 1);

	rank = comm_ptr->rank;
	size = comm_ptr->local_size;

	for (i = 0, total_count = 0; i < size; i++) {
		total_count += recvcnts[i];
	}

	/* Reduce to rank 0 (root) and scatterv */
	tmprbuf = (char *) recvbuf;
	if (MPI_IN_PLACE == sendbuf) {
		/* recvbuf on root (0) is big enough to hold whole data */
		if (root == rank) {
			mpi_errno = MPIR_Reduce_MV2(MPI_IN_PLACE, tmprbuf, total_count,
										datatype, op, root, comm_ptr, errflag);
		} else {
			mpi_errno = MPIR_Reduce_MV2(tmprbuf, NULL, total_count,
										datatype, op, root, comm_ptr, errflag);
		}
	} else {
        /* Allocate temporary receive buffer on root to ensure that
         *                rbuf is big enough */
        MPI_Aint true_lb, true_extent, extent;
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
        MPID_Datatype_get_extent_macro(datatype, extent);

        MPIU_CHKLMEM_MALLOC(tmprbuf_free, void *, total_count *
                (MPIR_MAX(extent, true_extent)),
                mpi_errno, "receive buffer");
        tmprbuf = (void *) ((char *) tmprbuf_free - true_lb);

		mpi_errno = MPIR_Reduce_MV2(sendbuf, tmprbuf, total_count,
									datatype, op, root, comm_ptr, errflag);
	}
	if (MPI_SUCCESS != mpi_errno) {
		MPIR_TIMER_END(coll,reduce_scatter,basic);
		return mpi_errno;
	}

	for (i = 1; i < size; i++) {
		if (recvcnts[i] != recvcnts[i - 1]) {
			use_scatterv = 1;
			break;
		}
	}

	if (use_scatterv == 0) {
        /* Use regular Scatter when possible */ 
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Scatter_MV2(tmprbuf, recvcnts[0], datatype,
                                         recvbuf, recvcnts[0], datatype,
                                         root, comm_ptr, errflag);
        } else {
            if (root == rank) {
                mpi_errno = MPIR_Scatter_MV2(tmprbuf, recvcnts[0], datatype,
                                             MPI_IN_PLACE, recvcnts[0], datatype,
                                             root, comm_ptr, errflag);
            } else {
                mpi_errno = MPIR_Scatter_MV2(NULL, recvcnts[0], datatype,
                                             recvbuf, recvcnts[0], datatype,
                                             root, comm_ptr, errflag);
            }
      }
        if (MPI_SUCCESS != mpi_errno) {
        	MPIR_TIMER_END(coll,reduce_scatter,basic);
			return mpi_errno;
		}
	} else {

		displs = MPIU_Malloc(size * sizeof (int));
		displs[0] = 0;
		for (i = 1; i < size; i++) {
			displs[i] = displs[i - 1] + recvcnts[i - 1];
		}
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Scatterv(tmprbuf, recvcnts, displs, datatype,
                                      recvbuf, recvcnts[rank], datatype,
                                      root, comm_ptr, errflag);
        } else {

            if (root == rank) {
                mpi_errno = MPIR_Scatterv(tmprbuf, recvcnts, displs, datatype,
                                             MPI_IN_PLACE, recvcnts[rank], datatype,
                                             root, comm_ptr, errflag);
            } else {
                mpi_errno = MPIR_Scatterv(NULL, recvcnts, displs, datatype, 
                                             recvbuf, recvcnts[rank], datatype,
                                             root, comm_ptr, errflag);
            }
        }
		if (MPI_SUCCESS != mpi_errno) {
			MPIR_TIMER_END(coll,reduce_scatter,basic);
			return mpi_errno;
		}
		MPIU_Free(displs);
	}

  fn_exit:
	MPIU_CHKLMEM_FREEALL();
	MPIR_TIMER_END(coll,reduce_scatter,basic);
	return mpi_errno;
  fn_fail:
	goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_Rec_Halving_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_Rec_Halving_MV2(const void *sendbuf,
                                        void *recvbuf,
                                        const int *recvcnts,
                                        MPI_Datatype datatype,
                                        MPI_Op op,
                                        MPID_Comm * comm_ptr,
                                        MPIR_Errflag_t *errflag)
{
	MPIR_TIMER_START(coll,reduce_scatter,rec_halving);
	int rank, comm_size, i;
	MPI_Aint extent, true_extent, true_lb;
	int *disps;
	void *tmp_recvbuf, *tmp_results;
	int mpi_errno = MPI_SUCCESS;
	int mpi_errno_ret = MPI_SUCCESS;
	int total_count, dst;
	int mask;
	int *newcnts, *newdisps, rem, newdst, send_idx, recv_idx,
		last_idx, send_cnt, recv_cnt;
	int pof2, old_i, newrank;
	MPI_User_function *uop;
	MPID_Op *op_ptr;
	MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
	int is_cxx_uop = 0;
#endif
	MPIU_CHKLMEM_DECL(5);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_scatter_rec_halving, 1);

	comm_size = comm_ptr->local_size;
	rank = comm_ptr->rank;

	/* set op_errno to 0. stored in perthread structure */
	MPIU_THREADPRIV_GET;
	MPIU_THREADPRIV_FIELD(op_errno) = 0;

	MPID_Datatype_get_extent_macro(datatype, extent);
	MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

	if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
		/* get the function by indexing into the op table */
		uop = MPIR_Op_table[op % 16 - 1];
	} else {
		MPID_Op_get_ptr(op, op_ptr);
#ifdef HAVE_CXX_BINDING
		if (op_ptr->language == MPID_LANG_CXX) {
			uop = (MPI_User_function *) op_ptr->function.c_function;
			is_cxx_uop = 1;
		} else
#endif
		if (op_ptr->language == MPID_LANG_C)
			uop = (MPI_User_function *) op_ptr->function.c_function;
		else
			uop = (MPI_User_function *) op_ptr->function.f77_function;
	}

	MPIU_CHKLMEM_MALLOC(disps, int *, comm_size * sizeof (int), mpi_errno,
						"disps");

	total_count = 0;
	for (i = 0; i < comm_size; i++) {
		disps[i] = total_count;
		total_count += recvcnts[i];
	}

	if (total_count == 0) {
		goto fn_exit;
	}

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

	/* total_count*extent eventually gets malloced. it isn't added to
	 * a user-passed in buffer */
	MPIU_Ensure_Aint_fits_in_pointer(total_count *
									 MPIR_MAX(true_extent, extent));

	/* commutative and short. use recursive halving algorithm */
	/* allocate temp. buffer to receive incoming data */
	MPIU_CHKLMEM_MALLOC(tmp_recvbuf, void *,
						total_count * (MPIR_MAX(true_extent, extent)),
						mpi_errno, "tmp_recvbuf");
	/* adjust for potential negative lower bound in datatype */
	tmp_recvbuf = (void *) ((char *) tmp_recvbuf - true_lb);

	/* need to allocate another temporary buffer to accumulate
	   results because recvbuf may not be big enough */
	MPIU_CHKLMEM_MALLOC(tmp_results, void *,
						total_count * (MPIR_MAX(true_extent, extent)),
						mpi_errno, "tmp_results");
	/* adjust for potential negative lower bound in datatype */
	tmp_results = (void *) ((char *) tmp_results - true_lb);

	/* copy sendbuf into tmp_results */
	if (sendbuf != MPI_IN_PLACE)
		mpi_errno = MPIR_Localcopy(sendbuf, total_count, datatype,
								   tmp_results, total_count, datatype);
	else
		mpi_errno = MPIR_Localcopy(recvbuf, total_count, datatype,
								   tmp_results, total_count, datatype);

	if (mpi_errno)
		MPIR_ERR_POP(mpi_errno);

	pof2 = comm_ptr->dev.ch.gpof2;

	rem = comm_size - pof2;

	/* In the non-power-of-two case, all even-numbered
	   processes of rank < 2*rem send their data to
	   (rank+1). These even-numbered processes no longer
	   participate in the algorithm until the very end. The
	   remaining processes form a nice power-of-two. */

	if (rank < 2 * rem) {
		if (rank % 2 == 0) {	/* even */
			MPIR_PVAR_INC(reduce_scatter, rec_halving, send, total_count, datatype);
			mpi_errno = MPIC_Send(tmp_results, total_count,
									 datatype, rank + 1,
									 MPIR_REDUCE_SCATTER_TAG, comm_ptr, errflag);
			if (mpi_errno) {
				/* for communication errors, just record the error but continue */
				*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
				MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
				MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
			}

			/* temporarily set the rank to -1 so that this
			   process does not pariticipate in recursive
			   doubling */
			newrank = -1;
		} else {				/* odd */
			MPIR_PVAR_INC(reduce_scatter, rec_halving, recv, total_count, datatype);
			mpi_errno = MPIC_Recv(tmp_recvbuf, total_count,
									 datatype, rank - 1,
									 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
									 MPI_STATUS_IGNORE, errflag);
			if (mpi_errno) {
				/* for communication errors, just record the error but continue */
				*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
				MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
				MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
			}

			/* do the reduction on received data. since the
			   ordering is right, it doesn't matter whether
			   the operation is commutative or not. */
#ifdef HAVE_CXX_BINDING
			if (is_cxx_uop) {
				(*MPIR_Process.cxx_call_op_fn) (tmp_recvbuf, tmp_results,
												total_count, datatype, uop);
			} else
#endif
				(*uop) (tmp_recvbuf, tmp_results, &total_count, &datatype);

			/* change the rank */
			newrank = rank / 2;
		}
	} else						/* rank >= 2*rem */
		newrank = rank - rem;

	if (newrank != -1) {
		/* recalculate the recvcnts and disps arrays because the
		   even-numbered processes who no longer participate will
		   have their result calculated by the process to their
		   right (rank+1). */
		MPID_Request *request = NULL;
		MPI_Status status;
		MPIU_CHKLMEM_MALLOC(newcnts, int *, pof2 * sizeof (int), mpi_errno,
							"newcnts");
		MPIU_CHKLMEM_MALLOC(newdisps, int *, pof2 * sizeof (int), mpi_errno,
							"newdisps");

		for (i = 0; i < pof2; i++) {
			/* what does i map to in the old ranking? */
			old_i = (i < rem) ? i * 2 + 1 : i + rem;
			if (old_i < 2 * rem) {
				/* This process has to also do its left neighbor's
				   work */
				newcnts[i] = recvcnts[old_i] + recvcnts[old_i - 1];
			} else
				newcnts[i] = recvcnts[old_i];
		}

		newdisps[0] = 0;
		for (i = 1; i < pof2; i++)
			newdisps[i] = newdisps[i - 1] + newcnts[i - 1];

		mask = pof2 >> 1;
		send_idx = recv_idx = 0;
		last_idx = pof2;
		while (mask > 0) {
			newdst = newrank ^ mask;
			/* find real rank of dest */
			dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

			send_cnt = recv_cnt = 0;
			if (newrank < newdst) {
				send_idx = recv_idx + mask;
				for (i = send_idx; i < last_idx; i++)
					send_cnt += newcnts[i];
				for (i = recv_idx; i < send_idx; i++)
					recv_cnt += newcnts[i];
			} else {
				recv_idx = send_idx + mask;
				for (i = send_idx; i < recv_idx; i++)
					send_cnt += newcnts[i];
				for (i = recv_idx; i < last_idx; i++)
					recv_cnt += newcnts[i];
			}
			/* Send data from tmp_results. Recv into tmp_recvbuf */
			if ((send_cnt != 0) && (recv_cnt != 0))
			{
				MPIR_PVAR_INC(reduce_scatter, rec_halving, send, send_cnt, datatype);
				MPIR_PVAR_INC(reduce_scatter, rec_halving, recv, recv_cnt, datatype);
				mpi_errno = MPIC_Sendrecv((char *) tmp_results +
											 newdisps[send_idx] * extent,
											 send_cnt, datatype,
											 dst, MPIR_REDUCE_SCATTER_TAG,
											 (char *) tmp_recvbuf +
											 newdisps[recv_idx] * extent,
											 recv_cnt, datatype, dst,
											 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
											 MPI_STATUS_IGNORE, errflag);
			}
			else if ((send_cnt == 0) && (recv_cnt != 0))
			{
				MPIR_PVAR_INC(reduce_scatter, rec_halving, recv, recv_cnt, datatype);
				mpi_errno = MPIC_Irecv((char *) tmp_recvbuf +
										  newdisps[recv_idx] * extent,
										  recv_cnt, datatype, dst,
										  MPIR_REDUCE_SCATTER_TAG, comm_ptr,
										  &request);
			}
			else if ((recv_cnt == 0) && (send_cnt != 0))
			{
				MPIR_PVAR_INC(reduce_scatter, rec_halving, send, send_cnt, datatype);
				mpi_errno = MPIC_Send((char *) tmp_results +
										 newdisps[send_idx] * extent,
										 send_cnt, datatype,
										 dst, MPIR_REDUCE_SCATTER_TAG,
										 comm_ptr, errflag);
			}

			if (mpi_errno) {
				/* for communication errors, just record the error but continue */
				*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
				MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
				MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
			}

			if ((send_cnt == 0) && (recv_cnt != 0)) {
				mpi_errno = MPIC_Waitall(1, &request, &status, errflag);
				if (mpi_errno) {
					/* for communication errors, just record the error but continue */
					*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
					MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
					MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
				}
			}

			/* tmp_recvbuf contains data received in this step.
			   tmp_results contains data accumulated so far */

			if (recv_cnt) {
#ifdef HAVE_CXX_BINDING
				if (is_cxx_uop) {
					(*MPIR_Process.cxx_call_op_fn) ((char *) tmp_recvbuf +
													newdisps[recv_idx] * extent,
													(char *) tmp_results +
													newdisps[recv_idx] * extent,
													recv_cnt, datatype, uop);
				} else
#endif
					(*uop) ((char *) tmp_recvbuf + newdisps[recv_idx] * extent,
							(char *) tmp_results + newdisps[recv_idx] * extent,
							&recv_cnt, &datatype);
			}

			/* update send_idx for next iteration */
			send_idx = recv_idx;
			last_idx = recv_idx + mask;
			mask >>= 1;
		}

		/* copy this process's result from tmp_results to recvbuf */
		if (recvcnts[rank]) {
			mpi_errno = MPIR_Localcopy((char *) tmp_results +
									   disps[rank] * extent,
									   recvcnts[rank], datatype, recvbuf,
									   recvcnts[rank], datatype);
			if (mpi_errno)
				MPIR_ERR_POP(mpi_errno);
		}

	}

	/* In the non-power-of-two case, all odd-numbered
	   processes of rank < 2*rem send to (rank-1) the result they
	   calculated for that process */
	if (rank < 2 * rem) {
		if (rank % 2) {			/* odd */
			if (recvcnts[rank - 1]) {
				MPIR_PVAR_INC(reduce_scatter, rec_halving, send, recvcnts[rank - 1], datatype);
				mpi_errno = MPIC_Send((char *) tmp_results +
										 disps[rank - 1] * extent,
										 recvcnts[rank - 1], datatype, rank - 1,
										 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
										 errflag);
				if (mpi_errno) {
					/* for communication errors, just record the error but continue */
					*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
					MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
					MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
				}
			}
		} else {				/* even */
			if (recvcnts[rank]) {
				MPIR_PVAR_INC(reduce_scatter, rec_halving, recv, recvcnts[rank], datatype);
				mpi_errno = MPIC_Recv(recvbuf, recvcnts[rank],
										 datatype, rank + 1,
										 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
										 MPI_STATUS_IGNORE, errflag);
				if (mpi_errno) {
					/* for communication errors, just record the error but continue */
					*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
					MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
					MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
				}
			}
		}
	}

  fn_exit:
	MPIU_CHKLMEM_FREEALL();

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

	if (MPIU_THREADPRIV_FIELD(op_errno))
		mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);

	if (mpi_errno_ret)
		mpi_errno = mpi_errno_ret;
	else if (*errflag)
		MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

	MPIR_TIMER_END(coll,reduce_scatter,rec_halving);
	return mpi_errno;
  fn_fail:
	goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_Pair_Wise_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_Pair_Wise_MV2(const void *sendbuf,
                                      void *recvbuf,
                                      const int *recvcnts,
                                      MPI_Datatype datatype,
                                      MPI_Op op,
                                      MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
	MPIR_TIMER_START(coll,reduce_scatter,pairwise);
	int rank, comm_size, i;
	MPI_Aint extent, true_extent, true_lb;
	int *disps;
	void *tmp_recvbuf;
	int mpi_errno = MPI_SUCCESS;
	int mpi_errno_ret = MPI_SUCCESS;
	int total_count, src, dst;
	int is_commutative;
	MPI_User_function *uop;
	MPID_Op *op_ptr;
	MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
	int is_cxx_uop = 0;
#endif
	MPIU_CHKLMEM_DECL(5);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_scatter_pairwise, 1);

	comm_size = comm_ptr->local_size;
	rank = comm_ptr->rank;

	/* set op_errno to 0. stored in perthread structure */
	MPIU_THREADPRIV_GET;
	MPIU_THREADPRIV_FIELD(op_errno) = 0;

	MPID_Datatype_get_extent_macro(datatype, extent);
	MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

	if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
		is_commutative = 1;
		/* get the function by indexing into the op table */
		uop = MPIR_Op_table[op % 16 - 1];
	} else {
		MPID_Op_get_ptr(op, op_ptr);
		if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
			is_commutative = 0;
		else
			is_commutative = 1;

#ifdef HAVE_CXX_BINDING
		if (op_ptr->language == MPID_LANG_CXX) {
			uop = (MPI_User_function *) op_ptr->function.c_function;
			is_cxx_uop = 1;
		} else
#endif
		if (op_ptr->language == MPID_LANG_C)
			uop = (MPI_User_function *) op_ptr->function.c_function;
		else
			uop = (MPI_User_function *) op_ptr->function.f77_function;
	}

	MPIU_CHKLMEM_MALLOC(disps, int *, comm_size * sizeof (int), mpi_errno,
						"disps");

	total_count = 0;
	for (i = 0; i < comm_size; i++) {
		disps[i] = total_count;
		total_count += recvcnts[i];
	}

	if (total_count == 0) {
		goto fn_exit;
	}

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

	MPIU_Ensure_Aint_fits_in_pointer(total_count *
									 MPIR_MAX(true_extent, extent));

	/* commutative and long message, or noncommutative and long message.
	   use (p-1) pairwise exchanges */

	if (sendbuf != MPI_IN_PLACE) {
		/* copy local data into recvbuf */
		mpi_errno = MPIR_Localcopy(((char *) sendbuf + disps[rank] * extent),
								   recvcnts[rank], datatype, recvbuf,
								   recvcnts[rank], datatype);
		if (mpi_errno)
			MPIR_ERR_POP(mpi_errno);
	}

	/* allocate temporary buffer to store incoming data */
	MPIU_CHKLMEM_MALLOC(tmp_recvbuf, void *,
						recvcnts[rank] * (MPIR_MAX(true_extent, extent)) + 1,
						mpi_errno, "tmp_recvbuf");
	/* adjust for potential negative lower bound in datatype */
	tmp_recvbuf = (void *) ((char *) tmp_recvbuf - true_lb);

	for (i = 1; i < comm_size; i++) {
		src = (rank - i + comm_size) % comm_size;
		dst = (rank + i) % comm_size;

		/* send the data that dst needs. recv data that this process
		   needs from src into tmp_recvbuf */
		if (sendbuf != MPI_IN_PLACE)
		{
			MPIR_PVAR_INC(reduce_scatter, pairwise, send, recvcnts[dst], datatype);
			MPIR_PVAR_INC(reduce_scatter, pairwise, recv, recvcnts[rank], datatype);
			mpi_errno =
				MPIC_Sendrecv(((char *) sendbuf + disps[dst] * extent),
								 recvcnts[dst], datatype, dst,
								 MPIR_REDUCE_SCATTER_TAG, tmp_recvbuf,
								 recvcnts[rank], datatype, src,
								 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
								 MPI_STATUS_IGNORE, errflag);
		}
		else
		{
			MPIR_PVAR_INC(reduce_scatter, pairwise, send, recvcnts[dst], datatype);
			MPIR_PVAR_INC(reduce_scatter, pairwise, recv, recvcnts[rank], datatype);
			mpi_errno =
				MPIC_Sendrecv(((char *) recvbuf + disps[dst] * extent),
								 recvcnts[dst], datatype, dst,
								 MPIR_REDUCE_SCATTER_TAG, tmp_recvbuf,
								 recvcnts[rank], datatype, src,
								 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
								 MPI_STATUS_IGNORE, errflag);
		}
		if (mpi_errno) {
			/* for communication errors, just record the error but continue */
			*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
			MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
			MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
		}

		if (is_commutative || (src < rank)) {
			if (sendbuf != MPI_IN_PLACE) {
#ifdef HAVE_CXX_BINDING
				if (is_cxx_uop) {
					(*MPIR_Process.cxx_call_op_fn) (tmp_recvbuf,
													recvbuf,
													recvcnts[rank],
													datatype, uop);
				} else
#endif
					(*uop) (tmp_recvbuf, recvbuf, (int *)&recvcnts[rank], &datatype);
			} else {
#ifdef HAVE_CXX_BINDING
				if (is_cxx_uop) {
					(*MPIR_Process.cxx_call_op_fn) (tmp_recvbuf,
													((char *) recvbuf +
													 disps[rank] * extent),
													recvcnts[rank], datatype,
													uop);
				} else
#endif
					(*uop) (tmp_recvbuf,
							((char *) recvbuf + disps[rank] * extent),
							(void *)&recvcnts[rank], &datatype);
				/* we can't store the result at the beginning of
				   recvbuf right here because there is useful data
				   there that other process/processes need. at the
				   end, we will copy back the result to the
				   beginning of recvbuf. */
			}
		} else {
			if (sendbuf != MPI_IN_PLACE) {
#ifdef HAVE_CXX_BINDING
				if (is_cxx_uop) {
					(*MPIR_Process.cxx_call_op_fn) (recvbuf,
													tmp_recvbuf,
													recvcnts[rank],
													datatype, uop);
				} else
#endif
					(*uop) (recvbuf, tmp_recvbuf, (int *)&recvcnts[rank], &datatype);
				/* copy result back into recvbuf */
				mpi_errno = MPIR_Localcopy(tmp_recvbuf, recvcnts[rank],
										   datatype, recvbuf,
										   recvcnts[rank], datatype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);
			} else {
#ifdef HAVE_CXX_BINDING
				if (is_cxx_uop) {
					(*MPIR_Process.cxx_call_op_fn) (((char *) recvbuf +
													 disps[rank] * extent),
													tmp_recvbuf, recvcnts[rank],
													datatype, uop);

				} else
#endif
					(*uop) (((char *) recvbuf + disps[rank] * extent),
							tmp_recvbuf, (int *)&recvcnts[rank], &datatype);
				/* copy result back into recvbuf */
				mpi_errno = MPIR_Localcopy(tmp_recvbuf, recvcnts[rank],
										   datatype,
										   ((char *) recvbuf +
											disps[rank] * extent),
										   recvcnts[rank], datatype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);
			}
		}
	}

	/* if MPI_IN_PLACE, move output data to the beginning of
	   recvbuf. already done for rank 0. */
	if ((sendbuf == MPI_IN_PLACE) && (rank != 0)) {
		mpi_errno = MPIR_Localcopy(((char *) recvbuf +
									disps[rank] * extent),
								   recvcnts[rank], datatype,
								   recvbuf, recvcnts[rank], datatype);
		if (mpi_errno)
			MPIR_ERR_POP(mpi_errno);
	}

  fn_exit:
	MPIU_CHKLMEM_FREEALL();

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

	if (MPIU_THREADPRIV_FIELD(op_errno))
		mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);

	if (mpi_errno_ret)
		mpi_errno = mpi_errno_ret;
	else if (*errflag)
		MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

	MPIR_TIMER_END(coll,reduce_scatter,pairwise);
	return mpi_errno;
  fn_fail:
	goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_ring(const void* sendbuf, void* recvbuf,
    const int *recvcnts, MPI_Datatype datatype,
    MPI_Op op, MPID_Comm *comm_ptr,
    MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
	if (PRINT_FUN_NAME){
    printf("[red_scat_osu rank = %d   count = (array) %d   host = %s] Func: MPIR_Reduce_scatter_ring @ Red_scat_osu.c [Un-sec]\n",comm_ptr->rank,recvcnts,hostname);fflush(stdout);}
#endif	
	
	MPIR_TIMER_START(coll,reduce_scatter,ring);
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;
    int rank          = comm_ptr->rank;

    MPIU_CHKLMEM_DECL(3);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* compute left and right neighbors */
    int rank_left  = (rank - 1 + comm_size) % comm_size;
    int rank_right = (rank + 1 + comm_size) % comm_size;

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* allocate memory for displacement array */
    int *displs       = NULL;
    MPIU_CHKLMEM_MALLOC(displs, int*, comm_size * sizeof(int), mpi_errno, "displs");

    /* total count of data for each rank */
    int total_count       = 0;
    int max_recv_count    = 0;
    int i;
    for (i = 0; i < comm_size; i++) {
        displs[i] = total_count;
        total_count += recvcnts[i];

        if (recvcnts[i] > max_recv_count) {
            max_recv_count = recvcnts[i];
        }
    }

    /* max number of elements a rank will receive */
    size_t max_elem_per_rank = (size_t) max_recv_count;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
    while (nread < max_elem_per_rank) {
        /* keep track of send and recv counts left to process */
        /* execute a lap through the ring */
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute rank of process whose data we're sending and rank
             * of process whose data we're receiving in this step */
            int send_rank = (rank + dist) % comm_size;
            int recv_rank = (rank + (dist - 1)) % comm_size;

            /* compute offset into input buffer to pull data for this chunk */
            unsigned long elem_offset = displs[send_rank] + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* compute the number of elements we're sending and receiving */
            int send_count = recvcnts[send_rank] - nread;
            int recv_count = recvcnts[recv_rank] - nread;

            if (send_count < 0) send_count = 0;
            if (recv_count < 0) recv_count = 0;

            if (send_count > elem_per_buffer) send_count = elem_per_buffer;
            if (recv_count > elem_per_buffer) recv_count = elem_per_buffer;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, send_count, datatype,
                    tmp_sendbuf, send_count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, send_count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
				
				/* unsigned int recv_enc_size = recv_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				unsigned int send_enc_size = send_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				
				unsigned int ciphertext_sendbuf_len,dec_count,max_out_len;

				//MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,&request[0]);
				MPIC_Irecv(large_recv_buffer, recv_enc_size, MPI_CHAR, rank_left,  0, comm_ptr,&request[0]);
						
				RAND_bytes(large_send_buffer, NONCE_SIZE);
				max_out_len = send_count*extent+ENC_MSG_TAG_SIZE; 
				if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer+NONCE_SIZE),
							   &ciphertext_sendbuf_len, max_out_len,
							   (&large_send_buffer[0]), NONCE_SIZE,
							   (char *)tmp_sendbuf, send_count*extent,
							   NULL, 0))
				{
					printf("Error in encryption: Red-Scat_2lvl  rank = %d\n",comm_ptr->rank);
					fflush(stdout);
				}
				
				MPIC_Isend(large_send_buffer, send_enc_size, MPI_CHAR, rank_right, 0, comm_ptr, &request[1], errflag); */
				
                MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, send_count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag);
                MPIC_Waitall(2, request, status, errflag);
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, send_count, datatype,
                        buf, send_count, datatype);
            }
        }

        /* assume we send the max buffer count in each step,
         * this means that nread may exceed the max value,
         * but that will end the loop */
        nread += elem_per_buffer;
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */

    MPIR_TIMER_END(coll,reduce_scatter,ring);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_ring_2lvl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

/************ Added by Mohsen *************/


int MPIR_Reduce_scatter_ring_2lvl(const void* sendbuf, void* recvbuf,
								  const int *recvcnts, MPI_Datatype datatype,
								  MPI_Op op, MPID_Comm *comm_ptr,
								  MPIR_Errflag_t *errflag)
{
    if (security_approach==2005) 
        return MPIR_Reduce_scatter_ring_2lvl_NodeAware(sendbuf, recvbuf, recvcnts, datatype, op, comm_ptr, errflag);
    else if (security_approach>0) 
        return MPIR_Reduce_scatter_ring_2lvl_Naive(sendbuf, recvbuf, recvcnts, datatype, op, comm_ptr, errflag);
    else return MPIR_Reduce_scatter_ring_2lvl_Original(sendbuf, recvbuf, recvcnts, datatype, op, comm_ptr, errflag);
}




int MPIR_Reduce_scatter_ring_2lvl_NodeAware(const void* sendbuf, void* recvbuf,
        const int *recvcnts, MPI_Datatype datatype,
        MPI_Op op, MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
	if (PRINT_FUN_NAME){
    printf("[red_scat_osu rank = %d   count = (array) %d   host = %s] Func: MPIR_Reduce_scatter_ring_2lvl_NodeAware @ Red_scat_osu.c [COLL_NAIVE_AESGCM_PLUS]\n",comm_ptr->rank,recvcnts,hostname);fflush(stdout);}
#endif		
	MPIR_TIMER_START(coll,reduce_scatter,ring_2lvl);
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;

    if (comm_ptr->dev.ch.rank_list == NULL) {
    	MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
        return MPIR_Reduce_scatter_ring(
                            sendbuf, recvbuf, recvcnts, datatype,
                            op, comm_ptr, errflag);
    }
	
	EVP_AEAD_CTX *global_coll_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);

    MPIU_CHKLMEM_DECL(3);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int index_left  = (comm_size + rank_index - 1) % comm_size;
    int index_right = (comm_size + rank_index + 1) % comm_size;
    int rank_left   = comm_ptr->dev.ch.rank_list[index_left];
    int rank_right  = comm_ptr->dev.ch.rank_list[index_right];
	
	MPID_Node_id_t my_node_id;
	MPID_Node_id_t rank_left_node_id;
	MPID_Node_id_t rank_right_node_id;

	MPID_Get_node_id(comm_ptr, rank_index, &my_node_id);
	MPID_Get_node_id(comm_ptr, rank_left, &rank_left_node_id);
	MPID_Get_node_id(comm_ptr, rank_right, &rank_right_node_id);

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* allocate memory for displacement array */
    int *displs       = NULL;
    MPIU_CHKLMEM_MALLOC(displs, int*, comm_size * sizeof(int), mpi_errno, "displs");

    /* total count of data for each rank */
    int total_count       = 0;
    int max_recv_count    = 0;
    int i;
    for (i = 0; i < comm_size; i++) {
        displs[i] = total_count;
        total_count += recvcnts[i];

        if (recvcnts[i] > max_recv_count) {
            max_recv_count = recvcnts[i];
        }
    }

    /* max number of elements a rank will receive */
    size_t max_elem_per_rank = (size_t) max_recv_count;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
	int jj=0;
	
    while (nread < max_elem_per_rank) {
        /* keep track of send and recv counts left to process */
        /* execute a lap through the ring */		
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute rank of process whose data we're sending and rank
             * of process whose data we're receiving in this step */
            int send_index = (rank_index + dist) % comm_size;
            int recv_index = (rank_index + (dist - 1)) % comm_size;
            int send_rank  = comm_ptr->dev.ch.rank_list[send_index];
            int recv_rank  = comm_ptr->dev.ch.rank_list[recv_index];

            /* compute offset into input buffer to pull data for this chunk */
            unsigned long elem_offset = displs[send_rank] + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* compute the number of elements we're sending and receiving */
            int send_count = recvcnts[send_rank] - nread;
            int recv_count = recvcnts[recv_rank] - nread;

            if (send_count < 0) send_count = 0;
            if (recv_count < 0) recv_count = 0;

            if (send_count > elem_per_buffer) send_count = elem_per_buffer;
            if (recv_count > elem_per_buffer) recv_count = elem_per_buffer;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, send_count, datatype,
                    tmp_sendbuf, send_count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, send_count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
				
				unsigned int recv_enc_size = recv_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				unsigned int send_enc_size = send_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				
				unsigned int ciphertext_sendbuf_len,dec_count,max_out_len;

				//MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,&request[0]);
				
				
				if (my_node_id == rank_left_node_id){					
					MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr, &request[0]);					
				} else {
					MPIC_Irecv(large_recv_buffer, recv_enc_size, MPI_CHAR, rank_left,  0, comm_ptr,&request[0]);				
				}
				
				if (my_node_id == rank_right_node_id){
					MPIC_Isend(tmp_sendbuf, send_count, datatype, rank_right, 0, comm_ptr, &request[1], errflag);
					
				} else {					
					RAND_bytes(large_send_buffer, NONCE_SIZE);
					max_out_len = send_count*extent+ENC_MSG_TAG_SIZE; 
					if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer+NONCE_SIZE),
								   &ciphertext_sendbuf_len, max_out_len,
								   (&large_send_buffer[0]), NONCE_SIZE,
								   (char *)tmp_sendbuf, send_count*extent,
								   NULL, 0))
					{
						printf("Error in encryption: Red-Scat_2lvl  rank = %d\n",comm_ptr->rank);
						fflush(stdout);
					}
					
					MPIC_Isend(large_send_buffer, send_enc_size, MPI_CHAR, rank_right, 0, comm_ptr, &request[1], errflag);
				}
				
                /* MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, send_count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag); */
                MPIC_Waitall(2, request, status, errflag);
				
				if (my_node_id != rank_left_node_id){					
				
					if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_recvbuf,
								   &dec_count, recv_count*extent,
								   (&large_recv_buffer[0]), NONCE_SIZE,
								   large_recv_buffer + NONCE_SIZE, (recv_count*extent + ENC_MSG_TAG_SIZE),
								   NULL, 0))
					{
						fprintf(stderr,"Error in Decryption Red-Scat_2lvl : Rank = %d --> dest =%d  recv_count = %d \n",comm_ptr->rank, rank_right, recv_count);						
					}
				}
				
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, send_count, datatype,
                        buf, send_count, datatype);
            }
			jj++;
        }

        /* assume we send the max buffer count in each step,
         * this means that nread may exceed the max value,
         * but that will end the loop */
        nread += elem_per_buffer;
		
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */

    MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
 

int MPIR_Reduce_scatter_ring_2lvl_Naive(const void* sendbuf, void* recvbuf,
        const int *recvcnts, MPI_Datatype datatype,
        MPI_Op op, MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
	if (PRINT_FUN_NAME){
    printf("[red_scat_osu rank = %d   count = (array) %d   host = %s] Func: MPIR_Reduce_scatter_ring_2lvl_Naive @ Red_scat_osu.c [COLL_NAIVE_AESGCM]\n",comm_ptr->rank,recvcnts,hostname);fflush(stdout);}
#endif		
	MPIR_TIMER_START(coll,reduce_scatter,ring_2lvl);
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;

    if (comm_ptr->dev.ch.rank_list == NULL) {
    	MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
        return MPIR_Reduce_scatter_ring(
                            sendbuf, recvbuf, recvcnts, datatype,
                            op, comm_ptr, errflag);
    }
	
	EVP_AEAD_CTX *global_coll_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);

    MPIU_CHKLMEM_DECL(3);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int index_left  = (comm_size + rank_index - 1) % comm_size;
    int index_right = (comm_size + rank_index + 1) % comm_size;
    int rank_left   = comm_ptr->dev.ch.rank_list[index_left];
    int rank_right  = comm_ptr->dev.ch.rank_list[index_right];
	
	MPID_Node_id_t my_node_id;
	MPID_Node_id_t rank_left_node_id;
	MPID_Node_id_t rank_right_node_id;

	MPID_Get_node_id(comm_ptr, rank_index, &my_node_id);
	MPID_Get_node_id(comm_ptr, rank_left, &rank_left_node_id);
	MPID_Get_node_id(comm_ptr, rank_right, &rank_right_node_id);

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* allocate memory for displacement array */
    int *displs       = NULL;
    MPIU_CHKLMEM_MALLOC(displs, int*, comm_size * sizeof(int), mpi_errno, "displs");

    /* total count of data for each rank */
    int total_count       = 0;
    int max_recv_count    = 0;
    int i;
    for (i = 0; i < comm_size; i++) {
        displs[i] = total_count;
        total_count += recvcnts[i];

        if (recvcnts[i] > max_recv_count) {
            max_recv_count = recvcnts[i];
        }
    }

    /* max number of elements a rank will receive */
    size_t max_elem_per_rank = (size_t) max_recv_count;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
    while (nread < max_elem_per_rank) {
        /* keep track of send and recv counts left to process */
        /* execute a lap through the ring */
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute rank of process whose data we're sending and rank
             * of process whose data we're receiving in this step */
            int send_index = (rank_index + dist) % comm_size;
            int recv_index = (rank_index + (dist - 1)) % comm_size;
            int send_rank  = comm_ptr->dev.ch.rank_list[send_index];
            int recv_rank  = comm_ptr->dev.ch.rank_list[recv_index];

            /* compute offset into input buffer to pull data for this chunk */
            unsigned long elem_offset = displs[send_rank] + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* compute the number of elements we're sending and receiving */
            int send_count = recvcnts[send_rank] - nread;
            int recv_count = recvcnts[recv_rank] - nread;

            if (send_count < 0) send_count = 0;
            if (recv_count < 0) recv_count = 0;

            if (send_count > elem_per_buffer) send_count = elem_per_buffer;
            if (recv_count > elem_per_buffer) recv_count = elem_per_buffer;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, send_count, datatype,
                    tmp_sendbuf, send_count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, send_count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
				
				unsigned int recv_enc_size = recv_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				unsigned int send_enc_size = send_count*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
				
				unsigned int ciphertext_sendbuf_len,dec_count,max_out_len;

				MPIC_Irecv(large_recv_buffer, recv_enc_size, MPI_CHAR, rank_left,  0, comm_ptr,&request[0]);
								
				RAND_bytes(large_send_buffer, NONCE_SIZE);
				max_out_len = send_count*extent+ENC_MSG_TAG_SIZE; 
				if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer+NONCE_SIZE),
							   &ciphertext_sendbuf_len, max_out_len,
							   (&large_send_buffer[0]), NONCE_SIZE,
							   (char *)tmp_sendbuf, send_count*extent,
							   NULL, 0))
				{
					printf("Error in encryption: Red-Scat_2lvl  rank = %d\n",comm_ptr->rank);
					fflush(stdout);
				}
				
				MPIC_Isend(large_send_buffer, send_enc_size, MPI_CHAR, rank_right, 0, comm_ptr, &request[1], errflag);
				
				
                /* MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, send_count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag); */
						
                MPIC_Waitall(2, request, status, errflag);

				
				if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_recvbuf,
							   &dec_count, recv_count*extent,
							   (&large_recv_buffer[0]), NONCE_SIZE,
							   large_recv_buffer + NONCE_SIZE, (recv_count*extent + ENC_MSG_TAG_SIZE),
							   NULL, 0))
				{
					fprintf(stderr,"Error in Decryption Red-Scat_2lvl : Rank = %d --> dest =%d  recv_count = %d \n",comm_ptr->rank, rank_right, recv_count);				
				}
				
				
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, send_count, datatype,
                        buf, send_count, datatype);
            }
        }

        /* assume we send the max buffer count in each step,
         * this means that nread may exceed the max value,
         * but that will end the loop */
        nread += elem_per_buffer;
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */

    MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
 



int MPIR_Reduce_scatter_ring_2lvl_Original(const void* sendbuf, void* recvbuf,
        const int *recvcnts, MPI_Datatype datatype,
        MPI_Op op, MPID_Comm *comm_ptr,
        MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
	if (PRINT_FUN_NAME){
    printf("[red_scat_osu rank = %d   count = (array) %d   host = %s] Func: MPIR_Reduce_scatter_ring_2lvl_Original @ Red_scat_osu.c [Un-sec]\n",comm_ptr->rank,recvcnts,hostname);fflush(stdout);}
#endif		
	MPIR_TIMER_START(coll,reduce_scatter,ring_2lvl);
    int mpi_errno     = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size     = comm_ptr->local_size;

    if (comm_ptr->dev.ch.rank_list == NULL) {
    	MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
        return MPIR_Reduce_scatter_ring(
                            sendbuf, recvbuf, recvcnts, datatype,
                            op, comm_ptr, errflag);
    }

    MPIU_CHKLMEM_DECL(3);

    /* get extent */
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* get true extent and lower bound of datatype */
    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    /* compute size of temporary buffers */
    size_t mpi_buf_size = 1024 * 1024;
    if (mpi_buf_size < true_extent) {
        /* ensure we allocate a temporary buffer
         * big enough to hold at least one element */
        mpi_buf_size = (size_t) true_extent;
    }

    /* allocate buffers to work with */
    void* tmp_sendbuf;
    void* tmp_recvbuf;
    MPIU_CHKLMEM_MALLOC(tmp_sendbuf, char *, mpi_buf_size, mpi_errno, "tmp_sendbuf");
    MPIU_CHKLMEM_MALLOC(tmp_recvbuf, char *, mpi_buf_size, mpi_errno, "tmp_recvbuf");

    /* adjust pointers for lower bounds */
    tmp_sendbuf -= true_lb;
    tmp_recvbuf -= true_lb;

    /* lookup our index in the rank list */
    int rank_index = comm_ptr->dev.ch.rank_list_index;

    /* compute the left and right neighbor ranks in the rank_list */
    int index_left  = (comm_size + rank_index - 1) % comm_size;
    int index_right = (comm_size + rank_index + 1) % comm_size;
    int rank_left   = comm_ptr->dev.ch.rank_list[index_left];
    int rank_right  = comm_ptr->dev.ch.rank_list[index_right];

    /* Reduce_scatter */
    MPID_Request *request[2];
    MPI_Status    status[2];

    const void* input_buf = sendbuf;
    void* output_buf = recvbuf;

    /* if the caller gave us MPI_IN_PLACE, pull the input data from the
     * receive buffer instead of the senf buffer.  We do not bother copying
     * the input data to a temporary buffer, because it will have been
     * read by the time we overwrite it with the result. */
    if (sendbuf == MPI_IN_PLACE) {
        input_buf = recvbuf;
    }

    /* allocate memory for displacement array */
    int *displs       = NULL;
    MPIU_CHKLMEM_MALLOC(displs, int*, comm_size * sizeof(int), mpi_errno, "displs");

    /* total count of data for each rank */
    int total_count       = 0;
    int max_recv_count    = 0;
    int i;
    for (i = 0; i < comm_size; i++) {
        displs[i] = total_count;
        total_count += recvcnts[i];

        if (recvcnts[i] > max_recv_count) {
            max_recv_count = recvcnts[i];
        }
    }

    /* max number of elements a rank will receive */
    size_t max_elem_per_rank = (size_t) max_recv_count;

    /* compute number of whole elements that can fit in the buffer */
    size_t elem_per_buffer = mpi_buf_size / true_extent;

    /* process data in chunks of size elem_per_buffer */
    size_t nread = 0;
    while (nread < max_elem_per_rank) {
        /* keep track of send and recv counts left to process */
        /* execute a lap through the ring */
        int dist;
        for(dist = comm_size-1; dist >= 0; dist--) {
            /* compute rank of process whose data we're sending and rank
             * of process whose data we're receiving in this step */
            int send_index = (rank_index + dist) % comm_size;
            int recv_index = (rank_index + (dist - 1)) % comm_size;
            int send_rank  = comm_ptr->dev.ch.rank_list[send_index];
            int recv_rank  = comm_ptr->dev.ch.rank_list[recv_index];

            /* compute offset into input buffer to pull data for this chunk */
            unsigned long elem_offset = displs[send_rank] + nread;
            const char* buf = (const char*)input_buf + elem_offset * extent;

            /* compute the number of elements we're sending and receiving */
            int send_count = recvcnts[send_rank] - nread;
            int recv_count = recvcnts[recv_rank] - nread;

            if (send_count < 0) send_count = 0;
            if (recv_count < 0) recv_count = 0;

            if (send_count > elem_per_buffer) send_count = elem_per_buffer;
            if (recv_count > elem_per_buffer) recv_count = elem_per_buffer;

            /* copy next set of bytes for this chunk from input buffer into sendbuf */
            MPIR_Localcopy(buf, send_count, datatype,
                    tmp_sendbuf, send_count, datatype);

            /* merge the blocks via reduce operation */
            if (dist < comm_size-1) {
                MPIR_Reduce_local_impl(tmp_recvbuf, tmp_sendbuf, send_count, datatype, op);
            }

            if (dist > 0) {
                /* exchange data with neighbors */
                MPIC_Irecv(tmp_recvbuf, recv_count, datatype, rank_left,  0, comm_ptr,
                        &request[0]);
                MPIC_Isend(tmp_sendbuf, send_count, datatype, rank_right, 0, comm_ptr,
                        &request[1], errflag);
                MPIC_Waitall(2, request, status, errflag);
            } else {
                /* write the result to the ouput buffer */
                char* buf = output_buf + nread * extent;
                MPIR_Localcopy(tmp_sendbuf, send_count, datatype,
                        buf, send_count, datatype);
            }
        }

        /* assume we send the max buffer count in each step,
         * this means that nread may exceed the max value,
         * but that will end the loop */
        nread += elem_per_buffer;
    }

    /* bump pointers back to start of buffers for free calls */
    tmp_sendbuf += true_lb;
    tmp_recvbuf += true_lb;

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    /* --BEGIN ERROR HANDLING-- */
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    /* --END ERROR HANDLING-- */

    MPIR_TIMER_END(coll,reduce_scatter,ring_2lvl);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
 


/************ ENd *************/

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_non_comm_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Reduce_scatter_non_comm_MV2(const void *sendbuf,
											void *recvbuf,
											const int *recvcnts,
											MPI_Datatype datatype,
											MPI_Op op,
											MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
	MPIR_TIMER_START(coll,reduce_scatter,non_comm);
	int rank, comm_size, i;
	MPI_Aint extent, true_extent, true_lb;
	int *disps;
	void *tmp_recvbuf, *tmp_results;
	int mpi_errno = MPI_SUCCESS;
	int mpi_errno_ret = MPI_SUCCESS;
	int dis[2], blklens[2], total_count, dst;
	int mask, dst_tree_root, my_tree_root, j, k;
	int pof2, received;
	MPI_Datatype sendtype, recvtype;
	int nprocs_completed, tmp_mask, tree_root, is_commutative;
	MPI_User_function *uop;
	MPID_Op *op_ptr;
	MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
	int is_cxx_uop = 0;
#endif
	MPIU_CHKLMEM_DECL(5);

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_reduce_scatter_non_comm, 1);

	comm_size = comm_ptr->local_size;
	rank = comm_ptr->rank;

	/* set op_errno to 0. stored in perthread structure */
	MPIU_THREADPRIV_GET;
	MPIU_THREADPRIV_FIELD(op_errno) = 0;

	MPID_Datatype_get_extent_macro(datatype, extent);
	MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

	if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
		is_commutative = 1;
		/* get the function by indexing into the op table */
		uop = MPIR_Op_table[op % 16 - 1];
	} else {
		MPID_Op_get_ptr(op, op_ptr);
		if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
			is_commutative = 0;
		else
			is_commutative = 1;

#ifdef HAVE_CXX_BINDING
		if (op_ptr->language == MPID_LANG_CXX) {
			uop = (MPI_User_function *) op_ptr->function.c_function;
			is_cxx_uop = 1;
		} else
#endif
		if (op_ptr->language == MPID_LANG_C)
			uop = (MPI_User_function *) op_ptr->function.c_function;
		else
			uop = (MPI_User_function *) op_ptr->function.f77_function;
	}

	MPIU_CHKLMEM_MALLOC(disps, int *, comm_size * sizeof (int), mpi_errno,
						"disps");

	total_count = 0;
	for (i = 0; i < comm_size; i++) {
		disps[i] = total_count;
		total_count += recvcnts[i];
	}

	if (total_count == 0) {
		goto fn_exit;
	}

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

	MPIU_Ensure_Aint_fits_in_pointer(total_count *
									 MPIR_MAX(true_extent, extent));

	if (!is_commutative) {
		int is_block_regular = 1;
		for (i = 0; i < (comm_size - 1); ++i) {
			if (recvcnts[i] != recvcnts[i + 1]) {
				is_block_regular = 0;
				break;
			}
		}

		/* slightly retask pof2 to mean pof2 equal or greater, not always greater as it is above */
		pof2 = 1;
		while (pof2 < comm_size)
			pof2 <<= 1;

		if (pof2 == comm_size && is_block_regular) {
			/* noncommutative, pof2 size, and block regular */
			mpi_errno =
				MPIR_Reduce_scatter_noncomm_MV2(sendbuf, recvbuf, recvcnts,
												datatype, op, comm_ptr,
												errflag);
			if (mpi_errno) {
				/* for communication errors, just record the error but continue */
				*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
				MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
				MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
			}
		} else {
			/* noncommutative and (non-pof2 or block irregular), use recursive doubling. */

			/* need to allocate temporary buffer to receive incoming data */
			MPIU_CHKLMEM_MALLOC(tmp_recvbuf, void *,
								total_count * (MPIR_MAX(true_extent, extent)),
								mpi_errno, "tmp_recvbuf");
			/* adjust for potential negative lower bound in datatype */
			tmp_recvbuf = (void *) ((char *) tmp_recvbuf - true_lb);

			/* need to allocate another temporary buffer to accumulate
			   results */
			MPIU_CHKLMEM_MALLOC(tmp_results, void *,
								total_count * (MPIR_MAX(true_extent, extent)),
								mpi_errno, "tmp_results");
			/* adjust for potential negative lower bound in datatype */
			tmp_results = (void *) ((char *) tmp_results - true_lb);

			/* copy sendbuf into tmp_results */
			if (sendbuf != MPI_IN_PLACE)
				mpi_errno = MPIR_Localcopy(sendbuf, total_count, datatype,
										   tmp_results, total_count, datatype);
			else
				mpi_errno = MPIR_Localcopy(recvbuf, total_count, datatype,
										   tmp_results, total_count, datatype);

			if (mpi_errno)
				MPIR_ERR_POP(mpi_errno);

			mask = 0x1;
			i = 0;
			while (mask < comm_size) {
				dst = rank ^ mask;

				dst_tree_root = dst >> i;
				dst_tree_root <<= i;

				my_tree_root = rank >> i;
				my_tree_root <<= i;

				/* At step 1, processes exchange (n-n/p) amount of
				   data; at step 2, (n-2n/p) amount of data; at step 3, (n-4n/p)
				   amount of data, and so forth. We use derived datatypes for this.

				   At each step, a process does not need to send data
				   indexed from my_tree_root to
				   my_tree_root+mask-1. Similarly, a process won't receive
				   data indexed from dst_tree_root to dst_tree_root+mask-1. */

				/* calculate sendtype */
				blklens[0] = blklens[1] = 0;
				for (j = 0; j < my_tree_root; j++)
					blklens[0] += recvcnts[j];
				for (j = my_tree_root + mask; j < comm_size; j++)
					blklens[1] += recvcnts[j];

				dis[0] = 0;
				dis[1] = blklens[0];
				for (j = my_tree_root;
					 (j < my_tree_root + mask) && (j < comm_size); j++)
					dis[1] += recvcnts[j];

				mpi_errno =
					MPIR_Type_indexed_impl(2, blklens, dis, datatype,
										   &sendtype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);

				mpi_errno = MPIR_Type_commit_impl(&sendtype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);

				/* calculate recvtype */
				blklens[0] = blklens[1] = 0;
				for (j = 0; j < dst_tree_root && j < comm_size; j++)
					blklens[0] += recvcnts[j];
				for (j = dst_tree_root + mask; j < comm_size; j++)
					blklens[1] += recvcnts[j];

				dis[0] = 0;
				dis[1] = blklens[0];
				for (j = dst_tree_root;
					 (j < dst_tree_root + mask) && (j < comm_size); j++)
					dis[1] += recvcnts[j];

				mpi_errno =
					MPIR_Type_indexed_impl(2, blklens, dis, datatype,
										   &recvtype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);

				mpi_errno = MPIR_Type_commit_impl(&recvtype);
				if (mpi_errno)
					MPIR_ERR_POP(mpi_errno);

				received = 0;
				if (dst < comm_size) {
					/* tmp_results contains data to be sent in each step. Data is
					   received in tmp_recvbuf and then accumulated into
					   tmp_results. accumulation is done later below.   */
					
					MPIR_PVAR_INC(reduce_scatter, non_comm, send, 1, sendtype);
					MPIR_PVAR_INC(reduce_scatter, non_comm, recv, 1, recvtype);
					mpi_errno = MPIC_Sendrecv(tmp_results, 1, sendtype, dst,
												 MPIR_REDUCE_SCATTER_TAG,
												 tmp_recvbuf, 1, recvtype, dst,
												 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
												 MPI_STATUS_IGNORE, errflag);
					received = 1;
					if (mpi_errno) {
						/* for communication errors, just record the error but continue */
						*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
						MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
						MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
					}
				}

				/* if some processes in this process's subtree in this step
				   did not have any destination process to communicate with
				   because of non-power-of-two, we need to send them the
				   result. We use a logarithmic recursive-halfing algorithm
				   for this. */

				if (dst_tree_root + mask > comm_size) {
					nprocs_completed = comm_size - my_tree_root - mask;
					/* nprocs_completed is the number of processes in this
					   subtree that have all the data. Send data to others
					   in a tree fashion. First find root of current tree
					   that is being divided into two. k is the number of
					   least-significant bits in this process's rank that
					   must be zeroed out to find the rank of the root */
					j = mask;
					k = 0;
					while (j) {
						j >>= 1;
						k++;
					}
					k--;

					tmp_mask = mask >> 1;
					while (tmp_mask) {
						dst = rank ^ tmp_mask;

						tree_root = rank >> k;
						tree_root <<= k;

						/* send only if this proc has data and destination
						   doesn't have data. at any step, multiple processes
						   can send if they have the data */
						if ((dst > rank) &&
							(rank < tree_root + nprocs_completed)
							&& (dst >= tree_root + nprocs_completed)) {
							/* send the current result */
							MPIR_PVAR_INC(reduce_scatter, non_comm, send, 1, recvtype);
							mpi_errno = MPIC_Send(tmp_recvbuf, 1, recvtype,
													 dst,
													 MPIR_REDUCE_SCATTER_TAG,
													 comm_ptr, errflag);
							if (mpi_errno) {
								/* for communication errors, just record the error but continue */
								*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
								MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER,
											 "**fail");
								MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
							}
						}
						/* recv only if this proc. doesn't have data and sender
						   has data */
						else if ((dst < rank) &&
								 (dst < tree_root + nprocs_completed) &&
								 (rank >= tree_root + nprocs_completed)) {
							MPIR_PVAR_INC(reduce_scatter, non_comm, recv, 1, recvtype);
							mpi_errno =
								MPIC_Recv(tmp_recvbuf, 1, recvtype, dst,
											 MPIR_REDUCE_SCATTER_TAG, comm_ptr,
											 MPI_STATUS_IGNORE, errflag);
							received = 1;
							if (mpi_errno) {
								/* for communication errors, just record the error but continue */
								*errflag = MPIR_ERR_GET_CLASS(mpi_errno);
								MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER,
											 "**fail");
								MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
							}
						}
						tmp_mask >>= 1;
						k--;
					}
				}

				/* The following reduction is done here instead of after 
				   the MPIC_Sendrecv or MPIC_Recv above. This is
				   because to do it above, in the noncommutative 
				   case, we would need an extra temp buffer so as not to
				   overwrite temp_recvbuf, because temp_recvbuf may have
				   to be communicated to other processes in the
				   non-power-of-two case. To avoid that extra allocation,
				   we do the reduce here. */
				if (received) {
					if (is_commutative || (dst_tree_root < my_tree_root)) {
#ifdef HAVE_CXX_BINDING
						if (is_cxx_uop) {
							(*MPIR_Process.cxx_call_op_fn) (tmp_recvbuf,
															tmp_results,
															blklens[0],
															datatype, uop);
							(*MPIR_Process.
							 cxx_call_op_fn) (((char *) tmp_recvbuf +
											   dis[1] * extent),
											  ((char *) tmp_results +
											   dis[1] * extent), blklens[1],
											  datatype, uop);
						} else
#endif
						{
							(*uop) (tmp_recvbuf, tmp_results, &blklens[0],
									&datatype);
							(*uop) (((char *) tmp_recvbuf + dis[1] * extent),
									((char *) tmp_results + dis[1] * extent),
									&blklens[1], &datatype);
						}
					} else {
#ifdef HAVE_CXX_BINDING
						if (is_cxx_uop) {
							(*MPIR_Process.cxx_call_op_fn) (tmp_results,
															tmp_recvbuf,
															blklens[0],
															datatype, uop);
							(*MPIR_Process.
							 cxx_call_op_fn) (((char *) tmp_results +
											   dis[1] * extent),
											  ((char *) tmp_recvbuf +
											   dis[1] * extent), blklens[1],
											  datatype, uop);
						} else
#endif
						{
							(*uop) (tmp_results, tmp_recvbuf, &blklens[0],
									&datatype);
							(*uop) (((char *) tmp_results + dis[1] * extent),
									((char *) tmp_recvbuf + dis[1] * extent),
									&blklens[1], &datatype);
						}
						/* copy result back into tmp_results */
						mpi_errno = MPIR_Localcopy(tmp_recvbuf, 1, recvtype,
												   tmp_results, 1, recvtype);
						if (mpi_errno)
							MPIR_ERR_POP(mpi_errno);
					}
				}

				MPIR_Type_free_impl(&sendtype);
				MPIR_Type_free_impl(&recvtype);

				mask <<= 1;
				i++;
			}

			/* now copy final results from tmp_results to recvbuf */
			mpi_errno =
				MPIR_Localcopy(((char *) tmp_results + disps[rank] * extent),
							   recvcnts[rank], datatype, recvbuf,
							   recvcnts[rank], datatype);
			if (mpi_errno)
				MPIR_ERR_POP(mpi_errno);
		}
	}

  fn_exit:
	MPIU_CHKLMEM_FREEALL();

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

	if (MPIU_THREADPRIV_FIELD(op_errno))
		mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);

	if (mpi_errno_ret)
		mpi_errno = mpi_errno_ret;
	else if (*errflag)
		MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");

	MPIR_TIMER_END(coll,reduce_scatter,non_comm);
	return mpi_errno;
  fn_fail:
	goto fn_exit;
}

/* MPIR_Reduce_Scatter performs an reduce_scatter using point-to-point
   messages.  This is intended to be used by device-specific
   implementations of reduce_scatter.  In all other cases
   MPIR_Reduce_Scatter_impl should be used. */

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_scatter_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_scatter_MV2(const void *sendbuf, void *recvbuf, const int *recvcnts,
							MPI_Datatype datatype, MPI_Op op,
							MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
	int mpi_errno = MPI_SUCCESS;
	int mpi_errno_ret = MPI_SUCCESS;
	int i = 0, comm_size = comm_ptr->local_size, total_count = 0, type_size =
		0, nbytes = 0;
    int range = 0;
    int range_threshold = 0;
	int is_commutative = 0;
	MPID_Op *op_ptr = NULL;
	int *disps = NULL;
#ifdef _ENABLE_CUDA_
    int recv_mem_type = 0, send_mem_type = 0, stride = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;
    int rank = comm_ptr->rank;
#endif
	MPIU_THREADPRIV_DECL;
	MPIU_THREADPRIV_GET;
	MPIU_THREADPRIV_FIELD(op_errno) = 0;
	MPIU_CHKLMEM_DECL(1);

	MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

	if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
		is_commutative = 1;
	} else {
		MPID_Op_get_ptr(op, op_ptr);
		if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE)
			is_commutative = 0;
		else
			is_commutative = 1;
	}

	MPIU_CHKLMEM_MALLOC(disps, int *, comm_size * sizeof (int), mpi_errno,
						"disps");

	for (i = 0; i < comm_size; i++) {
		disps[i] = total_count;
		total_count += recvcnts[i];
	}

	if (total_count == 0) {
		goto fn_exit;
	}

	MPID_Datatype_get_size_macro(datatype, type_size);
	nbytes = total_count * type_size;

#ifdef _ENABLE_CUDA_
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = total_count * MPIR_MAX(extent, true_extent);

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(recvbuf);
       if ( sendbuf != MPI_IN_PLACE ){
           send_mem_type = is_device_buffer(sendbuf);
       }
    }
    if (rdma_enable_cuda && send_mem_type) {
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)sendbuf, 
                            stride, 
                            cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if (rdma_enable_cuda && recv_mem_type) {
        /* recvbuf will be treated as sendbuf if sendbuf is MPI_IN_PLACE */
        if (sendbuf == MPI_IN_PLACE) {
            recv_host_buf = (char*) MPIU_Malloc(stride);
            MPIU_Memcpy_CUDA((void *)recv_host_buf,
                                (void *)recvbuf,
                                stride,
                                cudaMemcpyDeviceToHost);
        } else {
            recv_host_buf = (char*) MPIU_Malloc(recvcnts[rank]*type_size);
        }
        recvbuf = recv_host_buf;
    }
#endif

	if (is_commutative) {

        if (mv2_red_scat_thresholds_table[0].numproc != 1 &&
                ((comm_ptr->dev.ch.allgather_comm_ok != 0 &&
                  comm_ptr->dev.ch.is_blocked == 0 &&
                  mv2_redscat_cyclic_algo_threshold <= nbytes) ||
                 mv2_red_scat_ring_algo_threshold <= nbytes)) {
            /* make sure that user has not forced any algorithm and 
             * then for large messages, use ring algorithm. Also, for cyclic
             * hostfile, use ring  */
            mpi_errno =  MPIR_Reduce_scatter_ring_2lvl(sendbuf, recvbuf, recvcnts,
                    datatype, op, comm_ptr, errflag);
            goto fn_exit;
        }

        /* Search for the corresponding system size inside the tuning table */
        while ((range < (mv2_size_red_scat_tuning_table - 1)) &&
               (comm_size > mv2_red_scat_thresholds_table[range].numproc)) {
            range++;
        }
        /* Search for corresponding inter-leader function */
        while ((range_threshold < (mv2_red_scat_thresholds_table[range].size_inter_table - 1))
               && (nbytes >
                   mv2_red_scat_thresholds_table[range].inter_leader[range_threshold].max)
               && (mv2_red_scat_thresholds_table[range].inter_leader[range_threshold].max !=
                   -1)) {
            range_threshold++;
        }
    
        /* Set inter-leader pt */
        MV2_Red_scat_function =
                              mv2_red_scat_thresholds_table[range].inter_leader[range_threshold].
                              MV2_pt_Red_scat_function;

		mpi_errno = MV2_Red_scat_function(sendbuf, recvbuf,
                                          recvcnts, datatype,
                                          op, comm_ptr,
                                          errflag);
	} else {
        mpi_errno = MPIR_Reduce_scatter_non_comm_MV2(sendbuf, recvbuf,
                                                     recvcnts, datatype,
                                                     op, comm_ptr, errflag);
	}
	if (mpi_errno) { 
		MPIR_ERR_POP(mpi_errno);
    } 

fn_exit:
#ifdef _ENABLE_CUDA_
    if (rdma_enable_cuda && recv_mem_type==1) {
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
                           (void *)recv_host_buf, 
                            recvcnts[rank]*type_size, 
                            cudaMemcpyHostToDevice);
    }
    if (rdma_enable_cuda && recv_mem_type) {
        if (recv_host_buf) {
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if (rdma_enable_cuda && send_mem_type) {
        if (send_host_buf) {
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif

	MPIU_CHKLMEM_FREEALL();

	/* check if multiple threads are calling this collective function */
	MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

	if (MPIU_THREADPRIV_FIELD(op_errno))
		mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);

	if (mpi_errno_ret)
		mpi_errno = mpi_errno_ret;
	else if (*errflag)
		MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    return mpi_errno;
  fn_fail:
	goto fn_exit;
}
