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

#include "iallreduce_tuning.h"

#if defined (_SHARP_SUPPORT_)
#include "api/sharp_coll.h"
#include "ibv_sharp.h"
extern int mv2_sharp_tuned_msg_size;
#endif

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_iallreduce_sharp);

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)

int (*MV2_Iallreduce_function) (const void *sendbuf, void *recvbuf, int count,
				    MPI_Datatype datatype, MPI_Op op,
				    MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

int (*MV2_Iallreduce_intra_node_function) (const void *sendbuf, void *recvbuf, int count,
				    MPI_Datatype datatype, MPI_Op op,
				    MPID_Comm *comm_ptr, MPID_Sched_t s) = NULL;

#if defined (_SHARP_SUPPORT_)
#undef FCNAME
#define FCNAME "MPIR_Sharp_Iallreduce_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Sharp_Iallreduce_MV2 (const void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPID_Comm * comm_ptr, int *errflag, MPID_Request **req)
{
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_iallreduce_sharp, 1);
    int mpi_errno = MPI_SUCCESS;
    int is_contig = 0;
    int is_inplace = 0;
    struct sharp_coll_reduce_spec reduce_spec;
    struct sharp_reduce_datatyepe_size * dt_size = NULL; 
    struct sharp_coll_comm * sharp_comm = NULL;
    MPID_Datatype * dtp = NULL;
    SHARP_REQ_HANDLE * sharp_req = NULL;
    
    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN) {
        is_contig = 1;
    } else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    mv2_get_sharp_datatype(datatype, &dt_size);
    reduce_spec.dtype = dt_size->sharp_data_type;

    if (reduce_spec.dtype == SHARP_DTYPE_NULL) {
        mpi_errno = SHARP_COLL_ENOT_SUPP;
        goto fn_fail;
    }
 
    reduce_spec.op = mv2_get_sharp_reduce_op(op);;     
    if (reduce_spec.op == SHARP_OP_NULL) {
        mpi_errno = SHARP_COLL_ENOT_SUPP;
        goto fn_fail;
    }

    if (is_contig == 1) {
        reduce_spec.sbuf_desc.buffer.length = count * dt_size->size;
        if (sendbuf != MPI_IN_PLACE) {
            reduce_spec.sbuf_desc.buffer.ptr    = (void *)sendbuf;
        } else {
            is_inplace = 1;
            reduce_spec.sbuf_desc.buffer.ptr    = MPIU_Malloc(reduce_spec.sbuf_desc.buffer.length);
            MPIU_Memcpy(reduce_spec.sbuf_desc.buffer.ptr, recvbuf, reduce_spec.sbuf_desc.buffer.length);
        }
        reduce_spec.sbuf_desc.type          = SHARP_DATA_BUFFER;
        reduce_spec.sbuf_desc.buffer.mem_handle = NULL;
        reduce_spec.rbuf_desc.buffer.ptr    = recvbuf;
        reduce_spec.rbuf_desc.buffer.length = count * dt_size->size;
        reduce_spec.rbuf_desc.type          = SHARP_DATA_BUFFER;
        reduce_spec.rbuf_desc.buffer.mem_handle = NULL;    
    } else {
        /* NOT implementated in Sharp */
        mpi_errno = SHARP_COLL_ENOT_SUPP;
        goto fn_fail;
    }

    reduce_spec.length = count;     
    sharp_comm = ((sharp_info_t *)comm_ptr->dev.ch.sharp_coll_info)->sharp_comm_module->sharp_coll_comm;
    mpi_errno = sharp_coll_do_allreduce_nb(sharp_comm, &reduce_spec, &sharp_req);
    if (mpi_errno != SHARP_COLL_SUCCESS) {
        goto fn_fail;
    }
    /* now create and populate the request */
    *req = MPID_Request_create();
    if(*req == NULL) { 
        mpi_errno = SHARP_COLL_ENOT_SUPP;
        goto fn_fail;
    }
    (*req)->sharp_req = sharp_req;
    (*req)->kind = MPID_COLL_REQUEST;
    mpi_errno = MPI_SUCCESS;

fn_exit:
    MPIU_Free(dt_size);
    if (is_inplace)
        MPIU_Free(reduce_spec.sbuf_desc.buffer.ptr);
    return (mpi_errno);

fn_fail:
    PRINT_DEBUG(DEBUG_Sharp_verbose, "Continue without SHArP: %s \n", sharp_coll_strerror(mpi_errno));
    mpi_errno = MPI_ERR_INTERN;
    goto fn_exit;
}

#endif /* end of defined (_SHARP_SUPPORT_) */


#undef FUNCNAME
#define FUNCNAME MPIR_Iallreduce_tune_helper_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Iallreduce_tune_helper_MV2(const void *sendbuf, void *recvbuf, int count,
				    MPI_Datatype datatype, MPI_Op op,
				    MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int is_homogeneous  ATTRIBUTE((unused)), pof2, comm_size;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous);

    comm_size = comm_ptr->local_size;
    pof2 = 1;
    while (pof2 <= comm_size) pof2 <<= 1;
    pof2 >>=1;
    
    if (MV2_Iallreduce_function == MPIR_Iallreduce_redscat_allgather) {
        if (count >= pof2) {
            mpi_errno = MV2_Iallreduce_function(sendbuf, recvbuf, count, datatype,
                                                op, comm_ptr, s);
        }
        else {
            mpi_errno = MPIR_Iallreduce_naive(sendbuf, recvbuf, count, datatype,
                                                op, comm_ptr, s);
        }
    }
    else {
        mpi_errno = MV2_Iallreduce_function(sendbuf, recvbuf, count, datatype,
					     op, comm_ptr, s);
    }

    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Iallreduce_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iallreduce_intra_MV2(const void *sendbuf, void *recvbuf, int count,
				    MPI_Datatype datatype, MPI_Op op,
				    MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size, is_homogeneous  ATTRIBUTE((unused));
    MPI_Aint sendtype_size, nbytes;
    
    int two_level_iallreduce = 1;
    int range = 0;
    int range_threshold = 0;
    int range_threshold_intra = 0;

    MPIU_Assert(comm_ptr->comm_kind == MPID_INTRACOMM);

    is_homogeneous = 1;
#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero)
        is_homogeneous = 0;
#endif
    MPIU_Assert(is_homogeneous); /* we don't handle the hetero case right now */
    comm_size = comm_ptr->local_size;
    MPID_Datatype_get_size_macro(datatype, sendtype_size);
    nbytes = count * sendtype_size;

    // Search for some parameters regardless of whether subsequent selected
    // algorithm is 2-level or not
    
    // Search for the corresponding system size inside the tuning table
    while ((range < (mv2_size_iallreduce_tuning_table - 1)) &&
           (comm_size > mv2_iallreduce_thresholds_table[range].numproc)) {
        range++;
    }
    
    // Search for corresponding inter-leader function
    while ((range_threshold < (mv2_iallreduce_thresholds_table[range].size_inter_table - 1))
           && (nbytes >
               mv2_iallreduce_thresholds_table[range].inter_leader[range_threshold].max)
           && (mv2_iallreduce_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
        range_threshold++;
    }

    // Search for corresponding intra-node function
    
    // Commenting this for the time being as none of
    // the algorithms are 2-level
    /*
    while ((range_threshold_intra <
            (mv2_iallreduce_thresholds_table[range].size_intra_table - 1))
           && (nbytes >
               mv2_iallreduce_thresholds_table[range].intra_node[range_threshold_intra].max)
           && (mv2_iallreduce_thresholds_table[range].intra_node[range_threshold_intra].max !=
               -1)) {
        range_threshold_intra++;
    }
    */

    MV2_Iallreduce_function =
        mv2_iallreduce_thresholds_table[range].inter_leader[range_threshold].
        MV2_pt_Iallreduce_function;

    MV2_Iallreduce_intra_node_function =
        mv2_iallreduce_thresholds_table[range].
        intra_node[range_threshold_intra].MV2_pt_Iallreduce_function;

    /* There are currently no two-level nb-allreduce functions hence
       setting to 0 by default */
    two_level_iallreduce = 
        mv2_iallreduce_thresholds_table[range].is_two_level_iallreduce[range_threshold]; 
    if (1 != two_level_iallreduce) {
        mpi_errno = MPIR_Iallreduce_tune_helper_MV2(sendbuf, recvbuf, count, datatype,
                                     op, comm_ptr, s);
    }
    else {
        /* Code path should not enter this with the current algorithms*/
    }

    return mpi_errno;
}
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */

#undef FUNCNAME
#define FUNCNAME MPIR_Iallreduce_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Iallreduce_MV2(const void *sendbuf, void *recvbuf, int count,
				    MPI_Datatype datatype, MPI_Op op,
				    MPID_Comm *comm_ptr, MPID_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;

    if (comm_ptr->comm_kind == MPID_INTRACOMM) {    
#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
      mpi_errno = MPIR_Iallreduce_intra_MV2(sendbuf, recvbuf, count, datatype,
					    op, comm_ptr, s);
#else
      mpi_errno = MPIR_Iallreduce_intra(sendbuf, recvbuf, count, datatype,
					op, comm_ptr, s);
#endif                          /*#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM) */
    }
    else {
      mpi_errno = MPIR_Iallreduce_inter(sendbuf, recvbuf, count, datatype,
					op, comm_ptr, s);
    }

    return mpi_errno;
}
