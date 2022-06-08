/* -*- Mode: C; c-basic-offset:4 ; -*- */         
/*
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
#include "coll_shmem.h"
#include "common_tuning.h"
#include "allreduce_tuning.h"
#include "bcast_tuning.h"
#if defined(CKPT)
#include <cr.h>
#endif

#if defined (_SHARP_SUPPORT_)
#include "api/sharp_coll.h"
#include "ibv_sharp.h"
extern int mv2_sharp_tuned_msg_size;
#endif

// unsigned char large_send_buffer[150401232];
// unsigned char large_recv_buffer[150401232];

unsigned char large_send_buffer[COMMON_LARGE_SEND_BUFFER_SIZE];
unsigned char large_recv_buffer[COMMON_LARGE_RECV_BUFFER_SIZE];
unsigned char tmp_enc_buf[COMMON_LARGE_RECV_BUFFER_SIZE];

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_subcomm);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_sharp);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_shm_rd);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_shm_rs);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_shm_intra);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_intra_p2p);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_2lvl);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_shmem);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_mcast);
MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(MV2, mv2_coll_timer_allreduce_reduce_scatter_allgather_colls);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_sharp);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_shm_rd);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_shm_rs);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_shm_intra);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_intra_p2p);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_2lvl);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_shmem);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_mcast);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_reduce_scatter_allgather_colls);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_subcomm);

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rd_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rs_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rd_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rs_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rd_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rs_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rd_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_pt2pt_rs_count_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_bytes_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_bytes_recv);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_count_send);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_coll_allreduce_count_recv);


int (*MV2_Allreduce_function)(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)=NULL;


int (*MV2_Allreduce_intra_function)(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)=NULL;

static int (*MPIR_Rank_list_mapper)(MPID_Comm *, int)=NULL;

#if defined (_SHARP_SUPPORT_)
#undef FCNAME
#define FCNAME "MPIR_Sharp_Allreduce_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Sharp_Allreduce_MV2 (const void *sendbuf, void *recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPID_Comm * comm_ptr, int *errflag)
{
    MPIR_TIMER_START(coll,allreduce,sharp);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_sharp, 1);

    int mpi_errno = MPI_SUCCESS;
    int is_contig = 0;
    int is_inplace = 0;
    struct sharp_coll_reduce_spec reduce_spec;
    struct sharp_reduce_datatyepe_size * dt_size = NULL; 
    struct sharp_coll_comm * sharp_comm = NULL;
    MPID_Datatype * dtp = NULL;
    
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
    mpi_errno = sharp_coll_do_allreduce(sharp_comm, &reduce_spec);
    if (mpi_errno != SHARP_COLL_SUCCESS) {
        goto fn_fail;
    }
    
    mpi_errno = MPI_SUCCESS;

fn_exit:
    MPIU_Free(dt_size);
    if (is_inplace)
        MPIU_Free(reduce_spec.sbuf_desc.buffer.ptr);

    MPIR_TIMER_END(coll,allreduce,sharp);  
    return (mpi_errno);

fn_fail:
    PRINT_DEBUG(DEBUG_Sharp_verbose, "Continue without SHArP: %s \n", sharp_coll_strerror(mpi_errno));
    mpi_errno = MPI_ERR_INTERN;
    goto fn_exit;
}

#endif /* end of defined (_SHARP_SUPPORT_) */


/* This is the default implementation of allreduce. The algorithm is:
   
   Algorithm: MPI_Allreduce

   For the heterogeneous case, we call MPI_Reduce followed by MPI_Bcast
   in order to meet the requirement that all processes must have the
   same result. For the homogeneous case, we use the following algorithms.

   For long messages and for builtin ops and if count >= pof2 (where
   pof2 is the nearest power-of-two less than or equal to the number
   of processes), we use Rabenseifner's algorithm (see 
   http://www.hlrs.de/organization/par/services/models/mpi/myreduce.html ).
   This algorithm implements the allreduce in two steps: first a
   reduce-scatter, followed by an allgather. A recursive-halving
   algorithm (beginning with processes that are distance 1 apart) is
   used for the reduce-scatter, and a recursive doubling 
   algorithm is used for the allgather. The non-power-of-two case is
   handled by dropping to the nearest lower power-of-two: the first
   few even-numbered processes send their data to their right neighbors
   (rank+1), and the reduce-scatter and allgather happen among the remaining
   power-of-two processes. At the end, the first few even-numbered
   processes get the result from their right neighbors.

   For the power-of-two case, the cost for the reduce-scatter is 
   lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma. The cost for the
   allgather lgp.alpha + n.((p-1)/p).beta. Therefore, the
   total cost is:
   Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma

   For the non-power-of-two case, 
   Cost = (2.floor(lgp)+2).alpha + (2.((p-1)/p) + 2).n.beta + n.(1+(p-1)/p).gamma

   
   For short messages, for user-defined ops, and for count < pof2 
   we use a recursive doubling algorithm (similar to the one in
   MPI_Allgather). We use this algorithm in the case of user-defined ops
   because in this case derived datatypes are allowed, and the user
   could pass basic datatypes on one process and derived on another as
   long as the type maps are the same. Breaking up derived datatypes
   to do the reduce-scatter is tricky. 

   Cost = lgp.alpha + n.lgp.beta + n.lgp.gamma

   Possible improvements: 

   End Algorithm: MPI_Allreduce
*/

int MPIR_Allreduce_mcst_reduce_two_level_helper_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{ 
    return 0;
}

int MPIR_Allreduce_mcst_reduce_redscat_gather_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    return 0;
}

/* not declared static because a machine-specific function may call this one 
   in some cases */
/* This is flat p2p recursive-doubling allreduce */


#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rd_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_pt2pt_rd_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    if (security_approach==2005)
        return MPIR_Allreduce_pt2pt_rd_MV2_NodeAware(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else if (security_approach==2001) 
        return MPIR_Allreduce_pt2pt_rd_MV2_Naive(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_pt2pt_rd_MV2_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}




#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rd_MV2_NodeAware"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)



int MPIR_Allreduce_pt2pt_rd_MV2_NodeAware(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_pt2pt_rd_MV2_NodeAware  [COLL_NAIVE_AESGCM_PLUS]\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_rd);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rd, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);
	
	

   // EVP_AEAD_CTX *local_ctx = global_openmp_ctx;
    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);
	

    int pos, max_out_len, enc_recv_sz;
    unsigned long ciphertext_sendbuf_len, dec_count;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **, 2*sizeof(MPID_Request*),
                        mpi_errno, "reqarray");
    
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, 2*sizeof(MPI_Status),
                        mpi_errno, "starray");

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rd);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    
    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
           if (op_ptr->language == MPID_LANG_C) {
               uop = (MPI_User_function *) op_ptr->function.c_function;
           } else {
               uop = (MPI_User_function *) op_ptr->function.f77_function;
           }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;


    MPID_Node_id_t my_node_id = -1;
	MPID_Node_id_t remote_node_id = -1;
	
	MPID_Get_node_id(comm_ptr, rank, &my_node_id);

    unsigned int enc_chunk_size = -1;
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
    
    int context_id;
    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
    MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */
	   
	
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);

            MPID_Get_node_id(comm_ptr, rank + 1, &remote_node_id);

            /*  mpi_errno = MPIC_Send(recvbuf, count,
                                     datatype, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);*/            
            
            if ((my_node_id == remote_node_id)){ // || (init_phase)){
                    mpi_errno = MPIC_Send(recvbuf, count, datatype, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            } else {

                     mpi_errno = MPIC_ENC_NAIVE_Send(recvbuf, count, datatype, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            }
            
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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);

            MPID_Get_node_id(comm_ptr, rank - 1, &remote_node_id);

            if ((my_node_id == remote_node_id)){ // || (init_phase)){

                mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            } else {

                mpi_errno = MPIC_ENC_NAIVE_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag); 
                                
            }

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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
               (*uop) (tmp_buf, recvbuf, &count, &datatype);

               /* change the rank */
               newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */	

	if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
               into tmp_buf */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
			
			MPID_Get_node_id(comm_ptr, dst, &remote_node_id);
			
			if ((my_node_id == remote_node_id)){ // || (init_phase)){

			    mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
											 dst, MPIR_ALLREDUCE_TAG,
											 tmp_buf, count, datatype, dst,
											 MPIR_ALLREDUCE_TAG, comm_ptr,
											 MPI_STATUS_IGNORE, errflag);

#if ENABLE_SECURE_MPI_DEBUG
				
				if (ENABLE_SECURE_DEBUG)				
					fprintf(stderr,COLOR_CYAN"RD_NP_MPIC_Sendrecv Rank = %d --> dest =%d  [%d,%d] [Plain]"COLOR_RESET"\n",rank, dst,my_node_id,remote_node_id);
#endif
			} else {
			
			   /* mpi_errno = MPIC_Irecv(tmp_buf, count, datatype, dst,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &reqarray[0]);*/
				mpi_errno = MPIC_Irecv(large_recv_buffer, count*extent+NONCE_SIZE+ENC_MSG_TAG_SIZE, MPI_CHAR, dst,
										  MPIR_ALLREDUCE_TAG, comm_ptr,
										  &reqarray[0]);                          
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);
				
				RAND_bytes(large_send_buffer, NONCE_SIZE);
				max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
				if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
									   &ciphertext_sendbuf_len, max_out_len,
									   (large_send_buffer), NONCE_SIZE,
									   (char *)recvbuf, count*extent,
									   NULL, 0))
					{
						printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG)
							fprintf(stderr,COLOR_GREEN"RD_NP ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, dst);
					}
#endif
				
				mpi_errno = MPIC_Isend(large_send_buffer, count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, dst,
											  MPIR_ALLREDUCE_TAG, comm_ptr,
											  &reqarray[1], errflag);

				
				mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); 

				 if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
									   &dec_count, (count*extent),
									   (large_recv_buffer), NONCE_SIZE,
									   (large_recv_buffer + NONCE_SIZE), (unsigned long)((count*extent) + ENC_MSG_TAG_SIZE),
									   NULL, 0))
					{
						printf("RD_NP Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, dst);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"RD_NP MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
					fflush(stdout);
				}
#endif          
				mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag);                                                           
			}
			
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            //printf("return from sendrecv\n");fflush(stdout);
            /* tmp_buf contains data received in this step.
               recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif
            } else {
                /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif

                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                                   recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                            MPI_ERR_OTHER, "**fail");
                }
                    mask <<= 1;
            }
        }

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);

            MPID_Get_node_id(comm_ptr, rank - 1, &remote_node_id);

            if ((my_node_id == remote_node_id)){ // || (init_phase)){
                    mpi_errno = MPIC_Send(recvbuf, count, datatype, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            } else {

                      mpi_errno = MPIC_ENC_NAIVE_Send(recvbuf, count, datatype, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);      
            }

        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);

            MPID_Get_node_id(comm_ptr, rank + 1, &remote_node_id);

            if ((my_node_id == remote_node_id)){ // || (init_phase)){

                mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            } else {

                mpi_errno = MPIC_ENC_NAIVE_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);     
            }                                           
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
	
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rd);

    return (mpi_errno);

  fn_fail:
    goto fn_exit;


}


#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rd_MV2_Naive"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)


int MPIR_Allreduce_pt2pt_rd_MV2_Naive(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
	 printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_rd_MV2_Naive  count = %d  [COLL_NAIVE_AESGCM]\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_rd);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rd, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

   // EVP_AEAD_CTX *local_ctx = global_openmp_ctx;
      
    int pos, max_out_len, enc_recv_sz;
    unsigned long ciphertext_sendbuf_len, dec_count;
    unsigned int temp_data;
    MPID_Request **reqarray;
    MPI_Status *starray;
    MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **, 4*sizeof(MPID_Request*),
                        mpi_errno, "reqarray");
    
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, 4*sizeof(MPI_Status),
                        mpi_errno, "starray");

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rd);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    
    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
           if (op_ptr->language == MPID_LANG_C) {
               uop = (MPI_User_function *) op_ptr->function.c_function;
           } else {
               uop = (MPI_User_function *) op_ptr->function.f77_function;
           }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;
    
    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            //RAND_bytes(large_send_buffer, NONCE_SIZE);
             temp_data = count * extent;
            large_send_buffer[0] = (temp_data >> 24) & 0xFF;
            large_send_buffer[1] = (temp_data >> 16) & 0xFF;
            large_send_buffer[2] = (temp_data >> 8) & 0xFF;
            large_send_buffer[3] = temp_data & 0xFF;
            RAND_bytes(&large_send_buffer[4], NONCE_SIZE);
            mpi_errno = MPIC_Send(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR,  rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (large_send_buffer+4), NONCE_SIZE,
                                   (char *)recvbuf, count*extent,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    printf(COLOR_GREEN"MPI_Naive_Allreduce 1: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, rank+1);
                    fflush(stdout);
                }
#endif
            mpi_errno = MPIC_Send(large_send_buffer+MSG_HEADER_SIZE, max_out_len, MPI_CHAR, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            
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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
             mpi_errno = MPIC_Recv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            
            temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);        
            mpi_errno = MPIC_Recv(large_recv_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE, MPI_CHAR, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
                                   &dec_count, temp_data,
                                   (large_recv_buffer+4), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (unsigned long)(temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, rank-1);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                printf(COLOR_BLUE"  MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, rank-1);
                fflush(stdout);
            }
#endif                                         

            //mpi_errno = MPIC_Recv(tmp_buf, count, datatype, rank - 1,
            //                         MPIR_ALLREDUCE_TAG, comm_ptr,
            //                         MPI_STATUS_IGNORE, errflag);
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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
               (*uop) (tmp_buf, recvbuf, &count, &datatype);

               /* change the rank */
               newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
               into tmp_buf */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
           /* mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                         dst, MPIR_ALLREDUCE_TAG,
                                         tmp_buf, count, datatype, dst,
                                         MPIR_ALLREDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);*/
			
             mpi_errno = MPIC_Irecv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr,
                                      &reqarray[0]);                            
            mpi_errno = MPIC_Irecv(large_recv_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE, MPI_CHAR, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr,
                                      &reqarray[1]);                          
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            
            //RAND_bytes(large_send_buffer, NONCE_SIZE);
            temp_data = count * extent;
            large_send_buffer[0] = (temp_data >> 24) & 0xFF;
            large_send_buffer[1] = (temp_data >> 16) & 0xFF;
            large_send_buffer[2] = (temp_data >> 8) & 0xFF;
            large_send_buffer[3] = temp_data & 0xFF;
            RAND_bytes(&large_send_buffer[4], NONCE_SIZE);
            
            mpi_errno = MPIC_Isend(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR,  dst,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, &reqarray[2], errflag);
            
            max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (large_send_buffer+4), NONCE_SIZE,
                                   (char *)recvbuf, count*extent,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    printf(COLOR_GREEN"MPI_Naive_Allreduce 1: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, dst);
                    fflush(stdout);
                }
#endif

            mpi_errno = MPIC_Isend(large_send_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE, MPI_CHAR, dst,
                                          MPIR_ALLREDUCE_TAG, comm_ptr,
                                          &reqarray[3], errflag);

            
            mpi_errno = MPIC_Waitall(1,&reqarray[0],&starray[0], errflag); // header
             temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);         
             mpi_errno = MPIC_Waitall(1,&reqarray[1],&starray[1], errflag); // body

             if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
                                   &dec_count, (temp_data),
                                   (large_recv_buffer+4), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (unsigned long)(temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, dst);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                printf(COLOR_BLUE"  MPI_Naive_Allreduce 1: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                fflush(stdout);
            }
#endif                  
            mpi_errno = MPIC_Waitall(1,&reqarray[2],&starray[2], errflag);  // send header
            mpi_errno = MPIC_Waitall(1,&reqarray[3],&starray[3], errflag); // send body                                                          

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            //printf("return from sendrecv\n");fflush(stdout);
            /* tmp_buf contains data received in this step.
               recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif
            } else {
                /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif
                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                                   recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                            MPI_ERR_OTHER, "**fail");
                }
                    mask <<= 1;
            }
        }

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            //RAND_bytes(large_send_buffer, NONCE_SIZE);
             temp_data = count * extent;
            large_send_buffer[0] = (temp_data >> 24) & 0xFF;
            large_send_buffer[1] = (temp_data >> 16) & 0xFF;
            large_send_buffer[2] = (temp_data >> 8) & 0xFF;
            large_send_buffer[3] = temp_data & 0xFF;
            RAND_bytes(&large_send_buffer[4], NONCE_SIZE);

            mpi_errno = MPIC_Send(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (large_send_buffer+4), NONCE_SIZE,
                                   (char *)recvbuf, count*extent,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPI_Naive_Allreduce rank = %d\n",rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    printf(COLOR_GREEN"MPI_Naive_Allreduce 2: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, rank, rank+1);
                    fflush(stdout);
                }
#endif
             mpi_errno = MPIC_Send(large_send_buffer+MSG_HEADER_SIZE, max_out_len, MPI_CHAR, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
          
        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
           
               mpi_errno = MPIC_Recv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            
            temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);                            
             mpi_errno = MPIC_Recv(large_recv_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE, MPI_CHAR, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                   &dec_count, temp_data,
                                   (large_recv_buffer+4), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (unsigned long)(temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPI_NAIVE_Allreduce  rank = %d source =%d  \n",rank, rank-1);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                printf(COLOR_BLUE"  MPI_Naive_Allreduce: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, rank-1);
                fflush(stdout);
            }
#endif                                            

        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
    
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rd);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;


}



#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rd_MV2_Original"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_pt2pt_rd_MV2_Original(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_rd_MV2_Original  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif      
    MPIR_TIMER_START(coll,allreduce,shm_rd);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rd, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rd);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    
    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
           if (op_ptr->language == MPID_LANG_C) {
               uop = (MPI_User_function *) op_ptr->function.c_function;
           } else {
               uop = (MPI_User_function *) op_ptr->function.f77_function;
           }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count, datatype, rank + 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
            mpi_errno = MPIC_Recv(tmp_buf, count, datatype, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
               (*uop) (tmp_buf, recvbuf, &count, &datatype);

               /* change the rank */
               newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recvbuf. Recv
               into tmp_buf */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
            mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                         dst, MPIR_ALLREDUCE_TAG,
                                         tmp_buf, count, datatype, dst,
                                         MPIR_ALLREDUCE_TAG, comm_ptr,
                                         MPI_STATUS_IGNORE, errflag);

            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* tmp_buf contains data received in this step.
               recvbuf contains data accumulated so far */

            if (is_commutative || (dst < rank)) {
                /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif
            } else {
                /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                    count, datatype,
                                                    uop);
                } else {
#endif
                    (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                }
#endif
                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                                   recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                            MPI_ERR_OTHER, "**fail");
                }
                    mask <<= 1;
            }
        }

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count,
                                     datatype, rank - 1,
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rd, recv, count, datatype);
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rd);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;


}



/* not declared static because a machine-specific function may call this one 
   in some cases */
/* This is flat reduce-scatter-allgather allreduce */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rs_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_pt2pt_rs_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    if (security_approach==2005)
        return MPIR_Allreduce_pt2pt_rs_MV2_NodeAware(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else if (security_approach==2001) 
        return MPIR_Allreduce_pt2pt_rs_MV2_Naive(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_pt2pt_rs_MV2_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}

#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rs_MV2_NodeAware"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_pt2pt_rs_MV2_NodeAware(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE) { char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    fprintf(stderr,"[allreduce rank = %d host = %s count=%d ] Func: MPIR_Allreduce_pt2pt_rs_MV2_NodeAware [COLL_NAIVE_AESGCM_PLUS]\n",
    comm_ptr->rank,hostname, count);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_rs);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rs, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps, *enc_disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;

    unsigned int temp_data;
    long unsigned int ciphertext_sendbuf_len, max_out_len, dec_count;

    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rs);
        return MPI_SUCCESS;
    }
	
											 
    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            } else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
            }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
        rank = comm_ptr->rank;
        MPIR_Rank_list_mapper = &Bunch_Rank_list_mapper;
    } else {
        /* my index on rank_list */
        rank = comm_ptr->dev.ch.rank_list_index;
        MPIU_Assert(rank >= 0);
        MPIR_Rank_list_mapper = &Cyclic_Rank_list_mapper;
    }

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;

    MPID_Node_id_t my_node_id;
	MPID_Node_id_t remote_node_id;

    MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, rank), &my_node_id);

    unsigned int enc_chunk_size = -1;
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
	
	int context_id;
    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
    MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {

        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);

            MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, rank + 1), &remote_node_id);

            if ((my_node_id == remote_node_id)){
					
                    mpi_errno = MPIC_Send(recvbuf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            } else {
				
                    mpi_errno = MPIC_ENC_NAIVE_Send(recvbuf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
				
            }
       
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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);			

            MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, rank - 1), &remote_node_id);
			
            if ((my_node_id == remote_node_id)){
			
                mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);			
            } else {			

                mpi_errno = MPIC_ENC_NAIVE_Recv(tmp_buf, count,
                                  datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
		    }
			
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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
                (*uop) (tmp_buf, recvbuf, &count, &datatype);

                /* change the rank */
                newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */

    if (newrank != -1) {
        if ((HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) || (count < pof2)) {  /* use recursive doubling */
            mask = 0x1;
			
            while (mask < pof2) {
				
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                /* Send the most current data, which is in recvbuf. Recv
                   into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
				
				MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, dst), &remote_node_id);
			
				if ((my_node_id == remote_node_id)){ // || (init_phase)){
								
					mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
												 MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
												 tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
												 MPIR_ALLREDUCE_TAG, comm_ptr,
												 MPI_STATUS_IGNORE, errflag);
				
				} else {
									
					 mpi_errno = MPIC_NAIVE_PLUS_Sendrecv(recvbuf, count, datatype,
												 MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
												 tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
												 MPIR_ALLREDUCE_TAG, comm_ptr,
												 MPI_STATUS_IGNORE, errflag);     
				}

                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */

                if (is_commutative || (dst < rank)) {
                    /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                } else {
                    /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                               recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                        MPI_ERR_OTHER, "**fail");
                }
                mask <<= 1;
            }
        } else {

            /* do a reduce-scatter followed by allgather */

            /* for the reduce-scatter, calculate the count that
               each process receives and the displacement within
               the buffer */

            MPIU_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof (int), mpi_errno,
                                "counts");
            MPIU_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof (int),
                                mpi_errno, "displacements");								
			MPIU_CHKLMEM_MALLOC(enc_disps, int *, pof2 * sizeof (int),
                                mpi_errno, "enc_displacements");	

            for (i = 0; i < (pof2 - 1); i++) {
                cnts[i] = count / pof2;
            }
            cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

            disps[0] = 0;
            enc_disps[0] = 0;
            for (i = 1; i < pof2; i++) {
                disps[i] = disps[i - 1] + cnts[i - 1];
                enc_disps[i] = enc_disps[i-1] + (cnts[i-1]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
            }

            mask = 0x1;
            send_idx = recv_idx = 0;
            last_idx = pof2;

            while (mask < pof2) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    send_idx = recv_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < send_idx; i++)
                        recv_cnt += cnts[i];
                } else {
                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < last_idx; i++)
                        recv_cnt += cnts[i];
                }
								
                /* Send data from recvbuf. Recv into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);                
				
				//MPID_Get_node_id(comm_ptr, dst, &remote_node_id);
				MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, dst), &remote_node_id);
			
				if ((my_node_id == remote_node_id)){
			
            		/* mpi_errno = MPIC_Sendrecv((char *) recvbuf +
												 disps[send_idx] * extent,
												 send_cnt, datatype,
												 MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
												 (char *) tmp_buf +
												 disps[recv_idx] * extent,
												 recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
												 MPIR_ALLREDUCE_TAG, comm_ptr,
												 MPI_STATUS_IGNORE, errflag); */
												 
					mpi_errno = MPID_Irecv((char *) tmp_buf + disps[recv_idx] * extent, recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);					
					if (mpi_errno) MPIR_ERR_POP(mpi_errno);						
					
					mpi_errno = MPID_Isend((char *) recvbuf + disps[send_idx] * extent, send_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);					
					if (mpi_errno) MPIR_ERR_POP(mpi_errno);

					mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
					if (mpi_errno) MPIR_ERR_POP(mpi_errno);

												 
                } else {
					
					enc_chunk_size = send_cnt*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;
                	
					MPID_Irecv(large_recv_buffer, enc_chunk_size, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, dst),  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);
																	
					RAND_bytes(large_send_buffer, NONCE_SIZE);
					max_out_len = enc_chunk_size; 
					if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
								   &ciphertext_sendbuf_len, max_out_len,
								   (&large_send_buffer[0]), NONCE_SIZE,
								   (char *) recvbuf + (disps[send_idx] * extent), recv_cnt*extent,
								   NULL, 0))
					{
						printf("Error in encryption: RSA rank = %d\n",comm_ptr->rank);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"RSA : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, MPIR_Rank_list_mapper(comm_ptr, dst));
					}
#endif
					
					MPID_Isend(large_send_buffer, ciphertext_sendbuf_len+12, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);	
					
					mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
					
					if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf + (disps[recv_idx] * extent),
								   &dec_count, max_out_len,
								   (&large_recv_buffer[0]), NONCE_SIZE,
								   large_recv_buffer + NONCE_SIZE, ciphertext_sendbuf_len,
								   NULL, 0))
					{
						if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA_NP_RS : rank = %d source =%d  iter = %d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, MPIR_Rank_list_mapper(comm_ptr, dst),i,count,mask,recv_cnt);
						//fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_BLUE"RSA_NP_RS : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
						fflush(stdout);
					}
#endif  

				}					
				
                (*uop) ((char *) tmp_buf + disps[recv_idx] * extent,
                        (char *) recvbuf + disps[recv_idx] * extent,
                        &recv_cnt, &datatype);
				

                /* update send_idx for next iteration */
                send_idx = recv_idx;
                mask <<= 1;
				
				if (my_node_id != remote_node_id){
				
					mpi_errno = MPIC_Wait(send_req_ptr, errflag);
					if (mpi_errno) MPIR_ERR_POP(mpi_errno);

					MPID_Request_release(send_req_ptr);
					MPID_Request_release(recv_req_ptr);
				}

                /* update last_idx, but not in last iteration
                   because the value is needed in the allgather
                   step below. */
                if (mask < pof2)
                    last_idx = recv_idx + pof2 / mask;
            }				

            /* now do the allgather */

            mask >>= 1;
			
			/* There is at least one inter-node communication  */
			int all_intra_node=0;
			
			
			if (my_node_id != remote_node_id) {

				RAND_bytes(large_recv_buffer + enc_disps[recv_idx] , NONCE_SIZE);
				max_out_len = recv_cnt * extent + ENC_MSG_TAG_SIZE; 
				if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer + (enc_disps[recv_idx] ) + NONCE_SIZE),
							   &ciphertext_sendbuf_len, max_out_len,
							   (&large_recv_buffer[enc_disps[recv_idx]]), NONCE_SIZE,
							   (char *) recvbuf + disps[recv_idx] * extent, recv_cnt * extent,
							   NULL, 0))
				{
					printf("Error in encryption: RSA rank = %d\n",comm_ptr->rank);
					fflush(stdout);
				}			
			} else {
				all_intra_node = 1;
			}

			int seg_count=0;
			int same_node = 0;
			int basic_seg_size = recv_cnt * extent;
			int basic_cipher_size = basic_seg_size + ENC_MSG_TAG_SIZE + NONCE_SIZE;

			int step_power = 0;
			int last_recv_idx=0;
			
            while (mask > 0) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    /* update last_idx except on first iteration */
                    if (mask != pof2 / 2) {
                        last_idx = last_idx + pof2 / (mask * 2);
                    }

                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                        //enc_cnt += (cnts[i]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = recv_idx-send_idx;
                } else {
                    recv_idx = send_idx - pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
						//enc_cnt += (cnts[i]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = last_idx-send_idx;
                }
							
				
                enc_chunk_size = (unsigned int) ( (send_cnt*extent) + (seg_count*(12+16)));
                max_out_len = enc_chunk_size;  
								
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);
				
				MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, dst), &remote_node_id);
				
				if ((my_node_id == remote_node_id)){
					
					if ((same_node == 0)  &&  (all_intra_node == 0)) {
						
						for (i=0; i<step_power; i++){
	
								if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent) +i*basic_seg_size),
											   &dec_count, max_out_len,
											   (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
											   large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
											   NULL, 0))
								{
									if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG-First-Same_node) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, MPIR_Rank_list_mapper(comm_ptr, dst),i,recv_cnt/2);
								}
#if ENABLE_SECURE_MPI_DEBUG
                                else
                                {
                                    if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  RSA_NP (ALG-First-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                                    fflush(stdout);
                                }
#endif 								
						}						
						same_node++;
					}
					
					/* mpi_errno = MPIC_Sendrecv((char *) recvbuf + disps[send_idx] * extent,
                                             send_cnt, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             (char *) recvbuf +  disps[recv_idx] * extent,
                                             recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag); */
											 
						MPID_Irecv((char *) recvbuf + disps[recv_idx] * extent, recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);				
												
						MPID_Isend((char *) recvbuf + disps[send_idx] * extent, send_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);
						
						mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
											 
				} else {

					MPID_Irecv(large_recv_buffer + (enc_disps[recv_idx]), enc_chunk_size, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, dst),  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);					
											
					MPID_Isend(large_recv_buffer + (enc_disps[send_idx]), enc_chunk_size, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

					if (step_power > 0)
					{						
						for (i=0; i<step_power; i++){												
					
							if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent)+i*basic_seg_size),
										   &dec_count, max_out_len,
										   (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
										   large_recv_buffer + (enc_disps[last_recv_idx]) + (i * basic_cipher_size + NONCE_SIZE), basic_cipher_size - NONCE_SIZE,
										   NULL, 0))
							{
								if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, MPIR_Rank_list_mapper(comm_ptr, dst),i,count);
							}
#if ENABLE_SECURE_MPI_DEBUG
							else
							{
								if (ENABLE_SECURE_DEBUG) fprintf(COLOR_GREEN"RSA_NP (ALG-Not-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
								fflush(stdout);
							}
#endif 								
						}
					} 
					
					mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
					if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
					
					// last_recv_idx=recv_idx;
										
				}
		             
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
				
				
				if ( step_power == 0) step_power=1;
				else step_power = step_power * 2;
				
				last_recv_idx=recv_idx;
				

                if (newrank > newdst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
;
                MPID_Request_release(recv_req_ptr);
            }
			
			
			/* When All the communications were Inter-node ones */
			if ((same_node == 0) && (all_intra_node == 0)) {
					
				for (i=0; i<step_power; i++){
				
					if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf + ((disps[last_recv_idx]*extent)+i*basic_seg_size),
								   &dec_count, max_out_len,
								   (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
								   large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
								   NULL, 0))
					{
						if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at RSA (ALG-Last-out) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, MPIR_Rank_list_mapper(comm_ptr, dst),i,recv_cnt);
					}
#if ENABLE_SECURE_MPI_DEBUG
					else
					{
						if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  RSA_NP (ALG-Last-Dec): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
						fflush(stdout);
					}
#endif 							
				}
			}			
        }
    }
	

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {


        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);

            MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, rank - 1), &remote_node_id);

            if ((my_node_id == remote_node_id)){

                mpi_errno = MPIC_Send(recvbuf, count,
                                     datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            } else {

                 mpi_errno = MPIC_ENC_NAIVE_Send(recvbuf, count,
                                     datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            }                         
                                                                          
        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);

            MPID_Get_node_id(comm_ptr, MPIR_Rank_list_mapper(comm_ptr, rank + 1), &remote_node_id);

            if ((my_node_id == remote_node_id)){

                mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            } else {

                mpi_errno = MPIC_ENC_NAIVE_Recv(recvbuf, count,
                                  datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            }
        }

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

	/* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rs);

    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}


#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rs_MV2_Naive"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)


int MPIR_Allreduce_pt2pt_rs_MV2_Naive(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {char hostname[100];
    int namelen;
    gethostname(hostname, &namelen);
    printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_rs_MV2_Naive   count = %d  [COLL_NAIVE_AESGCM]\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_rs);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rs, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;

    unsigned int temp_data;
    long unsigned int ciphertext_sendbuf_len, max_out_len, dec_count;

    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rs);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            } else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
            }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
        rank = comm_ptr->rank;
        MPIR_Rank_list_mapper = &Bunch_Rank_list_mapper;
    } else {
        /* my index on rank_list */
        rank = comm_ptr->dev.ch.rank_list_index;
        MPIU_Assert(rank >= 0);
        MPIR_Rank_list_mapper = &Cyclic_Rank_list_mapper;
    }
	
	
    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
            /*mpi_errno = MPIC_Send(recvbuf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);*/
            temp_data = count * extent;
            large_send_buffer[0] = (temp_data >> 24) & 0xFF;
            large_send_buffer[1] = (temp_data >> 16) & 0xFF;
            large_send_buffer[2] = (temp_data >> 8) & 0xFF;
            large_send_buffer[3] = temp_data & 0xFF;
            RAND_bytes(&large_send_buffer[4], NONCE_SIZE);
            mpi_errno = MPIC_Send(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);

            max_out_len = temp_data+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[4]), NONCE_SIZE,
                                   (char *)recvbuf, temp_data,
                                   NULL, 0))
                {
                    printf("Error in encryption: allreduce_rs_mv2 rank = %d\n",comm_ptr->rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    printf(COLOR_GREEN"allreduce_rs_mv2: ciphertext_sendbuf_len = %d Rank = %d "COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank);
                    fflush(stdout);
                }
#endif   
            mpi_errno = MPIC_Send(large_send_buffer+MSG_HEADER_SIZE, count * extent+ENC_MSG_TAG_SIZE, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);                      

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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
           
            mpi_errno = MPIC_Recv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);
            
            temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
            
            mpi_errno = MPIC_Recv(large_recv_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE, MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
                                     MPI_STATUS_IGNORE, errflag);

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)tmp_buf,
                                   &dec_count, temp_data,
                                   (&large_recv_buffer[4]), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error allreduce_rs_mv2  rank = %d  \n",comm_ptr->rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                printf(COLOR_BLUE"  allreduce_rs_mv2: decrypted = %lu Rank = %d "COLOR_RESET"\n", dec_count, comm_ptr->rank);
                fflush(stdout);
            }
#endif                                                                                 
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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
                (*uop) (tmp_buf, recvbuf, &count, &datatype);

                /* change the rank */
                newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */

    if (newrank != -1) {
        if ((HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) || (count < pof2)) {  /* use recursive doubling */
            mask = 0x1;
            while (mask < pof2) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                /* Send the most current data, which is in recvbuf. Recv
                   into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
				/* if (init_phase){
                mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);
				} else  */{									
                 mpi_errno = MPIC_ENC_NAIVE_Sendrecv(recvbuf, count, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);                             
				}
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */

                if (is_commutative || (dst < rank)) {
                    /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                } else {
                    /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                               recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                        MPI_ERR_OTHER, "**fail");
                }
                mask <<= 1;
            }
        } else {

            /* do a reduce-scatter followed by allgather */

            /* for the reduce-scatter, calculate the count that
               each process receives and the displacement within
               the buffer */

            MPIU_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof (int), mpi_errno,
                                "counts");
            MPIU_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof (int),
                                mpi_errno, "displacements");

            for (i = 0; i < (pof2 - 1); i++) {
                cnts[i] = count / pof2;
            }
            cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

            disps[0] = 0;
            for (i = 1; i < pof2; i++) {
                disps[i] = disps[i - 1] + cnts[i - 1];
            }

            mask = 0x1;
            send_idx = recv_idx = 0;
            last_idx = pof2;
            while (mask < pof2) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    send_idx = recv_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < send_idx; i++)
                        recv_cnt += cnts[i];
                } else {
                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < last_idx; i++)
                        recv_cnt += cnts[i];
                }

                /* Send data from recvbuf. Recv into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);
											 
                mpi_errno = MPIC_ENC_NAIVE_Sendrecv((char *) recvbuf +
                                             disps[send_idx] * extent,
                                             send_cnt, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             (char *) tmp_buf +
                                             disps[recv_idx] * extent,
                                             recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);      
											 
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */

                /* This algorithm is used only for predefined ops
                   and predefined ops are always commutative. */

                (*uop) ((char *) tmp_buf + disps[recv_idx] * extent,
                        (char *) recvbuf + disps[recv_idx] * extent,
                        &recv_cnt, &datatype);

                /* update send_idx for next iteration */
                send_idx = recv_idx;
                mask <<= 1;

                /* update last_idx, but not in last iteration
                   because the value is needed in the allgather
                   step below. */
                if (mask < pof2)
                    last_idx = recv_idx + pof2 / mask;
            }

            /* now do the allgather */

            mask >>= 1;
            while (mask > 0) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    /* update last_idx except on first iteration */
                    if (mask != pof2 / 2) {
                        last_idx = last_idx + pof2 / (mask * 2);
                    }

                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
                } else {
                    recv_idx = send_idx - pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
                }
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);
											 
                mpi_errno = MPIC_ENC_NAIVE_Sendrecv((char *) recvbuf +
                                             disps[send_idx] * extent,
                                             send_cnt, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             (char *) recvbuf +
                                             disps[recv_idx] * extent,
                                             recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag); 
															 
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                if (newrank > newdst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
            }
        }
    }

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
            
            temp_data = count * extent;
            large_send_buffer[0] = (temp_data >> 24) & 0xFF;
            large_send_buffer[1] = (temp_data >> 16) & 0xFF;
            large_send_buffer[2] = (temp_data >> 8) & 0xFF;
            large_send_buffer[3] = temp_data & 0xFF;
            RAND_bytes(&large_send_buffer[4], NONCE_SIZE); 

            mpi_errno = MPIC_Send(large_send_buffer, MSG_HEADER_SIZE,
                                     MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            max_out_len = temp_data+ENC_MSG_TAG_SIZE; 
            
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[4]), NONCE_SIZE,
                                   (char *)recvbuf, temp_data,
                                   NULL, 0))
                {
                    printf("Error in encryption: allreduce_rs_mv2 rank = %d\n",comm_ptr->rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    printf(COLOR_GREEN"allreduce_rs_mv2: ciphertext_sendbuf_len = %d Rank = %d "COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank);
                    fflush(stdout);
                }
#endif                           

            mpi_errno = MPIC_Send(large_send_buffer+MSG_HEADER_SIZE, count*extent+ENC_MSG_TAG_SIZE,
                                     MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);                                                 
        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);

            mpi_errno = MPIC_Recv(large_recv_buffer, MSG_HEADER_SIZE,
                                  MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);                      

            mpi_errno = MPIC_Recv(large_recv_buffer+MSG_HEADER_SIZE,count*extent+ENC_MSG_TAG_SIZE ,
                                  MPI_CHAR, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                   &dec_count, temp_data,
                                   (&large_recv_buffer[4]), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error allreduce_rs_mv2  rank = %d  \n",comm_ptr->rank);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                printf(COLOR_BLUE"  allreduce_rs_mv2: decrypted = %lu Rank = %d "COLOR_RESET"\n", dec_count, comm_ptr->rank);
                fflush(stdout);
            }
#endif                                                                               
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rs);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}


#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_rs_MV2_Original"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)


int MPIR_Allreduce_pt2pt_rs_MV2_Original(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_rs_MV2_Original  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_rs);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_rs, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_rs);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else {
#endif
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            } else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
            }
#ifdef HAVE_CXX_BINDING
        }
#endif
    }

    if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
        rank = comm_ptr->rank;
        MPIR_Rank_list_mapper = &Bunch_Rank_list_mapper;
    } else {
        /* my index on rank_list */
        rank = comm_ptr->dev.ch.rank_list_index;
        MPIU_Assert(rank >= 0);
        MPIR_Rank_list_mapper = &Cyclic_Rank_list_mapper;
    }

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno =
            MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                           datatype);
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
    }

    /* find nearest power-of-two less than or equal to comm_size */
    pof2 = comm_ptr->dev.ch.gpof2;

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
       processes of rank < 2*rem send their data to
       (rank+1). These even-numbered processes no longer
       participate in the algorithm until the very end. The
       remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
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
        } else {
            /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
            mpi_errno = MPIC_Recv(tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr,
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
                (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                datatype, uop);
            } else {
#endif
                (*uop) (tmp_buf, recvbuf, &count, &datatype);

                /* change the rank */
                newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
            }
#endif
        }
    } else {                /* rank >= 2*rem */
        newrank = rank - rem;
    }

    /* If op is user-defined or count is less than pof2, use
       recursive doubling algorithm. Otherwise do a reduce-scatter
       followed by allgather. (If op is user-defined,
       derived datatypes are allowed and the user could pass basic
       datatypes on one process and derived on another as long as
       the type maps are the same. Breaking up derived
       datatypes to do the reduce-scatter is tricky, therefore
       using recursive doubling in that case.) */

    if (newrank != -1) {
        if ((HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) || (count < pof2)) {  /* use recursive doubling */
            mask = 0x1;
            while (mask < pof2) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                /* Send the most current data, which is in recvbuf. Recv
                   into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
                mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             tmp_buf, count, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);

                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */

                if (is_commutative || (dst < rank)) {
                    /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                } else {
                    /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif
                        (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                    }
#endif
                    /* copy result back into recvbuf */
                    mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                               recvbuf, count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                        MPI_ERR_OTHER, "**fail");
                }
                mask <<= 1;
            }
        } else {

            /* do a reduce-scatter followed by allgather */

            /* for the reduce-scatter, calculate the count that
               each process receives and the displacement within
               the buffer */

            MPIU_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof (int), mpi_errno,
                                "counts");
            MPIU_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof (int),
                                mpi_errno, "displacements");

            for (i = 0; i < (pof2 - 1); i++) {
                cnts[i] = count / pof2;
            }
            cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

            disps[0] = 0;
            for (i = 1; i < pof2; i++) {
                disps[i] = disps[i - 1] + cnts[i - 1];
            }

            mask = 0x1;
            send_idx = recv_idx = 0;
            last_idx = pof2;
            while (mask < pof2) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    send_idx = recv_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < send_idx; i++)
                        recv_cnt += cnts[i];
                } else {
                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++)
                        send_cnt += cnts[i];
                    for (i = recv_idx; i < last_idx; i++)
                        recv_cnt += cnts[i];
                }

                /* Send data from recvbuf. Recv into tmp_buf */
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);
                mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                             disps[send_idx] * extent,
                                             send_cnt, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             (char *) tmp_buf +
                                             disps[recv_idx] * extent,
                                             recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */

                /* This algorithm is used only for predefined ops
                   and predefined ops are always commutative. */

                (*uop) ((char *) tmp_buf + disps[recv_idx] * extent,
                        (char *) recvbuf + disps[recv_idx] * extent,
                        &recv_cnt, &datatype);

                /* update send_idx for next iteration */
                send_idx = recv_idx;
                mask <<= 1;

                /* update last_idx, but not in last iteration
                   because the value is needed in the allgather
                   step below. */
                if (mask < pof2)
                    last_idx = recv_idx + pof2 / mask;
            }

            /* now do the allgather */

            mask >>= 1;
            while (mask > 0) {
                newdst = newrank ^ mask;
                /* find real rank of dest */
                dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                send_cnt = recv_cnt = 0;
                if (newrank < newdst) {
                    /* update last_idx except on first iteration */
                    if (mask != pof2 / 2) {
                        last_idx = last_idx + pof2 / (mask * 2);
                    }

                    recv_idx = send_idx + pof2 / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
                } else {
                    recv_idx = send_idx - pof2 / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
                }
                MPIR_PVAR_INC(allreduce, pt2pt_rs, send, send_cnt, datatype);
                MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, recv_cnt, datatype);
                mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                             disps[send_idx] * extent,
                                             send_cnt, datatype,
                                             MPIR_Rank_list_mapper(comm_ptr, dst), MPIR_ALLREDUCE_TAG,
                                             (char *) recvbuf +
                                             disps[recv_idx] * extent,
                                             recv_cnt, datatype, MPIR_Rank_list_mapper(comm_ptr, dst),
                                             MPIR_ALLREDUCE_TAG, comm_ptr,
                                             MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }

                if (newrank > newdst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
            }
        }
    }

    /* In the non-power-of-two case, all odd-numbered
       processes of rank < 2*rem send the result to
       (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) {     /* odd */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, send, count, datatype);
            mpi_errno = MPIC_Send(recvbuf, count,
                                     datatype, MPIR_Rank_list_mapper(comm_ptr, rank - 1),
                                     MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
        } else {            /* even */
            MPIR_PVAR_INC(allreduce, pt2pt_rs, recv, count, datatype);
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, MPIR_Rank_list_mapper(comm_ptr, rank + 1),
                                  MPIR_ALLREDUCE_TAG, comm_ptr,
                                  MPI_STATUS_IGNORE, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rs);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}


/* not declared static because a machine-specific function may call this one 
   in some cases */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_old_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_old_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{

#ifdef MPID_HAS_HETERO
    int rc;
    int is_homogeneous = 1;
#endif
    int comm_size, rank;
    MPI_Aint type_size;
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int mask, dst, is_commutative, pof2, newrank = 0, rem, newdst, i,
        send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    MPIU_THREADPRIV_DECL;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero) {
        is_homogeneous = 0;
    }

    if (!is_homogeneous) {
        /* heterogeneous. To get the same result on all processes, we
           do a reduce to 0 and then broadcast. */
        mpi_errno = MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype,
                                    op, 0, comm_ptr, errflag);
        /* 
           FIXME: mpi_errno is error CODE, not necessarily the error
           class MPI_ERR_OP.  In MPICH2, we can get the error class 
           with errorclass = mpi_errno & ERROR_CLASS_MASK;
         */
        if (mpi_errno == MPI_ERR_OP || mpi_errno == MPI_SUCCESS) {
            /* Allow MPI_ERR_OP since we can continue from this error */
            rc = MPIR_Bcast_impl(recvbuf, count, datatype, 0, comm_ptr,
                                 errflag);
            if (rc)
                mpi_errno = rc;
        }
    } else
#endif                          /* MPID_HAS_HETERO */
    {
        /* homogeneous */

        /* set op_errno to 0. stored in perthread structure */
        MPIU_THREADPRIV_FIELD(op_errno) = 0;

        comm_size = comm_ptr->local_size;
        rank = comm_ptr->rank;

        if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
            is_commutative = 1;
            /* get the function by indexing into the op table */
            uop = MPIR_Op_table[op % 16 - 1];
        } else {
            MPID_Op_get_ptr(op, op_ptr);
            if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
                is_commutative = 0;
            } else {
                is_commutative = 1;
            }
#ifdef HAVE_CXX_BINDING
            if (op_ptr->language == MPID_LANG_CXX) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
                is_cxx_uop = 1;
            } else {
#endif
                if (op_ptr->language == MPID_LANG_C) {
                    uop = (MPI_User_function *) op_ptr->function.c_function;
                } else {
                    uop = (MPI_User_function *) op_ptr->function.f77_function;
                }
#ifdef HAVE_CXX_BINDING
            }
#endif
        }

        /* need to allocate temporary buffer to store incoming data */
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
        MPID_Datatype_get_extent_macro(datatype, extent);

        MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                            count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                            "temporary buffer");

        /* adjust for potential negative lower bound in datatype */
        tmp_buf = (void *) ((char *) tmp_buf - true_lb);

        /* copy local data into recvbuf */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count,
                               datatype);
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
        }

        MPID_Datatype_get_size_macro(datatype, type_size);

        /* find nearest power-of-two less than or equal to comm_size */
        pof2 = comm_ptr->dev.ch.gpof2;

        rem = comm_size - pof2;
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
		if ((count * type_size <= mv2_coll_param.allreduce_short_msg) || (HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) || (count < pof2)){
			printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_old_MV2 == RD count = %d\n",comm_ptr->rank,hostname,count);
			//MPIR_Allreduce_pt2pt_rd_MV2	(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
			//return MPI_SUCCESS;
		}else{
			printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_old_MV2 == RSA  count = %d\n",comm_ptr->rank,hostname,count);
		}
	}
#endif     		

        /* In the non-power-of-two case, all even-numbered
           processes of rank < 2*rem send their data to
           (rank+1). These even-numbered processes no longer
           participate in the algorithm until the very end. The
           remaining processes form a nice power-of-two. */

        if (rank < 2 * rem) {
            if (rank % 2 == 0) {
                /* even */
                mpi_errno = MPIC_Send(recvbuf, count, datatype, rank + 1,
                                         MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
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
            } else {
                /* odd */
                mpi_errno = MPIC_Recv(tmp_buf, count, datatype, rank - 1,
                                         MPIR_ALLREDUCE_TAG, comm_ptr,
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
                    (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf, count,
                                                    datatype, uop);
                } else {
#endif
                    (*uop) (tmp_buf, recvbuf, &count, &datatype);

                    /* change the rank */
                    newrank = rank / 2;
#ifdef HAVE_CXX_BINDING
                }
#endif
            }
        } else {                /* rank >= 2*rem */
            newrank = rank - rem;
        }

        /* If op is user-defined or count is less than pof2, use
           recursive doubling algorithm. Otherwise do a reduce-scatter
           followed by allgather. (If op is user-defined,
           derived datatypes are allowed and the user could pass basic
           datatypes on one process and derived on another as long as
           the type maps are the same. Breaking up derived
           datatypes to do the reduce-scatter is tricky, therefore
           using recursive doubling in that case.) */

        if (newrank != -1) {
            if ((count * type_size <= mv2_coll_param.allreduce_short_msg) || (HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) || (count < pof2)) {  /* use recursive doubling */
                mask = 0x1;
                while (mask < pof2) {
                    newdst = newrank ^ mask;
                    /* find real rank of dest */
                    dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                    /* Send the most current data, which is in recvbuf. Recv
                       into tmp_buf */
                    mpi_errno = MPIC_Sendrecv(recvbuf, count, datatype,
                                                 dst, MPIR_ALLREDUCE_TAG,
                                                 tmp_buf, count, datatype, dst,
                                                 MPIR_ALLREDUCE_TAG, comm_ptr,
                                                 MPI_STATUS_IGNORE, errflag);

                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }

                    /* tmp_buf contains data received in this step.
                       recvbuf contains data accumulated so far */

                    if (is_commutative || (dst < rank)) {
                        /* op is commutative OR the order is already right */
#ifdef HAVE_CXX_BINDING
                        if (is_cxx_uop) {
                            (*MPIR_Process.cxx_call_op_fn) (tmp_buf, recvbuf,
                                                            count, datatype,
                                                            uop);
                        } else {
#endif
                            (*uop) (tmp_buf, recvbuf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                        }
#endif
                    } else {
                        /* op is noncommutative and the order is not right */
#ifdef HAVE_CXX_BINDING
                        if (is_cxx_uop) {
                            (*MPIR_Process.cxx_call_op_fn) (recvbuf, tmp_buf,
                                                            count, datatype,
                                                            uop);
                        } else {
#endif
                            (*uop) (recvbuf, tmp_buf, &count, &datatype);
#ifdef HAVE_CXX_BINDING
                        }
#endif
                        /* copy result back into recvbuf */
                        mpi_errno = MPIR_Localcopy(tmp_buf, count, datatype,
                                                   recvbuf, count, datatype);
                        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno,
                                            MPI_ERR_OTHER, "**fail");
                    }
                    mask <<= 1;
                }
            } else {

                /* do a reduce-scatter followed by allgather */

                /* for the reduce-scatter, calculate the count that
                   each process receives and the displacement within
                   the buffer */

                MPIU_CHKLMEM_MALLOC(cnts, int *, pof2 * sizeof (int), mpi_errno,
                                    "counts");
                MPIU_CHKLMEM_MALLOC(disps, int *, pof2 * sizeof (int),
                                    mpi_errno, "displacements");

                for (i = 0; i < (pof2 - 1); i++) {
                    cnts[i] = count / pof2;
                }
                cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

                disps[0] = 0;
                for (i = 1; i < pof2; i++) {
                    disps[i] = disps[i - 1] + cnts[i - 1];
                }

                mask = 0x1;
                send_idx = recv_idx = 0;
                last_idx = pof2;
                while (mask < pof2) {
                    newdst = newrank ^ mask;
                    /* find real rank of dest */
                    dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                    send_cnt = recv_cnt = 0;
                    if (newrank < newdst) {
                        send_idx = recv_idx + pof2 / (mask * 2);
                        for (i = send_idx; i < last_idx; i++)
                            send_cnt += cnts[i];
                        for (i = recv_idx; i < send_idx; i++)
                            recv_cnt += cnts[i];
                    } else {
                        recv_idx = send_idx + pof2 / (mask * 2);
                        for (i = send_idx; i < recv_idx; i++)
                            send_cnt += cnts[i];
                        for (i = recv_idx; i < last_idx; i++)
                            recv_cnt += cnts[i];
                    }

                    /* Send data from recvbuf. Recv into tmp_buf */
                    mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                                 disps[send_idx] * extent,
                                                 send_cnt, datatype,
                                                 dst, MPIR_ALLREDUCE_TAG,
                                                 (char *) tmp_buf +
                                                 disps[recv_idx] * extent,
                                                 recv_cnt, datatype, dst,
                                                 MPIR_ALLREDUCE_TAG, comm_ptr,
                                                 MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }

                    /* tmp_buf contains data received in this step.
                       recvbuf contains data accumulated so far */

                    /* This algorithm is used only for predefined ops
                       and predefined ops are always commutative. */

                    (*uop) ((char *) tmp_buf + disps[recv_idx] * extent,
                            (char *) recvbuf + disps[recv_idx] * extent,
                            &recv_cnt, &datatype);

                    /* update send_idx for next iteration */
                    send_idx = recv_idx;
                    mask <<= 1;

                    /* update last_idx, but not in last iteration
                       because the value is needed in the allgather
                       step below. */
                    if (mask < pof2)
                        last_idx = recv_idx + pof2 / mask;
                }

                /* now do the allgather */

                mask >>= 1;
                while (mask > 0) {
                    newdst = newrank ^ mask;
                    /* find real rank of dest */
                    dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

                    send_cnt = recv_cnt = 0;
                    if (newrank < newdst) {
                        /* update last_idx except on first iteration */
                        if (mask != pof2 / 2) {
                            last_idx = last_idx + pof2 / (mask * 2);
                        }

                        recv_idx = send_idx + pof2 / (mask * 2);
                        for (i = send_idx; i < recv_idx; i++) {
                            send_cnt += cnts[i];
                        }
                        for (i = recv_idx; i < last_idx; i++) {
                            recv_cnt += cnts[i];
                        }
                    } else {
                        recv_idx = send_idx - pof2 / (mask * 2);
                        for (i = send_idx; i < last_idx; i++) {
                            send_cnt += cnts[i];
                        }
                        for (i = recv_idx; i < send_idx; i++) {
                            recv_cnt += cnts[i];
                        }
                    }

                    mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                                 disps[send_idx] * extent,
                                                 send_cnt, datatype,
                                                 dst, MPIR_ALLREDUCE_TAG,
                                                 (char *) recvbuf +
                                                 disps[recv_idx] * extent,
                                                 recv_cnt, datatype, dst,
                                                 MPIR_ALLREDUCE_TAG, comm_ptr,
                                                 MPI_STATUS_IGNORE, errflag);
                    if (mpi_errno) {
                        /* for communication errors, just record the error but continue */
                        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                    }

                    if (newrank > newdst) {
                        send_idx = recv_idx;
                    }

                    mask >>= 1;
                }
            }
        }

        /* In the non-power-of-two case, all odd-numbered
           processes of rank < 2*rem send the result to
           (rank-1), the ranks who didn't participate above. */
        if (rank < 2 * rem) {
            if (rank % 2) {     /* odd */
                mpi_errno = MPIC_Send(recvbuf, count,
                                         datatype, rank - 1,
                                         MPIR_ALLREDUCE_TAG, comm_ptr, errflag);
            } else {            /* even */

                mpi_errno = MPIC_Recv(recvbuf, count,
                                      datatype, rank + 1,
                                      MPIR_ALLREDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, errflag);
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

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}

/* intra-node shm reduce as the first reduce in allreduce */
int MPIR_Allreduce_reduce_shmem_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_reduce_shmem_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shm_intra);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shm_intra, 1);
    int mpi_errno = MPI_SUCCESS;
    int i = 0, is_commutative = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    char *shmem_buf = NULL;
    MPI_Comm shmem_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL;
    int local_rank = -1, local_size = 0;
    void *local_buf = NULL;
    int stride = 0;
    is_commutative = 0;
    int shmem_comm_rank;

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,shm_intra);
        return MPI_SUCCESS;
    }

    
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_shmem_coll_calls, 1);
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);
 
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    PMPI_Comm_size(shmem_comm, &local_size);
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);   
    if (count * (MPIR_MAX(extent, true_extent)) >= mv2_g_shmem_coll_max_msg_size) {
                mpi_errno =
                    MPIR_Reduce_intra(sendbuf, recvbuf, count, datatype, op, 0,
                                      shmem_commptr, errflag);
        goto fn_exit;
    }

    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }

    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;
    shmem_comm_rank = shmem_commptr->dev.ch.shmem_comm_rank;

    /* Doing the shared memory gather and reduction by the leader */
    if (local_rank == 0) {
        /* Message size is smaller than the shmem_reduce threshold. 
         * The intra-node communication is done through shmem */
        if (local_size > 1) {
            /* Node leader waits till all the non-leaders have written 
             * the data into the shmem buffer */
            MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                              shmem_comm_rank,
                                              (void *) &shmem_buf);
            if (is_commutative) {
                for (i = 1; i < local_size; i++) {
                    local_buf = (char *) shmem_buf + stride * i;
#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (local_buf, recvbuf,
                                                        count, datatype,
                                                        uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (local_buf, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif                          /* defined(HAVE_CXX_BINDING) */

                }
                MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size,
                                                        local_rank,
                                                        shmem_comm_rank);
            }
        }
    } else {
        MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                          shmem_comm_rank,
                                          (void *) &shmem_buf);
        local_buf = (char *) shmem_buf + stride * local_rank;
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, local_buf,
                                       count, datatype);
        } else {
            mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, local_buf,
                                       count, datatype);
        }
        MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                            "**fail");
        MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size, local_rank,
                                                    shmem_comm_rank);
    }

  fn_exit:
    MPIR_TIMER_END(coll,allreduce,shm_intra);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}

/* intra-node p2p reduce as the first reduce in allreduce */
int MPIR_Allreduce_reduce_p2p_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_reduce_p2p_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,intra_p2p);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_intra_p2p, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL;
    int local_rank = -1, local_size = 0;

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,intra_p2p);
        return MPI_SUCCESS;
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    PMPI_Comm_size(shmem_comm, &local_size);
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    /* Doing the shared memory gather and reduction by the leader */
    if (local_rank == 0) {
        /* Message size is larger than the shmem_reduce threshold. 
         * The leader will spend too much time doing the math operation
         * for messages that are larger. So, we use a point-to-point
         * based reduce to balance the load across all the processes within
         * the same node*/
        mpi_errno =
            MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype, op, 0,
                            shmem_commptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

    } else {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
                MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype, op, 0,
                                shmem_commptr, errflag);
        } else {
            /* MPI_Allreduce was called with MPI_IN_PLACE as the sendbuf.
             * Since we are doing Reduce now, we need to be careful. In
             * MPI_Reduce, only the root can use MPI_IN_PLACE as sendbuf.
             * Also, the recvbuf is not relevant at all non-root processes*/
            mpi_errno = MPIR_Reduce_MV2(recvbuf, NULL, count, datatype, op,
                                        0, shmem_commptr, errflag);
        }
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    MPIR_TIMER_END(coll,allreduce,intra_p2p);
    return (mpi_errno);
}

/* general two level allreduce helper function */


#undef FCNAME
#define FCNAME "MPIR_Allreduce_two_level_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_two_level_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    if (super_node==1 && security_approach==2005)
        return MPIR_Allreduce_two_level_MV2_SEC_Super_Node(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);    
    else return MPIR_Allreduce_two_level_MV2_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}




int MPIR_Allreduce_two_level_MV2_SEC_Super_Node(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{

#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE && (comm_ptr->rank == 0)) {		
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_two_level_MV2_SEC_Super_Node  count = %d [SUPER_NODE]\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    
    int mpi_errno = MPI_SUCCESS;
    MPIR_TIMER_START(coll,allreduce,2lvl);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_2lvl, 1);

    int mpi_errno_ret = MPI_SUCCESS;
    int total_size = 0;
    MPI_Aint true_lb, true_extent,extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int i,local_rank = -1, local_size = 0;
	//void *tmp_enc_buf;

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,2lvl);
        return MPI_SUCCESS;
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
	
	/* MPIU_CHKLMEM_MALLOC(tmp_enc_buf, void *,
                        (count * (MPIR_MAX(extent, true_extent))) + 28, mpi_errno,  // ENC_MSG_TAG_SIZE + NONCE_SIZE = 28 = 12+16
                        "temporary enc buffer");
    tmp_enc_buf = (void *) ((char *) tmp_enc_buf - true_lb); */ 

    total_size = comm_ptr->local_size;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    PMPI_Comm_size(shmem_comm, &local_size);
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;
    int ppn = shmem_commptr->local_size; // number of ranks per node
    int node = total_size/ppn; 
    if  (total_size % ppn !=0)  node++; // number of nodes
    int enc_size = count * extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;
	
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

    unsigned long ciphertext_sendbuf_len, dec_count, max_out_len = enc_size;
    
    
    if (local_rank == 0) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                       count, datatype);
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
        }
    }

    /* Doing the shared memory gather and reduction by the leader */
    if ((MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_shmem_MV2) || 
        (MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_p2p_MV2) ) {
        mpi_errno = MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
        }
    else {
        mpi_errno = MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype, op, shmem_commptr, errflag);
    }

    if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
	
#if 1
        //int leader_size = leader_commptr->local_size;
        //int leader_rank = leader_commptr->rank;        

		MPI_User_function *uop;
		MPID_Op *op_ptr;
		int is_commutative;
#ifdef HAVE_CXX_BINDING
		int is_cxx_uop = 0;
#endif
		MPIU_THREADPRIV_DECL;
		MPIU_CHKLMEM_DECL(3);

		MPIU_THREADPRIV_GET;

		/* check if multiple threads are calling this collective function */
		MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

		/* homogeneous */

		/* set op_errno to 0. stored in perthread structure */
		MPIU_THREADPRIV_FIELD(op_errno) = 0;
		
		/* Get the operator and check whether it is commutative or not */
		if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
			is_commutative = 1;
			/* get the function by indexing into the op table */
			uop = MPIR_Op_table[op % 16 - 1];
		} else {
			MPID_Op_get_ptr(op, op_ptr);
			if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
				is_commutative = 0;
			} else {
				is_commutative = 1;
			}

#if defined(HAVE_CXX_BINDING)
			if (op_ptr->language == MPID_LANG_CXX) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
				is_cxx_uop = 1;
			} else
#endif                          /* defined(HAVE_CXX_BINDING) */
			if (op_ptr->language == MPID_LANG_C) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
			} else {
				uop = (MPI_User_function *) op_ptr->function.f77_function;
			}
		}
           
#endif  
	
    
    double level;
	level = ceil (log2 (node) / log2 (ppn+1));

    int leader_cnt =1;

	for (i=1; i<level; i++)
		leader_cnt = leader_cnt * (ppn+1);

	// Step 3
	while (leader_cnt >= 1){    
#if 1
        // Step 1
        int Sender_cnt=0;  // number of leaders who send own data
        if (node > leader_cnt *(ppn+1)) Sender_cnt = leader_cnt * ppn;  //leader_cnt*(ppn+1) - leader_cnt; 
        else Sender_cnt = node - leader_cnt; 

        int list_rank;
        // list_rank = comm_ptr->dev.ch.rank_list_index;
		
		if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
			list_rank = comm_ptr->rank;
			
		} else {
			/* my index on rank_list */
			list_rank = comm_ptr->dev.ch.rank_list_index;			
		}
		
		
        int node_rank = list_rank / ppn;
        
		char hostname[100];
		int namelen;    
		 
		gethostname(hostname, &namelen);

        // Step 2
        int msg_recv_cnt=0;	// number of message each Receiver node will gather

        if (list_rank< (leader_cnt * ppn)){
            
            msg_recv_cnt = (Sender_cnt/leader_cnt);

            if ((Sender_cnt % leader_cnt) > (node_rank)) msg_recv_cnt++;
            
        } else msg_recv_cnt = 0;
        
        if (local_rank == 0) {
              
			if ((node_rank < leader_cnt) && (msg_recv_cnt>0)) { // The Receivers Leaders (R-Leaders)
				
                // Step 4
                int remote_index = (node_rank + leader_cnt) * ppn;
				
                // int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;
                
                mpi_errno = MPID_Irecv((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag); 
            
                //Decrypt (tmp_enc_buf, shmem_buffer);
                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)shmem_buffer,
                                        &dec_count, max_out_len,
                                        (&tmp_enc_buf[0]), NONCE_SIZE,
                                        tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                        NULL, 0))
                                        {
                                            fprintf(stderr,"Super Node: err Decryption in R-leader rank = %d source =%d  count = %d\n",comm_ptr->rank, remote_rank,count);
                                        }

                /* shmem_buffer contains data received in this step.
			        recvbuf contains data accumulated so far */ 
                // operation (shmem_buffer,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (shmem_buffer, recvbuf,
                                                    count, datatype, uop);
                } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                    (*uop) (shmem_buffer, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                }
#endif

                MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

                // Step 5
                for (i = 1; i < msg_recv_cnt; i++){
				    // operation (shmem_buffer + i * count ,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (shmem_buffer + (i * (count*extent)), recvbuf,
                                                        count, datatype, uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (shmem_buffer + (i * (count*extent)), recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif
			    }

            } else if (node_rank < leader_cnt + Sender_cnt){

                // Calculate the index of receiver rank
                int remote_node = (node_rank - leader_cnt) % leader_cnt;
                int remote_index = (remote_node * ppn) + ((node_rank - leader_cnt) / leader_cnt);
                // int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;
                
                
                // Step 3
                //Encrypt(recvbuf,tmp_enc_buf);
                RAND_bytes(tmp_enc_buf, NONCE_SIZE);
                if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, tmp_enc_buf + NONCE_SIZE,
                                &ciphertext_sendbuf_len, max_out_len,
                                (&tmp_enc_buf[0]), NONCE_SIZE,
                                (char *) recvbuf, count*extent,
                                NULL, 0))
                {
                    fprintf(stderr,"Super Node: err Encryption in S-leader rank = %d\n",comm_ptr->rank);
                }
      
                //mpi_errno = MPID_Send(tmp_enc_buf, enc_size, MPI_CHAR, comm_ptr->dev.ch.rank_list[remote_rank], 12, comm_ptr);
                mpi_errno = MPID_Isend((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank, 12, comm_ptr, context_id, &send_req_ptr);					
            }         
        } else if (node_rank < leader_cnt){ //  Non-leader processes who help R-leaders in decryption (Child)

            // Step 4
            if (local_rank < msg_recv_cnt) {
                
                int remote_index = (leader_cnt * (local_rank + 1) + node_rank) * ppn ;
                //int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;          
                // mpi_errno = MPID_Recv(tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr);	
                mpi_errno = MPID_Irecv((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                
                // Decrypt (tmp_enc_buf, shmem_buf + local_rank * count); // Copy decrypted message to shared memory buffer
                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)shmem_buffer + (local_rank * count * extent),
                                        &dec_count, max_out_len,
                                        (&tmp_enc_buf[0]), NONCE_SIZE,
                                        tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                        NULL, 0))
                                        {
                                            fprintf(stderr,"Super Node: err Decryption in Child rank = %d source =%d  count = %d\n",comm_ptr->rank, remote_rank,count);
                                        }

            }
            
            // We need to put this barrier here to inform R-leaders that do not wait for 
            // Non-leader processes who do not participate in receiver node, 
            // otherwise R-leaders will wait forever.
            
            MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

        }

#endif        

        leader_cnt = leader_cnt/(ppn+1);
	}


    // Step 4
	/* Broadcasting the message from Super-leader to the rest of leaders */
    if (local_rank == 0){

        // int leader_size = leader_commptr->local_size;
        int leader_rank = leader_commptr->rank;			

		if (leader_rank == 0){  // Super-leader
			// Encrypt(recvbuf,tmp_enc_buf);
            RAND_bytes(tmp_enc_buf, NONCE_SIZE);
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, tmp_enc_buf + NONCE_SIZE,
                                &ciphertext_sendbuf_len, max_out_len,
                                (&tmp_enc_buf[0]), NONCE_SIZE,
                                (char *) recvbuf, count * extent,
                                NULL, 0))
            {
                fprintf(stderr,"Super Node: err Encryption in Bcast rank = %d\n",comm_ptr->rank);        
            }
			MPIR_Bcast_MV2(tmp_enc_buf, enc_size, MPI_CHAR, 0, leader_commptr, errflag);            
		} else {
			MPIR_Bcast_MV2(tmp_enc_buf, enc_size, MPI_CHAR, 0, leader_commptr, errflag);
			// Decrypt(tmp_enc_buf,recvbuf);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                    &dec_count, max_out_len,
                                    (&tmp_enc_buf[0]), NONCE_SIZE,
                                    tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                    NULL, 0))
            {
                fprintf(stderr,"Super Node: err Decryption in Bcast rank = %d source =%d  count = %d\n",comm_ptr->rank, 0,count);
            }
		}
	}
    
  /* Broadcasting the mesage from leader to the rest */
    mpi_errno = MPIR_Shmem_Bcast_MV2(recvbuf, count, datatype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
        
	
#if ALLREDUCE_PRINT_FUN    
    
    if (SHOW_FINAL_RESULT && init_phase==0 && count==128  && count!=65) {    
        
        int s , match=1;
		int namelen1;
		char hostname1[100];
		gethostname(hostname1, &namelen1);
		int step = (count/8)*extent;        

        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,"[%d] dismatch s = %d (%f , %f , %f) host=%s  count=%d  [end]\n",
                comm_ptr->rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+s*extent)),*((float *) (recvbuf+1000*extent)),hostname1,count);
                match =0;
                break;
            }

        }
        if (match==1) {
            fprintf(stderr,"[Super-Node: two_level Rank = %d] is match [value = %f] host=%s  count=%d  [end]\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,"[Super-Node: two_level Rank = %d] recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]\n",comm_ptr->rank, *((float *) (recvbuf+extent)),*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),count,hostname1);
    }
#endif		

  fn_exit:
    MPIR_TIMER_END(coll,allreduce,2lvl);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}




int MPIR_Allreduce_two_level_MV2_Original(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{

#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE && (comm_ptr->rank == 0)) {		
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_two_level_MV2_Original  Allgather_Reduce=%d  count = %d\n",comm_ptr->rank,hostname,Allgather_Reduce,count);fflush(stdout);}
#endif 
	
	
    MPIR_TIMER_START(coll,allreduce,2lvl);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_2lvl, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int total_size = 0;
    MPI_Aint true_lb, true_extent,extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = -1, local_size = 0;

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,2lvl);
        return MPI_SUCCESS;
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    total_size = comm_ptr->local_size;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    PMPI_Comm_size(shmem_comm, &local_size);
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);
	

    if (local_rank == 0) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                       count, datatype);
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
        }
    }	

    /* Doing the shared memory gather and reduction by the leader */
    if (local_rank == 0) {

       if ((MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_shmem_MV2) || 
              (MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_p2p_MV2) ) {
        mpi_errno =
        MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype,
                                     op, comm_ptr, errflag);
        }
        else {
        mpi_errno =
        MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype,
                                     op, shmem_commptr, errflag);
        }

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

        if (local_size != total_size) {
            /* inter-node allreduce */

/* *************************** Mohsen ************************** */		


        if (Allgather_Reduce){

        int leader_size = leader_commptr->local_size;
        int leader_rank = leader_commptr->rank;

		MPI_User_function *uop;
		MPID_Op *op_ptr;
		int is_commutative;
#ifdef HAVE_CXX_BINDING
		int is_cxx_uop = 0;
#endif
		MPIU_THREADPRIV_DECL;
		MPIU_CHKLMEM_DECL(3);

		MPIU_THREADPRIV_GET;

		/* check if multiple threads are calling this collective function */
		MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

		/* homogeneous */

		/* set op_errno to 0. stored in perthread structure */
		MPIU_THREADPRIV_FIELD(op_errno) = 0;
		
		/* Get the operator and check whether it is commutative or not */
		if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
			is_commutative = 1;
			/* get the function by indexing into the op table */
			uop = MPIR_Op_table[op % 16 - 1];
		} else {
			MPID_Op_get_ptr(op, op_ptr);
			if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
				is_commutative = 0;
			} else {
				is_commutative = 1;
			}

#if defined(HAVE_CXX_BINDING)
			if (op_ptr->language == MPID_LANG_CXX) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
				is_cxx_uop = 1;
			} else
#endif                          /* defined(HAVE_CXX_BINDING) */
			if (op_ptr->language == MPID_LANG_C) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
			} else {
				uop = (MPI_User_function *) op_ptr->function.f77_function;
			}
		}	
		
        void *tmp_buf=NULL;
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * leader_size * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");
        tmp_buf = (void *) ((char *) tmp_buf - true_lb);   


            MPIR_Allgather_RD_MV2(recvbuf, count, datatype, tmp_buf, count, datatype, leader_commptr, errflag);
 
            int i;

            int step = (count/8)*extent;

            int stride = count * MPIR_MAX(extent, true_extent);
            void *local_buf = NULL;

            for (i=0; i<leader_size; i++){

                if (i != leader_rank){

                    local_buf = (char *) tmp_buf + stride * i;

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */    

#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (local_buf, recvbuf,
                                                count, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (local_buf, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
                }
            }
		

        }
	
        else 
        {
		
            if(MV2_Allreduce_function == &MPIR_Allreduce_pt2pt_rd_MV2){
                mpi_errno =
                    MPIR_Allreduce_pt2pt_rd_MV2(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                      leader_commptr, errflag);
            } else if (MV2_Allreduce_function == &MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2) {
                mpi_errno =
                    MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2(MPI_IN_PLACE,
                            recvbuf, count, datatype, op, leader_commptr,
                            errflag); 
            } else 
#if defined (_SHARP_SUPPORT_)
            if (MV2_Allreduce_function == &MPIR_Sharp_Allreduce_MV2) {
                mpi_errno =
                    MPIR_Sharp_Allreduce_MV2(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                  comm_ptr , errflag);
                if (mpi_errno != MPI_SUCCESS) {
                    /* fall back to RD algorithm if SHArP is not supported */
                    mpi_errno =
                        MPIR_Allreduce_pt2pt_rd_MV2(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                          leader_commptr, errflag); 
                }
            } else
#endif 
            {
                mpi_errno =
                    MPIR_Allreduce_pt2pt_rs_MV2(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                      leader_commptr, errflag);
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
    else 
    {
		// local_rank != 0    
        /* insert the first reduce here */
        if ((MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_shmem_MV2) || 
              (MV2_Allreduce_intra_function == &MPIR_Allreduce_reduce_p2p_MV2) ) {
        mpi_errno =
        MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype,
                                     op, comm_ptr, errflag);
        }
        else {
        mpi_errno =
        MV2_Allreduce_intra_function(sendbuf, recvbuf, count, datatype,
                                     op, shmem_commptr, errflag);
        }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
    }

    /* Broadcasting the mesage from leader to the rest */
    mpi_errno = MPIR_Shmem_Bcast_MV2(recvbuf, count, datatype, 0, shmem_commptr, errflag);
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
    
	
#if ALLREDUCE_PRINT_FUN    
    
    if (SHOW_FINAL_RESULT && init_phase==0 && count>16  && count!=65) {    
        
        int s , match=1;
		int namelen1;
		char hostname1[100];
		gethostname(hostname1, &namelen1);
		int step = (count/8)*extent;        

        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,"[%d] dismatch s = %d (%f , %f , %f) host=%s  count=%d  [end]\n",
                comm_ptr->rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+s*extent)),*((float *) (recvbuf+1000*extent)),hostname1,count);
                match =0;
                break;
            }

        }
        if (match==1) {
            fprintf(stderr,"[two_level Rank = %d] is match [value = %f] host=%s  count=%d  [end]\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,"[two_level Rank = %d] recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]\n",comm_ptr->rank, *((float *) (recvbuf+extent)),*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),count,hostname1);
    }
#endif		

  

	fn_exit:
    MPIR_TIMER_END(coll,allreduce,2lvl);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;

}


#undef FCNAME
#define FCNAME "MPIR_Allreduce_socket_aware_two_level_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_socket_aware_two_level_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    if (super_node==1 && security_approach==2005)
        return MPIR_Allreduce_socket_aware_two_level_MV2_SEC_Super_Node(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else if (super_node==1 && security_approach==0) 
        return MPIR_Allreduce_socket_aware_two_level_MV2_Unsecure(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_socket_aware_two_level_MV2_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}



#undef FCNAME
#define FCNAME "MPIR_Allreduce_socket_aware_two_level_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_socket_aware_two_level_MV2_SEC_Super_Node(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_socket_aware_two_level_MV2_SEC_Super_Node  count = %d [SUPER_NODE-Sec]\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIU_Assert(comm_ptr->dev.ch.use_intra_sock_comm && comm_ptr->dev.ch.shmem_coll_ok == 1);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_2lvl, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = -1;
    if (count == 0) {
        return MPI_SUCCESS;
    }
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    int intra_sock_rank = -1, intra_sock_size;
    MPID_Comm * intra_sock_commptr;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
    intra_sock_rank = intra_sock_commptr->rank;
    intra_sock_size = intra_sock_commptr->local_size;

    MPIDI_msg_sz_t nbytes;
    MPI_Aint type_size;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    MPI_Aint extent = type_size;
	
    /* Step 1. Socket leaders do an intra-socket reduce using shared memory*/

    if (intra_sock_rank == 0) {
        /* root ranks */
        mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                          op, 0, intra_sock_commptr, errflag);
        if(mpi_errno)
        { 
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

    } else {
        /* non-root ranks */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_shmem_MV2(recvbuf, NULL,  count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        }

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = TRUE;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
	
    /* Step 2. Socket level leaders within the node do an intra-node reduce to rank 0*/
    
    MPID_Comm *shmem_leader_commptr = NULL;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm,shmem_leader_commptr);
    if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Reduce_binomial_MV2(MPI_IN_PLACE,recvbuf,count,datatype,op,0,shmem_leader_commptr,errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 3. Leaders across nodes do an inter-node allreduce */


    // Mohsen
    
    if (1){
        	
        int total_size = comm_ptr->local_size;
        int ppn = shmem_commptr->local_size; // number of ranks per node
        int node = total_size/ppn; 
        if  (total_size % ppn !=0)  node++; // number of nodes
        int enc_size = count * extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;
	
        MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

        unsigned long ciphertext_sendbuf_len, dec_count, max_out_len = enc_size;

        
		MPI_User_function *uop;
		MPID_Op *op_ptr;
		int is_commutative;
#ifdef HAVE_CXX_BINDING
		int is_cxx_uop = 0;
#endif
		MPIU_THREADPRIV_DECL;
		MPIU_CHKLMEM_DECL(3);

		MPIU_THREADPRIV_GET;

		/* check if multiple threads are calling this collective function */
		MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

		/* homogeneous */

		/* set op_errno to 0. stored in perthread structure */
		MPIU_THREADPRIV_FIELD(op_errno) = 0;
		
		/* Get the operator and check whether it is commutative or not */
		if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
			is_commutative = 1;
			/* get the function by indexing into the op table */
			uop = MPIR_Op_table[op % 16 - 1];
		} else {
			MPID_Op_get_ptr(op, op_ptr);
			if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
				is_commutative = 0;
			} else {
				is_commutative = 1;
			}

#if defined(HAVE_CXX_BINDING)
			if (op_ptr->language == MPID_LANG_CXX) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
				is_cxx_uop = 1;
			} else
#endif                          /* defined(HAVE_CXX_BINDING) */
			if (op_ptr->language == MPID_LANG_C) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
			} else {
				uop = (MPI_User_function *) op_ptr->function.f77_function;
			}
		}


    double level;
	level = ceil (log2 (node) / log2 (ppn+1));

    int i, leader_cnt =1;

	for (i=1; i<level; i++)
		leader_cnt = leader_cnt * (ppn+1);

	// Step 3
	while (leader_cnt >= 1){
    
        // Step 1
        int Sender_cnt=0;  // number of leaders who send own data
        if (node > leader_cnt *(ppn+1)) Sender_cnt = leader_cnt * ppn;  //leader_cnt*(ppn+1) - leader_cnt; 
        else Sender_cnt = node - leader_cnt; 

        int list_rank;
        // list_rank = comm_ptr->dev.ch.rank_list_index;
		
		if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
			list_rank = comm_ptr->rank;
			
		} else {
			/* my index on rank_list */
			list_rank = comm_ptr->dev.ch.rank_list_index;			
		}
		
		
        int node_rank = list_rank / ppn;
        
		char hostname[100];
		int namelen;    
		 
		gethostname(hostname, &namelen);

        // Step 2
        int msg_recv_cnt=0;	// number of message each Receiver node will gather

        if (list_rank< (leader_cnt * ppn)){
            
            msg_recv_cnt = (Sender_cnt/leader_cnt);

            if ((Sender_cnt % leader_cnt) > (node_rank)) msg_recv_cnt++;
            
        } else msg_recv_cnt = 0;
        
        if (local_rank == 0) {
        
			if ((node_rank < leader_cnt) && (msg_recv_cnt>0)) { // The Receivers Leaders (R-Leaders)
				
                // Step 4
                int remote_index = (node_rank + leader_cnt) * ppn;
				
                int remote_rank = remote_index;
                
                mpi_errno = MPID_Irecv((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag); 
          
                //Decrypt (tmp_enc_buf, shmem_buffer);
                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)shmem_buffer,
                                        &dec_count, max_out_len,
                                        (&tmp_enc_buf[0]), NONCE_SIZE,
                                        tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                        NULL, 0))
                                        {
                                            fprintf(stderr,"Super Node: err Decryption in R-leader rank = %d source =%d  count = %d\n",comm_ptr->rank, remote_rank,count);
                                        }
                /* shmem_buffer contains data received in this step.
			        recvbuf contains data accumulated so far */ 
                // operation (shmem_buffer,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (shmem_buffer, recvbuf,
                                                    count, datatype, uop);
                } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                    (*uop) (shmem_buffer, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                }
#endif

                MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

                // Step 5
                for (i = 1; i < msg_recv_cnt; i++){
				    // operation (shmem_buffer + i * count ,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (shmem_buffer + (i * (count*extent)), recvbuf,
                                                        count, datatype, uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (shmem_buffer + (i * (count*extent)), recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif
			    }

            } else if (node_rank < leader_cnt + Sender_cnt){

                // Calculate the index of receiver rank
                int remote_node = (node_rank - leader_cnt) % leader_cnt;
                int remote_index = (remote_node * ppn) + ((node_rank - leader_cnt) / leader_cnt);
                // int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;
                
                
                // Step 3
                //Encrypt(recvbuf,tmp_enc_buf);
                RAND_bytes(tmp_enc_buf, NONCE_SIZE);
                if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, tmp_enc_buf + NONCE_SIZE,
                                &ciphertext_sendbuf_len, max_out_len,
                                (&tmp_enc_buf[0]), NONCE_SIZE,
                                (char *) recvbuf, count*extent,
                                NULL, 0))
                {
                    fprintf(stderr,"Super Node: err Encryption in S-leader rank = %d\n",comm_ptr->rank);
                }           
                //mpi_errno = MPID_Send(tmp_enc_buf, enc_size, MPI_CHAR, comm_ptr->dev.ch.rank_list[remote_rank], 12, comm_ptr);
                mpi_errno = MPID_Isend((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank, 12, comm_ptr, context_id, &send_req_ptr);					
            }         
        } else if (node_rank < leader_cnt){ //  Non-leader processes who help R-leaders in decryption (Child)

            // Step 4
            if (local_rank < msg_recv_cnt) {
                
                int remote_index = (leader_cnt * (local_rank + 1) + node_rank) * ppn ;
                //int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;                       
                mpi_errno = MPID_Irecv((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
 
                // Decrypt (tmp_enc_buf, shmem_buf + local_rank * count); // Copy decrypted message to shared memory buffer
                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)shmem_buffer + (local_rank * count * extent),
                                        &dec_count, max_out_len,
                                        (&tmp_enc_buf[0]), NONCE_SIZE,
                                        tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                        NULL, 0))
                                        {
                                            fprintf(stderr,"Super Node: err Decryption in Child rank = %d source =%d  count = %d\n",comm_ptr->rank, remote_rank,count);
                                        }

            }
            
            // We need to put this barrier here to inform R-leaders that do not wait for 
            // Non-leader processes who do not participate in receiver node, 
            // otherwise R-leaders will wait forever.
            
            MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

        }

        leader_cnt = leader_cnt/(ppn+1);
	}


    // Step 4
	/* Broadcasting the message from Super-leader to the rest of leaders */
    if (local_rank == 0){

        // int leader_size = leader_commptr->local_size;
        int leader_rank = leader_commptr->rank;			

		if (leader_rank == 0){  // Super-leader
			// Encrypt(recvbuf,tmp_enc_buf);
            RAND_bytes(tmp_enc_buf, NONCE_SIZE);
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, tmp_enc_buf + NONCE_SIZE,
                                &ciphertext_sendbuf_len, max_out_len,
                                (&tmp_enc_buf[0]), NONCE_SIZE,
                                (char *) recvbuf, count * extent,
                                NULL, 0))
            {
                fprintf(stderr,"Super Node: err Encryption in Bcast rank = %d\n",comm_ptr->rank);        
            }
			MPIR_Bcast_MV2(tmp_enc_buf, enc_size, MPI_CHAR, 0, leader_commptr, errflag);            
		} else {
			MPIR_Bcast_MV2(tmp_enc_buf, enc_size, MPI_CHAR, 0, leader_commptr, errflag);
			// Decrypt(tmp_enc_buf,recvbuf);
            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                    &dec_count, max_out_len,
                                    (&tmp_enc_buf[0]), NONCE_SIZE,
                                    tmp_enc_buf + NONCE_SIZE, count * extent + ENC_MSG_TAG_SIZE,
                                    NULL, 0))
            {
                fprintf(stderr,"Super Node: err Decryption in Bcast rank = %d source =%d  count = %d\n",comm_ptr->rank, 0,count);
            }
		}
	}

    }    
    /* else if(local_rank == 0 && leader_commptr->local_size > 1)
    {
        mv2_use_socket_aware_allreduce = 0;
		
            mpi_errno = MPIR_Allreduce_impl(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                                        leader_commptr, errflag);
            if (mpi_errno) {
               
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }                    
        mv2_use_socket_aware_allreduce = 1;
    } */

    /* Step 4. Rank 0 broadcasts message to socket-leaders within the node */

     if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Bcast_binomial_MV2(recvbuf, count, datatype,0, shmem_leader_commptr, errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 5. All socket leaders do an intra-node shared memory broadcast. Part of the bcast
               shmem code has been taken in due to incompatability between communicator and
               slotted shmem collectives. 
    */

    int shmem_comm_rank;
    shmem_comm_rank = intra_sock_commptr->dev.ch.shmem_comm_rank;
    void *shmem_buf = NULL;

    if (intra_sock_rank == 0) {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, shmem_buf, nbytes, MPI_BYTE);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    } else {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(shmem_buf, nbytes, MPI_BYTE, recvbuf, count, datatype);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    }

    if (mpi_errno) {
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    #if ALLREDUCE_PRINT_FUN    

    if (SHOW_FINAL_RESULT && init_phase==0 && count>64  && count!=65) {    
        
        int s , match=1;
		int namelen1;
		char hostname1[100];
		gethostname(hostname1, &namelen1);
		int step = (count/8)*extent;        

        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,"[two_level Rank = %d] dismatch s = %d (%f , %f , %f) host=%s  count=%d  [end]\n",
                comm_ptr->rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+s*extent)),*((float *) (recvbuf+(s-1)*extent)),hostname1,count);
                match =0;
                break;
            }

        }
        if (match==1) {
            fprintf(stderr,"[SUOER-NODE_socket_aware Rank = %d] is match [value = %f] host=%s  count=%d  [end]\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,"[SUOER-NODE_socket_aware Rank = %d] dismatch recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]\n",comm_ptr->rank, *((float *) (recvbuf+extent)),*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),count,hostname1);
    }
#endif
    
    fn_exit:
        return (mpi_errno);

    fn_fail:
        goto fn_exit;
}



int MPIR_Allreduce_socket_aware_two_level_MV2_Unsecure(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_socket_aware_two_level_MV2_Unsecure  count = %d [SUPER_NODE-Unsec]\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIU_Assert(comm_ptr->dev.ch.use_intra_sock_comm && comm_ptr->dev.ch.shmem_coll_ok == 1);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_2lvl, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = -1;
    if (count == 0) {
        return MPI_SUCCESS;
    }
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    int intra_sock_rank = -1, intra_sock_size;
    MPID_Comm * intra_sock_commptr;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
    intra_sock_rank = intra_sock_commptr->rank;
    intra_sock_size = intra_sock_commptr->local_size;

    MPIDI_msg_sz_t nbytes;
    MPI_Aint type_size;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    MPI_Aint extent = type_size;
	
    /* Step 1. Socket leaders do an intra-socket reduce using shared memory*/

    if (intra_sock_rank == 0) {
        /* root ranks */
        mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                          op, 0, intra_sock_commptr, errflag);
        if(mpi_errno)
        { 
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

    } else {
        /* non-root ranks */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_shmem_MV2(recvbuf, NULL,  count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        }

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = TRUE;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
	
    /* Step 2. Socket level leaders within the node do an intra-node reduce to rank 0*/
    
    MPID_Comm *shmem_leader_commptr = NULL;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm,shmem_leader_commptr);
    if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Reduce_binomial_MV2(MPI_IN_PLACE,recvbuf,count,datatype,op,0,shmem_leader_commptr,errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 3. Leaders across nodes do an inter-node allreduce */


    // Mohsen
    
    if (1){
        	
#if 1
        int total_size = comm_ptr->local_size;
        int ppn = shmem_commptr->local_size; // number of ranks per node
        int node = total_size/ppn; 
        if  (total_size % ppn !=0)  node++; // number of nodes
        int enc_size = count * extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;
	
        MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

        unsigned long ciphertext_sendbuf_len, dec_count, max_out_len = enc_size;

        
		MPI_User_function *uop;
		MPID_Op *op_ptr;
		int is_commutative;
#ifdef HAVE_CXX_BINDING
		int is_cxx_uop = 0;
#endif
		MPIU_THREADPRIV_DECL;
		MPIU_CHKLMEM_DECL(3);

		MPIU_THREADPRIV_GET;

		/* check if multiple threads are calling this collective function */
		MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

		/* homogeneous */

		/* set op_errno to 0. stored in perthread structure */
		MPIU_THREADPRIV_FIELD(op_errno) = 0;
		
		/* Get the operator and check whether it is commutative or not */
		if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
			is_commutative = 1;
			/* get the function by indexing into the op table */
			uop = MPIR_Op_table[op % 16 - 1];
		} else {
			MPID_Op_get_ptr(op, op_ptr);
			if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
				is_commutative = 0;
			} else {
				is_commutative = 1;
			}

#if defined(HAVE_CXX_BINDING)
			if (op_ptr->language == MPID_LANG_CXX) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
				is_cxx_uop = 1;
			} else
#endif                          /* defined(HAVE_CXX_BINDING) */
			if (op_ptr->language == MPID_LANG_C) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
			} else {
				uop = (MPI_User_function *) op_ptr->function.f77_function;
			}
		}
           
#endif  
	
    


    double level;
	level = ceil (log2 (node) / log2 (ppn+1));

    int i, leader_cnt =1;

	for (i=1; i<level; i++)
		leader_cnt = leader_cnt * (ppn+1);

	// Step 3
	while (leader_cnt >= 1){
    
        // Step 1
        int Sender_cnt=0;  // number of leaders who send own data
        if (node > leader_cnt *(ppn+1)) Sender_cnt = leader_cnt * ppn;  //leader_cnt*(ppn+1) - leader_cnt; 
        else Sender_cnt = node - leader_cnt; 

        int list_rank;
        // list_rank = comm_ptr->dev.ch.rank_list_index;
		
		if (comm_ptr->dev.ch.rank_list == NULL || is_commutative != 1) {
			list_rank = comm_ptr->rank;			
		} else {
			/* my index on rank_list */
			list_rank = comm_ptr->dev.ch.rank_list_index;			
		}
		
		
        int node_rank = list_rank / ppn;
        
		char hostname[100];
		int namelen;    
		 
		gethostname(hostname, &namelen);

        // Step 2
        int msg_recv_cnt=0;	// number of message each Receiver node will gather

        if (list_rank< (leader_cnt * ppn)){
            
            msg_recv_cnt = (Sender_cnt/leader_cnt);

            if ((Sender_cnt % leader_cnt) > (node_rank)) msg_recv_cnt++;
            
        } else msg_recv_cnt = 0;
        
        if (local_rank == 0) {
        
			if ((node_rank < leader_cnt) && (msg_recv_cnt>0)) { // The Receivers Leaders (R-Leaders)
				
                // Step 4
                int remote_index = (node_rank + leader_cnt) * ppn;
				    
                // int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;
                      
                mpi_errno = MPID_Irecv((char *) shmem_buffer, count * extent, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag); 
          
                /* shmem_buffer contains data received in this step.
			        recvbuf contains data accumulated so far */ 
                // operation (shmem_buffer,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                if (is_cxx_uop) {
                    (*MPIR_Process.cxx_call_op_fn) (shmem_buffer, recvbuf,
                                                    count, datatype, uop);
                } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                    (*uop) (shmem_buffer, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                }
#endif

                MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

                // Step 5
                for (i = 1; i < msg_recv_cnt; i++){
				    // operation (shmem_buffer + i * count ,recvbuf,count, datatype, op);
#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (shmem_buffer + (i * (count*extent)), recvbuf,
                                                        count, datatype, uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (shmem_buffer + (i * (count*extent)), recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif
			    }

            } else if (node_rank < leader_cnt + Sender_cnt){

                // Calculate the index of receiver rank
                int remote_node = (node_rank - leader_cnt) % leader_cnt;
                int remote_index = (remote_node * ppn) + ((node_rank - leader_cnt) / leader_cnt);
                // int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;
                
                
                // Step 3				
                mpi_errno = MPID_Isend((char *) recvbuf, count * extent, MPI_CHAR, remote_rank, 12, comm_ptr, context_id, &send_req_ptr);					
            }         
        } else if (node_rank < leader_cnt){ //  Non-leader processes who help R-leaders in decryption (Child)

            // Step 4
            if (local_rank < msg_recv_cnt) {
                
                int remote_index = (leader_cnt * (local_rank + 1) + node_rank) * ppn ;
                //int remote_rank = comm_ptr->dev.ch.rank_list[remote_index];
                int remote_rank = remote_index;                       
                //mpi_errno = MPID_Irecv((char *) tmp_enc_buf, enc_size, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 
                mpi_errno = MPID_Irecv((char *) shmem_buffer + (local_rank * count * extent), count * extent, MPI_CHAR, remote_rank , 12, comm_ptr, context_id, &recv_req_ptr); 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
 
            }
            
            // We need to put this barrier here to inform R-leaders that do not wait for 
            // Non-leader processes who do not participate in receiver node, 
            // otherwise R-leaders will wait forever.
            
            MPIR_Barrier_impl(comm_ptr->node_comm, errflag);

        }

        leader_cnt = leader_cnt/(ppn+1);
	}


    // Step 4
	/* Broadcasting the message from Super-leader to the rest of leaders */
    if (local_rank == 0){

        // int leader_size = leader_commptr->local_size;
        int leader_rank = leader_commptr->rank;			

		if (leader_rank == 0){  // Super-leader
			
			MPIR_Bcast_MV2(recvbuf, count * extent, MPI_CHAR, 0, leader_commptr, errflag);            
		} else {
			
			MPIR_Bcast_MV2(recvbuf, count * extent, MPI_CHAR, 0, leader_commptr, errflag);
			
		}
	}

    }    
    else if(local_rank == 0 && leader_commptr->local_size > 1)
    {
        mv2_use_socket_aware_allreduce = 0;

        mpi_errno = MPIR_Allreduce_impl(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                                    leader_commptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }                    
        mv2_use_socket_aware_allreduce = 1;
    }

    /* Step 4. Rank 0 broadcasts message to socket-leaders within the node */

 
    if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Bcast_binomial_MV2(recvbuf, count, datatype,0, shmem_leader_commptr, errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 5. All socket leaders do an intra-node shared memory broadcast. Part of the bcast
               shmem code has been taken in due to incompatability between communicator and
               slotted shmem collectives. 
    */

    int shmem_comm_rank;
    shmem_comm_rank = intra_sock_commptr->dev.ch.shmem_comm_rank;
    void *shmem_buf = NULL;

    if (intra_sock_rank == 0) {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, shmem_buf, nbytes, MPI_BYTE);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    } else {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(shmem_buf, nbytes, MPI_BYTE, recvbuf, count, datatype);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    }

    if (mpi_errno) {
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    #if ALLREDUCE_PRINT_FUN    

    if (SHOW_FINAL_RESULT && init_phase==0 && count>64  && count!=65) {    
        
        int s , match=1;
		int namelen1;
		char hostname1[100];
		gethostname(hostname1, &namelen1);
		int step = (count/8)*extent;        

        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,"[two_level Rank = %d] dismatch s = %d (%f , %f , %f) host=%s  count=%d  [end]\n",
                comm_ptr->rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+s*extent)),*((float *) (recvbuf+(s-1)*extent)),hostname1,count);
                match =0;
                break;
            }

        }
        if (match==1) {
            fprintf(stderr,"[SUOER-NODE_socket_aware Rank = %d] is match [value = %f] host=%s  count=%d  [end]\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,"[SUOER-NODE_socket_aware Rank = %d] dismatch recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]\n",comm_ptr->rank, *((float *) (recvbuf+extent)),*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),count,hostname1);
    }
#endif
    
    fn_exit:
        return (mpi_errno);

    fn_fail:
        goto fn_exit;
}




int MPIR_Allreduce_socket_aware_two_level_MV2_Original(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_socket_aware_two_level_MV2_Original  Allgather_Reduce=%d  count = %d\n",comm_ptr->rank,hostname,Allgather_Reduce,count);fflush(stdout);}
#endif     
    MPIU_Assert(comm_ptr->dev.ch.use_intra_sock_comm && comm_ptr->dev.ch.shmem_coll_ok == 1);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_2lvl, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = -1;
    if (count == 0) {
        return MPI_SUCCESS;
    }
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    int intra_sock_rank = -1, intra_sock_size;
    MPID_Comm * intra_sock_commptr;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_comm, intra_sock_commptr);
    intra_sock_rank = intra_sock_commptr->rank;
    intra_sock_size = intra_sock_commptr->local_size;

    MPIDI_msg_sz_t nbytes;
    MPI_Aint type_size;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = (MPIDI_msg_sz_t) (count) * (type_size);
    MPI_Aint extent = type_size;
	
    /* Step 1. Socket leaders do an intra-socket reduce using shared memory*/

    if (intra_sock_rank == 0) {
        /* root ranks */
        mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                          op, 0, intra_sock_commptr, errflag);
        if(mpi_errno)
        { 
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }

    } else {
        /* non-root ranks */
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Reduce_shmem_MV2(sendbuf, recvbuf, count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        } else {
            mpi_errno = MPIR_Reduce_shmem_MV2(recvbuf, NULL,  count, datatype, 
                                              op, 0, intra_sock_commptr, errflag);
        }

        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = TRUE;
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }
	
    /* Step 2. Socket level leaders within the node do an intra-node reduce to rank 0*/
    
    MPID_Comm *shmem_leader_commptr = NULL;
    MPID_Comm_get_ptr(shmem_commptr->dev.ch.intra_sock_leader_comm,shmem_leader_commptr);
    if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Reduce_binomial_MV2(MPI_IN_PLACE,recvbuf,count,datatype,op,0,shmem_leader_commptr,errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 3. Leaders across nodes do an inter-node allreduce */

    if(local_rank == 0 && leader_commptr->local_size > 1)
    {
        mv2_use_socket_aware_allreduce = 0;
		
		
		// Mohsen _ socket_aware_two_level_MV2 (High Iteration)
        if (Allgather_Reduce){
	
        int leader_size = leader_commptr->local_size;
        int leader_rank = leader_commptr->rank;

		MPI_User_function *uop;
		MPID_Op *op_ptr;
		int is_commutative;
#ifdef HAVE_CXX_BINDING
		int is_cxx_uop = 0;
#endif
		MPIU_THREADPRIV_DECL;
		MPIU_CHKLMEM_DECL(3);

		MPIU_THREADPRIV_GET;

		/* check if multiple threads are calling this collective function */
		MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

		/* homogeneous */

		/* set op_errno to 0. stored in perthread structure */
		MPIU_THREADPRIV_FIELD(op_errno) = 0;
		
		/* Get the operator and check whether it is commutative or not */
		if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
			is_commutative = 1;
			/* get the function by indexing into the op table */
			uop = MPIR_Op_table[op % 16 - 1];
		} else {
			MPID_Op_get_ptr(op, op_ptr);
			if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
				is_commutative = 0;
			} else {
				is_commutative = 1;
			}

#if defined(HAVE_CXX_BINDING)
			if (op_ptr->language == MPID_LANG_CXX) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
				is_cxx_uop = 1;
			} else
#endif                          /* defined(HAVE_CXX_BINDING) */
			if (op_ptr->language == MPID_LANG_C) {
				uop = (MPI_User_function *) op_ptr->function.c_function;
			} else {
				uop = (MPI_User_function *) op_ptr->function.f77_function;
			}
		}
		
        void *tmp_buf=NULL;
        
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                        count * leader_size * (MPIR_MAX(extent, true_extent)), mpi_errno,
                        "temporary buffer");
        tmp_buf = (void *) ((char *) tmp_buf - true_lb);   

			
            MPIR_Allgather_MV2(recvbuf, count, datatype, tmp_buf, count, datatype, leader_commptr, errflag);
            
            int i;

            int step = (count/8)*extent;

            int stride = count * MPIR_MAX(extent, true_extent);
            void *local_buf = NULL;

            for (i=0; i<leader_size; i++){

                if (i != leader_rank){

                    local_buf = (char *) tmp_buf + stride * i;

                /* tmp_buf contains data received in this step.
                   recvbuf contains data accumulated so far */    

#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (local_buf, recvbuf,
                                                count, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (local_buf, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
                }
            }
 
        }	
        else 
        {
            mpi_errno = MPIR_Allreduce_impl(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                                        leader_commptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }            
        }
        mv2_use_socket_aware_allreduce = 1;
    }

    /* Step 4. Rank 0 broadcasts message to socket-leaders within the node */

     if(intra_sock_rank == 0)
    {
        // Binomial seemed to give the best performance for intra-node socket leaders.
        mpi_errno = MPIR_Bcast_binomial_MV2(recvbuf, count, datatype,0, shmem_leader_commptr, errflag);
        if (mpi_errno) {
             /* for communication errors, just record the error but continue */
             *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
             MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
             MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

    /* Step 5. All socket leaders do an intra-node shared memory broadcast. Part of the bcast
               shmem code has been taken in due to incompatability between communicator and
               slotted shmem collectives. 
    */

    int shmem_comm_rank;
    shmem_comm_rank = intra_sock_commptr->dev.ch.shmem_comm_rank;
    void *shmem_buf = NULL;

    if (intra_sock_rank == 0) {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, shmem_buf, nbytes, MPI_BYTE);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    } else {
        MPIDI_CH3I_SHMEM_Bcast_GetBuf(intra_sock_size, intra_sock_rank,
                                      shmem_comm_rank, (void *) &shmem_buf);
        mpi_errno = MPIR_Localcopy(shmem_buf, nbytes, MPI_BYTE, recvbuf, count, datatype);
        MPIDI_CH3I_SHMEM_Bcast_Complete(intra_sock_size, intra_sock_rank, shmem_comm_rank);
    }

    if (mpi_errno) {
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    #if ALLREDUCE_PRINT_FUN    
    //if (SHOW_FINAL_RESULT && count==64) {
    //if (Print_Ring && SHOW_FINAL_RESULT && init_phase==0 && count>512) {
    if (0 && SHOW_FINAL_RESULT && init_phase==0 && count>16  && count!=65) {    
        
        int s , match=1;
		int namelen1;
		char hostname1[100];
		gethostname(hostname1, &namelen1);
		int step = (count/8)*extent;        

        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,"[two_level Rank = %d] dismatch s = %d (%f , %f , %f) host=%s  count=%d  [end]\n",
                comm_ptr->rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+s*extent)),*((float *) (recvbuf+1000*extent)),hostname1,count);
                match =0;
                break;
            }

        }
        if (match==1) {
            fprintf(stderr,"[two_level Rank = %d] is match [value = %f] host=%s  count=%d  [end]\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,"[two_level Rank = %d] dismatch recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]\n",comm_ptr->rank, *((float *) (recvbuf+extent)),*((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),count,hostname1);
    }
#endif
    
    fn_exit:
        return (mpi_errno);

    fn_fail:
        goto fn_exit;
}



int MPIR_Allreduce_shmem_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        int my_comm_size = comm_ptr->local_size;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_shmem_MV2  comm_size = %d  count = %d\n",comm_ptr->rank,hostname,my_comm_size,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,shmem);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_shmem, 1);
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int i = 0, is_commutative = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    char *shmem_buf = NULL;
    MPI_Comm shmem_comm = MPI_COMM_NULL, leader_comm = MPI_COMM_NULL;
    MPID_Comm *shmem_commptr = NULL, *leader_commptr = NULL;
    int local_rank = -1, local_size = 0;
    void *local_buf = NULL;
    int stride = 0;
    is_commutative = 0;
    int total_size, shmem_comm_rank;

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);

    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }

    total_size = comm_ptr->local_size;
    shmem_comm = comm_ptr->dev.ch.shmem_comm;
    PMPI_Comm_size(shmem_comm, &local_size);
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    local_rank = shmem_commptr->rank;
    local_size = shmem_commptr->local_size;
    shmem_comm_rank = shmem_commptr->dev.ch.shmem_comm_rank;

    leader_comm = comm_ptr->dev.ch.leader_comm;
    MPID_Comm_get_ptr(leader_comm, leader_commptr);

    if (local_rank == 0) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                       count, datatype);
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
        }
    }

    /* Doing the shared memory gather and reduction by the leader */
    if (local_rank == 0) {
        if (stride <= mv2_coll_param.shmem_allreduce_msg) {
            /* Message size is smaller than the shmem_reduce threshold. 
             * The intra-node communication is done through shmem */
            if (local_size > 1) {
                /* Node leader waits till all the non-leaders have written 
                 * the data into the shmem buffer */
                MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                                  shmem_comm_rank,
                                                  (void *) &shmem_buf);
                if (is_commutative) {
                    for (i = 1; i < local_size; i++) {
                        local_buf = (char *) shmem_buf + stride * i;
#if defined(HAVE_CXX_BINDING)
                        if (is_cxx_uop) {
                            (*MPIR_Process.cxx_call_op_fn) (local_buf, recvbuf,
                                                            count, datatype,
                                                            uop);
                        } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                            (*uop) (local_buf, recvbuf, &count, &datatype);
#if defined(HAVE_CXX_BINDING)
                        }
#endif                          /* defined(HAVE_CXX_BINDING) */

                    }
                    MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size,
                                                            local_rank,
                                                            shmem_comm_rank);
                }
            }
        } else {
            /* Message size is larger than the shmem_reduce threshold. 
             * The leader will spend too much time doing the math operation
             * for messages that are larger. So, we use a point-to-point
             * based reduce to balance the load across all the processes within
             * the same node*/
            mpi_errno =
                MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype, op, 0,
                                shmem_commptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

        }
        if (local_size != total_size) {
            mpi_errno =
                MPIR_Allreduce_MV2(MPI_IN_PLACE, recvbuf, count, datatype, op,
                                   leader_commptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } else {
        if (stride <= mv2_coll_param.shmem_allreduce_msg) {
            MPIDI_CH3I_SHMEM_COLL_GetShmemBuf(local_size, local_rank,
                                              shmem_comm_rank,
                                              (void *) &shmem_buf);
            local_buf = (char *) shmem_buf + stride * local_rank;
            if (sendbuf != MPI_IN_PLACE) {
                mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, local_buf,
                                           count, datatype);
            } else {
                mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, local_buf,
                                           count, datatype);
            }
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
            MPIDI_CH3I_SHMEM_COLL_SetGatherComplete(local_size, local_rank,
                                                    shmem_comm_rank);
        } else {
            if (sendbuf != MPI_IN_PLACE) {
                mpi_errno =
                    MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype, op, 0,
                                    shmem_commptr, errflag);
            } else {
                /* MPI_Allreduce was called with MPI_IN_PLACE as the sendbuf.
                 * Since we are doing Reduce now, we need to be careful. In
                 * MPI_Reduce, only the root can use MPI_IN_PLACE as sendbuf.
                 * Also, the recvbuf is not relevant at all non-root processes*/
                mpi_errno = MPIR_Reduce_MV2(recvbuf, NULL, count, datatype, op,
                                            0, shmem_commptr, errflag);
            }
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    }

    /* Broadcasting the mesage from leader to the rest */
    /* Note: shared memory broadcast could improve the performance */
    if (local_size > 1) {
        MPIR_Bcast_MV2(recvbuf, count, datatype, 0, shmem_commptr, errflag);
        if (mpi_errno) {
            /* for communication errors, just record the error but continue */
            *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
            MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
        }
    }

  fn_exit:
    MPIR_TIMER_END(coll,allreduce,shmem);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}


#if defined(_MCST_SUPPORT_)
#undef FCNAME
#define FCNAME "MPIR_Allreduce_mcst_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_mcst_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_mcst_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,mcast);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_mcast, 1);
    MPI_Aint true_lb, true_extent;
   /*We use reduce (at rank =0) followed by mcst-bcast to implement the 
    * allreduce operation */
    int root=0, nbytes=0;
    MPI_Aint type_size=0, position=0; 
    int mpi_errno=MPI_SUCCESS;
    int mpi_errno_ret=MPI_SUCCESS;
    int rank = comm_ptr->rank, is_contig=0, is_commutative=0;
    MPIU_CHKLMEM_DECL(1);
    MPID_Datatype *dtp=NULL;
    void *tmp_buf=NULL; 
    MPID_Op *op_ptr=NULL;
    MPID_Datatype_get_size_macro(datatype, type_size);
    nbytes = type_size * count;

 
    if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN) { 
        is_contig = 1;
    } else {
        MPID_Datatype_get_ptr(datatype, dtp);
        is_contig = dtp->is_contig;
    }

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

   if(is_commutative == 0) { 
       reduce_fn = &MPIR_Reduce_binomial_MV2; 
   } else { 
       if(MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2) {
            reduce_fn = &MPIR_Reduce_MV2;
       } else {
            reduce_fn = &MPIR_Reduce_redscat_gather_MV2;
       } 
   } 

    /* First do a reduction at rank = 0 */
    if(rank == root) {
        mpi_errno = reduce_fn(sendbuf, recvbuf, count, datatype,
                                op, root, comm_ptr, errflag);
    } else {
        if(sendbuf != MPI_IN_PLACE) {
            mpi_errno = reduce_fn(sendbuf, recvbuf, count, datatype,
                                op, root, comm_ptr, errflag);
        } else {
            mpi_errno = reduce_fn(recvbuf, NULL, count, datatype,
                                op, root, comm_ptr, errflag);
        }
    }
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }

    /* Now do a mcst-bcast operation with rank0 as the root */
    if(!is_contig) {
        /* Mcast cannot handle non-regular datatypes. We need to pack
         * as bytes before sending it*/ 
        MPIU_CHKLMEM_MALLOC(tmp_buf, void *, nbytes, mpi_errno, "tmp_buf");

        position = 0;
        if (rank == root) {
            mpi_errno = MPIR_Pack_impl(recvbuf, count, datatype, tmp_buf, nbytes,
                                       &position);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = MPIR_Mcast_inter_node_MV2(tmp_buf, nbytes, MPI_BYTE,
                                     root, comm_ptr, errflag);
    } else { 
        mpi_errno = MPIR_Mcast_inter_node_MV2(recvbuf, count, datatype,
                                     root, comm_ptr, errflag);
    } 
   
    if (mpi_errno) {
        /* for communication errors, just record the error but continue */
        *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
        MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
        MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
    }
    
    if (!is_contig) {
        /* We are done, lets pack the data back the way the user 
         * needs it */ 
        if (rank != root) {
            position = 0;
            mpi_errno = MPIR_Unpack_impl(tmp_buf, nbytes, &position, recvbuf,
                                         count, datatype);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    /* check to see if the intra-node mcast is not done. 
     * if this is the case, do it either through shmem or knomial */ 
    if(comm_ptr->dev.ch.intra_node_done == 0) { 
        MPID_Comm *shmem_commptr=NULL; 
        MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_commptr); 
        int local_size = shmem_commptr->local_size; 
        if (local_size > 1) {
            MPIR_Bcast_MV2(recvbuf, count, datatype, 0, shmem_commptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
        }
    } 


  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,mcast);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}
#endif /*  #if defined(_MCST_SUPPORT_) */ 

#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_new_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_new_MV2(const void *sendbuf,
                       void *recvbuf,
                       int count,
                       MPI_Datatype datatype,
                       MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_new_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
#ifdef MPID_HAS_HETERO
    int rc;
    int is_homogeneous = 1;
#endif

    int mpi_errno = MPI_SUCCESS;
    int rank = 0, comm_size = 0;
   
    mpi_errno = PMPI_Comm_size(comm_ptr->handle, &comm_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    MPI_Aint sendtype_size = 0;
    int nbytes = 0;
    int range = 0, range_threshold = 0, range_threshold_intra = 0;
    int is_two_level = 0;
    int is_commutative = 0;
    MPI_Aint true_lb, true_extent;

    MPID_Datatype_get_size_macro(datatype, sendtype_size);
    nbytes = count * sendtype_size;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Op *op_ptr;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

#ifdef _ENABLE_CUDA_
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);
    int stride = 0;
    stride = count * MPIR_MAX(extent, true_extent);
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;
    const char *temp_sendbuf = sendbuf;

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(recvbuf);
       if ( sendbuf != MPI_IN_PLACE ){
           send_mem_type = is_device_buffer(sendbuf);
       }
    }

    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)sendbuf, 
                            stride, 
                            cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
                            (void *)recvbuf, 
                            stride, 
                            cudaMemcpyDeviceToHost);
        recvbuf = recv_host_buf;
    }
#endif

#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero) {
        is_homogeneous = 0;
    }

    if (!is_homogeneous) {
        /* heterogeneous. To get the same result on all processes, we
           do a reduce to 0 and then broadcast. */
        mpi_errno = MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype,
                                    op, 0, comm_ptr, errflag);
        /* 
           FIXME: mpi_errno is error CODE, not necessarily the error
           class MPI_ERR_OP.  In MPICH2, we can get the error class 
           with errorclass = mpi_errno & ERROR_CLASS_MASK;
         */
        if (mpi_errno == MPI_ERR_OP || mpi_errno == MPI_SUCCESS) {
            /* Allow MPI_ERR_OP since we can continue from this error */
            rc = MPIR_Bcast_impl(recvbuf, count, datatype, 0, comm_ptr,
                                 errflag);
            if (rc)
                mpi_errno = rc;
        }
    } else
#endif /* MPID_HAS_HETERO */
    {
        /* Search for the corresponding system size inside the tuning table */
        while ((range < (mv2_size_allreduce_tuning_table - 1)) &&
               (comm_size > mv2_allreduce_thresholds_table[range].numproc)) {
            range++;
        }
        /* Search for corresponding inter-leader function */
        /* skip mcast poiters if mcast is not available */
        if(mv2_allreduce_thresholds_table[range].mcast_enabled != 1){
            while ((range_threshold < (mv2_allreduce_thresholds_table[range].size_inter_table - 1)) 
                    && ((mv2_allreduce_thresholds_table[range].
                    inter_leader[range_threshold].MV2_pt_Allreduce_function 
                    == &MPIR_Allreduce_mcst_reduce_redscat_gather_MV2) ||
                    (mv2_allreduce_thresholds_table[range].
                    inter_leader[range_threshold].MV2_pt_Allreduce_function
                    == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2)
                    )) {
                    range_threshold++;
            }
        }
        while ((range_threshold < (mv2_allreduce_thresholds_table[range].size_inter_table - 1))
               && (nbytes >
               mv2_allreduce_thresholds_table[range].inter_leader[range_threshold].max)
               && (mv2_allreduce_thresholds_table[range].inter_leader[range_threshold].max != -1)) {
               range_threshold++;
        }
        if(mv2_allreduce_thresholds_table[range].is_two_level_allreduce[range_threshold] == 1){
               is_two_level = 1;    
        }
        /* Search for corresponding intra-node function */
        while ((range_threshold_intra <
               (mv2_allreduce_thresholds_table[range].size_intra_table - 1))
                && (nbytes >
                mv2_allreduce_thresholds_table[range].intra_node[range_threshold_intra].max)
                && (mv2_allreduce_thresholds_table[range].intra_node[range_threshold_intra].max !=
                -1)) {
                range_threshold_intra++;
        }

        MV2_Allreduce_function = mv2_allreduce_thresholds_table[range].inter_leader[range_threshold]
                                .MV2_pt_Allreduce_function;

        MV2_Allreduce_intra_function = mv2_allreduce_thresholds_table[range].intra_node[range_threshold_intra]
                                .MV2_pt_Allreduce_function;

        /* check if mcast is ready, otherwise replace mcast with other algorithm */
        if((MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_redscat_gather_MV2)||
          (MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2)){
#if defined(_MCST_SUPPORT_)
            /* fall back to RD algorithm if:
             *  1) two level is disabled
             *  2) mcast is not ready or supported */
            if(is_two_level != 1 ||
                mv2_use_mcast_allreduce != 1 ||
                comm_ptr->dev.ch.is_mcast_ok != 1 ||
                comm_ptr->dev.ch.shmem_coll_ok !=1)
#endif  /* #if defined(_MCST_SUPPORT_) */
            {
                MV2_Allreduce_function = &MPIR_Allreduce_pt2pt_rd_MV2;
            }
        } 

        if(is_two_level == 1){
#if defined(_MCST_SUPPORT_)
            if((MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_redscat_gather_MV2)||
            (MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2)){ 

                mpi_errno = MPIR_Allreduce_mcst_MV2(sendbuf, recvbuf, count,
                                               datatype, op, comm_ptr, errflag);
            } else
#endif  /* #if defined(_MCST_SUPPORT_) */
            { 
                /* check if shm is ready, if not use other algorithm first */
                if ((comm_ptr->dev.ch.shmem_coll_ok == 1)
                    && (mv2_enable_shmem_allreduce)
                    && (is_commutative)
                    && (mv2_enable_shmem_collectives)) {
                    mpi_errno = MPIR_Allreduce_two_level_MV2(sendbuf, recvbuf, count,
                                                     datatype, op, comm_ptr, errflag);
                } else {
                    mpi_errno = MPIR_Allreduce_pt2pt_rd_MV2(sendbuf, recvbuf, count,
                                                     datatype, op, comm_ptr, errflag);
                }
            }
        } else { 
            mpi_errno = MV2_Allreduce_function(sendbuf, recvbuf, count,
                                           datatype, op, comm_ptr, errflag);
        }
    } 

#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type){
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
                            (void *)recv_host_buf, 
                            stride, 
                            cudaMemcpyHostToDevice);
    }
    if(rdma_enable_cuda && recv_mem_type){
        if(recv_host_buf){
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if(rdma_enable_cuda && send_mem_type){
        sendbuf = temp_sendbuf;
        if(send_host_buf){
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif
	comm_ptr->dev.ch.intra_node_done=0;
	
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    return (mpi_errno);

  fn_fail:
    goto fn_exit;

}


#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_index_tuned_intra_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_index_tuned_intra_MV2(const void *sendbuf,
                       void *recvbuf,
                       int count,
                       MPI_Datatype datatype,
                       MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_index_tuned_intra_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
#ifdef MPID_HAS_HETERO
    int rc;
    int is_homogeneous = 1;
#endif

    int mpi_errno = MPI_SUCCESS;
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
    int rank = 0, comm_size = 0;
    MPI_Aint sendtype_size = 0;
    int nbytes = 0;
    int is_two_level = 0;
    int is_commutative = 0;
    MPI_Aint true_lb, true_extent;
   
    mpi_errno = PMPI_Comm_size(comm_ptr->handle, &comm_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_rank(comm_ptr->handle, &rank);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    MPID_Datatype_get_size_macro(datatype, sendtype_size);
    nbytes = count * sendtype_size;

#if defined (_SHARP_SUPPORT_)
    int is_socket_aware = mv2_enable_socket_aware_collectives
                          && mv2_use_socket_aware_allreduce
                          && mv2_use_socket_aware_sharp_allreduce;

    //If is_socket_aware == 1 then Sharp allreduce will be called from within the socket-aware 
    //allreduce function
    if (comm_ptr->dev.ch.is_sharp_ok == 1 && nbytes <= mv2_sharp_tuned_msg_size
        && mv2_enable_sharp_coll == 2 && !is_socket_aware) {
        /* Direct flat algorithm in which every process calls Sharp
         * MV2_ENABLE_SHARP should be set to 2 */
        mpi_errno = MPIR_Sharp_Allreduce_MV2(sendbuf, recvbuf, count,
                                             datatype, op, comm_ptr, errflag);
        if (mpi_errno == MPI_SUCCESS) {
            goto fn_exit;
        }
        /* SHArP collective is not supported, continue without using SHArP */
    }
#endif /* end of defined (_SHARP_SUPPORT_) */

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Op *op_ptr;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

#ifdef _ENABLE_CUDA_
    MPI_Aint extent;
    MPID_Datatype_get_extent_macro(datatype, extent);
    int stride = 0;
    stride = count * MPIR_MAX(extent, true_extent);
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;

    if (rdma_enable_cuda) {
	recv_mem_type = is_device_buffer(recvbuf);
	if ( sendbuf != MPI_IN_PLACE ){
	    send_mem_type = is_device_buffer(sendbuf);
	}
    }

    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
			 (void *)sendbuf, 
			 stride, 
			 cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
			 (void *)recvbuf, 
			 stride, 
			 cudaMemcpyDeviceToHost);
        recvbuf = recv_host_buf;
    }
#endif
    
    /* check if safe to use partial subscription mode */
    if (comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.is_uniform) {
    
        shmem_comm = comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
        local_size = shmem_commptr->local_size;
        if (mv2_allreduce_indexed_table_ppn_conf[0] == -1) {
            // Indicating user defined tuning
            conf_index = 0;
            goto conf_check_end;
        }
		
		
		///Mohsen
        if (mv2_allred_use_ring == 1 && mv2_allreduce_ring_algo_threshold <=  nbytes && local_size <= mv2_allreduce_ring_algo_ppn_threshold) {
        //if (mv2_allred_use_ring == 1 && mv2_allreduce_ring_algo_threshold <=  nbytes && local_size <= 16) {
            /* flat ring-based Allreduce */
			
            mpi_errno = MPIR_Allreduce_pt2pt_ring_wrapper_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag, comm_size, sendtype_size);
				
				return mpi_errno;
        }

        if(nbytes <= mv2_socket_aware_allreduce_max_msg && nbytes >= mv2_socket_aware_allreduce_min_msg 
           && mv2_enable_socket_aware_collectives
           && mv2_use_socket_aware_allreduce && comm_ptr->dev.ch.shmem_coll_ok == 1
           && comm_ptr->dev.ch.allreduce_coll_count >= shmem_coll_count_threshold
           && comm_ptr->dev.ch.use_intra_sock_comm == 1
           && is_commutative) {

            mpi_errno = MPIR_Allreduce_socket_aware_two_level_MV2(sendbuf, recvbuf, count, datatype, op,
                                                                  comm_ptr, errflag);
            return mpi_errno;
        }

        if (likely(mv2_enable_shmem_allreduce && mv2_enable_skip_tuning_table_search && (nbytes <= mv2_coll_skip_table_threshold))) {
            /* for small messages, force Shmem + RD */
            MV2_Allreduce_intra_function = MPIR_Allreduce_reduce_shmem_MV2;
            MV2_Allreduce_function = MPIR_Allreduce_pt2pt_rd_MV2;
            is_two_level = 1;
            goto skip_tuning_tables;
        }

        FIND_PPN_INDEX  (allreduce, local_size,conf_index, partial_sub_ok)

    }

    if (partial_sub_ok != 1) {
        conf_index = mv2_allreduce_indexed_num_ppn_conf/2;
    }

 ///Mohsen
    if ((comm_ptr->dev.ch.allgather_comm_ok != 0 &&
          comm_ptr->dev.ch.is_blocked == 0 &&
          mv2_allreduce_cyclic_algo_threshold <= nbytes) ||
        (mv2_allreduce_red_scat_allgather_algo_threshold <= nbytes &&
                  //local_size > 16)) {
                  local_size > mv2_allreduce_red_scat_allgather_algo_ppn_threshold)) {

        /* for large messages or cyclic hostfiles for medium messages, use
         * red-scat-allgather algorithm  */
        return MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2(sendbuf, recvbuf, count,
                datatype, op, comm_ptr, errflag);
    }

conf_check_end:


#ifdef MPID_HAS_HETERO
    if (comm_ptr->is_hetero) {
        is_homogeneous = 0;
    }

    if (!is_homogeneous) {
        /* heterogeneous. To get the same result on all processes, we
           do a reduce to 0 and then broadcast. */
        mpi_errno = MPIR_Reduce_MV2(sendbuf, recvbuf, count, datatype,
                                    op, 0, comm_ptr, errflag);
        /* 
           FIXME: mpi_errno is error CODE, not necessarily the error
           class MPI_ERR_OP.  In MPICH2, we can get the error class 
           with errorclass = mpi_errno & ERROR_CLASS_MASK;
	*/
        if (mpi_errno == MPI_ERR_OP || mpi_errno == MPI_SUCCESS) {
            /* Allow MPI_ERR_OP since we can continue from this error */
            rc = MPIR_Bcast_impl(recvbuf, count, datatype, 0, comm_ptr,
                                 errflag);
            if (rc)
                mpi_errno = rc;
        }
    } else
#endif /* MPID_HAS_HETERO */
	{
    
	    /* Search for the corresponding system size inside the tuning table */
	    /*
	     * Comm sizes progress in powers of 2. Therefore comm_size can just be indexed instead
	     */
	    table_min_comm_size = mv2_allreduce_indexed_thresholds_table[conf_index][0].numproc;
	    table_max_comm_size =
		mv2_allreduce_indexed_thresholds_table[conf_index][mv2_size_allreduce_indexed_tuning_table[conf_index] - 1].
		numproc;
    
	    if (comm_size < table_min_comm_size) {
		/* Comm size smaller than smallest configuration in table: use smallest available */
		comm_size_index = 0;
	    }
	    else if (comm_size > table_max_comm_size) {
		/* Comm size larger than largest configuration in table: use largest available */
		comm_size_index = mv2_size_allreduce_indexed_tuning_table[conf_index] - 1;
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
	    /* Search for corresponding inter-leader function */
	    /* skip mcast poiters if mcast is not available */
    
	    last_inter = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].size_inter_table - 1;
	    table_min_inter_size = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].inter_leader[0].msg_sz;
	    table_max_inter_size = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].
		inter_leader[last_inter].msg_sz;
	    last_intra = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].size_intra_table - 1;
	    table_min_intra_size = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].intra_node[0].msg_sz;
	    table_max_intra_size = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].
		intra_node[last_intra].msg_sz;
    
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
	    
	    if(mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].
	       is_two_level_allreduce[inter_node_algo_index] == 1){
		is_two_level = 1;    
	    }

	    MV2_Allreduce_function = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].
		inter_leader[inter_node_algo_index].MV2_pt_Allreduce_function;

	    MV2_Allreduce_intra_function = mv2_allreduce_indexed_thresholds_table[conf_index][comm_size_index].
		intra_node[intra_node_algo_index].MV2_pt_Allreduce_function;

skip_tuning_tables:
	    /* check if mcast is ready, otherwise replace mcast with other algorithm */
	    if((MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_redscat_gather_MV2)||
	       (MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2)){
#if defined(_MCST_SUPPORT_)
            /* fall back to RD algorithm if:
             *  1) two level is disabled
             *  2) mcast is not ready or supported */
            if(is_two_level != 1 ||
                mv2_use_mcast_allreduce != 1 ||
                comm_ptr->dev.ch.is_mcast_ok != 1 ||
                comm_ptr->dev.ch.shmem_coll_ok != 1)
#endif  /* #if defined(_MCST_SUPPORT_) */
            {
                MV2_Allreduce_function = &MPIR_Allreduce_pt2pt_rd_MV2;
            }
	    }
#if defined (_SHARP_SUPPORT_)
        if (comm_ptr->dev.ch.is_sharp_ok == 1 && nbytes <= mv2_sharp_tuned_msg_size
                && mv2_enable_sharp_coll == 1) {
            is_two_level = 1;
            MV2_Allreduce_function = &MPIR_Sharp_Allreduce_MV2;

        }
#endif 
                
	    if(is_two_level == 1) {
#if defined(_MCST_SUPPORT_)
		if((MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_redscat_gather_MV2)||
		   (MV2_Allreduce_function == &MPIR_Allreduce_mcst_reduce_two_level_helper_MV2)){ 

		    mpi_errno = MPIR_Allreduce_mcst_MV2(sendbuf, recvbuf, count,
							datatype, op, comm_ptr, errflag);
		} else
#endif  /* #if defined(_MCST_SUPPORT_) */
		    { 
			/* check if shm is ready, if not use other algorithm first */
			if ((comm_ptr->dev.ch.shmem_coll_ok == 1)
			    && (mv2_enable_shmem_allreduce)
			    && (is_commutative)
			    && (mv2_enable_shmem_collectives)) {
			    mpi_errno = MPIR_Allreduce_two_level_MV2(sendbuf, recvbuf, count,
								     datatype, op, comm_ptr, errflag);
			} else {
			    mpi_errno = MPIR_Allreduce_pt2pt_rd_MV2(sendbuf, recvbuf, count,
								    datatype, op, comm_ptr, errflag);
			}
		    }
	    } else { 
		mpi_errno = MV2_Allreduce_function(sendbuf, recvbuf, count,
						   datatype, op, comm_ptr, errflag);
	    }
	}

#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type){
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
			 (void *)recv_host_buf, 
			 stride, 
			 cudaMemcpyHostToDevice);
    }
    if(rdma_enable_cuda && recv_mem_type){
        if(recv_host_buf){
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if(rdma_enable_cuda && send_mem_type){
        if(send_host_buf){
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif
    comm_ptr->dev.ch.intra_node_done=0;
	
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

 fn_exit:
    return (mpi_errno);

 fn_fail:
    goto fn_exit;

}



/* This is flat reduce-scatter-allgather allreduce */
/* This function uses reduce_scatter and allgather colls */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_reduce_scatter_allgather_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_TIMER_START(coll,allreduce,reduce_scatter_allgather_colls);
    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_reduce_scatter_allgather_colls, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int i, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPIU_THREADPRIV_DECL;
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        MPIR_TIMER_END(coll,allreduce,reduce_scatter_allgather_colls);
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    if (count < comm_size) {
        mpi_errno = MPIR_Allreduce_pt2pt_rs_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag); 
        return mpi_errno;
    } 

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
 
    {
        /* do a reduce-scatter followed by allgather */

        /* for the reduce-scatter, calculate the count that
           each process receives and the displacement within
           the buffer */

       MPIU_CHKLMEM_MALLOC(cnts, int *, comm_size * sizeof (int), mpi_errno,
                           "counts");
       for (i = 0; i < (comm_size - 1); i++) {
           cnts[i] = count / comm_size;
       }
       cnts[comm_size - 1] = count - (count / comm_size) * (comm_size - 1);

       MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                           cnts[rank] * (MPIR_MAX(extent, true_extent)), mpi_errno,
                           "temporary buffer");
        
        /* adjust for potential negative lower bound in datatype */
       tmp_buf = (void *) ((char *) tmp_buf - true_lb);
       MPIU_CHKLMEM_MALLOC(disps, int *, comm_size * sizeof (int),
                           mpi_errno, "displacements");
       disps[0] = 0;
       for (i = 1; i < comm_size; i++) {
           disps[i] = disps[i - 1] + cnts[i - 1];
       }

       void * tmp_recvbuf = recvbuf + disps[rank] * (MPIR_MAX(extent, true_extent));
       if (sendbuf != MPI_IN_PLACE) {
            mpi_errno =
               MPIR_Reduce_scatter_MV2(sendbuf, tmp_recvbuf, cnts, datatype, op, comm_ptr,
                   errflag); 
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");
       } else {
            mpi_errno =
               MPIR_Reduce_scatter_MV2(MPI_IN_PLACE, recvbuf, cnts, datatype, op, comm_ptr,
                   errflag); 
            MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                "**fail");

            if (recvbuf != tmp_recvbuf) {
                /* need to shift the resutls to the location pointed by
                 * tmp_recbuf so that the following Allgather IN_PLACE works properly */
                if (disps[rank] >= cnts[rank]) {
                    /* make sure that there is no overlap between src and dst */
                    mpi_errno =
                       MPIR_Localcopy(recvbuf, cnts[rank], datatype, tmp_recvbuf, cnts[rank],
                                       datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                        "**fail");
                } else {
                    /* there is overlap between src and dst of shift */
                    void * shift_recvbuf = tmp_recvbuf + disps[rank] *
                        (MPIR_MAX(extent, true_extent)); 
                    MPI_Aint overlapped_count = cnts[rank] - disps[rank];
                    MPIU_Assert(overlapped_count > 0);

                    /* first shift the overlapped data */
                    mpi_errno =
                       MPIR_Localcopy(tmp_recvbuf, overlapped_count, datatype,
                               shift_recvbuf, overlapped_count, datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                        "**fail");
                    /* now copy the non overlapped data */
                    mpi_errno =
                       MPIR_Localcopy(recvbuf, disps[rank], datatype, tmp_recvbuf, disps[rank],
                                       datatype);
                    MPIR_ERR_CHKANDJUMP((mpi_errno), mpi_errno, MPI_ERR_OTHER,
                                        "**fail");
                }
            }
       }

       if (count % comm_size == 0) {
           MPIR_Allgather_MV2(MPI_IN_PLACE, cnts[rank], datatype, recvbuf, cnts[rank],
                   datatype, comm_ptr, errflag);
       } else {
           MPIR_Allgatherv_MV2(MPI_IN_PLACE, cnts[rank], datatype, recvbuf, cnts,
                disps, datatype, comm_ptr, errflag);
       }
    }
    

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,reduce_scatter_allgather_colls);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}







#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_old_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_old_MV2(const void *sendbuf,
                       void *recvbuf,
                       int count,
                       MPI_Datatype datatype,
                       MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_old_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    int mpi_errno = MPI_SUCCESS;

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    int stride = 0, is_commutative = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);
    MPID_Op *op_ptr;

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }
    }

#ifdef _ENABLE_CUDA_
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = recvbuf;
    const char *temp_sendbuf = sendbuf;

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(recvbuf);
       if ( sendbuf != MPI_IN_PLACE ){
           send_mem_type = is_device_buffer(sendbuf);
       }
    }

    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)sendbuf, 
                            stride, 
                            cudaMemcpyDeviceToHost);
        sendbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
                            (void *)recvbuf, 
                            stride, 
                            cudaMemcpyDeviceToHost);
        recvbuf = recv_host_buf;
    }
#endif

#if defined(_MCST_SUPPORT_)
    if(comm_ptr->dev.ch.is_mcast_ok == 1
       && comm_ptr->dev.ch.shmem_coll_ok == 1
       && mv2_use_mcast_allreduce == 1
       && stride >= mv2_mcast_allreduce_small_msg_size 
       && stride <= mv2_mcast_allreduce_large_msg_size){
        mpi_errno = MPIR_Allreduce_mcst_MV2(sendbuf, recvbuf, count, datatype,
                                     op, comm_ptr, errflag);
    } else
#endif /* #if defined(_MCST_SUPPORT_) */ 
    {
        if ((comm_ptr->dev.ch.shmem_coll_ok == 1)
            && (stride < mv2_coll_param.allreduce_2level_threshold)
            && (mv2_enable_shmem_allreduce)
            && (is_commutative)
            && (mv2_enable_shmem_collectives)) {
            mpi_errno = MPIR_Allreduce_shmem_MV2(sendbuf, recvbuf, count, datatype,
                                                 op, comm_ptr, errflag);

        } else {
       
            mpi_errno = MPIR_Allreduce_pt2pt_old_MV2(sendbuf, recvbuf, count, 
                                            datatype, op, comm_ptr, errflag);

        }
    } 

#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type){
        recvbuf = temp_recvbuf;
        MPIU_Memcpy_CUDA((void *)recvbuf, 
                            (void *)recv_host_buf, 
                            stride, 
                            cudaMemcpyHostToDevice);
    }
    if(rdma_enable_cuda && recv_mem_type){
        if(recv_host_buf){
            MPIU_Free(recv_host_buf);
            recv_host_buf = NULL;
        }
    }
    if(rdma_enable_cuda && send_mem_type){
        sendbuf = temp_sendbuf;
        if(send_host_buf){
            MPIU_Free(send_host_buf);
            send_host_buf = NULL;
        }
    }
#endif
	comm_ptr->dev.ch.intra_node_done=0;
	
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    return (mpi_errno);

  fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_MV2
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_MV2(const void *sendbuf,
                       void *recvbuf,
                       int count,
                       MPI_Datatype datatype,
                       MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
    MPIR_T_PVAR_COMM_COUNTER_INC(MV2, mv2_coll_allreduce_subcomm, 1, comm_ptr);
    MPIR_T_PVAR_COMM_TIMER_START(MV2, mv2_coll_timer_allreduce_subcomm, comm_ptr);

    int mpi_errno = MPI_SUCCESS;
    if (count == 0) {
        MPIR_T_PVAR_COMM_TIMER_END(MV2, mv2_coll_timer_allreduce_subcomm, comm_ptr);
        return MPI_SUCCESS;
    }
	
	// int max_eager = comm_ptr->dev.eager_max_msg_sz;

    if(mv2_use_old_allreduce == 1){
        mpi_errno = MPIR_Allreduce_old_MV2(sendbuf, recvbuf, count,
                                        datatype, op, comm_ptr, errflag);
    } else {
        if (mv2_use_indexed_tuning || mv2_use_indexed_allreduce_tuning) {
		
            mpi_errno = MPIR_Allreduce_index_tuned_intra_MV2(sendbuf, recvbuf, count,
                    datatype, op, comm_ptr, errflag);

        }
        else {
            mpi_errno = MPIR_Allreduce_new_MV2(sendbuf, recvbuf, count,
                    datatype, op, comm_ptr, errflag);
        }
    }

    MPIR_T_PVAR_COMM_TIMER_END(MV2, mv2_coll_timer_allreduce_subcomm, comm_ptr);

    return (mpi_errno);
}

#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_wrapper_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_ring_wrapper_MV2(const void *sendbuf, void *recvbuf,
        int count, MPI_Datatype datatype, MPI_Op op, MPID_Comm * comm_ptr,
        MPIR_Errflag_t *errflag, int comm_size, int sendtype_size) {
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_ring_wrapper_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif 
    int mpi_errno;
    int chunk           = count / comm_size;
    int new_count       = chunk * comm_size;
    int remaining_count = count - new_count;

		if (sendbuf != MPI_IN_PLACE){

        mpi_errno = MPIR_Allreduce_pt2pt_ring_MV2(sendbuf, recvbuf, new_count, datatype,
            op, comm_ptr, errflag);

		}
    else  {
        mpi_errno = MPIR_Allreduce_pt2pt_ring_inplace_MV2(sendbuf, recvbuf, new_count, datatype,
            op, comm_ptr, errflag);
	}
    if (mpi_errno != MPI_SUCCESS || (remaining_count == 0)) 
        return mpi_errno;
    /* now Allreduce the remaining count */ 
    if (sendbuf != MPI_IN_PLACE){
        return MPIR_Allreduce_pt2pt_rd_MV2(sendbuf+new_count*sendtype_size,
                recvbuf+new_count*sendtype_size, remaining_count, datatype,
                op, comm_ptr, errflag);
	}
    else {

        return MPIR_Allreduce_pt2pt_rd_MV2(MPI_IN_PLACE,
                recvbuf+new_count*sendtype_size, remaining_count, datatype,
                op, comm_ptr, errflag);
	
	}
}



#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIR_Allreduce_pt2pt_ring_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    if (security_approach==2005)
        return MPIR_Allreduce_pt2pt_ring_MV2_NodeAware(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else if (UNSEC_ALLREDUCE_MULTI_LEADER==1) 
        return MPIR_Allreduce_pt2pt_ring_MV2_Unsec_MultiLeader(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
    else return MPIR_Allreduce_pt2pt_ring_MV2_Original(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}



/* This is Naive+ ring-allreduce  allreduce */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_MV2_NodeAware"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_ring_MV2_NodeAware(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
//    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_ring_colls, 1);
#if ALLREDUCE_PRINT_FUN
     if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
	char hostname[100];
    int namelen;    
	 
    gethostname(hostname, &namelen); 
    
    printf("[allreduce rank = %d  RS_MET = %d  INTER_MET = %d  ALG_MET = %d  leader_cnt = %d  host = %s count=%d] Func: MPIR_Allreduce_pt2pt_ring_MV2_NodeAware [COLL_NAIVE_AESGCM_PLUS]\n",
    comm_ptr->rank,CONCUR_RS_METHOD,CONCUR_INTER_METHOD,CONCUR_AllGTHER_METHOD,leader_cnt,hostname,count);
    fflush(stdout);
    }
#endif   
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int i, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    int is_commutative;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_THREADPRIV_DECL;
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        return MPI_SUCCESS;
    }
	
	
    MPIU_THREADPRIV_GET;	
	
	int mpi_errno_ret = MPI_SUCCESS;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
	
    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }
    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    int type_size = MPIR_MAX(extent, true_extent);

    int srep2 = extent*count;

    if (count % comm_size != 0 || sendbuf == MPI_IN_PLACE ||
            count < comm_size || !is_commutative) {
        mpi_errno = MPIR_Allreduce_pt2pt_rs_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag);
        return mpi_errno;
    }

    MPIU_Assert((count * type_size) % comm_size == 0);
    		
	MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
	
	//void *tmp_buf;
	MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                           count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                           "temporary buffer");

    


	 /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);						
	
	 /* get our rank and the size of this communicator */
    int size = comm_ptr->local_size;
	
	// comm_size = comm_ptr->local_size;
    
    int ppn = shmem_commptr->local_size; // number of ranks per node
    int node = (int) (size / ppn); // number of nodes
	
	leader_cnt = ppn;
		
	MPID_Node_id_t node_id, left_node_id, right_node_id;    

    MPID_Get_node_id(comm_ptr, rank, &node_id);

    int rank_index = comm_ptr->dev.ch.rank_list_index, send_req_idx=0; 

    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
	char hostname1[100];
    int namelen1;
    gethostname(hostname1, &namelen1);

    int step = 1000*extent;

    int pof2 = shmem_commptr->dev.ch.gpof2;

    int rem = ppn - pof2;

	int right_index = (rank_index + 1);
		if(right_index >= size){
			right_index -= ppn;
		}else{
			MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[right_index], &right_node_id);
			if(right_node_id != node_id){
				right_index -= ppn;
			}
		}
	int	right = comm_ptr->dev.ch.rank_list[right_index];
		
	int	left_index = (rank_index - 1);
		if(left_index<0){
			left_index += ppn;
		}else{
			MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[left_index], &left_node_id);
			if(left_node_id != node_id){
				(left_index += ppn);
			}
		}
	int	left = comm_ptr->dev.ch.rank_list[left_index];
	
	int chunk_size_inter  = (count * type_size) / (leader_cnt*node);
	int chunk_size_intra  = (count * type_size) / ppn;
    int chunk_size_leader  = (count * type_size) / leader_cnt;
    int Is_leader = 1;
    

    int chunk_count_inter = count / (leader_cnt*node);
    int chunk_count_intra = count / ppn;
    int chunk_count_leader = count / leader_cnt;
	
	int shmem_rank = rank_index - (node_id * ppn);
	int inter_prefix = (shmem_rank * count * type_size) / leader_cnt; //ppn;


	void * recv_chunk = NULL, * send_chunk = NULL, * comp_chunk = NULL;
	MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
	
	int max_out_len;
	unsigned long ciphertext_sendbuf_len, dec_count;
	
	unsigned int chunk_enc_size = chunk_size_inter + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
	
	unsigned int recv_enc_chunk , send_enc_chunk;
	
	int tmp_INTER_METHOD=0;

    choping_sz= 8192; //16384; //32768;
	
	if ((chunk_size_leader<(2*choping_sz)) && (CONCUR_INTER_METHOD == 4))  tmp_INTER_METHOD=1;
#if 1	
	if (ppn > 1) {
        /* do a Intra-node reduce-scatter followed by allgather */
        MPIU_Assert(sendbuf != MPI_IN_PLACE);

        int i = 0;
		/* compute the left and right neighbor ranks in the rank_list */		 

    if (CONCUR_RS_METHOD == 1) { // Pt2pt-method

        for (i = 2; i < (ppn+1); i++) {

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */
			 
			if (i == 2)
                send_chunk  = sendbuf + ((shmem_rank - (i-1) + ppn) % ppn)*chunk_size_intra;
            else 
            send_chunk  = tmp_buf + ((shmem_rank - (i-1) + ppn) % ppn)*chunk_size_intra;

            recv_chunk  = tmp_buf + ((shmem_rank - i + ppn) % ppn)*chunk_size_intra;

            comp_chunk  = sendbuf + ((shmem_rank - i + ppn) % ppn)*chunk_size_intra; 					
			
			mpi_errno = MPID_Irecv(recv_chunk, chunk_count_intra, datatype, left, 11, comm_ptr, context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPID_Isend(send_chunk, chunk_count_intra, datatype, right, 11, comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);


            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count_intra, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count_intra, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

			MPID_Request_release(recv_req_ptr);
		
		} 
    } 
    else if (CONCUR_RS_METHOD == 2)  // ShMem-method
		{      

            mpi_errno = MPIR_Localcopy((void*)((char*) sendbuf), count, datatype, 
									    (void*)((char*) (shmem_buffer + shmem_rank * count * type_size)), count, datatype);
            
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
				goto fn_fail;
			}


/* ********** Only leaders contribute on RS *********** */	

if (shmem_rank < leader_cnt)    {   

                
                recv_chunk  = sendbuf + (shmem_rank * chunk_size_leader);

                for (i = 0; i < ppn; i++) {

                    if (i != shmem_rank) {
                                 
                         comp_chunk  = shmem_buffer + (i * count * type_size) + (shmem_rank * chunk_size_leader);                    

#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                        chunk_count_leader, datatype, uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (comp_chunk, recv_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif
                }
            }

            mpi_errno = MPIR_Localcopy((void*)((char*) (sendbuf + (shmem_rank*chunk_size_leader)) ), (count / leader_cnt), datatype, 
                                    (void*)((char*) (tmp_buf + (shmem_rank*chunk_size_leader))), (count / leader_cnt), datatype);

               
            } else Is_leader = 0;

#endif

	
        }
	
    }


    if (Is_leader && (size > ppn)) {  

    void * rep_chunk = NULL;

    if (CONCUR_AllGTHER_METHOD == 2) rep_chunk = shmem_buffer;
    else rep_chunk = recvbuf;

    if (Print_Ring && SHOW_FINAL_RESULT && SHOW_FINAL_RESULT && count==8192) fprintf(stderr,"Ran_leader = %d  host = % s chunk_size_leader = %d\n",rank,hostname1,chunk_size_leader); 
    if (CONCUR_INTER_METHOD == 1 || tmp_INTER_METHOD == 1)  // RD-method
    {
        int mask = ppn;  //leader_cnt
        int dst, dst_index = 0;

        int pof2 = comm_ptr->dev.ch.gpof2;

        chunk_enc_size = chunk_size_leader + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        //int i = ppn;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        while (mask < pof2) {
			
            dst_index = rank_index ^ mask;

            dst  = comm_ptr->dev.ch.rank_list[dst_index];
			
            mpi_errno = MPID_Irecv(large_recv_buffer, chunk_enc_size, MPI_CHAR, dst, 11, comm_ptr, context_id, &recv_req_ptr);  
			
			RAND_bytes(large_send_buffer, NONCE_SIZE);
			
			max_out_len = chunk_size_leader+ENC_MSG_TAG_SIZE; 
			if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer+NONCE_SIZE),
										   &ciphertext_sendbuf_len, max_out_len,
										   (&large_send_buffer[0]), NONCE_SIZE,
										   (char *)send_chunk, chunk_size_leader,
										   NULL, 0))
						{
							printf("Error in encryption: Concur-Inter: RD  rank = %d\n",comm_ptr->rank);
							fflush(stdout);
						}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"Concur-Inter: RD : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dst);
					fflush(stdout);
				}
#endif

            
			mpi_errno = MPID_Isend(large_send_buffer, chunk_enc_size, MPI_CHAR, dst, 11, comm_ptr, context_id, &send_req_ptr);                       
			//if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			//if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
			
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_leader,
							   (&large_recv_buffer[0]), NONCE_SIZE,
							   large_recv_buffer+(NONCE_SIZE), (chunk_size_leader + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Concur-Inter: RD  rank = %d source = %d  mask = %d  count = %d\n",comm_ptr->rank, dst,mask,count);
				fflush(stdout);
            }


            recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

            send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (recv_chunk, send_chunk,
                                                chunk_count_leader, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (recv_chunk, send_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);

            mask <<= 1;

        }
    }
        
    else if (CONCUR_INTER_METHOD == 2)  // RSA-method
    {
		 
        int mask, dst, i, dst_index, send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps, *enc_disps;  

        int dst_offset = ppn;     
		
		// int pof2 = comm_ptr->dev.ch.gpof2;
        

        MPIU_CHKLMEM_MALLOC(cnts, int *, node * sizeof (int), mpi_errno, "counts");
        MPIU_CHKLMEM_MALLOC(disps, int *, node * sizeof (int), mpi_errno, "displacements");								
        MPIU_CHKLMEM_MALLOC(enc_disps, int *, node * sizeof (int), mpi_errno, "enc_displacements");
   
        for (i = 0; i < (node - 1); i++) {
                cnts[i] = chunk_count_leader / node;
            }
        cnts[node - 1] = chunk_count_leader - (chunk_count_leader / node) * (node - 1);

        disps[0] = 0;
        enc_disps[0] = 0;
        for (i = 1; i < node; i++) {
            disps[i] = disps[i - 1] + cnts[i - 1];
            enc_disps[i] = enc_disps[i-1] + (cnts[i-1]*extent) + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        }

        mask = 0x1; //ppn;
        send_idx = recv_idx = 0; //pof2*(shmem_rank)/node;
        last_idx = node; //pof2*(shmem_rank+1)/node;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        if (Print_Ring && SHOW_FINAL_RESULT && count==8192 && 0) {

            void * print_chunk = tmp_buf + (shmem_rank * chunk_size_leader); 

            int tempp = step;
            step = extent;
            step = tempp;

        } 


        while (mask < node) { 

            dst_index = rank_index ^ (mask * dst_offset);

            dst  = comm_ptr->dev.ch.rank_list[dst_index];

            //dst  = rank ^ mask;

            send_cnt = recv_cnt = 0;
//            if (rank_index < dst) {
            if (rank < dst) {
                send_idx = recv_idx + node / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + node / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            chunk_enc_size = send_cnt*extent + ENC_MSG_TAG_SIZE + NONCE_SIZE;

			MPID_Irecv(large_recv_buffer, chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

            RAND_bytes(large_send_buffer, NONCE_SIZE);
            max_out_len = chunk_enc_size; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                            &ciphertext_sendbuf_len, max_out_len,
                            (&large_send_buffer[0]), NONCE_SIZE,
                            (char *) send_chunk + (disps[send_idx] * extent), recv_cnt*extent,
                            NULL, 0))
            {
                printf("Error in encryption: ConCur_Inter : RSA rank = %d\n",comm_ptr->rank);
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_GREEN"ConCur_Inter : RSA ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dst);
            }
#endif

            MPID_Isend(large_send_buffer, ciphertext_sendbuf_len+12, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk + (disps[recv_idx] * extent),
                            &dec_count, max_out_len,
                            (&large_recv_buffer[0]), NONCE_SIZE,
                            large_recv_buffer + NONCE_SIZE, ciphertext_sendbuf_len,
                            NULL, 0))
            {
                if (ENABLE_SECURE_DEBUG) fprintf(stderr,"ConCur_Inter : RSA : rank = %d source =%d  count = %d  mask=%d  recv_cnt=%d \n",comm_ptr->rank, dst,count,mask,recv_cnt);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) fprintf(stderr,COLOR_BLUE"ConCur_Inter : RSA : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                fflush(stdout);
            }
#endif 
         
            if (mask == node/2){

                recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

                send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            }                      
            
            (*uop) ((char *) recv_chunk + disps[recv_idx] * extent,
                    (char *) send_chunk + disps[recv_idx] * extent,
                    &recv_cnt, &datatype);


            send_idx = recv_idx;
            mask <<= 1;

            MPID_Request_release(recv_req_ptr);

            if (mask < node)
                last_idx = recv_idx + node / mask;

        }


        RAND_bytes(large_recv_buffer + enc_disps[recv_idx] , NONCE_SIZE);
        max_out_len = recv_cnt * extent + ENC_MSG_TAG_SIZE; 
        if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer + (enc_disps[recv_idx] ) + NONCE_SIZE),
                        &ciphertext_sendbuf_len, max_out_len,
                        (&large_recv_buffer[enc_disps[recv_idx]]), NONCE_SIZE,
                        (char *) (send_chunk + disps[recv_idx] * extent), recv_cnt * extent,
                        NULL, 0))
        {
            printf("Error in encryption: ConCur_Inter : RSA : rank = %d\n",comm_ptr->rank);
            fflush(stdout);
        }


        mask >>= 1;

        int seg_count = 0;
        int step_power = 0;
		int last_recv_idx = 0;
        int basic_seg_size = recv_cnt * extent;
		int basic_cipher_size = basic_seg_size + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        while (mask > 0) {

                dst_index = rank_index ^ (mask * dst_offset);

                dst  = comm_ptr->dev.ch.rank_list[dst_index];

                send_cnt = recv_cnt = 0;
				
				int node_half = node ;
				
                if (rank < dst) {
                    /* update last_idx except on first iteration */
                    if (mask != node_half / 2) {
                        last_idx = last_idx + node_half / (mask * 2);
                    }

                    recv_idx = send_idx + node_half / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = recv_idx-send_idx;
                } else {
                    recv_idx = send_idx - node_half / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = last_idx-send_idx;
                }

                chunk_enc_size = (send_cnt*extent) + (seg_count*(12+16));
                max_out_len = chunk_enc_size;

                MPID_Irecv(large_recv_buffer + (enc_disps[recv_idx]), chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

                MPID_Isend(large_recv_buffer + (enc_disps[send_idx]), chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

                if (step_power > 0)
                {
                    for (i=0; i<step_power; i++){

                        if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *) (send_chunk + ((disps[last_recv_idx]*extent)+i*basic_seg_size)),
                                        &dec_count, max_out_len,
                                        (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                                        large_recv_buffer + (enc_disps[last_recv_idx]) + (i * basic_cipher_size + NONCE_SIZE), basic_cipher_size - NONCE_SIZE,
                                        NULL, 0))
                        {
                            if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at ConCur_Inter : RSA (ALG) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, dst,i,count);
                        }
#if ENABLE_SECURE_MPI_DEBUG
                        else
                        {
                            if (ENABLE_SECURE_DEBUG) fprintf(COLOR_GREEN"ConCur_Inter : RSA (ALG-Not-Same_node): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                            fflush(stdout);
                        }
#endif 					
                    }
                } 

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
                			
			last_recv_idx=recv_idx;

                if ( step_power == 0) step_power=1;
				else step_power = step_power * 2;
				                
                MPID_Request_release(recv_req_ptr);
				
                if ( rank > dst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
			
        }

        for (i=0; i<step_power; i++){	

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *) (send_chunk + ((disps[last_recv_idx]*extent)+i*basic_seg_size)),
                            &dec_count, max_out_len,
                            (&large_recv_buffer[(enc_disps[last_recv_idx]) + i * basic_cipher_size]), NONCE_SIZE,
                            large_recv_buffer + (enc_disps[last_recv_idx]) + i * basic_cipher_size + NONCE_SIZE, basic_cipher_size - NONCE_SIZE,
                            NULL, 0))
            {
                if (ENABLE_SECURE_DEBUG) fprintf(stderr,"Decryption error at ConCur_Inter : RSA (ALG-Last-out) : rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, dst,i,recv_cnt);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) fprintf(COLOR_BLUE"  ConCur_Inter : RSA (ALG-Last-Dec): decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, rank, dst);
                fflush(stdout);
            }
#endif 		

    
     
     }
    }

    else if (CONCUR_INTER_METHOD == 3)  // Ring-method
    {   
        int i = 0;
		
        void * recv_enc = NULL, * send_enc = NULL;

        void * rep_buffer = recvbuf;
        
		/* compute the left and right neighbor ranks in the rank_list */
		left_index  = (size + rank_index - ppn) % size;
		right_index = (size + rank_index + ppn) % size;
		int left_Inter  = comm_ptr->dev.ch.rank_list[left_index];
		int right_Inter = comm_ptr->dev.ch.rank_list[right_index];
		

        for (i = 1; i < node; i++) {

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */
    
            if (i == 1)
                send_chunk  = tmp_buf + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;
                        
            else 
                send_chunk  = rep_buffer + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;

            comp_chunk  = tmp_buf + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;   		 			

            recv_enc_chunk = ((node_id - i + node) % node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-1) + node) % node)*chunk_enc_size + inter_prefix;
							   
			mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);  
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);	

			RAND_bytes(large_recv_buffer+send_enc_chunk, NONCE_SIZE);
			max_out_len = chunk_size_inter+ENC_MSG_TAG_SIZE; 
			if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer+(send_enc_chunk + NONCE_SIZE)),
										   &ciphertext_sendbuf_len, max_out_len,
										   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
										   (char *)send_chunk, chunk_size_inter,
										   NULL, 0))
						{
							printf("Error in encryption: Red-Scat step  rank = %d\n",comm_ptr->rank);
							fflush(stdout);
						}
#if ENABLE_SECURE_MPI_DEBUG
				else
				{
					if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"Ring: first allgather step : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, right_Inter);
					fflush(stdout);
				}
#endif

			mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11, comm_ptr, context_id, &send_req_ptr);                       
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            if (i == (node -1))  rep_buffer = rep_chunk;

            recv_chunk  = rep_buffer + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;
			
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
			
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_inter,
							   (&large_recv_buffer[recv_enc_chunk]), NONCE_SIZE,
							   large_recv_buffer+(recv_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Red-Scat step: rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
				fflush(stdout);
            }
			
			/* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count_inter, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count_inter, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
            MPID_Request_release(recv_req_ptr);
		
		}
			
		i=1;
		
		recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk ;
			
		send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
		recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
		send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;
						   
		mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);  

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);	

		RAND_bytes(large_recv_buffer+send_enc_chunk, NONCE_SIZE);
		max_out_len = chunk_size_inter+ENC_MSG_TAG_SIZE; 
		if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_recv_buffer+(send_enc_chunk + NONCE_SIZE)),
									   &ciphertext_sendbuf_len, max_out_len,
									   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
									   (char *)send_chunk, chunk_size_inter,
									   NULL, 0))
					{
						printf("Error in encryption: first allgather step  rank = %d\n",comm_ptr->rank);
						fflush(stdout);
					}
#if ENABLE_SECURE_MPI_DEBUG
			else
			{
				if (ENABLE_SECURE_DEBUG) printf(COLOR_GREEN"first allgather step : ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, right_Inter);
				fflush(stdout);
			}
#endif

		mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11,
					   comm_ptr, context_id, &send_req_ptr);                       

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);

		
		mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
		if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

		//MPID_Request_release(send_req_ptr);
		MPID_Request_release(recv_req_ptr);
		
        for (i = 2; i < node; i++) {		

			recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;
			 						
            mpi_errno = MPID_Irecv(large_recv_buffer+recv_enc_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            
            mpi_errno = MPID_Isend(large_recv_buffer+send_enc_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11, comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
						
			if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)send_chunk,
										   &dec_count, chunk_size_inter,
										   (&large_recv_buffer[send_enc_chunk]), NONCE_SIZE,
										   large_recv_buffer+(send_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
										   NULL, 0))
						{
							printf("Decryption error inside allgather step rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
							fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"First allgather step : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, left_Inter);
                fflush(stdout);
            }
#endif  

			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
				
            MPID_Request_release(recv_req_ptr);			
        }
		
		/* Last received chunk Decryption */

		if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recv_chunk,
							   &dec_count, chunk_size_inter,
							   (&large_recv_buffer[recv_enc_chunk]), NONCE_SIZE,
							   large_recv_buffer+(recv_enc_chunk+NONCE_SIZE), (chunk_size_inter + ENC_MSG_TAG_SIZE),
							   NULL, 0))
			{
				printf("Decryption error at Last allgather step: rank = %d source =%d  iter = %d  count = %d\n",comm_ptr->rank, left_Inter,i,count);
				fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG) printf(COLOR_BLUE"First allgather step : decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, left_Inter);
                fflush(stdout);
            }
#endif 
		
    }
  	
   	
	else if (CONCUR_INTER_METHOD == 4)  // RD-chop-method
    {
        
#if ALLREDUCE_PRINT_FUN
   
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) fprintf(stderr,"[allreduce rank = %d  CONCUR_INTER_METHOD = %d  choping_sz=%d count=%d  [COLL_NAIVE_AESGCM_PLUS]\n",
    comm_ptr->rank,CONCUR_INTER_METHOD,choping_sz,count);
    
#endif         

		// TODO: The chunk of Decryption shows erro.
		// It must be fixed. 
		
        int mask = ppn;  //leader_cnt
        int dst, dst_index = 0;

        MPID_Request **reqarray;
        MPI_Status *starray;

        int pof2 = comm_ptr->dev.ch.gpof2;

        chunk_enc_size = chunk_size_leader; // + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        //int i = ppn;
        int k,datasz,pos,req_counter,r_req_counter,segment_counter,dest,recv_len,decrypted_len;
        unsigned char nonce[100];
        /* Prepare Nonce */
        nonce[0] = '0';
        nonce[1] = '0';
        nonce[2] = '0';
        nonce[3] = '0';
        nonce[4] = '0';
        nonce[5] = '0';
        nonce[6] = '0';

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);
        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        void * large_recv_buffer2 = recv_chunk;
        void * large_send_buffer2 = send_chunk;

        int totaldata = chunk_size_leader; // + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE;
        //int choping_sz=32768;

        int segments_no = totaldata/ choping_sz;

        MPIU_CHKLMEM_MALLOC(reqarray, MPID_Request **, segments_no * sizeof(MPID_Request *),
                        mpi_errno, "reqarray");

        MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, segments_no * sizeof(MPI_Status),
                        mpi_errno, "starray");

        while (mask < pof2) {		
            
            dst_index = rank_index ^ mask;

            dst  = comm_ptr->dev.ch.rank_list[dst_index];
			
	
            pos=0; 
            req_counter=0;         

            for (k = 0; k < segments_no; k++)
            {
                mpi_errno = MPIC_Irecv(large_recv_buffer + pos,
                                        enc_choping_sz, MPI_CHAR, dst,
                                        11, comm_ptr,
                                        &reqarray[req_counter++]);                
                pos += enc_choping_sz;                
            }

			mpi_errno = MPI_SEC_Chopping_Send(large_send_buffer2, chunk_enc_size, MPI_CHAR, dst, 11, comm_ptr, &send_req_ptr,errflag); 

            r_req_counter=0;

            mpi_errno = MPIC_Waitall(1, &reqarray[r_req_counter], &starray[r_req_counter], errflag);
            r_req_counter++;   

            dest = 0;
            pos=0;
            segment_counter = 0;
			nonce[0] = '0';
			nonce[1] = '0';
			nonce[2] = '0';
			nonce[3] = '0';
			nonce[4] = '0';
			nonce[5] = '0';
			nonce[6] = '0';
            nonce[7] = '0';
            nonce[8] = (segment_counter >> 24) & 0xFF;
            nonce[9] = (segment_counter >> 16) & 0xFF;
            nonce[10] = (segment_counter >> 8) & 0xFF;
            nonce[11] = segment_counter & 0xFF;
            
            recv_len = choping_sz; // - ENC_MSG_TAG_SIZE ;

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, large_recv_buffer2,
                                &decrypted_len, (recv_len),
                                nonce, 12,
                                &large_recv_buffer[0], enc_choping_sz,
                                NULL, 0))
            {
                fprintf(stderr,"Decryption error: Chooping Recv First [C=%d R=%d  D=%d]\n",decrypted_len,comm_ptr->rank,dst);
            
            }   

            dest += decrypted_len;   
            pos += enc_choping_sz ; 
            
            int recv_data = 0;
            for (i = 1, segment_counter = 1; i < segments_no; i++, segment_counter++)
            {
                recv_len = choping_sz ;

                if (i != segments_no - 1)
                {
                    nonce[7] = '0';                    
                }
                else
                {
                    nonce[7] = '1';                                        
                }
				
                nonce[8] = (segment_counter >> 24) & 0xFF;
                nonce[9] = (segment_counter >> 16) & 0xFF;
                nonce[10] = (segment_counter >> 8) & 0xFF;
                nonce[11] = segment_counter & 0xFF;

                mpi_errno = MPIC_Waitall(1, &reqarray[r_req_counter], &starray[r_req_counter], errflag);
                r_req_counter++; 

                if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (large_recv_buffer2 + dest),
                                    &recv_data, (recv_len),
                                    nonce, 12,
                                    large_recv_buffer+pos, (enc_choping_sz),
                                    NULL, 0))
                {
                    fprintf(stderr,"Decryption error: chopping wait [C=%d R=%d  D=%d SC=%d]\n",recv_data,comm_ptr->rank,dst,segment_counter);
                    
                }

                dest += recv_data;
                pos += enc_choping_sz; 

            } 
			
            recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

            send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (recv_chunk, send_chunk,
                                                chunk_count_leader, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (recv_chunk, send_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
             
            mask <<= 1;

        }

    }
    
	} 
    
   
	if (ppn > 1) {
        /* do a Intra-node allgather */

		if (CONCUR_AllGTHER_METHOD == 1)  // Ring-method
		{
			int i = 0;
			
			for (i = 2; i < (ppn+1); i++) {		

				recv_chunk = ((shmem_rank - (i-1) + ppn)%ppn)*chunk_size_intra + recvbuf;

				mpi_errno = MPID_Irecv(recv_chunk, chunk_count_intra, datatype, left, 11, comm_ptr,
						context_id, &recv_req_ptr);
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);
				
				send_chunk = ((shmem_rank - (i-2) + ppn)%ppn)*chunk_size_intra + recvbuf;     



                if (SHOW_FINAL_RESULT && count==8192 && i==2 && 0) {

            void * print_chunk = send_chunk; 

            int tempp = step;
            step = extent;
          
            step = tempp;

        }

				mpi_errno = MPID_Isend(send_chunk, chunk_count_intra, datatype, right, 11,
									   comm_ptr, context_id, &send_req_ptr);
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);

				mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
				if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
				
				// MPID_Request_release(send_req_ptr);
				MPID_Request_release(recv_req_ptr);    
			}

		}
		
		else if (CONCUR_AllGTHER_METHOD == 2)  // ShMem-method
		{		
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
				goto fn_fail;
			}
 

            mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
									    (void*)((char*)recvbuf), count, datatype);
 
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
			}
			
		}
    }
	
#if ALLREDUCE_PRINT_FUN    
    if (SHOW_FINAL_RESULT && count==8192) {
        int s , match=1;
		step = (count/8)*extent;  
        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf+extent))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,COLOR_RED"[%d] dismatch at %d th element (%f , %f , %f) count = %d   host=%s  [end]"COLOR_RESET"\n",
                rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+(s-1)*extent)),*((float *) (recvbuf+s*extent)),count,hostname1);
                match =0;
                break;
            }

        }
        if (match==1) {
            //fprintf(stderr,COLOR_GREEN"[%d] is match host=%s [end]"COLOR_RESET"\n",rank,hostname1);
			fprintf(stderr,"[ML_Ring Rank = %d] is match [value = %f] host=%s  count=%d  [end]"COLOR_RESET"\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,COLOR_CYAN"[ML_Ring Rank = %d]  recvbuf = %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  %.1f  count=%d   host=%s  [end]"COLOR_RESET"\n",rank, *((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),*((float *) (recvbuf+8*step)),count,hostname1);
    }
#endif	


    /* 1check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

    if (PRINT_Ring_FUN_NAME)     fprintf(stderr,"[12] Conc-Ring [r=%d]\n",rank);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}



/* This is Naive+ ring-allreduce  allreduce */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_MV2_Unsec_MultiLeader"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_ring_MV2_Unsec_MultiLeader(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
//    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_ring_colls, 1);
#if ALLREDUCE_PRINT_FUN
     if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
	char hostname[100];
    int namelen;    
	 
    gethostname(hostname, &namelen); 
    
    printf("[allreduce rank = %d  CONCUR_RS_METHOD = %d  CONCUR_INTER_METHOD = %d  CONCUR_AllGTHER_METHOD = %d  leader_cnt = %d  host = %s count=%d] Func: MPIR_Allreduce_pt2pt_ring_MV2_Unsec_MultiLeader [UNSEC_ALLREDUCE_MULTI_LEADER]\n",
    comm_ptr->rank,CONCUR_RS_METHOD,CONCUR_INTER_METHOD,CONCUR_AllGTHER_METHOD,leader_cnt,hostname,count);
    fflush(stdout);
    }
#endif   
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int i, *cnts, *disps;
    MPI_Aint true_lb, true_extent, extent;
    void *tmp_buf;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    int is_commutative;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_THREADPRIV_DECL;
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        return MPI_SUCCESS;
    }

    
    MPIU_THREADPRIV_GET;	
	
	int mpi_errno_ret = MPI_SUCCESS;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;
	
    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }
    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    int type_size = MPIR_MAX(extent, true_extent);

    int srep2 = extent*count;

    


    if (count % comm_size != 0 || sendbuf == MPI_IN_PLACE ||
            count < comm_size || !is_commutative) {
        mpi_errno = MPIR_Allreduce_pt2pt_rs_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag);
        return mpi_errno;
    }

    MPIU_Assert((count * type_size) % comm_size == 0);
    		
	MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
	
	//void *tmp_buf;
	MPIU_CHKLMEM_MALLOC(tmp_buf, void *,
                           count * (MPIR_MAX(extent, true_extent)), mpi_errno,
                           "temporary buffer");

    


	 /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);						
	
	 /* get our rank and the size of this communicator */
    int size = comm_ptr->local_size;
	
	comm_size = comm_ptr->local_size;
    
    int ppn = shmem_commptr->local_size; // number of ranks per node
    int node = (int) (size / ppn); // number of nodes
	
	leader_cnt = ppn;
		
	MPID_Node_id_t node_id, left_node_id, right_node_id;    

    MPID_Get_node_id(comm_ptr, rank, &node_id);

    int rank_index = comm_ptr->dev.ch.rank_list_index, send_req_idx=0; 

    
    /* execute ring exchange, start by sending our own data to the right
     * and receiving the data from the rank to our left */
	char hostname1[100];
    int namelen1;
    gethostname(hostname1, &namelen1);

    int step = 1000*extent;

    int pof2 = shmem_commptr->dev.ch.gpof2;

    int rem = ppn - pof2;
	
	int right_index = (rank_index + 1);
		if(right_index >= size){
			right_index -= ppn;
		}else{
			MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[right_index], &right_node_id);
			if(right_node_id != node_id){
				right_index -= ppn;
			}
		}
	int	right = comm_ptr->dev.ch.rank_list[right_index];
		
	int	left_index = (rank_index - 1);
		if(left_index<0){
			left_index += ppn;
		}else{
			MPID_Get_node_id(comm_ptr, comm_ptr->dev.ch.rank_list[left_index], &left_node_id);
			if(left_node_id != node_id){
				(left_index += ppn);
			}
		}
	int	left = comm_ptr->dev.ch.rank_list[left_index];

    if (leader_cnt > ppn) leader_cnt = ppn;
    else if (leader_cnt < 1) leader_cnt = 1;



	int chunk_size_inter  = (count * type_size) / (leader_cnt*node);
	int chunk_size_intra  = (count * type_size) / ppn;
    int chunk_size_leader  = (count * type_size) / leader_cnt;
    int Is_leader = 1;
    

    int chunk_count_inter = count / (leader_cnt*node);
    int chunk_count_intra = count / ppn;
    int chunk_count_leader = count / leader_cnt;
	
	int shmem_rank = rank_index - (node_id * ppn);
	int inter_prefix = (shmem_rank * count * type_size) / leader_cnt; //ppn;


	void * recv_chunk = NULL, * send_chunk = NULL, * comp_chunk = NULL;
	MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
	
	int max_out_len;
	unsigned long ciphertext_sendbuf_len, dec_count;
	
	unsigned int chunk_enc_size = chunk_size_inter; // + ENC_MSG_TAG_SIZE + NONCE_SIZE; 
	
	unsigned int recv_enc_chunk , send_enc_chunk;

    
	if (ppn > 1) {
        /* do a Intra-node reduce-scatter followed by allgather */
        MPIU_Assert(sendbuf != MPI_IN_PLACE);

        int i = 0;
		/* compute the left and right neighbor ranks in the rank_list */
	

    if (CONCUR_RS_METHOD == 1) { // Pt2pt-method

        for (i = 2; i < (ppn+1); i++) {

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */
			 
			if (i == 2)
                send_chunk  = sendbuf + ((shmem_rank - (i-1) + ppn) % ppn)*chunk_size_intra;
            else 
            send_chunk  = tmp_buf + ((shmem_rank - (i-1) + ppn) % ppn)*chunk_size_intra;

            recv_chunk  = tmp_buf + ((shmem_rank - i + ppn) % ppn)*chunk_size_intra;

            comp_chunk  = sendbuf + ((shmem_rank - i + ppn) % ppn)*chunk_size_intra; 					
			
 			/* mpi_errno = MPIC_Sendrecv(send_chunk, chunk_count_intra, datatype,
												 right, 11,
												 recv_chunk, chunk_count_intra, datatype, 
												 left, 11, 
												 comm_ptr, MPI_STATUS_IGNORE, errflag); 
			
			
			if (mpi_errno) {                    
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                } */

			mpi_errno = MPID_Irecv(recv_chunk, chunk_count_intra, datatype, left, 11, comm_ptr, context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPID_Isend(send_chunk, chunk_count_intra, datatype, right, 11, comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);


            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count_intra, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count_intra, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

			MPID_Request_release(recv_req_ptr);
		
		} 
    } 
    else if (CONCUR_RS_METHOD == 2)  // ShMem-method
		{
			
            mpi_errno = MPIR_Localcopy((void*)((char*) sendbuf), count, datatype, 
									    (void*)((char*) (shmem_buffer + shmem_rank * count * type_size)), count, datatype);
            
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
				goto fn_fail;
			}
      

/* ********** Only leaders contribute on RS *********** */	

if (shmem_rank < leader_cnt)    {   

                //mpi_errno = MPIR_Localcopy((void*)((char*) (shmem_buffer + (shmem_rank*chunk_size_leader)) ), (count / leader_cnt), datatype, 
                                               // (void*)((char*) (tmp_buf + (shmem_rank*chunk_size_leader))), (count / leader_cnt), datatype); 

                recv_chunk  = sendbuf + (shmem_rank * chunk_size_leader);

                for (i = 0; i < ppn; i++) {

                    if (i != shmem_rank) {
                                 
                         comp_chunk  = shmem_buffer + (i * count * type_size) + (shmem_rank * chunk_size_leader);                    

#if defined(HAVE_CXX_BINDING)
                    if (is_cxx_uop) {
                        (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                        chunk_count_leader, datatype, uop);
                    } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                        (*uop) (comp_chunk, recv_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
                    }
#endif
                }
            }

            mpi_errno = MPIR_Localcopy((void*)((char*) (sendbuf + (shmem_rank*chunk_size_leader)) ), (count / leader_cnt), datatype, 
                                    (void*)((char*) (tmp_buf + (shmem_rank*chunk_size_leader))), (count / leader_cnt), datatype);

               
            } else Is_leader = 0;

        }
	
    }


    if (Is_leader && (size > ppn)) {

       

    void * rep_chunk = NULL;


    if (CONCUR_AllGTHER_METHOD == 2) rep_chunk = shmem_buffer;
    else rep_chunk = recvbuf;

    if (CONCUR_INTER_METHOD == 1)  // RD-method
    {
        int mask = ppn;  //leader_cnt
        int dst, dst_index = 0;

        int pof2 = comm_ptr->dev.ch.gpof2;

        chunk_enc_size = chunk_size_leader; // + ENC_MSG_TAG_SIZE + NONCE_SIZE;

        //int i = ppn;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);
        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        void * large_recv_buffer2 = recv_chunk;
        void * large_send_buffer2 = send_chunk;


        while (mask < pof2) {

            dst_index = rank_index ^ mask;

            dst  = comm_ptr->dev.ch.rank_list[dst_index];

            mpi_errno = MPID_Irecv(large_recv_buffer2, chunk_enc_size, MPI_CHAR, dst, 11, comm_ptr, context_id, &recv_req_ptr);  
			//if (mpi_errno) MPIR_ERR_POP(mpi_errno);	
			mpi_errno = MPID_Isend(large_send_buffer2, chunk_enc_size, MPI_CHAR, dst, 11, comm_ptr, context_id, &send_req_ptr);                       
			//if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			//if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
			
            recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

            send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (recv_chunk, send_chunk,
                                                chunk_count_leader, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (recv_chunk, send_chunk, &chunk_count_leader, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif
            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);
            mask <<= 1;

        }

    }
        
    else if (CONCUR_INTER_METHOD == 2)  // RSA-method
    {
		
		 
        int mask, dst, i, dst_index, send_idx, recv_idx, last_idx, send_cnt, recv_cnt, *cnts, *disps, *enc_disps;  

        int dst_offset = ppn;     
		
		// int pof2 = comm_ptr->dev.ch.gpof2;
        

        MPIU_CHKLMEM_MALLOC(cnts, int *, node * sizeof (int), mpi_errno, "counts");
        MPIU_CHKLMEM_MALLOC(disps, int *, node * sizeof (int), mpi_errno, "displacements");						
        MPIU_CHKLMEM_MALLOC(enc_disps, int *, node * sizeof (int), mpi_errno, "enc_displacements");
   
        for (i = 0; i < (node - 1); i++) {
                cnts[i] = chunk_count_leader / node;
            }
        cnts[node - 1] = chunk_count_leader - (chunk_count_leader / node) * (node - 1);

        disps[0] = 0;
        enc_disps[0] = 0;
        for (i = 1; i < node; i++) {
            disps[i] = disps[i - 1] + cnts[i - 1];
            enc_disps[i] = enc_disps[i-1] + (cnts[i-1]*extent); // + ENC_MSG_TAG_SIZE + NONCE_SIZE;
        }

        mask = 0x1; //ppn;
        send_idx = recv_idx = 0; //pof2*(shmem_rank)/node;
        last_idx = node; //pof2*(shmem_rank+1)/node;

        send_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

        recv_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);

        
        void * large_send_buffer2 = send_chunk;
        void * large_recv_buffer2 = recv_chunk;
        

        while (mask < node) { 

            dst_index = rank_index ^ (mask * dst_offset);

            dst  = comm_ptr->dev.ch.rank_list[dst_index];

            //dst  = rank ^ mask;

            send_cnt = recv_cnt = 0;
//            if (rank_index < dst) {
            if (rank < dst) {
                send_idx = recv_idx + node / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + node / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            
            chunk_enc_size = send_cnt*extent;             
					
			MPID_Irecv(recv_chunk + (disps[recv_idx] * extent), chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

            MPID_Isend(send_chunk + (disps[send_idx] * extent), chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);            

         
            if (mask == node/2){

                recv_chunk  = tmp_buf + (shmem_rank * chunk_size_leader);

                send_chunk  = rep_chunk + (shmem_rank * chunk_size_leader);
            }          
                        
            
            (*uop) ((char *) recv_chunk + disps[recv_idx] * extent,
                    (char *) send_chunk + disps[recv_idx] * extent,
                    &recv_cnt, &datatype);


            send_idx = recv_idx;
            mask <<= 1;

            MPID_Request_release(recv_req_ptr);

            if (mask < node)
                last_idx = recv_idx + node / mask;

        }

        mask >>= 1;

        large_recv_buffer2 = send_chunk;

        int seg_count = 0;
        int step_power = 0;
		int last_recv_idx = 0;
        int basic_seg_size = recv_cnt * extent;
		int basic_cipher_size = basic_seg_size; 

        while (mask > 0) {

                dst_index = rank_index ^ (mask * dst_offset);

                dst  = comm_ptr->dev.ch.rank_list[dst_index];

                send_cnt = recv_cnt = 0;
				
				int node_half = node ;
				
                if (rank < dst) {
                    /* update last_idx except on first iteration */
                    if (mask != node_half / 2) {
                        last_idx = last_idx + node_half / (mask * 2);
                    }

                    recv_idx = send_idx + node_half / (mask * 2);
                    for (i = send_idx; i < recv_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < last_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = recv_idx-send_idx;
                } else {
                    recv_idx = send_idx - node_half / (mask * 2);
                    for (i = send_idx; i < last_idx; i++) {
                        send_cnt += cnts[i];
                    }
                    for (i = recv_idx; i < send_idx; i++) {
                        recv_cnt += cnts[i];
                    }
					seg_count = last_idx-send_idx;
                }

                chunk_enc_size = (send_cnt*extent); 

                MPID_Irecv(large_recv_buffer2 + (enc_disps[recv_idx]), chunk_enc_size, MPI_CHAR, dst,  MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &recv_req_ptr);

                MPID_Isend(large_recv_buffer2 + (enc_disps[send_idx]), chunk_enc_size, MPI_CHAR, dst, MPIR_ALLREDUCE_TAG, comm_ptr, context_id, &send_req_ptr);

                mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
                if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
      			
			    last_recv_idx=recv_idx;

                if ( step_power == 0) step_power=1;
				else step_power = step_power * 2;
				
      
                // MPID_Request_release(send_req_ptr);
                MPID_Request_release(recv_req_ptr);
				

      
                if ( rank > dst) {
                    send_idx = recv_idx;
                }

                mask >>= 1;
			
        }   
     
     }

    else if (CONCUR_INTER_METHOD == 3)  // Ring-method
    {    
        //MPIU_Assert(sendbuf != MPI_IN_PLACE);

        int i = 0;
		
        void * recv_enc = NULL, * send_enc = NULL;

        void * rep_buffer = recvbuf;
        
        
		
		/* compute the left and right neighbor ranks in the rank_list */
		left_index  = (size + rank_index - ppn) % size;
		right_index = (size + rank_index + ppn) % size;
		int left_Inter  = comm_ptr->dev.ch.rank_list[left_index];
		int right_Inter = comm_ptr->dev.ch.rank_list[right_index];

        
        for (i = 1; i < node; i++) {

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */
    
            if (i == 1)
                send_chunk  = tmp_buf + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;
                        
            else 
                send_chunk  = rep_buffer + ((node_id - (i-1) + node) % node)*chunk_size_inter + inter_prefix;


            comp_chunk  = tmp_buf + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;   
			


            recv_enc_chunk = ((node_id - i + node) % node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-1) + node) % node)*chunk_enc_size + inter_prefix;
							   
	
			mpi_errno = MPID_Isend(send_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11, comm_ptr, context_id, &send_req_ptr);                       
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            if (i == (node -1))  rep_buffer = rep_chunk;

            recv_chunk  = rep_buffer + ((node_id - i + node) % node)*chunk_size_inter + inter_prefix;

            mpi_errno = MPID_Irecv(recv_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);  
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);	
			
			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
			if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
	
			/* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count_inter, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count_inter, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

            MPID_Request_release(recv_req_ptr);
		
		}
			
		i=1;
		
		recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk ;
			
		send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
		recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
		send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;

        //void * large_recv_buffer2 = send_chunk;
						   
		mpi_errno = MPID_Irecv(recv_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);  

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);	

		mpi_errno = MPID_Isend(send_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11,
					   comm_ptr, context_id, &send_req_ptr);                       

		if (mpi_errno) MPIR_ERR_POP(mpi_errno);

		
		mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
		if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

		//MPID_Request_release(send_req_ptr);
		MPID_Request_release(recv_req_ptr);
		
        for (i = 2; i < node; i++) {		

			recv_chunk = ((node_id - (i-1) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			send_chunk = ((node_id - (i-2) + node)%node)*chunk_size_inter + inter_prefix + rep_chunk;
			
			recv_enc_chunk = ((node_id - (i-1) + node)%node)*chunk_enc_size + inter_prefix;
			
			send_enc_chunk = ((node_id - (i-2) + node)%node)*chunk_enc_size + inter_prefix;

            //void * large_recv_buffer2 = send_chunk;
			 						
            mpi_errno = MPID_Irecv(recv_chunk, chunk_enc_size, MPI_CHAR, left_Inter, 11, comm_ptr, context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            
            mpi_errno = MPID_Isend(send_chunk, chunk_enc_size, MPI_CHAR, right_Inter, 11, comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

			mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
				
            //mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            //if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            //MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);
			
        }
		
		/* Last received chunk Decryption */

    }

	
    } 
    
	if (ppn > 1) {
        /* do a Intra-node allgather */     
		if (CONCUR_AllGTHER_METHOD == 1)  // Ring-method
		{
			int i = 0;
			for (i = 2; i < (ppn+1); i++) {		
              	
				recv_chunk = ((shmem_rank - (i-1) + ppn)%ppn)*chunk_size_intra + recvbuf;

				mpi_errno = MPID_Irecv(recv_chunk, chunk_count_intra, datatype, left, 11, comm_ptr,
						context_id, &recv_req_ptr);
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);

				send_chunk = ((shmem_rank - (i-2) + ppn)%ppn)*chunk_size_intra + recvbuf;     

                if (SHOW_FINAL_RESULT && count==8192 && i==2 && 0) {

            void * print_chunk = send_chunk; 

            int tempp = step;
            step = extent;            

            step = tempp;

        }

				mpi_errno = MPID_Isend(send_chunk, chunk_count_intra, datatype, right, 11,
									   comm_ptr, context_id, &send_req_ptr);
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);

				mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
				if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
				MPID_Request_release(recv_req_ptr);

			}

		}
		
		else if (CONCUR_AllGTHER_METHOD == 2)  // ShMem-method
		{
			
            mpi_errno = MPIR_Barrier_impl(comm_ptr->node_comm, errflag);
			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
				goto fn_fail;
			}
            
            mpi_errno = MPIR_Localcopy((void*)((char*)shmem_buffer), count, datatype, 
									    (void*)((char*)recvbuf), count, datatype);

 			if (mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
			}
			
		}
    }
	
	
#if ALLREDUCE_PRINT_FUN  
    if (SHOW_FINAL_RESULT && count==8192) {
        int s , match=1;
		step = (count/8)*extent;  
        for (s=1 ; s<count ; s++){
            if ( (*((float *) (recvbuf+extent))) != ( *((float *) (recvbuf+s*extent)))){
                fprintf(stderr,COLOR_RED"[%d] dismatch at %d th element (%f , %f , %f) count = %d   host=%s  [end]"COLOR_RESET"\n",
                rank,s, *((float *) (recvbuf+extent)),*((float *) (recvbuf+(s-1)*extent)),*((float *) (recvbuf+s*extent)),count,hostname1);
                match =0;
                break;
            }

        }
        if (match==1) {
            //fprintf(stderr,COLOR_GREEN"[%d] is match host=%s [end]"COLOR_RESET"\n",rank,hostname1);
			fprintf(stderr,"[unsec_ML_Ring Rank = %d] is match [value = %f] host=%s  count=%d  [end]"COLOR_RESET"\n",comm_ptr->rank,*((float *) (recvbuf+extent)),hostname1,count);
        } else       
            fprintf(stderr,COLOR_CYAN"[unsec_ML_Ring Rank = %d]  recvbuf = %f  %f  %f  %f  %f  %f  %f  %f  count=%d   host=%s  [end]"COLOR_RESET"\n",rank, *((float *) (recvbuf+step)),*((float *) (recvbuf+2*step)),*((float *) (recvbuf+3*step)),*((float *) (recvbuf+4*step)),*((float *) (recvbuf+5*step)),*((float *) (recvbuf+6*step)),*((float *) (recvbuf+7*step)),*((float *) (recvbuf+8*step)),count,hostname1);
    }
#endif	


    /* 1check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}




/* This is flat ring-allreduce  allreduce */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_MV2_Original"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_ring_MV2_Original(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_ring_MV2_Original  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
//    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_ring_colls, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent, extent;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    int is_commutative;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_THREADPRIV_DECL;
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }
    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    int type_size = MPIR_MAX(extent, true_extent);


    if (count % comm_size != 0 || sendbuf == MPI_IN_PLACE ||
            count < comm_size || !is_commutative) {
        mpi_errno = MPIR_Allreduce_pt2pt_rs_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag);
        return mpi_errno;
    }
    //memset(recvbuf, 0, type_size*count);

    MPIU_Assert((count * type_size) % comm_size == 0);
    int chunk_size  = (count * type_size) / comm_size;
    int chunk_count = count / comm_size;

    {
        /* do a reduce-scatter followed by allgather */
        MPIU_Assert(sendbuf != MPI_IN_PLACE);

        int i = 0;
        int left    = (rank - 1 + comm_size) % comm_size;
        int right   = (rank + 1) % comm_size;
        void * recv_chunk = NULL, * send_chunk = NULL, * comp_chunk = NULL;
        MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

        for (i = 1; i < comm_size; i++) {

            MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */

            if (i == 1 && sendbuf != MPI_IN_PLACE)
                send_chunk  = (void*) sendbuf + ((rank - (i-1) + comm_size) % comm_size)*chunk_size;
            else 
                send_chunk  = recvbuf + ((rank - (i-1) + comm_size) % comm_size)*chunk_size;

            recv_chunk  = recvbuf + ((rank - i + comm_size) % comm_size)*chunk_size;

            comp_chunk  = (void*) sendbuf + ((rank - i + comm_size) % comm_size)*chunk_size;

            /*
            PRINT_DEBUG(1, "i = %d, sending chunk=%d to rank=%d and receiving chunk=%d from rank=%d and doing compute on chunk=%d \n",
                   i, ((rank - (i-1) + comm_size) % comm_size), right, ((rank -
                            i + comm_size) % comm_size), left, ((rank - i +
                                comm_size) % comm_size));
            */
            mpi_errno = MPID_Irecv(recv_chunk, chunk_count, datatype, left, 11, comm_ptr,
                    context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPID_Isend(send_chunk, chunk_count, datatype, right, 11,
                                   comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (comp_chunk, recv_chunk,
                                                chunk_count, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (comp_chunk, recv_chunk, &chunk_count, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);

        }


        for (i = 1; i < comm_size; i++) {
        
            recv_chunk = ((rank - (i-1) + comm_size)%comm_size)*chunk_size + recvbuf;
            mpi_errno = MPID_Irecv(recv_chunk, chunk_count, datatype, left, 11, comm_ptr,
                    context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            send_chunk = ((rank - (i-2) + comm_size)%comm_size)*chunk_size + recvbuf;
            mpi_errno = MPID_Isend(send_chunk, chunk_count, datatype, right, 11,
                                   comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);
        }
    }




    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}



/* This is flat ring-allreduce  allreduce with MPI_INPLACE for sendbuf */
#undef FCNAME
#define FCNAME "MPIR_Allreduce_pt2pt_ring_MV2"
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_pt2pt_ring_inplace_MV2(const void *sendbuf,
                             void *recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
#if ALLREDUCE_PRINT_FUN
    if (PRINT_FUN_NAME || DEBUG_INIT_FILE) {
        char hostname[100];
        int namelen;
        gethostname(hostname, &namelen);
        printf("[allreduce rank = %d host = %s] Func: MPIR_Allreduce_pt2pt_ring_inplace_MV2  count = %d\n",comm_ptr->rank,hostname,count);fflush(stdout);}
#endif     
//    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_coll_allreduce_ring_colls, 1);
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint true_lb, true_extent, extent;
    MPI_User_function *uop;
    MPID_Op *op_ptr;
    int is_commutative;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
    MPIU_THREADPRIV_DECL;
    MPIU_CHKLMEM_DECL(3);

    if (count == 0) {
        return MPI_SUCCESS;
    }

    MPIU_THREADPRIV_GET;

    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_ENTER(comm_ptr);

    /* homogeneous */

    /* set op_errno to 0. stored in perthread structure */
    MPIU_THREADPRIV_FIELD(op_errno) = 0;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Get the operator and check whether it is commutative or not */
    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        is_commutative = 1;
        /* get the function by indexing into the op table */
        uop = MPIR_Op_table[op % 16 - 1];
    } else {
        MPID_Op_get_ptr(op, op_ptr);
        if (op_ptr->kind == MPID_OP_USER_NONCOMMUTE) {
            is_commutative = 0;
        } else {
            is_commutative = 1;
        }

#if defined(HAVE_CXX_BINDING)
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        } else
#endif                          /* defined(HAVE_CXX_BINDING) */
        if (op_ptr->language == MPID_LANG_C) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
        } else {
            uop = (MPI_User_function *) op_ptr->function.f77_function;
        }
    }
    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
                MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    int type_size = MPIR_MAX(extent, true_extent);


    if (count % comm_size != 0 || sendbuf == MPI_IN_PLACE ||
            count < comm_size || !is_commutative) {
        mpi_errno = MPIR_Allreduce_pt2pt_rs_MV2(sendbuf, recvbuf, count, datatype,
                op, comm_ptr, errflag);
        return mpi_errno;
    }
    //memset(recvbuf, 0, type_size*count);

    MPIU_Assert((count * type_size) % comm_size == 0);
    int chunk_size  = (count * type_size) / comm_size;
    int chunk_count = count / comm_size;
    

    /* TODO: try to do this malloc only once for entire job run */
    void  * temp_buf = MPIU_Malloc(chunk_size);

    {
        /* do a reduce-scatter followed by allgather */
        MPIU_Assert(sendbuf == MPI_IN_PLACE);

        int i = 0;
        int left    = (rank - 1 + comm_size) % comm_size;
        int right   = (rank + 1) % comm_size;
        void * recv_chunk = NULL, * send_chunk = NULL, * comp_chunk = NULL;
        MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

        for (i = 1; i < comm_size; i++) {

            MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;

            /* iteration 1: read chunk murank - 1 from myrank - 1 and do compute locally, at the
             * same time, send chunk myrank + 1 to myrank + 1 */

            send_chunk  = recvbuf + ((rank - (i-1) + comm_size) % comm_size)*chunk_size;

            //recv_chunk  = recvbuf + ((rank - i + comm_size) % comm_size)*chunk_size;
            recv_chunk = temp_buf;

            //comp_chunk  = sendbuf + ((rank - i + comm_size) % comm_size)*chunk_size;
            comp_chunk  = recvbuf + ((rank - i + comm_size) % comm_size)*chunk_size;

            /*
            PRINT_DEBUG(1, "i = %d, sending chunk=%d to rank=%d and receiving chunk=%d from rank=%d and doing compute on chunk=%d \n",
                   i, ((rank - (i-1) + comm_size) % comm_size), right, ((rank -
                            i + comm_size) % comm_size), left, ((rank - i +
                                comm_size) % comm_size));
            */
            mpi_errno = MPID_Irecv(recv_chunk, chunk_count, datatype, left, 11, comm_ptr,
                    context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPID_Isend(send_chunk, chunk_count, datatype, right, 11,
                                   comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

            /* now do the computation with some MPI_Test (TODO) in between */
#if defined(HAVE_CXX_BINDING)
            if (is_cxx_uop) {
                (*MPIR_Process.cxx_call_op_fn) (recv_chunk, comp_chunk, 
                                                chunk_count, datatype, uop);
            } else {
#endif                          /* defined(HAVE_CXX_BINDING) */
                (*uop) (recv_chunk, comp_chunk, &chunk_count, &datatype);
#if defined(HAVE_CXX_BINDING)
            }
#endif

            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);

        }



        /*
        if (rank == 0) {
        PRINT_DEBUG(1, "Final Results count = %d\n", count);
        for (i = 0; i < count; i++) {
            fprintf(stderr, "%d ", ((float*)recvbuf)[i]);
        }
        PRINT_DEBUG(1, "\n");
        }
        */
           /* Allgather on recvbuf with  one elemented  shifted to right */
          // MPIR_Allgather_MV2(MPI_IN_PLACE, chunk_count, datatype, recvbuf, chunk_count,
            //       datatype, comm_ptr, errflag);
           

        for (i = 1; i < comm_size; i++) {
        
            recv_chunk = ((rank - (i-1) + comm_size)%comm_size)*chunk_size + recvbuf;
            mpi_errno = MPID_Irecv(recv_chunk, chunk_count, datatype, left, 11, comm_ptr,
                    context_id, &recv_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            send_chunk = ((rank - (i-2) + comm_size)%comm_size)*chunk_size + recvbuf;
            mpi_errno = MPID_Isend(send_chunk, chunk_count, datatype, right, 11,
                                   comm_ptr, context_id, &send_req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
            mpi_errno = MPIC_Wait(send_req_ptr, errflag);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);

            MPID_Request_release(send_req_ptr);
            MPID_Request_release(recv_req_ptr);
        }
    }




    /* check if multiple threads are calling this collective function */
    MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT(comm_ptr);

    if (MPIU_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPIU_THREADPRIV_FIELD(op_errno);
    }

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIU_Free(temp_buf);
    return (mpi_errno);

  fn_fail:
    goto fn_exit;
}
