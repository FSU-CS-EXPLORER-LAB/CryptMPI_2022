/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2009 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Reduce_local */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Reduce_local = PMPI_Reduce_local
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Reduce_local  MPI_Reduce_local
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Reduce_local as PMPI_Reduce_local
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype,
                     MPI_Op op)
                     __attribute__((weak,alias("PMPI_Reduce_local")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Reduce_local
#define MPI_Reduce_local PMPI_Reduce_local
/* any utility functions should go here, usually prefixed with PMPI_LOCAL to
 * correctly handle weak symbols and the profiling interface */

#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_local_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Reduce_local_impl(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Op *op_ptr;
    MPI_User_function *uop;
#ifdef HAVE_CXX_BINDING
    int is_cxx_uop = 0;
#endif
#if defined(HAVE_FORTRAN_BINDING) && !defined(HAVE_FINT_IS_INT)
    int is_f77_uop = 0;
#endif
    MPID_THREADPRIV_DECL;

    if (count == 0) goto fn_exit;

    MPID_THREADPRIV_GET;
    MPID_THREADPRIV_FIELD(op_errno) = MPI_SUCCESS;

#ifdef _ENABLE_CUDA_
    int stride = 0;
    MPI_Aint true_lb, true_extent, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);
    stride = count * MPIR_MAX(extent, true_extent);
    int recv_mem_type = 0;
    int send_mem_type = 0;
    char *recv_host_buf = NULL;
    char *send_host_buf = NULL;
    char *temp_recvbuf = inoutbuf;
    const char *temp_sendbuf = inbuf;

    if (rdma_enable_cuda) {
       recv_mem_type = is_device_buffer(inoutbuf);
       if ( inbuf != MPI_IN_PLACE ){
         send_mem_type = is_device_buffer(inbuf);
       }
    }
    if(rdma_enable_cuda && send_mem_type){
        send_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)send_host_buf, 
                            (void *)inbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        inbuf = send_host_buf;
    }

    if(rdma_enable_cuda && recv_mem_type){
        recv_host_buf = (char*) MPIU_Malloc(stride);
        MPIU_Memcpy_CUDA((void *)recv_host_buf, 
                            (void *)inoutbuf, 
                            stride,
                            cudaMemcpyDeviceToHost);
        inoutbuf = recv_host_buf;
    }
#endif

    if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
        /* get the function by indexing into the op table */
        uop = MPIR_OP_HDL_TO_FN(op);
    }
    else {
        MPID_Op_get_ptr(op, op_ptr);

#ifdef HAVE_CXX_BINDING
        if (op_ptr->language == MPID_LANG_CXX) {
            uop = (MPI_User_function *) op_ptr->function.c_function;
            is_cxx_uop = 1;
        }
        else
#endif
        {
            if (op_ptr->language == MPID_LANG_C) {
                uop = (MPI_User_function *) op_ptr->function.c_function;
            }
            else {
                uop = (MPI_User_function *) op_ptr->function.f77_function;
#if defined(HAVE_FORTRAN_BINDING) && !defined(HAVE_FINT_IS_INT)
                is_f77_uop = 1;
#endif
            }
        }
    }

    /* actually perform the reduction */
#ifdef HAVE_CXX_BINDING
    if (is_cxx_uop) {
        (*MPIR_Process.cxx_call_op_fn)(inbuf, inoutbuf, count, datatype, uop);
    }
    else
#endif
    {
#if defined(HAVE_FORTRAN_BINDING) && !defined(HAVE_FINT_IS_INT)
        if (is_f77_uop) {
            MPI_Fint lcount = (MPI_Fint)count;
            MPI_Fint ldtype = (MPI_Fint)datatype;
            MPIR_F77_User_function *uop_f77 = (MPIR_F77_User_function *)uop;

            (*uop_f77)((void *) inbuf, inoutbuf, &lcount, &ldtype);
        }
        else {
            (*uop)((void *) inbuf, inoutbuf, &count, &datatype);
        }
#else
        (*uop)((void *) inbuf, inoutbuf, &count, &datatype);
#endif
    }
    
#ifdef _ENABLE_CUDA_
    if(rdma_enable_cuda && recv_mem_type){
        inoutbuf = temp_recvbuf;
        inbuf = temp_sendbuf;
        MPIU_Memcpy_CUDA((void *)inoutbuf, 
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

    /* --BEGIN ERROR HANDLING-- */
    if (MPID_THREADPRIV_FIELD(op_errno))
        mpi_errno = MPID_THREADPRIV_FIELD(op_errno);
    /* --END ERROR HANDLING-- */

fn_exit:
    return mpi_errno;
}

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Reduce_local
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Reduce_local - Applies a reduction operator to local arguments.

Input Parameters:
+ inbuf - address of the input buffer (choice)
. count - number of elements in each buffer (integer)
. datatype - data type of elements in the buffers (handle)
- op - reduction operation (handle)

Output Parameters:
. inoutbuf - address of input-output buffer (choice)

.N ThreadSafe

.N Fortran

.N collops

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COUNT
.N MPI_ERR_TYPE
.N MPI_ERR_BUFFER
.N MPI_ERR_BUFFER_ALIAS
@*/
int MPI_Reduce_local(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_REDUCE_LOCAL);

    MPIR_ERRTEST_INITIALIZED_ORDIE();

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_COLL_FUNC_ENTER(MPID_STATE_MPI_REDUCE_LOCAL);

    /* Validate parameters */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_OP(op, mpi_errno);

            if (HANDLE_GET_KIND(op) != HANDLE_KIND_BUILTIN) {
                MPID_Op *op_ptr;
                MPID_Op_get_ptr(op, op_ptr);
                MPID_Op_valid_ptr( op_ptr, mpi_errno );
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }
            if (HANDLE_GET_KIND(op) == HANDLE_KIND_BUILTIN) {
                mpi_errno = (*MPIR_OP_HDL_TO_DTYPE_FN(op))(datatype);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }
            if (count != 0) {
                MPIR_ERRTEST_ALIAS_COLL(inbuf, inoutbuf, mpi_errno);
            }
            MPIR_ERRTEST_NAMED_BUF_INPLACE(inbuf, "inbuf", count, mpi_errno);
            MPIR_ERRTEST_NAMED_BUF_INPLACE(inoutbuf, "inoutbuf", count, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */


    /* ... body of routine ...  */

    mpi_errno = MPIR_Reduce_local_impl(inbuf, inoutbuf, count, datatype, op);

    /* ... end of body of routine ... */

  fn_exit:
    MPID_MPI_COLL_FUNC_EXIT(MPID_STATE_MPI_REDUCE_LOCAL);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_reduce_local",
            "**mpi_reduce_local %p %p %d %D %O", inbuf, inoutbuf, count, datatype, op);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( NULL, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

