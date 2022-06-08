/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"
#include "secure_allgather.h"

/* -- Begin Profiling Symbol Block for routine MPI_Neighbor_allgather */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Neighbor_allgather = PMPI_Neighbor_allgather
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Neighbor_allgather  MPI_Neighbor_allgather
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Neighbor_allgather as PMPI_Neighbor_allgather
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
                           __attribute__((weak,alias("PMPI_Neighbor_allgather")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Neighbor_allgather
#define MPI_Neighbor_allgather PMPI_Neighbor_allgather

/* any non-MPI functions go here, especially non-static ones */

#undef FUNCNAME
#define FUNCNAME MPIR_Neighbor_allgather_default
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Neighbor_allgather_default(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPID_Comm *comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Request req;

    /* just call the nonblocking version and wait on it */
    mpi_errno = MPIR_Ineighbor_allgather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_ptr, &req);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPIR_Wait_impl(&req, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


/************* Added by Mehran ***************/
#undef FUNCNAME
#define FUNCNAME MPIR_Naive_Neighbor_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Naive_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPID_Comm *comm_ptr)
{

    MPI_Request req;
    int mpi_errno = MPI_SUCCESS;
    int tag = -1;
    
    MPID_Request *reqp = NULL;
    MPID_Sched_t s = MPID_SCHED_NULL;

    req = MPI_REQUEST_NULL;

    mpi_errno = MPID_Sched_next_tag(comm_ptr, &tag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPID_Sched_create(&s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);


    int indegree, outdegree, weighted;
    int k,l;
    int *srcs, *dsts;
    MPI_Aint recvtype_extent;
    MPIU_CHKLMEM_DECL(2);

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_ptr->local_size * recvcount * recvtype_extent));

    
    mpi_errno = MPIR_Topo_canon_nhb_count(comm_ptr, &indegree, &outdegree, &weighted);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIU_CHKLMEM_MALLOC(srcs, int *, indegree*sizeof(int), mpi_errno, "srcs");
    MPIU_CHKLMEM_MALLOC(dsts, int *, outdegree*sizeof(int), mpi_errno, "dsts");
    mpi_errno = MPIR_Topo_canon_nhb(comm_ptr,
                                    indegree, srcs, MPI_UNWEIGHTED,
                                    outdegree, dsts, MPI_UNWEIGHTED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    //encrypt

    unsigned long  ciphertext_sendbuf_len = 0;
    RAND_bytes(ciphertext_sendbuf, 12); // 12 bytes of nonce

    unsigned long t=0;
    t = (unsigned long)(recvtype_extent*sendcount);
    unsigned long   max_out_len = (unsigned long) (16 + (recvtype_extent*sendcount));

    if(!EVP_AEAD_CTX_seal(ctx, ciphertext_sendbuf+12,
                        &ciphertext_sendbuf_len, max_out_len,
                        ciphertext_sendbuf, 12,
                        sendbuf,  t,
                        NULL, 0))
    {
            printf("Error in encryption: Naive Ineighbor_allgather\n");
            fflush(stdout);
    }


    //send

    for (k = 0; k < outdegree; ++k) {
        mpi_errno = MPID_Sched_send(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR, dsts[k], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    
    //recv

    for (l = 0; l < indegree; ++l) {
        char *rb = ((char *)ciphertext_recvbuf) + l * (recvcount * recvtype_extent + 16 + 12);
        mpi_errno = MPID_Sched_recv(rb, ((recvcount*recvtype_extent) + 16+12), MPI_CHAR, srcs[l], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    MPID_SCHED_BARRIER(s);

    

    mpi_errno = MPID_Sched_start(&s, comm_ptr, tag, &reqp);
    if (reqp)
        req = reqp->handle;
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIR_Wait_impl(&req, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    //decrypt

    unsigned long count=0;
    unsigned long next, dest;
    unsigned int i;
    for (i = 0; i < indegree; ++i) {
        next =(unsigned long )(i*((recvcount*recvtype_extent) + 16+12));
        dest =(unsigned long )(i*(recvcount*recvtype_extent));
        if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                    &count, (unsigned long )((recvcount*recvtype_extent)+16),
                        (ciphertext_recvbuf+next), 12,
                    (ciphertext_recvbuf+next+12), (unsigned long )((recvcount*recvtype_extent)+16),
                    NULL, 0)){
                printf("Decryption error: Naive Ineighbor_allgather while %d tried to decrypt from %d to %d\n", comm_ptr->rank, next, dest);fflush(stdout);        
        }  
    }


  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}



#undef FUNCNAME
#define FUNCNAME MPIR_NEW_Naive_Neighbor_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_NEW_Naive_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPID_Comm *comm_ptr)
{

    MPI_Request req;
    int mpi_errno = MPI_SUCCESS;
    int tag = -1;
    
    MPID_Request *reqp = NULL;
    MPID_Sched_t s = MPID_SCHED_NULL;

    req = MPI_REQUEST_NULL;

    mpi_errno = MPID_Sched_next_tag(comm_ptr, &tag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPID_Sched_create(&s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);


    int indegree, outdegree, weighted;
    int k,l;
    int *srcs, *dsts;
    MPI_Aint recvtype_extent;
    MPIU_CHKLMEM_DECL(2);

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_ptr->local_size * recvcount * recvtype_extent));

    
    mpi_errno = MPIR_Topo_canon_nhb_count(comm_ptr, &indegree, &outdegree, &weighted);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIU_CHKLMEM_MALLOC(srcs, int *, indegree*sizeof(int), mpi_errno, "srcs");
    MPIU_CHKLMEM_MALLOC(dsts, int *, outdegree*sizeof(int), mpi_errno, "dsts");
    mpi_errno = MPIR_Topo_canon_nhb(comm_ptr,
                                    indegree, srcs, MPI_UNWEIGHTED,
                                    outdegree, dsts, MPI_UNWEIGHTED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    //encrypt
    int encryption_overhead = 16, nonce_len = 12;
    unsigned long  ciphertext_sendbuf_len = 0;
   
    mpi_errno = MPID_Sched_crypt(sendbuf, recvtype_extent*sendcount, encryption_overhead, nonce_len, 
                                ciphertext_sendbuf, &ciphertext_sendbuf_len, encryption_overhead + (recvtype_extent*sendcount), 'E', s);

    if (mpi_errno) MPIR_ERR_POP(mpi_errno);


    //send

    for (k = 0; k < outdegree; ++k) {
        mpi_errno = MPID_Sched_send(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR, dsts[k], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    
    //recv
    MPI_Status status;
    for (l = 0; l < indegree; ++l) {
        char *rb = ((char *)ciphertext_recvbuf) + l * (recvcount * recvtype_extent + 16 + 12);
        mpi_errno = MPID_Sched_brecv(rb, ((recvcount*recvtype_extent) + 16+12), MPI_CHAR, srcs[l], comm_ptr, &status, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    //decrypt

    unsigned long count=0;
    unsigned long next, dest;
    unsigned int i;
    for (i = 0; i < indegree; ++i) {
        next =(unsigned long )(i*((recvcount*recvtype_extent) + 16+12));
        dest =(unsigned long )(i*(recvcount*recvtype_extent));
        mpi_errno = MPID_Sched_crypt(ciphertext_recvbuf+next, recvcount*recvtype_extent+encryption_overhead, encryption_overhead, nonce_len, 
                                recvbuf+dest, &count, encryption_overhead + (recvtype_extent*recvcount), 'D', s);
    }


    MPID_SCHED_BARRIER(s);

    

    mpi_errno = MPID_Sched_start(&s, comm_ptr, tag, &reqp);
    if (reqp)
        req = reqp->handle;
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIR_Wait_impl(&req, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

   

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}








#undef FUNCNAME
#define FUNCNAME MPIR_NaivePlus_Neighbor_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_NaivePlus_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPID_Comm *comm_ptr)
{

    MPI_Request req;
    int mpi_errno = MPI_SUCCESS;
    int tag = -1;
    bool encrypt = true;
    MPID_Request *reqp = NULL;
    MPID_Sched_t s = MPID_SCHED_NULL;

    req = MPI_REQUEST_NULL;

    mpi_errno = MPID_Sched_next_tag(comm_ptr, &tag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPID_Sched_create(&s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);


    int indegree, outdegree, weighted;
    int k,l;
    int *srcs, *dsts;
    MPI_Aint recvtype_extent;
    MPIU_CHKLMEM_DECL(2);

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIU_Ensure_Aint_fits_in_pointer(MPIU_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_ptr->local_size * recvcount * recvtype_extent));

    
    mpi_errno = MPIR_Topo_canon_nhb_count(comm_ptr, &indegree, &outdegree, &weighted);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIU_CHKLMEM_MALLOC(srcs, int *, indegree*sizeof(int), mpi_errno, "srcs");
    MPIU_CHKLMEM_MALLOC(dsts, int *, outdegree*sizeof(int), mpi_errno, "dsts");
    mpi_errno = MPIR_Topo_canon_nhb(comm_ptr,
                                    indegree, srcs, MPI_UNWEIGHTED,
                                    outdegree, dsts, MPI_UNWEIGHTED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    unsigned long  ciphertext_sendbuf_len = 0;
    RAND_bytes(ciphertext_sendbuf, 12); // 12 bytes of nonce

    unsigned long t=0;
    t = (unsigned long)(recvtype_extent*sendcount);
    unsigned long   max_out_len = (unsigned long) (16 + (recvtype_extent*sendcount));


    //send
    //    printf("%d @ Check 0\n", comm_ptr->rank);
    int my_node = comm_ptr->dev.ch.leader_map[comm_ptr->rank];
    for (k = 0; k < outdegree; ++k) {

        if(my_node != comm_ptr->dev.ch.leader_map[dsts[k]]){//Inter Node
            //printf("%d @ Check 01-%d, node=%d , dst=%d, dst_node=%d\n", comm_ptr->rank, k, my_node, dsts[k], comm_ptr->dev.ch.leader_map[dsts[k]]);
            if(encrypt){
                //encrypt
                //printf("%d @ Check 02-%d\n", comm_ptr->rank, k);
                if(!EVP_AEAD_CTX_seal(ctx, ciphertext_sendbuf+12,
                                    &ciphertext_sendbuf_len, max_out_len,
                                    ciphertext_sendbuf, 12,
                                    sendbuf,  t,
                                    NULL, 0))
                {
                        printf("Error in encryption: Naive Ineighbor_allgather\n");
                        fflush(stdout);
                }
                encrypt = false;
            }
            //printf("%d @ Check 03-%d\n", comm_ptr->rank, k);
            mpi_errno = MPID_Sched_send(ciphertext_sendbuf, ciphertext_sendbuf_len+12, MPI_CHAR, dsts[k], comm_ptr, s);
            //printf("%d @ Check 04-%d\n", comm_ptr->rank, k);
        }
        else{ //Intra Node
            mpi_errno = MPID_Sched_send(sendbuf, sendcount, sendtype, dsts[k], comm_ptr, s);
            //printf("%d @ Check 05-%d\n", comm_ptr->rank, k);
        }
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    //    printf("%d @ Check 1\n", comm_ptr->rank);
    //recv

    for (l = 0; l < indegree; ++l) {
        if(comm_ptr->dev.ch.leader_map[comm_ptr->rank] != comm_ptr->dev.ch.leader_map[srcs[l]]){//Inter Node
            char *rb = ((char *)ciphertext_recvbuf) + l * (recvcount * recvtype_extent + 16 + 12);
            mpi_errno = MPID_Sched_recv(rb, ((recvcount*recvtype_extent) + 16+12), MPI_CHAR, srcs[l], comm_ptr, s);
        }else{
            char *rb = ((char *)recvbuf) + l * recvcount * recvtype_extent;
            mpi_errno = MPID_Sched_recv(rb, recvcount, recvtype, srcs[l], comm_ptr, s);
        }
        
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    //    printf("%d @ Check 2\n", comm_ptr->rank);
    MPID_SCHED_BARRIER(s);

    

    mpi_errno = MPID_Sched_start(&s, comm_ptr, tag, &reqp);
    if (reqp)
        req = reqp->handle;
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIR_Wait_impl(&req, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    //decrypt
    //    printf("%d @ Check 3\n", comm_ptr->rank);
    unsigned long count=0;
    unsigned long next, dest;
    unsigned int i;
    for (i = 0; i < indegree; ++i) {
        if(comm_ptr->dev.ch.leader_map[comm_ptr->rank] != comm_ptr->dev.ch.leader_map[srcs[i]]){//Inter Node
            next =(unsigned long )(i*((recvcount*recvtype_extent) + 16+12));
            dest =(unsigned long )(i*(recvcount*recvtype_extent));
            if(!EVP_AEAD_CTX_open(ctx, ((recvbuf+dest)),
                        &count, (unsigned long )((recvcount*recvtype_extent)+16),
                            (ciphertext_recvbuf+next), 12,
                        (ciphertext_recvbuf+next+12), (unsigned long )((recvcount*recvtype_extent)+16),
                        NULL, 0)){
                    printf("Decryption error: Naive Ineighbor_allgather while %d tried to decrypt from %d to %d\n", comm_ptr->rank, next, dest);fflush(stdout);        
            }  
        }//end if
    }//end for
    //    printf("%d @ Check 4\n", comm_ptr->rank);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


    /*******************************************************************************/



#undef FUNCNAME
#define FUNCNAME MPIR_Neighbor_allgather_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Neighbor_allgather_impl(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPID_Comm *comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIU_Assert(comm_ptr->coll_fns != NULL);
    MPIU_Assert(comm_ptr->coll_fns->Neighbor_allgather != NULL);
    mpi_errno = comm_ptr->coll_fns->Neighbor_allgather(sendbuf, sendcount, sendtype,
                                                       recvbuf, recvcount, recvtype,
                                                       comm_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#endif /* MPICH_MPI_FROM_PMPI */

#undef FUNCNAME
#define FUNCNAME MPI_Neighbor_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Neighbor_allgather - In this function, each process i gathers data items
from each process j if an edge (j,i) exists in the topology graph, and each
process i sends the same data items to all processes j where an edge (i,j)
exists. The send buffer is sent to each neighboring process and the l-th block
in the receive buffer is received from the l-th neighbor.

Input Parameters:
+ sendbuf - starting address of the send buffer (choice)
. sendcount - number of elements sent to each neighbor (non-negative integer)
. sendtype - data type of send buffer elements (handle)
. recvcount - number of elements received from each neighbor (non-negative integer)
. recvtype - data type of receive buffer elements (handle)
- comm - communicator (handle)

Output Parameters:
. recvbuf - starting address of the receive buffer (choice)

.N ThreadSafe

.N Fortran

.N Errors
@*/
int MPI_Neighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_NEIGHBOR_ALLGATHER);

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_FUNC_ENTER(MPID_STATE_MPI_NEIGHBOR_ALLGATHER);

    /* Validate parameters, especially handles needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS
        {
            MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
            MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
            MPIR_ERRTEST_COMM(comm, mpi_errno);

            /* TODO more checks may be appropriate */
        }
        MPID_END_ERROR_CHECKS
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPID_Comm_get_ptr(comm, comm_ptr);

    /* Validate parameters and objects (post conversion) */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS
        {
            if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                MPID_Datatype *sendtype_ptr = NULL;
                MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                MPID_Datatype_valid_ptr(sendtype_ptr, mpi_errno);
                MPID_Datatype_committed_ptr(sendtype_ptr, mpi_errno);
            }

            if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                MPID_Datatype *recvtype_ptr = NULL;
                MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                MPID_Datatype_valid_ptr(recvtype_ptr, mpi_errno);
                MPID_Datatype_committed_ptr(recvtype_ptr, mpi_errno);
            }

            MPID_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            /* TODO more checks may be appropriate (counts, in_place, buffer aliasing, etc) */
            if (mpi_errno != MPI_SUCCESS) goto fn_fail;
        }
        MPID_END_ERROR_CHECKS
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPIR_Neighbor_allgather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* ... end of body of routine ... */

fn_exit:
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPI_NEIGHBOR_ALLGATHER);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
            "**mpi_neighbor_allgather", "**mpi_neighbor_allgather %p %d %D %p %d %D %C", sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm(NULL, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
