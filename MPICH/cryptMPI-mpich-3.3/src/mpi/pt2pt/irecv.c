/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Added by Abu Naser(an16e@my.fsu.edu) */
unsigned char Ideciphertext[NON_BLOCKING_SND_RCV_1][NON_BLOCKING_SND_RCV_2];
int ranksIrecvS[MAX_RANKS_LIMIT];
/* End of add by Abu Naser */

/* -- Begin Profiling Symbol Block for routine MPI_Irecv */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Irecv = PMPI_Irecv
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Irecv  MPI_Irecv
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Irecv as PMPI_Irecv
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request * request) __attribute__ ((weak, alias("PMPI_Irecv")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Irecv
#define MPI_Irecv PMPI_Irecv

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Irecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
    MPI_Irecv - Begins a nonblocking receive

Input Parameters:
+ buf - initial address of receive buffer (choice)
. count - number of elements in receive buffer (integer)
. datatype - datatype of each receive buffer element (handle)
. source - rank of source (integer)
. tag - message tag (integer)
- comm - communicator (handle)

Output Parameters:
. request - communication request (handle)

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_COUNT
.N MPI_ERR_TYPE
.N MPI_ERR_TAG
.N MPI_ERR_RANK
.N MPI_ERR_EXHAUSTED
@*/
int MPI_Irecv_original(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_Request *request_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_IRECV);

    MPIR_ERRTEST_INITIALIZED_ORDIE();

    MPID_THREAD_CS_ENTER(VNI_GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_PT2PT_ENTER_BACK(MPID_STATE_MPI_IRECV);

    /* Validate handle parameters needing to be converted */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_COMM(comm, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPIR_Comm_get_ptr(comm, comm_ptr);

    /* Validate parameters if error checking is enabled */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_Comm_valid_ptr(comm_ptr, mpi_errno, FALSE);
            if (mpi_errno)
                goto fn_fail;

            MPIR_ERRTEST_COUNT(count, mpi_errno);
            MPIR_ERRTEST_RECV_RANK(comm_ptr, source, mpi_errno);
            MPIR_ERRTEST_RECV_TAG(tag, mpi_errno);
            MPIR_ERRTEST_ARGNULL(request, "request", mpi_errno);

            /* Validate datatype handle */
            MPIR_ERRTEST_DATATYPE(datatype, "datatype", mpi_errno);

            /* Validate datatype object */
            if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN) {
                MPIR_Datatype *datatype_ptr = NULL;

                MPIR_Datatype_get_ptr(datatype, datatype_ptr);
                MPIR_Datatype_valid_ptr(datatype_ptr, mpi_errno);
                if (mpi_errno)
                    goto fn_fail;
                MPIR_Datatype_committed_ptr(datatype_ptr, mpi_errno);
                if (mpi_errno)
                    goto fn_fail;
            }

            /* Validate buffer */
            MPIR_ERRTEST_USERBUFFER(buf, count, datatype, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPID_Irecv(buf, count, datatype, source, tag, comm_ptr,
                           MPIR_CONTEXT_INTRA_PT2PT, &request_ptr);
    /* return the handle of the request to the user */
    /* MPIU_OBJ_HANDLE_PUBLISH is unnecessary for irecv, lower-level access is
     * responsible for its own consistency, while upper-level field access is
     * controlled by the completion counter */
    *request = request_ptr->handle;

    /* Put this part after setting the request so that if the request is
     * pending (which is still considered an error), it will still be set
     * correctly here. For real error cases, the user might get garbage as
     * their request value, but that's fine since the definition is
     * undefined anyway. */
    if (mpi_errno != MPI_SUCCESS)
        goto fn_fail;

    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_PT2PT_EXIT_BACK(MPID_STATE_MPI_IRECV);
    MPID_THREAD_CS_EXIT(VNI_GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
                                 "**mpi_irecv", "**mpi_irecv %p %d %D %i %t %C %p", buf, count,
                                 datatype, source, tag, comm, request);
    }
#endif
    mpi_errno = MPIR_Err_return_comm(comm_ptr, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}



/* Added by Abu Naser */
/**********************/

int MPI_SEC_MultiThread_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status sta;
    MPI_Request req;

    unsigned long ciphertext_len = 0;
    int recvtype_sz, m, segments_no, totaldata;
    int th_data, th_start, i;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &recvtype_sz);

    nonblock_req_handler[nonBlockCounter].source = source;
    nonblock_req_handler[nonBlockCounter].dest = comm_ptr->rank;
    ;
    nonblock_req_handler[nonBlockCounter].tag = tag;

    nonblock_req_handler[nonBlockCounter].req_type = 2;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    *request = nonBlockCounter;

    int req_counter = 0;
    totaldata = recvtype_sz * count;
    mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], totaldata + MSG_HEADER_SIZE + 16, MPI_UNSIGNED_CHAR, source, tag, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);

    th_start = MSG_HEADER_SIZE;

    int my_thread_no = 1; //MY_OPENMP_THREADS_NO;
#if OMP_DYNAMIC_THREADS
    if (totaldata < SIXTY_4K)
        my_thread_no = 1;
    else if (totaldata < ONE_TWO_8K)
        my_thread_no = 2;
    else if (totaldata < FIVE_ONE_2K)
        my_thread_no = 4;
    else if (totaldata >= FIVE_ONE_2K)
        my_thread_no = 8;
#elif OMP_DYNAMIC_THREADS_INNOVATION
    my_thread_no = 14; // posting a max thread number, so that posting data would be large
#endif

    segments_no = 1;

    mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start], totaldata + (16 + 12) * segments_no, MPI_CHAR, source,
                                   tag, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);

    nonblock_req_handler[nonBlockCounter].segment_number = 1;
    nonblock_req_handler[nonBlockCounter].total_request = 2;

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_Irecv2_largeSegment_3(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status sta;
    MPI_Request req;

    unsigned long ciphertext_len = 0;
    int recvtype_sz, m, segments_no, totaldata;
    int th_data, th_start, i;
    int my_thread_no;
    int flag;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &recvtype_sz);

    nonblock_req_handler[nonBlockCounter].source = source;
    nonblock_req_handler[nonBlockCounter].dest = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].tag = tag;

    nonblock_req_handler[nonBlockCounter].req_type = 2;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    *request = nonBlockCounter;

    int req_counter = 0;
    totaldata = recvtype_sz * count;
    //    printf(COLOR_BLUE"    %d recv from %d totaldata = %d tag =%d"COLOR_RESET"\n", comm_ptr->rank, source, totaldata,tag );fflush(stdout);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION

    /* Check for Probe */
    if (Crypthandle_probe[source].no_tag > 0)
    {
        flag = 0;
        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
        {
            if (Crypthandle_probe[source].tags[i] == tag)
            {
                flag = 1;
                tag++;
                req_counter++;
                ranksIrecvS[source] += 1;
                //printf("irecv: source = %d i = %d no_tag = %d\n",source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
                break;
            }
        }

        if (flag == 0)
        {
            mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], totaldata + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
            ranksIrecvS[source] += 1;
        }
    }
    else
    {
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], totaldata + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        ranksIrecvS[source] += 1;
    }

#else
    mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], totaldata + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
#endif
    // tag++;

    int max_data_recv_size;
    int inner_segments = 0;

#if OMP_DYNAMIC_THREADS_PIPELINE
    choping_sz = PIPELINE_SIZE;
    max_data_recv_size = totaldata + MAX_OMP_THREADS_LIMIT * (NONCE_HEADER + ENC_MSG_TAG_SIZE);

    inner_segments = MAX_OMP_THREADS_LIMIT;

#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    choping_sz = 16384;
    max_data_recv_size = totaldata + MAX_OMP_THREADS_LIMIT * (NONCE_HEADER + ENC_MSG_TAG_SIZE);
#endif

    if ((totaldata > PIPELINE_SIZE) && (totaldata > LARGE_SEGMENT_SIZE))
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    if (ranksIrecvS[source] + segments_no > MAX_PENDING_ISEND_LIMIT)
    { // already increased for the header
        segments_no = 1;
        ranksIrecvS[source] += 1;
        //printf(COLOR_MAGENTA"    >ranksIrecvS[%d] = %d "COLOR_RESET"\n",source,ranksIrecvS[source]); fflush(stdout);
    }
    else
    {
        ranksIrecvS[source] += segments_no;
        // printf(COLOR_CYAN"    ranksIrecvS[%d] = %d "COLOR_RESET"\n",source,ranksIrecvS[source]); fflush(stdout);
    }
#endif

    th_start = MSG_HEADER_SIZE;
    for (m = 0; m < segments_no; m++)
    {
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start], max_data_recv_size + NONCE_HEADER, MPI_UNSIGNED_CHAR, source,
                                       tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        //  printf(COLOR_BLUE" rank %d recv body from source %d tag %d"COLOR_RESET"\n",comm_ptr->rank,source,tag);
        /* mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start],131452, MPI_UNSIGNED_CHAR, source,
                                                                   tag, comm, &irecvRequestQ[front_of_irecvRequestQ++]); */
        //printf("m = %d segments_no = %d th_start = %d\n",m, segments_no, th_start);fflush(stdout);
        th_start += PIPELINE_SIZE + inner_segments * (NONCE_HEADER + ENC_MSG_TAG_SIZE);
#else
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start], choping_sz + NONCE_HEADER + ENC_MSG_TAG_SIZE + 1000, MPI_UNSIGNED_CHAR, source,
                                       tag, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        th_start += choping_sz + NONCE_HEADER + ENC_MSG_TAG_SIZE;

#endif
    }
    nonblock_req_handler[nonBlockCounter].segment_number = segments_no;
    nonblock_req_handler[nonBlockCounter].total_request = req_counter;

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_BaseCounter_Pipeline_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status sta;
    MPI_Request req;

    unsigned long ciphertext_len = 0;
    int recvtype_sz, m, segments_no, totaldata;
    int th_data, th_start, i;
    int my_thread_no;
    int flag;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &recvtype_sz);

    nonblock_req_handler[nonBlockCounter].source = source;
    nonblock_req_handler[nonBlockCounter].dest = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].tag = tag;

    nonblock_req_handler[nonBlockCounter].req_type = 2;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    *request = nonBlockCounter;

    int req_counter = 0;
    totaldata = recvtype_sz * count;
    //  printf(COLOR_BLUE"    %d recv from %d totaldata = %d tag =%d"COLOR_RESET"\n", comm_ptr->rank, source, totaldata,tag );fflush(stdout);

    /* Check for Probe */
    if (Crypthandle_probe[source].no_tag > 0)
    {
        flag = 0;
        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
        {
            if (Crypthandle_probe[source].tags[i] == tag)
            {
                flag = 1;
                tag++;
                req_counter++;

                //printf("irecv: source = %d i = %d no_tag = %d\n",source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
                break;
            }
        }

        if (flag == 0)
        {
            mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        }
    }
    else
    {
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
    }

    if ((totaldata > PIPELINE_SIZE) && (totaldata > LARGE_SEGMENT_SIZE))
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }

    if (BASE_COUNTER_NO_PIPELINE)
        segments_no = 1;

    th_start = COUNTER_HEADER_SIZE;
    for (m = 0; m < segments_no; m++)
    {
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start], totaldata, MPI_UNSIGNED_CHAR, source,
                                       tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        //printf("m = %d segments_no = %d th_start = %d\n",m, segments_no, th_start);fflush(stdout);
        th_start += PIPELINE_SIZE;
    }
    nonblock_req_handler[nonBlockCounter].segment_number = segments_no;
    nonblock_req_handler[nonBlockCounter].total_request = req_counter;
    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_PreComputation_Pipeline_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status sta;
    MPI_Request req;

    unsigned long ciphertext_len = 0;
    int recvtype_sz, m, segments_no, totaldata;
    int th_data, th_start, i;
    int my_thread_no;
    int flag;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &recvtype_sz);

    nonblock_req_handler[nonBlockCounter].source = source;
    nonblock_req_handler[nonBlockCounter].dest = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].tag = tag;

    nonblock_req_handler[nonBlockCounter].req_type = 2;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    *request = nonBlockCounter;

    int req_counter = 0;
    totaldata = recvtype_sz * count;
    //    printf(COLOR_BLUE"    %d recv from %d totaldata = %d tag =%d"COLOR_RESET"\n", comm_ptr->rank, source, totaldata,tag );fflush(stdout);

    /* Check for Probe */
    if (Crypthandle_probe[source].no_tag > 0)
    {
        flag = 0;
        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
        {
            if (Crypthandle_probe[source].tags[i] == tag)
            {
                flag = 1;
                tag++;
                req_counter++;
                ranksIrecvS[source] += 1;
                //printf("irecv: source = %d i = %d no_tag = %d\n",source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
                break;
            }
        }

        if (flag == 0)
        {
            mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
            ranksIrecvS[source] += 1;
        }
    }
    else
    {
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        ranksIrecvS[source] += 1;
    }

    if ((totaldata > PIPELINE_SIZE) && (totaldata > LARGE_SEGMENT_SIZE))
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }
    if (ranksIrecvS[source] + segments_no > MAX_PENDING_ISEND_LIMIT)
    { // already increased for the header
        //choping_sz = max_data_recv_size;
        segments_no = 1;
        ranksIrecvS[source] += 1;
        //printf(COLOR_MAGENTA"    >ranksIrecvS[%d] = %d "COLOR_RESET"\n",source,ranksIrecvS[source]); fflush(stdout);
    }
    else
    {
        ranksIrecvS[source] += segments_no;
        // printf(COLOR_CYAN"    ranksIrecvS[%d] = %d "COLOR_RESET"\n",source,ranksIrecvS[source]); fflush(stdout);
    }

    th_start = COUNTER_HEADER_SIZE;
    for (m = 0; m < segments_no; m++)
    {
        mpi_errno = MPI_Irecv_original(&Ideciphertext[nonBlockCounter][th_start], totaldata, MPI_UNSIGNED_CHAR, source,
                                       tag++, comm, &nonblock_req_handler[nonBlockCounter].request[req_counter++]);
        //printf("m = %d segments_no = %d th_start = %d\n",m, segments_no, th_start);fflush(stdout);
        th_start += PIPELINE_SIZE;
    }
    nonblock_req_handler[nonBlockCounter].segment_number = segments_no;
    nonblock_req_handler[nonBlockCounter].total_request = req_counter;

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    int var, recvtype_sz;
    MPIR_Comm *comm_ptr = NULL;
    MPI_Type_size(datatype, &recvtype_sz);
    int datasz = count * recvtype_sz;
    MPIR_Comm_get_ptr(comm, comm_ptr);
    int isSamenode = 0;
    if (source > -1)
        isSamenode = sameNode[source];

    //printf(COLOR_CYAN "[Rank %d]  receiving from source %d dest = %d tag =%d datasize=%d" COLOR_RESET "\n",  comm_ptr->rank, source,comm_ptr->rank, tag,count*recvtype_sz);fflush(stdout);

    if (comm_ptr->rank == source || source < 0)
        mpi_errno = MPI_Irecv_original(buf, count, datatype, source, tag, comm, request);
    //#if SKIP_INTRA_ENC
    else if (isSamenode > -1)
    {
        if (security_approach == 600) // encryt intra also
            mpi_errno = MPI_SEC_MultiThread_Irecv(buf, count, datatype, source, tag, comm, request);
        else if (security_approach == 700) // encryt intra also
            mpi_errno = MPI_SEC_BaseCounter_Pipeline_Irecv(buf, count, datatype, source, tag, comm, request);
        else
            mpi_errno = MPI_Irecv_original(buf, count, datatype, source, tag, comm, request);
    }
    //#endif
    else if (security_approach == 600 || security_approach == 601) //(MULTI_THREAD_ONLY || OPENMP_MULTI_THREAD_ONLY)
        mpi_errno = MPI_SEC_MultiThread_Irecv(buf, count, datatype, source, tag, comm, request);
    else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        mpi_errno = MPI_SEC_Irecv2_largeSegment_3(buf, count, datatype, source, tag, comm, request);
    else if (security_approach == 700 || security_approach == 701) //if(BASE_COUNTER_MODE)
        mpi_errno = MPI_SEC_BaseCounter_Pipeline_Irecv(buf, count, datatype, source, tag, comm, request);
    else if (security_approach == 702) //(PRE_COMPUTE_COUNTER_MODE)
        mpi_errno = MPI_SEC_PreComputation_Pipeline_Irecv(buf, count, datatype, source, tag, comm, request);

    else
    {
        mpi_errno = MPI_Irecv_original(buf, count, datatype, source, tag, comm, request);
    }

    return mpi_errno;
}

/* End of add by Abu Naser */
