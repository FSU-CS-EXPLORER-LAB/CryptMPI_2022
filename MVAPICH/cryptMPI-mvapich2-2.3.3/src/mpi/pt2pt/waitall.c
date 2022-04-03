/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

#if !defined(MPID_REQUEST_PTR_ARRAY_SIZE)
/* use a larger default size of 64 in order to enhance SQMR performance */
#define MPID_REQUEST_PTR_ARRAY_SIZE 64
#endif

/* -- Begin Profiling Symbol Block for routine MPI_Waitall */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Waitall = PMPI_Waitall
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Waitall  MPI_Waitall
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Waitall as PMPI_Waitall
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) __attribute__((weak,alias("PMPI_Waitall")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Waitall
#define MPI_Waitall PMPI_Waitall


/* The "fastpath" version of MPIR_Request_complete.  It only handles
 * MPID_REQUEST_SEND and MPID_REQUEST_RECV kinds, and it does not attempt to
 * deal with status structures under the assumption that bleeding fast code will
 * pass either MPI_STATUS_IGNORE or MPI_STATUSES_IGNORE as appropriate.  This
 * routine (or some a variation of it) is an unfortunately necessary stunt to
 * get high message rates on key benchmarks for high-end systems.
 */
#undef FUNCNAME
#define FUNCNAME request_complete_fastpath
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int request_complete_fastpath(MPI_Request *request, MPID_Request *request_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIU_Assert(request_ptr->kind == MPID_REQUEST_SEND || request_ptr->kind == MPID_REQUEST_RECV);

    if (request_ptr->kind == MPID_REQUEST_SEND) {
        /* FIXME: are Ibsend requests added to the send queue? */
        MPIR_SENDQ_FORGET(request_ptr);
    }

    /* the completion path for SEND and RECV is the same at this time, modulo
     * the SENDQ hook above */
    mpi_errno = request_ptr->status.MPI_ERROR;
    MPID_Request_release(request_ptr);
    *request = MPI_REQUEST_NULL;

    /* avoid normal fn_exit/fn_fail jump pattern to reduce jumps and compiler confusion */
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Waitall_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Waitall_impl(int count, MPI_Request array_of_requests[],
                      MPI_Status array_of_statuses[])
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Request * request_ptr_array[MPID_REQUEST_PTR_ARRAY_SIZE];
    MPID_Request ** request_ptrs = request_ptr_array;
    MPI_Status * status_ptr;
    MPID_Progress_state progress_state;
    int i, j;
    int n_completed;
    int active_flag;
    int rc;
    int n_greqs;
    int proc_failure = FALSE;
    int disabled_anysource = FALSE;
    const int ignoring_statuses = (array_of_statuses == MPI_STATUSES_IGNORE);
    int optimize = ignoring_statuses; /* see NOTE-O1 */
    MPIU_CHKLMEM_DECL(1);

    /* Convert MPI request handles to a request object pointers */
    if (count > MPID_REQUEST_PTR_ARRAY_SIZE)
    {
	MPIU_CHKLMEM_MALLOC(request_ptrs, MPID_Request **, count * sizeof(MPID_Request *), mpi_errno, "request pointers");
    }

    n_greqs = 0;
    n_completed = 0;
    for (i = 0; i < count; i++)
    {
	if (array_of_requests[i] != MPI_REQUEST_NULL)
	{
	    MPID_Request_get_ptr(array_of_requests[i], request_ptrs[i]);
	    /* Validate object pointers if error checking is enabled */
#           ifdef HAVE_ERROR_CHECKING
	    {
		MPID_BEGIN_ERROR_CHECKS;
		{
		    MPID_Request_valid_ptr( request_ptrs[i], mpi_errno );
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    MPIR_ERR_CHKANDJUMP1((request_ptrs[i]->kind == MPID_REQUEST_MPROBE),
                                         mpi_errno, MPI_ERR_ARG, "**msgnotreq", "**msgnotreq %d", i);
		}
		MPID_END_ERROR_CHECKS;
	    }
#           endif
            if (request_ptrs[i]->kind != MPID_REQUEST_RECV &&
                request_ptrs[i]->kind != MPID_REQUEST_SEND
#if defined (_SHARP_SUPPORT_)
                && request_ptrs[i]->sharp_req == NULL
#endif
                )
            {
                optimize = FALSE;
            }

            if (request_ptrs[i]->kind == MPID_UREQUEST)
                ++n_greqs;

            /* If one of the requests is an anysource on a communicator that's
             * disabled such communication, convert this operation to a testall
             * instead to prevent getting stuck in the progress engine. */
            if (unlikely(MPIR_CVAR_ENABLE_FT &&
                        MPID_Request_is_anysource(request_ptrs[i]) &&
                        !MPID_Request_is_complete(request_ptrs[i]) &&
                        !MPID_Comm_AS_enabled(request_ptrs[i]->comm))) {
                disabled_anysource = TRUE;
            }
	}
	else
	{
	    status_ptr = (array_of_statuses != MPI_STATUSES_IGNORE) ? &array_of_statuses[i] : MPI_STATUS_IGNORE;
	    MPIR_Status_set_empty(status_ptr);
	    request_ptrs[i] = NULL;
	    n_completed += 1;
            optimize = FALSE;
	}
    }
    
    if (n_completed == count)
    {
	goto fn_exit;
    }

    if (unlikely(disabled_anysource)) {
        mpi_errno = MPIR_Testall_impl(count, array_of_requests, &disabled_anysource, array_of_statuses);
        goto fn_exit;
    }

    /* NOTE-O1: high-message-rate optimization.  For simple send and recv
     * operations and MPI_STATUSES_IGNORE we use a fastpath approach that strips
     * out as many unnecessary jumps and error handling as possible.
     *
     * Possible variation: permit request_ptrs[i]==NULL at the cost of an
     * additional branch inside the for-loop below. */
    if (optimize) {
        MPID_Progress_start(&progress_state);
        for (i = 0; i < count; ++i) {
#if defined (_SHARP_SUPPORT_)
            if (request_ptrs[i]->sharp_req != NULL) {
                mpi_errno = MPID_SHARP_COLL_REQ_WAIT(request_ptrs[i]);
                if (mpi_errno != MPID_SHARP_COLL_SUCCESS) {
                    PRINT_ERROR("SHArP non-blocking collective failed \n ");
                    MPID_SHARP_COLL_REQ_FREE(request_ptrs[i]);
                    mpi_errno = MPI_ERR_INTERN;
                    MPIR_ERR_POP(mpi_errno);
                }
                continue; 
            }
#endif
            while (!MPID_Request_is_complete(request_ptrs[i])) {
                mpi_errno = MPID_Progress_wait(&progress_state);
                /* must check and handle the error, can't guard with HAVE_ERROR_CHECKING, but it's
                 * OK for the error case to be slower */
                if (unlikely(mpi_errno)) {
                    /* --BEGIN ERROR HANDLING-- */
                    if (unlikely(MPIR_CVAR_ENABLE_FT &&
                                MPID_Request_is_anysource(request_ptrs[i]) &&
                                !MPID_Request_is_complete(request_ptrs[i]) &&
                                !MPID_Comm_AS_enabled(request_ptrs[i]->comm))) {
                        MPIR_ERR_SET(mpi_errno, MPI_ERR_IN_STATUS, "**instatus");
                    }
                    MPID_Progress_end(&progress_state);
                    MPIR_ERR_POP(mpi_errno);
                    /* --END ERROR HANDLING-- */
                }
            }
            mpi_errno = request_complete_fastpath(&array_of_requests[i], request_ptrs[i]);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }

        MPID_Progress_end(&progress_state);

        goto fn_exit;
    }

    /* ------ "slow" code path below ------ */

    /* Grequest_waitall may run the progress engine - thus, we don't 
       invoke progress_start until after running Grequest_waitall */
    /* first, complete any generalized requests */
    if (n_greqs)
    {
        mpi_errno = MPIR_Grequest_waitall(count, request_ptrs);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    
    MPID_Progress_start(&progress_state);

    for (i = 0; i < count; i++)
    {
        if (request_ptrs[i] == NULL)
        {
            if (!ignoring_statuses)
                array_of_statuses[i].MPI_ERROR = MPI_SUCCESS;
            continue;
        }
#if defined (_SHARP_SUPPORT_)
        if (request_ptrs[i]->sharp_req != NULL) {
            mpi_errno = MPID_SHARP_COLL_REQ_WAIT(request_ptrs[i]);
            if (mpi_errno != MPID_SHARP_COLL_SUCCESS) {
                PRINT_ERROR("SHArP non-blocking collective failed \n ");
                MPID_SHARP_COLL_REQ_FREE(request_ptrs[i]);
                mpi_errno = MPI_ERR_INTERN;
                MPIR_ERR_POP(mpi_errno);
            }
            continue; 
        }
#endif       
        /* wait for ith request to complete */
        while (!MPID_Request_is_complete(request_ptrs[i]))
        {
            /* generalized requests should already be finished */
            MPIU_Assert(request_ptrs[i]->kind != MPID_UREQUEST);
            
            mpi_errno = MPID_Progress_wait(&progress_state);
            if (mpi_errno != MPI_SUCCESS) {
                /* --BEGIN ERROR HANDLING-- */
                MPID_Progress_end(&progress_state);
                MPIR_ERR_POP(mpi_errno);
                /* --END ERROR HANDLING-- */
            } else if (unlikely(MPIR_CVAR_ENABLE_FT &&
                        MPID_Request_is_anysource(request_ptrs[i]) &&
                        !MPID_Request_is_complete(request_ptrs[i]) &&
                        !MPID_Comm_AS_enabled(request_ptrs[i]->comm))) {
                /* Check for pending failures */
                MPID_Progress_end(&progress_state);
                MPIR_ERR_SET(rc, MPIX_ERR_PROC_FAILED_PENDING, "**failure_pending");
                status_ptr = (ignoring_statuses) ? MPI_STATUS_IGNORE : &array_of_statuses[i];
                if (status_ptr != MPI_STATUS_IGNORE) status_ptr->MPI_ERROR = mpi_errno;
                proc_failure = TRUE;
                break;
            }
        }

        if (MPID_Request_is_complete(request_ptrs[i])) {
            /* complete the request and check the status */
            status_ptr = (ignoring_statuses) ? MPI_STATUS_IGNORE : &array_of_statuses[i];
            rc = MPIR_Request_complete(&array_of_requests[i], request_ptrs[i], status_ptr, &active_flag);
        }

        if (rc == MPI_SUCCESS)
        {
            request_ptrs[i] = NULL;
            if (!ignoring_statuses)
                status_ptr->MPI_ERROR = MPI_SUCCESS;
        }
        else
        {
            /* req completed with an error */
            MPIR_ERR_SET(mpi_errno, MPI_ERR_IN_STATUS, "**instatus");

            if (!proc_failure) {
                if (MPIX_ERR_PROC_FAILED == MPIR_ERR_GET_CLASS(rc))
                    proc_failure = TRUE;
            }

            if (!ignoring_statuses)
            {
                /* set the error code for this request */
                status_ptr->MPI_ERROR = rc;

                /* set the error codes for the rest of the uncompleted requests to PENDING */
                for (j = i+1; j < count; ++j)
                {
                    if (!ignoring_statuses)
                    {
                        if (request_ptrs[j] == NULL)
                        {
                            /* either the user specified MPI_REQUEST_NULL, or this is a completed greq */
                            array_of_statuses[j].MPI_ERROR = MPI_SUCCESS;
                        }
                        else
                        {
                            if (!proc_failure)
                                array_of_statuses[j].MPI_ERROR = MPI_ERR_PENDING;
                            else
                                array_of_statuses[j].MPI_ERROR = MPIX_ERR_PROC_FAILED_PENDING;
                        }
                    }
                }
            }
            break;
        }
    }
    MPID_Progress_end(&progress_state);
        
 fn_exit:
     if (count > MPID_REQUEST_PTR_ARRAY_SIZE)
    {
	MPIU_CHKLMEM_FREEALL();
    }

   return mpi_errno;
 fn_fail:
    goto fn_exit;
}

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Waitall
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
    MPI_Waitall - Waits for all given MPI Requests to complete

Input Parameters:
+ count - list length (integer) 
- array_of_requests - array of request handles (array of handles)

Output Parameters:
. array_of_statuses - array of status objects (array of Statuses).  May be
  'MPI_STATUSES_IGNORE'.

Notes:

If one or more of the requests completes with an error, 'MPI_ERR_IN_STATUS' is
returned.  An error value will be present is elements of 'array_of_status'
associated with the requests.  Likewise, the 'MPI_ERROR' field in the status
elements associated with requests that have successfully completed will be
'MPI_SUCCESS'.  Finally, those requests that have not completed will have a 
value of 'MPI_ERR_PENDING'.

While it is possible to list a request handle more than once in the
array_of_requests, such an action is considered erroneous and may cause the
program to unexecpectedly terminate or produce incorrect results.

.N waitstatus

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_REQUEST
.N MPI_ERR_ARG
.N MPI_ERR_IN_STATUS
@*/
/* Added by Abu Naser(an16e@my.fsu.edu) */
#if 1 //ENABLE_SECURE_MPI
int MPI_Waitall_original(int count, MPI_Request array_of_requests[],
                         MPI_Status array_of_statuses[])
#else
int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[])
#endif
{
    int mpi_errno = MPI_SUCCESS;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_WAITALL);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_PT2PT_FUNC_ENTER(MPID_STATE_MPI_WAITALL);

    /* Check the arguments */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            int i;
	    MPIR_ERRTEST_COUNT(count, mpi_errno);

	    if (count != 0) {
		MPIR_ERRTEST_ARGNULL(array_of_requests, "array_of_requests", mpi_errno);
		/* NOTE: MPI_STATUSES_IGNORE != NULL */
	    
		MPIR_ERRTEST_ARGNULL(array_of_statuses, "array_of_statuses", mpi_errno);
	    }

	    for (i = 0; i < count; i++) {
		MPIR_ERRTEST_ARRAYREQUEST_OR_NULL(array_of_requests[i], i, mpi_errno);
	    }
	}
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */
    
    /* ... body of routine ...  */

    mpi_errno = MPIR_Waitall_impl(count, array_of_requests, array_of_statuses);
    if (mpi_errno) goto fn_fail;

    /* ... end of body of routine ... */
    
 fn_exit:
    MPID_MPI_PT2PT_FUNC_EXIT(MPID_STATE_MPI_WAITALL);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#ifdef HAVE_ERROR_CHECKING
    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
				     FCNAME, __LINE__, MPI_ERR_OTHER,
				     "**mpi_waitall",
				     "**mpi_waitall %d %p %p",
				     count, array_of_requests,
				     array_of_statuses);
#endif
    mpi_errno = MPIR_Err_return_comm(NULL, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/* Added by Abu Naser */
int MPI_SEC_Waitall_MultiThreads_openmp(int req_count, MPI_Request arr_req[],
                                   MPI_Status arr_sta[])
{
    int mpi_errno = MPI_SUCCESS;

    int var, j;
    int sendtype_sz, tag, i, recv_len, len;
    unsigned long next, datasize, dest;
    char newkey[40];
    unsigned char nonce[50], V[50];
    unsigned long count, totaldata, index, decrypted_len;
    unsigned int segment_counter;
    int segments_no;
    MPI_Request request[300];
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;

    unsigned int nonce_counter;
    int m, th_data, my_thread_no, th_start, temp_chunk, n;
    int guided_chunk;

    EVP_AEAD_CTX *local_ctx = NULL;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);

    int temp_req_counter = 0;
    int cancel_req_no;
    int pos, start, max_out_len, th_sum;
    struct timeval t_s1, t_s2;
    int isSamenode = 0;

    for (n = 0; n < req_count; n++)
    {
        index = arr_req[n];

        if ((index >= 0 && index < ISEND_REQ))
        {
            if (nonblock_req_handler[index].req_type == 2)
            {

                source = nonblock_req_handler[index].source;
                dest = nonblock_req_handler[index].dest;
                tag = nonblock_req_handler[index].tag;
                nonblock_req_handler[index].req_type = 0;
                buf = nonblock_req_handler[index].buffer;
                int comm = MPI_COMM_WORLD;
                if (dest != source)
                {
#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s1, NULL);
#endif
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);

                    totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);
                    my_thread_no = 1; //MY_OPENMP_THREADS_NO;
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
                    if (Ideciphertext[index][20] == '2')
                    { // sent from isend
                        if (totaldata < SIXTY_4K)
                            my_thread_no = 1;
                        else if (totaldata < TWO_FIVE_6K)
                            my_thread_no = 2;
                        else
                            my_thread_no = 4;
                    }
                    else
                    { // from send
                        if (totaldata < THIRTY_2K)
                            my_thread_no = 1;
                        else if (totaldata >= THIRTY_2K && totaldata < ONE_TWO_8K)
                            my_thread_no = 2;
                        else if (totaldata >= ONE_TWO_8K && totaldata < FIVE_ONE_2K)
                            my_thread_no = 4;
                        else if (totaldata >= FIVE_ONE_2K && totaldata < ONE_M)
                            my_thread_no = 8;
                        else if (totaldata >= ONE_M)
                            my_thread_no = 12;
                    }
#endif

#if OMP_DYNAMIC_THREADS
                    if (my_thread_no > cyptmpi_series_thread)
                        my_thread_no = cyptmpi_series_thread;
#endif

                    unsigned int data = 0;
                    data = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);

                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

                    start = MSG_HEADER_SIZE;
                    pos = 0;
                    /* prepare for decryption */

                    int segment_no;
                    segment_no = 1;
#if 0
                    if(totaldata <=16){
                        segment_no = 1;
                        data = totaldata;
                    }
                    else
                    {
                        segment_no = (totaldata - 1) / data + 1;
                    }
                    
                    if((!OMP_DYNAMIC_THREADS) && (!OMP_DYNAMIC_THREADS_INNOVATION)){
                        if (totaldata <= MULTI_THREAD_START_SIZE)
                        {
                            data = totaldata;
                            segment_no = 1;
                        }
                    }
#endif

                    /* prepare for decryption */
#if OPENMP_THREADS
#pragma omp parallel for schedule(static) default(none) private(segment_counter, count, max_out_len, pos) shared(buf, Ideciphertext, segment_no, data, stdout, global_openmp_ctx, totaldata, index) num_threads(my_thread_no)
#endif
                    for (segment_counter = 0; segment_counter < segment_no; segment_counter++)
                    {

                        EVP_AEAD_CTX *local_ctx = global_openmp_ctx;

                        max_out_len = data + 16;
                        pos = MSG_HEADER_SIZE + segment_counter * (data + 12 + 16);
                        int enc_data = data;
                        // active_threads = omp_get_num_threads();
                        // printf("waitall active_threads=%d\n",active_threads);fflush(stdout);

                        if (segment_counter == segment_no - 1)
                        {
                            enc_data = totaldata - data * (segment_no - 1);
                            max_out_len = enc_data + 16;
                        }

                        if (!EVP_AEAD_CTX_open(local_ctx, buf + segment_counter * data,
                                               &count, enc_data,
                                               &Ideciphertext[index][pos], 12,
                                               &Ideciphertext[index][pos + 12], (unsigned long)(max_out_len),
                                               NULL, 0))
                        {
                            printf("[T = %d] OpenMP multi-thread Error in encryption: segment_counter=%d pos = %d\n", omp_get_thread_num(), segment_counter, pos);
                            fflush(stdout);
                        }
#if OMP_DEBUG
                        else
                        {
                            printf(COLOR_BLUE "  [T = %d] OpenMP multi-thread Decrypted data is %d in segment_counter = %d pos = %d" COLOR_RESET "\n",
                                   omp_get_thread_num(), count, segment_counter, pos);
                            fflush(stdout);
                        }
#endif
                    }

#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s2, NULL);
                    double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);

                    isSamenode = sameNode[source];
                    if (isSamenode < 0)
                    { /* inter node */
                        if (totaldata < FOUR_K)
                        {
                            inter_less_than_4K += t;
                        }
                        else if (totaldata < SIXTEEN_K)
                        {
                            inter_less_than_16K += t;
                        }
                        else if (totaldata < THIRTY_2K)
                        {
                            inter_less_than_32K += t;
                        }
                        else if (totaldata < SIXTY_4K)
                        {
                            inter_less_than_64K += t;
                        }
                        else if (totaldata < ONE_TWO_8K)
                        {
                            inter_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            inter_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            inter_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            inter_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            inter_more_than_1M += t;
                        }
                    }
                    else
                    {
                        if (totaldata < ONE_TWO_8K)
                        {
                            intra_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            intra_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            intra_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            intra_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            intra_more_than_1M += t;
                        }
                    }
#endif
                }

                else
                {
                    /*if (arr_sta == MPI_STATUS_IGNORE)
                        MPI_Wait_original(&arr_req[n], &sta);
                    else
                        mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);*/
                     mpi_errno = MPI_Wait_original(&arr_req[n], &sta);    
                }
            }
            else if (nonblock_req_handler[index].req_type == 1)
            {
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s1, NULL);
#endif
                for (i = 0; i < nonblock_req_handler[index].total_request; i++)
                {
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[i], &sta);
                }
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s2, NULL);
                double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
                totaldata = nonblock_req_handler[index].totaldata;
                //total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
                dest = nonblock_req_handler[index].dest;
                // isSamenode = 0;
                //if (dest > -1)
                isSamenode = sameNode[dest];

                if (isSamenode < 0)
                { /* inter node */
                    if (totaldata < FOUR_K)
                    {
                        inter_less_than_4K += t;
                    }
                    else if (totaldata < SIXTEEN_K)
                    {
                        inter_less_than_16K += t;
                    }
                    else if (totaldata < THIRTY_2K)
                    {
                        inter_less_than_32K += t;
                    }
                    else if (totaldata < SIXTY_4K)
                    {
                        inter_less_than_64K += t;
                    }
                    else if (totaldata < ONE_TWO_8K)
                    {
                        inter_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        inter_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        inter_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        inter_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        inter_more_than_1M += t;
                    }
                }
                else
                {
                    if (totaldata < ONE_TWO_8K)
                    {
                        intra_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        intra_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        intra_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        intra_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        intra_more_than_1M += t;
                    }
                }

#endif
                nonblock_req_handler[index].req_type = 0;
                //printf("waitall done for isend index=%d\n",index);fflush(stdout);
            }
        }
        else
        { /* Looks like, sending to own process, same node, unencruypted comm*/
#if ALL_COMM_PLUS_ENC_TIME
            int datasz;
            //struct timeval t_s1, t_s2;
            gettimeofday(&t_s1, NULL);
#endif
            /* if (arr_sta == MPI_STATUS_IGNORE)
                        MPI_Wait_original(&arr_req[n], &sta);
                    else
                        mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);*/
            mpi_errno =  MPI_Wait_original(&arr_req[n], &sta);            
#if ALL_COMM_PLUS_ENC_TIME
            gettimeofday(&t_s2, NULL);
            double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
            MPI_Get_count(&arr_sta[n], MPI_CHAR, &datasz);
            if (datasz < ONE_TWO_8K)
            {
                intra_less_than_128K += t;
            }
            else if (datasz < TWO_FIVE_6K)
            {
                intra_128K_256K += t;
            }
            else if (datasz < FIVE_ONE_2K)
            {
                intra_256K_512K += t;
            }
            else if (datasz < ONE_M)
            {
                intra_512K_1M += t;
            }
            else if (datasz >= ONE_M)
            {
                intra_more_than_1M += t;
            }
#endif
        }
    }
    return mpi_errno;
}

int MPI_SEC_Waitall_pipeline_openmp_largeSegment_3(int req_count, MPI_Request arr_req[],
                                                   MPI_Status arr_sta[])
{

    int mpi_errno = MPI_SUCCESS;

    int var, j;
    int sendtype_sz, tag, i, recv_len, len;
    unsigned long next, datasize, dest;
    char newkey[40];
    unsigned char nonce[50], V[50];
    unsigned long count, totaldata, index, decrypted_len;
    unsigned int segment_counter;
    int segments_no;
    MPI_Request request[300];
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;

    unsigned int nonce_counter;
    int m, th_data, my_thread_no, th_start, temp_chunk, n;
    int guided_chunk;

    EVP_AEAD_CTX *local_ctx = NULL;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);

    int temp_req_counter = 0;
    int cancel_req_no;
    int th_sum;
    int pos, dec_pos;
    int flag;
    unsigned char local_nonce[20];
    EVP_AEAD_CTX *segment_ctx[100]; // total threads will not excede 100
    struct timeval t_s1, t_s2;
    int isSamenode = 0;

    for (n = 0; n < req_count; n++)
    {
        index = arr_req[n];
        // printf("index =%d n=%d req_count=%d\n",index,n,req_count);fflush(stdout);

        if ((index >= 0 && index < ISEND_REQ))
        {
            if (nonblock_req_handler[index].req_type == 2)
            {
                source = nonblock_req_handler[index].source;
                dest = nonblock_req_handler[index].dest;
                tag = nonblock_req_handler[index].tag;
                nonblock_req_handler[index].req_type = 0;
                buf = nonblock_req_handler[index].buffer;
                int comm = MPI_COMM_WORLD;

                if (dest != source)
                {
#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s1, NULL);
#endif
                    /* Check for Probe */
                    if (Crypthandle_probe[source].no_tag > 0)
                    {
                        flag = 0;
                        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
                        {
                            if (Crypthandle_probe[source].tags[i] == tag)
                            {
                                memcpy(&Ideciphertext[index][0], &Crypthandle_probe[source].header[i][0], MSG_HEADER_SIZE);
                                flag = 1;
                                Crypthandle_probe[source].tags[i] = -1;
                                tag++;
                                //printf("    waitall: source = %d i = %d no_tag = %d\n", source, i, Crypthandle_probe[source].no_tag);
                                fflush(stdout);
                                break;
                            }
                        }

                        if (flag == 0)
                            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);
                    }
                    else
                    {
                        mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);
                    }
                    // mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                    ranksIrecvS[source] -= 1;
#endif
                    totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);

#if CRYPTMPI_ADAPTIVE_CHOPP
                    choping_sz = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
                    // printf(COLOR_YELLOW"wait recv: choping_sz =%d flag=%c totaldata=%lu "COLOR_RESET "\n",choping_sz, Ideciphertext[nonBlockCounter][20],totaldata); fflush(stdout);
#endif
                    //  printf(COLOR_BLUE"[rank=%d] wait for recv source=%d dest=%d totaldata=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest,totaldata);fflush(stdout);

                    if ((totaldata > PIPELINE_SIZE) && totaldata > LARGE_SEGMENT_SIZE)
                    {
                        segments_no = 1;
                        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
                    }
                    else
                    {
                        segments_no = 1;
                    }

                    //   printf("index=%d source=%d dest=%d totaldat=%d segments_no=%d\n",index,source,dest,totaldata,segments_no);fflush(stdout);

                    th_data = choping_sz; // - ENC_MSG_TAG_SIZE;
                    th_start = 0;
                    if (segments_no != nonblock_req_handler[index].segment_number)
                    {
                        cancel_req_no = nonblock_req_handler[index].segment_number - segments_no;
                        for (m = 0; m < cancel_req_no; m++)
                        {
                            int temp = nonblock_req_handler[index].total_request - 1 - m;
                            // printf("rank %d post cancel for temp %d\n",comm_ptr->rank, temp);
                            mpi_errno = MPI_Cancel(&nonblock_req_handler[index].request[temp]);
                            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[temp], &sta);
                            // newtag[source] -= 1;
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                            ranksIrecvS[source] -= 1;
#endif
                            // printf("   rank %d posted cancel done for temp %d\n",comm_ptr->rank, temp);
                        }
                        // nonblock_req_handler[index].segment_number = segments_no;
                        // nonblock_req_handler[index].total_request -= cancel_req_no;
                    }

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                    if (Ideciphertext[index][20] == '3' || Ideciphertext[index][20] == '4') // for '2' above equation will fine
                        segments_no = 1;
#endif

                    local_ctx = global_small_msg_ctx;

#if NONBLOCK_RECV_DEBUG
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    printf("[Rank = %d ]:Openmp-pipeline   index = %d  totaldata = %lu source = %d dest = %d\n",
                           rank, index, totaldata, source, dest);
                    fflush(stdout);
#endif

                    if ((totaldata <= PIPELINE_SIZE) && (Ideciphertext[index][20] != '3' && Ideciphertext[index][20] != '4'))
                    {
                        segment_counter = 0;
                        //my_thread_no = MY_MAX_NO_THREADS;
                        // m = my_thread_no - 1;
                        pos = MSG_HEADER_SIZE + NONCE_HEADER;

                        mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                        ranksIrecvS[source] -= 1;
#endif

                        if (!EVP_AEAD_CTX_open(local_ctx, buf,
                                               &decrypted_len, totaldata,
                                               &Ideciphertext[index][4], 12,
                                               &Ideciphertext[index][pos], totaldata + 16,
                                               NULL, 0))
                        {
                            printf("Openmp-pipeline: wait Decryption error: dec_pos=%d nonce_counter=%d pos=%d\n", dec_pos, nonce_counter, pos);
                            fflush(stdout);
                        }
#if NONBLOCK_RECV_DEBUG
                        else
                        {
                            printf("[Rank = %d ]:Openmp-pipeline: wait Decrypted data is %d in process nonce_counter %d\n", rank, decrypted_len, nonce_counter);
                            fflush(stdout);
                        }
#endif

                        if ((totaldata > SUBKEY_GEN_START))
                            EVP_AEAD_CTX_free(local_ctx);
                    }
                    else
                    {

                        count = 0;

                        /* prepare for decryption */
                        my_thread_no = MY_OPENMP_THREADS_NO;

#if OMP_DYNAMIC_THREADS_PIPELINE

                        if (Ideciphertext[index][20] == '1')
                        {

#if PSC_BRIDGE_TUNE
                            if (totaldata < SIXTY_4K)
                                my_thread_no = 1;
                            else if (totaldata < ONE_TWO_8K)
                                my_thread_no = 4;
                            else if (totaldata < TWO_FIVE_6K)
                                my_thread_no = 4;
                            else if (totaldata < FIVE_ONE_2K)
                                my_thread_no = 8;
                            else if (totaldata >= FIVE_ONE_2K)
                                my_thread_no = 16;
#else
                            if (totaldata < SIXTY_4K)
                                my_thread_no = 1;
                            else if (totaldata < ONE_TWO_8K)
                                my_thread_no = 2;
                            else if (totaldata < FIVE_ONE_2K)
                                my_thread_no = 4;
                            else if (totaldata >= FIVE_ONE_2K)
                                my_thread_no = 8;
#endif
                        }
                        else
                        { // '2' or '3'

#if PSC_BRIDGE_TUNE
                            if (totaldata < SIXTY_4K)
                                my_thread_no = 1;
                            else if (totaldata < ONE_TWO_8K)
                                my_thread_no = 4;
                            else if (totaldata < TWO_FIVE_6K)
                                my_thread_no = 4;
                            else if (totaldata < FIVE_ONE_2K)
                                my_thread_no = 8;
                            else if (totaldata >= FIVE_ONE_2K)
                                my_thread_no = 16;
#else
                            if (totaldata < SIXTY_4K)
                                my_thread_no = 1;
                            else if (totaldata < ONE_TWO_8K)
                                my_thread_no = 2;
                            else if (totaldata < FIVE_ONE_2K)
                                my_thread_no = 4;
                            else if (totaldata >= FIVE_ONE_2K)
                                my_thread_no = 8;
#endif
                        }
#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                        if (totaldata < SIXTY_4K)
                            my_thread_no = 1;
                        else if (totaldata < TWO_FIVE_6K)
                            my_thread_no = 2;
                        else
                            my_thread_no = 4;
#endif
#if OMP_DYNAMIC_THREADS_PIPELINE

                        if (my_thread_no > cyptmpi_series_thread)
                            my_thread_no = cyptmpi_series_thread;
#endif

/* Switch Between CryptMPI-R and CryptMPI-C */
#if ENABLE_CRYPTMPI_RC_SWITCH
                        if (omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
                            my_thread_no = 1;
#endif

#if CRYPTMPI_SINGLE_THREAD_MODE
                        my_thread_no = 1;
#endif

                        if ((totaldata > SUBKEY_GEN_START))
                        {
                            memcpy(V, &Ideciphertext[index][4], 16);
                            if (symmetric_key_size == 32)
                            {
                                for (i = 0; i < 16; i++)
                                    V[i + 16] = (~(int)V[i]);
                            }

                            /* Derive the new key k` */
                            if (1 != EVP_EncryptUpdate(ctx_enc, newkey, &len, V, symmetric_key_size))
                                printf("1");

                            if (symmetric_key_size == 16)
                            {
                                for (i = 0; i < my_thread_no; i++)
                                    segment_ctx[i] = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                                                      newkey,
                                                                      16, 0);
                            }
                            else if (symmetric_key_size == 32)
                            {
                                for (i = 0; i < my_thread_no; i++)
                                    segment_ctx[i] = EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(),
                                                                      newkey,
                                                                      32, 0);
                            }
                        }
                        th_data = choping_sz; // - ENC_MSG_TAG_SIZE;
                        //  printf("th_data = %d and recv_len = %d segments_no=%d\n",th_data, recv_len,segments_no);fflush(stdout);
                        th_start = NONCE_HEADER + MSG_HEADER_SIZE;

#if NONBLOCK_RECV_DEBUG
                        printf("    Oenmp-pipeline my_thread_no=%d th_data=%d choping_sz=%d segments_no=%d th_start=%d\n", my_thread_no, th_data, choping_sz, segments_no, th_start);
                        fflush(stdout);
#endif
                        int active_thread_num = my_thread_no;
                        int ii;

                        //  if(OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION)
                        //    omp_set_dynamic(my_thread_no);
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE
                        if (Ideciphertext[index][20] == '3' || Ideciphertext[index][20] == '4')
                        {
                            th_data = choping_sz;

                            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                            ranksIrecvS[source] -= 1;
#endif
                            // printf("wait to get data done\n");fflush(stdout);
                            pos = 0;
                            int new_segments;
                            new_segments = (totaldata - 1) / th_data + 1;
                            //   printf("   waitall totaldata =%d choping_sz=%d new_segments=%d\n",totaldata,choping_sz,new_segments);fflush(stdout);

//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, Ideciphertext, th_data, segment_ctx,new_segments, local_ctx, stdout, index,totaldata) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, Ideciphertext, th_data, segment_ctx, new_segments, local_ctx, stdout, index, totaldata) num_threads(my_thread_no)
                            {
#pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter)

                                for (segment_counter = 0; segment_counter < new_segments; segment_counter++)
                                {

                                    pos = MSG_HEADER_SIZE + segment_counter * (th_data + NONCE_HEADER + ENC_MSG_TAG_SIZE) + NONCE_HEADER;
                                    int enc_data = th_data;
                                    // active_threads = omp_get_num_threads();
                                    // printf("waitall active_threads=%d\n",active_threads);fflush(stdout);

                                    if (segment_counter == new_segments - 1)
                                    {
                                        enc_data = totaldata - th_data * (new_segments - 1);
                                    }

                                    local_nonce[0] = '0';
                                    local_nonce[1] = '0';
                                    local_nonce[2] = '0';
                                    local_nonce[3] = '0';
                                    local_nonce[4] = '0';
                                    local_nonce[5] = '0';
                                    local_nonce[6] = '0';
                                    local_nonce[7] = Ideciphertext[index][pos - 5];
                                    local_nonce[8] = Ideciphertext[index][pos - 4];
                                    local_nonce[9] = Ideciphertext[index][pos - 3];
                                    local_nonce[10] = Ideciphertext[index][pos - 2];
                                    local_nonce[11] = Ideciphertext[index][pos - 1];

                                    nonce_counter = ((unsigned char)Ideciphertext[index][pos - 1] << 0) | ((unsigned char)Ideciphertext[index][pos - 2] << 8) | ((unsigned char)Ideciphertext[index][pos - 3] << 16) | ((unsigned char)Ideciphertext[index][pos - 4] << 24);

                                    dec_pos = nonce_counter * th_data;
                                    // segment_ctx[omp_get_thread_num()] = local_ctx;

                                    // printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d %u %u %u %u => %u pos=%d\n",
                                    // omp_get_thread_num(),segment_counter,m,ii,th_data,nonce_counter,dec_pos,large_recv_buffer[pos-1],large_recv_buffer[pos-2],large_recv_buffer[pos-3],large_recv_buffer[pos-4],0,pos );fflush(stdout);
                                    if (!EVP_AEAD_CTX_open(segment_ctx[omp_get_thread_num()], buf + dec_pos,
                                                           &decrypted_len, enc_data,
                                                           local_nonce, 12,
                                                           &Ideciphertext[index][pos], enc_data + 16,
                                                           NULL, 0))
                                    {
                                        printf(COLOR_RED "OpenMP Pipeline: Decryption error waitall:  thread %d nonce_counter %d segment_counter %d new_segments=%d th_data=%d" COLOR_RESET "\n",
                                               omp_get_thread_num(), nonce_counter, segment_counter, new_segments, th_data);
                                        fflush(stdout);
                                    }
#if OMP_DEBUG_DYNAMIC_THREAD_PIPELINE
                                    else
                                    {
                                        printf(COLOR_BLUE "    OpenMP Dynamic Thread Pipeline: Decrypted data is %d in thread = %d nonce_counter=%d" COLOR_RESET "\n",
                                               decrypted_len, omp_get_thread_num(), nonce_counter);
                                        fflush(stdout);
                                    }
#endif
                                } // end of parallel loop
                            }     // end of parallel region
                        }
                        else
                        {
#endif

                            int prsd_segment = 0;
                            int inner_totaldata = 0;
                            int enc_loc = 0;
                            int recv_loc = 0;
                            for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
                            {

                                mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[segment_counter + 1], &sta);
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                                ranksIrecvS[source] -= 1;
#endif
                                th_data = choping_sz;

                                if (segment_counter == segments_no - 1)
                                {
                                    inner_totaldata = totaldata - (PIPELINE_SIZE * (segments_no - 1));
                                    ii = (inner_totaldata - 1) / th_data + 1;
                                }
                                else
                                {
                                    inner_totaldata = PIPELINE_SIZE;
                                    ii = (PIPELINE_SIZE - 1) / th_data + 1;
                                }

                                // printf("my_thread_no=%d ii=%d\n",my_thread_no,ii); fflush(stdout);
                                if (my_thread_no > ii)
                                    my_thread_no = ii;

                                    /* Decrypt received data */
//#if OPENMP_DYNAMIC_SCH
//#pragma omp parallel for default(none) schedule(dynamic) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter,index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
//#elif OPENMP_STATIC_SCH
//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter,index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, index, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
                                {
#pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos)

                                    //#elif OPENMP_GUIDED_SCH
                                    //#pragma omp parallel for default(none) schedule(guided) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter,index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
                                    //#else
                                    //#pragma omp parallel for default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter,index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
                                    //#endif
                                    for (m = 0; m < ii; m++)
                                    {
                                        pos = MSG_HEADER_SIZE + enc_loc + m * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;

                                        local_nonce[0] = '0';
                                        local_nonce[1] = '0';
                                        local_nonce[2] = '0';
                                        local_nonce[3] = '0';
                                        local_nonce[4] = '0';
                                        local_nonce[5] = '0';
                                        local_nonce[6] = '0';
                                        local_nonce[7] = Ideciphertext[index][pos - 5];
                                        local_nonce[8] = Ideciphertext[index][pos - 4];
                                        local_nonce[9] = Ideciphertext[index][pos - 3];
                                        local_nonce[10] = Ideciphertext[index][pos - 2];
                                        local_nonce[11] = Ideciphertext[index][pos - 1];

                                        nonce_counter = ((unsigned char)Ideciphertext[index][pos - 1] << 0) | ((unsigned char)Ideciphertext[index][pos - 2] << 8) | ((unsigned char)Ideciphertext[index][pos - 3] << 16) | ((unsigned char)Ideciphertext[index][pos - 4] << 24);

                                        //dec_pos = nonce_counter * th_data;
                                        dec_pos = recv_loc + m * th_data;
                                        // segment_ctx[omp_get_thread_num()] = local_ctx;

                                        int enc_data = th_data;
                                        if (m == ii - 1)
                                        {
                                            enc_data = inner_totaldata - th_data * (ii - 1);
                                        }

                                        /* printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d pos=%d enc_loc=%d inner_totaldata=%d\n",
                                        omp_get_thread_num(), segment_counter, m, ii, th_data, nonce_counter, dec_pos, pos, enc_loc,inner_totaldata,enc_data);
                                 fflush(stdout);*/

                                        if (!EVP_AEAD_CTX_open(segment_ctx[omp_get_thread_num()], buf + dec_pos,
                                                               &decrypted_len, enc_data,
                                                               local_nonce, 12,
                                                               &Ideciphertext[index][pos], enc_data + 16,
                                                               NULL, 0))
                                        {
                                            printf(COLOR_RED "OpenMP Pipeline: Decryption error waitall:  thread %d nonce_counter %d segment_counter %d enc_data=%d th_data=%d inner_totaldata=%d ii=%d recv_loc=%d pos=%d " COLOR_RESET "\n",
                                                   omp_get_thread_num(), nonce_counter, segment_counter, enc_data, th_data, inner_totaldata, ii, recv_loc, pos);
                                            fflush(stdout);
                                        }
#if OMP_DEBUG
                                        else
                                        {
                                            printf(COLOR_BLUE "**OpenMP Pipeline: Decrypted data is %d in thread = %d" COLOR_RESET "\n", decrypted_len, omp_get_thread_num());
                                            fflush(stdout);
                                        }
#endif
                                    }
                                } // end of parallel region
                                // enc_loc += inner_totaldata + (ENC_MSG_TAG_SIZE + NONCE_HEADER) * ii;
                                enc_loc += inner_totaldata + (ENC_MSG_TAG_SIZE + NONCE_HEADER) * MAX_OMP_THREADS_LIMIT;
                                recv_loc += inner_totaldata;
                                //  prsd_segment += ii;
                            }

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE
                        } // end of else
#endif
                        if ((totaldata > SUBKEY_GEN_START))
                        {
                            for (i = 0; i < my_thread_no; i++)
                                EVP_AEAD_CTX_free(segment_ctx[i]);
                        }
                    }

#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s2, NULL);
                    double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);

                    /* if(totaldata > ONE_M)
                        total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
                    else
                        total_comm_plus_enc_time_small_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);*/

                    if (source > -1)
                        isSamenode = sameNode[source];
                    if (isSamenode < 0)
                    { /* inter node */
                        if (totaldata < FOUR_K)
                        {
                            inter_less_than_4K += t;
                        }
                        else if (totaldata < SIXTEEN_K)
                        {
                            inter_less_than_16K += t;
                        }
                        else if (totaldata < THIRTY_2K)
                        {
                            inter_less_than_32K += t;
                        }
                        else if (totaldata < SIXTY_4K)
                        {
                            inter_less_than_64K += t;
                        }
                        else if (totaldata < ONE_TWO_8K)
                        {
                            inter_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            inter_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            inter_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            inter_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            inter_more_than_1M += t;
                        }
                    }
                    else
                    {
                        /* printf("[rank %d]: source %d sending to dest %d datasize %d isSamenode = %d\n",
                        comm_ptr->rank, source, dest, totaldata, isSamenode);fflush(stdout);*/
                        if (totaldata < ONE_TWO_8K)
                        {
                            intra_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            intra_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            intra_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            intra_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            intra_more_than_1M += t;
                        }
                    }
#endif

                } // if source and dest are same

                else
                {

                    //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
                     mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
                }
            }
            else if (nonblock_req_handler[index].req_type == 1)
            {

#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s1, NULL);
#endif
                for (i = 0; i < nonblock_req_handler[index].total_request; i++)
                {
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[i], &sta);
                    pendingIsendRequestCount[nonblock_req_handler[index].dest] -= 1;
                }
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s2, NULL);
                double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
                totaldata = nonblock_req_handler[index].totaldata;

                dest = nonblock_req_handler[index].dest;
                source = nonblock_req_handler[index].source;

                if (dest > -1)
                    isSamenode = sameNode[dest];

                if (isSamenode < 0)
                { /* inter node */
                    if (totaldata < FOUR_K)
                    {
                        inter_less_than_4K += t;
                    }
                    else if (totaldata < SIXTEEN_K)
                    {
                        inter_less_than_16K += t;
                    }
                    else if (totaldata < THIRTY_2K)
                    {
                        inter_less_than_32K += t;
                    }
                    else if (totaldata < SIXTY_4K)
                    {
                        inter_less_than_64K += t;
                    }
                    else if (totaldata < ONE_TWO_8K)
                    {
                        inter_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        inter_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        inter_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        inter_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        inter_more_than_1M += t;
                    }
                }
                else
                {
                    /* printf("[rank %d] waitall in sending: source %d sending to dest %d datasize %d isSamenode = %d sameNode[%d] = %d\n",
                        comm_ptr->rank, source, dest, totaldata, isSamenode, dest,sameNode[dest]);fflush(stdout);*/

                    if (totaldata < ONE_TWO_8K)
                    {
                        intra_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        intra_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        intra_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        intra_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        intra_more_than_1M += t;
                    }
                }

#endif
                nonblock_req_handler[index].req_type = 0;
            }
        }
        else
        { /* Looks like, sending to own process, same node, unencrypted comm */
#if ALL_COMM_PLUS_ENC_TIME
            int datasz;
            //struct timeval t_s1, t_s2;
            gettimeofday(&t_s1, NULL);
#endif
            //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
             mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
#if ALL_COMM_PLUS_ENC_TIME
            gettimeofday(&t_s2, NULL);
            double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
            MPI_Get_count(&arr_sta[n], MPI_CHAR, &datasz);
            if (datasz < ONE_TWO_8K)
            {
                intra_less_than_128K += t;
            }
            else if (datasz < TWO_FIVE_6K)
            {
                intra_128K_256K += t;
            }
            else if (datasz < FIVE_ONE_2K)
            {
                intra_256K_512K += t;
            }
            else if (datasz < ONE_M)
            {
                intra_512K_1M += t;
            }
            else if (datasz >= ONE_M)
            {
                intra_more_than_1M += t;
            }
#endif
        }
    }
    return mpi_errno;
}

int MPI_SEC_BaseCounter_Pipeline_Waitall(int req_count, MPI_Request arr_req[],
                                         MPI_Status arr_sta[])
{
    int mpi_errno = MPI_SUCCESS;
    int sendtype_sz, tag, i, len;
    int datasize, dest;
    unsigned char iv_buffer[20];
    int count, totaldata, index, decrypted_len;
    unsigned int segment_counter;
    int segments_no;
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;

    int m, th_data, my_thread_no, n;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
    int thisrank = comm_ptr->rank;

    int temp_req_counter = 0;
    int cancel_req_no;
    int next;
    struct timeval t_s1, t_s2;
    int inner_totaldata, ii;
    int isSamenode = 0;

    for (n = 0; n < req_count; n++)
    {
        index = arr_req[n];

        if ((index >= 0 && index < ISEND_REQ))
        {
            if (nonblock_req_handler[index].req_type == 2)
            {

                source = nonblock_req_handler[index].source;
                dest = nonblock_req_handler[index].dest;
                tag = nonblock_req_handler[index].tag;
                nonblock_req_handler[index].req_type = 0;
                buf = nonblock_req_handler[index].buffer;
                int comm = MPI_COMM_WORLD;
                if (dest != source)
                {
#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s1, NULL);
#endif
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);

                    unsigned int temp_totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);
                    totaldata = temp_totaldata;
                    my_thread_no = 1; //MY_OPENMP_THREADS_NO;

                    choping_sz = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
                    long unsigned int recv_counter = (((unsigned char)Ideciphertext[index][8] << 0) | ((unsigned char)Ideciphertext[index][7] << 8) | ((unsigned char)Ideciphertext[index][6] << 16) | ((unsigned char)Ideciphertext[index][5] << 24));
                    //mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

                    if (BASE_COUNTER_LIBRARY_NONCE)
                    {
                        memcpy(iv_buffer, &Ideciphertext[index][4], 16);
                        recv_counter = 0;
                    }
                    else
                    {
                        memcpy(iv_buffer, &Recv_common_IV[source * 16], 16);
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

                    /* cancel the extra pipeline request */
                    if (segments_no != nonblock_req_handler[index].segment_number)
                    {
                        cancel_req_no = nonblock_req_handler[index].segment_number - segments_no;
                        for (m = 0; m < cancel_req_no; m++)
                        {
                            int temp = nonblock_req_handler[index].total_request - 1 - m;
                            // printf("rank %d post cancel for temp %d\n",comm_ptr->rank, temp);
                            mpi_errno = MPI_Cancel(&nonblock_req_handler[index].request[temp]);
                            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[temp], &sta);
                            // printf("   rank %d posted cancel done for temp %d\n",comm_ptr->rank, temp);
                        }
                    }

                    next = 0;
                    for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
                    {

                        mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[segment_counter + 1], &sta);
                        th_data = choping_sz;
                        if (segment_counter == segments_no - 1)
                        {
                            inner_totaldata = totaldata - (PIPELINE_SIZE * (segments_no - 1));
                            ii = (inner_totaldata - 1) / th_data + 1;
                        }
                        else
                        {
                            inner_totaldata = PIPELINE_SIZE;
                            ii = (PIPELINE_SIZE - 1) / th_data + 1;
                        }

                        IV_Count(iv_buffer, recv_counter);

                        if (1 != EVP_DecryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer))
                            printf(COLOR_RED "___Error in initialization Base counter waitall___" COLOR_RESET "\n");

                        if (1 != EVP_DecryptUpdate(global_counter_ctx, (buf + next), &decrypted_len, &Ideciphertext[index][COUNTER_HEADER_SIZE + next], inner_totaldata))
                            printf(COLOR_RED "___Error in decryption___ Base counter waitall " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
                        else
                        {
                            printf(COLOR_BLUE "BaseCounter waitall [rank %d]: Decrypted data is %d  next %d segment_counter %d segments_no %d" COLOR_RESET "\n",
                                   thisrank, decrypted_len, next, segment_counter, segments_no);
                        }
#endif
                        next += inner_totaldata;
                        recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
                    }

#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s2, NULL);
                    double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
                    //int isSamenode = 0;
                    //if (source > -1)
                    isSamenode = sameNode[source];
                    if (isSamenode < 0)
                    { /* inter node */
                        if (totaldata < FOUR_K)
                        {
                            inter_less_than_4K += t;
                        }
                        else if (totaldata < SIXTEEN_K)
                        {
                            inter_less_than_16K += t;
                        }
                        else if (totaldata < THIRTY_2K)
                        {
                            inter_less_than_32K += t;
                        }
                        else if (totaldata < SIXTY_4K)
                        {
                            inter_less_than_64K += t;
                        }
                        else if (totaldata < ONE_TWO_8K)
                        {
                            inter_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            inter_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            inter_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            inter_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            inter_more_than_1M += t;
                        }
                    }
                    else
                    {
                        if (totaldata < ONE_TWO_8K)
                        {
                            intra_less_than_128K += t;
                        }
                        else if (totaldata < TWO_FIVE_6K)
                        {
                            intra_128K_256K += t;
                        }
                        else if (totaldata < FIVE_ONE_2K)
                        {
                            intra_256K_512K += t;
                        }
                        else if (totaldata < ONE_M)
                        {
                            intra_512K_1M += t;
                        }
                        else if (totaldata >= ONE_M)
                        {
                            intra_more_than_1M += t;
                        }
                    }
#endif
                }

                else
                {
                    //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
                     mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
                }
            }
            else if (nonblock_req_handler[index].req_type == 1)
            {
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s1, NULL);
#endif
                for (i = 0; i < nonblock_req_handler[index].total_request; i++)
                {
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[i], &sta);
                }
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s2, NULL);
                double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
                totaldata = nonblock_req_handler[index].totaldata;
                //total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
                dest = nonblock_req_handler[index].dest;

                //int isSamenode = 0;
                //if (dest > -1)
                isSamenode = sameNode[dest];
                if (isSamenode < 0)
                { /* inter node */
                    if (totaldata < FOUR_K)
                    {
                        inter_less_than_4K += t;
                    }
                    else if (totaldata < SIXTEEN_K)
                    {
                        inter_less_than_16K += t;
                    }
                    else if (totaldata < THIRTY_2K)
                    {
                        inter_less_than_32K += t;
                    }
                    else if (totaldata < SIXTY_4K)
                    {
                        inter_less_than_64K += t;
                    }
                    else if (totaldata < ONE_TWO_8K)
                    {
                        inter_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        inter_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        inter_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        inter_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        inter_more_than_1M += t;
                    }
                }
                else
                {
                    if (totaldata < ONE_TWO_8K)
                    {
                        intra_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        intra_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        intra_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        intra_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        intra_more_than_1M += t;
                    }
                }

#endif
                nonblock_req_handler[index].req_type = 0;
                //printf("waitall done for isend index=%d\n",index);fflush(stdout);
            }
        }
        else
        { /* Looks like, sending to own process, same node, unencruypted comm*/
#if ALL_COMM_PLUS_ENC_TIME
            int datasz;
            //struct timeval t_s1, t_s2;
            gettimeofday(&t_s1, NULL);
#endif

            //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
             mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
            //printf("    [rank=%d] else done index =%d n=%d \n",comm_ptr->rank,index,n);fflush(stdout);
#if ALL_COMM_PLUS_ENC_TIME
            gettimeofday(&t_s2, NULL);
            double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
            MPI_Get_count(&arr_sta[n], MPI_CHAR, &datasz);
            if (datasz < ONE_TWO_8K)
            {
                intra_less_than_128K += t;
            }
            else if (datasz < TWO_FIVE_6K)
            {
                intra_128K_256K += t;
            }
            else if (datasz < FIVE_ONE_2K)
            {
                intra_256K_512K += t;
            }
            else if (datasz < ONE_M)
            {
                intra_512K_1M += t;
            }
            else if (datasz >= ONE_M)
            {
                intra_more_than_1M += t;
            }
#endif
        }
    }

    return mpi_errno;
}

int MPI_SEC_PreComputeCounter_Pipeline_Waitall_v4(int req_count, MPI_Request arr_req[],
                                   MPI_Status arr_sta[])
{
    int mpi_errno = MPI_SUCCESS;
    int sendtype_sz, tag, i, len;
    int datasize, dest;
    unsigned char iv_buffer[20];
    int count, totaldata, index, decrypted_len;
    unsigned int segment_counter;
    int segments_no;
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;

    int m, th_data, n;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
    int thisrank = comm_ptr->rank;

    int temp_req_counter = 0;
    int cancel_req_no;
    int next;
    struct timeval t_s1, t_s2;
    int inner_totaldata, ii;
    int recv_pos, mask_pos, dec_pos, flag;
    int common_counter_gen_sz;
    int temp_thread, t_counter_data, my_thread_no, dec_loc,pos,enc_data;
    long unsigned int t_counter;
    int decryption_mask;

    for (n = 0; n < req_count; n++)
    {
        index = arr_req[n];

        if ((index >= 0 && index < ISEND_REQ))
        {
            if (nonblock_req_handler[index].req_type == 2)
            {

                source = nonblock_req_handler[index].source;
                dest = nonblock_req_handler[index].dest;
                tag = nonblock_req_handler[index].tag;
                nonblock_req_handler[index].req_type = 0;
                buf = nonblock_req_handler[index].buffer;
                int comm = MPI_COMM_WORLD;
                if (dest != source)
                {
#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s1, NULL);
#endif
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);
                    ranksIrecvS[source] -=1;
                    unsigned int temp_totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);
                    totaldata = temp_totaldata;

                    choping_sz = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
                    unsigned long   recv_counter = (((unsigned char)Ideciphertext[index][8] << 0) | ((unsigned char)Ideciphertext[index][7] << 8) | ((unsigned char)Ideciphertext[index][6] << 16) | ((unsigned char)Ideciphertext[index][5] << 24));
                    unsigned long temp_recv_counter = recv_counter;
                    char preCTRflag = Ideciphertext[index][4];
                    //mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

                   // printf("waitall: totaldata %d recv_counter %lu\n",totaldata, recv_counter);fflush(stdout);                   

                    if ((totaldata > PIPELINE_SIZE) && (totaldata > LARGE_SEGMENT_SIZE))
                    {
                        segments_no = 1;
                        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
                    }
                    else
                    {
                        segments_no = 1;
                    }


                    /* cancel the extra pipeline request */
                     if (segments_no != nonblock_req_handler[index].segment_number)
                    {
                        cancel_req_no = nonblock_req_handler[index].segment_number - segments_no;
                        for (m = 0; m < cancel_req_no; m++)
                        {
                            int temp = nonblock_req_handler[index].total_request - 1 - m;
                            // printf("rank %d post cancel for temp %d\n",comm_ptr->rank, temp);
                            mpi_errno = MPI_Cancel(&nonblock_req_handler[index].request[temp]);
                            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[temp], &sta);
                            ranksIrecvS[source] -=1;
                            // printf("   rank %d posted cancel done for temp %d\n",comm_ptr->rank, temp);
                        }
                    }

                    if (totaldata < PRE_COM_DATA_RANGE)
                    {
                        MPI_Test(&nonblock_req_handler[index].request[1], &flag, &sta);
                       

                        if (!flag)
                        {
                            decryption_mask = 0;

                            if (totaldata > 1024)
                            {
                                common_counter_gen_sz = 512; //1024;

                                    my_thread_no = 1;

                                if (my_thread_no == 1)
                                {
                                    while (!flag)
                                    {
                                        //memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                                        //IV_Count(iv_buffer, recv_counter);
                                        if(preCTRflag == '0')
                                            IV_Count_out(iv_buffer, recv_counter, &Recv_common_IV[source * 32]);
                                        else
                                             IV_Count_out(iv_buffer, recv_counter, &Recv_common_IV[source * 32+16]);  
                                        EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                                        EVP_EncryptUpdate(global_counter_ctx, &dec_common_buffer[decryption_mask], &len, zeros, common_counter_gen_sz);
                                        decryption_mask += common_counter_gen_sz;
                                        recv_counter += (unsigned long)((common_counter_gen_sz - 1) / 16 + 1);
                                        /*printf("wait: rank %d source %d dest %d totaldata %d decryption_mask %d\n",
                                    comm_ptr->rank, source, dest,totaldata,decryption_mask); fflush(stdout);*/

                                        if (decryption_mask >= totaldata)
                                            break;

                                         MPI_Test(&nonblock_req_handler[index].request[1], &flag, &sta);
                                    }
                                }

                                else
                                {

                                    while (!flag)
                                    {
#pragma omp parallel default(none) private(m, iv_buffer, len) shared(my_thread_no, dec_common_buffer, local_counter_ctx, stdout, totaldata, source, recv_counter, decryption_mask, Recv_common_IV, common_counter_gen_sz, zeros,preCTRflag) num_threads(my_thread_no)
                                        {
#pragma omp for schedule(static) nowait private(m, iv_buffer, len)
                                            for (m = 0; m < my_thread_no; m++)
                                            {
                                                unsigned long int t_recv_counter;
                                                if (m > 0)
                                                {
                                                    t_recv_counter = recv_counter + ((unsigned long)((common_counter_gen_sz * m - 1) / 16 + 1));
                                                }
                                                else
                                                {
                                                    t_recv_counter = recv_counter;
                                                }
                                                 if(preCTRflag == '0')    
                                                    IV_Count_out(iv_buffer, t_recv_counter, &Recv_common_IV[source * 32]);
                                                else
                                                    IV_Count_out(iv_buffer, t_recv_counter, &Recv_common_IV[source * 32+16]);    
                                                EVP_EncryptInit_ex(local_counter_ctx[m], NULL, NULL, NULL, iv_buffer);
                                                EVP_EncryptUpdate(local_counter_ctx[m], &dec_common_buffer[decryption_mask + m * common_counter_gen_sz], &len, zeros, common_counter_gen_sz);

                                                /*printf("wait: rank %d source %d dest %d totaldata %d decryption_mask %d\n",
                                    comm_ptr->rank, source, dest,totaldata,decryption_mask); fflush(stdout);*/
                                            }
                                        }
                                        decryption_mask += (common_counter_gen_sz * my_thread_no);
                                        recv_counter += (unsigned long)((common_counter_gen_sz * my_thread_no - 1) / 16 + 1);
                                        if (decryption_mask >= totaldata)
                                            break;

                                         MPI_Test(&nonblock_req_handler[index].request[1], &flag, &sta);
                                    }
                                }
                               

                                if (!flag)
                                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);
                            }
                            else
                            {
                                if(preCTRflag == '0')  
                                    memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                                else
                                    memcpy(iv_buffer, &Recv_common_IV[source * 32+16], 16);
                                IV_Count(iv_buffer, recv_counter);
                                EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                                EVP_EncryptUpdate(global_counter_ctx, dec_common_buffer, &len, zeros, totaldata);
                                decryption_mask = totaldata;
                                mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);
                            }

                            ranksIrecvS[source] -= 1;
                            mask_pos = 0;
                            recv_pos = COUNTER_HEADER_SIZE;
                            dec_pos = 0;
                            // printf("waitall dec: rank %d source %d dest %d totaldata %d decryption_mask %d\n",
                            //        comm_ptr->rank, source, dest,totaldata,decryption_mask); fflush(stdout);
                            if(decryption_mask < totaldata)
                            {
                                decryption_mask = 0;
                                recv_counter = temp_recv_counter;    
                            }  
                            decryption_common_counter_ivflag(buf, decryption_mask, totaldata, recv_pos, dec_pos, mask_pos,
                                                      recv_counter, source, &Ideciphertext[index][0],preCTRflag);
                            // printf("1. n %d decryption done data size %d\n",n, totaldata);
                            /* Debug */
                            /*
                            double percentage = (((double)decryption_mask)/totaldata)*100;
                            if(percentage >= 100)
                                {printf(COLOR_GREEN"[waitall dec: rank: %d]percentage %lf decryption_mask %d totaldata %d"COLOR_RESET"\n",comm_ptr->rank,percentage,decryption_mask,totaldata);fflush(stdout);}
                            else if(percentage >= 75)
                                {printf(COLOR_BLUE"[waitall dec: rank: %d]percentage %lf decryption_mask %d totaldata %d"COLOR_RESET"\n",comm_ptr->rank,percentage, decryption_mask,totaldata);fflush(stdout);}
                            else if(percentage >= 50)
                                {printf(COLOR_CYAN"[waitall dec: rank: %d]percentage %lf decryption_mask %d totaldata %d"COLOR_RESET"\n",comm_ptr->rank,percentage, decryption_mask,totaldata);fflush(stdout);} 
                            else if(percentage >= 25)
                                {printf(COLOR_MAGENTA"[waitall dec: rank: %d]percentage %lf decryption_mask %d totaldata %d"COLOR_RESET"\n",comm_ptr->rank,percentage, decryption_mask,totaldata);fflush(stdout);}
                            else
                                {printf(COLOR_YELLOW"[waitall dec: rank: %d]percentage %lf decryption_mask %d totaldata %d"COLOR_RESET"\n",comm_ptr->rank,percentage, decryption_mask,totaldata);fflush(stdout);}                                              
                            */
                            
                        }
                        else
                        {
                            ranksIrecvS[source] -= 1;
                            if(preCTRflag == '0')  
                                memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                            else
                                memcpy(iv_buffer, &Recv_common_IV[source * 32+16], 16);    
                            IV_Count(iv_buffer, recv_counter);
                            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                            EVP_EncryptUpdate(global_counter_ctx, buf, &len, &Ideciphertext[index][COUNTER_HEADER_SIZE], totaldata);
                           // printf("2. decryption done data size %d\n", totaldata);
                        }
                        
                    }
                    else
                    {
                        /* because of too many request, no pipeline */
                        if (Ideciphertext[index][20] == '3' || Ideciphertext[index][20] == '4')
                            segments_no = 1;

                        /* prepare for decryption */
                        my_thread_no = MY_OPENMP_THREADS_NO;

#if PSC_BRIDGE_TUNE
                        if (totaldata < SIXTY_4K)
                            my_thread_no = 1;
                        else if (totaldata < ONE_TWO_8K)
                            my_thread_no = 4;
                        else if (totaldata < TWO_FIVE_6K)
                            my_thread_no = 4;
                        else if (totaldata < FIVE_ONE_2K)
                            my_thread_no = 8;
                        else if (totaldata >= FIVE_ONE_2K)
                            my_thread_no = 16;
#else
                         if (totaldata < SIXTY_4K)
                            my_thread_no = 1; 
                        else if (totaldata < TWO_FIVE_6K)
                            my_thread_no = 8;
                        else if (totaldata >= TWO_FIVE_6K)
                            my_thread_no = 12;  
#endif

#if DYNAMIC_PIPELINE
                        if (my_thread_no > cyptmpi_series_thread)
                            my_thread_no = cyptmpi_series_thread;
#endif

#if DYNAMIC_PIPELINE
                        if (totaldata > LARGE_SEGMENT_SIZE)
                        {
                            int temp_thread;
#if PSC_BRIDGE_TUNE
                            if (PIPELINE_SIZE < SIXTY_4K)
                                temp_thread = 1;
                            else if (PIPELINE_SIZE < ONE_TWO_8K)
                                temp_thread = 4;
                            else if (PIPELINE_SIZE < TWO_FIVE_6K)
                                temp_thread = 4;
                            else if (PIPELINE_SIZE < FIVE_ONE_2K)
                                temp_thread = 8;
                            else if (PIPELINE_SIZE >= FIVE_ONE_2K)
                                temp_thread = 16;
#else
                            if (PIPELINE_SIZE < THIRTY_2K)
                                temp_thread = 1;
                            else if (PIPELINE_SIZE < SIXTY_4K)
                                temp_thread = 4;
                            else if (PIPELINE_SIZE <= TWO_FIVE_6K)
                                temp_thread = 12;
                            else if (PIPELINE_SIZE > TWO_FIVE_6K)
                                temp_thread = 12; //16;  
#endif
                            if (temp_thread > cyptmpi_series_thread)
                                temp_thread = cyptmpi_series_thread;
                            my_thread_no = temp_thread;
                        }

#endif
                        recv_pos = COUNTER_HEADER_SIZE;
                        dec_loc = 0;
                        for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
                        {
                            /* encryption size for each thread */
                            th_data = choping_sz;

                             mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[segment_counter+ 1], &sta);                                    
                            ranksIrecvS[source] -=1;
                           // printf("waitall:ranksIrecvS[%d] %d  \n",source, ranksIrecvS[source]);fflush(stdout);


                            if (segment_counter == segments_no - 1)
                            {
                                inner_totaldata = totaldata - (PIPELINE_SIZE * (segments_no - 1));
                                ii = (inner_totaldata - 1) / th_data + 1;
                            }
                            else
                            {
                                inner_totaldata = PIPELINE_SIZE;
                                ii = (PIPELINE_SIZE - 1) / th_data + 1;
                            }

                            if (my_thread_no > ii)
                                my_thread_no = ii;
//#pragma omp parallel for default(none) schedule(static) private(m, pos, dec_pos, enc_data, t_counter, t_counter_data) shared(buf, Ideciphertext, th_data, local_counter_ctx, segments_no, stdout, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, recv_counter,index) num_threads(my_thread_no)
#pragma omp parallel default(none) private(m, pos, dec_pos, enc_data, t_counter, t_counter_data) shared(buf, Ideciphertext, th_data, local_counter_ctx, segments_no, stdout, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, recv_counter,index) num_threads(my_thread_no)
    {
                        #pragma omp for schedule(static) nowait private(m, pos, dec_pos, enc_data, t_counter, t_counter_data)
    
                            for (m = 0; m < ii; m++)
                            {
                                enc_data = th_data;
                                if (m == ii - 1)
                                {
                                    enc_data = inner_totaldata - th_data * (ii - 1);
                                }
                                pos = recv_pos + m * th_data;
                                dec_pos = dec_loc + m * th_data;

                                    t_counter_data = th_data * m; // - common_counter_gen_sz;
                                    if (t_counter_data < 1)
                                        t_counter = recv_counter;
                                    else
                                        t_counter = recv_counter + (unsigned long)((t_counter_data - 1) / 16 + 1);
                                
                                multithread_decryption_common_counter_large_msg(buf, enc_data, pos, dec_pos,  t_counter, &Ideciphertext[index][0],local_counter_ctx[m], source);
                               // multithread_decryption_common_counter(buf, t_mask, enc_data, pos, dec_pos, t_mask_pos, t_counter, large_recv_buffer, dec_common_buffer, local_counter_ctx[m], source);
                                //  multithread_decryption_common_counter(buf,  tinfo[m].compute_size, enc_data, pos, dec_pos, tinfo[m].tstart,  tinfo[m].counter, large_recv_buffer, dec_common_buffer,local_counter_ctx[m], source);
#if DEBUG_PRECOMPUTE_COUNTER
                                int tid = omp_get_thread_num();
                                //printf("tid = %d precompute recv: totaldata =%d enc_data =%d m =%d common_counter_gen_sz %d t_mask %d t_mask_pos %d t_counter %lu\n",
                                //       tid, totaldata, enc_data, m, common_counter_gen_sz, t_mask, t_mask_pos, t_counter);
                                printf("tid = %d precompute waitall: totaldata =%d enc_data =%d m =%d  t_counter %lu recv_counter %lu pos %d dec_pos %d segment_counter %d segments_no %d\n",
                                       tid, totaldata, enc_data, m, t_counter,recv_counter,pos,dec_pos, segment_counter, segments_no);
                                fflush(stdout);
#endif
                            } // end of for loop
                        } // end of parallel region
                            //printf("common_counter_gen_sz =%d\n",common_counter_gen_sz);fflush(stdout);
                          
                            recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
                            recv_pos += inner_totaldata;
                            dec_loc += inner_totaldata;
                            
                        }
                    }

#if ALL_COMM_PLUS_ENC_TIME
                    gettimeofday(&t_s2, NULL);
                    double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);

                    if (totaldata < FOUR_K)
                    {
                        inter_less_than_4K += t;
                    }
                    else if (totaldata < SIXTEEN_K)
                    {
                        inter_less_than_16K += t;
                    }
                    else if (totaldata < THIRTY_2K)
                    {
                        inter_less_than_32K += t;
                    }
                    else if (totaldata < SIXTY_4K)
                    {
                        inter_less_than_64K += t;
                    }
                    else if (totaldata < ONE_TWO_8K)
                    {
                        inter_less_than_128K += t;
                    }
                    else if (totaldata < TWO_FIVE_6K)
                    {
                        inter_128K_256K += t;
                    }
                    else if (totaldata < FIVE_ONE_2K)
                    {
                        inter_256K_512K += t;
                    }
                    else if (totaldata < ONE_M)
                    {
                        inter_512K_1M += t;
                    }
                    else if (totaldata >= ONE_M)
                    {
                        inter_more_than_1M += t;
                    }
#endif
                }

                else
                {
                    //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
                     mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
                }
            }
            else if (nonblock_req_handler[index].req_type == 1)
            {
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s1, NULL);
#endif
                totaldata = nonblock_req_handler[index].totaldata;
                if(totaldata <= 16)
                    common_counter_gen_sz = 16;
                else if(totaldata <= 2048)
                      common_counter_gen_sz = totaldata;  
                else if(totaldata > 2048)
                     common_counter_gen_sz = 1048;    
              

                    int j = 0;


                for (i = 0; i < nonblock_req_handler[index].total_request; i++)
                {
                    /* Dynamic pre-computation */
                    
                    flag = 0;
                    MPI_Test(&nonblock_req_handler[index].request[i], &flag, &sta);

                    int enc_mask =0;

                    while(!flag)
                    {
                      generateCommonEncMask(common_counter_gen_sz);
                     // enc_mask += common_counter_gen_sz;
                     // printf("n %d gen size waitall %d\n",n, common_counter_gen_sz);fflush(stdout);
                      MPI_Test(&nonblock_req_handler[index].request[i], &flag, &sta);
                    }
                  
                       pendingIsendRequestCount[nonblock_req_handler[index].dest]-=1;
                   
                }
 //}
#if ALL_COMM_PLUS_ENC_TIME
                gettimeofday(&t_s2, NULL);
                double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
                totaldata = nonblock_req_handler[index].totaldata;
                //total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);

                if (totaldata < FOUR_K)
                {
                    inter_less_than_4K += t;
                }
                else if (totaldata < SIXTEEN_K)
                {
                    inter_less_than_16K += t;
                }
                else if (totaldata < THIRTY_2K)
                {
                    inter_less_than_32K += t;
                }
                else if (totaldata < SIXTY_4K)
                {
                    inter_less_than_64K += t;
                }
                else if (totaldata < ONE_TWO_8K)
                {
                    inter_less_than_128K += t;
                }
                else if (totaldata < TWO_FIVE_6K)
                {
                    inter_128K_256K += t;
                }
                else if (totaldata < FIVE_ONE_2K)
                {
                    inter_256K_512K += t;
                }
                else if (totaldata < ONE_M)
                {
                    inter_512K_1M += t;
                }
                else if (totaldata >= ONE_M)
                {
                    inter_more_than_1M += t;
                }

#endif
                nonblock_req_handler[index].req_type = 0;
                //printf("waitall done for isend index=%d\n",index);fflush(stdout);
            }
        }
        else
        { /* Looks like, sending to own process, same node, unencruypted comm*/
#if ALL_COMM_PLUS_ENC_TIME
            int datasz;
            //struct timeval t_s1, t_s2;
            gettimeofday(&t_s1, NULL);
#endif
           
            //mpi_errno = MPI_Wait_original(&arr_req[n], &arr_sta[n]);
             mpi_errno = MPI_Wait_original(&arr_req[n], &sta);
            //printf("    [rank=%d] else done index =%d n=%d \n",comm_ptr->rank,index,n);fflush(stdout);
#if ALL_COMM_PLUS_ENC_TIME
            gettimeofday(&t_s2, NULL);
            double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
            MPI_Get_count(&arr_sta[n], MPI_CHAR, &datasz);
            if (datasz < ONE_TWO_8K)
            {
                intra_less_than_128K += t;
            }
            else if (datasz < TWO_FIVE_6K)
            {
                intra_128K_256K += t;
            }
            else if (datasz < FIVE_ONE_2K)
            {
                intra_256K_512K += t;
            }
            else if (datasz < ONE_M)
            {
                intra_512K_1M += t;
            }
            else if (datasz >= ONE_M)
            {
                intra_more_than_1M += t;
            }
#endif
        }
    }

    return mpi_errno;
}


int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[])
{

#if ALL_COMM_PLUS_ENC_TIME
    gettimeofday(&comm_start_time, NULL);
#endif

    int mpi_errno = MPI_SUCCESS;

    if (security_approach == 600 || security_approach == 601) // (OPENMP_MULTI_THREAD_ONLY)
        mpi_errno = MPI_SEC_Waitall_MultiThreads_openmp(count, array_of_requests, array_of_statuses);
    else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        mpi_errno = MPI_SEC_Waitall_pipeline_openmp_largeSegment_3(count, array_of_requests, array_of_statuses);
    else if (security_approach == 700 || security_approach == 701) //(BASE_COUNTER_MODE)
        mpi_errno = MPI_SEC_BaseCounter_Pipeline_Waitall(count, array_of_requests, array_of_statuses);
    else if (security_approach == 702) //(PRE_COMPUTE_COUNTER_MODE)
        mpi_errno = MPI_SEC_PreComputeCounter_Pipeline_Waitall_v4(count, array_of_requests, array_of_statuses);

    else
    {
        mpi_errno = MPI_Waitall_original(count, array_of_requests, array_of_statuses);
    }

#if ALL_COMM_PLUS_ENC_TIME
    gettimeofday(&comm_end_time, NULL);
    double t = (double)(comm_end_time.tv_usec - comm_start_time.tv_usec) / 1000000 + (double)(comm_end_time.tv_sec - comm_start_time.tv_sec);
    total_comm_plus_enc_time += t;
    waitall_total_time += t;
#endif

    return mpi_errno;
}

/* End of add by Abu Naser */
