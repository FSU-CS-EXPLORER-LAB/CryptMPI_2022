/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* Added by Abu Naser(an16e@my.fsu.edu) */

struct isend_req nonblock_req_handler[ISEND_REQ+5];
//int max_send_data_size = 0;
unsigned char Iciphertext[NON_BLOCK_SEND][NON_BLOCK_SEND2];;
//int nonBlockCounter = 0;
long pendingIsendRequestCount[MAX_RANKS_LIMIT];

/* End of add by Abu Naser */

/* -- Begin Profiling Symbol Block for routine MPI_Wait */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Wait = PMPI_Wait
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Wait  MPI_Wait
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Wait as PMPI_Wait
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Wait(MPI_Request * request, MPI_Status * status) __attribute__ ((weak, alias("PMPI_Wait")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Wait
#define MPI_Wait PMPI_Wait

#undef FUNCNAME
#define FUNCNAME MPIR_Wait_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Wait_impl(MPIR_Request * request_ptr, MPI_Status * status)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Progress_state progress_state;
    if (request_ptr == NULL)
        goto fn_exit;

    MPID_Progress_start(&progress_state);
    while (!MPIR_Request_is_complete(request_ptr)) {
        mpi_errno = MPID_Progress_wait(&progress_state);
        if (mpi_errno) {
            /* --BEGIN ERROR HANDLING-- */
            MPID_Progress_end(&progress_state);
            MPIR_ERR_POP(mpi_errno);
            /* --END ERROR HANDLING-- */
        }

        if (unlikely(MPIR_Request_is_anysrc_mismatched(request_ptr))) {
            mpi_errno = MPIR_Request_handle_proc_failed(request_ptr);
            goto fn_fail;
        }
    }
    MPID_Progress_end(&progress_state);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Wait
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Wait(MPI_Request * request, MPI_Status * status)
{
    int mpi_errno = MPI_SUCCESS;
    int active_flag;
    MPIR_Request *request_ptr = NULL;

    /* If this is a null request handle, then return an empty status */
    if (*request == MPI_REQUEST_NULL) {
        MPIR_Status_set_empty(status);
        goto fn_exit;
    }

    MPIR_Request_get_ptr(*request, request_ptr);
    MPIR_Assert(request_ptr != NULL);

    if (!MPIR_Request_is_complete(request_ptr)) {
        /* If this is an anysource request including a communicator with
         * anysource disabled, convert the call to an MPI_Test instead so we
         * don't get stuck in the progress engine. */
        if (unlikely(MPIR_Request_is_anysrc_mismatched(request_ptr))) {
            mpi_errno = MPIR_Test(request, &active_flag, status);
            goto fn_exit;
        }

        if (MPIR_Request_has_poll_fn(request_ptr)) {
            while (!MPIR_Request_is_complete(request_ptr)) {
                mpi_errno = MPIR_Grequest_poll(request_ptr, status);
                if (mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                /* Avoid blocking other threads since I am inside an infinite loop */
                MPID_THREAD_CS_YIELD(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
            }
        } else {
            mpi_errno = MPID_Wait(request_ptr, status);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

    mpi_errno = MPIR_Request_completion_processing(request_ptr, status, &active_flag);
    if (!MPIR_Request_is_persistent(request_ptr)) {
        MPIR_Request_free(request_ptr);
        *request = MPI_REQUEST_NULL;
    }
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Wait
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
    MPI_Wait - Waits for an MPI request to complete

Input Parameters:
. request - request (handle)

Output Parameters:
. status - status object (Status).  May be 'MPI_STATUS_IGNORE'.

.N waitstatus

.N ThreadSafe

.N Fortran

.N FortranStatus

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_REQUEST
.N MPI_ERR_ARG
@*/
int MPI_Wait_original(MPI_Request * request, MPI_Status * status)
{
    MPIR_Request *request_ptr = NULL;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_WAIT);

    MPIR_ERRTEST_INITIALIZED_ORDIE();

    MPID_THREAD_CS_ENTER(VNI_GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_REQUEST_ENTER(MPID_STATE_MPI_WAIT);

    /* Check the arguments */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_ARGNULL(request, "request", mpi_errno);
            /* NOTE: MPI_STATUS_IGNORE != NULL */
            MPIR_ERRTEST_ARGNULL(status, "status", mpi_errno);
            MPIR_ERRTEST_REQUEST_OR_NULL(*request, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* If this is a null request handle, then return an empty status */
    if (*request == MPI_REQUEST_NULL) {
        MPIR_Status_set_empty(status);
        goto fn_exit;
    }

    /* Convert MPI request handle to a request object pointer */
    MPIR_Request_get_ptr(*request, request_ptr);

    /* Validate object pointers if error checking is enabled */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_Request_valid_ptr(request_ptr, mpi_errno);
            if (mpi_errno)
                goto fn_fail;
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */

    /* save copy of comm because request will be freed */
    if (request_ptr)
        comm_ptr = request_ptr->comm;
    mpi_errno = MPIR_Wait(request, status);
    if (mpi_errno)
        goto fn_fail;

    /* ... end of body of routine ... */

  fn_exit:
    MPIR_FUNC_TERSE_REQUEST_EXIT(MPID_STATE_MPI_WAIT);
    MPID_THREAD_CS_EXIT(VNI_GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#ifdef HAVE_ERROR_CHECKING
    mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE,
                                     FCNAME, __LINE__, MPI_ERR_OTHER,
                                     "**mpi_wait", "**mpi_wait %p %p", request, status);
#endif
    mpi_errno = MPIR_Err_return_comm(comm_ptr, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

// ======================= wait for non blocking send ===========================
/*
  wait for isend
  - multi-thread
  - all shceduling algorithms
  - for openmp pipeline
*/
int MPI_SEC_Wait_for_isend(MPI_Request *request, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status sta;
    MPI_Request req;

    int index, i;
    struct timeval t_s1, t_s2;
    int totaldata, common_counter_gen_sz, flag;

#if NONBLOCK_SEND_DEBUG
    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
#endif
    index = *request;

    if (nonblock_req_handler[index].source != nonblock_req_handler[index].dest)
    {

#if ALL_COMM_PLUS_ENC_TIME
        gettimeofday(&t_s1, NULL);
#endif

        totaldata = nonblock_req_handler[index].totaldata;
#if PRE_COMPUTE_COUNTER_MODE          
        if (totaldata <= 16)
            common_counter_gen_sz = 16;
        else if (totaldata <= 2048)
            common_counter_gen_sz = totaldata;
        else if (totaldata > 2048)
            common_counter_gen_sz = 1048;
#endif            
        for (i = 0; i < nonblock_req_handler[index].total_request; i++)
        {
#if PRE_COMPUTE_COUNTER_MODE            
            flag = 0;
            MPI_Test(&nonblock_req_handler[index].request[i], &flag, &sta);
            //int enc_mask =0;
            while (!flag)
            {
                generateCommonEncMask(common_counter_gen_sz);
                // enc_mask += common_counter_gen_sz;
                // printf("n %d gen size waitall %d\n",n, common_counter_gen_sz);fflush(stdout);
                MPI_Test(&nonblock_req_handler[index].request[i], &flag, &sta);
            }
#else
            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[i], &sta);
#endif            

            pendingIsendRequestCount[nonblock_req_handler[index].dest] -= 1;
        }
#if ALL_COMM_PLUS_ENC_TIME
        gettimeofday(&t_s2, NULL);
        double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);
        int totaldata = nonblock_req_handler[index].totaldata;
        int isSamenode = 0;
        int dest = nonblock_req_handler[index].dest;

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
    else
    {
        mpi_errno = MPI_Wait_original(request, status);
    }
#if NONBLOCK_SEND_DEBUG
    printf("  >>>   [rank=%d]    Wait for isend is done  index=%d  <<<<<\n", comm_ptr->rank, index);
    fflush(stdout);
#endif
    return mpi_errno;
}

// ==============================================================================

/*
    - openmp
    - Multi-thread irecv
*/
int MPI_SEC_Multi_Thread_irecv_openmp(MPI_Request *req, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;

    unsigned long ciphertext_len, max_out_len;
    unsigned long decrypted_len = 0;

    int recvtype_sz, totaldata, start, pos, m, i, s;
    int index, source, dest, tag;
    int cancel_req_no;
    char *buf;
    MPI_Status sta;
    

    index = *req;
    source = nonblock_req_handler[index].source;
    dest = nonblock_req_handler[index].dest;
    tag = nonblock_req_handler[index].tag;
    //totaldata = nonblock_req_handler[index].totaldata;
    nonblock_req_handler[index].req_type = 0;
    buf = nonblock_req_handler[index].buffer;
    int comm = MPI_COMM_WORLD;
    int segment_counter;
    unsigned long count;
    struct timeval t_s1, t_s2;

#if ALL_COMM_PLUS_ENC_TIME 
    gettimeofday(&t_s1, NULL);
#endif  
    
    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[0], &sta);

//#if COMM_TIME_DEBUG    
//    gettimeofday(&wait_time_end, NULL);
//    total_recv_wait_time += (double) (wait_time_end.tv_usec - wait_time_start.tv_usec)/1000000  + (double) (wait_time_end.tv_sec - wait_time_start.tv_sec);
//#endif    
    totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);
     if(totaldata > ONE_M)
        long_msg_flag=1;
    int my_thread_no = 1; // MY_OPENMP_THREADS_NO;
#if OMP_DYNAMIC_THREADS
    if (totaldata < SIXTY_4K)
        my_thread_no = 1;
    else if (totaldata < ONE_TWO_8K)
        my_thread_no = 2;
    else if (totaldata < FIVE_ONE_2K)
        my_thread_no = 4;
    else if (totaldata >= FIVE_ONE_2K)
        my_thread_no = 8;
    /* if (totaldata < SIXTY_4K)
        my_thread_no = 1;
      else if (totaldata < ONE_TWO_8K)
        my_thread_no = 2;
      else if (totaldata < FIVE_ONE_2K)
        my_thread_no = 4;
      else if (totaldata <= ONE_M) 
        my_thread_no = 8;
      else
      {
          my_thread_no = 4;
      } */   
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
  /*  cyptmpi_series_thread = ((no_of_max_omp_threads/2) - (cryptmpi_local_process*2)) / cryptmpi_local_process;
     
     if(cyptmpi_series_thread < 1)
        cyptmpi_series_thread = 1;*/

   if(my_thread_no > cyptmpi_series_thread)
    my_thread_no = cyptmpi_series_thread;
#endif

   /* if((cyptmpi_series_thread+2) * cryptmpi_local_process < (no_of_max_omp_threads - cryptmpi_local_process*2)){
       cyptmpi_series_thread +=2;
   }
   else if((cyptmpi_series_thread+1) * cryptmpi_local_process < (no_of_max_omp_threads - cryptmpi_local_process*2)){
       cyptmpi_series_thread +=1;
   }*/
   //cyptmpi_series_thread = (no_of_max_omp_threads - cryptmpi_local_process*2) / cryptmpi_local_process;
   // if(cyptmpi_series_thread < 1)
    //    cyptmpi_series_thread = 1;
#if OMP_DYNAMIC_THREADS   
   /* cyptmpi_series_thread = ((no_of_max_omp_threads/2) - (cryptmpi_local_process*2)) / cryptmpi_local_process;
     
     if(cyptmpi_series_thread < 1)
        cyptmpi_series_thread = 1;*/

   if(my_thread_no > cyptmpi_series_thread)
    my_thread_no = cyptmpi_series_thread;
#endif   
 //int data = totaldata / my_thread_no;
    //data++;

    unsigned int data = 0;
    data = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
    int segment_no;

    if (totaldata <= 16)
    {
        segment_no = 1;
        data = totaldata;
    }
    else
    {
        segment_no = (totaldata - 1) / data + 1;
    }

    int active_threads;
    
     if((!OMP_DYNAMIC_THREADS) && (!OMP_DYNAMIC_THREADS_INNOVATION)){
        if (totaldata <= MULTI_THREAD_START_SIZE)
        {
            data = totaldata;
            segment_no = 1;
            active_threads = 1;
        }
    }

    // my_thread_no = segment_no;   
    

#if CRYPTMPI_MONITOR_THREAD_USE
    /* Check how many threads are using by others */
    int diff = -1;
    int flag = 0;
    if (!(cryptmpi_local_process * 12 < no_of_max_omp_threads))
    {
        int thread_sum = 0;
        for (i = 0; i < cryptmpi_local_process; i++)
            thread_sum += *(cryptmpi_ranks + i);

        if (!(thread_sum + my_thread_no < no_of_max_omp_threads ))
        {

            diff = no_of_max_omp_threads - thread_sum;

            if (diff > 1)
            {
                my_thread_no = diff - 1;
            }
            else
                my_thread_no = 1;
        }

        flag = 1;
        *(cryptmpi_ranks + cryptmpi_local_rank) = my_thread_no;
#if CRYPTMPI_THREAD_MONITOR_DEBUG
        printf("        wait rank =%d  thread_sum=%d cryptmpi_local_process=%d diff = %d no_of_max_omp_threads=%d my_thread_no=%d\n", comm_ptr->rank, thread_sum, cryptmpi_local_process, diff, no_of_max_omp_threads, my_thread_no);
        fflush(stdout);
#endif
    }
#endif

        active_threads = my_thread_no;         

//#if COMM_TIME_DEBUG
//    gettimeofday(&wait_time_start, NULL);
//#endif      

    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

//#if COMM_TIME_DEBUG    
//    gettimeofday(&wait_time_end, NULL);
//    total_recv_wait_time += (double) (wait_time_end.tv_usec - wait_time_start.tv_usec)/1000000  + (double) (wait_time_end.tv_sec - wait_time_start.tv_sec);
//#endif        

    start = MSG_HEADER_SIZE;
    pos = 0;
    /* prepare for decryption */
    

    
    
   
    

#if NONBLOCK_RECV_DEBUG
    printf("Openmp-wait-irecv: totaldata=%d data=%d segment_no=%d my_thread_no=%d\n",totaldata,data,segment_no,my_thread_no);fflush(stdout);
#endif    

   
   // if(OMP_DYNAMIC_THREADS || OMP_DYNAMIC_THREADS_INNOVATION)
    //   omp_set_dynamic(my_thread_no);





    /* prepare for decryption */
#if ENC_DEC_TIME_DEBUG    
    gettimeofday(&omp_tv3, NULL);
#endif
  /*  if (segment_no == 1)
    {
        EVP_AEAD_CTX *local_ctx = global_openmp_ctx;

        max_out_len = totaldata + 16;
        pos = MSG_HEADER_SIZE;

      

        if (!EVP_AEAD_CTX_open(local_ctx, buf,
                               &count, totaldata,
                               &Ideciphertext[index][pos], 12,
                               &Ideciphertext[index][pos + 12], (unsigned long)(max_out_len),
                               NULL, 0))
        {
            printf("[T = %d] OpenMP multi-thread-irecv Error in encryption: segment_counter=%d pos = %d\n", omp_get_thread_num(), segment_counter, pos);
            fflush(stdout);
        }
#if OMP_DEBUG
        else
        {
            printf(COLOR_BLUE "  [T = %d] OpenMP multi-thread-irecv Decrypted data is %d in segment_counter = %d pos = %d" COLOR_RESET "\n",
                   omp_get_thread_num(), count, segment_counter, pos);
            fflush(stdout);
        }
#endif
    }
    else
    {*/
#if OPENMP_THREADS
#pragma omp parallel for schedule(static) default(none) private(segment_counter, count, max_out_len, pos) shared(buf, Ideciphertext, segment_no, data, global_openmp_ctx, totaldata, index) num_threads(my_thread_no)
#endif
        for (segment_counter = 0; segment_counter < segment_no; segment_counter++)
        {

            EVP_AEAD_CTX *local_ctx = global_openmp_ctx;

            max_out_len = data + 16;
            pos = MSG_HEADER_SIZE + segment_counter * (data + 12 + 16);
            int enc_data = data;
            //active_threads = omp_get_num_threads();
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
                printf("[T = %d] OpenMP multi-thread-irecv Error in encryption: segment_counter=%d pos = %d\n", omp_get_thread_num(), segment_counter, pos);
                //fflush(stdout);
            }
#if OMP_DEBUG
            else
            {
                printf(COLOR_BLUE "  [T = %d] OpenMP multi-thread-irecv Decrypted data is %d in segment_counter = %d pos = %d" COLOR_RESET "\n",
                       omp_get_thread_num(), count, segment_counter, pos);
                fflush(stdout);
            }
#endif
        }
    //}
#if ENC_DEC_TIME_DEBUG    
    gettimeofday(&omp_tv4, NULL);
    omp_t2 += (double) (omp_tv4.tv_usec - omp_tv3.tv_usec)/1000000  + (double) (omp_tv4.tv_sec - omp_tv3.tv_sec);
#endif

/* reduce the thread counter */
#if CRYPTMPI_MONITOR_THREAD_USE
    if (flag == 1)
    {
        *(cryptmpi_ranks + cryptmpi_local_rank) = 0;
    }
#endif

#if ALL_COMM_PLUS_ENC_TIME    
    gettimeofday(&t_s2, NULL);
    double t = (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
    /*if(totaldata > ONE_M)
        total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
    else
        total_comm_plus_enc_time_small_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);*/
        int isSamenode = 0;

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
            else if(totaldata < ONE_TWO_8K){
                inter_less_than_128K += t;
            }
            else if(totaldata < TWO_FIVE_6K){
                inter_128K_256K +=t;
            }
            else if(totaldata < FIVE_ONE_2K){
                inter_256K_512K +=t;
            }
            else if(totaldata < ONE_M){
                inter_512K_1M +=t;
            }
            else if(totaldata >= ONE_M){
                inter_more_than_1M +=t;
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
    //  printf(">> MPI_SEC_Multi_Thread_irecv openmp\n");fflush(stdout);
    return mpi_errno;
}

/*
- use different tag for header

*/
int MPI_SEC_MThreads_PipeLine_Wait_irecv_openmp_largeSegment_3(MPI_Request *req, MPI_Status *status)
{

    int mpi_errno = MPI_SUCCESS;
    int var, j;
    int sendtype_sz, chopping_tag, i, recv_len, len;
    unsigned long next, datasize, dest;
    char newkey[40];
    unsigned char nonce[50], V[50];
    unsigned long count,  index, decrypted_len;
    int totaldata;
    unsigned int segment_counter;
    int segments_no;
    MPI_Request request[300];
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;
    struct timeval t_s1, t_s2;

    unsigned int nonce_counter;
    int m, th_data, my_thread_no, th_start, temp_chunk;
    int cancel_req_no;
    int th_sum = 0;
    int flag;

    //chopping_tag = 0;
    EVP_AEAD_CTX *local_ctx = NULL;

    int pos, dec_pos;
    unsigned char local_nonce[20];
    EVP_AEAD_CTX *segment_ctx[100]; // total threads will not excede 100
    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);

    index = *req;
    source = nonblock_req_handler[index].source;
    dest = nonblock_req_handler[index].dest;
    int tag = nonblock_req_handler[index].tag;
    //totaldata = nonblock_req_handler[index].totaldata;
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
                   // printf("    wait: source = %d i = %d no_tag = %d\n", source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
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

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION                   
                    ranksIrecvS[source] -=1;
#endif         
        totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);

#if CRYPTMPI_ADAPTIVE_CHOPP
        choping_sz = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
      //   printf(COLOR_YELLOW"rank %d wait recv: choping_sz =%d flag=%c totaldata=%d source=%d dest=%d index=%d tag=%d"COLOR_RESET "\n",comm_ptr->rank,choping_sz, Ideciphertext[nonBlockCounter][20],totaldata, source,dest,index,tag); fflush(stdout);

        // printf(COLOR_YELLOW"rank %d wait recv: choping_sz =%d flag=%c totaldata=%d source=%d dest=%d index=%d tag=%d"COLOR_RESET "\n",comm_ptr->rank,choping_sz, Ideciphertext[nonBlockCounter][20],totaldata, source,dest,index,tag); fflush(stdout);

#endif
          //printf(COLOR_BLUE"[rank=%d] wait for recv source=%d dest=%d totaldata=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest,totaldata);fflush(stdout);
        if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
        {
            segments_no = 1;
            segments_no += (int)(totaldata - (PIPELINE_SIZE) - 1) / (PIPELINE_SIZE) + 1;
        }
        else
        {
            segments_no = 1;
        }

        //   printf("index=%d source=%d dest=%d totaldat=%d segments_no=%d\n",index,source,dest,totaldata,segments_no);fflush(stdout);

        th_data = choping_sz;

        if(segments_no != nonblock_req_handler[index].segment_number){
        cancel_req_no = nonblock_req_handler[index].segment_number - segments_no;
         for (m = 0; m < cancel_req_no; m++){
            int temp = nonblock_req_handler[index].total_request -1 - m;
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

        th_start = 0;
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE
        if(Ideciphertext[index][20] == '3' || Ideciphertext[index][20] == '4') // for '2' above equation will fine. '3' from isend, '4' from send  
            segments_no = 1;
#endif

      
    
              local_ctx = global_small_msg_ctx;
       

#if NONBLOCK_RECV_DEBUG
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("[Rank = %d ]:openmp-pipeline-irecv  *req = %lu index = %d  totaldata = %lu source = %d dest = %d\n",
               rank, ((unsigned long)*req), index, totaldata, source, dest);
        fflush(stdout);
#endif

        if ((totaldata <= PIPELINE_SIZE ) && (Ideciphertext[index][20] != '3' && Ideciphertext[index][20] != '4'))
        {
            segment_counter = 0;
            pos = MSG_HEADER_SIZE + NONCE_HEADER;

//#if COMM_TIME_DEBUG    
//                    gettimeofday(&wait_time_start, NULL);
//#endif 

            mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

//#if COMM_TIME_DEBUG    
//            gettimeofday(&wait_time_end, NULL);
//            total_recv_wait_time += (double) (wait_time_end.tv_usec - wait_time_start.tv_usec)/1000000  + (double) (wait_time_end.tv_sec - wait_time_start.tv_sec);
//#endif 

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION                    
            ranksIrecvS[source] -=1;
#endif 


#if ENC_DEC_TIME_DEBUG                     
            gettimeofday(&omp_tv3, NULL);
#endif  
            /*local_nonce[0] = '0';
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
            local_nonce[11] = Ideciphertext[index][pos - 1];*/

            //nonce_counter = ((unsigned char)Ideciphertext[index][pos - 1] << 0) | ((unsigned char)Ideciphertext[index][pos - 2] << 8) | ((unsigned char)Ideciphertext[index][pos - 3] << 16) | ((unsigned char)Ideciphertext[index][pos - 4] << 24);
            //dec_pos = nonce_counter * totaldata;
            //printf("Openmp-pipeline: dec_pos=%d nonce_counter=%d pos=%d\n",dec_pos,nonce_counter,pos);
            //fflush(stdout);

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

#if ENC_DEC_TIME_DEBUG                    
                    gettimeofday(&omp_tv4, NULL);
                    omp_t2 += (double) (omp_tv4.tv_usec - omp_tv3.tv_usec)/1000000  + (double) (omp_tv4.tv_sec - omp_tv3.tv_sec);
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
            if(Ideciphertext[index][20] == '1'){
               
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
                else { // '2' , '3' , '4'
                 
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
            else if(totaldata < TWO_M)
                my_thread_no = 2;
            else
                my_thread_no = 8;
#endif 

#if OMP_DYNAMIC_THREADS_PIPELINE
                if (my_thread_no > cyptmpi_series_thread)
                    my_thread_no = cyptmpi_series_thread;
#endif
/* Switch Between CryptMPI-R and CryptMPI-C */
#if ENABLE_CRYPTMPI_RC_SWITCH
    if( omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
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

            th_data = choping_sz;
            //  printf("th_data = %d and recv_len = %d segments_no=%d\n",th_data, recv_len,segments_no);fflush(stdout);
            th_start = NONCE_HEADER + MSG_HEADER_SIZE;

#if NONBLOCK_RECV_DEBUG
            printf("    Oenmp-pipeline-irecv my_thread_no=%d th_data=%d choping_sz=%d segments_no=%d th_start=%d\n", my_thread_no, th_data, choping_sz, segments_no, th_start);
            fflush(stdout);
#endif
            int active_thread_num = my_thread_no; //omp_get_num_threads();
            int ii;

          


#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
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

#if ENC_DEC_TIME_DEBUG                     
                gettimeofday(&omp_tv3, NULL);
#endif                 
//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, Ideciphertext, th_data, segment_ctx, new_segments, local_ctx, stdout, index, totaldata) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, Ideciphertext, th_data, segment_ctx, new_segments, local_ctx, index, totaldata) num_threads(my_thread_no)
{
    #pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) 
      
                for (segment_counter = 0; segment_counter < new_segments; segment_counter++)
                {
                    //pos = (segment_counter - ii + m) * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER);
                    //pos += NONCE_HEADER+MSG_HEADER_SIZE;
                    // max_out_len = th_data + 16;
                    pos = MSG_HEADER_SIZE + segment_counter * (th_data + NONCE_HEADER + ENC_MSG_TAG_SIZE) + NONCE_HEADER;
                    int enc_data = th_data;
                    // active_threads = omp_get_num_threads();
                    // printf("waitall active_threads=%d\n",active_threads);fflush(stdout);

                    if (segment_counter == new_segments - 1)
                    {
                        enc_data = totaldata - th_data * (new_segments - 1);
                        //max_out_len = enc_data + 16;
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
                    //segment_ctx[omp_get_thread_num()] = local_ctx;

                    // printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d %u %u %u %u => %u pos=%d\n",
                    // omp_get_thread_num(),segment_counter,m,ii,th_data,nonce_counter,dec_pos,large_recv_buffer[pos-1],large_recv_buffer[pos-2],large_recv_buffer[pos-3],large_recv_buffer[pos-4],0,pos );fflush(stdout);
                    if (!EVP_AEAD_CTX_open(segment_ctx[omp_get_thread_num()], buf + dec_pos,
                                           &decrypted_len, enc_data,
                                           local_nonce, 12,
                                           &Ideciphertext[index][pos], enc_data + 16,
                                           NULL, 0))
                    {
                        printf(COLOR_RED "OpenMP Pipeline: Decryption error wait:  thread %d nonce_counter %d segment_counter %d new_segments=%d th_data=%d" COLOR_RESET "\n",
                               omp_get_thread_num(), nonce_counter, segment_counter, new_segments, th_data);
                        //fflush(stdout);
                    }
#if OMP_DEBUG_DYNAMIC_THREAD_PIPELINE
                    else
                    {
                        printf(COLOR_BLUE "  wait: OpenMP Dynamic Thread Pipeline: Decrypted data is %d in thread = %d nonce_counter=%d" COLOR_RESET "\n",
                               decrypted_len, omp_get_thread_num(), nonce_counter);
                        fflush(stdout);
                    }
#endif
                } // end of parallel loop
    } // end of parallel region
#if ENC_DEC_TIME_DEBUG                    
                gettimeofday(&omp_tv4, NULL);
                omp_t2 += (double) (omp_tv4.tv_usec - omp_tv3.tv_usec)/1000000  + (double) (omp_tv4.tv_sec - omp_tv3.tv_sec);
#endif                  
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
                  //  printf("waiting to recv data source = %d dest =%d totaldata =%d\n",source,dest, totaldata);fflush(stdout);
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[segment_counter + 1], &sta);
                   // printf("	wait done  source = %d dest =%d totaldata =%d\n",source,dest, totaldata);fflush(stdout); 
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

                    if (my_thread_no > ii)
                        my_thread_no = ii;

                        /* Decrypt received data */
#if OPENMP_DYNAMIC_SCH
#pragma omp parallel for default(none) schedule(dynamic) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
#elif OPENMP_STATIC_SCH
//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, ii, segment_counter, index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
{
    #pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) 

#elif OPENMP_GUIDED_SCH
#pragma omp parallel for default(none) schedule(guided) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
#else
#pragma omp parallel for default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, Ideciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, index,totaldata, enc_loc, inner_totaldata,recv_loc) num_threads(my_thread_no)
#endif
                for (m = 0; m < ii; m++)
                {

                    //pos = (prsd_segment + m) * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER);
                    //pos += NONCE_HEADER+MSG_HEADER_SIZE;
                     pos =  MSG_HEADER_SIZE + enc_loc + m * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;

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

                   // dec_pos = nonce_counter * th_data;
                     dec_pos = recv_loc + m * th_data;
                    //segment_ctx[omp_get_thread_num()] = local_ctx;

                    int enc_data = th_data;
                    if (m == ii - 1)
                    {
                        enc_data = inner_totaldata - th_data * (ii - 1);
                    }

                    /*printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d pos=%d prsd_segment=%d\n",
                           omp_get_thread_num(), segment_counter, m, ii, th_data, nonce_counter, dec_pos, pos, prsd_segment);
                    fflush(stdout);*/

                    // printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d %u %u %u %u => %u pos=%d\n",
                    // omp_get_thread_num(),segment_counter,m,ii,th_data,nonce_counter,dec_pos,large_recv_buffer[pos-1],large_recv_buffer[pos-2],large_recv_buffer[pos-3],large_recv_buffer[pos-4],0,pos );fflush(stdout);
                    if (!EVP_AEAD_CTX_open(segment_ctx[omp_get_thread_num()], buf + dec_pos,
                                           &decrypted_len, enc_data,
                                           local_nonce, 12,
                                           &Ideciphertext[index][pos], enc_data + 16,
                                           NULL, 0))
                    {
                        printf(COLOR_RED "OpenMP Pipeline : Decryption error wait:  thread %d nonce_counter %d segment_counter %d inner_totaldata=%d totaldata=%d pos=%d ii=%d m=%d th_data=%d enc_loc=%d" COLOR_RESET "\n", 
                        omp_get_thread_num(), nonce_counter, segment_counter,inner_totaldata, totaldata,pos,ii,m,th_data,enc_loc);
                        //fflush(stdout);
                    }

#if OMP_DEBUG
                    else
                    {
                        printf(COLOR_BLUE "wait: OpenMP Pipeline irecv: Decrypted data is %d in thread = %d nonce_counter %d segment_counter %d inner_totaldata=%d totaldata=%d pos=%d ii=%d m=%d th_data=%d enc_loc=%d" COLOR_RESET "\n", 
                        decrypted_len, omp_get_thread_num(),nonce_counter, segment_counter,inner_totaldata, totaldata,pos,ii,m,th_data,enc_loc);
                        fflush(stdout);
                    }
#endif
                }
        } // end of parallel region

                // prsd_segment += ii; 
                //enc_loc += inner_totaldata + (ENC_MSG_TAG_SIZE + NONCE_HEADER) * ii;
                enc_loc += inner_totaldata + (ENC_MSG_TAG_SIZE + NONCE_HEADER) * MAX_OMP_THREADS_LIMIT;
                recv_loc += inner_totaldata;                


            }



#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
        } // end of else
#endif 

          
            if ((totaldata > SUBKEY_GEN_START)){   
                for (i = 0; i < my_thread_no; i++)
                    EVP_AEAD_CTX_free( segment_ctx[i]);
            }
        }
#if PSC_THREAD_USE_DEBUG
        if(max_wait_thread_used < my_thread_no)
            max_wait_thread_used = my_thread_no;
#endif

#if ALL_COMM_PLUS_ENC_TIME    
        gettimeofday(&t_s2, NULL);
        double t = (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
       /* if(totaldata > ONE_M)
            total_comm_plus_enc_time_long_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);
        else
            total_comm_plus_enc_time_small_msg += (double) (t_s2.tv_usec - t_s1.tv_usec)/1000000  + (double) (t_s2.tv_sec - t_s1.tv_sec);*/
        
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
        else if(totaldata < ONE_TWO_8K){
            inter_less_than_128K += t;
        }
        else if(totaldata < TWO_FIVE_6K){
            inter_128K_256K +=t;
        }
        else if(totaldata < FIVE_ONE_2K){
            inter_256K_512K +=t;
        }
        else if(totaldata < ONE_M){
            inter_512K_1M +=t;
        }
        else if(totaldata >= ONE_M){
            inter_more_than_1M +=t;
        }    
#endif 

    } // if source and dest are same
    else
    {
        mpi_errno = MPI_Wait_original(req, &sta);
    }
    //printf(COLOR_YELLOW"[rank=%d] wait for recv is done source=%d dest=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest);fflush(stdout);
    return mpi_errno;
}

/*
    - Begin counter mode
*/
#if CRYPTMPI_COUNTER_MODE   

/*
    - baseline counter 
*/
int MPI_SEC_BaseCounter_Pipeline_Wait(MPI_Request *req, MPI_Status *status)
{

    int mpi_errno = MPI_SUCCESS;
    int var, j;
    int sendtype_sz, chopping_tag, i, recv_len, len;
    unsigned long next, datasize, dest;
    unsigned long count, index, decrypted_len;
    int totaldata;
    unsigned int segment_counter;
    int segments_no;
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;
    struct timeval t_s1, t_s2;

    unsigned int nonce_counter;
    int m, th_data, my_thread_no, th_start, temp_chunk;
    int cancel_req_no;
    int th_sum = 0;
    int flag;
    int pos, dec_pos;
    unsigned char iv_buffer[20];
    int ii, inner_totaldata;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
    int thisrank = comm_ptr->rank;

    index = *req;
    source = nonblock_req_handler[index].source;
    dest = nonblock_req_handler[index].dest;
    int tag = nonblock_req_handler[index].tag;
    //totaldata = nonblock_req_handler[index].totaldata;
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
                    // printf("    wait: source = %d i = %d no_tag = %d\n", source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
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

        totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);

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
        //printf(COLOR_BLUE"[rank=%d] wait for recv source=%d dest=%d totaldata=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest,totaldata);fflush(stdout);
        if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
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

        //   printf("index=%d source=%d dest=%d totaldat=%d segments_no=%d\n",index,source,dest,totaldata,segments_no);fflush(stdout);

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


#if DEBUG_BASE_COUNTER
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("[Rank = %d ]:basecounter wait  *req = %lu index = %d  totaldata = %lu source = %d dest = %d\n",
               rank, ((unsigned long)*req), index, totaldata, source, dest);
        fflush(stdout);
#endif

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
                printf(COLOR_RED "___Error in initialization Base counter wait___" COLOR_RESET "\n");

            if (1 != EVP_DecryptUpdate(global_counter_ctx, (buf + next), &decrypted_len, &Ideciphertext[index][COUNTER_HEADER_SIZE + next], inner_totaldata))
                printf(COLOR_RED "___Error in decryption___ Base counter wait " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
            else
            {
                printf(COLOR_BLUE "BaseCounter wait [rank %d]: Decrypted data is %d  next %d segment_counter %d segments_no %d" COLOR_RESET "\n",
                       thisrank, decrypted_len, next, segment_counter, segments_no);
            }
#endif
            next += inner_totaldata;
            recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
        }

#if PSC_THREAD_USE_DEBUG
        if (max_wait_thread_used < my_thread_no)
            max_wait_thread_used = my_thread_no;
#endif

#if ALL_COMM_PLUS_ENC_TIME
        gettimeofday(&t_s2, NULL);
        double t = (double)(t_s2.tv_usec - t_s1.tv_usec) / 1000000 + (double)(t_s2.tv_sec - t_s1.tv_sec);

        int isSamenode = 0;
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
        mpi_errno = MPI_Wait_original(req, &sta);
    }
    //printf(COLOR_YELLOW"[rank=%d] wait for recv is done source=%d dest=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest);fflush(stdout);
    return mpi_errno;
}

/*
* v4
*
*/
int MPI_SEC_PreComputeCounter_Pipeline_Wait_v4(MPI_Request *req, MPI_Status *status)
{

    int mpi_errno = MPI_SUCCESS;
    int var, j;
    int sendtype_sz, chopping_tag, i, recv_len, len;
    unsigned long next, datasize, dest;
    unsigned long count, index, decrypted_len;
    int totaldata;
    unsigned int segment_counter;
    int segments_no;
    MPI_Status sta;
    int source, rank;
    unsigned char *buf;
    struct timeval t_s1, t_s2;

    unsigned int nonce_counter;
    int m, th_data, th_start, temp_chunk;
    int cancel_req_no;
    int th_sum = 0;
    unsigned char iv_buffer[20];
    int ii, inner_totaldata;
    int recv_pos, mask_pos, dec_pos, flag;
    int common_counter_gen_sz;
    int temp_thread, t_counter_data, my_thread_no, dec_loc,pos,enc_data;
    long unsigned int t_counter;
    int decryption_mask;

    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
    int thisrank = comm_ptr->rank;

    index = *req;
    source = nonblock_req_handler[index].source;
    dest = nonblock_req_handler[index].dest;
    int tag = nonblock_req_handler[index].tag;
    //totaldata = nonblock_req_handler[index].totaldata;
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
                    // printf("    wait: source = %d i = %d no_tag = %d\n", source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
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

        ranksIrecvS[source] -= 1;
        totaldata = ((unsigned char)Ideciphertext[index][3] << 0) | ((unsigned char)Ideciphertext[index][2] << 8) | ((unsigned char)Ideciphertext[index][1] << 16) | ((unsigned char)Ideciphertext[index][0] << 24);

        choping_sz = ((unsigned char)Ideciphertext[index][24] << 0) | ((unsigned char)Ideciphertext[index][23] << 8) | ((unsigned char)Ideciphertext[index][22] << 16) | ((unsigned char)Ideciphertext[index][21] << 24);
        long unsigned int recv_counter = (((unsigned char)Ideciphertext[index][8] << 0) | ((unsigned char)Ideciphertext[index][7] << 8) | ((unsigned char)Ideciphertext[index][6] << 16) | ((unsigned char)Ideciphertext[index][5] << 24));
         char preCTRflag = Ideciphertext[index][4];
        //mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

        //printf(COLOR_BLUE"[rank=%d] wait for recv source=%d dest=%d totaldata=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest,totaldata);fflush(stdout);
        if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
        {
            segments_no = 1;
            segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
        }
        else
        {
            segments_no = 1;
        }

        //   printf("index=%d source=%d dest=%d totaldat=%d segments_no=%d\n",index,source,dest,totaldata,segments_no);fflush(stdout);

        if (segments_no != nonblock_req_handler[index].segment_number)
        {
            cancel_req_no = nonblock_req_handler[index].segment_number - segments_no;
            for (m = 0; m < cancel_req_no; m++)
            {
                int temp = nonblock_req_handler[index].total_request - 1 - m;
                // printf("rank %d post cancel for temp %d\n",comm_ptr->rank, temp);
                mpi_errno = MPI_Cancel(&nonblock_req_handler[index].request[temp]);
                mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[temp], &sta);
                ranksIrecvS[source] -= 1;
                // printf("   rank %d posted cancel done for temp %d\n",comm_ptr->rank, temp);
            }
        }

#if DEBUG_BASE_COUNTER
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //if(totaldata < SIXTY_4K)
        printf("[Rank = %d ]:precompute wait  *req = %lu index = %d  totaldata = %lu source = %d dest = %d recv_counter %lu\n",
               rank, ((unsigned long)*req), index, totaldata, source, dest,recv_counter);
        fflush(stdout);
#endif

        if (totaldata < SIXTY_4K)
        {
            MPI_Test(&nonblock_req_handler[index].request[1], &flag, &sta);

            if (!flag)
            {

                if (totaldata > 1024)
                {
                    common_counter_gen_sz = 512; //1024;
                    decryption_mask = 0;

                    while (!flag)
                    {
                        if(preCTRflag == '0')
                            memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                        else
                            memcpy(iv_buffer, &Recv_common_IV[source * 32+16], 16);    
                        IV_Count(iv_buffer, recv_counter);
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

                    if (!flag)
                        mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);

                   
                  /*
                    memcpy(iv_buffer, &Recv_common_IV[source * 32], 16); 
                    IV_Count(iv_buffer, recv_counter); 
                    EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(global_counter_ctx, dec_common_buffer, &len, zeros, totaldata);
                    decryption_mask = totaldata;
                    mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[1], &sta);   
                    */
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
                // printf("wait dec: rank %d source %d dest %d totaldata %d decryption_mask %d recv_counter %lu\n",
                //                    comm_ptr->rank, source, dest,totaldata,decryption_mask, recv_counter); fflush(stdout);
                decryption_common_counter_ivflag(buf, decryption_mask, totaldata, recv_pos, dec_pos, mask_pos,
                                          recv_counter, source, &Ideciphertext[index][0],preCTRflag);
                // printf("1. n %d decryption done data size %d\n",n, totaldata);
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
            //  memcpy(iv_buffer, &Recv_common_IV[source * 32+16], 16);
            // IV_Count(iv_buffer, recv_counter);
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

                mpi_errno = MPI_Wait_original(&nonblock_req_handler[index].request[segment_counter + 1], &sta);
                ranksIrecvS[source] -= 1;
                
                /*flag = 0;
                MPI_Test(&nonblock_req_handler[index].request[segment_counter + 1], &flag, &sta);
                while (!flag)
                {
                    generateCommonEncMask(4096);
                    MPI_Test(&nonblock_req_handler[index].request[segment_counter + 1], &flag, &sta);
                }*/
        
        
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
//#pragma omp parallel for default(none) schedule(static) private(m, pos, dec_pos, enc_data, t_counter, t_counter_data) shared(buf, Ideciphertext, th_data, local_counter_ctx, segments_no, stdout, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, recv_counter, index) num_threads(my_thread_no)
#pragma omp parallel default(none) private(m, pos, dec_pos, enc_data, t_counter, t_counter_data,iv_buffer, len) shared(buf, Ideciphertext, th_data, local_counter_ctx, segments_no, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, recv_counter, index,Recv_common_IV) num_threads(my_thread_no)
            {
            #pragma omp for schedule(static) nowait private(m, pos, dec_pos, enc_data, t_counter, t_counter_data,iv_buffer,len)
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

                    //multithread_decryption_common_counter_large_msg(buf, enc_data, pos, dec_pos, t_counter, &Ideciphertext[index][0], local_counter_ctx[m], source);
                    IV_Count_out(iv_buffer, t_counter,&Recv_common_IV[source * 32 + 16]);
                    EVP_EncryptInit_ex(local_counter_ctx[m], NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(local_counter_ctx[m], (buf+dec_pos), &len, (&Ideciphertext[index][0]+pos), enc_data);
                    
                    // multithread_decryption_common_counter(buf, t_mask, enc_data, pos, dec_pos, t_mask_pos, t_counter, large_recv_buffer, dec_common_buffer, local_counter_ctx[m], source);
                    //  multithread_decryption_common_counter(buf,  tinfo[m].compute_size, enc_data, pos, dec_pos, tinfo[m].tstart,  tinfo[m].counter, large_recv_buffer, dec_common_buffer,local_counter_ctx[m], source);
#if DEBUG_PRECOMPUTE_COUNTER
                    int tid = omp_get_thread_num();
                    //printf("tid = %d precompute recv: totaldata =%d enc_data =%d m =%d common_counter_gen_sz %d t_mask %d t_mask_pos %d t_counter %lu\n",
                    //       tid, totaldata, enc_data, m, common_counter_gen_sz, t_mask, t_mask_pos, t_counter);
                    printf("tid = %d precompute wait: totaldata =%d enc_data =%d m =%d  t_counter %lu recv_counter %lu pos %d dec_pos %d segment_counter %d segments_no %d\n",
                           tid, totaldata, enc_data, m, t_counter, recv_counter, pos, dec_pos, segment_counter, segments_no);
                    //fflush(stdout);
#endif
                } // end of for
            } // end of parallel region
        
               
                recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
                recv_pos += inner_totaldata;
                dec_loc += inner_totaldata;
            }
        }

#if PSC_THREAD_USE_DEBUG
        if (max_wait_thread_used < my_thread_no)
            max_wait_thread_used = my_thread_no;
#endif

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

    } // if source and dest are same
    else
    {
        mpi_errno = MPI_Wait_original(req, &sta);
    }
    //printf(COLOR_YELLOW"[rank=%d] wait for recv is done source=%d dest=%d"COLOR_RESET"\n",comm_ptr->rank,source,dest);fflush(stdout);
    return mpi_errno;
}

#endif 
/*
  End of counter mode
*/


int MPI_Wait(MPI_Request *request, MPI_Status *status)
{

    int mpi_errno = MPI_SUCCESS;
    int req_index = *request;

    if ((req_index >= 0 && req_index < ISEND_REQ) && nonblock_req_handler[req_index].req_type == 1)
    {
        mpi_errno = MPI_SEC_Wait_for_isend(request, status);
    }
    else if ((req_index >= 0 && req_index < ISEND_REQ) && nonblock_req_handler[req_index].req_type == 2)
    {

        if (security_approach == 600 || security_approach == 601) //(OPENMP_MULTI_THREAD_ONLY)
        {
            mpi_errno = MPI_SEC_Multi_Thread_irecv_openmp(request, status);
        }
        else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        {
            mpi_errno = MPI_SEC_MThreads_PipeLine_Wait_irecv_openmp_largeSegment_3(request, status);
        }

        else if (security_approach == 700 || security_approach == 701) //(BASE_COUNTER_MODE)
{
#if CRYPTMPI_COUNTER_MODE
            mpi_errno = MPI_SEC_BaseCounter_Pipeline_Wait(request, status);
#endif 
}
        else if (security_approach == 702) //(PRE_COMPUTE_COUNTER_MODE)
{
#if CRYPTMPI_COUNTER_MODE
                mpi_errno = MPI_SEC_PreComputeCounter_Pipeline_Wait_v4(request, status);
#endif                
}
        else
        {
            printf(COLOR_YELLOW "wait: Did not set security_approach" COLOR_RESET "\n");
            fflush(stdout);
        }
    }
    else
    {

        mpi_errno = MPI_Wait_original(request, status);
    }

    return mpi_errno;
}