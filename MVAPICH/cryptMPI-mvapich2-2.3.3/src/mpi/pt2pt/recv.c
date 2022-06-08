/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Recv */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Recv = PMPI_Recv
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Recv  MPI_Recv
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Recv as PMPI_Recv
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status) __attribute__((weak,alias("PMPI_Recv")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Recv
#define MPI_Recv PMPI_Recv

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Recv

/*@
    MPI_Recv - Blocking receive for a message

Output Parameters:
+ buf - initial address of receive buffer (choice) 
- status - status object (Status) 

Input Parameters:
+ count - maximum number of elements in receive buffer (integer) 
. datatype - datatype of each receive buffer element (handle) 
. source - rank of source (integer) 
. tag - message tag (integer) 
- comm - communicator (handle) 

Notes:
The 'count' argument indicates the maximum length of a message; the actual 
length of the message can be determined with 'MPI_Get_count'.  

.N ThreadSafe

.N Fortran

.N FortranStatus

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_TYPE
.N MPI_ERR_COUNT
.N MPI_ERR_TAG
.N MPI_ERR_RANK

@*/
/* Added by Abu Naser */
#if 1 //ENABLE_SECURE_MPI
int MPI_Recv_original(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                      MPI_Comm comm, MPI_Status *status)
#else
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
#endif
{
    static const char FCNAME[] = "MPI_Recv";
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPID_Request * request_ptr = NULL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_RECV);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_PT2PT_FUNC_ENTER_BACK(MPID_STATE_MPI_RECV);
    
    /* Validate handle parameters needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPIR_ERRTEST_COMM(comm, mpi_errno);
	    /* NOTE: MPI_STATUS_IGNORE != NULL */
	    MPIR_ERRTEST_ARGNULL(status, "status", mpi_errno);
	}
        MPID_END_ERROR_CHECKS;
    }
    
#   endif /* HAVE_ERROR_CHECKING */
    
    /* Convert MPI object handles to object pointers */
    MPID_Comm_get_ptr( comm, comm_ptr );

    /* Validate parameters if error checking is enabled */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPID_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            if (mpi_errno) goto fn_fail;
	    
	    MPIR_ERRTEST_COUNT(count, mpi_errno);
	    MPIR_ERRTEST_RECV_RANK(comm_ptr, source, mpi_errno);
	    MPIR_ERRTEST_RECV_TAG(tag, mpi_errno);
	    
	    /* Validate datatype handle */
	    MPIR_ERRTEST_DATATYPE(datatype, "datatype", mpi_errno);
	    
	    /* Validate datatype object */
	    if (HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN)
	    {
		MPID_Datatype *datatype_ptr = NULL;

		MPID_Datatype_get_ptr(datatype, datatype_ptr);
		MPID_Datatype_valid_ptr(datatype_ptr, mpi_errno);
		if (mpi_errno) goto fn_fail;
		MPID_Datatype_committed_ptr(datatype_ptr, mpi_errno);
		if (mpi_errno) goto fn_fail;
	    }
	    
	    /* Validate buffer */
	    MPIR_ERRTEST_USERBUFFER(buf,count,datatype,mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    /* MT: Note that MPID_Recv may release the SINGLE_CS if it
       decides to block internally.  MPID_Recv in that case will
       re-aquire the SINGLE_CS before returnning */
    mpi_errno = MPID_Recv(buf, count, datatype, source, tag, comm_ptr, 
			  MPID_CONTEXT_INTRA_PT2PT, status, &request_ptr);
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    if (request_ptr == NULL)
    {
	goto fn_exit;
    }
    
    /* If a request was returned, then we need to block until the request is 
       complete */
    if (!MPID_Request_is_complete(request_ptr))
    {
	MPID_Progress_state progress_state;
	    
	MPID_Progress_start(&progress_state);
        while (!MPID_Request_is_complete(request_ptr))
	{
	    /* MT: Progress_wait may release the SINGLE_CS while it
	       waits */
	    mpi_errno = MPID_Progress_wait(&progress_state);
	    if (mpi_errno != MPI_SUCCESS)
	    { 
		/* --BEGIN ERROR HANDLING-- */
		MPID_Progress_end(&progress_state);
		goto fn_fail;
		/* --END ERROR HANDLING-- */
	    }

            if (unlikely(MPIR_CVAR_ENABLE_FT &&
                        !MPID_Request_is_complete(request_ptr) &&
                        MPID_Request_is_anysource(request_ptr) &&
                        !MPID_Comm_AS_enabled(request_ptr->comm))) {
                /* --BEGIN ERROR HANDLING-- */
                MPID_Cancel_recv(request_ptr);
                MPIR_STATUS_SET_CANCEL_BIT(request_ptr->status, FALSE);
                MPIR_ERR_SET(request_ptr->status.MPI_ERROR, MPIX_ERR_PROC_FAILED, "**proc_failed");
                mpi_errno = request_ptr->status.MPI_ERROR;
                goto fn_fail;
                /* --END ERROR HANDLING-- */
            }
	}
	MPID_Progress_end(&progress_state);
    }

    mpi_errno = request_ptr->status.MPI_ERROR;
    MPIR_Request_extract_status(request_ptr, status);
    MPID_Request_release(request_ptr);

    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    /* ... end of body of routine ... */
    
  fn_exit:
    MPID_MPI_PT2PT_FUNC_EXIT_BACK(MPID_STATE_MPI_RECV);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_recv",
	    "**mpi_recv %p %d %D %i %t %C %p", buf, count, datatype, source, tag, comm, status);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/* Added by Abu Naser */
/**********************/

int MPI_SEC_Multi_Thread_Recv_OpenMP(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                                     MPI_Comm comm, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;

    unsigned long ciphertext_len, max_out_len;
    unsigned long decrypted_len = 0;

    int recvtype_sz, start, pos, m, i, s;
    int totaldata;
    unsigned int temp_data;
    MPI_Type_size(datatype, &recvtype_sz);

    mpi_errno = MPI_Recv_original(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, source, tag, comm, status);

    temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
    totaldata = (int)temp_data;
    long_msg_flag = totaldata;
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
    if (large_recv_buffer[20] == '1')
    {
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
    else
    {
        if (totaldata < SIXTY_4K)
            my_thread_no = 1;
        else
            my_thread_no = 2;
    }

#endif

#if OMP_DYNAMIC_THREADS
    if (my_thread_no > cyptmpi_series_thread)
        my_thread_no = cyptmpi_series_thread;
#endif

    int data = 0;
    data = ((unsigned char)large_recv_buffer[24] << 0) | ((unsigned char)large_recv_buffer[23] << 8) | ((unsigned char)large_recv_buffer[22] << 16) | ((unsigned char)large_recv_buffer[21] << 24);

    if (totaldata >= MULTI_THREAD_START_SIZE)
        mpi_errno = MPI_Recv_original(large_recv_buffer, totaldata + (my_thread_no * (12 + 16)), MPI_CHAR, source, tag, comm, status);
    else
    {
        mpi_errno = MPI_Recv_original(large_recv_buffer, totaldata + (my_thread_no * (12 + 16)), MPI_CHAR, source, tag, comm, status);
    }

    int segment_no, segment_counter;
#if 0
    segment_no = (totaldata - 1) / data + 1;

    if (!OMP_DYNAMIC_THREADS)
    {
        if (totaldata <= MULTI_THREAD_START_SIZE)
        {
            data = totaldata;
            segment_no = 1;
        }
    }
#endif
    segment_no = 1;
    // printf("  data at recv = %d segment_no=%d\n",data,segment_no);fflush(stdout);

#if OPENMP_THREADS
#pragma omp parallel for schedule(static) default(none) private(segment_counter, count, max_out_len, pos) shared(buf, large_recv_buffer, segment_no, data, stdout, global_openmp_ctx, totaldata) num_threads(my_thread_no)
#endif
    for (segment_counter = 0; segment_counter < segment_no; segment_counter++)
    {

        EVP_AEAD_CTX *local_ctx = global_openmp_ctx;

        max_out_len = data + 16;
        pos = segment_counter * (data + 12 + 16);
        int enc_data = data;

        if (segment_counter == segment_no - 1)
        {
            enc_data = totaldata - data * (segment_no - 1);
            max_out_len = enc_data + 16;
        }

        if (!EVP_AEAD_CTX_open(local_ctx, buf + segment_counter * data,
                               &count, enc_data,
                               &large_recv_buffer[pos], 12,
                               large_recv_buffer + pos + 12, (unsigned long)(max_out_len),
                               NULL, 0))
        {
            printf("[T = %d] Error in encryption: segment_counter=%d\n", omp_get_thread_num(), segment_counter);
            fflush(stdout);
        }
#if OMP_DEBUG
        else
        {
            printf(COLOR_BLUE "  [T = %d] Decrypted data is %d in segment_counter = %d" COLOR_RESET "\n", omp_get_thread_num(), count, segment_counter);
            fflush(stdout);
        }
#endif
    }

    return mpi_errno;
}

int MPI_SEC_MThreads_Pipelined_OpenMP_Recv_largeSegment_3(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                                                          MPI_Comm comm, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned char newkey[40], V[50];
    unsigned long next, decrypted_len, dest;
    MPI_Status sta;
    int i, recv_len, segments_no, len, recvtype_sz, flag;
    MPI_Type_size(datatype, &recvtype_sz);
    int totaldata = 0;
    unsigned int temp_data;
    unsigned int segment_counter, nonce_counter;
    int th_data, th_pos, th_start, th_sum, m, temp_chunk;
    int my_thread_no;
    MPI_Request request[300];
    int pos, dec_pos;
    unsigned char local_nonce[20];

    volatile EVP_AEAD_CTX *local_ctx = NULL;
    EVP_AEAD_CTX *segment_ctx[100]; // total threads will not excede 100

    MPID_Comm *comm_ptr = NULL;

    //printf("recv: Thread# %d omp_get_team_size %d omp_get_ancestor_thread_num %d omp_get_level %d omp_get_max_active_levels %d\n",
    //omp_get_thread_num(),omp_get_team_size(1), omp_get_ancestor_thread_num(1),omp_get_level(),omp_get_max_active_levels());

    /* Received Header bytes  for data size with original tag */
    /* Check for Probe */
    if (Crypthandle_probe[source].no_tag > 0)
    {
        flag = 0;
        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
        {
            if (Crypthandle_probe[source].tags[i] == tag)
            {
                memcpy(large_recv_buffer, &Crypthandle_probe[source].header[i][0], MSG_HEADER_SIZE);
                flag = 1;
                Crypthandle_probe[source].tags[i] = -1;
                tag++;
                //printf("recv: source = %d i = %d no_tag = %d\n",source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
                break;
            }
        }

        if (flag == 0)
            mpi_errno = MPI_Recv_original(large_recv_buffer, (recvtype_sz * count) + MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &sta);
    }
    else
    {
        mpi_errno = MPI_Recv_original(large_recv_buffer, (recvtype_sz * count) + MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &sta);
    }
    temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
    totaldata = (int)temp_data;

    long_msg_flag = totaldata;

    choping_sz = ((unsigned char)large_recv_buffer[24] << 0) | ((unsigned char)large_recv_buffer[23] << 8) | ((unsigned char)large_recv_buffer[22] << 16) | ((unsigned char)large_recv_buffer[21] << 24);
    //  printf(COLOR_YELLOW"openmp: recv:choping_sz =%d flag=%c totaldata=%lu "COLOR_RESET "\n",choping_sz, large_recv_buffer[20],totaldata); fflush(stdout);

    local_ctx = global_small_msg_ctx;

#if ENABLE_SECURE_MPI_DEBUG
    MPID_Comm_get_ptr(comm, comm_ptr);
    printf("MPI_SEC_Chopping_Recv[rank=%d]: totaldata will receive from sender =%lu choping_sz=%d\n", comm_ptr->rank, totaldata, choping_sz);
    fflush(stdout);
#endif

    nonce_counter = 0;
    if (totaldata <= PIPELINE_SIZE && (large_recv_buffer[20] != '3' && large_recv_buffer[20] != '4'))
    {
        segment_counter = 0;
        my_thread_no = 1;
        th_data = totaldata;

        //th_start = 0;

        m = my_thread_no - 1;

        mpi_errno = MPI_Irecv_original(&large_recv_buffer[MSG_HEADER_SIZE], totaldata + 16 + NONCE_HEADER, MPI_CHAR, source,
                                       tag++, comm, &request[1]);

        mpi_errno = MPI_Wait_original(&request[1], &sta);

        if (!EVP_AEAD_CTX_open(local_ctx, buf,
                               &decrypted_len, totaldata,
                               &large_recv_buffer[4], 12,
                               &large_recv_buffer[MSG_HEADER_SIZE + NONCE_HEADER], totaldata + 16,
                               NULL, 0))
        {
            printf("OMP_DEBUG: Decryption error: Chooping Recv\n");
            fflush(stdout);
        }
#if OMP_DEBUG
        else
        {
            printf("OMP_DEBUG: Decrypted data is %d in process = %d\n", decrypted_len, m);
            fflush(stdout);
        }
#endif
    }
    else
    {
        if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
        {
            segments_no = 1;
            segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
        }
        else
        {
            segments_no = 1;
        }

        count = 0;
        th_pos = 0;

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

#elif OMP_DYNAMIC_THREADS_PIPELINE

        if (totaldata < SIXTY_4K)
            my_thread_no = 1;
        else if (totaldata < ONE_TWO_8K)
            my_thread_no = 2;
        else if (totaldata < FIVE_ONE_2K)
            my_thread_no = 4;
        else if (totaldata >= FIVE_ONE_2K)
            my_thread_no = 8;

#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
        if (totaldata < SIXTY_4K)
            my_thread_no = 1;
        else if (totaldata < TWO_M)
            my_thread_no = 2;
        else
            my_thread_no = 8;
#endif
#if OMP_DYNAMIC_THREADS

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

        th_data = choping_sz;
        //  printf("th_data = %d and recv_len = %d segments_no=%d\n",th_data, recv_len,segments_no);fflush(stdout);
        th_start = MSG_HEADER_SIZE;

        if (large_recv_buffer[20] == '3' || large_recv_buffer[20] == '4')
        {
            mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], totaldata + MAX_OMP_THREADS_LIMIT * (ENC_MSG_TAG_SIZE + NONCE_HEADER), MPI_UNSIGNED_CHAR, source,
                                           tag++, comm, &request[1]);
        }
        else
        {
            int inside_segments = (PIPELINE_SIZE - 1) / choping_sz + 1;

            for (m = 0; m < segments_no - 1; m++)
            {
                mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], PIPELINE_SIZE + inside_segments * (16 + NONCE_HEADER), MPI_UNSIGNED_CHAR, source,
                                               tag++, comm, &request[m]);
                th_start += PIPELINE_SIZE + inside_segments * (16 + NONCE_HEADER);
            }

            mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], totaldata - PIPELINE_SIZE * (segments_no - 1) + inside_segments * (16 + NONCE_HEADER), MPI_UNSIGNED_CHAR, source,
                                           tag++, comm, &request[m]);
        }

        int active_thread_num = my_thread_no; //omp_get_num_threads();
        int ii;

#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
        if (large_recv_buffer[20] == '3' || large_recv_buffer[20] == '4')
        {
            th_data = choping_sz;
            mpi_errno = MPI_Wait_original(&request[1], &sta);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
            //  ranksIrecvS[source] -= 1;
#endif
            // printf("wait to get data done\n");fflush(stdout);
            pos = 0;
            int new_segments;
            new_segments = (totaldata - 1) / th_data + 1;

            if ((totaldata > SUBKEY_GEN_START))
            {
                memcpy(V, &large_recv_buffer[4], 16);
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

//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, large_recv_buffer, th_data, segment_ctx, new_segments, local_ctx, stdout, totaldata) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter) shared(buf, large_recv_buffer, th_data, segment_ctx, new_segments, local_ctx, stdout, totaldata) num_threads(my_thread_no)
            {
#pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos, segment_counter)

                for (segment_counter = 0; segment_counter < new_segments; segment_counter++)
                {

                    pos = MSG_HEADER_SIZE + segment_counter * (th_data + NONCE_HEADER + ENC_MSG_TAG_SIZE) + NONCE_HEADER;
                    int enc_data = th_data;

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
                    local_nonce[7] = large_recv_buffer[pos - 5];
                    local_nonce[8] = large_recv_buffer[pos - 4];
                    local_nonce[9] = large_recv_buffer[pos - 3];
                    local_nonce[10] = large_recv_buffer[pos - 2];
                    local_nonce[11] = large_recv_buffer[pos - 1];

                    nonce_counter = ((unsigned char)large_recv_buffer[pos - 1] << 0) | ((unsigned char)large_recv_buffer[pos - 2] << 8) | ((unsigned char)large_recv_buffer[pos - 3] << 16) | ((unsigned char)large_recv_buffer[pos - 4] << 24);

                    dec_pos = nonce_counter * th_data;

                    // printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d %u %u %u %u => %u pos=%d\n",
                    // omp_get_thread_num(),segment_counter,m,ii,th_data,nonce_counter,dec_pos,large_recv_buffer[pos-1],large_recv_buffer[pos-2],large_recv_buffer[pos-3],large_recv_buffer[pos-4],0,pos );fflush(stdout);
                    if (!EVP_AEAD_CTX_open(segment_ctx[segment_counter], buf + dec_pos,
                                           &decrypted_len, enc_data,
                                           local_nonce, 12,
                                           &large_recv_buffer[pos], enc_data + 16,
                                           NULL, 0))
                    {
                        printf(COLOR_RED "OpenMP Pipeline: Decryption error:  thread %d nonce_counter %d segment_counter %d new_segments=%d th_data=%d" COLOR_RESET "\n",
                               omp_get_thread_num(), nonce_counter, segment_counter, new_segments, th_data);
                        fflush(stdout);
                    }
#if OMP_DEBUG_DYNAMIC_THREAD_PIPELINE
                    else
                    {
                        printf(COLOR_BLUE "  wait: OpenMP Dynamic Thread Pipeline: Decrypted data is %d in thread = %d nonce_counter=%d" COLOR_RESET "\n",
                               decrypted_len, omp_get_thread_num(), nonce_counter);
                        fflush(stdout);
                    }
#endif
                } // end  loop
            }     // end of parallel region
        }
        else
        {
#endif

#if OMP_DYNAMIC_THREADS_PIPELINE
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
                if (PIPELINE_SIZE < SIXTY_4K)
                    temp_thread = 1;
                else if (PIPELINE_SIZE < ONE_TWO_8K)
                    temp_thread = 2;
                else if (PIPELINE_SIZE < FIVE_ONE_2K)
                    temp_thread = 4; //4;
                else if (PIPELINE_SIZE >= FIVE_ONE_2K)
                    temp_thread = 8;
#endif
                if (temp_thread > cyptmpi_series_thread)
                    temp_thread = cyptmpi_series_thread;
                my_thread_no = temp_thread;
/* Switch Between CryptMPI-R and CryptMPI-C */
#if ENABLE_CRYPTMPI_RC_SWITCH
                if (omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
                    my_thread_no = 1;
#endif
#if CRYPTMPI_SINGLE_THREAD_MODE
                my_thread_no = 1;
#endif
            }

#endif

            if ((totaldata > SUBKEY_GEN_START))
            {
                memcpy(V, &large_recv_buffer[4], 16);
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

            int prsd_segment = 0;
            int inner_totaldata = 0;
            int enc_loc = 0;
            int recv_loc = 0;
            for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
            {

                mpi_errno = MPI_Wait_original(&request[segment_counter], &sta);
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
//#if OPENMP_DYNAMIC_SCH
//#pragma omp parallel for default(none) schedule(dynamic) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, large_recv_buffer, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
//#elif OPENMP_STATIC_SCH
//#pragma omp parallel for default(none) schedule(static) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, large_recv_buffer, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, large_recv_buffer, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
                {
#pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos)

                    //#elif OPENMP_GUIDED_SCH
                    //#pragma omp parallel for default(none) schedule(guided) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, large_recv_buffer, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
                    //#else
                    //#pragma omp parallel for default(none) private(nonce_counter, local_nonce, decrypted_len, mpi_errno, m, pos, dec_pos) shared(buf, large_recv_buffer, th_data, segment_ctx, segments_no, local_ctx, stdout, ii, segment_counter, totaldata, enc_loc, inner_totaldata, recv_loc) num_threads(my_thread_no)
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
                        local_nonce[7] = large_recv_buffer[pos - 5];
                        local_nonce[8] = large_recv_buffer[pos - 4];
                        local_nonce[9] = large_recv_buffer[pos - 3];
                        local_nonce[10] = large_recv_buffer[pos - 2];
                        local_nonce[11] = large_recv_buffer[pos - 1];

                        nonce_counter = ((unsigned char)large_recv_buffer[pos - 1] << 0) | ((unsigned char)large_recv_buffer[pos - 2] << 8) | ((unsigned char)large_recv_buffer[pos - 3] << 16) | ((unsigned char)large_recv_buffer[pos - 4] << 24);

                        dec_pos = recv_loc + m * th_data;

                        int enc_data = th_data;
                        if (m == ii - 1)
                        {
                            enc_data = inner_totaldata - th_data * (ii - 1);
                        }

                        /*printf("    [Thread id =%d] omp-recv:segment_counter=%d m=%d ii=%d th_data=%d nonce_counter=%d dec_pos=%d %u %u %u %u => %u pos=%d prsd_segment=%d\n",
                               omp_get_thread_num(), segment_counter, m, ii, th_data, nonce_counter, dec_pos, large_recv_buffer[pos - 1], large_recv_buffer[pos - 2], large_recv_buffer[pos - 3], large_recv_buffer[pos - 4], 0, pos,prsd_segment);
                        fflush(stdout);*/
                        if (!EVP_AEAD_CTX_open(segment_ctx[omp_get_thread_num()], buf + dec_pos,
                                               &decrypted_len, enc_data,
                                               local_nonce, 12,
                                               &large_recv_buffer[pos], enc_data + 16,
                                               NULL, 0))
                        {
                            printf(COLOR_RED "OpenMP Pipeline: Decryption error:  thread %d nonce_counter %d segment_counter %d" COLOR_RESET "\n", omp_get_thread_num(), nonce_counter, segment_counter);
                            fflush(stdout);
                        }
#if OMP_DEBUG
                        else
                        {
                            printf(COLOR_BLUE "OMP_DEBUG: Decrypted data is %d in thread = %d" COLOR_RESET "\n", decrypted_len, omp_get_thread_num());
                            fflush(stdout);
                        }
#endif
                    }
                } // end of parallel region
                prsd_segment += ii;
                enc_loc += inner_totaldata + (ENC_MSG_TAG_SIZE + NONCE_HEADER) * ii;
                recv_loc += inner_totaldata;
            }

#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
        } // end of else
#endif
        if ((totaldata > SUBKEY_GEN_START))
        {
            for (i = 0; i < my_thread_no; i++)
                EVP_AEAD_CTX_free(segment_ctx[i]);
        }
    }

    return mpi_errno;
}

/* Base counter */
int MPI_SEC_BaseCounter_Pipeline_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                                      MPI_Comm comm, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned char iv_buffer[20];
    int decrypted_len, next, dest;
    MPI_Status sta;
    int i, recv_len, segments_no, len, recvtype_sz;
    MPI_Type_size(datatype, &recvtype_sz);
    unsigned int totaldata = 0;
    unsigned int segment_counter = 0;
    uint32_t n, c;
    int m, th_start, th_data, inner_totaldata, ii;
    MPI_Request request[300];

    /* Received Header bytes */
    mpi_errno = MPI_Recv_original(large_recv_buffer, COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &sta);

    totaldata = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
    //memcpy(CTR_IV, &large_recv_buffer[4], 16);
    long unsigned int recv_counter = (((unsigned char)large_recv_buffer[8] << 0) | ((unsigned char)large_recv_buffer[7] << 8) | ((unsigned char)large_recv_buffer[6] << 16) | ((unsigned char)large_recv_buffer[5] << 24));

    if (BASE_COUNTER_LIBRARY_NONCE)
    {
        memcpy(iv_buffer, &large_recv_buffer[4], 16);
        recv_counter = 0;
    }
    else
    {
        memcpy(iv_buffer, &Recv_common_IV[source * 16], 16);
    }

    choping_sz = ((unsigned char)large_recv_buffer[24] << 0) | ((unsigned char)large_recv_buffer[23] << 8) | ((unsigned char)large_recv_buffer[22] << 16) | ((unsigned char)large_recv_buffer[21] << 24);

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

#if DEBUG_BASE_COUNTER
    MPID_Comm *comm_ptr_debug = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr_debug);
    int thisrank = comm_ptr_debug->rank;
    //printf(COLOR_YELLOW " [Rank %d] MPI_SEC_BaseCounter_Pipeline_Recv totaldata %d segments_no %d choping_sz %d" COLOR_RESET "\n",
    //       thisrank, totaldata, segments_no, choping_sz);
    //fflush(stdout);
#endif

    th_start = COUNTER_HEADER_SIZE;
    if (totaldata <= PIPELINE_SIZE)
    {
        mpi_errno = MPI_Irecv_original(large_recv_buffer + th_start, totaldata, MPI_UNSIGNED_CHAR, source,
                                       tag++, comm, &request[0]);
        mpi_errno = MPI_Wait_original(&request[0], &sta);

        /* incresed counter for optimized nonce gen */
        if (!BASE_COUNTER_LIBRARY_NONCE)
            IV_Count(iv_buffer, recv_counter);

        if (1 != EVP_DecryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer))
            printf(COLOR_RED "___Error in initialization Base counter recv___" COLOR_RESET "\n");

        if (1 != EVP_DecryptUpdate(global_counter_ctx, buf, &decrypted_len, large_recv_buffer + COUNTER_HEADER_SIZE, totaldata))
            printf(COLOR_RED "___Error in decryption___ Base counter recv " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
        else
        {
            printf(COLOR_BLUE "Base counter recv[rank %d]: Decrypted data is %d  totaldata %d segments_no %d" COLOR_RESET "\n",
                   thisrank, decrypted_len, totaldata, segments_no);
        }
#endif
    }
    else
    {
        for (m = 0; m < segments_no - 1; m++)
        {
            mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], PIPELINE_SIZE, MPI_UNSIGNED_CHAR, source,
                                           tag++, comm, &request[m]);
            th_start += PIPELINE_SIZE;
        }

        mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], totaldata - PIPELINE_SIZE * (segments_no - 1), MPI_UNSIGNED_CHAR, source,
                                       tag++, comm, &request[m]);

        next = 0;
        for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
        {

            mpi_errno = MPI_Wait_original(&request[segment_counter], &sta);
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
                printf(COLOR_RED "___Error in initialization Base counter recv___" COLOR_RESET "\n");

            if (1 != EVP_DecryptUpdate(global_counter_ctx, buf + next, &decrypted_len, &large_recv_buffer[COUNTER_HEADER_SIZE + next], inner_totaldata))
                printf(COLOR_RED "___Error in decryption___ Base counter recv " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
            else
            {
                printf(COLOR_BLUE "Base counter recv[rank %d]: Decrypted data is %d  next %d segment_counter %d segments_no %d" COLOR_RESET "\n",
                       thisrank, decrypted_len, next, segment_counter, segments_no);
            }
#endif
            next += inner_totaldata;
            recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
        }
    }

    return mpi_errno;
}
void multithread_decryption_common_counter_large_msg(void *buf, int datasz, int recv_pos, int dec_pos, unsigned long dec_counter, 
                            unsigned char * in, EVP_CIPHER_CTX * t_counter_ctx, int source)
{
    unsigned char iv_buffer[20];
    int len;
   
        IV_Count_out(iv_buffer, dec_counter,&Recv_common_IV[source * 32 + 16]);
        EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
        EVP_EncryptUpdate(t_counter_ctx, (buf+dec_pos), &len, (in+recv_pos), datasz);
     //  printf("datasz %d dec_counter %d tempdata %d pre_com_dec_mask_sz %d\n",datasz,dec_counter,tempdata, pre_com_dec_mask_sz );fflush(stdout);
    return;
}
void decryption_common_counter_ivflag(void *buf, int pre_com_dec_mask_sz, int datasz, int recv_pos, int dec_pos, int mask_pos,
                                      unsigned long dec_counter, int source, unsigned char *in_buf, char pflag)
{
    unsigned char *enc_buffer, *in_buffer, *out_buffer;
    unsigned char iv_buffer[20];
    long unsigned int v1, v2, n;
    int len, i, tempdata;
    // printf("datasz %d *dec_counter %lu pre_com_dec_mask_sz %d\n",datasz,dec_counter,pre_com_dec_mask_sz);fflush(stdout);
    if (datasz > pre_com_dec_mask_sz)
    {
        len = pre_com_dec_mask_sz;
        datasz = datasz - pre_com_dec_mask_sz;
    }
    else
    {
        len = datasz;
        datasz = 0;
    }
    tempdata = 0;

    if (len)
    {
        tempdata = len;
        //*pre_com_dec_mask_sz = *pre_com_dec_mask_sz - len;
        enc_buffer = (unsigned char *)(dec_common_buffer + mask_pos);
        //in_buffer = (unsigned char *)(large_recv_buffer + recv_pos);
        in_buffer = (unsigned char *)(in_buf + recv_pos);
        out_buffer = (unsigned char *)(buf + dec_pos);

        while (len >= 16)
        {
            for (n = 0; n < 16; n += sizeof(long unsigned int))
            {
                memcpy(&v1, enc_buffer + n, sizeof(long unsigned int));
                memcpy(&v2, in_buffer + n, sizeof(long unsigned int));
                v1 = v1 ^ v2;
                memcpy(out_buffer + n, &v1, sizeof(long unsigned int));
            }
            len -= 16;
            enc_buffer += 16;
            in_buffer += 16;
            out_buffer += 16;
            n = 0;
        }
        if (len)
        {
            i = tempdata - len;
            while (len--)
            {
                *((unsigned char *)(buf + dec_pos + i)) = (unsigned char)(dec_common_buffer[i + mask_pos] ^ in_buf[i + recv_pos]);
                i++;
            }
        }
    }

    if (datasz)
    {
        if (pflag == '0')
            memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
        else
            memcpy(iv_buffer, &Recv_common_IV[source * 32 + 16], 16);
        IV_Count(iv_buffer, dec_counter);
        EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
        EVP_EncryptUpdate(global_counter_ctx, (buf + dec_pos + tempdata), &len, (in_buf + recv_pos + tempdata), datasz);
        //*dec_counter += (unsigned long)((datasz - 1) / 16 + 1);
        //printf("datasz %d dec_counter %d tempdata %d pre_com_dec_mask_sz %d\n",datasz,dec_counter,tempdata, pre_com_dec_mask_sz );fflush(stdout);
    }

    return;
}

int MPI_SEC_PreComputeCounter_Recv_v4(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                                      MPI_Comm comm, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned long next, decrypted_len, dest;
    MPI_Status sta;
    int i, recv_len, segments_no, len, recvtype_sz, flag;
    MPI_Type_size(datatype, &recvtype_sz);
    int totaldata = 0;
    unsigned int temp_data;
    unsigned int segment_counter, nonce_counter;
    int th_data, th_pos, th_start, th_sum, m, temp_chunk;
    int my_thread_no;
    MPI_Request request[300];
    int pos, request_counter;

    unsigned long common_recv_counter;
    unsigned char iv_buffer[20 * MAX_OMP_THREADS_LIMIT];
    unsigned char *enc_buffer, *in_buffer, *out_buffer;
    long unsigned int v1, v2, n;
    int recv_pos, mask_pos, dec_pos, inner_totaldata, ii, common_counter_gen_sz, dec_loc, enc_data;
    int j, temp2;
    int dec_cmn_start, dec_cmn_end;
    int t_mask, t_mask_pos, t_counter_data, gen_size, decryption_mask;
    long unsigned int t_counter;

    dec_cmn_end = dec_cmn_start = request_counter = 0;

#if 1 //DEBUG_PRECOMPUTE_COUNTER
    MPID_Comm *comm_ptr_debug = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr_debug);
    int thisrank = comm_ptr_debug->rank;
#endif

    /* Received Header bytes  for data size with original tag */
    /* Check for Probe */
    if (Crypthandle_probe[source].no_tag > 0)
    {
        flag = 0;
        for (i = 1; i <= Crypthandle_probe[source].no_tag; i++)
        {
            if (Crypthandle_probe[source].tags[i] == tag)
            {
                memcpy(large_recv_buffer, &Crypthandle_probe[source].header[i][0], MSG_HEADER_SIZE);
                flag = 1;
                Crypthandle_probe[source].tags[i] = -1;
                tag++;
                //printf("recv: source = %d i = %d no_tag = %d\n",source, i, Crypthandle_probe[source].no_tag);fflush(stdout);
                break;
            }
        }

        if (flag == 0)
            mpi_errno = MPI_Recv_original(large_recv_buffer, (recvtype_sz * count) + COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &sta);
    }
    else
    {
        mpi_errno = MPI_Recv_original(large_recv_buffer, (recvtype_sz * count) + COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, source, tag++, comm, &sta);
    }

    temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
    totaldata = (int)temp_data;

    long_msg_flag = totaldata;
    choping_sz = ((unsigned char)large_recv_buffer[24] << 0) | ((unsigned char)large_recv_buffer[23] << 8) | ((unsigned char)large_recv_buffer[22] << 16) | ((unsigned char)large_recv_buffer[21] << 24);
    common_recv_counter = (((unsigned char)large_recv_buffer[8] << 0) | ((unsigned char)large_recv_buffer[7] << 8) | ((unsigned char)large_recv_buffer[6] << 16) | ((unsigned char)large_recv_buffer[5] << 24));
    unsigned int temp_recv_counter = common_recv_counter;

    //if (totaldata <= PIPELINE_SIZE && (large_recv_buffer[20] != '3' && large_recv_buffer[20] != '4'))
    if (totaldata < PRE_COM_DATA_RANGE)
    {
#if DEBUG_PRECOMPUTE_COUNTER
        printf("    rank = %d precompute recv: totaldata =%d choping_sz =%d common_recv_counter =%lu\n",
               thisrank, totaldata, choping_sz, common_recv_counter);
        fflush(stdout);
#endif

        char preCTRflag = large_recv_buffer[4];
        mpi_errno = MPI_Irecv_original(large_recv_buffer, totaldata, MPI_CHAR, source, tag++, comm, &request[request_counter++]);
        MPI_Test(&request[0], &flag, &sta);
        //mpi_errno = MPI_Wait_original(&request[0], &sta);

        if (!flag)
        {
            decryption_mask = 0;

            if (totaldata > 1024)
            {
                common_counter_gen_sz = 512;

                my_thread_no = 1;

                if (my_thread_no == 1)
                {
                    while (!flag)
                    {
                        //memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                        //IV_Count(iv_buffer, common_recv_counter);
                        if (preCTRflag == '0')
                            IV_Count_out(iv_buffer, common_recv_counter, &Recv_common_IV[source * 32]);
                        else
                            IV_Count_out(iv_buffer, common_recv_counter, &Recv_common_IV[source * 32 + 16]);
                        EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                        EVP_EncryptUpdate(global_counter_ctx, &dec_common_buffer[decryption_mask], &len, zeros, common_counter_gen_sz);
                        decryption_mask += common_counter_gen_sz;
                        common_recv_counter += (unsigned long)((common_counter_gen_sz - 1) / 16 + 1);
                        /*printf("wait: rank %d source %d dest %d totaldata %d decryption_mask %d\n",
                                    comm_ptr->rank, source, dest,totaldata,decryption_mask); fflush(stdout);*/

                        if (decryption_mask >= totaldata)
                            break;

                        MPI_Test(&request[0], &flag, &sta);
                    }
                }
                //#if 0
                else
                {

                    while (!flag)
                    {
#pragma omp parallel default(none) private(m, iv_buffer, len) shared(my_thread_no, dec_common_buffer, local_counter_ctx, stdout, totaldata, source, common_recv_counter, decryption_mask, Recv_common_IV, common_counter_gen_sz, zeros, preCTRflag) num_threads(my_thread_no)
                        {
#pragma omp for schedule(static) nowait private(m, iv_buffer, len)
                            for (m = 0; m < my_thread_no; m++)
                            {
                                unsigned long int t_recv_counter;
                                if (m > 0)
                                {
                                    t_recv_counter = common_recv_counter + ((unsigned long)((common_counter_gen_sz * m - 1) / 16 + 1));
                                }
                                else
                                {
                                    t_recv_counter = common_recv_counter;
                                }

                                if (preCTRflag == '0')
                                    IV_Count_out(iv_buffer, t_recv_counter, &Recv_common_IV[source * 32]);
                                else
                                    IV_Count_out(iv_buffer, t_recv_counter, &Recv_common_IV[source * 32 + 16]);
                                EVP_EncryptInit_ex(local_counter_ctx[m], NULL, NULL, NULL, iv_buffer);
                                EVP_EncryptUpdate(local_counter_ctx[m], &dec_common_buffer[decryption_mask + m * common_counter_gen_sz], &len, zeros, common_counter_gen_sz);

                                /*printf("wait: rank %d source %d dest %d totaldata %d decryption_mask %d\n",
                                    comm_ptr->rank, source, dest,totaldata,decryption_mask); fflush(stdout);*/
                            }
                        }
                        decryption_mask += (common_counter_gen_sz * my_thread_no);
                        common_recv_counter += (unsigned long)((common_counter_gen_sz * my_thread_no - 1) / 16 + 1);
                        if (decryption_mask >= totaldata)
                            break;

                        MPI_Test(&request[0], &flag, &sta);
                    }
                }
                //#endif

                if (!flag)
                    mpi_errno = MPI_Wait_original(&request[0], &sta);
            }
            else
            {
                if (preCTRflag == '0')
                    memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
                else
                    memcpy(iv_buffer, &Recv_common_IV[source * 32 + 16], 16);
                IV_Count(iv_buffer, common_recv_counter);
                EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                EVP_EncryptUpdate(global_counter_ctx, dec_common_buffer, &len, zeros, totaldata);
                decryption_mask = totaldata;
                mpi_errno = MPI_Wait_original(&request[0], &sta);
            }

            mask_pos = 0;
            recv_pos = 0;
            dec_pos = 0;
            /* printf("recv dec: thisrank %d source %d  totaldata %d decryption_mask %d common_recv_counter %lu\n",
                                    thisrank, source, totaldata,decryption_mask, common_recv_counter); fflush(stdout);*/
            if (decryption_mask < totaldata)
            {
                decryption_mask = 0;
                common_recv_counter = temp_recv_counter;
            }
            decryption_common_counter_ivflag(buf, decryption_mask, totaldata, recv_pos, dec_pos, mask_pos,
                                             common_recv_counter, source, large_recv_buffer, preCTRflag);
            //my_thread_no = 2;
        }
        else
        {
            if (preCTRflag == '0')
                memcpy(iv_buffer, &Recv_common_IV[source * 32], 16);
            else
                memcpy(iv_buffer, &Recv_common_IV[source * 32 + 16], 16);
            IV_Count(iv_buffer, common_recv_counter);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, buf, &len, large_recv_buffer, totaldata);
            // printf("2. decryption done data size %d\n", totaldata);
        }
    }
    else
    {
        if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
        {
            segments_no = 1;
            segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
        }
        else
        {
            segments_no = 1;
        }

        /* because of too many request, no pipeline */
        if (large_recv_buffer[20] == '3' || large_recv_buffer[20] == '4')
            segments_no = 1;

        th_start = 0;
        for (m = 0; m < segments_no - 1; m++)
        {
            mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], PIPELINE_SIZE, MPI_CHAR, source,
                                           tag++, comm, &request[request_counter++]);
            th_start += PIPELINE_SIZE;
        }
        mpi_errno = MPI_Irecv_original(&large_recv_buffer[th_start], totaldata - PIPELINE_SIZE * (segments_no - 1), MPI_CHAR, source,
                                       tag++, comm, &request[request_counter++]);

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
        //  if (totaldata < THIRTY_2K)
        //    my_thread_no = 1;
        if (totaldata < SIXTY_4K)
            my_thread_no = 1;
        else if (totaldata < TWO_FIVE_6K)
            my_thread_no = 8;
        else if (totaldata >= TWO_FIVE_6K)
            my_thread_no = 12; //16;
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

#if DEBUG_PRECOMPUTE_COUNTER
        printf("rank = %d precompute recv: totaldata =%d choping_sz =%d common_recv_counter =%lu common_counter_gen_sz %d\n",
               thisrank, totaldata, choping_sz, common_recv_counter, common_counter_gen_sz);
        fflush(stdout);
#endif

        common_counter_gen_sz = 0;
        dec_cmn_start = 0;
        dec_cmn_end = 0;
        recv_pos = 0;
        dec_loc = 0;
        mask_pos = 0;
        int temp_totaldata = totaldata;
        int so_far_done = 0;
        long unsigned int old_common_recv_counter = common_recv_counter;
        // long unsigned int temp_common_recv_counter = common_recv_counter;
        for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
        {

            mpi_errno = MPI_Wait_original(&request[segment_counter], &sta);

            /* encryption size for each thread */
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

#if DEBUG_PRECOMPUTE_COUNTER
            printf("rank = %d precompute recv: totaldata =%d choping_sz =%d common_recv_counter =%lu common_counter_gen_sz %d\n",
                   thisrank, totaldata, choping_sz, common_recv_counter, common_counter_gen_sz);
            fflush(stdout);
#endif

//#pragma omp parallel for default(none) schedule(static) private( m, pos, dec_pos, enc_data, t_mask, t_mask_pos, t_counter,t_counter_data) shared(buf, large_recv_buffer,dec_common_buffer, th_data, local_counter_ctx, segments_no, stdout, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, common_recv_counter, common_counter_gen_sz, mask_pos) num_threads(my_thread_no)
#pragma omp parallel default(none) private(m, pos, dec_pos, enc_data, t_mask, t_mask_pos, t_counter, t_counter_data, iv_buffer, len) shared(buf, large_recv_buffer, dec_common_buffer, th_data, local_counter_ctx, segments_no, stdout, ii, segment_counter, totaldata, dec_loc, inner_totaldata, recv_pos, source, common_recv_counter, common_counter_gen_sz, mask_pos, Recv_common_IV) num_threads(my_thread_no)
            {
#pragma omp for schedule(static) nowait private(m, pos, dec_pos, enc_data, t_mask, t_mask_pos, t_counter, t_counter_data, iv_buffer, len)
                for (m = 0; m < ii; m++)
                {
                    enc_data = th_data;
                    if (m == ii - 1)
                    {
                        enc_data = inner_totaldata - th_data * (ii - 1);
                    }
                    pos = recv_pos + m * th_data;
                    dec_pos = dec_loc + m * th_data;

                    t_mask = 0;
                    t_mask_pos = 0;
                    t_counter_data = th_data * m; // - common_counter_gen_sz;
                    if (t_counter_data < 1)
                        t_counter = common_recv_counter;
                    else
                        t_counter = common_recv_counter + (unsigned long)((t_counter_data - 1) / 16 + 1);

                    IV_Count_out(iv_buffer, t_counter, &Recv_common_IV[source * 32 + 16]);
                    EVP_EncryptInit_ex(local_counter_ctx[m], NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(local_counter_ctx[m], (buf + dec_pos), &len, (large_recv_buffer + pos), enc_data);
                    //multithread_decryption_common_counter_large_msg(buf, enc_data, pos, dec_pos,  t_counter, large_recv_buffer,local_counter_ctx[m], source);

#if DEBUG_PRECOMPUTE_COUNTER
                    int tid = omp_get_thread_num();
                    //printf("tid = %d precompute recv: totaldata =%d enc_data =%d m =%d common_counter_gen_sz %d t_mask %d t_mask_pos %d t_counter %lu\n",
                    //       tid, totaldata, enc_data, m, common_counter_gen_sz, t_mask,t_mask_pos, t_counter);
                    printf("tid = %d precompute recv: totaldata =%d enc_data =%d m =%d   t_counter %lu common_recv_counter %lu recv_pos %d dec_pos %d\n",
                           tid, totaldata, enc_data, m, t_counter, common_recv_counter, recv_pos, dec_pos);
                    fflush(stdout);
#endif
                } // end of for loop
            }     // end of parallel region
            //printf("common_counter_gen_sz =%d\n",common_counter_gen_sz);fflush(stdout);

            common_recv_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
            recv_pos += inner_totaldata;
            dec_loc += inner_totaldata;
            temp_totaldata -= inner_totaldata;
        }
    }

    return mpi_errno;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
{

    int mpi_errno = MPI_SUCCESS;
    int var;
    MPID_Comm *comm_ptr = NULL;
    int recvtype_sz;
    var = MPI_Type_size(datatype, &recvtype_sz);
    int datasz = count * recvtype_sz;

    MPID_Comm_get_ptr(comm, comm_ptr);
    int isSamenode = 0;

    if (source > -1)
        isSamenode = sameNode[source];
    //isSamenode = comm_ptr->intranode_table[source];

    if (comm_ptr->rank == source || source < 0)
        mpi_errno = MPI_Recv_original(buf, count, datatype, source, tag, comm, status);
    //#if SKIP_INTRA_ENC
    else if (isSamenode > -1)
    {
        if (security_approach == 600) // encryt intra also
            mpi_errno = MPI_SEC_Multi_Thread_Recv_OpenMP(buf, count, datatype, source, tag, comm, status);
        else if (security_approach == 700) // encryt intra also
            mpi_errno = MPI_SEC_BaseCounter_Pipeline_Recv(buf, count, datatype, source, tag, comm, status);
        else
            mpi_errno = MPI_Recv_original(buf, count, datatype, source, tag, comm, status);
    }
    //#endif
    else if (security_approach == 600 || security_approach == 601) //(OPENMP_MULTI_THREAD_ONLY)
        mpi_errno = MPI_SEC_Multi_Thread_Recv_OpenMP(buf, count, datatype, source, tag, comm, status);
    else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        mpi_errno = MPI_SEC_MThreads_Pipelined_OpenMP_Recv_largeSegment_3(buf, count, datatype, source, tag, comm, status);

    else if (security_approach == 700 || security_approach == 701) //(BASE_COUNTER_MODE)
        mpi_errno = MPI_SEC_BaseCounter_Pipeline_Recv(buf, count, datatype, source, tag, comm, status);
    else if (security_approach == 702) //(PRE_COMPUTE_COUNTER_MODE)
        mpi_errno = MPI_SEC_PreComputeCounter_Recv_v4(buf, count, datatype, source, tag, comm, status);
    else
    {
        mpi_errno = MPI_Recv_original(buf, count, datatype, source, tag, comm, status);
    }

    return mpi_errno;
}

/* end of add */