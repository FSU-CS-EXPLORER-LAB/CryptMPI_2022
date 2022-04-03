/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */
#include "mpiimpl.h"

/* Added by Abu Naser(an16e@my.fsu.edu) */
#if 1 //ENABLE_SECURE_MPI  
struct isend_req nonblock_req_handler[ISEND_REQ+5];
int max_send_data_size = 0;
unsigned char Iciphertext[NON_BLOCK_SEND][NON_BLOCK_SEND2];;
int nonBlockCounter = 0;
long pendingIsendRequestCount[MAX_RANKS_LIMIT];
#endif
/* End of add by Abu Naser */

/* -- Begin Profiling Symbol Block for routine MPI_Isend */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Isend = PMPI_Isend
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Isend  MPI_Isend
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Isend as PMPI_Isend
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) __attribute__((weak,alias("PMPI_Isend")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Isend
#define MPI_Isend PMPI_Isend

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Isend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
    MPI_Isend - Begins a nonblocking send

Input Parameters:
+ buf - initial address of send buffer (choice) 
. count - number of elements in send buffer (integer) 
. datatype - datatype of each send buffer element (handle) 
. dest - rank of destination (integer) 
. tag - message tag (integer) 
- comm - communicator (handle) 

Output Parameters:
. request - communication request (handle) 

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
#if 1 //ENABLE_SECURE_MPI 
int MPI_Isend_original(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	      MPI_Comm comm, MPI_Request *request)
#else
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	      MPI_Comm comm, MPI_Request *request)
#endif   
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPID_Request *request_ptr = NULL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_ISEND);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_PT2PT_FUNC_ENTER_FRONT(MPID_STATE_MPI_ISEND);

    /* Validate handle parameters needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
	    MPIR_ERRTEST_COMM(comm, mpi_errno);
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
	    MPIR_ERRTEST_SEND_RANK(comm_ptr, dest, mpi_errno);
	    MPIR_ERRTEST_SEND_TAG(tag, mpi_errno);
	    MPIR_ERRTEST_ARGNULL(request,"request",mpi_errno);

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
    
    mpi_errno = MPID_Isend(buf, count, datatype, dest, tag, comm_ptr,
			   MPID_CONTEXT_INTRA_PT2PT, &request_ptr);
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    MPIR_SENDQ_REMEMBER(request_ptr,dest,tag,comm_ptr->context_id);

    /* return the handle of the request to the user */
    /* MPIU_OBJ_HANDLE_PUBLISH is unnecessary for isend, lower-level access is
     * responsible for its own consistency, while upper-level field access is
     * controlled by the completion counter */
    *request = request_ptr->handle;

    /* ... end of body of routine ... */
    
  fn_exit:
    MPID_MPI_PT2PT_FUNC_EXIT(MPID_STATE_MPI_ISEND);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;
    
  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_isend",
	    "**mpi_isend %p %d %D %i %t %C %p", buf, count, datatype, dest, tag, comm, request);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/* Added by Abu Naser */

int MPI_SEC_Multi_Thread_OpenMP_isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                      MPI_Comm comm, MPI_Request *request)
{
    // printf(">> MPI_SEC_Multi_Thread_Send start\n");fflush(stdout);
    int mpi_errno = MPI_SUCCESS;
    unsigned long max_out_len, ciphertext_len = 0;
    int sendtype_sz, totaldata, m, start, pos, i, s;

    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &sendtype_sz);
    totaldata = count * sendtype_sz;
    unsigned int tempdata = (unsigned int)totaldata;
    /* setting the size of the data */
    Iciphertext[nonBlockCounter][0] = (tempdata >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][1] = (tempdata >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][2] = (tempdata >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][3] = tempdata & 0xFF;

#if OMP_DYNAMIC_THREADS
    Iciphertext[nonBlockCounter][20] = '2';
#elif OMP_DYNAMIC_THREADS_INNOVATION
    Iciphertext[nonBlockCounter][20] = '2';
#endif

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
        // else if(totaldata >= ONE_M)
        //    my_thread_no = 12;
#elif OMP_DYNAMIC_THREADS_INNOVATION
    if (totaldata < SIXTY_4K)
        my_thread_no = 1;
    else if (totaldata < TWO_FIVE_6K)
        my_thread_no = 2;
    else
        my_thread_no = 4;
#endif

#if OMP_DYNAMIC_THREADS
    if (my_thread_no > cyptmpi_series_thread)
        my_thread_no = cyptmpi_series_thread;
#endif

    int data = totaldata / my_thread_no;
    unsigned int adap_chop = (unsigned int)data;
    Iciphertext[nonBlockCounter][21] = (adap_chop >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][22] = (adap_chop >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][23] = (adap_chop >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][24] = adap_chop & 0xFF;
    // printf("   *data=%d totaldata=%d my_thread_n0=%d\n",data,totaldata,my_thread_no);fflush(stdout);
    int segment_no, segment_counter;

#if 0
    if(totaldata <=16){
        segment_no = 1;
        data = totaldata;
    }
    else
    {
        segment_no = (totaldata -1)/data + 1;
    }

    if(!OMP_DYNAMIC_THREADS){
        if(totaldata <= MULTI_THREAD_START_SIZE){
            data = totaldata;
            segment_no = 1;
        }
    }
#endif
    segment_no = 1;
    //mpi_errno = MPI_Send_original(large_send_buffer,MSG_HEADER_SIZE , MPI_UNSIGNED_CHAR, dest, tag, comm);
    int request_counter = 0;
    mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][0], MSG_HEADER_SIZE, MPI_CHAR, dest, tag, comm,
                                   &nonblock_req_handler[nonBlockCounter].request[request_counter++]);

#if OPENMP_THREADS
#pragma omp parallel for schedule(static) default(none) private(segment_counter, count, max_out_len, pos, ciphertext_len) shared(buf, Iciphertext, segment_no, data, stdout, global_openmp_ctx, totaldata, nonBlockCounter) num_threads(my_thread_no)
#endif
    for (segment_counter = 0; segment_counter < segment_no; segment_counter++)
    {

        EVP_AEAD_CTX *local_ctx = global_openmp_ctx;
        max_out_len = data + 16;
        pos = MSG_HEADER_SIZE + segment_counter * (data + 12 + 16);
        RAND_bytes(&Iciphertext[nonBlockCounter][pos], 12);
        int enc_data = data;

        if (segment_counter == segment_no - 1)
        {
            enc_data = totaldata - data * (segment_no - 1);
            max_out_len = enc_data + 16;
        }

        if (!EVP_AEAD_CTX_seal(local_ctx, &Iciphertext[nonBlockCounter][pos + 12],
                               &ciphertext_len, max_out_len,
                               &Iciphertext[nonBlockCounter][pos], 12,
                               buf + segment_counter * data, (unsigned long)(enc_data),
                               NULL, 0))
        {
            printf("[T = %d] openmp-isend Error in encryption: segment_counter=%d\n", omp_get_thread_num(), segment_counter);
            fflush(stdout);
        }
#if OMP_DEBUG
        else
        {
            printf("[T = %d] openmp-isend Encrypted data is %lu in segment_counter = %d total segment = %d\n", omp_get_thread_num(), ciphertext_len, segment_counter, segment_no);
            fflush(stdout);
        }
#endif
    }

    mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][MSG_HEADER_SIZE], totaldata + (segment_no * (12 + 16)), MPI_CHAR, dest, tag, comm,
                                   &nonblock_req_handler[nonBlockCounter].request[request_counter++]);

    nonblock_req_handler[nonBlockCounter].source = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].dest = dest;
    nonblock_req_handler[nonBlockCounter].tag = tag;
    nonblock_req_handler[nonBlockCounter].totaldata = totaldata;
    nonblock_req_handler[nonBlockCounter].req_type = 1;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    nonblock_req_handler[nonBlockCounter].total_request = request_counter;
    *request = nonBlockCounter;

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_MThreads_PipeLine_OpenMP_isend_largeSegment_3(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                                          MPI_Comm comm, MPI_Request *reqst)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned long ciphertext_len, datasize, next;
    ciphertext_len = 0;
    int i, len, sendtype_sz, segments_no, chopping_tag, random_len;
    unsigned char newkey[40], V[50];
    unsigned int own_rank, segment_counter, nonce_counter, d;
    int send_pos, temp_chunk, base, j;

    volatile EVP_AEAD_CTX *local_ctx = NULL;
    unsigned char local_nonce[50];

    EVP_AEAD_CTX *segment_ctx[20];

    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);
    own_rank = (unsigned int)comm_ptr->rank;
    d = (unsigned int)dest;

    int th_data, th_pos, th_start, th_sum, m;
    int my_thread_no;

    int request_counter;

#if CRYPTMPI_ADAPTIVE_CHOPP
    choping_sz = BLOCKING_CHOP_SZ;
#endif

    MPI_Type_size(datatype, &sendtype_sz);
    unsigned long max_out_len = (unsigned long)(16 + (sendtype_sz * count));
    int totaldata = count * sendtype_sz;
    unsigned int temp_data = (unsigned int)totaldata;

    /* setting the size of the data */
    Iciphertext[nonBlockCounter][0] = (temp_data >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][1] = (temp_data >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][2] = (temp_data >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][3] = temp_data & 0xFF;

#if OMP_DYNAMIC_THREADS_PIPELINE
    choping_sz = PIPELINE_SIZE;

#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    choping_sz = 16384;
#endif

    if ((totaldata > PIPELINE_SIZE) && totaldata > LARGE_SEGMENT_SIZE)
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }

    my_thread_no = MY_OPENMP_THREADS_NO; //  omp_get_num_threads();

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
    if (omp_get_team_size(1) > -1) // if true then called from parallel region and switch to CryptMPI-C
        my_thread_no = 1;
#endif

#if CRYPTMPI_SINGLE_THREAD_MODE
    my_thread_no = 1;
#endif

/* just for 1 thread pipeline, rest of the time commnet */
#if ONLY_ONE_THREAD_PIPELINE
    if ((totaldata >= SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE))
    {
        large_send_buffer[20] = '4';
        choping_sz = totaldata / my_thread_no;
        choping_sz++;
    }
#endif

    /* Mark this is a Non-Blocking Send from iSend */
#if OMP_DYNAMIC_THREADS_PIPELINE
    if ((pendingIsendRequestCount[dest] + segments_no + 1 > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata >= SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE))
    {
        Iciphertext[nonBlockCounter][20] = '3';
        choping_sz = totaldata / my_thread_no;
        choping_sz++;
        //choping_sz = (totaldata -1) / my_thread_no +1;
    }
    else
    {
        Iciphertext[nonBlockCounter][20] = '2';
        /* Based on pipeline size thread no will be different */
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
                temp_thread = 8; //4;
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

        choping_sz = (PIPELINE_SIZE - 1) / my_thread_no + 1; // if data less than 64K then thread will be 1
        //choping_sz++;
    }
#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    if ((pendingIsendRequestCount[dest] + segments_no + 1 > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata >= SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE))
    {
        Iciphertext[nonBlockCounter][20] = '3';
        choping_sz = totaldata / my_thread_no;
        choping_sz++;
    }
    else
    {
        Iciphertext[nonBlockCounter][20] = '2';
        if (totaldata > LARGE_SEGMENT_SIZE)
        {
            int temp_thread;
            if (PIPELINE_SIZE < SIXTY_4K)
                temp_thread = 1;
            else if (PIPELINE_SIZE < TWO_M)
                temp_thread = 2;
            else
                temp_thread = 8;

            if (temp_thread > cyptmpi_series_thread)
                temp_thread = cyptmpi_series_thread;
            my_thread_no = temp_thread;
        }

        choping_sz = (PIPELINE_SIZE - 1) / my_thread_no + 1;
        //choping_sz++;
    }
#else
    Iciphertext[nonBlockCounter][20] = '2';
#endif

#if CRYPTMPI_ADAPTIVE_CHOPP
    unsigned int adap_chop = (unsigned int)choping_sz;
    Iciphertext[nonBlockCounter][21] = (adap_chop >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][22] = (adap_chop >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][23] = (adap_chop >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][24] = adap_chop & 0xFF;
#endif

    /* Generate random bytes */
    if ((totaldata > SUBKEY_GEN_START))
    {
        RAND_bytes(&Iciphertext[nonBlockCounter][4], 16);
        memcpy(V, &Iciphertext[nonBlockCounter][4], 16);
        if (symmetric_key_size == 32)
        {
            for (i = 0; i < 16; i++)
                V[i + 16] = (~(int)V[i]);
        }

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
    else
    {
        RAND_bytes(&Iciphertext[nonBlockCounter][4], 16);
        segment_ctx[0] = global_small_msg_ctx;
    }

    request_counter = 0;
    mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][0], MSG_HEADER_SIZE, MPI_CHAR, dest, tag++, comm,
                                   &nonblock_req_handler[nonBlockCounter].request[request_counter++]);

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    pendingIsendRequestCount[dest] += 1;
#endif

    next = 0;
    nonce_counter = 0;
    th_start = 0;
    th_pos = 0;

#if OMP_DEBUG
    printf("OMP_DEBUG isend pipeline: my_thread_no = %d choping_sz=%d  segments_no=%d totaldata=%d\n",
           my_thread_no, choping_sz, segments_no, totaldata);
    fflush(stdout);
#endif

#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
    if (Iciphertext[nonBlockCounter][20] == '3')
    {
        th_data = choping_sz;
        segments_no = (totaldata - 1) / th_data + 1;
        //  printf("isend totaldata =%d choping_sz=%d segments_no=%d\n",totaldata,choping_sz,segments_no);fflush(stdout);

//#pragma omp parallel for default(none) schedule(static) private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base,max_out_len) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter,totaldata) num_threads(my_thread_no)
#pragma omp parallel default(none) private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base, max_out_len) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, totaldata) num_threads(my_thread_no)
        {
#pragma omp for schedule(static) nowait private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base, max_out_len)

            for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
            {
                nonce_counter = segment_counter;
                local_nonce[0] = '0';
                local_nonce[1] = '0';
                local_nonce[2] = '0';
                local_nonce[3] = '0';
                local_nonce[4] = '0';
                local_nonce[5] = '0';
                local_nonce[6] = '0';
                local_nonce[7] = '0';
                local_nonce[8] = (nonce_counter >> 24) & 0xFF;
                local_nonce[9] = (nonce_counter >> 16) & 0xFF;
                local_nonce[10] = (nonce_counter >> 8) & 0xFF;
                local_nonce[11] = nonce_counter & 0xFF;

                base = MSG_HEADER_SIZE + segment_counter * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;
                Iciphertext[nonBlockCounter][base - 5] = '0'; //local_nonce[7];
                Iciphertext[nonBlockCounter][base - 4] = (nonce_counter >> 24) & 0xFF;
                Iciphertext[nonBlockCounter][base - 3] = (nonce_counter >> 16) & 0xFF;
                Iciphertext[nonBlockCounter][base - 2] = (nonce_counter >> 8) & 0xFF;
                Iciphertext[nonBlockCounter][base - 1] = nonce_counter & 0xFF;
                int th_id = omp_get_thread_num();

                int enc_data = th_data;
                max_out_len = enc_data + 16;

                if (segment_counter == segments_no - 1)
                {
                    enc_data = totaldata - th_data * (segments_no - 1);
                    max_out_len = enc_data + 16;
                }

                // printf("th_id=%d base=%d segment_counter=%d\n",th_id,base,segment_counter);fflush(stdout);
                if (!EVP_AEAD_CTX_seal(segment_ctx[th_id], &Iciphertext[nonBlockCounter][base],
                                       &ciphertext_len, max_out_len,
                                       local_nonce, 12,
                                       buf + th_data * segment_counter, enc_data,
                                       NULL, 0))
                {
                    printf(COLOR_RED "Error in encryption: OpenMP Dynamic-thread Pipeline" COLOR_RESET "\n");
                    fflush(stdout);
                }
#if OMP_DEBUG_DYNAMIC_THREAD_PIPELINE
                else
                {
                    printf(COLOR_GREEN "isend-dynamic-thread-pipeline: Thread=%d Encrypted data is %lu in segment_counter = %d nonce_counter=%d totaldata=%d" COLOR_RESET "\n", omp_get_thread_num(), ciphertext_len, segment_counter, nonce_counter, totaldata);
                    fflush(stdout);
                }
#endif
            } // end of thread loop
        }     // end of parallel region

        /* now send the data */
        mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][MSG_HEADER_SIZE], totaldata + (segments_no * (NONCE_HEADER + 16)), MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                       &nonblock_req_handler[nonBlockCounter].request[request_counter++]);
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
        pendingIsendRequestCount[dest] += 1;
#endif
    }
    else
    {
#endif

        int inner_segment_counter;
        int prsd_segment = 0;
        int ii, inner_totaldata;
        int send_loc = 0;
        int enc_loc = 0;
        for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
        {

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

//#if OPENMP_DYNAMIC_SCH
//#pragma omp parallel for default(none) schedule(dynamic) private( nonce_counter, local_nonce, ciphertext_len, mpi_errno, base) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, request_counter, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
//#elif OPENMP_STATIC_SCH
//#pragma omp parallel for default(none) schedule(static) private( nonce_counter, local_nonce, ciphertext_len, mpi_errno, base) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, request_counter, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
#pragma omp parallel default(none) private(nonce_counter, local_nonce, ciphertext_len, mpi_errno, base) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, request_counter, pendingIsendRequestCount, inner_totaldata, segment_counter, ii, prsd_segment, enc_loc, totaldata, send_loc) num_threads(my_thread_no)

            {
#pragma omp for schedule(static) nowait private(nonce_counter, local_nonce, ciphertext_len, mpi_errno, base)

                //#elif OPENMP_GUIDED_SCH
                //#pragma omp parallel for default(none) schedule(guided) private( nonce_counter, local_nonce, ciphertext_len, mpi_errno, base) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, request_counter, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
                //#else
                //#pragma omp parallel for default(none) private( nonce_counter, local_nonce, ciphertext_len, mpi_errno, base) shared(buf, Iciphertext, th_data, segment_ctx, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, nonblock_req_handler, nonBlockCounter, request_counter, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
                //#endif
                for (inner_segment_counter = 0; inner_segment_counter < ii; inner_segment_counter++)
                {

                    int enc_data = th_data;
                    if (inner_segment_counter == ii - 1)
                    {
                        enc_data = inner_totaldata - th_data * (ii - 1);
                    }

                    base = MSG_HEADER_SIZE + send_loc + inner_segment_counter * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;

                    if (totaldata >= SIXTY_4K)
                    {
                        nonce_counter = (unsigned int)(prsd_segment + inner_segment_counter);
                        local_nonce[0] = '0';
                        local_nonce[1] = '0';
                        local_nonce[2] = '0';
                        local_nonce[3] = '0';
                        local_nonce[4] = '0';
                        local_nonce[5] = '0';
                        local_nonce[6] = '0';
                        local_nonce[7] = '0';
                        local_nonce[8] = (nonce_counter >> 24) & 0xFF;
                        local_nonce[9] = (nonce_counter >> 16) & 0xFF;
                        local_nonce[10] = (nonce_counter >> 8) & 0xFF;
                        local_nonce[11] = nonce_counter & 0xFF;

                        if (segment_counter == segments_no - 1)
                        {
                            local_nonce[7] = '1';
                        }

                        Iciphertext[nonBlockCounter][base - 5] = local_nonce[7];
                        Iciphertext[nonBlockCounter][base - 4] = (nonce_counter >> 24) & 0xFF;
                        Iciphertext[nonBlockCounter][base - 3] = (nonce_counter >> 16) & 0xFF;
                        Iciphertext[nonBlockCounter][base - 2] = (nonce_counter >> 8) & 0xFF;
                        Iciphertext[nonBlockCounter][base - 1] = nonce_counter & 0xFF;
                    }
                    else
                    {
                        memcpy(local_nonce, &Iciphertext[nonBlockCounter][4], 12);
                    }
                    //int th_id = omp_get_thread_num();
                    // printf("th_id=%d base=%d segment_counter=%d\n",th_id,base,segment_counter);fflush(stdout);
                    if (!EVP_AEAD_CTX_seal(segment_ctx[inner_segment_counter], &Iciphertext[nonBlockCounter][base],
                                           &ciphertext_len, (enc_data + 16),
                                           local_nonce, 12,
                                           buf + enc_loc + inner_segment_counter * th_data, enc_data,
                                           NULL, 0))
                    {
                        printf("Error in encryption: OpenMP Pipeline\n");
                        fflush(stdout);
                    }
#if OMP_DEBUG
                    else
                    {
                        printf("isend-pipeline: Thread=%d Encrypted data is %lu in segment_counter = %d nonce_counter=%d inner_totaldata=%d totaldata=%d\n", omp_get_thread_num(), ciphertext_len, segment_counter, nonce_counter, inner_totaldata, totaldata);
                        fflush(stdout);
                    }
#endif

                } // ii
            }     // parallel region

            mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][MSG_HEADER_SIZE + send_loc], inner_totaldata + ii * (ENC_MSG_TAG_SIZE + NONCE_HEADER), MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                           &nonblock_req_handler[nonBlockCounter].request[request_counter++]);
            prsd_segment += ii;
            send_loc += (inner_totaldata + ii * (ENC_MSG_TAG_SIZE + NONCE_HEADER));
            enc_loc += inner_totaldata;

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
            pendingIsendRequestCount[dest] += 1;
#endif
        }

#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
    }
#endif

    if ((totaldata > SUBKEY_GEN_START))
    {
        for (i = 0; i < my_thread_no; i++)
            EVP_AEAD_CTX_free(segment_ctx[i]);
    }

    nonblock_req_handler[nonBlockCounter].source = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].dest = dest;
    nonblock_req_handler[nonBlockCounter].tag = tag;
    nonblock_req_handler[nonBlockCounter].totaldata = totaldata;
    nonblock_req_handler[nonBlockCounter].req_type = 1;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    nonblock_req_handler[nonBlockCounter].total_request = request_counter;
    *reqst = nonBlockCounter;

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_BaseCounter_Pipeline_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                       MPI_Comm comm, MPI_Request *request)
{

    int mpi_errno = MPI_SUCCESS;
    int outlen_enc, datasize, next;
    next = 0;
    int i, len, sendtype_sz, segments_no, chopping_tag;
    unsigned char iv_buffer[20];
    int th_data, ii, segment_counter, inner_totaldata, request_counter, inner_segment_counter;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);

    MPI_Status status;

    MPI_Type_size(datatype, &sendtype_sz);
    int totaldata = count * sendtype_sz;

    if ((totaldata > PIPELINE_SIZE) && totaldata > LARGE_SEGMENT_SIZE)
    {
        segments_no = 1;
        segments_no += (totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }

    if (BASE_COUNTER_NO_PIPELINE)
        segments_no = 1;

    choping_sz = PIPELINE_SIZE;
    unsigned int adap_chop = (unsigned int)choping_sz;
    Iciphertext[nonBlockCounter][21] = (adap_chop >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][22] = (adap_chop >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][23] = (adap_chop >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][24] = adap_chop & 0xFF;

    /* setting the size of the data */
    unsigned int temp_data = (unsigned int)totaldata;
    Iciphertext[nonBlockCounter][0] = (temp_data >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][1] = (temp_data >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][2] = (temp_data >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][3] = temp_data & 0xFF;

    /* Generate random bytes */
    if (BASE_COUNTER_LIBRARY_NONCE)
    {
        RAND_bytes(&Iciphertext[nonBlockCounter][4], 16);
        memcpy(iv_buffer, &Iciphertext[nonBlockCounter][4], 16);
    }
    else
    {
        memcpy(iv_buffer, Send_common_IV, 16);
        /* Prepare the counter */
        unsigned long temp_counter_to_send = base_global_counter;
        Iciphertext[nonBlockCounter][5] = (temp_counter_to_send >> 24) & 0xFF;
        Iciphertext[nonBlockCounter][6] = (temp_counter_to_send >> 16) & 0xFF;
        Iciphertext[nonBlockCounter][7] = (temp_counter_to_send >> 8) & 0xFF;
        Iciphertext[nonBlockCounter][8] = temp_counter_to_send & 0xFF;
    }
    int normal_counter_extra_header = 0; //8;

#if DEBUG_BASE_COUNTER
    MPID_Comm *comm_ptr_debug = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr_debug);
    int thisrank = comm_ptr_debug->rank;
    //printf(COLOR_YELLOW "[Rank %d] BaseCounter_ISend totaldata %d segments_no %d choping_sz %d" COLOR_RESET "\n",
    //       thisrank, totaldata, segments_no, choping_sz); fflush(stdout);
#endif

    request_counter = 0;
    mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                   &nonblock_req_handler[nonBlockCounter].request[request_counter++]);

    /* encrypt each message segment */
    next = 0;
    uint32_t n, c;
    long unsigned int temp_counter = 0;
    for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
    {

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

        if (BASE_COUNTER_LIBRARY_NONCE)
            IV_Count(iv_buffer, temp_counter);
        else
            IV_Count(iv_buffer, base_global_counter);

        if (1 != EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer))
            printf(COLOR_RED "___Error in initialization Base counter send___" COLOR_RESET "\n");

        if (1 != EVP_EncryptUpdate(global_counter_ctx, &Iciphertext[nonBlockCounter][COUNTER_HEADER_SIZE + next], &outlen_enc, buf + next, inner_totaldata))
            printf(COLOR_RED "___Error in encryption___ Base counter send " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
        else
        {
            printf(COLOR_GREEN "Base counter isend[rank %d]: Encrypted data is %d  next %d segment_counter %d segments_no %d" COLOR_RESET "\n",
                   thisrank, outlen_enc, next, segment_counter, segments_no);
        }
#endif

        mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][COUNTER_HEADER_SIZE + next], inner_totaldata, MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                       &nonblock_req_handler[nonBlockCounter].request[request_counter++]);
        next += inner_totaldata;
        base_global_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
        temp_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
    }

    nonblock_req_handler[nonBlockCounter].source = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].dest = dest;
    nonblock_req_handler[nonBlockCounter].tag = tag;
    nonblock_req_handler[nonBlockCounter].totaldata = totaldata;
    nonblock_req_handler[nonBlockCounter].req_type = 1;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    nonblock_req_handler[nonBlockCounter].total_request = request_counter;
    *request = nonBlockCounter;

    //printf("pendingIsendRequestCount[%d] =%d\n",dest, pendingIsendRequestCount[dest]);fflush(stdout);

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_SEC_PreComputeCounter_Isend_v4(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                       MPI_Comm comm, MPI_Request *request)
{

    int mpi_errno = MPI_SUCCESS;
    int var, sendtype_sz, i, j;
    MPID_Comm *comm_ptr = NULL;
    unsigned long index_need_to_send;
    int totaldata, next;
    int pre_compute_size, segments_no, datasize;
    int flag;
    MPI_Status status;

    MPI_Type_size(datatype, &sendtype_sz);
    totaldata = count * sendtype_sz;

    int tempamount, blockamount, len;
    unsigned long tempnext, ori_datasize;
    unsigned char iv_buffer[50];
    unsigned char *enc_buffer, *out_buffer, *in_buffer;
    long unsigned int v1, v2;
    unsigned int n;
    int temporary_datasize, enc_data, send_loc;
    int request_counter, inner_totaldata, segment_counter;
    int common_counter_gen_sz, th_data, ii, pos, my_thread_no;
    int base, inner_segment_counter;
    int t_mask, t_mask_pos, t_counter_data;
    long unsigned int t_counter;

    MPID_Comm_get_ptr(comm, comm_ptr);
    int rank = comm_ptr->rank;
    request_counter = 0;

    /* setting the size of the data */
    unsigned int temp_data = (unsigned int)totaldata;
    Iciphertext[nonBlockCounter][0] = (temp_data >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][1] = (temp_data >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][2] = (temp_data >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][3] = temp_data & 0xFF;

    choping_sz = PIPELINE_SIZE;

#if DEBUG_PRECOMPUTE_COUNTER
    printf(COLOR_CYAN " [Rank = %d]: dest =%d  totaldata =%d common_compute_size = %d counter_needto_send %lu counter_needto_send_large_msg %lu" COLOR_RESET "\n",
           comm_ptr->rank, dest, totaldata, common_compute_size, counter_needto_send, counter_needto_send_large_msg);
    fflush(stdout);
#endif

    /* use common mode */

    if (totaldata > PIPELINE_SIZE && totaldata > LARGE_SEGMENT_SIZE)
    {
        segments_no = 1;
        segments_no += (totaldata - (PIPELINE_SIZE)-1) / (PIPELINE_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    }

    my_thread_no = MY_OPENMP_THREADS_NO; //  omp_get_num_threads();

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
#else // noleland
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

/* Mark this is a nonBlocking Send from iSend */
#if DYNAMIC_PIPELINE
    if ((pendingIsendRequestCount[dest] + segments_no + 1 > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata >= SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE))
    {
        Iciphertext[nonBlockCounter][20] = '3';
        choping_sz = (totaldata - 1) / my_thread_no + 1;
        choping_sz = (choping_sz - 1) / 16 * 16 + 16; // make chooping size as multiple of 16
        segments_no = 1;                              // mark the segments_no to 1 segment
    }
    else
    {
        Iciphertext[nonBlockCounter][20] = '1';
        /* Based on pipeline size thread no will be different */
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
        choping_sz = (PIPELINE_SIZE - 1) / my_thread_no + 1; // if data less than 64K then thread will be 1
        choping_sz = (choping_sz - 1) / 16 * 16 + 16;        // make chooping size as multiple of 16
    }
#else
    Iciphertext[nonBlockCounter][20] = '1';
#endif

    unsigned int adap_chop = (unsigned int)choping_sz;
    Iciphertext[nonBlockCounter][21] = (adap_chop >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][22] = (adap_chop >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][23] = (adap_chop >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][24] = adap_chop & 0xFF;

    /* Prepare the counter */
    //unsigned long temp_counter_to_send = counter_needto_send;
    unsigned long temp_counter_to_send;
    if (totaldata < PRE_COM_DATA_RANGE)
    {
        if (common_compute_size < totaldata)
        {
            Iciphertext[nonBlockCounter][4] = '1';
            temp_counter_to_send = counter_needto_send_large_msg; // save partial precomputation
        }
        else
        {
            Iciphertext[nonBlockCounter][4] = '0';
            temp_counter_to_send = counter_needto_send;
        }
    }
    else
    {
        temp_counter_to_send = counter_needto_send_large_msg;
    }
    Iciphertext[nonBlockCounter][5] = (temp_counter_to_send >> 24) & 0xFF;
    Iciphertext[nonBlockCounter][6] = (temp_counter_to_send >> 16) & 0xFF;
    Iciphertext[nonBlockCounter][7] = (temp_counter_to_send >> 8) & 0xFF;
    Iciphertext[nonBlockCounter][8] = temp_counter_to_send & 0xFF;

    request_counter = 0;
    //mpi_errno = MPI_Send_original(large_send_buffer, COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, tag++, comm);
    mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][0], COUNTER_HEADER_SIZE, MPI_CHAR, dest, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[request_counter++]);
    pendingIsendRequestCount[dest] += 1;

#if DEBUG_PRECOMPUTE_COUNTER
    MPID_Comm_get_ptr(comm, comm_ptr);
    printf(COLOR_CYAN "[Rank = %d]: dest =%d  totaldata =%d choping_sz = %d segments_no %d my_thread_no %d counter_needto_send %lu" COLOR_RESET "\n",
           comm_ptr->rank, dest, totaldata, choping_sz, segments_no, my_thread_no, counter_needto_send);
    fflush(stdout);

    /* Do a sanity check here 
     * enc_common_counter, start, end, compute size
    */
#endif
    if (totaldata < PRE_COM_DATA_RANGE)
    {
        next = 0;
        pos = COUNTER_HEADER_SIZE;

        if (common_compute_size >= totaldata)
            encryption_common_counter(buf, next, totaldata, pos, &Iciphertext[nonBlockCounter][0]);
        else
        {
            memcpy(iv_buffer, (Send_common_IV + 16), 16);
            IV_Count(iv_buffer, enc_common_counter_long_msg);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, &Iciphertext[nonBlockCounter][pos], &len, buf, totaldata);
            enc_common_counter_long_msg += (unsigned long)(totaldata - 1) / 16 + 1;
            counter_needto_send_large_msg += ((totaldata - 1) / 16) + 1;
        }
        // printf("%2x %2x\n",enc_common_buffer[0],large_send_buffer[0]);fflush(stdout);
        mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][pos], totaldata, MPI_CHAR, dest, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[request_counter++]);
        pendingIsendRequestCount[dest] += 1;
    }
    else
    {
        next = 0;
        pos = COUNTER_HEADER_SIZE;
        send_loc = 0;
        int temp, temp2;

        /* Encrypt each message segment */
        for (segment_counter = 0; segment_counter < segments_no; segment_counter++)
        {
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
            MPID_Comm_get_ptr(comm, comm_ptr);
            printf(COLOR_CYAN "[Rank = %d]: dest =%d enc_common_start=%d enc_common_end=%d common_compute_size = %d enc_common_counter_long_msg %lu totaldata =%d segmentno=%d send_loc=%d" COLOR_RESET "\n",
                   comm_ptr->rank, dest, enc_common_start, enc_common_end, common_compute_size, enc_common_counter_long_msg, totaldata, segment_counter, send_loc);
            fflush(stdout);
#endif

//#pragma omp parallel for default(none) schedule(static) private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data) shared(buf, Iciphertext, enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, inner_totaldata, segment_counter, ii, totaldata, send_loc, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter_long_msg,nonBlockCounter) num_threads(my_thread_no)
#pragma omp parallel default(none) private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data) shared(buf, Iciphertext, enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, inner_totaldata, segment_counter, ii, totaldata, send_loc, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter_long_msg, nonBlockCounter) num_threads(my_thread_no)
            {
#pragma omp for schedule(static) nowait private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data)

                for (inner_segment_counter = 0; inner_segment_counter < ii; inner_segment_counter++)
                {
                    enc_data = th_data;
                    if (inner_segment_counter == ii - 1)
                    {
                        enc_data = inner_totaldata - th_data * (ii - 1);
                    }

                    base = send_loc + inner_segment_counter * th_data;

                    // enc_buffer = (unsigned char *)(enc_common_buffer);
                    in_buffer = (unsigned char *)(buf + base);
                    out_buffer = (unsigned char *)(&Iciphertext[nonBlockCounter][COUNTER_HEADER_SIZE + base]);
                    int tid = omp_get_thread_num();

                    t_counter_data = th_data * inner_segment_counter; // - common_compute_size;
                    if (t_counter_data < 1)
                        t_counter = enc_common_counter_long_msg;
                    else
                        t_counter = enc_common_counter_long_msg + (unsigned long)((t_counter_data - 1) / 16 + 1);

#if DEBUG_PRECOMPUTE_COUNTER
                    j = inner_segment_counter;
                    printf(COLOR_MAGENTA "[rank %d tid = %d]: inner_segment_counter %d tinfo_start=%d enc_common_end=%d tinfo_comsize = %d counter %lu enc_data =%d segment_counter=%d send_loc=%d base %d" COLOR_RESET "\n",
                           rank, tid, j, tinfo[j].tstart, enc_common_end, tinfo[j].compute_size, tinfo[j].counter, enc_data, segment_counter, send_loc, base);
                    fflush(stdout);
#endif

                    multithreads_encryption_common_counter_large_msg(in_buffer, enc_data, out_buffer,
                                                                     local_counter_ctx[inner_segment_counter], t_counter, tid);
                    /* multithreads_encryption_common_counter(in_buffer, enc_data, enc_buffer, out_buffer, t_mask_pos,
                                                       enc_common_end, t_mask, local_counter_ctx[inner_segment_counter], t_counter, tid);*/

                    /*multithreads_encryption_common_counter(in_buffer, enc_data, enc_buffer, out_buffer, tinfo[inner_segment_counter].tstart,
                                                       enc_common_end, tinfo[inner_segment_counter].compute_size, local_counter_ctx[tid], tinfo[inner_segment_counter].counter, tid);*/
                } //  end of for
            }     // end of parallel region

            mpi_errno = MPI_Isend_original(&Iciphertext[nonBlockCounter][COUNTER_HEADER_SIZE + send_loc], inner_totaldata, MPI_CHAR, dest, tag++, comm, &nonblock_req_handler[nonBlockCounter].request[request_counter++]);

            enc_common_counter_long_msg += (unsigned long)(inner_totaldata - 1) / 16 + 1;
            send_loc += inner_totaldata;
            pendingIsendRequestCount[dest] += 1;
            //next += inner_totaldata;
            //pos += inner_totaldata;
        }
        // update counter_needto_send_large_msg
        counter_needto_send_large_msg += ((totaldata - 1) / 16) + 1;

#if DEBUG_PRECOMPUTE_COUNTER
        MPID_Comm_get_ptr(comm, comm_ptr);
        printf(COLOR_CYAN "[Rank = %d]: dest =%d enc_common_start=%d enc_common_end=%d common_compute_size = %d enc_common_counter %lu counter_needto_send =%d send_loc=%d" COLOR_RESET "\n",
               comm_ptr->rank, dest, enc_common_start, enc_common_end, common_compute_size, enc_common_counter, counter_needto_send, send_loc);
        fflush(stdout);
#endif
    }

    nonblock_req_handler[nonBlockCounter].source = comm_ptr->rank;
    nonblock_req_handler[nonBlockCounter].dest = dest;
    nonblock_req_handler[nonBlockCounter].tag = tag;
    nonblock_req_handler[nonBlockCounter].totaldata = totaldata;
    nonblock_req_handler[nonBlockCounter].req_type = 1;
    nonblock_req_handler[nonBlockCounter].buffer = buf;
    nonblock_req_handler[nonBlockCounter].total_request = request_counter;
    *request = nonBlockCounter;

    // printf("isend pendingIsendRequestCount[%d] =%d\n",dest, pendingIsendRequestCount[dest]);fflush(stdout);

    nonBlockCounter++;
    if (nonBlockCounter == ISEND_REQ)
        nonBlockCounter = 0;

    return mpi_errno;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    int sendtype_sz, datasz;
    MPID_Comm *comm_ptr = NULL;

    MPID_Comm_get_ptr(comm, comm_ptr);
    MPI_Type_size(datatype, &sendtype_sz);
    datasz = count * sendtype_sz;

    int isSamenode = 0;
    if (dest > -1)
        isSamenode = sameNode[dest];
    // isSamenode = comm_ptr->intranode_table[dest];

    if (comm_ptr->rank == dest || dest < 0)
    {
        mpi_errno = MPI_Isend_original(buf, count, datatype, dest, tag, comm, request);
    }
    //#if SKIP_INTRA_ENC
    else if (isSamenode > -1)
    {
        if (security_approach == 600) // encryt intra also
            mpi_errno = MPI_SEC_Multi_Thread_OpenMP_isend(buf, count, datatype, dest, tag, comm, request);
        else if (security_approach == 700) // encryt intra also
            mpi_errno = MPI_SEC_BaseCounter_Pipeline_Isend(buf, count, datatype, dest, tag, comm, request);
        else
            mpi_errno = MPI_Isend_original(buf, count, datatype, dest, tag, comm, request);
    }
    //#endif
    else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        mpi_errno = MPI_SEC_MThreads_PipeLine_OpenMP_isend_largeSegment_3(buf, count, datatype, dest, tag, comm, request);
    else if (security_approach == 600 || security_approach == 601) //(OPENMP_MULTI_THREAD_ONLY)
        mpi_errno = MPI_SEC_Multi_Thread_OpenMP_isend(buf, count, datatype, dest, tag, comm, request);
    else if (security_approach == 700 || security_approach == 701) //if(BASE_COUNTER_MODE)
        mpi_errno = MPI_SEC_BaseCounter_Pipeline_Isend(buf, count, datatype, dest, tag, comm, request);
    else if (security_approach == 702) //(PRE_COMPUTE_COUNTER_MODE)
        mpi_errno = MPI_SEC_PreComputeCounter_Isend_v4(buf, count, datatype, dest, tag, comm, request);
    else
    {
        mpi_errno = MPI_Isend_original(buf, count, datatype, dest, tag, comm, request);
    }

    return mpi_errno;
}
/* end of add */