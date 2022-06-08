/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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

/* Added by Abu Naser */
unsigned char large_send_buffer[COMMON_LARGE_SEND_BUFFER_SIZE];
unsigned char large_recv_buffer[COMMON_LARGE_RECV_BUFFER_SIZE];

struct CryptHandleProbe Crypthandle_probe[2048]; 
#if CRYPTMPI_ADAPTIVE_CHOPP
int cryptMPI_adap_chop;
#endif
/* end of add */

/* -- Begin Profiling Symbol Block for routine MPI_Send */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Send = PMPI_Send
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Send  MPI_Send
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Send as PMPI_Send
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm) __attribute__((weak,alias("PMPI_Send")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Send
#define MPI_Send PMPI_Send

#endif

#undef FUNCNAME
#define FUNCNAME MPI_Send

/*@
    MPI_Send - Performs a blocking send

Input Parameters:
+ buf - initial address of send buffer (choice) 
. count - number of elements in send buffer (nonnegative integer) 
. datatype - datatype of each send buffer element (handle) 
. dest - rank of destination (integer) 
. tag - message tag (integer) 
- comm - communicator (handle) 

Notes:
This routine may block until the message is received by the destination 
process.

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_COMM
.N MPI_ERR_COUNT
.N MPI_ERR_TYPE
.N MPI_ERR_TAG
.N MPI_ERR_RANK

.seealso: MPI_Isend, MPI_Bsend
@*/

/* Added by Abu Naser */
#if 1 //ENABLE_SECURE_MPI
int MPI_Send_original(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                      MPI_Comm comm)
#else
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	     MPI_Comm comm)
#endif  
{
    static const char FCNAME[] = "MPI_Send";
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;
    MPID_Request * request_ptr = NULL;
    MPID_MPI_STATE_DECL(MPID_STATE_MPI_SEND);

    MPIR_ERRTEST_INITIALIZED_ORDIE();
    
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPID_MPI_PT2PT_FUNC_ENTER_FRONT(MPID_STATE_MPI_SEND);
    
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
    
    mpi_errno = MPID_Send(buf, count, datatype, dest, tag, comm_ptr, 
			  MPID_CONTEXT_INTRA_PT2PT, &request_ptr);
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    if (request_ptr == NULL)
    {
#if defined(CHANNEL_MRAIL)
        mpi_errno = MPID_Progress_test();
        if (mpi_errno != MPI_SUCCESS)
        {
            goto fn_fail;
        }
#endif /* defined(CHANNEL_MRAIL) */
	goto fn_exit;
    }

    /* If a request was returned, then we need to block until the request 
       is complete */
    if (!MPID_Request_is_complete(request_ptr))
    {
	MPID_Progress_state progress_state;
	    
	MPID_Progress_start(&progress_state);
        while (!MPID_Request_is_complete(request_ptr))
	{
	    mpi_errno = MPID_Progress_wait(&progress_state);
	    if (mpi_errno != MPI_SUCCESS)
	    {
		/* --BEGIN ERROR HANDLING-- */
		MPID_Progress_end(&progress_state);
		goto fn_fail;
		/* --END ERROR HANDLING-- */
	    }
	}
	MPID_Progress_end(&progress_state);
    }

    mpi_errno = request_ptr->status.MPI_ERROR;
    MPID_Request_release(request_ptr);
    
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    /* ... end of body of routine ... */
    
  fn_exit:
    MPID_MPI_PT2PT_FUNC_EXIT(MPID_STATE_MPI_SEND);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, "**mpi_send", 
	    "**mpi_send %p %d %D %i %t %C", buf, count, datatype, dest, tag, comm);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( comm_ptr, FCNAME, mpi_errno );
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/* Added by Abu Naser */
int MPI_SEC_Multi_Thread_Send_OpenMP(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
	     MPI_Comm comm)
{
    // printf(">> MPI_SEC_Multi_Thread_Send start\n");fflush(stdout);
    int mpi_errno = MPI_SUCCESS;
    unsigned long max_out_len, ciphertext_len = 0;
    int sendtype_sz, totaldata, m, start, pos, i, s;

    MPI_Request request1, request2;
    MPI_Status status;

    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);

    MPI_Type_size(datatype, &sendtype_sz);
    totaldata = count * sendtype_sz;

    unsigned int temp_data = (unsigned int)totaldata;

    /* setting the size of the data */
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;

    /* Mark this is a Blocking Send from Send */
#if 1 //OMP_DYNAMIC_THREADS
    large_send_buffer[20] = '1';
#elif OMP_DYNAMIC_THREADS_INNOVATION
    large_send_buffer[20] = '1';
#endif

    int my_thread_no = 1; //MY_OPENMP_THREADS_NO;

#if 0 //OMP_DYNAMIC_THREADS   
   if(my_thread_no > cyptmpi_series_thread)
    my_thread_no = cyptmpi_series_thread;
#endif

    int data = totaldata / my_thread_no;

    unsigned int adap_chop = (unsigned int)data;
    large_send_buffer[21] = (adap_chop >> 24) & 0xFF;
    large_send_buffer[22] = (adap_chop >> 16) & 0xFF;
    large_send_buffer[23] = (adap_chop >> 8) & 0xFF;
    large_send_buffer[24] = adap_chop & 0xFF;

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

    mpi_errno = MPI_Isend_original(large_send_buffer, MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, tag, comm,
                                   &request1);

#if OPENMP_THREADS
#pragma omp parallel for schedule(static) default(none) private(segment_counter, count, max_out_len, pos, ciphertext_len) shared(buf, large_send_buffer, segment_no, data, stdout, global_openmp_ctx, totaldata) num_threads(my_thread_no)
#endif
    for (segment_counter = 0; segment_counter < segment_no; segment_counter++)
    {

        EVP_AEAD_CTX *local_ctx = global_openmp_ctx;
        RAND_bytes(&large_send_buffer[MSG_HEADER_SIZE + segment_counter * (data + 12 + 16)], 12);

        max_out_len = data + 16;
        pos = MSG_HEADER_SIZE + segment_counter * (data + 12 + 16);

        int enc_data = data;

        if (segment_counter == segment_no - 1)
        {
            enc_data = totaldata - data * (segment_no - 1);
            max_out_len = enc_data + 16;
        }

        if (!EVP_AEAD_CTX_seal(local_ctx, &large_send_buffer[pos + 12],
                               &ciphertext_len, max_out_len,
                               &large_send_buffer[pos], 12,
                               buf + segment_counter * data, (unsigned long)(enc_data),
                               NULL, 0))
        {
            printf("[T = %d] omp-send Error in encryption: segment_counter=%d\n", omp_get_thread_num(), segment_counter);
            fflush(stdout);
        }
#if OMP_DEBUG
        else
        {
            printf("[T = %d] omp-send Encrypted data is %lu in segment_counter = %d total segment = %d\n", omp_get_thread_num(), ciphertext_len, segment_counter, segment_no);
            fflush(stdout);
        }
#endif
    }
    // }

    mpi_errno = MPI_Isend_original(large_send_buffer + MSG_HEADER_SIZE, totaldata + (segment_no * (12 + 16)), MPI_UNSIGNED_CHAR, dest, tag, comm,
                                   &request2);

    mpi_errno = MPI_Wait_original(&request1, &status);
    mpi_errno = MPI_Wait_original(&request2, &status);

    return mpi_errno;
}

int MPI_SEC_MThreads_PipeLine_OpenMP_Send__largeSegment_3(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                          MPI_Comm comm)
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

    EVP_AEAD_CTX *segment_ctx[100]; // total threads will not excede 100
   

    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);
    own_rank = (unsigned int)comm_ptr->rank;
    d = (unsigned int)dest;

    int th_data, th_pos, th_start, th_sum, m;
    int my_thread_no;
    MPI_Request request[300];
    MPI_Status status[300];
    int request_counter;

    MPI_Type_size(datatype, &sendtype_sz);
    unsigned long max_out_len = (unsigned long)(16 + (sendtype_sz * count));
    int totaldata = count * sendtype_sz;

   // printf("%d sending to %d data = %d tag =%d\n",own_rank, dest, totaldata,tag );fflush(stdout);

    unsigned int temp_data = (unsigned int)totaldata;
    /* setting the size of the data */
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;

                
    /* Mark this is a Blocking Send from Send */
    large_send_buffer[20] = '1'; 

    
#if OMP_DYNAMIC_THREADS_PIPELINE
    choping_sz = PIPELINE_SIZE;  //LARGE_SEGMENT_SIZE; //THIRTY_2K ; // ; //TWO_FIVE_6K; // SIXTY_4K; // THIRTY_2K

#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    choping_sz = 16384;
  
#endif 


    if ((totaldata > PIPELINE_SIZE) && totaldata > LARGE_SEGMENT_SIZE)
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (PIPELINE_SIZE ) - 1) / (PIPELINE_SIZE) + 1;
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
       if(totaldata < SIXTY_4K)
        my_thread_no = 1;
    else if( totaldata < ONE_TWO_8K)
        my_thread_no = 2;
     else if( totaldata < FIVE_ONE_2K)
        my_thread_no = 4;
     else if(totaldata >= FIVE_ONE_2K)
        my_thread_no = 8;      
#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    if (totaldata < SIXTY_4K)
        my_thread_no = 1;
    else if(totaldata < TWO_M)
        my_thread_no = 2;
    else
        my_thread_no = 8;
#endif 



#if OMP_DYNAMIC_THREADS_PIPELINE
   
   if(my_thread_no > cyptmpi_series_thread)
    my_thread_no = cyptmpi_series_thread;

#endif

/* Switch Between CryptMPI-R and CryptMPI-C */
#if ENABLE_CRYPTMPI_RC_SWITCH
    // printf("send: Thread# %d omp_get_team_size %d omp_get_ancestor_thread_num %d omp_get_level %d omp_get_max_active_levels %d\n",
    //  omp_get_thread_num(),omp_get_team_size(1), omp_get_ancestor_thread_num(1),omp_get_level(),omp_get_max_active_levels());
    if( omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
        my_thread_no = 1;
#endif

#if CRYPTMPI_SINGLE_THREAD_MODE
    my_thread_no = 1;
#endif

/* just for 1 thread pipeline, rest of the time commnet */
#if ONLY_ONE_THREAD_PIPELINE 
 if ( (totaldata>=SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE)){
        large_send_buffer[20] = '4';
          choping_sz = (totaldata - 1) / my_thread_no + 1;
       // choping_sz++;
    }
#endif   


    /* Mark this is a Blocking Send from Send */
#if OMP_DYNAMIC_THREADS_PIPELINE
    if ((pendingIsendRequestCount[dest] + segments_no > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata>=SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE)){
        large_send_buffer[20] = '4';
        choping_sz = (totaldata - 1) / my_thread_no + 1;
        //choping_sz++;
    }
    else
    {
        large_send_buffer[20] = '1';
    /* Based on pipeline size thread no will be different */
#if OMP_DYNAMIC_THREADS_PIPELINE
        if (totaldata > LARGE_SEGMENT_SIZE)
        {
            int temp_thread;
#if PSC_BRIDGE_TUNE
            if (PIPELINE_SIZE < SIXTY_4K)
                temp_thread = 1;
            else if (PIPELINE_SIZE < ONE_TWO_8K)
                temp_thread = 4;
            else if(PIPELINE_SIZE < TWO_FIVE_6K)
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
    if( omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
        my_thread_no = 1;
#endif
#if CRYPTMPI_SINGLE_THREAD_MODE
            my_thread_no = 1;
#endif            
        }

#endif

        choping_sz = (PIPELINE_SIZE -1) / my_thread_no +1; // if data less than 64K then thread will be 1
        //choping_sz++;
    }
#elif OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    if ((pendingIsendRequestCount[dest] + segments_no > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata>=SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE)){
         large_send_buffer[20] = '4';
           choping_sz = (totaldata - 1) / my_thread_no + 1;
        //choping_sz++;
    }
    else
    {
        large_send_buffer[20] = '1';   
#if OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
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
/* Switch Between CryptMPI-R and CryptMPI-C */
#if ENABLE_CRYPTMPI_RC_SWITCH
    if( omp_get_team_size(1) > -1) // if true then called from parallel region switch to CryptMPI-C
        my_thread_no = 1;
#endif            
#if CRYPTMPI_SINGLE_THREAD_MODE
            my_thread_no = 1;
#endif            
        }
#endif         
        choping_sz = (PIPELINE_SIZE -1) / my_thread_no +1;
        //choping_sz++;
    }
#else
     large_send_buffer[20] = '1';
#endif    

#if CRYPTMPI_ADAPTIVE_CHOPP
    unsigned int adap_chop = (unsigned int) choping_sz;
    large_send_buffer[21] = (adap_chop >> 24) & 0xFF;
    large_send_buffer[22] = (adap_chop >> 16) & 0xFF;
    large_send_buffer[23] = (adap_chop >> 8) & 0xFF;
    large_send_buffer[24] = adap_chop & 0xFF;
#endif  

     /* Generate random bytes */
    if ((totaldata > SUBKEY_GEN_START))
    {
        RAND_bytes(&large_send_buffer[4], 16);
        memcpy(V, &large_send_buffer[4], 16);

        if (symmetric_key_size == 32)
        {
            for (i = 0; i < 16; i++)
                V[i + 16] = (~(int)V[i]);
        }

        if (1 != EVP_EncryptUpdate(ctx_enc, newkey, &len, V, symmetric_key_size))
            printf("1");

        if (symmetric_key_size == 16)
        {
            for(i=0;i<my_thread_no; i++)
                segment_ctx[i] = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                         newkey,
                                         16, 0);
        }
        else if (symmetric_key_size == 32)
        {
            for(i=0;i<my_thread_no; i++)
                segment_ctx[i] = EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(),
                                         newkey,
                                         32, 0);
        }
    }
    else
    {
          RAND_bytes(&large_send_buffer[4], 12);
         segment_ctx[0] = global_small_msg_ctx;
    }

    
    //newtag[dest] = newtag[dest] + 1;
 //   tag = sendtag[dest] + 1;
    

    mpi_errno = MPI_Send_original(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR, dest, tag++, comm);
   // tag++;
   // printf(COLOR_GREEN"rank %d sending header to %d tag = %d totaldata =%d"COLOR_RESET"\n",own_rank,dest,tag-1,totaldata);fflush(stdout);

    pendingIsendRequestCount[dest]+=1; // decrese it at the end
  
   
   // newtag[dest] = newtag[dest] + 1;
   // tag = newtag[dest];
   
    next = 0;


    nonce_counter = 0;
    request_counter = 0;
    th_start = 0;
    th_pos = 0;

#if  OMP_DEBUG
            printf("OMP_DEBUG: my_thread_no = %d choping_sz=%d th_data=%d segments_no=%d\n",
            my_thread_no,choping_sz,th_data,segments_no);fflush(stdout);    
#endif




#if  (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
    if ( large_send_buffer[20] == '4')
    {
        th_data = choping_sz;
        segments_no = (totaldata -1)/th_data + 1;


//#pragma omp parallel for default(none) schedule(static) private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base,max_out_len) shared(buf, large_send_buffer, segment_ctx, th_data, segments_no, local_ctx, stdout, my_thread_no,totaldata) num_threads(my_thread_no)
#pragma omp parallel default(none) private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base,max_out_len) shared(buf, large_send_buffer, segment_ctx, th_data, segments_no, local_ctx, stdout, my_thread_no,totaldata) num_threads(my_thread_no)
    {
        #pragma omp for schedule(static) nowait private(segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base,max_out_len) 
        for (segment_counter = 0; segment_counter < segments_no ; segment_counter++)
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
            large_send_buffer[base - 5] = '0'; //local_nonce[7];
            large_send_buffer[base - 4] = (nonce_counter >> 24) & 0xFF;
            large_send_buffer[base - 3] = (nonce_counter >> 16) & 0xFF;
            large_send_buffer[base - 2] = (nonce_counter >> 8) & 0xFF;
            large_send_buffer[base - 1] = nonce_counter & 0xFF;
            //int th_id = omp_get_thread_num();
           // segment_ctx[th_id] = local_ctx;
           

            int enc_data = th_data;
             max_out_len = enc_data+16; 
              //if(enc_data < 4000)
                //    printf("send tid %d enc_data %d\n",th_id,enc_data);fflush(stdout);
            // active_threads = omp_get_num_threads();
            // printf("active_threads=%d\n",active_threads);fflush(stdout);

            if(segment_counter == segments_no -1){
                enc_data = totaldata - th_data*(segments_no-1);
                max_out_len = enc_data+16; 
            }
            

            // printf("th_id=%d base=%d segment_counter=%d\n",th_id,base,segment_counter);fflush(stdout);
            if (!EVP_AEAD_CTX_seal(segment_ctx[segment_counter], &large_send_buffer[base],
                                   &ciphertext_len, max_out_len,
                                   local_nonce, 12,
                                   buf + th_data * segment_counter, enc_data,
                                   NULL, 0))
            {
                printf(COLOR_RED"Error in encryption: OpenMP Dynamic-thread Pipeline"COLOR_RESET"\n");
                fflush(stdout);
            }
#if OMP_DEBUG_DYNAMIC_THREAD_PIPELINE
            else
            {
                printf(COLOR_GREEN"1 send-dynamic-thread-pipeline: Thread=%d Encrypted data is %lu in segment_counter = %d nonce_counter=%d"COLOR_RESET"\n", omp_get_thread_num(), ciphertext_len, segment_counter, nonce_counter);
                fflush(stdout);
            }
#endif
        }// end of thread loop
    } // end parallel
      

        /* now send the data */
        mpi_errno = MPI_Isend_original(&large_send_buffer[MSG_HEADER_SIZE], totaldata+(segments_no*(NONCE_HEADER+16)), MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                                &request[request_counter++]);

      //  printf(COLOR_GREEN"    rank %d sending header to %d tag = %d data = %d"COLOR_RESET"\n",own_rank,dest,tag,totaldata+(segments_no*(NONCE_HEADER+16)));fflush(stdout);                                        

       // printf(COLOR_GREEN"    rank %d sending header to %d tag = %d data = %d"COLOR_RESET"\n",own_rank,dest,tag,totaldata+(segments_no*(NONCE_HEADER+16)));fflush(stdout);                                        

#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION        
        pendingIsendRequestCount[dest]+=1;
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
           // printf("my_thread_no =%d\n",my_thread_no);fflush(stdout);    



//#if OPENMP_DYNAMIC_SCH
//#pragma omp parallel for default(none) schedule(dynamic) private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) shared(buf, large_send_buffer, segment_ctx, th_data, request, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
//#elif OPENMP_STATIC_SCH
//#pragma omp parallel for default(none) schedule(static)  private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) shared(buf, large_send_buffer, segment_ctx, th_data, request, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
#pragma omp parallel default(none)  private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) shared(buf, large_send_buffer, segment_ctx, th_data, request, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
{
    #pragma omp for schedule(static) nowait private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) 
//#elif OPENMP_GUIDED_SCH
//#pragma omp parallel for default(none) schedule(guided) private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) shared(buf, large_send_buffer, segment_ctx, th_data, request, segments_no, dest, tag, comm, local_ctx, stdout, my_thread_no, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
//#else
//#pragma omp parallel for default(none) private(inner_segment_counter, nonce_counter, local_nonce, ciphertext_len, mpi_errno, i, base) shared(buf, large_send_buffer, segment_ctx, th_data, request, segments_no, dest, tag, comm , local_ctx, stdout, my_thread_no, pendingIsendRequestCount, inner_totaldata,segment_counter,ii,prsd_segment,enc_loc,totaldata,send_loc) num_threads(my_thread_no)
//#endif
            for (inner_segment_counter = 0; inner_segment_counter < ii; inner_segment_counter++)
            {
                int enc_data = th_data;
                if (inner_segment_counter == ii - 1)
                {
                    enc_data = inner_totaldata - th_data * (ii - 1);
                }

               // base = MSG_HEADER_SIZE + (prsd_segment + inner_segment_counter) * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;
               base = MSG_HEADER_SIZE + send_loc+ inner_segment_counter * (th_data + ENC_MSG_TAG_SIZE + NONCE_HEADER) + NONCE_HEADER;
                /* printf("    [Thread id =%d] omp-send:segment_counter=%d inner_segment_counter=%d ii=%d th_data=%d \
                            nonce_counter=%d base=%d %u %u %u %u => %u segments_no=%d totaldata=%d inner_totaldata=%d\n",
                               omp_get_thread_num(), segment_counter, inner_segment_counter, ii, th_data, nonce_counter, base, 
                               large_send_buffer[base - 1], large_send_buffer[base - 2], large_send_buffer[base - 3], 
                               large_send_buffer[base - 4], 0, segments_no, totaldata,inner_totaldata);
                        fflush(stdout);*/

                if(totaldata >= SIXTY_4K)
                {
                    nonce_counter = (unsigned int) (prsd_segment + inner_segment_counter);
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

                    large_send_buffer[base - 5] = local_nonce[7];
                    large_send_buffer[base - 4] = (nonce_counter >> 24) & 0xFF;
                    large_send_buffer[base - 3] = (nonce_counter >> 16) & 0xFF;
                    large_send_buffer[base - 2] = (nonce_counter >> 8) & 0xFF;
                    large_send_buffer[base - 1] = nonce_counter & 0xFF;
                }
                else
                {
                    memcpy(local_nonce, &large_send_buffer[4], 12);
                }
                
               
                if (!EVP_AEAD_CTX_seal(segment_ctx[inner_segment_counter], large_send_buffer + base,
                                       &ciphertext_len, (enc_data + 16),
                                       local_nonce, 12,
                                       buf + enc_loc + inner_segment_counter*th_data, enc_data,
                                       NULL, 0))
                {
                    printf("Error in encryption: OpenMP Pipeline\n");
                    fflush(stdout);
                }
#if OMP_DEBUG
                else
                {
                    printf("Thread=%d Encrypted data is %lu in segment_counter = %d nonce_counter=%d\n", omp_get_thread_num(), ciphertext_len, segment_counter, nonce_counter);
                    fflush(stdout);
                }
#endif


            } // for
        } // end of parallel region

           
            mpi_errno = MPI_Isend_original(large_send_buffer + MSG_HEADER_SIZE+ send_loc, inner_totaldata+ii*(ENC_MSG_TAG_SIZE + NONCE_HEADER), MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                               &request[request_counter++]);
           //  printf(COLOR_GREEN"    rank %d sending header to %d tag = %d data = %d"COLOR_RESET"\n",own_rank,dest,tag,inner_totaldata+ii*(ENC_MSG_TAG_SIZE + NONCE_HEADER));fflush(stdout);                                   
             prsd_segment += ii;
             send_loc += (inner_totaldata+ii*(ENC_MSG_TAG_SIZE + NONCE_HEADER));
             enc_loc += inner_totaldata; 
            // send_loc += (inner_totaldata+ii*(ENC_MSG_TAG_SIZE + NONCE_HEADER));
            // enc_loc += inner_totaldata;                                  
                                                 
#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
                pendingIsendRequestCount[dest] += 1;
#endif
#if OMP_DEBUG_V
                printf("        Thread=%d sent data segment_counter=%d\n", omp_get_thread_num(), segment_counter);
#endif

        } // segment_counter



#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || ONLY_ONE_THREAD_PIPELINE)
}
#endif

if ((totaldata > SUBKEY_GEN_START)){    
    for(i=0;i<my_thread_no;i++)
        EVP_AEAD_CTX_free(segment_ctx[i]);
}

#if PSC_THREAD_USE_DEBUG
    if(max_send_thread_used < my_thread_no)
        max_send_thread_used = my_thread_no;
#endif

 //  sendtag[dest] = tag;  

    // printf("request_counter=%d \n",request_counter);fflush(stdout);
    for(m=0;m<request_counter;m++){
        mpi_errno = MPI_Wait_original(&request[m],&status[m]);
#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION)   
        pendingIsendRequestCount[dest] -= 1; 
#endif           
    }
#if (OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION)   
        pendingIsendRequestCount[dest] -= 1;  // for the header
#endif

   // printf(COLOR_GREEN"    [send done ]%d sending to %d data = %d tag =%d"COLOR_RESET"\n",own_rank, dest, totaldata,tag );fflush(stdout);
   // printf("all send done\n");fflush(stdout);
    return mpi_errno;
}

/* base counter */
int MPI_SEC_BaseCounter_Pipeline_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                      MPI_Comm comm)
{

    int mpi_errno = MPI_SUCCESS;
    int outlen_enc, datasize, next;
    next = 0;
    int i, len, sendtype_sz, segments_no, chopping_tag;
    unsigned char iv_buffer[20];
    int th_data, ii, segment_counter, inner_totaldata, request_counter, inner_segment_counter;
    MPI_Request request[300];
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

    if(BASE_COUNTER_NO_PIPELINE)
        segments_no = 1;

    choping_sz = PIPELINE_SIZE;
    unsigned int adap_chop = (unsigned int)choping_sz;
    large_send_buffer[21] = (adap_chop >> 24) & 0xFF;
    large_send_buffer[22] = (adap_chop >> 16) & 0xFF;
    large_send_buffer[23] = (adap_chop >> 8) & 0xFF;
    large_send_buffer[24] = adap_chop & 0xFF;

    /* setting the size of the data */
    unsigned int temp_data = (unsigned int)totaldata;
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;

    /* Generate random bytes */
    if(BASE_COUNTER_LIBRARY_NONCE)
    {
        RAND_bytes(&large_send_buffer[4], 16);
        memcpy(iv_buffer, &large_send_buffer[4], 16);
    }
    else
    {
        memcpy(iv_buffer, Send_common_IV, 16);
       /* Prepare the counter */
        unsigned long temp_counter_to_send = base_global_counter;
        large_send_buffer[5] = (temp_counter_to_send >> 24) & 0xFF;
        large_send_buffer[6] = (temp_counter_to_send >> 16) & 0xFF;
        large_send_buffer[7] = (temp_counter_to_send >> 8) & 0xFF;
        large_send_buffer[8] = temp_counter_to_send & 0xFF;
    }

    

    int normal_counter_extra_header = 0; //8;

#if DEBUG_BASE_COUNTER
    MPID_Comm *comm_ptr_debug = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr_debug);
    int thisrank = comm_ptr_debug->rank;
   // printf(COLOR_YELLOW "[Rank %d] MPI_SEC_BaseCounter_Pipeline_Send totaldata %d segments_no %d choping_sz %d" COLOR_RESET "\n",
    //       thisrank, totaldata, segments_no, choping_sz); fflush(stdout);
#endif

    request_counter = 0;
    //mpi_errno = MPI_Isend_original(large_send_buffer, COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, tag++, comm,
    //                               &request[request_counter++]);
      mpi_errno = MPI_Send_original(large_send_buffer, COUNTER_HEADER_SIZE, MPI_CHAR, dest, tag++, comm);

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


         if(BASE_COUNTER_LIBRARY_NONCE)    
            IV_Count(iv_buffer, temp_counter);
         else
            IV_Count(iv_buffer, base_global_counter);

        if (1 != EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer))
            printf(COLOR_RED "___Error in initialization Base counter send___" COLOR_RESET "\n");

        if (1 != EVP_EncryptUpdate(global_counter_ctx, large_send_buffer + COUNTER_HEADER_SIZE + next, &outlen_enc, buf + next, inner_totaldata))
            printf(COLOR_RED "___Error in encryption___ Base counter send " COLOR_RESET "\n");
#if DEBUG_BASE_COUNTER
        else
        {
            printf(COLOR_GREEN "Base counter send[rank %d]: Encrypted data is %d  next %d segment_counter %d segments_no %d" COLOR_RESET "\n",
                   thisrank, outlen_enc, next, segment_counter, segments_no);
        }
#endif

        mpi_errno = MPI_Isend_original(&large_send_buffer[COUNTER_HEADER_SIZE + next], inner_totaldata, MPI_UNSIGNED_CHAR, dest, tag++, comm,
                                       &request[request_counter++]);
        next += inner_totaldata;
        base_global_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
        temp_counter += (unsigned long)(inner_totaldata - 1) / 16 + 1;
    }
    
    for(i = 0; i<request_counter; i++)
    {
        mpi_errno = MPI_Wait_original(&request[i],&status);
    }

    return mpi_errno;
}


void IV_Count(unsigned char *IV, unsigned long cter)
{
    uint32_t n = 16, c = (uint32_t)cter;
    do
    {
        --n;
        c += IV[n];
        IV[n] = (uint8_t)c;
        c >>= 8;
    } while (n);
}

 inline void IV_Count_out(unsigned char *IV, unsigned long cter, unsigned char *in)
{
    uint32_t n = 16, c = (uint32_t)cter;
    do
    {
        --n;
        c += in[n];
        IV[n] = (uint8_t)c;
        c >>= 8;
    } while (n);
}

/*
* Generate Common Encryption Mask with multi-threads
* --------------------------------------------------
* enc_common_end      = t_common_end
* enc_common_start    = t_common_start
* common_compute_size = t_compute_size
* enc_common_counter  = t_common_counter
*
*/
void multithreaded_generateCommonEncMask(int common_counter_gen_sz, int t_common_start, int t_common_end, int t_compute_size, 
                                     long unsigned int t_common_counter, unsigned char * encbuf,  EVP_CIPHER_CTX * t_counter_ctx)
{
    int blockamount, len, tempamount;
    unsigned char iv_buffer[20];

    if (t_compute_size <= (MAX_COMMON_COUNTER_SZ - common_counter_gen_sz - 32))
    {

        blockamount = ((common_counter_gen_sz - 1) / 16) * 16 + 16;

        if (t_common_end > t_common_start && t_common_end + blockamount <= MAX_COMMON_COUNTER_SZ)
        {
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, t_common_counter);
            EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, blockamount);
            //t_compute_size += blockamount;
            //t_common_end += blockamount;
            //t_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"Generated amount is blockamount %d common_counter_gen_sz %d t_common_end %d t_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, t_common_end, t_compute_size);fflush(stdout);
        }
        else if (t_common_end > t_common_start && t_common_end + blockamount > MAX_COMMON_COUNTER_SZ)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - t_common_end;

            /* check if blockamount is greater */
            if (blockamount > tempamount)
            { // added
                if (tempamount)
                {
                    memcpy(iv_buffer, Send_common_IV, 16);
                    IV_Count(iv_buffer, t_common_counter);
                    EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, tempamount);
                    t_compute_size += tempamount;
                    t_common_end += tempamount;
                    t_common_counter += (tempamount / 16);
                }
                blockamount = blockamount - tempamount;
                t_common_end = 0;
            }
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, t_common_counter);
            EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, blockamount);
            //t_compute_size += blockamount;
            //t_common_end += blockamount;
            //t_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"<>Generated amount is blockamount %d common_counter_gen_sz %d t_common_end %d t_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, t_common_end, t_compute_size);fflush(stdout);
        }
        else if (t_common_end < t_common_start && blockamount + t_common_end < t_common_start)
        {
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, t_common_counter);
            EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, blockamount);
            //t_compute_size += blockamount;
            //t_common_end += blockamount;
            //t_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"<..>Generated amount is blockamount %d common_counter_gen_sz %d t_common_end %d t_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, t_common_end, t_compute_size);fflush(stdout);
        }
        else if (t_common_end == t_common_start && t_compute_size == 0)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - t_common_end;
            if (blockamount > tempamount)
            {
                if (tempamount)
                {
                    memcpy(iv_buffer, Send_common_IV, 16);
                    IV_Count(iv_buffer, t_common_counter);
                    EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, tempamount);
                    t_compute_size += tempamount;
                    t_common_end += tempamount;
                    t_common_counter += (tempamount / 16);
                }
                blockamount = blockamount - tempamount;
                t_common_end = 0;
            }
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, t_common_counter);
            EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(t_counter_ctx, (encbuf+t_common_end), &len, zeros, blockamount);
            //t_compute_size += blockamount;
            //t_common_end += blockamount;
            //t_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"...Generated amount is blockamount %d common_counter_gen_sz %d t_common_end %d t_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, t_common_end, t_compute_size);fflush(stdout);
        }
        else
        {
            MPID_Comm *comm_ptr = NULL;
            MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
            printf(COLOR_RED "[Rank = %d]: ___ERROR___ in generation:: blockamount=%d, t_common_end=%d t_common_start=%d gnerateAmount=%d t_compute_size=%d" COLOR_RESET "\n",
                       comm_ptr->rank, blockamount, t_common_end, t_common_start, common_counter_gen_sz, t_compute_size);
            fflush(stdout);
            exit(EXIT_FAILURE);
        }
    }
    return;
}


/*
* Generate Common Encryption Mask
*/
void generateCommonEncMask(int common_counter_gen_sz)
{
    int blockamount, len, tempamount;
    unsigned char iv_buffer[20];

    if (common_compute_size <= (MAX_COMMON_COUNTER_SZ - common_counter_gen_sz - 1024))
    {

        blockamount = ((common_counter_gen_sz - 1) / 16) * 16 + 16;

        if (enc_common_end > enc_common_start && enc_common_end + blockamount <= MAX_COMMON_COUNTER_SZ)
        {
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, enc_common_counter);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, blockamount);
            common_compute_size += blockamount;
            enc_common_end += blockamount;
            enc_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"Generated amount is blockamount %d common_counter_gen_sz %d enc_common_end %d common_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, enc_common_end, common_compute_size);fflush(stdout);
        }
        else if (enc_common_end > enc_common_start && enc_common_end + blockamount > MAX_COMMON_COUNTER_SZ)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - enc_common_end;

            /* check if blockamount is greater */
            if (blockamount > tempamount)
            { // added
                if (tempamount)
                {
                    memcpy(iv_buffer, Send_common_IV, 16);
                    IV_Count(iv_buffer, enc_common_counter);
                    EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, tempamount);
                    common_compute_size += tempamount;
                    enc_common_end += tempamount;
                    enc_common_counter += (tempamount / 16);
                }
                blockamount = blockamount - tempamount;
                enc_common_end = 0;
            }
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, enc_common_counter);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, blockamount);
            common_compute_size += blockamount;
            enc_common_end += blockamount;
            enc_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"<>Generated amount is blockamount %d common_counter_gen_sz %d enc_common_end %d common_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, enc_common_end, common_compute_size);fflush(stdout);
        }
        else if (enc_common_end < enc_common_start && blockamount + enc_common_end < enc_common_start)
        {
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, enc_common_counter);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, blockamount);
            common_compute_size += blockamount;
            enc_common_end += blockamount;
            enc_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"<..>Generated amount is blockamount %d common_counter_gen_sz %d enc_common_end %d common_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, enc_common_end, common_compute_size);fflush(stdout);
        }
        else if (enc_common_end == enc_common_start && common_compute_size == 0)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - enc_common_end;
            if (blockamount > tempamount)
            {
                if (tempamount)
                {
                    memcpy(iv_buffer, Send_common_IV, 16);
                    IV_Count(iv_buffer, enc_common_counter);
                    EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
                    EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, tempamount);
                    common_compute_size += tempamount;
                    enc_common_end += tempamount;
                    enc_common_counter += (tempamount / 16);
                }
                blockamount = blockamount - tempamount;
                enc_common_end = 0;
            }
            memcpy(iv_buffer, Send_common_IV, 16);
            IV_Count(iv_buffer, enc_common_counter);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, &enc_common_buffer[enc_common_end], &len, zeros, blockamount);
            common_compute_size += blockamount;
            enc_common_end += blockamount;
            enc_common_counter += (blockamount / 16);
            //printf(COLOR_YELLOW"...Generated amount is blockamount %d common_counter_gen_sz %d enc_common_end %d common_compute_size %d"COLOR_RESET"\n",
            //blockamount, common_counter_gen_sz, enc_common_end, common_compute_size);fflush(stdout);
        }
        else
        {
            MPID_Comm *comm_ptr = NULL;
            MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
            /* possible reason: enc_common_start was not updated properly */
            printf(COLOR_RED "[Rank = %d]: ___ERROR___ in generation:: blockamount=%d, enc_common_end=%d enc_common_start=%d gnerateAmount=%d common_compute_size=%d" COLOR_RESET "\n",
                       comm_ptr->rank, blockamount, enc_common_end, enc_common_start, common_counter_gen_sz, common_compute_size);
            fflush(stdout);
            exit(EXIT_FAILURE);
        }
    }
    return;
}

/*
*   Encryption with common counter
*   next is buf position
*   pos is large_send_buf position
*/
void encryption_common_counter(const void *buf, int next, int enc_datasize, int pos, unsigned char * out_buf)
{
    int how_much_generate, temporary_datasize, datasize;
    unsigned char * enc_buffer, * in_buffer, * out_buffer;
    int len, n, tempamount, tempnext, blockamount;
    long unsigned int v1, v2;
    unsigned char iv_buffer[20]; 

    if (enc_datasize > common_compute_size)
    {
        how_much_generate = enc_datasize - common_compute_size;
        datasize = temporary_datasize = common_compute_size;
    }
    else
    {
        how_much_generate = 0;
        temporary_datasize = datasize = enc_datasize;
    }
#if DEBUG_PRECOMPUTE_COUNTER    
    printf("how_much_generate %d enc_common_counter %lu common_compute_size %d enc_datasize %d\n",
    how_much_generate,enc_common_counter,common_compute_size, enc_datasize);fflush(stdout);
#endif    
    if(common_compute_size > 0)
    {
        if (enc_common_end > enc_common_start)
        {
            if (enc_common_start + datasize <= enc_common_end)
                tempamount = datasize;
            else
                tempamount = enc_common_end - enc_common_start;
            len = tempamount;
            enc_buffer = &enc_common_buffer[enc_common_start];
            in_buffer = (unsigned char *)(buf + next);
            //out_buffer = &large_send_buffer[pos];
            out_buffer = out_buf+pos;
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
                while (len--)
                {
                    out_buf[pos + tempamount - len - 1] = (unsigned char)(enc_common_buffer[enc_common_start + tempamount - len - 1] ^ *((unsigned char *)(buf + next + tempamount - len - 1)));
                }
            }

            //printf("common_compute_size %d enc_common_start %d enc_common_end %d\n",common_compute_size, enc_common_start, enc_common_end);fflush(stdout);
            enc_common_start += ((tempamount - 1) / 16) * 16 + 16;
            if (enc_common_start >= MAX_COMMON_COUNTER_SZ)
                enc_common_start = 0;
            common_compute_size -= (((tempamount - 1) / 16) * 16 + 16);
            counter_needto_send += ((tempamount - 1) / 16) + 1;
        }
        else if (enc_common_end < enc_common_start)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - enc_common_start;
            tempnext = 0;
            if (datasize > tempamount)
            {
                if (tempamount)
                {
                    len = tempamount;
                    enc_buffer = &enc_common_buffer[enc_common_start];
                    in_buffer = (unsigned char *)(buf + next);
                    //out_buffer = &large_send_buffer[pos];
                    out_buffer = out_buf+pos;
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
                        while (len--)
                        {
                            out_buf[pos + tempamount - len - 1] = (unsigned char)(enc_common_buffer[enc_common_start + tempamount - len - 1] ^ *((unsigned char *)(buf + next + tempamount - len - 1)));
                        }
                    }

                    tempnext = tempamount;
                }
                enc_common_start = 0;
                datasize = datasize - tempamount;
            }
            len = datasize;
            enc_buffer = &enc_common_buffer[enc_common_start];
            in_buffer = (unsigned char *)(buf + next + tempnext);
            //out_buffer = &large_send_buffer[pos + tempnext];
            out_buffer = out_buf+pos+tempnext;
            
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
                while (len--)
                {
                    out_buf[pos + temporary_datasize - len - 1] = (unsigned char)(enc_common_buffer[enc_common_start + datasize - len - 1] ^ *((unsigned char *)(buf + next + temporary_datasize - len - 1)));
                }
            }

            //printf("<>common_compute_size %d enc_common_start %d enc_common_end %d\n",common_compute_size, enc_common_start, enc_common_end);fflush(stdout);
            if (datasize > 0)
                enc_common_start += ((datasize - 1) / 16) * 16 + 16;
            if (enc_common_start >= MAX_COMMON_COUNTER_SZ)
                enc_common_start = 0;
            common_compute_size -= (((temporary_datasize - 1) / 16) * 16 + 16);
            counter_needto_send += ((temporary_datasize - 1) / 16) + 1;
        }
        else if (enc_common_end == enc_common_start)
        {
            printf("___Here it should not come____\n");
            fflush(stdout);
        }
    }
    else if(common_compute_size < 0)
    {
        printf("common_compute_size = %d < 0: ERROR \n",common_compute_size);
    }

    /* api optimization */
    if (how_much_generate)
    {
       // printf("how_much_generate %d enc_common_counter %lu\n",how_much_generate,enc_common_counter);
        memcpy(iv_buffer, Send_common_IV, 16);
        IV_Count(iv_buffer, enc_common_counter);
        EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
        EVP_EncryptUpdate(global_counter_ctx, &out_buf[pos + temporary_datasize], &len, (buf + next + temporary_datasize), how_much_generate);

        enc_common_counter += (unsigned long)((how_much_generate - 1) / 16 + 1);
        counter_needto_send += ((how_much_generate - 1) / 16) + 1;
       /* blockamount = ((how_much_generate - 1) / 16) * 16 + 16;
        if (enc_common_start + blockamount >= MAX_COMMON_COUNTER_SZ)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - enc_common_start;
            enc_common_start = blockamount - tempamount;
        }
        else
        {
            enc_common_start += blockamount;
        }

        if (enc_common_end + blockamount >= MAX_COMMON_COUNTER_SZ)
        {
            tempamount = MAX_COMMON_COUNTER_SZ - enc_common_end;
            enc_common_end = blockamount - tempamount;
        }
        else
        {
            enc_common_end += blockamount;
        }*/

        // printf("how_much_generate = %d\n",how_much_generate);
    }

    return;
}

/*
*   Multi-thread Encryption for large messg
*   next is buf position
*   pos is large_send_buf position
*   t_start = enc_common_start
*   t_end = enc_common_end
*   t_cmnCM_sz = common_compute_size
*   mask = enc_common_buffer
*   out = large_send_buffer
* 
*/
void multithreads_encryption_common_counter_large_msg(const void *buf, int enc_datasize,  unsigned char * out,
                                             EVP_CIPHER_CTX * t_counter_ctx, long unsigned int counter, int tid)
{
    int len;
    unsigned char iv_buffer[20]; 

#if DEBUG_PRECOMPUTE_COUNTER    
    printf("[tid %d] counter %lu enc_datasize %d\n", tid,counter, enc_datasize);fflush(stdout);
#endif    
   
        memcpy(iv_buffer, (Send_common_IV+16), 16);
        IV_Count(iv_buffer, counter);
        EVP_EncryptInit_ex(t_counter_ctx, NULL, NULL, NULL, iv_buffer);
        EVP_EncryptUpdate(t_counter_ctx, (out), &len, (buf), enc_datasize);

        // printf("[tid %d] how_much_generate = %d\n",tid, how_much_generate);
    

    return;
}



/* precompute counter mode */
int MPI_SEC_PreComputeCounter_Send_v4(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                                   MPI_Comm comm)
{

    int mpi_errno = MPI_SUCCESS;
    int var, sendtype_sz, i, j;
    MPID_Comm *comm_ptr = NULL;
    unsigned long index_need_to_send;
    int totaldata, next;
    int pre_compute_size, segments_no, datasize;
    int flag;
    MPI_Status status;
    MPI_Request request[300];

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
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;

    /* Mark this is a Blocking Send */
    //large_send_buffer[13] = '1';

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

/* Mark this is a Blocking Send from Send */
#if DYNAMIC_PIPELINE
    if ((pendingIsendRequestCount[dest] + segments_no > MAX_PENDING_ISEND_LIMIT && segments_no > 1) || (totaldata >= SIXTY_4K && totaldata <= LARGE_SEGMENT_SIZE))
    {
        large_send_buffer[20] = '4';
        choping_sz = (totaldata - 1) / my_thread_no + 1;
        choping_sz = (choping_sz - 1) / 16 * 16 + 16; // make chooping size as multiple of 16
        segments_no = 1;                              // mark the segments_no to 1 segment
    }
    else
    {
        large_send_buffer[20] = '1';
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
    large_send_buffer[20] = '1';
#endif

    unsigned int adap_chop = (unsigned int)choping_sz;
    large_send_buffer[21] = (adap_chop >> 24) & 0xFF;
    large_send_buffer[22] = (adap_chop >> 16) & 0xFF;
    large_send_buffer[23] = (adap_chop >> 8) & 0xFF;
    large_send_buffer[24] = adap_chop & 0xFF;

    /* Prepare the counter */
    unsigned long temp_counter_to_send;
    if(totaldata < PRE_COM_DATA_RANGE)
    {
       if(common_compute_size < totaldata)
       {
        large_send_buffer[4] ='1';
        temp_counter_to_send = counter_needto_send_large_msg; // save it for later  
       }
       else
       {
        large_send_buffer[4] ='0';   
        temp_counter_to_send = counter_needto_send;
       }
    }
    else
    {
        temp_counter_to_send = counter_needto_send_large_msg;
    }
    
    large_send_buffer[5] = (temp_counter_to_send >> 24) & 0xFF;
    large_send_buffer[6] = (temp_counter_to_send >> 16) & 0xFF;
    large_send_buffer[7] = (temp_counter_to_send >> 8) & 0xFF;
    large_send_buffer[8] = temp_counter_to_send & 0xFF;

    request_counter = 0;
    mpi_errno = MPI_Send_original(large_send_buffer, COUNTER_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, tag++, comm);
    pendingIsendRequestCount[dest]+=1;

 
#if DEBUG_PRECOMPUTE_COUNTER
    MPID_Comm_get_ptr(comm, comm_ptr);
    printf(COLOR_CYAN "[Rank = %d]: dest =%d  totaldata =%d choping_sz = %d segments_no %d my_thread_no %d" COLOR_RESET "\n",
           comm_ptr->rank, dest, totaldata, choping_sz, segments_no, my_thread_no);
    fflush(stdout);

    /* Do a sanity check here 
     * enc_common_counter, start, end, compute size
    */
#endif
    if (totaldata < PRE_COM_DATA_RANGE)
    {
        next = 0;
        pos = 0;
        /*
        double percentage = (double)(common_compute_size/totaldata)*100;
        if(percentage >= 100)
            {printf(COLOR_GREEN"percentage %lf common_compute_size %d totaldata %d"COLOR_RESET"\n",percentage,common_compute_size,totaldata);fflush(stdout);}
        else if(percentage >= 75)
            {printf(COLOR_BLUE"percentage %lf common_compute_size %d totaldata %d"COLOR_RESET"\n",percentage, common_compute_size,totaldata);fflush(stdout);}
        else if(percentage >= 50)
            {printf(COLOR_CYAN"percentage %lf common_compute_size %d totaldata %d"COLOR_RESET"\n",percentage, common_compute_size,totaldata);fflush(stdout);} 
         else if(percentage >= 25)
            {printf(COLOR_MAGENTA"percentage %lf common_compute_size %d totaldata %d"COLOR_RESET"\n",percentage, common_compute_size,totaldata);fflush(stdout);}
        else
            {printf(COLOR_YELLOW"percentage %lf common_compute_size %d totaldata %d"COLOR_RESET"\n",percentage, common_compute_size,totaldata);fflush(stdout);}                                              
        */
        
        /*
         if(common_compute_size < totaldata)
        {
            common_compute_size = 0;
            enc_common_start = 0;
            enc_common_end = 0;
            counter_needto_send = enc_common_counter;
        } */
        
        if(common_compute_size >= totaldata)
            encryption_common_counter(buf, next, totaldata, pos, large_send_buffer);
        else
        {
            memcpy(iv_buffer, (Send_common_IV+16), 16);
            IV_Count(iv_buffer, enc_common_counter_long_msg);
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, iv_buffer);
            EVP_EncryptUpdate(global_counter_ctx, large_send_buffer, &len, buf, totaldata);
            enc_common_counter_long_msg += (unsigned long)(totaldata - 1) / 16 + 1;
            counter_needto_send_large_msg  += ((totaldata - 1) / 16) + 1;

        }     
       // printf("%2x %2x\n",enc_common_buffer[0],large_send_buffer[0]);fflush(stdout);
        mpi_errno = MPI_Isend_original(large_send_buffer, totaldata, MPI_CHAR, dest, tag++, comm, &request[request_counter++]);
        pendingIsendRequestCount[dest]+=1;

    }
    else
    {
        next = 0;
        pos = 0;
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

//#pragma omp parallel for default(none) schedule(static) private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data) shared(buf, large_send_buffer, enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, inner_totaldata, segment_counter, ii, totaldata, send_loc, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter_long_msg) num_threads(my_thread_no)
#pragma omp parallel default(none) private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data,iv_buffer,len) shared(buf, large_send_buffer, enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, inner_totaldata, segment_counter, ii, totaldata, send_loc, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter_long_msg,Send_common_IV) num_threads(my_thread_no)
        {
            #pragma omp for schedule(static) nowait private(inner_segment_counter, base, enc_data, enc_buffer, in_buffer, out_buffer, j, t_mask, t_mask_pos, t_counter, t_counter_data,iv_buffer,len) 
            for (inner_segment_counter = 0; inner_segment_counter < ii; inner_segment_counter++)
            {
                enc_data = th_data;
                if (inner_segment_counter == ii - 1)
                {
                    enc_data = inner_totaldata - th_data * (ii - 1);
                }

                base = send_loc + inner_segment_counter * th_data;

                //enc_buffer = (unsigned char *)(enc_common_buffer);
                in_buffer = (unsigned char *)(buf + base);
                out_buffer = (unsigned char *)(large_send_buffer + base);
                int tid = omp_get_thread_num();

                 
                    t_mask = 0;
                    t_mask_pos = 0;
                    t_counter_data = th_data * inner_segment_counter; // - common_compute_size;  
                    if(t_counter_data < 1)
                        t_counter = enc_common_counter_long_msg;
                    else
                        t_counter = enc_common_counter_long_msg + (unsigned long)((t_counter_data - 1) / 16 + 1);
                

#if DEBUG_PRECOMPUTE_COUNTER
                j = inner_segment_counter;
               // printf(COLOR_MAGENTA "[rank %d tid = %d]: inner_segment_counter %d tinfo_start=%d enc_common_end=%d tinfo_comsize = %d counter %lu enc_data =%d segment_counter=%d send_loc=%d base %d" COLOR_RESET "\n",
                //       rank, tid, j, tinfo[j].tstart, enc_common_end, tinfo[j].compute_size, tinfo[j].counter, enc_data, segment_counter, send_loc, base);
                 printf(COLOR_MAGENTA "[rank %d tid = %d]: inner_segment_counter  %d t_counter %lu enc_common_end=%d enc_data =%d segment_counter=%d send_loc=%d base %d" COLOR_RESET "\n",
                       rank, tid, j,  t_counter, enc_common_end,  enc_data, segment_counter, send_loc, base);       
                fflush(stdout);
#endif

                memcpy(iv_buffer, (Send_common_IV+16), 16);
                IV_Count(iv_buffer, t_counter);
                EVP_EncryptInit_ex(local_counter_ctx[inner_segment_counter], NULL, NULL, NULL, iv_buffer);
                EVP_EncryptUpdate(local_counter_ctx[inner_segment_counter], (out_buffer), &len, (in_buffer), enc_data);
                
                //multithreads_encryption_common_counter_large_msg(in_buffer, enc_data, out_buffer,
                //                                       local_counter_ctx[inner_segment_counter], t_counter, tid);


                
            } // end of for loop
        }// end of parallel region
            //encryption_common_counter(buf, next, inner_totaldata, pos, large_send_buffer);

            mpi_errno = MPI_Isend_original(large_send_buffer + send_loc, inner_totaldata, MPI_CHAR, dest, tag++, comm, &request[request_counter++]);
            
            // update common_compute_size
            // update enc_common_start
            // update enc_common_counter
           /* if(common_compute_size >= inner_totaldata)
            {
                common_compute_size -= (((inner_totaldata - 1) / 16) * 16 + 16);
                enc_common_start += ((inner_totaldata - 1) / 16) * 16 + 16;
                if (enc_common_start >= MAX_COMMON_COUNTER_SZ)
                {
                    enc_common_start = enc_common_start - MAX_COMMON_COUNTER_SZ;
                }
            }
            else
            {
                enc_common_counter += (unsigned long)(inner_totaldata - common_compute_size - 1) / 16 + 1;
                enc_common_start += common_compute_size; //((common_compute_size - 1) / 16) * 16 + 16;
                if (enc_common_start >= MAX_COMMON_COUNTER_SZ)
                {
                    enc_common_start = enc_common_start - MAX_COMMON_COUNTER_SZ;
                }
                common_compute_size = 0; 
            }*/
            
            enc_common_counter_long_msg += (unsigned long)(inner_totaldata - 1) / 16 + 1;
            send_loc += inner_totaldata;
            pendingIsendRequestCount[dest] += 1;
            //next += inner_totaldata;
            //pos += inner_totaldata;
        }
        // update counter_needto_send
        //counter_needto_send += ((totaldata - 1) / 16) + 1;
        counter_needto_send_large_msg  += ((totaldata - 1) / 16) + 1;
#if DEBUG_PRECOMPUTE_COUNTER
        MPID_Comm_get_ptr(comm, comm_ptr);
        printf(COLOR_CYAN "[Rank = %d]: dest =%d enc_common_start=%d enc_common_end=%d common_compute_size = %d enc_common_counter_long_msg %lu counter_needto_send =%d send_loc=%d" COLOR_RESET "\n",
               comm_ptr->rank, dest, enc_common_start, enc_common_end, common_compute_size, enc_common_counter_long_msg, counter_needto_send, send_loc);
        fflush(stdout);
#endif

    }

    /* Dynamic pre-computation */

    if (totaldata <= 16)
        common_counter_gen_sz = 16;
    else if (totaldata < 1024)
        common_counter_gen_sz = totaldata;
    else if (totaldata < 4096)
        common_counter_gen_sz = 1024;
    else
       common_counter_gen_sz = 4096;    
    
    
    /* Determine the thread number for pre-computation */

    if (totaldata < SIXTY_4K) // SIXTY_4K or 32K does not make any differences
    {
        //generateCommonEncMask(totaldata);
        for (i = 0; i < request_counter; i++)
        {
            flag = 0;
            MPI_Test(&request[i], &flag, &status);
            while (!flag)
            {
                generateCommonEncMask(totaldata);
                MPI_Test(&request[i], &flag, &status);
            }
            pendingIsendRequestCount[dest] -= 1;
        }
        pendingIsendRequestCount[dest] -= 1; // for header
    }        
    else
    {

        if (common_counter_gen_sz < THIRTY_2K)
            my_thread_no = 1;
        else if (common_counter_gen_sz < SIXTY_4K)
            my_thread_no = 4;
        else if (common_counter_gen_sz <= TWO_FIVE_6K)
            my_thread_no = 8; //8;
        else if (common_counter_gen_sz > TWO_FIVE_6K)
            my_thread_no = 16;

        if (my_thread_no > cyptmpi_series_thread)
            my_thread_no = cyptmpi_series_thread;

        /* Each thread will generate th_data amount mask, th_data%16 == 0*/
        // my_thread_no = 24;
        // th_data = FOUR_K; //common_counter_gen_sz;
        th_data = common_counter_gen_sz / my_thread_no;
        th_data = ((th_data - 1) / 16) * 16 + 16;
        segments_no = my_thread_no;
        int pre_com_data = 0;
        if (totaldata > ONE_M)
            totaldata = totaldata / 2;

        int t_end_pos, t_start_pos;

        for (i = 0; i < request_counter; i++)
        {
            flag = 0;
            MPI_Test(&request[i], &flag, &status);
            while (!flag)
            {
                //if((common_compute_size + th_data*segments_no) <= (MAX_COMMON_COUNTER_SZ - 16))
                if ((common_compute_size + th_data * segments_no) <= (MAX_COMMON_COUNTER_SZ - 16) && (pre_com_data + th_data * segments_no <= totaldata))
                {
//#pragma omp parallel for default(none) schedule(static) private(enc_buffer, j, t_counter, t_end_pos, t_start_pos) shared(enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter) num_threads(my_thread_no)
#pragma omp parallel default(none) private(enc_buffer, j, t_counter, t_end_pos, t_start_pos) shared(enc_common_buffer, local_counter_ctx, th_data, segments_no, stdout, my_thread_no, enc_common_end, rank, enc_common_start, common_compute_size, enc_common_counter) num_threads(my_thread_no)
            {
                #pragma omp for schedule(static) nowait private(enc_buffer, j, t_counter, t_end_pos, t_start_pos) 
    
                    for (j = 0; j < segments_no; j++)
                    {
                        t_end_pos = enc_common_end + th_data * j;
                        if (t_end_pos >= MAX_COMMON_COUNTER_SZ)
                        {
                            t_end_pos = t_end_pos - MAX_COMMON_COUNTER_SZ;
                        }
                        if (j > 0)
                            t_counter = enc_common_counter + (unsigned long)(((th_data * j) - 1) / 16 + 1);
                        else
                            t_counter = enc_common_counter;

                        t_start_pos = enc_common_start;
                        enc_buffer = enc_common_buffer;
#if DEBUG_PRECOMPUTE_COUNTER
                        int tid = omp_get_thread_num();

                        printf(COLOR_MAGENTA "[rank %d tid = %d]: j %d t_start_pos=%d t_end_pos=%d th_data = %d t_counter %lu common_compute_size =%d segments_no=%d" COLOR_RESET "\n",
                               rank, tid, j, t_start_pos, t_end_pos, th_data, t_counter, common_compute_size, segments_no);
                        fflush(stdout);
#endif
                        multithreaded_generateCommonEncMask(th_data, t_start_pos, t_end_pos, common_compute_size,
                                                            t_counter, enc_common_buffer, local_counter_ctx[j]);
                    } // end of for
                } // end of parallel region
                    // update enc_common_end
                    enc_common_end += (th_data * segments_no);
                    if (enc_common_end >= MAX_COMMON_COUNTER_SZ)
                    {
                        enc_common_end = enc_common_end - MAX_COMMON_COUNTER_SZ;
                    }
                    // update enc_common_counter
                    enc_common_counter += (unsigned long)(((th_data * segments_no) - 1) / 16 + 1);

                    // update common compute size
                    common_compute_size += (th_data * segments_no);
                    pre_com_data += (th_data * segments_no);
                    MPI_Test(&request[i], &flag, &status);
                }
                else
                {
                    mpi_errno = MPI_Wait_original(&request[i], &status);
                    flag = 1; // exit from while
                }
            }
            pendingIsendRequestCount[dest] -= 1;
        }
        pendingIsendRequestCount[dest] -= 1; // for header
        // printf("rank %d common_compute_size %d\n",rank,common_compute_size);fflush(stdout);
        //printf("my_thread_no %d common_compute_size %d\n",my_thread_no,common_compute_size);fflush(stdout);
    }
  

    return mpi_errno;
}



int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
             MPI_Comm comm)
{
    int mpi_errno = MPI_SUCCESS;
    int var, sendtype_sz, datasz;
    MPID_Comm *comm_ptr = NULL;

    MPID_Comm_get_ptr(comm, comm_ptr);
    var = MPI_Type_size(datatype, &sendtype_sz);
    datasz = count * sendtype_sz;

    int isSamenode = 0;

    if (dest > -1)
        isSamenode = sameNode[dest];
    //isSamenode = comm_ptr->intranode_table[dest];

    if (comm_ptr->rank == dest || dest < 0)
        mpi_errno = MPI_Send_original(buf, count, datatype, dest, tag, comm);
    //#if SKIP_INTRA_ENC
    else if (isSamenode > -1)
    {
        if (security_approach == 600) // encryt intra also
            mpi_errno = MPI_SEC_Multi_Thread_Send_OpenMP(buf, count, datatype, dest, tag, comm);
        else if (security_approach == 700) // encryt intra also    
             mpi_errno = MPI_SEC_BaseCounter_Pipeline_Send(buf, count, datatype, dest, tag, comm);
        else
            mpi_errno = MPI_Send_original(buf, count, datatype, dest, tag, comm);
    }
    //#endif
    else if (security_approach == 600 || security_approach == 601) //(OPENMP_MULTI_THREAD_ONLY)
        mpi_errno = MPI_SEC_Multi_Thread_Send_OpenMP(buf, count, datatype, dest, tag, comm);
    else if (security_approach == 602) //(OPENMP_PIPE_LINE)
        mpi_errno = MPI_SEC_MThreads_PipeLine_OpenMP_Send__largeSegment_3(buf, count, datatype, dest, tag, comm);

    else if (security_approach == 700 || security_approach == 701) //(BASE_COUNTER_MODE)
        mpi_errno = MPI_SEC_BaseCounter_Pipeline_Send(buf, count, datatype, dest, tag, comm);
    else if (security_approach == 702) // (PRE_COMPUTE_COUNTER_MODE)
        mpi_errno = MPI_SEC_PreComputeCounter_Send_v4(buf, count, datatype, dest, tag, comm);
    else
    {
        mpi_errno = MPI_Send_original(buf, count, datatype, dest, tag, comm);
    }

    return mpi_errno;
}

/* end of add */