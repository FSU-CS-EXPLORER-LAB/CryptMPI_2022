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
#include "datatype.h"

#define COPY_BUFFER_SZ 16384
#if !defined(MPIC_REQUEST_PTR_ARRAY_SIZE)
#define MPIC_REQUEST_PTR_ARRAY_SIZE 64
#endif

/* These functions are used in the implementation of collective
   operations. They are wrappers around MPID send/recv functions. They do
   sends/receives by setting the context offset to
   MPID_CONTEXT_INTRA_COLL or MPID_CONTEXT_INTER_COLL. */

#undef FUNCNAME
#define FUNCNAME MPIC_Probe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPID_Comm *comm_ptr;

    MPID_Comm_get_ptr( comm, comm_ptr );

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;
    
    mpi_errno = MPID_Probe(source, tag, comm_ptr, context_id, status);
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

 fn_exit:
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIR_Localcopy
#undef FCNAME
#define FCNAME "MPIR_Localcopy"
int MPIR_Localcopy(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                   void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype)
{
    int mpi_errno = MPI_SUCCESS;
    int sendtype_iscontig, recvtype_iscontig;
    MPI_Aint sendsize, recvsize, sdata_sz, rdata_sz, copy_sz;
    MPI_Aint true_extent, sendtype_true_lb, recvtype_true_lb;
    MPIU_CHKLMEM_DECL(1);
    MPID_MPI_STATE_DECL(MPID_STATE_MPIR_LOCALCOPY);

    MPID_MPI_FUNC_ENTER(MPID_STATE_MPIR_LOCALCOPY);

    MPID_Datatype_get_size_macro(sendtype, sendsize);
    MPID_Datatype_get_size_macro(recvtype, recvsize);

    sdata_sz = sendsize * sendcount;
    rdata_sz = recvsize * recvcount;

    /* if there is no data to copy, bail out */
    if (!sdata_sz || !rdata_sz)
        goto fn_exit;

#if defined(HAVE_ERROR_CHECKING)
    if (sdata_sz > rdata_sz) {
        MPIR_ERR_SET2(mpi_errno, MPI_ERR_TRUNCATE, "**truncate", "**truncate %d %d", sdata_sz, rdata_sz);
        copy_sz = rdata_sz;
    }
    else
#endif /* HAVE_ERROR_CHECKING */
        copy_sz = sdata_sz;

#if defined(_ENABLE_CUDA_)
    int sbuf_isdev = 0, rbuf_isdev = 0;
    MPID_Datatype *sdt_ptr, *rdt_ptr;
    MPID_Datatype_get_ptr(sendtype, sdt_ptr);
    MPID_Datatype_get_ptr(recvtype, rdt_ptr);
    if (rdma_enable_cuda) {
        sbuf_isdev = is_device_buffer(sendbuf);
        rbuf_isdev = is_device_buffer(recvbuf);
    }
#endif

    /* Builtin types is the common case; optimize for it */
    if ((HANDLE_GET_KIND(sendtype) == HANDLE_KIND_BUILTIN) &&
        HANDLE_GET_KIND(recvtype) == HANDLE_KIND_BUILTIN) {
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) {
            MPIU_Memcpy_CUDA((void *) recvbuf, (void *) sendbuf,
                    copy_sz, cudaMemcpyDefault);
            goto fn_exit;
        }
#endif
        MPIU_Memcpy(recvbuf, sendbuf, copy_sz);
        goto fn_exit;
    }

    MPIR_Datatype_iscontig(sendtype, &sendtype_iscontig);
    MPIR_Datatype_iscontig(recvtype, &recvtype_iscontig);

    MPIR_Type_get_true_extent_impl(sendtype, &sendtype_true_lb, &true_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &true_extent);

    if (sendtype_iscontig && recvtype_iscontig)
    {
#if defined(_ENABLE_CUDA_)
        if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) { 
                MPIU_Memcpy_CUDA((void *) ((char *)recvbuf + recvtype_true_lb),
                    (void *) ((char *)sendbuf + sendtype_true_lb),
                    copy_sz, cudaMemcpyDefault); 
        } else {     
#endif
#if defined(HAVE_ERROR_CHECKING)
        MPIR_ERR_CHKMEMCPYANDJUMP(mpi_errno,
                                  ((char *)recvbuf + recvtype_true_lb),
                                  ((char *)sendbuf + sendtype_true_lb),
                                  copy_sz);
#endif
            MPIU_Memcpy(((char *) recvbuf + recvtype_true_lb),
                   ((char *) sendbuf + sendtype_true_lb),
                   copy_sz);
#if defined(_ENABLE_CUDA_)
        }
#endif 
    }
    else if (sendtype_iscontig)
    {
        MPID_Segment seg;
	MPI_Aint last;

	MPID_Segment_init(recvbuf, recvcount, recvtype, &seg, 0);
	last = copy_sz;
#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) {
	    MPID_Segment_unpack_cuda(&seg, 0, &last, rdt_ptr, (char*)sendbuf + sendtype_true_lb);
    } else
#endif
    {
	    MPID_Segment_unpack(&seg, 0, &last, (char*)sendbuf + sendtype_true_lb);
    }
        MPIR_ERR_CHKANDJUMP(last != copy_sz, mpi_errno, MPI_ERR_TYPE, "**dtypemismatch");
    }
    else if (recvtype_iscontig)
    {
        MPID_Segment seg;
	MPI_Aint last;

	MPID_Segment_init(sendbuf, sendcount, sendtype, &seg, 0);
	last = copy_sz;
#if defined(_ENABLE_CUDA_)
    if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) {
	    MPID_Segment_pack_cuda(&seg, 0, &last, sdt_ptr, (char*)recvbuf + recvtype_true_lb);
    } else
#endif
    {
	    MPID_Segment_pack(&seg, 0, &last, (char*)recvbuf + recvtype_true_lb);
    }
        MPIR_ERR_CHKANDJUMP(last != copy_sz, mpi_errno, MPI_ERR_TYPE, "**dtypemismatch");
    }
    else
    {
	char * buf;
	MPIDI_msg_sz_t buf_off;
	MPID_Segment sseg;
	MPIDI_msg_sz_t sfirst;
	MPID_Segment rseg;
	MPIDI_msg_sz_t rfirst;
#if defined (_ENABLE_CUDA_)
    if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) {
         MPIU_Malloc_CUDA(buf, copy_sz);
    } else
#endif   
    {
        MPIU_CHKLMEM_MALLOC(buf, char *, COPY_BUFFER_SZ, mpi_errno, "buf");
    }

	MPID_Segment_init(sendbuf, sendcount, sendtype, &sseg, 0);
	MPID_Segment_init(recvbuf, recvcount, recvtype, &rseg, 0);

	sfirst = 0;
	rfirst = 0;
	buf_off = 0;

#if defined (_ENABLE_CUDA_)
    if (rdma_enable_cuda && (sbuf_isdev || rbuf_isdev)) {
        MPI_Aint last;
        last = copy_sz;

        MPID_Segment_pack_cuda(&sseg, sfirst, &last, rdt_ptr, buf);
        MPID_Segment_unpack_cuda(&rseg, rfirst, &last, sdt_ptr, buf);
        MPIU_Free_CUDA(buf);
    } else 
#endif
        {
	    while (1)
	    {
	        MPI_Aint last;
	        char * buf_end;
             
	        if (copy_sz - sfirst > COPY_BUFFER_SZ - buf_off)
	        {
	            last = sfirst + (COPY_BUFFER_SZ - buf_off);
	        }
	        else
	        {
	            last = copy_sz;
	        }
	        
	        MPID_Segment_pack(&sseg, sfirst, &last, buf + buf_off);
	        MPIU_Assert(last > sfirst);
	        
	        buf_end = buf + buf_off + (last - sfirst);
	        sfirst = last;
	        
	        MPID_Segment_unpack(&rseg, rfirst, &last, buf);
	        MPIU_Assert(last > rfirst);
             
	        rfirst = last;
             
	        if (rfirst == copy_sz)
	        {
	            /* successful completion */
	            break;
	        }
             
                /* if the send side finished, but the recv side couldn't unpack it, there's a datatype mismatch */
                MPIR_ERR_CHKANDJUMP(sfirst == copy_sz, mpi_errno, MPI_ERR_TYPE, "**dtypemismatch");        
             
                /* if not all data was unpacked, copy it to the front of the buffer for next time */
	        buf_off = sfirst - rfirst;
	        if (buf_off > 0)
	        {
	            memmove(buf, buf_end - buf_off, buf_off);
	        }
	    }
        }
    }
    
    
  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPID_MPI_FUNC_EXIT(MPID_STATE_MPIR_LOCALCOPY);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}


/* FIXME: For the brief-global and finer-grain control, we must ensure that
   the global lock is *not* held when this routine is called. (unless we change
   progress_start/end to grab the lock, in which case we must *still* make
   sure that the lock is not held when this routine is called). */
#undef FUNCNAME
#define FUNCNAME MPIC_Wait
#undef FCNAME
#define FCNAME "MPIC_Wait"
int MPIC_Wait(MPID_Request * request_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_WAIT);

    MPIDI_PT2PT_FUNC_ENTER(MPID_STATE_MPIC_WAIT);

    MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    if (request_ptr->kind == MPID_REQUEST_SEND)
        request_ptr->status.MPI_TAG = 0;

    if (!MPID_Request_is_complete(request_ptr))
    {
	MPID_Progress_state progress_state;
	
	MPID_Progress_start(&progress_state);
        while (!MPID_Request_is_complete(request_ptr))
	{
	    mpi_errno = MPID_Progress_wait(&progress_state);
	    if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
	}
	MPID_Progress_end(&progress_state);
    }

    if (request_ptr->kind == MPID_REQUEST_RECV)
        MPIR_Process_status(&request_ptr->status, errflag);

    MPIR_TAG_CLEAR_ERROR_BITS(request_ptr->status.MPI_TAG);

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_PT2PT_FUNC_EXIT(MPID_STATE_MPIC_WAIT);
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/* Fault-tolerance versions.  When a process fails, collectives will
   still complete, however the result may be invalid.  Processes
   directly communicating with the failed process can detect the
   failure, however another mechanism is needed to commuinicate the
   failure to other processes receiving the invalid data.  To do this
   we introduce the _ft versions of the MPIC_ helper functions.  These
   functions take a pointer to an error flag.  When this is set to
   TRUE, the send functions will communicate the failure to the
   receiver.  If a function detects a failure, either by getting a
   failure in the communication operation, or by receiving an error
   indicator from a remote process, it sets the error flag to TRUE.

   In this implementation, we indicate an error to a remote process by
   sending an empty message instead of the requested buffer.  When a
   process receives an empty message, it knows to set the error flag.
   We count on the fact that collectives that exchange data (as
   opposed to barrier) will never send an empty message.  The barrier
   collective will not communicate failure information this way, but
   this is OK since there is no data that can be received corrupted. */

#undef FUNCNAME
#define FUNCNAME MPIC_Send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Send(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                 MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPID_Request *request_ptr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SEND);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SEND);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        default:
            MPIR_TAG_SET_ERROR_BIT(tag);
    }

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    mpi_errno = MPID_Send(buf, count, datatype, dest, tag, comm_ptr,
                          context_id, &request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    if (request_ptr) {
        mpi_errno = MPIC_Wait(request_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPID_Request_release(request_ptr);
    }

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SEND);
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    if (request_ptr) MPID_Request_release(request_ptr);
    if (mpi_errno && !*errflag) {
        if (MPIX_ERR_PROC_FAILED == MPIR_ERR_GET_CLASS(mpi_errno)) {
            *errflag = MPIR_ERR_PROC_FAILED;
        } else {
            *errflag = MPIR_ERR_OTHER;
        }
    }
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}



/****************** Added by Mehran ********************/

#undef FUNCNAME
#define FUNCNAME MPIC_Send_Plus
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Send_Plus(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                 MPID_Comm *comm_ptr, MPID_Request **send_req_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SEND_PLUS);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SEND_PLUS);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        default:
            MPIR_TAG_SET_ERROR_BIT(tag);
    }

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    mpi_errno = MPID_Isend(buf, count, datatype, dest, tag, comm_ptr,
                          context_id, send_req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    // if (request_ptr) {
    //     mpi_errno = MPIC_Wait(request_ptr, errflag);
    //     if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    //     MPID_Request_release(request_ptr);
    // }

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SEND_PLUS);
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    // if (request_ptr) MPID_Request_release(request_ptr);
    if (mpi_errno && !*errflag) {
        if (MPIX_ERR_PROC_FAILED == MPIR_ERR_GET_CLASS(mpi_errno)) {
            *errflag = MPIR_ERR_PROC_FAILED;
        } else {
            *errflag = MPIR_ERR_OTHER;
        }
    }
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/*******************************************************/


/************* Added by Mohsen **************/
/**
 This functions Send/Recv msg by choppin
 /************************************************************************/
 
 
#undef FUNCNAME
#define FUNCNAME MPI_SEC_Chopping_Send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPI_SEC_Chopping_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                          MPID_Comm * comm_ptr, MPID_Request ** request , MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned long ciphertext_len, datasize, next;
    ciphertext_len = 0;
    int i, len, sendtype_sz, segments_no, chopping_tag, random_len;
    unsigned char newkey[40], V[50], nonce[50];
    unsigned int rank, segment_counter, d;
    MPID_Request *send_req_ptr = NULL;
    //chopping_tag = 1000;

    int context_id;

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    //EVP_AEAD_CTX *local_ctx = NULL;
    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);

    // MPID_Comm *comm_ptr = NULL;
    // MPID_Comm_get_ptr(comm, comm_ptr);
    rank = (unsigned int)comm_ptr->rank;
    d = (unsigned int)dest;
    //int choping_sz=32768;

#if 0 //CRYPTMPI_ADAPTIVE_CHOPP
    choping_sz = BLOCKING_CHOP_SZ;   
   // printf("AES: Send::  choping_sz=%d\n", choping_sz);fflush(stdout);
#endif 

    /* update the tag */
    //chopping_tag =  comm_ptr->rank + (tag+1) * 1024;
    chopping_tag = tag;

    MPI_Type_size(datatype, &sendtype_sz);
    unsigned long max_out_len = (unsigned long)(16 + (sendtype_sz * count));
    unsigned long totaldata = (unsigned long)count * sendtype_sz;

    /* setting the size of the data 
    large_send_buffer[0] = (totaldata >> 24) & 0xFF;
    large_send_buffer[1] = (totaldata >> 16) & 0xFF;
    large_send_buffer[2] = (totaldata >> 8) & 0xFF;
    large_send_buffer[3] = totaldata & 0xFF;*/

    /* Generate random bytes 
    RAND_bytes(&large_send_buffer[4], 16);
    memcpy(V, &large_send_buffer[4], 16);*/


    RAND_bytes(&large_send_buffer[0], 16);
    memcpy(V, &large_send_buffer[0], 16);
               
    /* Mark this is a Blocking Send from Send */
    large_send_buffer[20] = '1'; 
#if 0 //CRYPTMPI_ADAPTIVE_CHOPP
    unsigned int adap_chop = (unsigned int) choping_sz;
    large_send_buffer[21] = (adap_chop >> 24) & 0xFF;
    large_send_buffer[22] = (adap_chop >> 16) & 0xFF;
    large_send_buffer[23] = (adap_chop >> 8) & 0xFF;
    large_send_buffer[24] = adap_chop & 0xFF;
#endif  


    if (symmetric_key_size == 32)
    {
        for (i = 0; i < 16; i++)
            V[i + 16] = (~(int)V[i]);
    }

    /* Prepare Nonce */
    nonce[0] = '0';
    nonce[1] = '0';
    nonce[2] = '0';
    nonce[3] = '0';
    nonce[4] = '0';
    nonce[5] = '0';
    nonce[6] = '0';
	
	

    /* if (totaldata > choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE)
    {
        segments_no = 1;
        segments_no += (int)(totaldata - (choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE) - 1) / (choping_sz - ENC_MSG_TAG_SIZE) + 1;
    }
    else
    {
        segments_no = 1;
    } */

    segments_no = totaldata/ choping_sz;

    //if (1 != EVP_EncryptUpdate(ctx_enc, newkey, &len, V, symmetric_key_size))
        //printf("1");

    /* if (symmetric_key_size == 16)
    {
        local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                     newkey,
                                     16, 0);
    }
    else if (symmetric_key_size == 32)
    {
        local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(),
                                     newkey,
                                     32, 0);
    } */

    next = 0;
    /* encrypt each message segment */
    for (i = 0, segment_counter = 0; i < segments_no; i++, segment_counter++)
    {
        datasize = choping_sz;
        max_out_len = enc_choping_sz; //datasize + ENC_MSG_TAG_SIZE;

        if (i != segments_no - 1) nonce[7] = '0';
        else nonce[7] = '1';

        /*if (i != segments_no - 1)
        {
            nonce[7] = '0';

             if (i == 0)
            {
                nonce[7] = '0';
                //datasize = choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE;
                //max_out_len = datasize + ENC_MSG_TAG_SIZE;
            }
            else
            {
                nonce[7] = '0';
                //datasize = choping_sz - ENC_MSG_TAG_SIZE;
                //max_out_len = datasize + ENC_MSG_TAG_SIZE;
            }
        }
        else
        {

            nonce[7] = '1';

            if (i == 0)
            {
                nonce[7] = '1';
                datasize = (unsigned long)(totaldata);
                max_out_len = datasize + ENC_MSG_TAG_SIZE;
            }
            else if (i == 1)
            {
                nonce[7] = '1';
                datasize = (unsigned long)(totaldata) - (choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE);
                max_out_len = datasize + ENC_MSG_TAG_SIZE;
            }
            else
            {
                nonce[7] = '1';
                datasize = (unsigned long)(totaldata) - (segments_no - 2) * (choping_sz - ENC_MSG_TAG_SIZE);
                datasize = datasize - (choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE); // deduct first message
                max_out_len = datasize + ENC_MSG_TAG_SIZE;
            } 
        }*/

        nonce[8] = (segment_counter >> 24) & 0xFF;
        nonce[9] = (segment_counter >> 16) & 0xFF;
        nonce[10] = (segment_counter >> 8) & 0xFF;
        nonce[11] = segment_counter & 0xFF;

        if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer+(i*enc_choping_sz),
                                   &ciphertext_len, max_out_len,
                                   nonce, 12,
                                   buf + next, datasize,
                                   NULL, 0))
            {
                printf("Error in encryption: MPI_SEC_Choping_Send\n");
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                MPID_Comm *debug_comm_ptr = NULL;
                MPID_Comm_get_ptr(comm, debug_comm_ptr);
                printf("MPI_SEC_Chopping_Send: Ciphertext_len = %d [Rank = %d] totaldata=%d i=%d, seg=%d\n",
                       ciphertext_len, debug_comm_ptr->rank, totaldata, i, segments_no);
                fflush(stdout);
            }
#endif

            // mpi_errno = MPI_Send_original(large_send_buffer, ciphertext_len, MPI_CHAR, dest, chopping_tag, comm);
            // mpi_errno = MPID_Isend(large_send_buffer+(i*choping_sz), ciphertext_len, MPI_CHAR, dest, chopping_tag, comm_ptr, context_id, &send_req_ptr);
            mpi_errno = MPID_Isend(large_send_buffer+(i*enc_choping_sz), ciphertext_len, MPI_CHAR, dest, chopping_tag, comm_ptr, context_id, &send_req_ptr);

            next += (unsigned long)(choping_sz);


        //printf("send next is %lu\n", next); fflush(stdout);
        /*
        if (i == 0)
        {

            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer + MSG_HEADER_SIZE,
                                   &ciphertext_len, max_out_len,
                                   nonce, 12,
                                   buf + next, datasize,
                                   NULL, 0))
            {
                printf("Error in encryption: MPI_SEC_Choping_Send\n");
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                MPID_Comm *debug_comm_ptr = NULL;
                MPID_Comm_get_ptr(comm, debug_comm_ptr);
                printf("MPI_SEC_Chopping_Send: Ciphertext_len = %d [Rank = %d] totaldata=%d i=%d, seg=%d choping_sz=%d\n",
                       ciphertext_len, debug_comm_ptr->rank, totaldata, i, segments_no, choping_sz);
                fflush(stdout);
            }
#endif
            // mpi_errno = MPI_Send_original(large_send_buffer, ciphertext_len + MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, chopping_tag, comm);

            mpi_errno = MPID_Isend(large_send_buffer, ciphertext_len + MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, chopping_tag, comm_ptr, context_id, &send_req_ptr);
            //mpi_errno = MPIC_Isend(large_send_buffer, ciphertext_len + MSG_HEADER_SIZE, MPI_UNSIGNED_CHAR, dest, chopping_tag, comm_ptr, send_req_ptr,errflag);
			//if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        }
        else
        {

            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, large_send_buffer+(i*choping_sz),
                                   &ciphertext_len, max_out_len,
                                   nonce, 12,
                                   buf + next, datasize,
                                   NULL, 0))
            {
                printf("Error in encryption: MPI_SEC_Choping_Send\n");
                fflush(stdout);
            }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                MPID_Comm *debug_comm_ptr = NULL;
                MPID_Comm_get_ptr(comm, debug_comm_ptr);
                printf("MPI_SEC_Chopping_Send: Ciphertext_len = %d [Rank = %d] totaldata=%d i=%d, seg=%d\n",
                       ciphertext_len, debug_comm_ptr->rank, totaldata, i, segments_no);
                fflush(stdout);
            }
#endif

            // mpi_errno = MPI_Send_original(large_send_buffer, ciphertext_len, MPI_CHAR, dest, chopping_tag, comm);
            mpi_errno = MPID_Isend(large_send_buffer+(i*choping_sz), ciphertext_len, MPI_CHAR, dest, chopping_tag, comm_ptr, context_id, &send_req_ptr);

        }

        

        if (i == 0)
            next += (unsigned long)(choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE);
        else
            next += (unsigned long)(choping_sz - ENC_MSG_TAG_SIZE);
        */
            
        //chopping_tag++;
    }

    // EVP_AEAD_CTX_free(local_ctx);

    /* fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIR_TIMER_END(coll,allreduce,shm_rd);
	 
	fn_fail:
    goto fn_exit; */
	
	return (mpi_errno);
}




#undef FUNCNAME
#define FUNCNAME MPI_SEC_Chopping_Recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPI_SEC_Chopping_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                          MPID_Comm * comm_ptr, MPID_Request ** request , MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned char nonce[100], newkey[40], V[50];
    unsigned long next, decrypted_len, dest;
    MPI_Status sta;
    int i, recv_len, segments_no, len, recvtype_sz, chopping_tag;
    MPI_Type_size(datatype, &recvtype_sz);
    unsigned long totaldata = 0;
    unsigned int segment_counter;
    MPID_Request *recv_req_ptr = NULL;

    //EVP_AEAD_CTX *local_ctx = NULL;
    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);

    // MPID_Comm *comm_ptr = NULL;

    /* update the tag */
    chopping_tag = tag;
    int choping_sz=32768;

    int context_id;

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    /* MPID_Request **recv_req_ptr = NULL;
    MPIU_CHKLMEM_MALLOC(recv_req_ptr, MPID_Request **,
                        2 * 100 * sizeof (MPID_Request*),
                        mpi_errno, "recv_req_ptr");

    MPI_Status *starray;
    
    MPIU_CHKLMEM_MALLOC(starray, MPI_Status *, 200*sizeof(MPI_Status),
                        mpi_errno, "starray");
    
    int reqs = 0; */

    /* Received Header bytes  for data size with original tag */
    // mpi_errno = MPI_Recv_original(large_recv_buffer, (recvtype_sz * count) + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag, comm, &sta);
     mpi_errno = MPID_Irecv(large_recv_buffer, (recvtype_sz * count) + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag, comm_ptr, context_id, &recv_req_ptr); 
    //mpi_errno = MPIC_Irecv(large_recv_buffer, (recvtype_sz * count) + MSG_HEADER_SIZE + ENC_MSG_TAG_SIZE, MPI_UNSIGNED_CHAR, source, tag, comm_ptr,  &recv_req_ptr); 

    mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
    // mpi_errno = MPIC_Waitall(1,&recv_req_ptr[0],&starray[0], errflag);     


    totaldata = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);

#if 0 //CRYPTMPI_ADAPTIVE_CHOPP
            choping_sz = ((unsigned char)large_recv_buffer[24] << 0) | ((unsigned char)large_recv_buffer[23] << 8) | ((unsigned char)large_recv_buffer[22] << 16) | ((unsigned char)large_recv_buffer[21] << 24);           
          //  printf(COLOR_YELLOW"GCM: recv:choping_sz =%d flag=%c totaldata=%lu "COLOR_RESET "\n",choping_sz, large_recv_buffer[20],totaldata); fflush(stdout);     
#endif 
    memcpy(V, &large_recv_buffer[4], 16);
    if (symmetric_key_size == 32)
    {
        for (i = 0; i < 16; i++)
            V[i + 16] = (~(int)V[i]);
    }


    // if (1 != EVP_EncryptUpdate(ctx_enc, newkey, &len, V, symmetric_key_size))   printf("1");
 

    /* if (symmetric_key_size == 16)
    {
        local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                     newkey,
                                     16, 0);
    }
    else if (symmetric_key_size == 32)
    {
        local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_256_gcm(),
                                     newkey,
                                     32, 0);
    } */

    /* Prepare Nonce */
    nonce[0] = '0';
    nonce[1] = '0';
    nonce[2] = '0';
    nonce[3] = '0';
    nonce[4] = '0';
    nonce[5] = '0';
    nonce[6] = '0';

#if ENABLE_SECURE_MPI_DEBUG
    MPID_Comm_get_ptr(comm, comm_ptr);
    printf("MPI_SEC_Chopping_Recv[rank=%d]: totaldata will receive from sender =%lu\n", comm_ptr->rank, totaldata);
    fflush(stdout);
#endif

    if (totaldata <= choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE)
    {
        segment_counter = 0;
        nonce[7] = '1';
        nonce[8] = (segment_counter >> 24) & 0xFF;
        nonce[9] = (segment_counter >> 16) & 0xFF;
        nonce[10] = (segment_counter >> 8) & 0xFF;
        nonce[11] = segment_counter & 0xFF;

        if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, buf,
                               &decrypted_len, (totaldata),
                               nonce, 12,
                               &large_recv_buffer[MSG_HEADER_SIZE], totaldata + ENC_MSG_TAG_SIZE,
                               NULL, 0))
        {
            printf("Decryption error: Chooping Recv Zero [R=%d  D=%d]\n",comm_ptr->rank,source);
            fflush(stdout);
        }
#if ENABLE_SECURE_MPI_DEBUG
        else
        {
            MPID_Comm_get_ptr(comm, comm_ptr);
            printf(COLOR_BLUE "MPI_SEC_Chopping_3_Wait:[1 segment] Decrypted = %d Rank = %d" COLOR_RESET "\n", decrypted_len, comm_ptr->rank);
            fflush(stdout);
        }
#endif

       // EVP_AEAD_CTX_free(local_ctx);
    }
    else
    {
        dest = 0;
        segment_counter = 0;
        nonce[7] = '0';
        nonce[8] = (segment_counter >> 24) & 0xFF;
        nonce[9] = (segment_counter >> 16) & 0xFF;
        nonce[10] = (segment_counter >> 8) & 0xFF;
        nonce[11] = segment_counter & 0xFF;
        recv_len = choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE;

        if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, buf,
                               &decrypted_len, (recv_len),
                               nonce, 12,
                               &large_recv_buffer[MSG_HEADER_SIZE], recv_len + ENC_MSG_TAG_SIZE,
                               NULL, 0))
        {
            fprintf(stderr,"Decryption error: Chooping Recv First [C=%d R=%d  D=%d]\n",count,comm_ptr->rank,source);
            
        }
#if ENABLE_SECURE_MPI_DEBUG
        else
        {
            //MPID_Comm *debug_comm_ptr = NULL;
            MPID_Comm_get_ptr(comm, comm_ptr);
            printf(COLOR_BLUE "MPI_SEC_Chopping_3_Wait:[1st of many] Decrypted = %d Rank = %d segment_counter=%d" COLOR_RESET "\n", decrypted_len, comm_ptr->rank, segment_counter);
            fflush(stdout);
        }
#endif

        dest += decrypted_len;
        /*segments_no = (int)totaldata / choping_sz;
            if( totaldata % choping_sz != 0)
                segments_no++;*/

        if (totaldata > choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE)
        {
            segments_no = 1;
            segments_no += (int)(totaldata - (choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE) - 1) / (choping_sz - ENC_MSG_TAG_SIZE) + 1;
        }
        else
        {
            segments_no = 1;
        }

        int recv_data = 0;
        for (i = 1, segment_counter = 1; i < segments_no; i++, segment_counter++)
        {

            if (i != segments_no - 1)
            {
                nonce[7] = '0';
                recv_len = choping_sz - ENC_MSG_TAG_SIZE;
            }
            else
            {

                nonce[7] = '1';
                recv_len = (unsigned long)(totaldata) - (segments_no - 2) * (choping_sz - ENC_MSG_TAG_SIZE);
                recv_len = recv_len - (choping_sz - MSG_HEADER_SIZE - ENC_MSG_TAG_SIZE); // deduct first message
            }
            nonce[8] = (segment_counter >> 24) & 0xFF;
            nonce[9] = (segment_counter >> 16) & 0xFF;
            nonce[10] = (segment_counter >> 8) & 0xFF;
            nonce[11] = segment_counter & 0xFF;

            //printf("Receiving %d bytes segment = %d chopping_tag=%d\n", recv_len, i, chopping_tag); fflush(stdout);
            // mpi_errno = MPI_Recv_original(large_recv_buffer, recv_len + ENC_MSG_TAG_SIZE, MPI_CHAR, source, chopping_tag, comm, &sta);

            mpi_errno = MPID_Irecv(large_recv_buffer, recv_len + ENC_MSG_TAG_SIZE, MPI_CHAR, source, chopping_tag, comm_ptr, context_id, &recv_req_ptr); 

            mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
            // mpi_errno = MPIC_Waitall(1,&recv_req_ptr[0],&starray[0], errflag);     

            if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (buf + dest),
                                   &recv_data, (recv_len),
                                   nonce, 12,
                                   large_recv_buffer, (recv_len + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
            {
                fprintf(stderr,"Decryption error: chopping wait [C=%d R=%d  D=%d]\n",count,comm_ptr->rank,source);
                
            }
#if ENABLE_SECURE_CHOPP_DEBUG
            MPI_Comm_rank(MPI_COMM_WORLD, comm_ptr);
            printf(COLOR_BLUE "MPI_SEC_Chopping_3_Wait [Rank = %d ]:  Decrypted = %d  sz = %lu s = %d tag = [%d,%d] sg=%d i=%d\n \
                bufptr[waitCounter] =%llx" COLOR_RESET "\n",
                   comm_ptr->rank, recv_data, totaldata, source, tag, chopping_tag, segments_no, i, buf);
            fflush(stdout);
#endif
            dest += recv_data;

        } //i
      //  EVP_AEAD_CTX_free(local_ctx);
    }

    return mpi_errno;
}



/*****************************  End  **************************************/



#undef FUNCNAME
#define FUNCNAME MPIC_Recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Recv(void *buf, MPI_Aint count, MPI_Datatype datatype, int source, int tag,
                 MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPI_Status mystatus;
    MPID_Request *request_ptr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_RECV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_RECV);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (status == MPI_STATUS_IGNORE)
        status = &mystatus;

    mpi_errno = MPID_Recv(buf, count, datatype, source, tag, comm_ptr,
                          context_id, status, &request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    if (request_ptr) {
        mpi_errno = MPIC_Wait(request_ptr, errflag);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);

        *status = request_ptr->status;
        mpi_errno = status->MPI_ERROR;
        MPID_Request_release(request_ptr);
    } else {
        MPIR_Process_status(status, errflag);

        MPIR_TAG_CLEAR_ERROR_BITS(status->MPI_TAG);
    }

    if (MPI_SUCCESS == MPIR_ERR_GET_CLASS(status->MPI_ERROR)) {
        MPIU_Assert(status->MPI_TAG == tag);
    }

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_RECV);
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    if (request_ptr) MPID_Request_release(request_ptr);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

#undef FUNCNAME
#define FUNCNAME MPIC_Ssend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Ssend(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                  MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPID_Request *request_ptr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SSEND);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SSEND);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
            "**countneg", "**countneg %d", count);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        default:
            MPIR_TAG_SET_ERROR_BIT(tag);
    }

    mpi_errno = MPID_Ssend(buf, count, datatype, dest, tag, comm_ptr,
                           context_id, &request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    if (request_ptr) {
        mpi_errno = MPIC_Wait(request_ptr, errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPID_Request_release(request_ptr);
    }

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SSEND);
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    if (request_ptr) MPID_Request_release(request_ptr);
    if (mpi_errno && !*errflag) {
        if (MPIX_ERR_PROC_FAILED == MPIR_ERR_GET_CLASS(mpi_errno)) {
            *errflag = MPIR_ERR_PROC_FAILED;
        } else {
            *errflag = MPIR_ERR_OTHER;
        }
    }
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}


/**************************** Added by Mehran ***************************/
/**
 * This function is almost the same as the MPIC_Sendrecv.
 * The only difference is that we do not wait for the send in this function
 * 
/************************************************************************/
#undef FUNCNAME
#define FUNCNAME MPIC_Sendrecv_Plus
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Sendrecv_Plus(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                     int dest, int sendtag, void *recvbuf, MPI_Aint recvcount, 
                     MPI_Datatype recvtype, int source, int recvtag,
                     MPID_Comm *comm_ptr, MPI_Status *status, MPID_Request **send_req_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPI_Status mystatus;
    MPID_Request *recv_req_ptr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SENDRECV_PLUS);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SENDRECV_PLUS);

    MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    MPIR_ERR_CHKANDJUMP1((sendcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", sendcount);
    MPIR_ERR_CHKANDJUMP1((recvcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", recvcount);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (status == MPI_STATUS_IGNORE) status = &mystatus;
    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(sendtag);
        default:
            MPIR_TAG_SET_ERROR_BIT(sendtag);
    }

    mpi_errno = MPID_Irecv(recvbuf, recvcount, recvtype, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPID_Isend(sendbuf, sendcount, sendtype, dest, sendtag,
                           comm_ptr, context_id, send_req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    // mpi_errno = MPIC_Wait(send_req_ptr, errflag);
    // if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

    *status = recv_req_ptr->status;

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = recv_req_ptr->status.MPI_ERROR;

        /*if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr->status.MPI_ERROR;
            }*/
    }

    //MPID_Request_release(send_req_ptr);
    MPID_Request_release(recv_req_ptr);

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV_PLUS);
    return mpi_errno;
 fn_fail:
    // if (send_req_ptr)
    //     MPID_Request_release(send_req_ptr);
    if (recv_req_ptr)
        MPID_Request_release(recv_req_ptr);
    goto fn_exit;
}



/************ Mohsen *************/

#undef FUNCNAME
#define FUNCNAME MPIC_ENC_NAIVE_Sendrecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_ENC_NAIVE_Sendrecv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                     int dest, int sendtag, void *recvbuf, MPI_Aint recvcount,
                     MPI_Datatype recvtype, int source, int recvtag,
                     MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{
#if HELPER_FNS_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen); 
    
    if (PRINT_FUN_NAME){
        printf("[HELPER_FNS rank = %d host = %s  count=%d] Func: MPIC_ENC_NAIVE_Sendrecv [Secure]\n", comm_ptr->rank, hostname, sendcount);
        fflush(stdout);
    }
#endif 

    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPI_Status mystatus;
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
    MPID_Request *recv_req_ptr_header = NULL, *send_req_ptr_header = NULL;
	
	// EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);

    int recv_extent, send_extent;
    int pos, max_out_len, enc_recv_sz;
    unsigned long ciphertext_sendbuf_len, dec_count;
    MPID_Datatype_get_extent_macro(sendtype, send_extent);
    MPID_Datatype_get_extent_macro(recvtype, recv_extent);
   //printf("MPIC_ENC_NAIVE_sendredv\n");fflush(stdout);	
   // MPIDI_STATE_DECL(MPID_STATE_MPIC_SENDRECV);

    //MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SENDRECV);

   // MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    MPIR_ERR_CHKANDJUMP1((sendcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", sendcount);
    MPIR_ERR_CHKANDJUMP1((recvcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", recvcount);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (status == MPI_STATUS_IGNORE) status = &mystatus;
    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(sendtag);
        default:
            MPIR_TAG_SET_ERROR_BIT(sendtag);
    }

    //printf("    rank = %d recvcount = %d %d sendcount = %d %d \n",comm_ptr->rank,recvcount,recvcount*recv_extent,sendcount,sendcount*send_extent);fflush(stdout);
    //mpi_errno = MPID_Irecv(recvbuf, recvcount, recvtype, source, recvtag,
    //                       comm_ptr, context_id, &recv_req_ptr);
    /* mpi_errno = MPID_Irecv(recvbuf, recvcount*recv_extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr); */   
    mpi_errno = MPID_Irecv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr_header); 
    mpi_errno = MPID_Irecv(large_recv_buffer+MSG_HEADER_SIZE, recvcount*recv_extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr);  
                                          
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    //mpi_errno = MPID_Isend(sendbuf, sendcount, sendtype, dest, sendtag,
    //                       comm_ptr, context_id, &send_req_ptr);
   
    unsigned int temp_data = sendcount*send_extent;
    /* setting the size of the data */
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;
    //memcpy(&large_send_buffer[4], sendbuf, sendcount*send_extent);
    //printf("[rank %d send] source = %d dest = %d temp_data = %d Func: MPIC_ENC_NAIVE_Sendrecv\n",comm_ptr->rank, source, dest, temp_data);fflush(stdout);
 
    RAND_bytes(&large_send_buffer[4], NONCE_SIZE);
    mpi_errno = MPID_Isend(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr_header);  
 
    max_out_len = sendcount*send_extent+ENC_MSG_TAG_SIZE; 
    if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[4]), NONCE_SIZE,
                                   (char *)sendbuf, sendcount*send_extent,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPIC_ENC_NAIVE_Sendrecv rank = %d  count = %d\n",comm_ptr->rank,sendcount);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_GREEN"MPIC_ENC_NAIVE_Sendrecv: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dest);
                    //fflush(stdout);
                }
#endif
  
    /*mpi_errno = MPID_Isend(large_send_buffer, sendcount*send_extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr);*/
    /*  mpi_errno = MPID_Isend(sendbuf, sendcount*send_extent, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr); */                         
    mpi_errno = MPID_Isend(large_send_buffer+MSG_HEADER_SIZE, sendcount*send_extent+ENC_MSG_TAG_SIZE, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr);                       
 //printf("3\n");fflush(stdout);                       
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIC_Wait(send_req_ptr_header, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIC_Wait(send_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

  
    mpi_errno = MPIC_Wait(recv_req_ptr_header, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
 
    temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
    //printf("[rank %d recv] source = %d dest = %d temp_data = %d Func: MPIC_ENC_NAIVE_Sendrecv\n",comm_ptr->rank, source, dest, temp_data);fflush(stdout);
    //memcpy(recvbuf, &large_recv_buffer[4], temp_data);
    mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
     
    if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                   &dec_count, temp_data,
                                   (&large_recv_buffer[4]), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPIC_ENC_NAIVE_Sendrecv  rank = %d   count = %d   source =%d  \n",comm_ptr->rank,sendcount, source);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_BLUE"  MPIC_ENC_NAIVE_Sendrecv: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, source);
                //fflush(stdout);
            }
#endif           
   

       
    *status = recv_req_ptr->status;

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = recv_req_ptr->status.MPI_ERROR;

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr->status.MPI_ERROR;
        }
    }
  

    MPID_Request_release(send_req_ptr);
    MPID_Request_release(recv_req_ptr);
    MPID_Request_release(send_req_ptr_header);
    MPID_Request_release(recv_req_ptr_header);

 fn_exit:
    //MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    //MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV);
    return mpi_errno;
 fn_fail:
    if (send_req_ptr)
        MPID_Request_release(send_req_ptr);
    if (recv_req_ptr)
        MPID_Request_release(recv_req_ptr);
    goto fn_exit;
}





#undef FUNCNAME
#define FUNCNAME MPIC_NAIVE_PLUS_Sendrecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_NAIVE_PLUS_Sendrecv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                     int dest, int sendtag, void *recvbuf, MPI_Aint recvcount,
                     MPI_Datatype recvtype, int source, int recvtag,
                     MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{
#if HELPER_FNS_PRINT_FUN
    char hostname[100];
    int namelen;
    gethostname(hostname, &namelen); 
    
    if (PRINT_FUN_NAME){
        printf("[HELPER_FNS rank = %d host = %s  count=%d] Func: MPIC_NAIVE_PLUS_Sendrecv [SecurePlus]\n", comm_ptr->rank, hostname, sendcount);
        fflush(stdout);
    }
#endif 
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPI_Status mystatus;
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
    MPID_Request *recv_req_ptr_header = NULL, *send_req_ptr_header = NULL;

    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);

    int recv_extent, send_extent;
    int pos, max_out_len, enc_recv_sz;
    unsigned long ciphertext_sendbuf_len, dec_count;
    MPID_Datatype_get_extent_macro(sendtype, send_extent);
    MPID_Datatype_get_extent_macro(recvtype, recv_extent);
   //printf("MPIC_ENC_NAIVE_sendredv\n");fflush(stdout);	
   // MPIDI_STATE_DECL(MPID_STATE_MPIC_SENDRECV);

    //MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SENDRECV);

   // MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    MPIR_ERR_CHKANDJUMP1((sendcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", sendcount);
    MPIR_ERR_CHKANDJUMP1((recvcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", recvcount);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (status == MPI_STATUS_IGNORE) status = &mystatus;
    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(sendtag);
        default:
            MPIR_TAG_SET_ERROR_BIT(sendtag);
    }

    // fprintf(stderr,"    rank = %d recvcount = %d %d sendcount = %d %d  [MPIC_NAIVE_PLUS_Sendrecv]\n",comm_ptr->rank,recvcount,recvcount*recv_extent,sendcount,sendcount*send_extent);fflush(stdout);
    //mpi_errno = MPID_Irecv(recvbuf, recvcount, recvtype, source, recvtag,
    //                       comm_ptr, context_id, &recv_req_ptr);
    /* mpi_errno = MPID_Irecv(recvbuf, recvcount*recv_extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr); */   
    
    
    unsigned int chunk_size = sendcount*send_extent;
    
    mpi_errno = MPID_Irecv(large_recv_buffer, (recvcount*recv_extent)+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, source, recvtag, comm_ptr, context_id, &recv_req_ptr);  
                                          
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
 
    RAND_bytes(large_send_buffer, NONCE_SIZE);
    max_out_len = chunk_size+ENC_MSG_TAG_SIZE; 
    if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[0]), NONCE_SIZE,
                                   (char *)sendbuf, chunk_size,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPIC_NAIVE_PLUS_Sendrecv rank = %d  count = %d\n",comm_ptr->rank,sendcount);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_GREEN"MPIC_NAIVE_PLUS_Sendrecv: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dest);
                    //fflush(stdout);
                }
#endif
  
    /*mpi_errno = MPID_Isend(large_send_buffer, sendcount*send_extent+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr);*/
    /*  mpi_errno = MPID_Isend(sendbuf, sendcount*send_extent, MPI_CHAR, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr); */                         
    mpi_errno = MPID_Isend(large_send_buffer, (chunk_size)+ENC_MSG_TAG_SIZE+NONCE_SIZE, MPI_CHAR, dest, sendtag, comm_ptr, context_id, &send_req_ptr);                       
 //printf("3\n");fflush(stdout);                       
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);    

    mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

    
     
    if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)recvbuf,
                                   &dec_count, chunk_size,
                                   (&large_recv_buffer[0]), NONCE_SIZE,
                                   (large_recv_buffer + NONCE_SIZE), (chunk_size + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPIC_NAIVE_PLUS_Sendrecv  rank = %d   count = %d   source =%d  \n",comm_ptr->rank,sendcount, source);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_BLUE"  MPIC_NAIVE_PLUS_Sendrecv: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, source);
                //fflush(stdout);
            }
#endif           
   
    mpi_errno = MPIC_Wait(send_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
       
    *status = recv_req_ptr->status;

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = recv_req_ptr->status.MPI_ERROR;

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr->status.MPI_ERROR;
        }
    }
  

    MPID_Request_release(send_req_ptr);
    MPID_Request_release(recv_req_ptr);


 fn_exit:
    //MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    //MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV);
    return mpi_errno;
 fn_fail:
    if (send_req_ptr)
        MPID_Request_release(send_req_ptr);
    if (recv_req_ptr)
        MPID_Request_release(recv_req_ptr);
    goto fn_exit;
}





int MPIC_ENC_NAIVE_Recv(void *buf, MPI_Aint count, MPI_Datatype datatype, int source, int tag,
                 MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{

#if HELPER_FNS_PRINT_FUN
    if (PRINT_FUN_NAME){
		char hostname[100];
		int namelen;
		gethostname(hostname, &namelen); 
    
        printf("[HELPER_FNS rank = %d host = %s  count=%d] Func: MPIC_ENC_NAIVE_Recv [Secure]\n", comm_ptr->rank, hostname, count);
        fflush(stdout);
    }
#endif 

    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);

    //fprintf(stderr,"[2] rank = %d  [HELPER-RCV]\n",comm_ptr->rank);

    int mpi_errno = MPI_SUCCESS;
    int extent;
    unsigned long dec_count;
    MPID_Request *req_ptr = NULL;
    MPID_Request *req_ptr_header = NULL;

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPID_Datatype_get_extent_macro(datatype, extent);
    int amount = count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE; 

    #if 0

    mpi_errno = MPID_Irecv(large_recv_buffer, MSG_HEADER_SIZE, MPI_CHAR, source, tag,
                           comm_ptr, context_id, &req_ptr_header);

    mpi_errno = MPID_Irecv(large_recv_buffer+MSG_HEADER_SIZE, amount, MPI_CHAR, source, tag,
                           comm_ptr, context_id, &req_ptr);  

    if (mpi_errno) MPIR_ERR_POP(mpi_errno); 

    mpi_errno = MPIC_Wait(req_ptr_header, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
 
    unsigned int temp_data = ((unsigned char)large_recv_buffer[3] << 0) | ((unsigned char)large_recv_buffer[2] << 8) | ((unsigned char)large_recv_buffer[1] << 16) | ((unsigned char)large_recv_buffer[0] << 24);
     
    mpi_errno = MPIC_Wait(req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);
     
    if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)buf,
                                   &dec_count, temp_data,
                                   (&large_recv_buffer[4]), NONCE_SIZE,
                                   (large_recv_buffer + MSG_HEADER_SIZE), (temp_data + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPIC_ENC_NAIVE_Recv  rank = %d   count = %d   source =%d  \n",comm_ptr->rank,count, source);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_BLUE"  MPIC_ENC_NAIVE_Sendrecv: decrypted = %lu Rank = %d <-- source =%d"COLOR_RESET"\n", dec_count, comm_ptr->rank, source);
                //fflush(stdout);
            }
#endif  


    MPID_Request_release(req_ptr);
    MPID_Request_release(req_ptr_header);                            

#endif
    
    

// fprintf(stderr,"[3] rank = %d  [HELPER-RCV]\n",comm_ptr->rank);
    mpi_errno = MPIC_Recv(large_recv_buffer, amount, MPI_CHAR, source,
                                    tag, comm_ptr, status, errflag); 

// fprintf(stderr,"[4] rank = %d  [HELPER-RCV]\n",comm_ptr->rank);                                                     
    if (!EVP_AEAD_CTX_open(global_coll_msg_ctx, (char *)buf,
                                   &dec_count, (count*extent),
                                   (&large_recv_buffer[0]), NONCE_SIZE,
                                   (large_recv_buffer + NONCE_SIZE), (unsigned long)((count*extent) + ENC_MSG_TAG_SIZE),
                                   NULL, 0))
                {
                    printf("Decryption error MPIC_ENC_NAIVE_Recv source =%d  \n",source);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
            else
            {
                if (ENABLE_SECURE_DEBUG)  {printf(COLOR_BLUE"  MPIC_ENC_NAIVE_Recv: decrypted = %lu source =%d"COLOR_RESET"\n", dec_count, source);
                fflush(stdout);}
            }
#endif       
//fprintf(stderr,"[5] rank = %d  [HELPER-RCV]\n",comm_ptr->rank);



    fn_exit:
    //MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    //MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV);
    return mpi_errno;
 fn_fail:
    if (req_ptr)
        MPID_Request_release(req_ptr);
    if (req_ptr_header)
        MPID_Request_release(req_ptr_header);
    goto fn_exit;
}



int MPIC_ENC_NAIVE_Send(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                 MPID_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{

    //fprintf(stderr,"[1] rank = %d  [HELPER-SND]\n",comm_ptr->rank);
#if HELPER_FNS_PRINT_FUN
    if (PRINT_FUN_NAME){
		char hostname[100];
		int namelen;
		gethostname(hostname, &namelen);     
        printf("[HELPER_FNS rank = %d host = %s  count=%d] Func: MPIC_ENC_NAIVE_Send [Secure]\n", comm_ptr->rank, hostname, count);
        fflush(stdout);
    }
#endif 
    int mpi_errno = MPI_SUCCESS;
    unsigned int max_out_len;
    unsigned int ciphertext_sendbuf_len;
    int extent;

    // EVP_AEAD_CTX *local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),symmetric_key,symmetric_key_size, 0);


//fprintf(stderr,"[2] rank = %d  [HELPER-SND]\n",comm_ptr->rank); 

    
    
    unsigned long dec_count;
    MPID_Request *req_ptr = NULL;
    MPID_Request *req_ptr_header = NULL;

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    int context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    MPID_Datatype_get_extent_macro(datatype, extent);
    int amount = count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE;

    unsigned int temp_data = count*extent;

#if 0  

    /* setting the size of the data */
    large_send_buffer[0] = (temp_data >> 24) & 0xFF;
    large_send_buffer[1] = (temp_data >> 16) & 0xFF;
    large_send_buffer[2] = (temp_data >> 8) & 0xFF;
    large_send_buffer[3] = temp_data & 0xFF;

    RAND_bytes(&large_send_buffer[4], NONCE_SIZE);
    mpi_errno = MPID_Isend(large_send_buffer, MSG_HEADER_SIZE, MPI_CHAR, dest, tag,
                           comm_ptr, context_id, &req_ptr_header);  
 //fprintf(stderr,"[3] rank = %d  [HELPER-SND]\n",comm_ptr->rank); 
    max_out_len = temp_data+ENC_MSG_TAG_SIZE; 
    if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + MSG_HEADER_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[4]), NONCE_SIZE,
                                   (char *)buf, temp_data,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPIC_ENC_NAIVE_Sendrecv rank = %d  count = %d\n",comm_ptr->rank,count);
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)
						fprintf(stderr,COLOR_GREEN"MPIC_ENC_NAIVE_Sendrecv: ciphertext_sendbuf_len = %d Rank = %d --> dest =%d"COLOR_RESET"\n", ciphertext_sendbuf_len, comm_ptr->rank, dest);
                    //fflush(stdout);
                }
#endif
//fprintf(stderr,"[4] rank = %d  [HELPER-SND]\n",comm_ptr->rank); 
    mpi_errno = MPID_Isend(large_send_buffer+MSG_HEADER_SIZE, temp_data+ENC_MSG_TAG_SIZE, MPI_CHAR, dest, tag, comm_ptr, context_id, &req_ptr);                       
 //fprintf(stderr,"[5] rank = %d  [HELPER-SND]\n",comm_ptr->rank); 
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIC_Wait(req_ptr_header, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIC_Wait(req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    MPID_Request_release(req_ptr);
    MPID_Request_release(req_ptr_header);
#endif     
     
    
    RAND_bytes(large_send_buffer, NONCE_SIZE);

    // fprintf(stderr,"[3] rank = %d  [HELPER-SND]\n",comm_ptr->rank);
    max_out_len = count*extent+ENC_MSG_TAG_SIZE; 
            if (!EVP_AEAD_CTX_seal(global_coll_msg_ctx, (large_send_buffer + NONCE_SIZE),
                                   &ciphertext_sendbuf_len, max_out_len,
                                   (&large_send_buffer[0]), NONCE_SIZE,
                                   (char *)buf, (unsigned long) count*extent,
                                   NULL, 0))
                {
                    printf("Error in encryption: MPIC_ENC_NAIVE_Send rs\n");
                    fflush(stdout);
                }
#if ENABLE_SECURE_MPI_DEBUG
                else
                {
                    if (ENABLE_SECURE_DEBUG)  {printf(COLOR_GREEN"MPIC_ENC_NAIVE_Send: ciphertext_sendbuf_len = %d dest =%d"COLOR_RESET"\n",
                     ciphertext_sendbuf_len, dest);
                    fflush(stdout);}
                }
#endif
// fprintf(stderr,"[4] rank = %d  [HELPER-SND]\n",comm_ptr->rank);
    //int amount = count*extent+ENC_MSG_TAG_SIZE+NONCE_SIZE;
    mpi_errno = MPIC_Send(large_send_buffer, amount, MPI_CHAR, dest,
                                     tag, comm_ptr, errflag);
// fprintf(stderr,"[5] rank = %d  [HELPER-SND]\n",comm_ptr->rank);


    fn_exit:
    //MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    //MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV);
    return mpi_errno;
 fn_fail:
    if (req_ptr)
        MPID_Request_release(req_ptr);
    if (req_ptr_header)
        MPID_Request_release(req_ptr_header);
    goto fn_exit;
}

/* *************** End ***************** */


#undef FUNCNAME
#define FUNCNAME MPIC_Sendrecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Sendrecv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                     int dest, int sendtag, void *recvbuf, MPI_Aint recvcount,
                     MPI_Datatype recvtype, int source, int recvtag,
                     MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPI_Status mystatus;
    MPID_Request *recv_req_ptr = NULL, *send_req_ptr = NULL;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SENDRECV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SENDRECV);

    MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    MPIR_ERR_CHKANDJUMP1((sendcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", sendcount);
    MPIR_ERR_CHKANDJUMP1((recvcount < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", recvcount);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (status == MPI_STATUS_IGNORE) status = &mystatus;
    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(sendtag);
        default:
            MPIR_TAG_SET_ERROR_BIT(sendtag);
    }

    mpi_errno = MPID_Irecv(recvbuf, recvcount, recvtype, source, recvtag,
                           comm_ptr, context_id, &recv_req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPID_Isend(sendbuf, sendcount, sendtype, dest, sendtag,
                           comm_ptr, context_id, &send_req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIC_Wait(send_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPIC_Wait(recv_req_ptr, errflag);
    if (mpi_errno) MPIR_ERR_POPFATAL(mpi_errno);

    *status = recv_req_ptr->status;

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = recv_req_ptr->status.MPI_ERROR;

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = send_req_ptr->status.MPI_ERROR;
        }
    }

    MPID_Request_release(send_req_ptr);
    MPID_Request_release(recv_req_ptr);

 fn_exit:
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV);
    return mpi_errno;
 fn_fail:
    if (send_req_ptr)
        MPID_Request_release(send_req_ptr);
    if (recv_req_ptr)
        MPID_Request_release(recv_req_ptr);
    goto fn_exit;
}

/* NOTE: for regular collectives (as opposed to irregular collectives) calling
 * this function repeatedly will almost always be slower than performing the
 * equivalent inline because of the overhead of the repeated malloc/free */
#undef FUNCNAME
#define FUNCNAME MPIC_Sendrecv_replace
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                             int dest, int sendtag,
                             int source, int recvtag,
                             MPID_Comm *comm_ptr, MPI_Status *status, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    MPI_Status mystatus;
    MPIU_Context_id_t context_id_offset;
    MPID_Request *sreq = NULL;
    MPID_Request *rreq = NULL;
    void *tmpbuf = NULL;
    MPI_Aint tmpbuf_size = 0;
    MPI_Aint tmpbuf_count = 0;
    MPIU_CHKLMEM_DECL(1);
    MPIDI_STATE_DECL(MPID_STATE_MPIC_SENDRECV_REPLACE);
#ifdef MPID_LOG_ARROWS
    /* The logging macros log sendcount and recvcount */
    int sendcount = count, recvcount = count;
#endif

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_SENDRECV_REPLACE);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    if (status == MPI_STATUS_IGNORE) status = &mystatus;
    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(sendtag);
        default:
            MPIR_TAG_SET_ERROR_BIT(sendtag);
    }

    context_id_offset = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    if (count > 0 && dest != MPI_PROC_NULL) {
        MPIR_Pack_size_impl(count, datatype, &tmpbuf_size);
        MPIU_CHKLMEM_MALLOC(tmpbuf, void *, tmpbuf_size, mpi_errno, "temporary send buffer");

        mpi_errno = MPIR_Pack_impl(buf, count, datatype, tmpbuf, tmpbuf_size, &tmpbuf_count);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = MPID_Irecv(buf, count, datatype, source, recvtag,
                           comm_ptr, context_id_offset, &rreq);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPID_Isend(tmpbuf, tmpbuf_count, MPI_PACKED, dest,
                           sendtag, comm_ptr, context_id_offset, &sreq);
    if (mpi_errno != MPI_SUCCESS) {
        /* --BEGIN ERROR HANDLING-- */
        /* FIXME: should we cancel the pending (possibly completed) receive
         * request or wait for it to complete? */
        MPID_Request_release(rreq);
        MPIR_ERR_POP(mpi_errno);
        /* --END ERROR HANDLING-- */
    }

    mpi_errno = MPIC_Wait(sreq, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPIC_Wait(rreq, errflag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    *status = rreq->status;

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = rreq->status.MPI_ERROR;

        if (mpi_errno == MPI_SUCCESS) {
            mpi_errno = sreq->status.MPI_ERROR;
        }
    }

    MPID_Request_release(sreq);
    MPID_Request_release(rreq);

 fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_SENDRECV_REPLACE);
    return mpi_errno;
 fn_fail:
     if (sreq)
         MPID_Request_release(sreq);
     if (rreq)
         MPID_Request_release(rreq);
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIC_Isend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Isend(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                  MPID_Comm *comm_ptr, MPID_Request **request_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_ISEND);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_ISEND);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        default:
            MPIR_TAG_SET_ERROR_BIT(tag);
    }

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    mpi_errno = MPID_Isend(buf, count, datatype, dest, tag, comm_ptr,
            context_id, request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

 fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_ISEND);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIC_Issend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Issend(const void *buf, MPI_Aint count, MPI_Datatype datatype, int dest, int tag,
                  MPID_Comm *comm_ptr, MPID_Request **request_ptr, MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_ISSEND);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_ISSEND);

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "IN: errflag = %d", *errflag);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    switch(*errflag) {
        case MPIR_ERR_NONE:
            break;
        case MPIR_ERR_PROC_FAILED:
            MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        default:
            MPIR_TAG_SET_ERROR_BIT(tag);
    }

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    mpi_errno = MPID_Issend(buf, count, datatype, dest, tag, comm_ptr,
            context_id, request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

 fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_ISSEND);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIC_Irecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Irecv(void *buf, MPI_Aint count, MPI_Datatype datatype, int source,
                  int tag, MPID_Comm *comm_ptr, MPID_Request **request_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_IRECV);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_IRECV);

    MPIR_ERR_CHKANDJUMP1((count < 0), mpi_errno, MPI_ERR_COUNT,
                         "**countneg", "**countneg %d", count);

    context_id = (comm_ptr->comm_kind == MPID_INTRACOMM) ?
        MPID_CONTEXT_INTRA_COLL : MPID_CONTEXT_INTER_COLL;

    mpi_errno = MPID_Irecv(buf, count, datatype, source, tag, comm_ptr,
            context_id, request_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

 fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_IRECV);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIC_Waitall
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIC_Waitall(int numreq, MPID_Request *requests[], MPI_Status statuses[], MPIR_Errflag_t *errflag)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    MPI_Request request_ptr_array[MPIC_REQUEST_PTR_ARRAY_SIZE];
    MPI_Request *request_ptrs = request_ptr_array;
    MPI_Status status_static_array[MPIC_REQUEST_PTR_ARRAY_SIZE];
    MPI_Status *status_array = statuses;
    MPIDI_STATE_DECL(MPID_STATE_MPIC_WAITALL);
    MPIU_CHKLMEM_DECL(2);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIC_WAITALL);

    MPIU_DBG_MSG_S(PT2PT, TYPICAL, "IN: errflag = %s", *errflag?"TRUE":"FALSE");

    if (statuses == MPI_STATUSES_IGNORE) {
        status_array = status_static_array;
    }

    if (numreq > MPIC_REQUEST_PTR_ARRAY_SIZE) {
        MPIU_CHKLMEM_MALLOC(request_ptrs, MPI_Request *, numreq * sizeof(MPI_Request), mpi_errno, "request pointers");
        MPIU_CHKLMEM_MALLOC(status_array, MPI_Status *, numreq * sizeof(MPI_Status), mpi_errno, "status objects");
    }

    for (i = 0; i < numreq; ++i) {
        /* The MPI_TAG field is not set for send operations, so if we want
        to check for the error bit in the tag below, we should initialize all
        tag fields here. */
        status_array[i].MPI_TAG = 0;
        status_array[i].MPI_SOURCE = MPI_PROC_NULL;

        /* Convert the MPID_Request objects to MPI_Request objects */
        request_ptrs[i] = requests[i]->handle;
    }

    mpi_errno = MPIR_Waitall_impl(numreq, request_ptrs, status_array);

    /* The errflag value here is for all requests, not just a single one.  If
     * in the future, this function is used for multiple collectives at a
     * single time, we may have to change that. */
    for (i = 0; i < numreq; ++i) {
        MPIR_Process_status(&status_array[i], errflag);

        MPIR_TAG_CLEAR_ERROR_BITS(status_array[i].MPI_TAG);
    }

 fn_exit:
    if (numreq > MPIC_REQUEST_PTR_ARRAY_SIZE)
        MPIU_CHKLMEM_FREEALL();

    MPIU_DBG_MSG_D(PT2PT, TYPICAL, "OUT: errflag = %d", *errflag);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIC_WAITALL);
    return mpi_errno;
 fn_fail:
    goto fn_exit;
}
