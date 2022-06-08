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

#ifndef MPIDI_CH3_RDMA_POST_H
#define MPIDI_CH3_RDMA_POST_H

#include "mpichconf.h"
#include "vbuf.h"
#include "dreg.h"

/* Use this header to add implementation specific structures
   that cannot be defined until after the mpich2 header files
   have been included.
*/

#ifdef CRC_CHECK
void gen_crc_table();
unsigned long update_crc(unsigned long crc_accum, char *data_blk_ptr,
			 int data_blk_size);
#endif

typedef enum{                       
    FREE = 2,        
    BUSY                    
}put_get_list_entry_stat; 

/* structure MPIDI_CH3I_RDMA_put_get_list is the queue pool to record every
 * issued signaled RDMA write and RDMA read operation. The address of
 * the entries are assigned to the id field of the descriptors when they
 * are posted. So it will be easy to find the corresponding operators of
 * the RDMA operations when a completion queue entry is polled.
 */
struct MPIDI_CH3I_RDMA_put_get_list_t{
    int     op_type;
    /* op_type, SIGNAL_FOR_PUT or
     * SIGNAL_FOR_GET */
    MPIDI_msg_sz_t     data_size;
    struct  dreg_entry * mem_entry; /* mem region registered on the fly*/
    void    *target_addr;   /* get use only */
    void    *origin_addr;   /* get use only, tmp buffer for small msg,
                             * NULL if big msg, do need to do mem cpy
                             */
    int     completion; /* deregister when complete is 0 */
    int     target_rank;
    MPID_Win    *win_ptr;
    struct MPIDI_VC* vc_ptr;
    put_get_list_entry_stat status;
};

#define Calculate_IOV_len(_iov, _n_iov, _len)                   \
{   int _i; (_len) = 0;                                         \
    for (_i = 0; _i < (_n_iov); _i ++) {                        \
        (_len) += (_iov)[_i].MPL_IOV_LEN;                      \
    }                                                           \
}

#if defined(_ENABLE_CUDA_)

#define IS_CUDA_RNDV_REQ(rreq)  (NONE != rreq->mrail.cuda_transfer_mode)            
#else
#define MPIDI_CH3I_MRAIL_FREE_CUDA_RNDV_BUFFER(rreq)           
#define IS_CUDA_RNDV_REQ(rreq)  (0)               
#endif

#define MPIDI_CH3I_MRAILI_RREQ_RNDV_FINISH(rreq)                    \
{                                                                   \
    if (rreq != NULL) {                                             \
        if (rreq->mrail.d_entry != NULL) {                          \
            dreg_unregister(rreq->mrail.d_entry);                   \
            rreq->mrail.d_entry = NULL;                             \
        }                                                           \
        MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(rreq);                    \
    }                                                               \
    rreq->mrail.d_entry = NULL;                                     \
    rreq->mrail.protocol = MV2_RNDV_PROTOCOL_RENDEZVOUS_UNSPECIFIED;    \
}                                                                   \

#define PUSH_FLOWLIST(c) {                                      \
    if (0 == c->mrail.inflow) {                                 \
        c->mrail.inflow = 1;                                    \
        c->mrail.nextflow = flowlist;                           \
        flowlist = c;                                           \
    }                                                           \
}
 
#define POP_FLOWLIST() {                                        \
    if (flowlist != NULL) {                                     \
        MPIDI_VC_t *_c;                                         \
        _c = flowlist;                                          \
        flowlist = _c->mrail.nextflow;                          \
        _c->mrail.inflow = 0;                                   \
        _c->mrail.nextflow = NULL;                              \
    }                                                           \
}

#define ADD_PENDING_FLOWLIST(_c, _list) {                   \
    _c->mrail.nextflow = _list;                             \
    _list = _c;                                             \
}                                                           \

#define REMOVE_PENDING_FLOWLIST(_c, _list) {                \
    _c = _list;                                             \
    _list = _c->mrail.nextflow;                             \
    _c->mrail.nextflow = NULL;                              \
}
/*
 * Attached to each connection is a list of send handles that
 * represent rendezvous sends that have been started and acked but not
 * finished. When the ack is received, the send is placed on the list;
 * when the send is complete, it is removed from the list.  The list
 * is an "in progress sends" queue for this connection.  We need it to
 * remember what sends are pending and to remember the order sends
 * were acked so that we complete them in that order. This
 * prevents a situation where we receive an ack for message 1, block
 * because of flow control, receive an ack for message 2, and are able
 * to complete message 2 based on the new piggybacked credits.
 *
 * The list head and tail are given by the shandle_head and
 * shandle_tail entries on mv2_connection_t and the list is linked
 * through the nexthandle entry on a send handle.
 *
 * The queue is FIFO because we must preserve order, so we maintain
 * both a head and a tail.
 *
 */

#define RENDEZVOUS_IN_PROGRESS(c, s) {                          \
    MPIR_Request_add_ref(s);                                    \
    if (NULL == (c)->mrail.sreq_tail) {                         \
        (c)->mrail.sreq_head = (void *)(s);                     \
    } else {                                                    \
        ((MPID_Request *)                                       \
         (c)->mrail.sreq_tail)->mrail.next_inflow =             \
            (void *)(s);                                        \
    }                                                           \
    (c)->mrail.sreq_tail = (void *)(s);                         \
    ((MPID_Request *)(s))->mrail.next_inflow = NULL;            \
}

#define RENDEZVOUS_DONE(c) {                                    \
    MPID_Request *req = (c)->mrail.sreq_head;                   \
    (c)->mrail.sreq_head =                                      \
    ((MPID_Request *)                                           \
     (c)->mrail.sreq_head)->mrail.next_inflow;                  \
        if (NULL == (c)->mrail.sreq_head) {                     \
            (c)->mrail.sreq_tail = NULL;                        \
        }                                                       \
    MPID_Request_release(req);                                  \
}

#define MPIDI_CH3I_MRAIL_REVERT_RPUT(_sreq)                     \
{                                                               \
    if (MV2_RNDV_PROTOCOL_RGET == (_sreq)->mrail.protocol)          \
        (_sreq)->mrail.protocol = MV2_RNDV_PROTOCOL_RPUT;           \
}

#ifdef _ENABLE_CUDA_
#define MPIDI_CH3I_MRAIL_SET_PKT_RNDV_CUDA(_pkt, _req)          \
{                                                               \
    int _i, _k;                                                     \
    (_pkt)->rndv.protocol = (_req)->mrail.protocol;             \
        (_pkt)->rndv.num_cuda_blocks = MIN(((_req)->mrail.num_cuda_blocks - (_req)->mrail.cuda_block_offset), rdma_num_cuda_rndv_blocks); \
    if ( (MV2_RNDV_PROTOCOL_RPUT == (_pkt)->rndv.protocol) ||       \
            (MV2_RNDV_PROTOCOL_RGET == (_pkt)->rndv.protocol) ) {   \
        for (_i = 0; _i < (_pkt)->rndv.num_cuda_blocks; _i++) {   \
        (_pkt)->rndv.buffer_addr[_i] = (_req)->mrail.cuda_vbuf[_i]->buffer;    \
            for(_k = 0; _k < rdma_num_hcas; _k++) {             \
                (_pkt)->rndv.buffer_rkey[_i][_k] =              \
                    (_req)->mrail.cuda_vbuf[_i]->region->mem_handle[_k]->rkey;\
            }                                                   \
        }                                                       \
        (_pkt)->rndv.cuda_block_offset = (_req)->mrail.cuda_block_offset;  \
    }                                                           \
} 
#endif

#if defined(_ENABLE_CUDA_) && defined(HAVE_CUDA_IPC)
#define MPIDI_CH3I_MRAIL_SET_PKT_RNDV_CUDA_IPC(_pkt, _req)      \
{                                                               \
    if ( MV2_RNDV_PROTOCOL_RGET == (_pkt)->rndv.protocol            \
         && IS_CUDA_RNDV_REQ(_req)) {                           \
        (_pkt)->rndv.ipc_displ = (_req)->mrail.ipc_displ;       \
        (_pkt)->rndv.ipc_baseptr = (_req)->mrail.ipc_baseptr;   \
        (_pkt)->rndv.ipc_size = (_req)->mrail.ipc_size;         \
        MPIU_Memcpy(&(_pkt)->rndv.ipc_memhandle,                \
                    &(_req)->mrail.ipc_memhandle,               \
                    sizeof(cudaIpcMemHandle_t));                \
        MPIU_Memcpy(&(_pkt)->rndv.ipc_eventhandle,              \
                    &(_req)->mrail.ipc_eventhandle,             \
                    sizeof(cudaIpcEventHandle_t));              \
    }                                                           \
}
#else
#define MPIDI_CH3I_MRAIL_SET_PKT_RNDV_CUDA_IPC(_pkt, _req)
#endif

#define MPIDI_CH3I_MRAIL_SET_PKT_RNDV(_pkt, _req)               \
{                                                               \
    int _i;                                                     \
    (_pkt)->rndv.protocol = (_req)->mrail.protocol;             \
    if ( (MV2_RNDV_PROTOCOL_RPUT == (_pkt)->rndv.protocol) ||       \
            (MV2_RNDV_PROTOCOL_RGET == (_pkt)->rndv.protocol) ) {   \
        if (!IS_CUDA_RNDV_REQ(_req) && ((_req)->mrail.d_entry)) {   \
            for (_i = 0; _i < rdma_num_hcas; _i ++) {           \
                (_pkt)->rndv.rkey[_i] =                         \
                ((_req)->mrail.d_entry)->memhandle[_i]->rkey;   \
            }                                                   \
        }                                                       \
        (_pkt)->rndv.buf_addr = (_req)->mrail.rndv_buf;         \
    }                                                           \
    (_pkt)->rndv.reqtype = MPIDI_Request_get_type(_req);       \
    MPIDI_CH3I_MRAIL_SET_PKT_RNDV_CUDA_IPC(_pkt, _req);         \
}

#define MPIDI_CH3I_MRAIL_FREE_RNDV_BUFFER(req)                  \
do {                                                            \
    if (1 == req->mrail.rndv_buf_alloc                          \
            && NULL != req->mrail.rndv_buf) {                   \
        MPIU_Free(req->mrail.rndv_buf);                         \
        req->mrail.rndv_buf_alloc = 0;                          \
        req->mrail.rndv_buf_off = 0;                            \
        req->mrail.rndv_buf_sz = 0;                             \
        req->mrail.rndv_buf = NULL;                             \
    } else {                                                    \
        req->mrail.rndv_buf_off = 0;                            \
        req->mrail.rndv_buf_sz = 0;                             \
    }                                                           \
}while(0)

#define MPIDI_CH3I_MRAIL_SET_REMOTE_RNDV_INFO(_rndv,_req)       \
{                                                               \
    int _i;                                                     \
    (_rndv)->protocol = (_req)->mrail.protocol;                 \
    for (_i = 0; _i < rdma_num_hcas; _i ++)                     \
    (_rndv)->rkey[_i] = (_req)->mrail.rkey[_i];                 \
    (_rndv)->buf_addr = (_req)->mrail.remote_addr;              \
}

#define MPIDI_CH3I_MRAIL_SET_REQ_REMOTE_RNDV(_req,_pkt)         \
{                                                               \
    int _i;                                                     \
    (_req)->mrail.protocol = (_pkt)->rndv.protocol;             \
    if (  (MV2_RNDV_PROTOCOL_RPUT == (_pkt)->rndv.protocol) ||  \
          (MV2_RNDV_PROTOCOL_RGET == (_pkt)->rndv.protocol) ) {     \
        (_req)->mrail.remote_addr = (_pkt)->rndv.buf_addr;          \
        for (_i = 0; _i < rdma_num_hcas; _i ++)                     \
        (_req)->mrail.rkey[_i] = (_pkt)->rndv.rkey[_i];             \
    }                                                               \
}

/* Return type of the sending interfaces */
#define MPI_MRAIL_MSG_QUEUED (-1)

int MPIDI_CH3I_RDMA_cq_poll(void);

void MRAILI_Init_vc(struct MPIDI_VC* vc);

int MPIDI_CH3I_MRAILI_Eager_send(   struct MPIDI_VC* vc,
                                    MPL_IOV * iov,
                                    int n_iov,
                                    size_t len,
                                    int * num_bytes_ptr,
                                    vbuf **buf_handle);

/* Following functions are defined in ibv_channel_manager.c */

/* return type predefinition */
#define T_CHANNEL_NO_ARRIVE 0   
#define T_CHANNEL_EXACT_ARRIVE 1
#define T_CHANNEL_OUT_OF_ORDER_ARRIVE 2
#define T_CHANNEL_CONTROL_MSG_ARRIVE 3
#define T_CHANNEL_HYBRID_MSG_ARRIVE 4
#define T_CHANNEL_ERROR -1

int MPIDI_CH3I_MRAILI_Get_next_vbuf_local(struct MPIDI_VC* vc, vbuf** vbuf_handle, int is_blocking);
        
int MPIDI_CH3I_MRAILI_Get_next_vbuf(struct MPIDI_VC** vc_ptr, vbuf** vbuf_ptr); 

int MPIDI_CH3I_MRAILI_Waiting_msg(struct MPIDI_VC* vc, vbuf**, int blocking);

int (*MPIDI_CH3I_MRAILI_Cq_poll) (vbuf**, struct MPIDI_VC*, int, int);

void MRAILI_Send_noop(struct MPIDI_VC* c, int rail);

int MRAILI_Send_noop_if_needed(struct MPIDI_VC* vc, int rail);

int MRAILI_Send_rdma_credit_if_needed(struct MPIDI_VC* vc);

/* Following interface for rndv msgs */
void MPIDI_CH3I_MRAILI_Rendezvous_rput_push(struct MPIDI_VC* vc, MPID_Request * sreq);

void MPIDI_CH3I_MRAILI_Rendezvous_rget_push(struct MPIDI_VC* vc, MPID_Request * sreq);

#ifdef _ENABLE_UD_
/* UD ZCOPY RNDV interface */
void MPIDI_CH3I_MRAILI_Rendezvous_zcopy_push(struct MPIDI_VC * vc,
                                             MPID_Request * sreq,
                                             mv2_ud_zcopy_info_t *zcopy_info);
#endif

void MRAILI_Release_recv_rdma(vbuf *v);

extern struct MPIDI_VC* flowlist;

/* Post the buffers on an SRQ associated with a particular HCA */
int mv2_post_srq_buffers(int, int);
void async_thread(void *ctx);

int MPIDI_CH3I_MRAILI_Flush(void);
#endif /* MPIDI_CH3_RDMA_POST_H */
