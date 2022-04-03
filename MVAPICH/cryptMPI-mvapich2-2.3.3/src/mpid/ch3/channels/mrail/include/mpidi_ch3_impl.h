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

#if !defined(MPICH_MPIDI_CH3_IMPL_H_INCLUDED)
#define MPICH_MPIDI_CH3_IMPL_H_INCLUDED

#include "mpichconf.h"
#include "mpidimpl.h"
#include "mpiu_os_wrappers.h"
#include "mpl_utlist.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#if defined(_SMP_LIMIC_) || defined (_SMP_CMA_)
#include "smp_smpi.h"
#endif

#define MPIDI_CH3I_SPIN_COUNT_DEFAULT   100
#define MPIDI_CH3I_YIELD_COUNT_DEFAULT  5000

#define MPIDI_CH3I_READ_STATE_IDLE    0
#define MPIDI_CH3I_READ_STATE_READING 1

#define MPIDI_CH3I_CM_DEFAULT_ON_DEMAND_THRESHOLD           64
#define MPIDI_CH3I_RDMA_CM_DEFAULT_ON_DEMAND_THRESHOLD      512
#define MPIDI_CH3I_CM_DEFAULT_IWARP_ON_DEMAND_THRESHOLD     16
#define MPIDI_CH3I_RDMA_CM_DEFAULT_BASE_LISTEN_PORT         12000

typedef enum {
    MPIDI_CH3I_CM_BASIC_ALL2ALL,
    MPIDI_CH3I_CM_ON_DEMAND,
    MPIDI_CH3I_CM_RDMA_CM,
}MPIDI_CH3I_CM_type_t;

typedef struct MPIDI_CH3I_Process_s
{
    struct MPIDI_VC *vc;
    MPIDI_CH3I_CM_type_t cm_type;
    /*a flag to indicate whether new connection been established*/
    volatile int new_conn_complete;
    int num_conn;
#ifdef CKPT
    /*a flag to indicate some reactivation has finished*/
    volatile int reactivation_complete;
#endif
    int has_dpm;
}
MPIDI_CH3I_Process_t;

extern MPIDI_CH3I_Process_t MPIDI_CH3I_Process;
extern MPIDI_VC_t *mv2_read_progress_pending_vc;

extern int mv2_eager_fast_send(MPIDI_VC_t* vc, const void *buf,
                                MPIDI_msg_sz_t data_sz, int rank, int tag,
                                MPID_Comm *comm, int context_offset, MPID_Request **sreq_p);
extern int mv2_eager_fast_coalesce_send(MPIDI_VC_t* vc, const void *buf,
                                MPIDI_msg_sz_t data_sz, int rank, int tag,
                                MPID_Comm *comm, int context_offset, MPID_Request **sreq_p);
extern int mv2_eager_fast_rfp_send(MPIDI_VC_t* vc, const void *buf,
                                MPIDI_msg_sz_t data_sz, int rank, int tag,
                                MPID_Comm *comm, int context_offset, MPID_Request **sreq_p);
extern int mv2_smp_fast_write_contig(MPIDI_VC_t* vc, const void *buf,
                                MPIDI_msg_sz_t data_sz, int rank, int tag,
                                MPID_Comm *comm, int context_offset, MPID_Request **sreq_p);

#define MPIDI_CH3I_SendQ_enqueue(vc, req)                       \
{                                                               \
    /* MT - not thread safe! */                                 \
    MPIDI_DBG_PRINTF((50, FCNAME, "enqueue vc=%08p req=0x%08x", \
                  vc, req->handle));                            \
    req->dev.next = NULL;                                       \
    if (vc->ch.sendq_tail != NULL) {                            \
        vc->ch.sendq_tail->dev.next = req;                      \
    } else {                                                    \
        vc->ch.sendq_head = req;                                \
    }                                                           \
    vc->ch.sendq_tail = req;                                    \
    /* Disable direct send */                                   \
    vc->eager_fast_fn = NULL;                                   \
}

#define MPIDI_CH3I_SendQ_enqueue_head(vc, req)                              \
{                                                                           \
    /* MT - not thread safe! */                                             \
    MPIDI_DBG_PRINTF((50, FCNAME, "SendQ_enqueue_head vc=%08p req=0x%08x",  \
                  vc, req->handle));                                        \
    req->dev.next = vc->ch.sendq_head;                                      \
    if (vc->ch.sendq_tail == NULL) {                                        \
        vc->ch.sendq_tail = req;                                            \
    }                                                                       \
    vc->ch.sendq_head = req;                                                \
    /* Disable direct send */                                               \
    vc->eager_fast_fn = NULL;                                               \
}

#define MPIDI_CH3I_SendQ_dequeue(vc)                                    \
{                                                                       \
    /* MT - not thread safe! */                                         \
    MPIDI_DBG_PRINTF((50, FCNAME, "SendQ_dequeue vc=%08p req=0x%08x",   \
                    vc, vc->ch.sendq_head));                            \
    vc->ch.sendq_head = vc->ch.sendq_head->dev.next;                    \
    if (vc->ch.sendq_head == NULL) {                                    \
        vc->ch.sendq_tail = NULL;                                       \
        /* Re-enable direct send */                                     \
        if (mv2_use_eager_fast_send &&                                  \
            !(SMP_INIT && (vc->smp.local_nodes >= 0))) {                \
            if (likely(rdma_use_coalesce)) {                            \
                vc->eager_fast_fn = mv2_eager_fast_coalesce_send;       \
            } else {                                                    \
                vc->eager_fast_fn = mv2_eager_fast_send;                \
            }                                                           \
        }                                                               \
    }                                                                   \
}

#define MPIDI_CH3I_SendQ_head(vc) (vc->ch.sendq_head)

#define MPIDI_CH3I_SendQ_empty(vc) (vc->ch.sendq_head == NULL)

/* #define XRC_DEBUG */

#define MPIDI_CH3I_CM_SendQ_enqueue(vc, req)                            \
{                                                                       \
    /* MT - not thread safe! */                                         \
    MPIDI_DBG_PRINTF((50, FCNAME, "CM_SendQ_enqueue vc=%08p req=0x%08x",\
                    vc, req->handle));                                  \
    req->dev.next = NULL;                                               \
    if (vc->ch.cm_sendq_tail != NULL) {                                 \
        vc->ch.cm_sendq_tail->dev.next = req;                           \
    }  else {                                                           \
        vc->ch.cm_sendq_head = req;                                     \
    }                                                                   \
    vc->ch.cm_sendq_tail = req;                                         \
    /* Disable direct send */                                           \
    vc->eager_fast_fn = NULL;                                           \
}

#define MPIDI_CH3I_CM_SendQ_dequeue(vc)                                 \
{                                                                       \
    /* MT - not thread safe! */                                         \
    MPIDI_DBG_PRINTF((50, FCNAME, "CM_SendQ_dequeue vc=%08p req=0x%08x",\
                  vc, vc->ch.sendq_head));                              \
    vc->ch.cm_sendq_head = vc->ch.cm_sendq_head->dev.next;              \
    if (vc->ch.cm_sendq_head == NULL) {                                 \
        vc->ch.cm_sendq_tail = NULL;                                    \
        /* Re-enable direct send */                                     \
        if (mv2_use_eager_fast_send &&                                  \
            !(SMP_INIT && (vc->smp.local_nodes >= 0))) {                \
            if (likely(rdma_use_coalesce)) {                            \
                vc->eager_fast_fn = mv2_eager_fast_coalesce_send;       \
            } else {                                                    \
                vc->eager_fast_fn = mv2_eager_fast_send;                \
            }                                                           \
        }                                                               \
    }                                                                   \
}

#define MPIDI_CH3I_CM_SendQ_head(vc) (vc->ch.cm_sendq_head)

#define MPIDI_CH3I_CM_SendQ_empty(vc) (vc->ch.cm_sendq_head == NULL)

/* One sided sendq */
#define MPIDI_CH3I_CM_One_Sided_SendQ_enqueue(vc, v)                        \
{                                                                           \
    /* MT - not thread safe! */                                             \
    MPIDI_DBG_PRINTF((50, FCNAME, "CM_SendQ_enqueue vc=%08p vbuf=0x%08x",   \
                  vc, v));                                                  \
    v->desc.next = NULL;                                                    \
    if (vc->ch.cm_1sc_sendq_head != NULL) {                                 \
        vc->ch.cm_1sc_sendq_tail->desc.next = v;                            \
    } else {                                                                \
        vc->ch.cm_1sc_sendq_head = v;                                       \
    }                                                                       \
    vc->ch.cm_1sc_sendq_tail = v;                                           \
    /* Disable direct send */                                               \
    vc->eager_fast_fn = NULL;                                               \
}

#define MPIDI_CH3I_CM_One_Sided_SendQ_dequeue(vc)                           \
{                                                                           \
    /* MT - not thread safe! */                                             \
    MPIDI_DBG_PRINTF((50, FCNAME, "CM_SendQ_dequeue vc=%08p", vc));         \
    vc->ch.cm_1sc_sendq_head = vc->ch.cm_1sc_sendq_head->desc.next;         \
    if (vc->ch.cm_1sc_sendq_head == NULL) {                                 \
        vc->ch.cm_1sc_sendq_tail = NULL;                                    \
        /* Re-enable direct send */                                         \
        if (mv2_use_eager_fast_send &&                                      \
            !(SMP_INIT && (vc->smp.local_nodes >= 0))) {                    \
            if (likely(rdma_use_coalesce)) {                            \
                vc->eager_fast_fn = mv2_eager_fast_coalesce_send;       \
            } else {                                                    \
                vc->eager_fast_fn = mv2_eager_fast_send;                \
            }                                                           \
        }                                                                   \
    }                                                                       \
}

#define MPIDI_CH3I_CM_One_Sided_SendQ_head(vc) (vc->ch.cm_1sc_sendq_head)

#define MPIDI_CH3I_CM_One_Sided_SendQ_empty(vc) (vc->ch.cm_1sc_sendq_head == NULL)

/* RDMA channel interface */

int MPIDI_CH3I_Progress_init(void);
int MPIDI_CH3I_Progress_finalize(void);
int MPIDI_CH3I_Progress_register_hook(int (*progress_fn)(int*), int *id);
int MPIDI_CH3I_Progress_deregister_hook(int id);
int MPIDI_CH3I_Progress_activate_hook(int id);
int MPIDI_CH3I_Progress_deactivate_hook(int id);
int MPIDI_CH3I_Request_adjust_iov(MPID_Request *, MPIDI_msg_sz_t);
int MPIDI_CH3_Rendezvous_push(MPIDI_VC_t * vc, MPID_Request * sreq);
void MPIDI_CH3_Rendezvous_r3_push(MPIDI_VC_t * vc, MPID_Request * sreq);
void MPIDI_CH3I_MRAILI_Process_rndv(void);

int MPIDI_CH3I_read_progress(MPIDI_VC_t **vc_pptr, vbuf **, int *, int);
int MPIDI_CH3I_post_read(MPIDI_VC_t *vc, void *buf, int len);
int MPIDI_CH3I_post_readv(MPIDI_VC_t *vc, MPL_IOV *iov, int n);

/* RDMA implementation interface */

/*
MPIDI_CH3I_RDMA_init_process_group is called before the RDMA channel is initialized.  It's job is to initialize the
process group and determine the process rank in the group.  has_parent is used by spawn and you
can set it to FALSE.  You can copy the implementation found in the shm directory for this 
function if you want to use the PMI interface for process management (recommended).
*/
int MPIDI_CH3I_RDMA_init_process_group(int * has_parent, MPIDI_PG_t ** pg_pptr, int * pg_rank_ptr);

/*
MPIDI_CH3I_RDMA_init is called after the RDMA channel has been initialized
and the VC structures have been allocated.  VC stands for Virtual
Connection.  This should be the main initialization routine that fills in
any implementation specific fields to the VCs, connects all the processes to
each other and performs all other global initialization.  After this
function is called all the processes must be connected.  The ch channel
assumes a fully connected network.
*/
/* int MPIDI_CH3I_RDMA_init(void); */
int MPIDI_CH3I_RDMA_init(MPIDI_PG_t * pg, int pg_rank);

/* finalize releases the RDMA memory and any other cleanup */
int MPIDI_CH3I_RDMA_finalize(void);

/*
MPIDI_CH3I_RDMA_put_datav puts data into the ch memory of the remote process
specified by the vc.  It returns the number of bytes successfully written in
the num_bytes_ptr parameter.  This may be zero or up to the total amount of
data described by the input iovec.  The data does not have to arrive at the
destination before this function returns but the local buffers may be
touched.
*/
int MPIDI_CH3I_RDMA_put_datav(MPIDI_VC_t *vc, 
        MPL_IOV *iov, int n, int *num_bytes_ptr);

/*
MPIDI_CH3I_RDMA_read_datav reads data from the local ch memory into the user
buffer described by the iovec.  This function sets num_bytes_ptr to the
amout of data successfully read which may be zero.  This function only reads
data that was previously put by the remote process indentified by the vc.
*/
int MPIDI_CH3I_RDMA_read_datav(MPIDI_VC_t *vc, 
        MPL_IOV *iov, int n, int *num_bytes_ptr);

/********** Added interface for OSU-MPI2 ************/
int MPIDI_CH3I_MRAIL_PG_Init(MPIDI_PG_t *pg);

int MPIDI_CH3I_MRAIL_PG_Destroy(MPIDI_PG_t *pg);

int MPIDI_CH3_Rendezvous_rput_finish(MPIDI_VC_t *, 
        MPIDI_CH3_Pkt_rput_finish_t *);

int MPIDI_CH3_Rendezvous_rget_recv_finish(MPIDI_VC_t *, 
        MPID_Request *);

int MPIDI_CH3_Rendezvous_rget_send_finish(MPIDI_VC_t *, 
        MPIDI_CH3_Pkt_rget_finish_t *);

int MPIDI_CH3_Packetized_send(MPIDI_VC_t * vc, MPID_Request *);

int MPIDI_CH3_Packetized_recv_data(MPIDI_VC_t * vc, vbuf * v);

int MPIDI_CH3_Rendezvouz_r3_recv_data(MPIDI_VC_t * vc, vbuf * v);

void MPIDI_CH3_Rendezvouz_r3_ack_recv(MPIDI_VC_t * vc, 
				MPIDI_CH3_Pkt_rndv_r3_ack_t *r3ack_pkt);

int MPIDI_CH3I_MRAILI_Rendezvous_r3_ack_send(MPIDI_VC_t *vc);

int MPIDI_CH3_Packetized_recv_req(MPIDI_VC_t * vc, MPID_Request *);

int MPIDI_CH3_Rendezvous_unpack_data(MPIDI_VC_t *vc, MPID_Request *); 
#ifdef _ENABLE_UD_
/* UD ZCOPY RNDV interface */
void MPIDI_CH3I_MRAIL_Prepare_rndv_zcopy(MPIDI_VC_t * vc, MPID_Request * req);

void MPIDI_CH3_Rendezvous_zcopy_finish(MPIDI_VC_t * vc,
                             MPIDI_CH3_Pkt_zcopy_finish_t * zcopy_finish);

void MPIDI_CH3_Rendezvous_zcopy_ack(MPIDI_VC_t * vc,
                             MPIDI_CH3_Pkt_zcopy_ack_t * zcopy_ack);
#endif

/* Mrail interfaces*/
int MPIDI_CH3I_MRAIL_Prepare_rndv(
                MPIDI_VC_t * vc, MPID_Request * rreq);
int MPIDI_CH3I_MRAIL_Prepare_rndv_transfer(MPID_Request * sreq,
                MPIDI_CH3I_MRAILI_Rndv_info_t *rndv);
#if defined(_ENABLE_CUDA_)
void MRAILI_RDMA_Put_finish_cuda(MPIDI_VC_t * vc,
          MPID_Request * sreq, int rail, int is_cuda_pipeline, 
            int cuda_pipeline_finish, int offset);
int MPIDI_CH3I_MRAILI_Process_cuda_finish(MPIDI_VC_t * vc, 
            MPID_Request * rreq, MPIDI_CH3_Pkt_rput_finish_t * rf_pkt);
void MPIDI_CH3I_MRAILI_Rendezvous_rput_push_cuda(MPIDI_VC_t * vc,
        MPID_Request * sreq);
void MPIDI_CH3I_MRAIL_Send_cuda_cts_conti(MPIDI_VC_t * vc, MPID_Request * req);
void MPIDI_CH3_Rendezvous_cuda_cts_conti(MPIDI_VC_t * vc,
                           MPIDI_CH3_Pkt_cuda_cts_cont_t * cts_pkt);
#endif

int MPIDI_CH3I_MRAILI_Get_rndv_rput(MPIDI_VC_t *vc,
        MPID_Request * req,
        MPIDI_CH3I_MRAILI_Rndv_info_t * rndv,
        MPL_IOV *);

int MPIDI_CH3I_MRAIL_Parse_header(MPIDI_VC_t * vc, 
        vbuf * v, void **, int *headersize);

int handle_read(MPIDI_VC_t * vc, vbuf * v);

int MPIDI_CH3I_MRAIL_Fill_Request(MPID_Request *, 
        vbuf *v, int header_size, int * nb);

void  MPIDI_CH3I_MRAIL_Release_vbuf(vbuf * v);

int MPIDI_CH3I_MRAIL_Finish_request(MPID_Request *);

/* Connection Management Interfaces */

/* MPIDI_CH3I_CM_Init should replace MPIDI_CH3I_RDMA_init if dynamic
 * connection is enabled. */
int MPIDI_CH3I_CM_Init(MPIDI_PG_t * pg, int pg_rank, char **str);

/* MPIDI_CH3I_CM_Finalize should be used if MPIDI_CH3I_CM_Init is used
 * in initialization */
int MPIDI_CH3I_CM_Finalize(void);

/* MPIDI_CH3I_RDMA_CM_Init should replace MPIDI_CH3I_CM_init if 
 * RDMA_CM is enabled. */
int MPIDI_CH3I_RDMA_CM_Init(MPIDI_PG_t * pg, int pg_rank, char **str);

/* MPIDI_CH3I_RDMA_CM_Finalize should be used if MPIDI_CH3I_RDMA_CM_Init is used
 * in initialization */
int MPIDI_CH3I_RDMA_CM_Finalize(void);

/* MPIDI_CH3I_CM_Get_port_info gets the connection information in ifname */
int MPIDI_CH3I_CM_Get_port_info(char *ifname, int max_len);

int MPIDI_CH3I_CM_Connect_raw_vc(MPIDI_VC_t *vc, char *ifname);

/* Let the lower layer flush out anything from the ext_sendq
 * and reclaim all WQEs
 */
int MPIDI_CH3_Flush(void);

/* MPIDI_CH3I_CM_Connect should be called before using a VC to do
 * communication */
int MPIDI_CH3I_CM_Connect(MPIDI_VC_t * vc);

/* Establish connection to self */
int MPIDI_CH3I_CM_Connect_self(MPIDI_VC_t * vc);

/* MPIDI_CH3I_CM_Establish should be called when detecting the first message
 * from a VC */
int MPIDI_CH3I_CM_Establish(MPIDI_VC_t * vc);
void MPIDI_CH3I_Cleanup_after_connection(MPIDI_VC_t *vc);
int MPIDI_CH3I_MRAIL_CM_Alloc(MPIDI_PG_t * pg);
int MPIDI_CH3I_MRAIL_CM_Dealloc(MPIDI_PG_t * pg);
void MPIDI_CH3I_Cleanup_cqes(void);

/*flag to check if cq_poll is success in the progressing loop*/
int cq_poll_completion;

#ifdef CKPT

/*Following CM functions will only be useful for CR*/

/*Disconnect a connection that is not used for a while*/
int MPIDI_CH3I_CM_Disconnect(MPIDI_VC_t * vc);

/*Suspend connections in use*/
/*vc_vector is an array of pg_size. Use NULL to fill the vc which should not
 *  * be suspended*/
int MPIDI_CH3I_CM_Suspend(MPIDI_VC_t ** vc_vector);

/*Reactivate previously suspended connections*/
int MPIDI_CH3I_CM_Reactivate(MPIDI_VC_t ** vc_vector);

/*Send all the logged message after channel reactivated*/
int MPIDI_CH3I_CM_Send_logged_msg(MPIDI_VC_t * vc);

/*CM message handler for RC message in progress engine*/
void MPIDI_CH3I_CM_Handle_recv(MPIDI_VC_t * vc, 
        MPIDI_CH3_Pkt_type_t msg_type, vbuf * v);

void MPIDI_CH3I_CM_Handle_send_completion(MPIDI_VC_t * vc, 
        MPIDI_CH3_Pkt_type_t msg_type, vbuf * v);

/* Initialization and finalization for CR */
int MPIDI_CH3I_CR_Init(MPIDI_PG_t *pg, int rank, int size);

int MPIDI_CH3I_CR_Finalize();

/* CR message handler in progress engine */
void MPIDI_CH3I_CR_Handle_recv(MPIDI_VC_t * vc, 
        MPIDI_CH3_Pkt_type_t msg_type, vbuf * v);

void MPIDI_CH3I_CR_Handle_send_completion(MPIDI_VC_t * vc, 
        MPIDI_CH3_Pkt_type_t msg_type, vbuf * v);

/* CR lock to protect upper layers from accessing communication channel */
void MPIDI_CH3I_CR_lock();

void MPIDI_CH3I_CR_unlock();

/* Functions to enqueue/dequeue request involving memory registration. e.g.
 *  * rndv recv */
void MPIDI_CH3I_CR_req_enqueue(MPID_Request * req, MPIDI_VC_t * vc);

void MPIDI_CH3I_CR_req_dequeue(MPID_Request * req);

typedef enum MPICR_cr_state
{
    MPICR_STATE_RUNNING,
    MPICR_STATE_REQUESTED,
    MPICR_STATE_PRE_COORDINATION,
    MPICR_STATE_CHECKPOINTING,
    MPICR_STATE_POST_COORDINATION,
    MPICR_STATE_RESTARTING,
    MPICR_STATE_ERROR,
} MPICR_cr_state;

MPICR_cr_state MPIDI_CH3I_CR_Get_state();

#endif

#define MPIDI_CH3I_SMP_SendQ_enqueue(vc, req)                            \
{                                                                        \
    /* MT - not thread safe! */                                          \
    PRINT_DEBUG(DEBUG_SHM_verbose>1,                                     \
            "SendQ_enqueue vc: %p, dst: %d, req: %p, type: %d, ch.reqtype: %d\n", \
            vc, vc->pg_rank, req, MPIDI_Request_get_type(req), req->ch.reqtype);  \
    req->dev.next = NULL;                                                \
    MPIR_Request_add_ref(req);                                           \
    if (vc->smp.sendq_tail != NULL)                                      \
    {                                                                    \
        vc->smp.sendq_tail->dev.next = req;                              \
    }                                                                    \
    else                                                                 \
    {                                                                    \
        vc->smp.sendq_head = req;                                        \
    }                                                                    \
    vc->smp.sendq_tail = req;                                            \
    if (vc->smp.send_active == NULL) {                                   \
          vc->smp.send_active =  vc->smp.sendq_head;                     \
    }                                                                    \
    /* Disable direct send */                                            \
    vc->eager_fast_fn = NULL;                                            \
}                                                                        

#define MPIDI_CH3I_SMP_SendQ_enqueue_head(vc, req)                            \
{                                                                             \
    /* MT - not thread safe! */                                               \
    PRINT_DEBUG(DEBUG_SHM_verbose>1,                                          \
            "SendQ_enqueue_head vc: %p, dst: %d, req: %p, type: %d, ch.reqtype: %d\n", \
            vc, vc->pg_rank, req, MPIDI_Request_get_type(req), req->ch.reqtype);       \
    MPIR_Request_add_ref(req);                                                \
    req->dev.next = vc->smp.sendq_head;                                       \
    if (vc->smp.sendq_tail == NULL)                                           \
    {                                                                         \
        vc->smp.sendq_tail = req;                                             \
    }                                                                         \
    vc->smp.sendq_head = req;                                                 \
    /* Disable direct send */                                                 \
    vc->eager_fast_fn = NULL;                                                 \
}

#define MPIDI_CH3I_SMP_SendQ_dequeue(vc)                                      \
{                                                                             \
    MPID_Request *req = vc->smp.sendq_head;                                   \
    /* MT - not thread safe! */                                               \
    PRINT_DEBUG(DEBUG_SHM_verbose>1,                                          \
            "SendQ_dequeue vc: %p, dst: %d, req: %p, type: %d, ch.reqtype: %d\n",   \
            vc, vc->pg_rank, req, MPIDI_Request_get_type(req), req->ch.reqtype);    \
    vc->smp.sendq_head = vc->smp.sendq_head->dev.next;                        \
    if (vc->smp.sendq_head == NULL)                                           \
    {                                                                         \
        vc->smp.sendq_tail = NULL;                                            \
        /* Enable direct send */                                              \
        if (mv2_use_eager_fast_send) {                                        \
            vc->eager_fast_fn = mv2_smp_fast_write_contig;                    \
        }                                                                     \
    }                                                                         \
    MPID_Request_release(req);                                                \
}

#define MPIDI_CH3I_SMP_SendQ_head(vc) (vc->smp.sendq_head)

#define MPIDI_CH3I_SMP_SendQ_empty(vc) (vc->smp.sendq_head == NULL)

extern int SMP_INIT;
extern int SMP_ONLY;

enum {
    MRAIL_RNDV_NOT_COMPLETE,
    MRAIL_RNDV_NEARLY_COMPLETE,
    MRAIL_SMP_RNDV_NOT_START
};

/* management informations */
struct smpi_var {
    volatile void *mmap_ptr;
    void *send_buf_pool_ptr;
    unsigned int my_local_id;
    unsigned int num_local_nodes;
    short int only_one_device;  /* to see if all processes are on one physical node */

    unsigned int *l2g_rank;
    size_t available_queue_length;
    int fd;
    int fd_pool;
    /*
    struct smpi_send_fifo_req *send_fifo_head;
    struct smpi_send_fifo_req *send_fifo_tail;
    unsigned int send_fifo_queued;
    unsigned int *local_nodes; 
    int pending;
    */
};

typedef enum{
    SMP_DMA_NONE,
    SMP_DMA_LIMIC,
    SMP_DMA_CMA
}smp_dma_flag_t;

typedef enum{
    ONE_FREE = 1,
    TWO_FREE
}smp_ctrl_avail_flag_t;

typedef enum{
    NO_FALLBACK = 0,
    FALLBACK
}smp_fallback_flag_t;

extern struct smpi_var g_smpi;

extern int mv2_shmem_pool_init;

void MPIDI_CH3I_set_smp_only();

void MPIDI_CH3I_SMP_Init_VC(MPIDI_VC_t *vc);

int MPIDI_CH3I_SMP_write_progress(MPIDI_PG_t *pg);

int MPIDI_CH3I_SMP_read_progress(MPIDI_PG_t *pg);

int MPIDI_CH3I_SMP_init(MPIDI_PG_t *pg);

int MPIDI_CH3I_SMP_attach_shm_pool();

int MPIDI_CH3I_SMP_finalize(void);

void MPIDI_CH3I_SMP_writev_rndv_header(MPIDI_VC_t * vc, const MPL_IOV * iov,
	const int n, int *num_bytes_ptr);
	
void MPIDI_CH3I_SMP_writev_rndv_data_cont(MPIDI_VC_t * vc, MPID_Request *req,
    const MPL_IOV * iov, const int n, int *num_bytes_ptr);
	
int MPIDI_CH3I_SMP_writev_rndv_data(MPIDI_VC_t * vc, MPID_Request *req, 
    const MPL_IOV * iov, const int n, int *num_bytes_ptr);

#if defined(_ENABLE_CUDA_)
int MPIDI_CH3I_SMP_writev_rndv_data_cuda(MPIDI_VC_t * vc, MPID_Request *req,
    const MPL_IOV * iov, const int n, int *num_bytes_ptr, int is_cont);

void smp_cuda_send_copy_complete(MPIDI_VC_t * vc, MPID_Request *req, 
    void *ptr_flag);

int MPIDI_CH3I_SMP_readv_rndv_cuda(MPIDI_VC_t *recv_vc_ptr, MPID_Request *req,
    const MPL_IOV * iov, const int iov_count, int index, size_t *num_bytes_ptr, 
    int is_cont);

void smp_cuda_recv_copy_complete(MPIDI_VC_t * vc, MPID_Request *req, 
    void *recv_buf_ptr);
#endif

void MPIDI_CH3I_SMP_writev(MPIDI_VC_t * vc, const MPL_IOV * iov,
                          const int n, int *num_bytes_ptr);

void MPIDI_CH3I_SMP_write_contig(MPIDI_VC_t * vc, MPIDI_CH3_Pkt_type_t reqtype,
                          const void * buf, MPIDI_msg_sz_t data_sz, int rank,
                          int tag, MPID_Comm * comm, int context_offset, 
                          int *num_bytes_ptr);
                          
int MPIDI_CH3I_SMP_readv_rndv_cont(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
        const int iovlen, int index, void * l_header,
        void *c_header, size_t *num_bytes_ptr, smp_dma_flag_t dma_flag);
	
int MPIDI_CH3I_SMP_readv_rndv(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
        const int iovlen, int index, void *l_header, 
        void *c_header, size_t *num_bytes_ptr, smp_dma_flag_t dma_flag);

int MPIDI_CH3I_SMP_readv(MPIDI_VC_t * recv_vc_ptr, const MPL_IOV * iov,
                         const int iovlen, size_t *num_bytes_ptr);

int MPIDI_CH3I_SMP_pull_header(MPIDI_VC_t * vc,
                               MPIDI_CH3_Pkt_t ** pkt_head);

/********* End of OSU-MPI2 *************************/

/* Shared memory window atomic/accumulate mutex implementation */

#define MPIDI_CH3I_SHM_MUTEX_LOCK(win_ptr)                                              \
 do {                                                                                   \
        int pt_err = pthread_mutex_lock((win_ptr)->shm_mutex);                          \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_lock",        \
                             "**pthread_lock %s", strerror(pt_err));                    \
    } while (0)

#define MPIDI_CH3I_SHM_MUTEX_UNLOCK(win_ptr)                                            \
do {                                                                                    \
        int pt_err = pthread_mutex_unlock((win_ptr)->shm_mutex);                        \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_unlock",      \
                             "**pthread_unlock %s", strerror(pt_err));                  \
    } while (0)

#define MPIDI_CH3I_SHM_MUTEX_INIT(win_ptr)                                              \
 do {                                                                                   \
        int pt_err;                                                                     \
        pthread_mutexattr_t attr;                                                       \
                                                                                        \
        pt_err = pthread_mutexattr_init(&attr);                                         \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",       \
                             "**pthread_mutex %s", strerror(pt_err));                   \
        pt_err = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);           \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",       \
                             "**pthread_mutex %s", strerror(pt_err));                   \
        pt_err = pthread_mutex_init((win_ptr)->shm_mutex, &attr);                       \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",       \
                             "**pthread_mutex %s", strerror(pt_err));                   \
        pt_err = pthread_mutexattr_destroy(&attr);                                      \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",       \
                             "**pthread_mutex %s", strerror(pt_err));                   \
    } while (0);

#define MPIDI_CH3I_SHM_MUTEX_DESTROY(win_ptr)                                           \
do {                                                                                    \
        int pt_err = pthread_mutex_destroy((win_ptr)->shm_mutex);                       \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",       \
                             "**pthread_mutex %s", strerror(pt_err));                   \
    } while (0)



/* Starting of shared window list */

typedef struct MPIDI_SHM_Win {
    struct MPIDI_SHM_Win *prev;
    struct MPIDI_SHM_Win *next;
    MPID_Win *win;
} MPIDI_SHM_Win_t;

typedef MPIDI_SHM_Win_t *MPIDI_SHM_Wins_list_t;

extern MPIDI_SHM_Wins_list_t shm_wins_list;

#define MPIDI_SHM_Wins_next_and_continue(elem) {elem = elem->next; continue;}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHM_Wins_append
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_CH3I_SHM_Wins_append(MPIDI_SHM_Wins_list_t * list, MPID_Win * win)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_SHM_Win_t *tmp_ptr;
    MPIU_CHKPMEM_DECL(1);

    /* FIXME: We should use a pool allocator here */
    MPIU_CHKPMEM_MALLOC(tmp_ptr, MPIDI_SHM_Win_t *, sizeof(MPIDI_SHM_Win_t),
                        mpi_errno, "SHM window entry");

    tmp_ptr->next = NULL;
    tmp_ptr->win = win;

    MPL_DL_APPEND(*list, tmp_ptr);

  fn_exit:
    MPIU_CHKPMEM_COMMIT();
    return mpi_errno;
  fn_fail:
    MPIU_CHKPMEM_REAP();
    goto fn_exit;
}

/* Unlink an element from the SHM window list
 *
 * @param IN    list      Pointer to the SHM window list
 * @param IN    elem      Pointer to the element to be unlinked
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHM_Wins_unlink
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline void MPIDI_CH3I_SHM_Wins_unlink(MPIDI_SHM_Wins_list_t * list, MPID_Win * shm_win)
{
    MPIDI_SHM_Win_t *elem = NULL;
    MPIDI_SHM_Win_t *tmp_elem = NULL;

    MPL_LL_SEARCH_SCALAR(*list, elem, win, shm_win);
    if (elem != NULL) {
        tmp_elem = elem;
        MPL_DL_DELETE(*list, elem);
        MPIU_Free(tmp_elem);
    }
}

#endif /* !defined(MPICH_MPIDI_CH3_IMPL_H_INCLUDED) */
