/*!\file */
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

#ifndef IB_CM_H
#define IB_CM_H

#include <infiniband/verbs.h>
#include "ib_channel_manager.h"
#include "ib_vbuf.h"
/* add mpidimpl.h to pass MPIDI_PG_t *pg to MPID_nem_ib_setup_conn(); */
typedef enum {
     MPID_NEM_IB_CONN_NONE        = 1,
     MPID_NEM_IB_CONN_IN_PROGRESS = 2,
     MPID_NEM_IB_CONN_RC          = 3,
} MPID_nem_ib_cm_conn_type_t;

/* TODO: move srp to a independent structure in order to keep modulized*/
typedef struct _ibv_backlog_queue_t {
    int         len;                  /* length of backlog queue */
    vbuf        *vbuf_head;           /* head of backlog queue */
    vbuf        *vbuf_tail;           /* tail of backlog queue */
} ibv_backlog_queue_t;

#define BACKLOG_ENQUEUE(q,v) {                      \
    v->desc.next = NULL;                            \
    if (q->vbuf_tail == NULL) {                     \
         q->vbuf_head = v;                          \
    } else {                                        \
         q->vbuf_tail->desc.next = v;               \
    }                                               \
    q->vbuf_tail = v;                               \
    q->len++;                                       \
}

#define BACKLOG_DEQUEUE(q,v)  {                     \
    v = q->vbuf_head;                               \
    q->vbuf_head = v->desc.next;                    \
    if (v == q->vbuf_tail) {                        \
        q->vbuf_tail = NULL;                        \
    }                                               \
    q->len--;                                       \
    v->desc.next = NULL;                            \
}

typedef struct MPIDI_CH3I_MRAILI_RDMAPATH_VC
{
    /**********************************************************
     * Following part of the structure is shared by all rails *
     **********************************************************/
    /* RDMA buffers */
    void        *RDMA_send_buf_DMA;
    void        *RDMA_recv_buf_DMA;
    struct vbuf *RDMA_send_buf;
    struct vbuf *RDMA_recv_buf;

    struct ibv_mr *RDMA_send_buf_mr[MAX_NUM_HCAS];
    struct ibv_mr *RDMA_recv_buf_mr[MAX_NUM_HCAS];
    uint32_t       RDMA_remote_buf_rkey[MAX_NUM_HCAS];

    /* current flow control credit accumulated for remote side */
    uint8_t     rdma_credit;

    /* RDMA buffer address on the remote side */
    char        *remote_RDMA_buf;

    int     phead_RDMA_send;
    int     ptail_RDMA_send;

    /* pointer to the head of free receive buffers
     * this is also where we should poll for incoming
     * rdma write messages */
    /* this pointer advances when we receive packets */
    int     p_RDMA_recv;
    int     p_RDMA_recv_tail;

    int     eager_start_cnt;
    int     in_polling_set;
    int     rdma_failed;

#ifndef MV2_DISABLE_HEADER_CACHING 
    void        *cached_outgoing;
    void        *cached_outgoing_iheader;
    void        *cached_incoming;
    void        *cached_incoming_iheader;
    int         cached_hit;
    int         cached_miss;
#endif
} MPIDI_CH3I_MRAILI_RDMAPATH_VC;

/**
 * MPIDI_CH3I_MRAILI_SR_VC
 */
typedef struct MPIDI_NEM_IB_SR_VC {
    struct {
        /* how many vbufs can be consumed on remote end. */
        uint8_t         remote_credit;
        /* accumulate vbuf credit locally here */
        uint8_t         local_credit;
        /* number of vbufs currently preposted */
        uint8_t         preposts;
        uint32_t         pending_r3_sends;
        uint8_t         remote_cc;
        uint8_t         initialized;
        int             rendezvous_packets_expected;
        ibv_backlog_queue_t backlog;
    } *credits;
} MPIDI_NEM_IB_SR_VC;

#define INDEX_GLOBAL(_cmanager,_global_index) (_global_index) 

#define INDEX_LOCAL(_cmanager,_local_index) \
    (((_cmanager)->num_channels - (_cmanager)->num_local_pollings) + (_local_index))

struct process_init_info {
    int         **hostid;
    uint16_t    **lid;
    uint32_t    **qp_num_rdma;
    union ibv_gid    **gid;
    uint64_t    *vc_addr;
    mv2_arch_hca_type    *arch_hca_type;
};

struct mrail_rail {
        int     hca_index;
        int     s_weight;
        int     used_send_cq;
        int     used_recv_cq;
        struct ibv_qp   *qp_hndl;
        int             send_wqes_avail;
        vbuf            *ext_sendq_head;
        vbuf            *ext_sendq_tail;
        int             ext_sendq_size;
};

typedef struct _MPID_nem_ib_connection_tag {
    struct  mrail_rail  *rails;

    /* move to channel_manager.h */
    /* MPID_nem_ib_channel_manager  cmanager; */

    /* number of send wqes available */
    uint16_t    next_packet_expected;
    uint16_t    next_packet_tosend;

    /* what buffer are we currently packing */
    struct vbuf *coalesce_vbuf;

    /* how many eager sends do we have outstanding */
    int outstanding_eager_vbufs;

    MPIDI_CH3I_MRAILI_RDMAPATH_VC   rfp;
    MPIDI_NEM_IB_SR_VC             srp;

    /* Buffered receiving request for packetized transfer */
    void                    *packetized_recv;
    
    /* these fields are used to remember data transfer operations
     * that are currently in progress on this connection. The
     * send handle list is a queue of send handles representing
     * in-progress rendezvous transfers. It is processed in FIFO
     * order (because of MPI ordering rules) so there is both a head
     * and a tail.
     *
     * The receive handle is a pointer to a single
     * in-progress eager receive. We require that an eager sender
     * send *all* packets associated with an eager receive before
     * sending any others, so when we receive the first packet of
     * an eager series, we remember it by caching the rhandle
     * on the connection.
     *
     */
    void    *sreq_head; /* "queue" of send handles to process */
    void    *sreq_tail;
    /* these two fields are used *only* by MPID_DeviceCheck to
     * build up a list of connections that have received new
     * flow control credit so that pending operations should be
     * pushed. nextflow is a pointer to the next connection on the
     * list, and inflow is 1 (true) or 0 (false) to indicate whether
     * the connection is currently on the flowlist. This is needed
     * to prevent a circular list. 
     */
    void    *nextflow;
    int     inflow;

    uint64_t remote_vc_addr; /* Used to find vc at remote side */
} MPID_nem_ib_connection_t;

/**
 * The list of all connections.
 */
typedef struct _MPID_nem_ib_connections {
    int size;
    int rank;
    MPID_nem_ib_connection_t *connections;
    struct process_init_info *init_info;
} MPID_nem_ib_conn_info_t;


/**
 *  all the connections existing for current process
 */
extern MPID_nem_ib_conn_info_t conn_info;

struct MPIDI_PG;
int MPID_nem_ib_init_connection(int size, int rank);
int MPID_nem_ib_alloc_process_init_info();
int MPID_nem_ib_setup_conn(struct MPIDI_PG *pg);
int MPID_nem_ib_establish_conn();
int MPID_nem_ib_exchange_conn(struct MPIDI_PG *pg, int rank);
int MPID_nem_ib_free_conn_info(int size);
int MPID_nem_ib_setup_startup_ring(struct MPIDI_PG *pg, int rank);

extern int mv2_pmi_max_keylen;
extern int mv2_pmi_max_vallen;
extern char *mv2_pmi_key;
extern char *mv2_pmi_val;

/*
 * mv2_allocate_pmi_keyval
 * Allocate a Key-Value pair of correct length
 * Return 0 on success, non-zero on failure
 */
int mv2_allocate_pmi_keyval(void);

/*
 * mv2_free_pmi_keyval
 * Free a previously allocated Key-Value pair
 */
void mv2_free_pmi_keyval(void);

/**
 * set credits info into packet header
 */
#define PACKET_SET_CREDIT(_p, _c, _rail_index)                  \
{                                                               \
    (_p)->rdma_credit     = (_c)->rfp.rdma_credit;\
    (_c)->rfp.rdma_credit = 0;                                  \
    (_p)->vbuf_credit     =  (_c)->srp.credits[(_rail_index)].local_credit;        \
    (_p)->remote_credit   =  (_c)->srp.credits[(_rail_index)].remote_credit;       \
    (_c)->srp.credits[(_rail_index)].local_credit = 0;    \
}

#define PACKET_SET_RDMA_CREDIT(_p, _c)                          \
{                                                               \
    (_p)->rdma_credit     = (_c)->rfp.rdma_credit;  \
    (_c)->rfp.rdma_credit = 0;                            \
    (_p)->vbuf_credit     = 0;                            \
    (_p)->remote_credit   = 0;                            \
}   

#endif
