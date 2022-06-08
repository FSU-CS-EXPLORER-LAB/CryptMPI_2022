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

#ifndef __IB_COMM__
#define __IB_COMM__

#include "thread_pool.h"
#include "bitmap.h"

#include "atomic.h"
#include "list.h"

#include "ckpt_file.h"
#include "openhash.h"

#define USE_MPI                 // undef it if we dont need multiple client nodes
#undef USE_MPI

#define    PAGE_SIZE    (4096)

#define        MAX_QP_PER_CONNECTION    (16)

#define     BUF_SLOT_COUNT    (256)

// endpoint of a QP
typedef struct qp_endpoint {
    int lid;                    // port's lid num
    int qpn;                    // qp num
    int psn;                    // psn of SQ in this qp
} qp_endpoint_t;

// a memory-region used by IB
typedef struct ib_buffer {
    pthread_mutex_t mutex;      //lock to protect the buf

    int num_slot;               // number of valid slots in the buf
    int free_slots;             // num of free slots

//  struct bitmap   bitmap; // bitmap of all slots in this buf

    //unsigned char state[BUF_SLOT_COUNT]; // state of each slot
    struct bitmap bitmap;       // bitmap for slots in this buf
    pthread_mutex_t lock[BUF_SLOT_COUNT];

    //this lock is to wake a thread waiting on send/recv at this slot

    sem_t buf_sem;              // if no free slots in this buf, wait on this sem

    struct ibv_mr *mr;          // used if this buf is ib-mr

    void *addr;                 //
    unsigned long size;         // total size of this buffer
    int slot_size;              // Each slot is of this size. i.e., each send()/recv() must <= this size

    char name[16];

} ib_buffer_t;

typedef struct qp_stat {
    int send;                   // num of ibv_send
    unsigned long send_size;    // total size of send

    int recv;                   // num of ibv_recv
    unsigned long recv_size;    // total size of recv

    int rdma_read;
    int rdma_write;

} qp_stat_t;

typedef struct ib_HCA {
    pthread_mutex_t mutex;
    int init;                   // 0: not inited, 1: has been initialized

    atomic_t ref;               // 
    // then iothreads fetch RR rqst from this queue

    struct ibv_context *context;
    struct ibv_comp_channel *comp_channel;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_srq *srq;        // for future use

    int max_rq_wqe;             // = num of recv-buf slots
    int rq_wqe;                 // num of remaining WQE on RQ(SRQ)
    int next_rbuf_slot;         // post srq WQE starting from this slot. 
    // next_rbuf_slot is mono-increasing, wrap-around by max_rq_wqe

    int comp_event_cnt;         // how many comp_channel event in comp_channel
    int total_cqe;              // how many CQE at CQ

    struct ib_buffer *send_buf;
    struct ib_buffer *recv_buf;
    struct ib_buffer *rdma_buf;

    int rx_depth;               // depth of CQ
    enum ibv_mtu mtu;
    int ib_port;

} ib_HCA_t;

/*a IB connection between 2 nodes.  Each side of a IB connection has this struct
(1 port, 1 CQ, multiple QP) in each side
*/
typedef struct ib_connection {
    ///////////////
/*    struct ibv_context  *context;
    struct ibv_comp_channel *comp_channel;
    struct ibv_pd       *pd;
    struct ibv_cq       *cq;
    struct ibv_srq      *srq;  // for future use
    
    struct ib_buffer    *send_buf;
    struct ib_buffer    *recv_buf;    
    struct ib_buffer    *rdma_buf;    */
    /////////////////

    struct ib_HCA *hca;         // a hca encaps all above elems
    unsigned int status;        // uninit -> active -> terminated

    struct ibv_qp *qp[MAX_QP_PER_CONNECTION];
    struct qp_stat qpstat[MAX_QP_PER_CONNECTION];   // statistics of each QP

    int num_qp;
    int next_use_qp;            // round-robin the qp[]. Next initiating msg goes to this qp

    /// some statistics about all QPs in this connection
    int send_cqe;
    int recv_cqe;
    int rr_cqe;
    /////////////////

    struct qp_endpoint my_ep[MAX_QP_PER_CONNECTION];
    struct qp_endpoint remote_ep[MAX_QP_PER_CONNECTION];

} ib_connection_t;

/////////////////////////////////////////

enum {
    SLOT_FREE = 0,
    SLOT_INUSE = 1,

    /// status of a connection
    connection_uninit = 0,
    connection_active = 1,
    connection_terminated = 2,

};

enum {
    qpidx_shift = 12,           // in send-wr: wr_id = qpidx | sbuf-id

};

/*
A request-pkt, will be sent by ibv_post_send, and recv's RQ will generate a CQE on this
So, it corresponds to a ibv_post_recv()
*/
typedef struct generic_pkt {
    unsigned char dummy[64];

} __attribute__ ((packed)) generic_pkt_t;

#define  arg_invalid  0xffffffffffffffff

/// define constants for ib_packet.command
enum {

    /// a generic request
    rqst_gen = 0x00000000,
    reply_gen = (1 << 31) | rqst_gen,

    /// client initiates RR by sending request
    rqst_RR = 0x00000001,
    reply_RR = (1 << 31) | rqst_RR,

    /// RDMA-write??

    /// terminate 
    rqst_terminate = 0x7fffffff,
    reply_terminate = 0xffffffff,
};

/* format of a pkt:

    u32 command
    {
        (request or reply) | cmd:
        cmd:  RDMA_READ        
    }

    RDMA_READ_request
    {
        u32    remote_proc_pid
        u64 remote_addr;
        u32 rkey;
        u32 size;
        u32 offset; // the data's offset in original     
    }

*/

///////////////////////////////////

extern int g_srv_tcpport;       // server listens on this tcp port for incoming connection requests
extern int g_num_qp;            // QP num in one connection
extern int g_rx_depth;          // RQ depth

// server listens on this tcp port for incoming connection requests
extern int numRR;               // num_qp * 4
extern int g_num_srv;

extern int sendbuf_size;        //4096; //8192;
extern int recvbuf_size;        //4096; //8192;
extern int rdmabuf_size;
extern long cli_rdmabuf_size;
extern long srv_rdmabuf_size;

extern int sendslot_size;
extern int recvslot_size;
extern int rdmaslot_size;

extern int g_iopool_size;

extern int g_exit;              // cli/srv shall exit??

//extern int g_ibtpool_size;
////////////////////////////////////

int ib_HCA_init(struct ib_HCA *hca, int is_srv);
void ib_HCA_destroy(struct ib_HCA *hca);
int get_HCA(struct ib_HCA *hca);
int put_HCA(struct ib_HCA *hca);

//int   ib_connection_init_1(struct ib_connection* conn, int num_qp, int rx_depth);
int ib_connection_init_1(struct ib_connection *conn, int num_qp, struct ib_HCA *hca);
int ib_connection_exchange_server(struct ib_connection *conn, int sock);
int ib_connection_exchange_client(struct ib_connection *conn, int sock);
int ib_connection_init_2(struct ib_connection *conn);

void ib_connection_release(struct ib_connection *conn);

//int   ib_connection_buf_init(struct ib_connection* conn, int sendsize, int recvsize, int rdmasize);
int ib_connection_buf_init(struct ib_HCA *hca, int sendsize, int recvsize, int rdmasize);

int ib_connection_post_send(struct ib_connection *conn, int qp_index, int sbuf_slot, int size);
int ib_connection_post_recv(struct ib_connection *conn, int qp_index, int rbuf_slot, int size);
int ib_connection_fillup_srq(struct ib_HCA *hca);
int ib_connection_send_RR_rqst(struct ib_connection *conn, int qpidx, int rbufid, int rprocid, int rckptid, int size, int offset, int is_last_chunk);
void ib_connection_send_terminate_rqst(struct ib_connection *connarray);

//int server_loop(struct ib_connection* conn );
int ib_server_loop(struct ib_connection *conn, struct thread_pool *tp);

int ib_client_loop(struct ib_connection *conn);

void pass_hash_table(hash_table_t * ht);

int ib_connection_post_RR(struct ib_connection *conn, int qpidx, ib_packet_t * rrpkt);
int ib_connection_send_chunk_RR_rqst(struct ib_connection *conn, ckpt_file_t * cfile, ckpt_chunk_t * chunk, void *arg);

#endif                          // __IB_COMM__
