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

#include <mpichconf.h>

#ifdef CR_AGGRE

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <fcntl.h>
#include <poll.h>

#include <string.h>
#include <strings.h>
//#include <malloc.h>
#include <netinet/in.h>
#include <byteswap.h>
#include <inttypes.h>

#include <infiniband/verbs.h>

#include "thread_pool.h"

#include "ib_comm.h"
#include "ibutil.h"
#include "debug.h"
#include "ib_buf.h"
#include "work_queue.h"

#include "openhash.h"

#include "ckpt_file.h"

/////////////////////////////////////
#define        POLL_COMP_CHANNEL

////////////////////////////////////

// server listens on this tcp port for incoming connection requests
int g_srv_tcpport = 15678;

///////////////////
int g_num_qp = 1;               // 8; // 4;  // QP num in one connection

int numRR = 65;                 // num_qp * 4  // NO use
int g_complete_RR = 0;          // NO use
int g_num_srv = 1;              /// NO use

char g_devname[32] = "mlx4_0";  //"mthca0"; // use this HCA device

//int g_client_complete = 0;
//int g_server_complete = 0;

////////////////////////////

int g_rx_depth = 512;           // RQ depth
int sendbuf_size = (16384);     //8192;
int recvbuf_size = (16384);     //8192;
long cli_rdmabuf_size = 8 * 1024 * 1024;
long srv_rdmabuf_size = 8 * 1024 * 1024;

int sendslot_size = 64;
int recvslot_size = 64;
int rdmaslot_size = 128 * 1024;

int g_device = 1;               // which IB-NIC to use.  0: mthca0;  1: mlx4_0

//int g_ibtpool_size = 16;  

int g_iopool_size = 8;          // NOTE: make sure iopool size >= num of RDMA-bufs

//int g_exit = 0;

////////////
// static struct ibv_wc *wc_array[512];
////////////////

//////////////  vars

static hash_table_t *ht_cfile;  /// open-hash table for ckpt-files

////////////////////////////////////

extern int g_exit;

//////////////////////////////////

void dump_ep(struct qp_endpoint *ep)
{
    printf("qp-ep: lid / qpn / psn = 0x%x / 0x%x / 0x%x\n", ep->lid, ep->qpn, ep->psn);
}

void dump_qpstat(struct ib_connection *conn)
{
    int i;
    struct qp_stat *stat;

    for (i = 0; i < conn->num_qp; i++) {
        stat = conn->qpstat + i;
        printf("\tqp %d: send %d, recv %d\n", i, stat->send, stat->recv);
    }

    printf("\tTotal: send-cqe %d, recv-cqe %d, rr-cqe %d, total cqe %d\n", conn->send_cqe, conn->recv_cqe, conn->rr_cqe, conn->hca->total_cqe);
    printf("====================\n");

}

void dump_ib_packet(struct ib_packet *pkt)
{
    return;

    char *cmd;
    switch (pkt->command) {
        /// a generic request
    case rqst_gen:
        cmd = "rqst_gen";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    case reply_gen:
        cmd = "reply_gen";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    case rqst_RR:
        cmd = "rqst_RR";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        printf("    rbuf = %lu, rbuf-id=%d, size = %lu\n", pkt->RR.raddr, pkt->RR.rbuf_id, pkt->RR.size);
        printf("    lbuf-id = %d\n", pkt->RR.lbuf_id);
        break;
    case reply_RR:
        cmd = "reply_RR";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    case rqst_terminate:
        cmd = "rqst_terminate";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    case reply_terminate:
        cmd = "reply_terminate";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    default:
        cmd = "unknown cmd";
        printf("ib-packet:  cmd = \"%s\" \n", cmd);
        break;
    }
}

static inline int get_next_qp(struct ib_connection *conn)
{
    return (conn->next_use_qp++) % conn->num_qp;
}

static inline int find_qp(int qpn, struct ib_connection *conn)
{
    int i;
    for (i = 0; i < conn->num_qp; i++) {
        if (qpn == conn->qp[i]->qp_num)
            return i;
    }
    return -1;
}

static inline int find_qp_from_conn_array(int qpn, struct ib_connection *conn, int num_con, int *conidx, int *qpidx)
{
    //printf(" num-conn %d, qpn 0x%x\n", num_con, qpn );
    *conidx = -1;
    *qpidx = -1;
    int i;
    for (i = 0; i < num_con; i++) {
        *qpidx = find_qp(qpn, conn + i);
        if (*qpidx >= 0) {
            // find a match in this connection "i"
            *conidx = i;        // 
            return 0;
        }
    }
    error(" num-conn %d, qpn 0x%x, fail to find qp\n", num_con, qpn);
    return -1;
}

////////////////////////////////////////////////////////////////////////////////////////
/////////////////////
int get_HCA(struct ib_HCA *hca)
{
    atomic_add(1, &hca->ref);
    return 0;
}

int put_HCA(struct ib_HCA *hca)
{
    if (atomic_sub_and_test(1, &hca->ref)) {    // no connection is referring it, release the HCA

    }
    return 0;
}

int ib_HCA_init(struct ib_HCA *hca, int is_srv)
{
    ///////////////////
    struct ibv_device **dev_list;
    struct ibv_port_attr port_attr;
    int i, ret;
    //int dev;

    ///////////
    memset(hca, 0, sizeof(*hca));
    pthread_mutex_init(&hca->mutex, NULL);

    //////////////
    pthread_mutex_lock(&hca->mutex);
    if (hca->init > 0) {        // has been inited, don't need to do anything
        pthread_mutex_unlock(&hca->mutex);
        return 0;
    }
    /// now, init a HCA 
    //memset(hca, 0, sizeof(*hca));
    atomic_set(&hca->ref, 0);
    hca->rq_wqe = 0;
    hca->next_rbuf_slot = 0;    // RQ wqe starts from this rbuf-slot
    hca->comp_event_cnt = 0;

    ////////// default properties
    hca->ib_port = 1;
    hca->rx_depth = g_rx_depth;
    hca->mtu = IBV_MTU_2048;    // default MTU

    //////////////////////////////////////////////////////////////////
    //// open IB device 

    dev_list = ibv_get_device_list(&ret);   // has "ret" num of devices in the system
    if (!dev_list) {
        error("Fail at get_device_list\n");
        goto err_out_1;
    }

    for (i = 0; i < ret; i++)   // open the device with "g_devname"
    {
        if (strcmp(g_devname, ibv_get_device_name(dev_list[i])) == 0) {
            hca->context = ibv_open_device(dev_list[i]);
            break;
        }
    }
    if (i >= ret)               // couldn't find the device
    {
        i = ret - 1;
        hca->context = ibv_open_device(dev_list[i]);
        //error("ERROR!!!!   ****** %d HCA,  Cannot find device %s\n", ret,g_devname);
        //return -1;
    }
    //dev = 1; //ret-1; // use device 0: // dev0: mthca0,  dev1: mlx4_0
/*    g_device = ret>1? g_device: 0;
    hca->context = ibv_open_device(dev_list[g_device]);
    if( ! hca->context ){
        error("Fail to open_device %d\n", g_device);
        ibv_free_device_list( dev_list ); // free all other devices
        goto err_out_1;
    }    */
    printf("------- Has opened device: %s\n", ibv_get_device_name(dev_list[i]));    //g_device]) );
    ibv_free_device_list(dev_list); // free all other devices

    /// get device's MTU    
    ibv_query_port(hca->context, hca->ib_port, &port_attr);
    hca->mtu = port_attr.active_mtu;

    ///// create completion-channel
    hca->comp_channel = ibv_create_comp_channel(hca->context);
    if (!hca->comp_channel) {
        error("Fail to create_comp_channel\n");
        goto err_out_1;
    }
    // change flags of the comp_channel fd:
#ifdef    POLL_COMP_CHANNEL
    i = fcntl(hca->comp_channel->fd, F_GETFL);
    ret = fcntl(hca->comp_channel->fd, F_SETFL, i | O_NONBLOCK);    // set the fd to be non-blocking
    if (ret < 0) {
        error("Fail to change comp_channel flag to NONBLOCK...\n");
        goto err_out_1;
    }
#endif

    // get pd
    hca->pd = ibv_alloc_pd(hca->context);
    if (!hca->pd) {
        error("Fail to alloc_pd\n");
        goto err_out_1;
    }
    // get CQ
    hca->cq = ibv_create_cq(hca->context, hca->rx_depth + 1,    // rx_depth+1,  /* CQE depth */
                            NULL,   //void*, user provide cq_context                
                            hca->comp_channel,  /*completion_channel */
                            0);
    if (!hca->cq) {
        error("Fail to create_cq\n");
        goto err_out_1;
    }
    ////////  get SRQ
    struct ibv_srq_init_attr srqattr = {
        .attr = {
                 .max_wr = g_rx_depth,
                 .max_sge = 1}
    };
    hca->srq = ibv_create_srq(hca->pd, &srqattr);
    if (!hca->srq) {
        error("Couldn't create SRQ\n");
        goto err_out_1;
    }
    ///////  alloc ib_buf..
    //dbg("create ib_buf\n");
    // send-buf size, recv-buf size,  RDMA-buf size
    int rdmabufsize = is_srv ? srv_rdmabuf_size : cli_rdmabuf_size;
    if (ib_connection_buf_init(hca, sendbuf_size, recvbuf_size, rdmabufsize) != 0) {
        error("Fail to create ib_buffer\n");
        goto err_out_1;
    }
    /// mark the hca as inited
    hca->init = 1;
    pthread_mutex_unlock(&hca->mutex);

    return 0;
    ///////////
    ///////////////////////////////////////////
  err_out_1:
    pthread_mutex_unlock(&hca->mutex);
    ib_HCA_destroy(hca);
    return -1;
}

void ib_HCA_destroy(struct ib_HCA *hca)
{
    ////////// free buffers
    printf("Release HCA...\n");
    if (hca->send_buf)
        free_ib_buffer(hca->send_buf);
    if (hca->recv_buf)
        free_ib_buffer(hca->recv_buf);
    if (hca->rdma_buf)
        free_ib_buffer(hca->rdma_buf);

    ///// ack remaining un-acked events
    printf("%d cq_event \n", hca->comp_event_cnt);
    if (hca->comp_event_cnt > 0) {
        if (hca->comp_event_cnt % 10 != 0) {
            ibv_ack_cq_events(hca->cq, hca->comp_event_cnt % 10);
        }
    }
    ///// free srq  
    if (hca->srq)
        if (ibv_destroy_srq(hca->srq)) {
            error("   Couldn't destroy SRQ\n");
        }
    /// free cq
    if (hca->cq)
        ibv_destroy_cq(hca->cq);

    /// free pd
    if (hca->pd) {
        if (ibv_dealloc_pd(hca->pd)) {
            error("Couldn't deallocate PD\n");
        }
    }
    // free comp_channel
    if (hca->comp_channel) {
        if (ibv_destroy_comp_channel(hca->comp_channel)) {
            error("Couldn't destroy comp_channel\n");
        }
    }
    // free context
    if (hca->context) {
        if (ibv_close_device(hca->context)) {
            fprintf(stderr, "Couldn't release context\n");
            //return 1;
        }
    }
    // 
    pthread_mutex_destroy(&hca->mutex);
    printf("Has released connection...\n");
}

////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/*
init the basic params in connection
*/
int ib_connection_init_1(struct ib_connection *conn, int num_qp, struct ib_HCA *hca)    // , int rx_depth)
{
    //// init some basic infor in the connection    
    memset(conn, 0, sizeof(struct ib_connection));

    ///////////////////
    struct ibv_port_attr port_attr;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    unsigned int attr_mask;
    int i, ret;
    //int dev;

    //////////////////////////////////////////////////////////////////
    //// get the HCA context
    get_HCA(hca);
    conn->hca = hca;

    /////////////////////////////////////////////////////////////////
    /////   open QPs
    conn->num_qp = num_qp;
    conn->next_use_qp = 0;

    for (i = 0; i < conn->num_qp; i++) {
        // create a new QP  
        //dbg("create qp %d\n", i);
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.send_cq = hca->cq;
        qp_init_attr.recv_cq = hca->cq;
        //////
        qp_init_attr.srq = hca->srq;
        /////
        qp_init_attr.cap.max_send_wr = 64,  // 4,
            qp_init_attr.cap.max_recv_wr = hca->rx_depth, qp_init_attr.cap.max_send_sge = 1, qp_init_attr.cap.max_recv_sge = 1;

        qp_init_attr.qp_type = IBV_QPT_RC;  // create a RC qp

        conn->qp[i] = ibv_create_qp(hca->pd, &qp_init_attr);
        if (!conn->qp[i]) {
            error("fail to create_qp %d, qp[%d]=%p\n", i, i, conn->qp[i]);
            goto err_out_1;
        }
    }
    dbg("Have created %d QPs for one connection...\n", conn->num_qp);
    /////////////////////////////////////////////////////////////// 
    //// set qp to INIT state

    for (i = 0; i < conn->num_qp; i++) {
        //dbg("transit qp %d to INIT\n", i);
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.pkey_index = 0; // Primary P_Key index 
        qp_attr.port_num = hca->ib_port;    //                                                                                                 

        //***** NOTE:::  must set qp_access_flag if RDMA is needed
        qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

        attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

        ret = ibv_modify_qp(conn->qp[i], &qp_attr, attr_mask);
        if (ret != 0) {
            error("Fail to transit qp %d to INIT\n", i);
            goto err_out_1;
        }
    }

    /////////////////////////////////////////////////////
    ///////     post WQE to RQ/SRQ
    //for( i=0; i< conn->rx_depth; i++){
    //for( i=0; i< conn->recv_buf->num_slot; i++){  
    //  slot = get_buf_slot( conn->recv_buf, NULL, 0 );
    //  ret = ib_connection_post_recv( conn, i, slot, 64 );
    //dbg("  Post recv req, slot %d, ret %d\n", i, ret);        
    //} }
    //ib_connection_fillup_srq( conn );

    //////////////////////////////////////////////////////////////////
    //// get IB device attr,  IB port attr
    ibv_query_port(hca->context, hca->ib_port, &port_attr);
    //conn->mtu = port_attr.active_mtu;

    // record the endpoint info for each QP     
    for (i = 0; i < conn->num_qp; i++) {
        conn->my_ep[i].lid = port_attr.lid;
        conn->my_ep[i].qpn = conn->qp[i]->qp_num;
        conn->my_ep[i].psn = 0;
        //dump_ep( conn->my_ep +i );
    }

    return 0;

    ///////////////////////////////////////////////////////////////
    ////    clean up
  err_out_1:
    ib_connection_release(conn);

    return -3;
}

/*
Init QPs: step 2
Have already exchanged qp-infor with remote peers. 
Remote-peer endpoints are stored at conn->remote_e
*/
int ib_connection_init_2(struct ib_connection *conn)
{
    struct ib_HCA *hca = conn->hca;

    struct ibv_qp_attr qp_attr;
    unsigned int attr_mask;
    int i;

    ////// . request first cq_event
    ibv_req_notify_cq(hca->cq, 0);  // each CQE will trigger a comp_channel event

    // modify QP's attributes: ready-to-read
    for (i = 0; i < conn->num_qp; i++) {
        memset(&qp_attr, 0, sizeof(qp_attr));

        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = hca->mtu;
        qp_attr.dest_qp_num = conn->remote_ep[i].qpn;   //***** remote-qpn
        qp_attr.rq_psn = conn->remote_ep[i].psn;    ///***** RQ-psn, match remote send-psn 
        qp_attr.max_dest_rd_atomic = 4; // num of outstanding "inbound" rdma-read / atomic, as responder 
        qp_attr.min_rnr_timer = 12;

        qp_attr.ah_attr.is_global = 0;
        qp_attr.ah_attr.dlid = conn->remote_ep[i].lid;  //***** remote lid 
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = hca->ib_port;

        printf("  local QP %d (qpn %x) match remote QP:  ", i, conn->qp[i]->qp_num);
        dump_ep(&conn->remote_ep[i]);

        ///// transit to RTR state
        attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
        ibv_modify_qp(conn->qp[i], &qp_attr, attr_mask);
    }

    /// modify QP to RTS
    for (i = 0; i < conn->num_qp; i++) {
        memset(&qp_attr, 0, sizeof(qp_attr));

        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.timeout = 14;
        qp_attr.retry_cnt = 7;
        qp_attr.rnr_retry = 7;

        qp_attr.sq_psn = conn->my_ep[i].psn;    // SQ PSN
        qp_attr.max_rd_atomic = 4;  // max outstanding rdma-read/ atomic, as initiator 

        //// transit to RTS state
        attr_mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
        ibv_modify_qp(conn->qp[i], &qp_attr, attr_mask);
    }

    dbg("ib-conn init-2 completed...\n");
    return 0;
}

/*
Release resources used by this connection
*/
void ib_connection_release(struct ib_connection *conn)
{
    struct ib_HCA *hca = conn->hca;

    int i, ret;

    ///// drain the QP
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_SQD;

    for (i = 0; i < conn->num_qp; i++) {
        if (!conn->qp[i])
            break;

        ret = ibv_modify_qp(conn->qp[i], &attr, IBV_QP_STATE);
        if (ret < 0) {
            error("Error when drain qp %d, ret=%d\n", i, ret);
        }
    }

    ////// free qp
    for (i = 0; i < conn->num_qp; i++) {
        if (!conn->qp[i])
            break;
        dbg("destroy qp %d\n", i);
        ibv_destroy_qp(conn->qp[i]);
    }

    put_HCA(hca);
    printf("Has released connection...\n");
}

/*
Create send-buf, recv-buf, rdma-buf for this ib_connection. 
By default, each buf has (256) slots. 
*/
//int   ib_connection_buf_init(struct ib_connection* conn, int sendsize, int recvsize, int rdmasize)
int ib_connection_buf_init(struct ib_HCA *hca, int sendsize, int recvsize, int rdmasize)
{
    int slot_size = sendslot_size;  //64; // max 256 slots, so max size = 64 * 256 = 16 KB 

    hca->send_buf = create_ib_buffer(sendsize, slot_size, "send-buf");
    hca->send_buf->mr = ibv_reg_mr(hca->pd, hca->send_buf->addr, sendsize, IBV_ACCESS_LOCAL_WRITE);
    //dbg("send-buf: addr=%p, size=%d, mr=%p\n", hca->send_buf->addr, 
    //      sendsize, hca->send_buf->mr );

    slot_size = recvslot_size;
    hca->recv_buf = create_ib_buffer(recvsize, slot_size, "recv-buf");
    hca->recv_buf->mr = ibv_reg_mr(hca->pd, hca->recv_buf->addr, recvsize, IBV_ACCESS_LOCAL_WRITE);
    //dbg("recv-buf: addr=%p, size=%d, mr=%p\n", hca->recv_buf->addr, 
    //      recvsize, hca->recv_buf->mr );

    /// max num of RQ WQE, determined by recv-buf size
    hca->max_rq_wqe = (hca->recv_buf->num_slot < hca->rx_depth) ? hca->recv_buf->num_slot : hca->rx_depth;

    slot_size = rdmaslot_size;  // 1*1024*1024; // 1M per slot in RDMA buf. max 256 slots
    if (rdmasize > 0) {
        hca->rdma_buf = create_ib_buffer(rdmasize, slot_size, "rdma-buf");
        hca->rdma_buf->mr = ibv_reg_mr(hca->pd, hca->rdma_buf->addr, rdmasize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        printf("%s: Create RDMA-buf: size=%d, slotsize=%d\n", __func__, rdmasize, rdmaslot_size);
        //hca->rdma_buf->addr, rdmasize, hca->rdma_buf->mr );
    }

    return 0;
}

/*
Server exchanges qp-infor with client
*/
int ib_connection_exchange_server(struct ib_connection *conn, int sock)
{
    int n = 0, r = 0;
    void *src, *dst;
    int size = sizeof(struct qp_endpoint) * conn->num_qp;   // expected total size

    src = (void *) (conn->my_ep);
    dst = (void *) (conn->remote_ep);

    /////  1. receive remote_endpoint, store to dest
    n = 0;
    while (n < size) {
        r = read(sock, dst + n, size - n);
        if (r < 0) {
            error("Error read: current %d, ret %d\n", n, r);
        }
        n += r;
    }
    /////  2. write my endpoint to remote
    n = 0;
    while (n < size) {
        r = write(sock, src + n, size - n);
        if (r < 0) {
            error("Error read: current %d, ret %d\n", n, r);
        }
        n += r;
    }
    ////  3. final handshake: a dummy int
    read(sock, &n, sizeof(int));

    //// dump remote infor
    for (n = 0; n < conn->num_qp; n++) {
        //dbg("local  EP %d :: ", n);
        //dump_ep( &conn->my_ep[n] );
    }
    for (n = 0; n < conn->num_qp; n++) {
        //dbg("Remote EP %d :: ", n);
        //dump_ep( &conn->remote_ep[n] );
    }
    return 0;
}

/*
Client exchanges qp-infor with server
*/
int ib_connection_exchange_client(struct ib_connection *conn, int sock)
{
    int n = 0, r = 0;
    void *src, *dst;
    int size = sizeof(struct qp_endpoint) * conn->num_qp;   // expected total size

    src = conn->my_ep;
    dst = conn->remote_ep;

    /////  1. write my endpoint to remote
    n = 0;
    while (n < size) {
        r = write(sock, src + n, size - n);
        if (r < 0) {
            error("Error read: current %d, ret %d\n", n, r);
        }
        n += r;
    }
    /////  2. receive remote_endpoint to dest
    n = 0;
    while (n < size) {
        r = read(sock, dst + n, size - n);
        if (r < 0) {
            error("Error read: current %d, ret %d\n", n, r);
        }
        n += r;
    }
    ////  3. final handshake: a dummy int
    write(sock, &n, sizeof(int));

    //// dump remote infor
    for (n = 0; n < conn->num_qp; n++) {
        //dbg("local  EP %d :: ", n);
        //dump_ep( &conn->my_ep[n] );
    }
    for (n = 0; n < conn->num_qp; n++) {
        //dbg("Remote EP %d :: ", n);
        //dump_ep( &conn->remote_ep[n] );
    }

    ////////////////
    return 0;

}

/*
post a send-req to qp[idx], the buf to send is at send_buf[slot], with size="size"
*/
int ib_connection_post_send(struct ib_connection *conn, int qp_index, int sbuf_slot, int size)
{
    struct ib_HCA *hca = conn->hca;

    /// ibv_send:  wc->wr_id =   connection-id | qp-id | buf-slot-id
    struct ibv_sge list = {
        .addr = (uint64_t) (hca->send_buf->addr + sbuf_slot * hca->send_buf->slot_size),
        .length = size,
        .lkey = hca->send_buf->mr->lkey,    //ctx->mr->lkey
    };

    /// send-wr.wr_id = (qp_idx ) | sendbuf-id
    struct ibv_send_wr wr = {
        .wr_id = ((uint64_t) qp_index << qpidx_shift) | sbuf_slot,
        // max 2^12 slots in the buf
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED,
    };
    struct ibv_send_wr *bad_wr;

    int i = ibv_post_send(conn->qp[qp_index], &wr, &bad_wr);
//  dbg("post-send at qp %d(qpn %x), send slot %d, size %d, msg=\"%s\", ret %d\n", 
//      qp_index, conn->qp[qp_index]->qp_num, sbuf_slot,  size, ib_buffer_slot_addr( hca->send_buf, sbuf_slot),  i ); 

    return i;
}

/*
IB-server calls this to post a RDMA-read to SQ. 
The "rrpkt" contains infor of RR in remote peer, it's a local-var from iop
*/
int ib_connection_post_RR(struct ib_connection *conn, int qpidx, ib_packet_t * rrpkt)
{
    struct ib_HCA *hca = conn->hca;

    void *addr;                 // local RDMA-buf addr

    /// this call may block, wait for a rdma_buf to be avail
    int slot = get_buf_slot(hca->rdma_buf, &addr, 0);

    rrpkt->RR.lbuf_id = slot;   // local buf-slot to receive the RR data
    rrpkt->RR.laddr = (uint64_t) addr;
    //backup->RR.larg1 = pkt.RR.larg1;

    struct ibv_sge list = {
        .addr = (uint64_t) addr,    // fetch remote data, store the data to this "addr"
        .length = rrpkt->RR.size,   // backup->RR.size,
        .lkey = hca->rdma_buf->mr->lkey,    //ctx->mr->lkey
    };
    // RR send-wr. wr = RR-ib_packet_t*
    struct ibv_send_wr wr = {
        .wr_id = (uint64_t) rrpkt,
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,

        .wr.rdma.remote_addr = rrpkt->RR.raddr, // backup->RR.raddr
        .wr.rdma.rkey = rrpkt->RR.rkey, // backup->RR.rkey,
    };
    struct ibv_send_wr *bad_wr;

    int i = ibv_post_send(conn->qp[qpidx], &wr, &bad_wr);
    dbg("post RR: qp %d(qpn %x) (%s) %ld@%ld, lbuf-slot %d, rmt-addr %p, rmt-bufid %d, rkey 0x%x, ret %d\n",
        qpidx, conn->qp[qpidx]->qp_num, rrpkt->RR.filename, rrpkt->RR.size, rrpkt->RR.offset, slot, rrpkt->RR.raddr, rrpkt->RR.rbuf_id, rrpkt->RR.rkey, i);
    //dump_ib_packet( rrpkt ); // backup );
    if (i >= 0)
        return slot;
    else
        return i;
}

/*
Send the data(in buf) of size(size), via QP(qpidx)
*/
int ib_connection_send(struct ib_connection *conn, int qpidx, void *buf, int size)
{
    struct ib_HCA *hca = conn->hca;
    void *addr;
    int i = get_buf_slot(hca->send_buf, &addr, 0);  // addr now points to slot "i"

    memcpy(addr, buf, size < hca->send_buf->slot_size ? size : hca->send_buf->slot_size);

    return ib_connection_post_send(conn, qpidx, i, 64);
}

/*
Post a RECV WQE to SRQ
Note: the qp_index is not really needed. It's a dummy
*/
int ib_connection_post_recv(struct ib_connection *conn, int qpidx, int rbuf_slot, int size)
{
    struct ib_HCA *hca = conn->hca;

    struct ibv_sge list = {
        .addr = (uint64_t) (hca->recv_buf->addr + rbuf_slot * hca->recv_buf->slot_size),
        .length = size,
        .lkey = hca->recv_buf->mr->lkey,
    };
    /// recv-wr.wr_id = recv-buf-slot-id
    struct ibv_recv_wr wr = {
        .wr_id = (uint64_t) rbuf_slot,  // (uint64_t) list.addr,
        ///((uint64_t)qpidx << 12) | rbuf_slot, // max 2^12 slots in the buf
        .sg_list = &list,
        .num_sge = 1,
        .next = NULL,
    };
    struct ibv_recv_wr *bad_wr;
    int i;
    //i = ibv_post_recv(conn->qp[qpidx], &wr, &bad_wr);
    //ibv_post_recv(qp, &wr, &bad_rcv);
    i = ibv_post_srq_recv(hca->srq, &wr, &bad_wr);
    //dbg("   recv slot %d, slot-size %d, size %d, ret %d\n", 
    //      rbuf_slot, hca->recv_buf->slot_size, size, i );
    return i;
}

/*

*/
int ib_connection_post_srq_recv(struct ib_HCA *hca, int qpidx, int rbuf_slot, int size)
{
    struct ibv_sge list = {
        .addr = (uint64_t) (hca->recv_buf->addr + rbuf_slot * hca->recv_buf->slot_size),
        .length = size,
        .lkey = hca->recv_buf->mr->lkey,
    };
    /// recv-wr.wr_id = recv-buf-slot-id
    struct ibv_recv_wr wr = {
        .wr_id = (uint64_t) rbuf_slot,  // (uint64_t) list.addr,
        ///((uint64_t)qpidx << 12) | rbuf_slot, // max 2^12 slots in the buf
        .sg_list = &list,
        .num_sge = 1,
        .next = NULL,
    };
    struct ibv_recv_wr *bad_wr;
    int i;
    i = ibv_post_srq_recv(hca->srq, &wr, &bad_wr);
    //dbg("   recv slot %d, slot-size %d, size %d, ret %d\n", 
    //      rbuf_slot, hca->recv_buf->slot_size, size, i );
    return i;
}

/*
Only the ib-loop thread should be able to call this func
*/
int ib_connection_fillup_srq(struct ib_HCA *hca) //struct ib_connection* conn)
{
    int i = 0;
    //if( atomic_read(&conn->rq_wqe) <= 20 ){ 
    if (hca->rq_wqe <= 20) {
        // too few RQ wqe left, fill it up
        int fill = hca->max_rq_wqe - hca->rq_wqe;
        int slot = hca->next_rbuf_slot;

        dbg("  fill srq, from slot %d, fill %d slots\n", slot, fill);
        //atomic_read( &conn->rq_weq ); //
        for (i = 0; i < fill; i++) {
            ib_connection_post_srq_recv(hca, 0, slot % hca->max_rq_wqe, hca->recv_buf->slot_size);
            slot++;
        }
        hca->recv_buf->free_slots -= fill;
        hca->rq_wqe += fill;
        hca->next_rbuf_slot = slot;
    }
    return i;
}

/// client sends a rqst to server: RR
/// rbufid:  the RDMA-buf-id on client
int ib_connection_send_RR_rqst(struct ib_connection *conn, int qpidx, int rbufid, int rprocid, int rckptid, int size, int offset, int is_last_chunk)
{
    struct ib_HCA *hca = conn->hca;
    struct ib_packet *pkt;

    // get a send-buf slot
    int slot = get_buf_slot(hca->send_buf, (void *) &pkt, 0);

    /// init the pkt fields
    pkt->command = rqst_RR;

    pkt->RR.rbuf_id = rbufid;
    pkt->RR.raddr = (uint64_t) ib_buffer_slot_addr(hca->rdma_buf, rbufid);
    pkt->RR.rkey = hca->rdma_buf->mr->rkey;

    pkt->RR.rprocid = rprocid;
    pkt->RR.rckptid = rckptid;
    pkt->RR.size = size;
    pkt->RR.offset = offset;
    pkt->RR.larg1 = is_last_chunk;
    // init the buf to some random text
    //sprintf((void*)pkt->RR.raddr, " raddr=%p, rbufid=%d, rkey=0x%x, size=%d\n", pkt->RR.raddr,
    //  pkt->RR.rbuf_id, pkt->RR.rkey, pkt->RR.size );

    ///////////////
    //dump_ib_packet( pkt );
    dbg(" rqst RR of rdma-bufid %d= %s, free-slots=%d\n", pkt->RR.rbuf_id, pkt->RR.raddr, hca->rdma_buf->free_slots);

    /// post send
    return ib_connection_post_send(conn, qpidx, slot, sizeof(*pkt));
}

/// mig-src node sends a RR rqst to tgt
int ib_connection_send_chunk_RR_rqst(struct ib_connection *conn, ckpt_file_t * cfile, ckpt_chunk_t * chunk, void *arg)
{
    struct ib_HCA *hca = conn->hca;
    struct ib_packet *pkt;

    int qpidx = get_next_qp(conn);

    // get a send-buf slot
    int slot = get_buf_slot(hca->send_buf, (void *) &pkt, 0);
    // init the pkt fields
    pkt->command = rqst_RR;

    pkt->RR.rbuf_id = chunk->bufid;
    if (chunk->curr_pos > 0)
        pkt->RR.raddr = (uint64_t) ib_buffer_slot_addr(hca->rdma_buf, chunk->bufid);
    else
        pkt->RR.raddr = 0;

    pkt->RR.rkey = hca->rdma_buf->mr->rkey;
    pkt->RR.rckptid = chunk->ckpt_id;
    pkt->RR.rprocid = chunk->proc_rank;
    pkt->RR.size = chunk->curr_pos;
    pkt->RR.offset = chunk->offset;
    pkt->RR.is_last_chunk = chunk->is_last_chunk;
    pkt->RR.rarg1 = (unsigned long) arg;

    //memset(pkt->RR.filename, 0, MAX_FILENAME_LENGTH);
    pkt->RR.namelen = cfile->filename_len;
    strncpy(pkt->RR.filename, cfile->filename, MAX_FILENAME_LENGTH);
    pkt->RR.filename[MAX_FILENAME_LENGTH - 1] = 0;

    dbg("send chunk-RR: (%s) %ld@%ld,  rdma-bufid %d,free-slots=%d\n", cfile->filename, chunk->curr_pos, chunk->offset, chunk->bufid, hca->rdma_buf->free_slots);

    /// post send
    return ib_connection_post_send(conn, qpidx, slot, sizeof(*pkt));
}

#define is_server(id) (id==1)

/*
process a CQE, taking different actions based on "server/client"
("conn", "qp") is the (connection-qp ) where this wc happens
"tpool" is iothread_pool
*/
int exam_cqe(struct ib_connection *conn, int qpidx, struct ibv_wc *wc, int server, struct thread_pool *tpool)
{
    struct ib_HCA *hca = conn->hca;
    uint64_t arg1, arg2;

    int slot, qpi;
    void *slotaddr;
    ib_packet_t *pkt;
    int ret = 0;
    //dump_wc( wc );    
    switch (wc->opcode) {
    case IBV_WC_SEND:          // ibv_send complete. Can free the send-buf
        /// ibv_send().wr_id = qpidx | sbuf-slot
        conn->send_cqe++;
        qpi = (int) (wc->wr_id >> qpidx_shift); // qpi must == qpidx
        if (qpi != qpidx) {
            error("WC_SEND: Error! qpi %d != qpidx %d\n", qpi, qpidx);
            return -5;
        }
        slot = ((int) wc->wr_id) & ((1UL << qpidx_shift) - 1);
        conn->qpstat[qpidx].send++;
        pkt = (ib_packet_t *) ib_buffer_slot_addr(hca->send_buf, slot);
        if (!pkt)
            return -5;
        dbg("send complete at conn 0 : qp %d : send-slot %d\n", qpidx, slot);
        ///// for client only: if I have sent rqst_terminate, 
        //       I should also terminate this connection 
        if (pkt->command == rqst_terminate) {
            ret = -2;
        }
        free_buf_slot(hca->send_buf, slot, 0);
        //if( send_cqe >=32 ){
        //finish = 1;
        //}
        dbg("  ===================\n\n");
        break;
    case IBV_WC_RDMA_WRITE:

        break;
    case IBV_WC_RDMA_READ:     // finished a RDMA-READ, only on server
        /// RR-wr.wr_id = ib_packet_t*
        pkt = (void *) wc->wr_id;
        dbg("finished RDMA-READ (%s) %ld@%ld,rbufid=%d, lbufid=%d\n",   //
            pkt->RR.filename, pkt->RR.size, pkt->RR.offset, pkt->RR.rbuf_id, pkt->RR.lbuf_id);  // pkt->RR.laddr );
        //dump_ib_packet(pkt);

        /// wake up the iothread who is waiting for this RR to complete
        sem_t *sem = (sem_t *) pkt->RR.larg1;
        if (sem)
            sem_post(sem);      // wake up the iothread, who is waiting for the RR to complete
        else {
            error(" Error:: RR-WC, but sem invalid!! rbuf-id=%d, lbuf-id=%d\n", pkt->RR.rbuf_id, pkt->RR.lbuf_id);
        }

        conn->rr_cqe++;

        break;
    case IBV_WC_COMP_SWAP:

        break;
    case IBV_WC_FETCH_ADD:

        break;
    case IBV_WC_BIND_MW:

        break;
        //// recv-side: inbound completion
    case IBV_WC_RECV:          // receive an inbound message
        conn->recv_cqe++;
        hca->rq_wqe--;
        ib_connection_fillup_srq(hca);

        //qpidx = find_qp( wc->qp_num, conn );
        slot = (int) wc->wr_id; // recv-buf slot id
        slotaddr = ib_buffer_slot_addr(hca->recv_buf, slot);
        if (!slotaddr) {
            error("Error!!!  cannot find recvbuf for wc\n");
            dump_wc(wc);
        }
        pkt = (ib_packet_t *) slotaddr;
        dbg("Recv a msg at conn %p: qp %d : recv-slot %d\n", conn, qpidx, slot);
        conn->qpstat[qpidx].recv++;
        ////
        dump_ib_packet(pkt);
        ////////
        if (is_server(server)) {
            ///////////// server
            if (pkt->command == rqst_terminate) {
                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf
                ret = -2;       /// will terminate the loop               
            } else if (pkt->command == rqst_gen) {
                /// will reply with a generic-pkt

                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf                   
            } else if (pkt->command == rqst_RR) {
                /* has gotten a RR rqst, add a ref to a record about this RR-ckpt-file
                   into a hash table, then add one work-item(for one chunk) to work-queue. 
                   One io-thread will wake up and fetch this work-item, lookup the hash
                   table to locate the ckpt-file record, open the ckpt file if needed,
                   and perform RR, write the RR data to this file
                 */
                //ckpt_file_t *cfile = hash_table_get_record(ht_cfile, pkt->RR.rckptid, pkt->RR.rprocid, pkt->RR.larg1 );
                //ckpt_file_t *cfile = 
                //hash_table_get_record(ht_cfile, "file", pkt->RR.larg1 );
                ckpt_file_t *cfile;
                cfile = hash_table_get_record(ht_cfile, pkt->RR.filename, pkt->RR.is_last_chunk);
                // dump_hash_table( ht_cfile );
                cfile->adv_size += pkt->RR.size;
                arg1 = (((uint64_t) qpidx) << 32) | slot;
                arg2 = (uint64_t) conn;
                /// insert the rqst to queue, one iothr will unblock and work on it
                workqueue_enqueue3(tpool->queue, pkt, sizeof(*pkt), arg1, arg2, (unsigned long) cfile);
                // now can free the recv-buf slot
                free_buf_slot(hca->recv_buf, slot, 0);

            } else {
                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf
                error("server:  unknown WC\n");
                ret = -1;
            }
        } else {
            /////////////// client side
            if (pkt->command == reply_RR) {
                // RR finished, the infor about RR is in pkt
                //dbg("  RR reply : procid=%d, buf=%p, bufid = %d, size = %d\n", 
                //  pkt->RR.rprocid, pkt->RR.raddr, pkt->RR.rbuf_id, pkt->RR.size );

                if (pkt->RR.size > 0)   // only free valid rdma-buf chunk
                    free_buf_slot(hca->rdma_buf, pkt->RR.rbuf_id, 0);

                dbg("Finished RR: (%s) %ld@%ld, free rdma-buf id %d, free-slots=%d\n", pkt->RR.filename, pkt->RR.size, pkt->RR.offset, pkt->RR.rbuf_id, hca->rdma_buf->free_slots);

                if (pkt->RR.rarg1)
                    sem_post((sem_t *) (pkt->RR.rarg1));    //wait up a io-thr

                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf
                conn->rr_cqe++;
                g_complete_RR++;
                //if( conn->rr_cqe >= numRR )   ret = -1;
                //if( g_complete_RR >= numRR )  ret = -1;

            } else if (pkt->command == reply_gen) {
                dbg("    msg = \"%s\"\n", slotaddr);
                free_buf_slot(hca->recv_buf, slot, 0);
            } else if (pkt->command == rqst_terminate) {
                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf
                ret = -1;       /// will terminate the loop               
            } else {
                free_buf_slot(hca->recv_buf, slot, 0);  // free the recv-buf
                error("Client WC-RECV:  unknown WC\n");
                ret = -1;
            }

        }
        //i = get_buf_slot( conn->send_buf, 0 );
        //sprintf(ib_buffer_slot_addr(conn->send_buf, i), "qp %d slot %d", qpidx, i );
        //ret = ib_connection_post_send( conn, qpidx, i, 64 );
        //sprintf(msg, "Server qp %d msg %d", qpidx, send_cqe);
        //ret = ib_connection_send( conn, qpidx, msg, 64 );
        //dbg("  Post send req, slot %d, ret %d\n", i, ret);
        dbg("  ===================\n\n");

        break;
    case IBV_WC_RECV_RDMA_WITH_IMM:

        break;
    default:

        break;
    }                           // switch(wc->opcode)

    return ret;
}

int all_connection_finished(struct ib_connection *connarray, int num_conn)
{
    int i;
    for (i = 0; i < num_conn; i++) {
        if (connarray[i].status == connection_active)
            return 0;
    }
    return 1;
}

/*
The "tp" is iothread_pool
*/
int ib_server_loop(struct ib_connection *connarray, struct thread_pool *tp)
{
    struct ibv_wc wc_array[8];
    struct ibv_cq *ev_cq;
    struct ibv_wc *wc;
    int num_wc = 8;

    int ret = 0;
    void *cq_ctx;
    int i, ne;

    int connidx;
    int qpidx;

    struct ib_HCA *hca = connarray[0].hca;
    int num_conn = atomic_read(&hca->ref);  // num of connections using this HCA, = num of client nodes
    struct pollfd pfd;

    //num_wc = 8; // 256; //hca->max_rq_wqe + num_conn * connarray[0].num_qp*16;    
    //wc_array = malloc( num_wc * sizeof(struct ibv_wc*) );

    dbg("begin ib-server loop...\n");
    while (!g_exit)
        //while( ! all_connection_finished(connarray, atomic_read( &hca->ref) ) )
    {
        ////////////  do a non-blocking poll on the comp_channel
#ifdef    POLL_COMP_CHANNEL
        pfd.fd = hca->comp_channel->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        ret = poll(&pfd, 1, 1500);  // poll 1 fd, wait 1000 ms inbetween

        if (ret == 0) {         // timeout, no event happens
            // check if user wants exit
            if (g_exit) {
                printf("%s: poll find exit...\n", __func__);
                break;
            }
            continue;
        }
        if (ret < 0) {
            error("poll failed, ret=%d\n", ret);
            perror("");
            break;
        }
#endif
        ///////////////////////////////////

        // wait for a comp_event
        ret = ibv_get_cq_event(hca->comp_channel, &ev_cq, &cq_ctx);
        //dbg("get_cq_event ret %d\n", ret);
        if (ret != 0) {
            error("Error to get comp_channel event\n");
            break;
        }
        hca->comp_event_cnt++;

        //dbg("get_cq_event ret %d, evt_cnt=%d\n", ret, conn->comp_event_cnt);
        if (ev_cq != hca->cq) {
            error("event_cq %p not the correct CQ %p!!\n", ev_cq, hca->cq);
            return -1;
        }
        /// cumulative ack
        if (hca->comp_event_cnt % 10 == 0) {
            ibv_ack_cq_events(hca->cq, 10);
        }
        // enable future comp_event
        ret = ibv_req_notify_cq(hca->cq, 0);
        if (ret) {
            error("Error req_notify_cq\n");
            return ret;
        }
        ///////////////////////////////
        /// poll the pending CQE in CQ
        do {
            ne = ibv_poll_cq(hca->cq, num_wc, wc_array);
            if (ne < 0) {       // this can never happen
                fprintf(stderr, "Failed to poll completions from the CQ\n");
                ret = ne;
                goto err_out_1;
            }
            //dbg(" Poll get %d cqe\n", ne);
            if (ne == 0)
                break;          // no cqe in CQ, can break the do-while   

            hca->total_cqe += ne;

            for (i = 0; i < ne; i++) {
                wc = &wc_array[i];
                //dump_wc( wc );
                /// find the (connection id, QP id) of this WC
                if (find_qp_from_conn_array(wc->qp_num, connarray, atomic_read(&hca->ref), &connidx, &qpidx) != 0) {
                    error("Error:: Fail to find connidx, qpidx for qpn %d\n", wc->qp_num);
                    //dump_wc(wc);
                } else {
                    dbg(" get CQE at conn %d : qpidx %d\n", connidx, qpidx);
                    ret = exam_cqe(connarray + connidx, qpidx, wc, 1, tp);
                    if (ret < 0) {
                        //finish = 1;
                        // this connection is terminated
                        connarray[connidx].status = connection_terminated;
                        //break;
                    }
                }
            }                   // end of for( ; i<ne; )
        }
        while (ne > 0);         // end of do { ibv_poll_cq() ... }
        ////////// end of ibv_poll_cq() for pending CQEs

    }                           // while( !finish )

  err_out_1:
    dbg("about to exit...\n");
    num_conn = atomic_read(&hca->ref);
    for (i = 0; i < num_conn; i++) {
        struct ib_connection *conn = connarray + i;
        printf("connection %d stat:: \n", i);
        dump_qpstat(conn);
        //printf("    send-cqe %d, recv-cqe %d, total-cqe %d\n", conn->send_cqe, conn->recv_cqe, conn->hca->total_cqe);
    }
    //free(wc_array);

    return ret;
}

int client_finished()
{
//  return (g_client_complete);
    return (g_complete_RR >= numRR);
}

int ib_client_loop(struct ib_connection *connarray)
{
    //struct ib_connection* conn;

    struct ibv_cq *ev_cq;
    struct ibv_wc wc_array[8];

    struct ibv_wc *wc;

    int num_wc;
    int ret = 0;
    void *cq_ctx;
    int i, ne;

    int connidx;
    int qpidx;
    int finish = 0;

    struct ib_HCA *hca = connarray[0].hca;
    int num_conn = atomic_read(&hca->ref);  // num of connections using this HCA
    struct pollfd pfd;

    /// uint64_t
    num_wc = 8;                 // 256; //hca->max_rq_wqe + num_conn * connarray[0].num_qp*16;
    //wc_array = malloc( num_wc * sizeof(struct ibv_wc*) );

    /////////////
    dbg("begin client loop...\n");

    //while( ! client_finished() )
    //while( !finish )
    while (!all_connection_finished(connarray, atomic_read(&hca->ref))) {
        ////////////  do a non-blocking poll on the comp_channel
#ifdef    POLL_COMP_CHANNEL
        pfd.fd = hca->comp_channel->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        ret = poll(&pfd, 1, 1500);  // poll 1 fd, wait 1000 ms inbetween

        if (ret == 0) {         // timeout, no event happens
            // check if user wants exit
            if (g_exit) {
                printf("%s: poll find exit...\n", __func__);
                break;
            }
            continue;
        }
        if (ret < 0) {
            error("poll failed, ret=%d\n", ret);
            perror("");
            break;
        }
#endif
        ///////////////////////////////////

        // wait for a comp_event
        ret = ibv_get_cq_event(hca->comp_channel, &ev_cq, &cq_ctx);
        //dbg("get_cq_event ret %d\n", ret);
        if (ret != 0) {
            error("Error to get comp_channel event\n");
            break;
        }

        hca->comp_event_cnt++;

        //dbg("get_cq_event ret %d, evt_cnt=%d\n", ret, conn->cq_event_cnt);
        if (ev_cq != hca->cq) {
            error("event_cq %p not the correct CQ %p!!\n", ev_cq, hca->cq);
            goto err_out_1;
        }
        // cumulative ack
        if (hca->comp_event_cnt % 10 == 0) {
            ibv_ack_cq_events(hca->cq, 10);
        }
        // enable future comp_event
        ret = ibv_req_notify_cq(hca->cq, 0);
        if (ret) {
            error("Error req_notify_cq\n");
            return ret;
        }
        ///////////////////////////////
        /// poll the just coming comp_event
        do {
            ne = ibv_poll_cq(hca->cq, num_wc, wc_array);
            if (ne < 0) {       // this can never happen
                fprintf(stderr, "Failed to poll completions from the CQ\n");
                goto err_out_1; // return -1;
            }
            //dbg(" Poll get %d cqe\n", ne);            
            if (ne == 0)
                break;
            hca->total_cqe += ne;
            ///////////////////////////////
            /// wc->wr_id =   connection-id | qp-id | buf-slot-id
            for (i = 0; i < ne; i++) {
                wc = &wc_array[i];
                //dump_wc( wc_array+i );
                if (find_qp_from_conn_array(wc->qp_num, connarray, atomic_read(&hca->ref), &connidx, &qpidx) != 0) {
                    error("Error:: Fail to find connidx, qpidx for qpn %d\n", wc->qp_num);
                    //dump_wc(wc);
                } else {
                    //ret = exam_cqe(conn, wc, 0, NULL );
                    ret = exam_cqe(connarray + connidx, qpidx, wc, 0, NULL);
                    if (ret < 0) {
                        // -1:  has finished numRR rqst, 
                        // -2:  has sent rqst_terminate to srv
                        //finish = 1;
                        connarray[connidx].status = connection_terminated;
                        //break;
                    }
                }
            }
        }
        while (ne && !finish);  // end of do { ... ibv_poll_cq() ... }
        ////////////////////////

    }                           // while( !finish )

    /// tell each server to exit
/*    //if( g_client_rank == 0 )    
    {
        for(i=0; i<g_num_srv; i++){
            qpidx = get_next_qp( &connarray[i] );
            printf(" ************* tell srv %d of %d to exit\n", i, g_num_srv);
            slot = get_buf_slot( hca->send_buf, (void**)&pkt, 0 );
            pkt->command = rqst_terminate;
            ib_connection_post_send( &connarray[i], qpidx, slot, sizeof(*pkt) );
            connarray[i].status = connection_terminated;
        }
    }    */

  err_out_1:
    num_conn = atomic_read(&hca->ref);
    for (i = 0; i < num_conn; i++) {
        struct ib_connection *conn = connarray + i;
        printf("connection %d:: \n", i);
        dump_qpstat(conn);
        //dbg("    send-cqe %d, recv-cqe %d, total-cqe %d\n", conn->send_cqe, conn->recv_cqe, conn->hca->total_cqe);
    }
    //free(wc_array);
    return ret;
}

/*
Send "terminate" rqst to other end.
Called before all ib_connections to be closed.
*/
void ib_connection_send_terminate_rqst(struct ib_connection *connarray)
{
    struct ib_HCA *hca = connarray[0].hca;
    int numconn = atomic_read(&hca->ref);
    int i, qpidx, slot;
    struct ib_packet *pkt;

    for (i = 0; i < numconn; i++) {
        qpidx = get_next_qp(&connarray[i]);
        //printf(" ************* tell srv %d of %d to exit\n", i, numconn ); 
        slot = get_buf_slot(hca->send_buf, (void *) &pkt, 0);
        pkt->command = rqst_terminate;
        ib_connection_post_send(&connarray[i], qpidx, slot, sizeof(*pkt));
        //connarray[i].status = connection_terminated;
    }
}

/*
IB-server-loop needs a hash-table
*/
void pass_hash_table(hash_table_t * ht)
{
    ht_cfile = ht;
}

#endif
