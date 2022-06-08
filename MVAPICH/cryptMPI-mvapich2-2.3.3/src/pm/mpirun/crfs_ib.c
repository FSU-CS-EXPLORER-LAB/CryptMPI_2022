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

// for pwrite()
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <mpichconf.h>

#ifdef CR_AGGRE

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>

#include <string.h>
#include <strings.h>
#include <malloc.h>
#include <netinet/in.h>
#include <byteswap.h>
#include <inttypes.h>
#include <signal.h>
#include <errno.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>

#include "ib_comm.h"
#include "ib_buf.h"
#include "debug.h"
//#include "ftb.h"
#include "bitmap.h"
#include "thread_pool.h"
#include "openhash.h"
#include "crfs.h"

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////// global vars, common to both cli & srv
#define MAX_CONN_NUM    (32)
int g_exit = 0;
struct ib_HCA hca;

int num_connection = 0;
struct ib_connection conn_array[MAX_CONN_NUM];

hash_table_t *g_ht_cfile;       // hash table for ckpt-files

struct thread_pool *iopool;     // pool of io-threads

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////       server side code

static pthread_t ibthr;         // ib-main-thread, takes care of ib_server_loop
static pthread_t listen_thr;    // istening on a port for connection rqst

static sem_t test_sem;

static pthread_mutex_t g_mutex;

static void *ibsrv_ioprocess(void *arg);
static void *ibsrv_listen_port(void *arg);
static void *ibsrv_main_loop(void *arg);
static void ibsrv_connection_server(void *arg);

int init_all_connections(struct ib_connection *carray, int num);
int release_all_connections(struct ib_connection *carray, int num);

/**
For CRFS: working as a server, accept data from migration source
**/
int ibsrv_main_entry(void *p)
{
    sendslot_size = recvslot_size = sizeof(ib_packet_t);

    sem_init(&test_sem, 0, 0);
    pthread_mutex_init(&g_mutex, NULL);

    ///     1. init the HCA, when a connection is rqsted, setup a connection
    if (ib_HCA_init(&hca, 1)) {
        printf("HCA-init error!!\n");
        goto err_out_1;
    }
    /// 2. create the IO-thread pool
    //tp_init_thread_pool( &io_pool, g_iopool_size, ibsrv_ioprocess, "io_pool" );
    iopool = tp_create_thread_pool(g_iopool_size, ibsrv_ioprocess, "io_pool");
    if (!iopool) {
        error("Error!! fail to create io-thread-pool\n");
        goto err_out_2;
    }
    /// 3. start a ib-main-loop to wait for IB-pkts
    if (pthread_create(&ibthr, NULL, ibsrv_main_loop, (void *) iopool) != 0) {
        error("Error!! fail to create ib-thr\n");
        goto err_out_3;
    }
    //// 4. start tcp-srv loop
    if (pthread_create(&listen_thr, NULL, ibsrv_listen_port, &g_srv_tcpport) != 0) {
        error("Error!! fail to create listen-thr\n");
        goto err_out_4;
    }
    g_ht_cfile = create_hash_table(128, "ckpt-hash-table");
    if (!g_ht_cfile) {
        error("Fail to create hash-table...\n");
        goto err_out_5;
    }
    pass_hash_table(g_ht_cfile);    /// tell ib_server: use this hash_table

    init_all_connections(conn_array, MAX_CONN_NUM);
    /////// test ftb
/*    if( ftb_init() != 0 ){
        error("fail to connect ftb...\n");
        goto err_out_6;
    }
    //ftb_publish_msg( MSG_TEST );   */

    return 0;

    /// release the hash table
    destroy_hash_table(g_ht_cfile);

  err_out_5:

  err_out_4:
    g_exit = 1;                 //g_ibsrv_finish = 1;
    pthread_join(listen_thr, NULL);

  err_out_3:                   // terminate the ib_main_loop
    g_exit = 1;                 //g_ibsrv_finish = 1;
    sem_post(&test_sem);
    pthread_join(ibthr, NULL);

  err_out_2:                   // free the io-pool
    tp_destroy_thread_pool(iopool);

  err_out_1:                   /// release the HCA
    ib_HCA_destroy(&hca);

    pthread_mutex_destroy(&g_mutex);
    sem_destroy(&test_sem);

    g_exit = 1;
    return -1;
}

int ibsrv_main_exit()
{
    dbg("dump ib-bufs before exit...\n");
    dump_ib_buffer(hca.send_buf);
    dump_ib_buffer(hca.recv_buf);
    dump_ib_buffer(hca.rdma_buf);

    if (!g_exit) {
        //////////// Now exit.    Clear resources, then exit
        dbg("ib-server exit...\n");
        //g_ibsrv_finish = 1;       
        g_exit = 1;

        /// release the hash table
        destroy_hash_table(g_ht_cfile);

        /////////// FTB
        //ftb_terminate();
        /////////////////////////////

        // terminate the ib_main_loop
        sem_post(&test_sem);

        // free the io-pool
        tp_destroy_thread_pool(iopool);

        /// wait the ib-main-loop to terminate
        pthread_join(ibthr, NULL);
        pthread_join(listen_thr, NULL);

        ib_HCA_destroy(&hca);

        dbg("ib-server exit...\n");

        pthread_mutex_destroy(&g_mutex);
        sem_destroy(&test_sem);
    }

    g_exit = 1;
    return 0;
}

/****
process one RR rqst from client
****/
void *ibsrv_ioprocess(void *arg)
{
    int i;

    struct thread_pool *ownertp = (struct thread_pool *) arg;
    sem_t sem;
    struct ib_connection *conn;
    struct work_elem welem;     // a RDMA rqst 
    struct ib_packet *rrpkt;

    pthread_t mytid = pthread_self();   // my-thread id
    int myid = -1;

    sem_init(&sem, 0, 0);

    sleep(1);                   // wait for the complete io-pool to be created

    /// find my id
    for (i = 0; i < ownertp->num_threads; i++) {
        if (mytid == ownertp->thread[i]) {
            myid = i;
            break;
        }
    }

    /////////
    if (myid < 0) {             // Error!!
        error("Error::  mytid = %lu, fail to find id\n", mytid);
        goto err_out_1;
    } else {
        printf(" ioprocess %lu, id = %d\n", mytid, myid);
    }

    while (1) {
        // wait to be waken up
        // get a RDMA rqst from the queue, copy this item to local welem
        workqueue_dequeue(ownertp->queue, &welem);

        // terminate?
        if (welem.arg1 == arg_invalid || welem.arg2 == arg_invalid) {
            printf(" iothread %d exit...\n", myid);
            break;
        }
        //printf("::::  iothr %d::   \n", myid);
        //dump_queue( ownertp->queue );
        //dump_work_elem( &welem );

        //////  welem.data is the RR-rqst pkt,              
        ///  arg1 =  (qpidx | recvbuf-slot), arg2: ib_connection*
        uint64_t t = welem.arg1;    // ownertp->arg1[myid];
        int qpidx = (int) (t >> 32);
        int slot = (int) (t & 0x0ffffffffUL);   // recv-buf slot: containing the RR info
        // this "slot" var is meaningless now... DON'T use it!!!
        conn = (struct ib_connection *) welem.arg2;
        ckpt_file_t *cfile = (ckpt_file_t *) welem.arg3;

        struct ib_HCA *hca = conn->hca;

        //printf("iop %d: arg1=0x%lx, arg2=0x%lx\n", myid, ownertp->arg1[myid], ownertp->arg2[myid] );  

        //////// 1.   start the RDMA-READ
        rrpkt = (struct ib_packet *) welem.data;
        rrpkt->RR.larg1 = (uint64_t) & sem;
        if (rrpkt->RR.size > 0) {
            int lbuf_id __attribute__((__unused__)) = ib_connection_post_RR(conn, qpidx, rrpkt);

            // Now, lbuf_id is local RDMA-buf id to store the RR data
            dbg("[iot_%d]: post RR using local-buf %d\n", myid, lbuf_id);

            ///// wait for the RR to complete
            sem_wait(&sem);

            //printf("[iot_%d]: done RR: (%s) %d@%d,is-last=%d, lbuf %d, \n", myid, rrpkt->RR.filename,
            //  rrpkt->RR.size, rrpkt->RR.offset, rrpkt->RR.is_last_chunk, rrpkt->RR.lbuf_id );
        } else {
            dbg("*** [iot_%d]: got an empty chk: (%s) %ld@%ld, is-last=%d\n", myid, rrpkt->RR.filename, rrpkt->RR.size, rrpkt->RR.offset, rrpkt->RR.is_last_chunk);
        }

        /////// 2. now, the RR is complete. Attach the new chunk of data to cfile's chunk-list
        add_chunk_to_ckpt_file(cfile, rrpkt);
        //cfile->write_size += rrpkt->RR.size;

        ///////  decrease the reference of ckpt-file in the hash-table
        hash_table_put_record(g_ht_cfile, cfile, 0);

        ////// now the data has been RDMA-READ to local buf,                
        {                       // Prepare a reply to remote peer, copy the RR-info-pkt
            struct ib_packet *spkt;
            slot = get_buf_slot(hca->send_buf, (void *) &spkt, 0);
            memcpy(spkt, rrpkt, sizeof(*spkt)); // copy RR-infor to the send-buf
            spkt->command = reply_RR;
            dbg("[iot_%d]: reply RR: (%s) %ld@%ld, is-last=%d\n", myid, rrpkt->RR.filename, rrpkt->RR.size, rrpkt->RR.offset, rrpkt->RR.is_last_chunk);

            ib_connection_post_send(conn, qpidx, slot, sizeof(*spkt));
        }

    }
    ///////
    ///close(fd);
    //////////////////

  err_out_1:
    sem_destroy(&sem);
    //dbg("iothr %d exit...\n", myid);
    pthread_exit(NULL);
}

// this loop monitors the comp_channel & CQ, binding to a HCA
void *ibsrv_main_loop(void *arg)
{
    struct thread_pool *iopool = (struct thread_pool *) arg;

    dbg(" ib-main-loop waiting to be waken up... \n");

    int ret = sem_wait(&test_sem);

    if (ret != 0) {
        printf("%s: sem ret %d...\n", __func__, ret);
        return NULL;
    }
    if (g_exit) {               //g_ibsrv_finish ) { // g_finish ){  
        printf("%s:  sem interrupted\n", __func__);
        return NULL;
    }
    dbg(" ib-main-loop start, go in ib_server_loop ... \n");

    ib_server_loop(conn_array, iopool);

    //g_exit = 1;

    release_all_connections(conn_array, MAX_CONN_NUM);
    dbg(" ib-main-loop exit ... \n");
    pthread_exit(NULL);

}

/// runs in one thread
void *ibsrv_listen_port(void *arg)
{
    int port = *((int *) (arg));
    char msg[64];
    int srv_sock;

    srv_sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in srv;

    srv.sin_family = AF_INET;
    srv.sin_port = (port == 0) ? htons(g_srv_tcpport) : htons(port);
    srv.sin_addr.s_addr = htonl(INADDR_ANY);

    bind(srv_sock, (struct sockaddr *) &srv, sizeof(srv));

    socklen_t namelen = sizeof(struct sockaddr_in);
    if (getsockname(srv_sock, (struct sockaddr *) &srv, &namelen) < 0) {
        perror("getting sock name");
        exit(3);
    }
    //char* pn = inet_ntop(srv.sin_addr);
    //char* pn = inet_ntop(AF_INET, (void*)&(srv.sin_addr), msg, INET_ADDRSTRLEN);
    dbg("Server listening on port %d\n",    /// inet_ntoa(srv.sin_addr), 
        ntohs(srv.sin_port));

    /////////////////////////////////////// 
    struct sockaddr_in cli_addr;
    socklen_t addrlen = sizeof(struct sockaddr_in);
    int msgsock;
    int i;
    fd_set fds;
    struct timeval timeout;

    listen(srv_sock, 8);        // srv listen on a port

    while (!g_exit)             //g_ibsrv_finish ) // g_finish )
    {
        FD_ZERO(&fds);
        FD_SET(srv_sock, &fds);

        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        i = select(FD_SETSIZE, &fds, NULL, NULL, &timeout);
        if (i < 0) {
            error("Error:: select() ret %d\n", i);
            continue;
        } else if (i == 0) {    // timeout
            if (g_exit)
                break;
            continue;
        }
        /// has a incoming request // FD_ISSET( srv_sock, &fds );
        msgsock = accept(srv_sock, (struct sockaddr *) (&cli_addr), &addrlen);
        if (msgsock < 0) {
            dbg("will break\n");
        }
        inet_ntop(AF_INET, (void *) &(cli_addr.sin_addr), msg, 64);
        dbg("\tSrv get con from %s: %d\n", msg, ntohs(cli_addr.sin_port));

        dbg("process connection %d...\n", num_connection);

        uint64_t t = (((uint64_t) num_connection) << 32) | msgsock;

        ibsrv_connection_server((void *) t);    // setup a connection to this coming client
        num_connection++;

    }

    close(srv_sock);
    printf("Has gotten %d client-requests\n", num_connection);

    /*
       ///and free the ib_connection
       for(i=0; i<num_connection; i++){
       //pthread_join( conn_thread[i], NULL );
       ib_connection_release( &conn_array[i] );
       }    */
    /////////   
    return NULL;
}

int init_all_connections(struct ib_connection *carray, int num)
{
    int i = 0;
    for (i = 0; i < num; i++) {
        carray[i].status = connection_uninit;
    }

    return num;
}

int release_all_connections(struct ib_connection *carray, int num)
{
    int cnt = 0;
    int i = 0;
    struct ib_connection *conn;

    for (i = 0; i < num; i++) {
        conn = carray + i;
        if (conn->status == connection_active) {
            printf("%s: conn_%d still active...\n", __func__, i);
            ib_connection_release(conn);
            cnt++;
        } else if (conn->status == connection_terminated) {
            ib_connection_release(conn);
            cnt++;
        }
    }
    dbg("Has released %d connections...\n", cnt);
    return cnt;
}

/*
When a client inits a connection to server, 
this function is called to setup one ib_connection to the client
*/
void ibsrv_connection_server(void *arg)
{
    static int has_init = 0;

    int newsock;
    int myid;

    uint64_t t = (uint64_t) arg;

    newsock = (int) t;
    myid = (int) (t >> 32);
    struct ib_connection *conn = conn_array + myid; // ib-connection for this ib-srv

    {
        dbg("build connection %d, using socket %d...\n", myid, newsock);

        // now newsock is used to commu with client
        /// 1. set up ib-conn step 1
        ib_connection_init_1(conn, g_num_qp, &hca); // g_rx_depth );
        if (has_init == 0)      // myid == 0 ) // the first connection-rqst, set up the HCA
            ib_connection_fillup_srq(&hca);

        /// 2. from server-side: exchange info with client
        ib_connection_exchange_server(conn, newsock);
        close(newsock);

        /// 3. finish ib-conn setup
        ib_connection_init_2(conn);

        conn->status = connection_active;

        // now IB port is ready. Wake up ib-main-loop to check the ib-channel           
        //if( myid == 0 ) 
        if (has_init == 0)      // 
            sem_post(&test_sem);

        dbg("connection %d is ready...\n", myid);

    }
    has_init++;

    //dbg("ib-conn-srv %d finished...\n", myid);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////       client side code

static int exchange_with_server(char *srvname, int srvport, struct ib_connection *conn);
static void *ibcli_ioprocess(void *arg);

int ibcli_main_entry(void *p)
{
    sendslot_size = recvslot_size = sizeof(ib_packet_t);

    /// 1. init the IB 
    if (ib_HCA_init(&hca, 1)) {
        printf("HCA-init error!!\n");
        goto err_out_1;
    }
    //client_work( xxx );

    /// 2. create the IO-thread pool for client-IO
    iopool = tp_create_thread_pool(g_iopool_size, ibcli_ioprocess, "cli_io_pool");
    if (!iopool) {
        error("Error!! fail to create io-thread-pool\n");
        goto err_out_2;
    }
    /// 3. create the ckpt-file hash-table
    g_ht_cfile = create_hash_table(32, "cli-ckpt-hash");
    if (!g_ht_cfile) {
        error("Fail to create hash-table...\n");
        goto err_out_3;
    }
    g_exit = 0;
    init_all_connections(conn_array, MAX_CONN_NUM);

    dbg("Has inited cli-main-entry...\n");
    return 0;

  err_out_3:
    tp_destroy_thread_pool(iopool);
    iopool = NULL;

  err_out_2:
    ib_HCA_destroy(&hca);

  err_out_1:
    g_exit = 1;
    return -1;
}

void ibcli_main_exit()
{
    dbg("dump ib-bufs before exit...\n");
    dump_ib_buffer(hca.send_buf);
    dump_ib_buffer(hca.recv_buf);
    dump_ib_buffer(hca.rdma_buf);

    if (g_exit)
        return;
    /// release the hash-table
    if (g_ht_cfile) {
        destroy_hash_table(g_ht_cfile);
        g_ht_cfile = NULL;
    }
    /// release the io-thr-pool
    if (iopool) {
        printf("free thread-pool\n");
        // free the io-pool
        tp_destroy_thread_pool(iopool);
        iopool = NULL;
    }
    /// release the hca-driver
    ib_HCA_destroy(&hca);

    g_exit = 1;
}

/// a client initiates connection to server, exchange qp_endpoint infor
int exchange_with_server(char *srvname, int srvport, struct ib_connection *conn)
{
    char msg[128];

    struct sockaddr_in srv_addr;
    struct hostent *hp;         // get srv's addr infor

    int cli_sock = socket(AF_INET, SOCK_STREAM, 0);

    // now setup srv's addr
    srv_addr.sin_family = AF_INET;
    srv_addr.sin_port = htons(srvport);

    hp = gethostbyname(srvname);    // get srv's addr
    if (!hp) {                  // fail to get srv's addr
        error("fail at gethostbyname: %s\n", srvname);
        return -1;
    }
    memcpy(&(srv_addr.sin_addr.s_addr), hp->h_addr, hp->h_length);

    /// now connect to srv
    if (connect(cli_sock, (struct sockaddr *) &srv_addr, sizeof(srv_addr)) < 0) {
        error("%s: fail to connect to srv\n", __func__);
        return -1;
    }

    inet_ntop(AF_INET, (void *) &(srv_addr.sin_addr), msg, 64);

    printf("cli connect to srv %s:: %s : %d\n", srvname, msg, ntohs(srv_addr.sin_port));

    ////////////////////////////////////////
    //// now exchange info with server
    ib_connection_exchange_client(conn, cli_sock);
    /////////////////////////////////////

    close(cli_sock);

    return 0;

}

/**

**/
void *ibcli_mig_src(void *arg)  // char* srv[], int numsrv, int port)
{
    int ret;

    struct ib_connection *conn;

    mig_info_t *minfo = (mig_info_t *) arg;

    struct timeval tstart, tend;

    //num_connection = 0;
    ////  establish one IB connections to each server
    //for(i=0; i<numsrv; i++)
    minfo->fail_flag = 0;
    gettimeofday(&tstart, NULL);
    {
        conn = &conn_array[num_connection];
        minfo->conn = conn;

        /// 1. set up ib-conn step 1
        ret = ib_connection_init_1(conn, g_num_qp, &hca);
        if (ret) {
            error("connection-init-1 fails ret %d\n", ret);
            return NULL;
        }
        if (num_connection == 0)    // post WQE to srq of the hca
            ib_connection_fillup_srq(&hca);

        /// 2. exchange info with server
        ret = exchange_with_server(minfo->tgt, minfo->port, conn);
        if (ret != 0) {         // fail to connect to server
            goto err_out_1;
        }
        /// 3. finish ib-conn setup
        ib_connection_init_2(conn);
        conn->status = connection_active;
        num_connection++;
    }

    gettimeofday(&tend, NULL);
    sem_post(&minfo->sem);

//  long us = (tend.tv_sec - tstart.tv_sec)*1000000 + (tend.tv_usec-tstart.tv_usec);
//  dbg(" Has connected: %s ==> %s, port %d, cost %ld us\n", 
//      minfo->src, minfo->tgt, minfo->port, us );

    //////////// create a flag file to release other procs
    extern char crfs_sessionid[128];
    char fname[128];
    snprintf(fname, 128, "/tmp/cr-%s-mig-begin", crfs_sessionid);
    int fd = open(fname, O_RDWR | O_CREAT, 0660);
    close(fd);

    //////////// start work
    gettimeofday(&tstart, NULL);
    //usleep(1000);     

    ///////  start working
    ib_client_loop(&conn_array[0]);

    gettimeofday(&tend, NULL);

    /////////////////////////////

    /*
       double sec = tv2sec(&tstart, &tend);
       double size = (1UL * rdmaslot_size) * numRR;
       printf("Client: send data %f MB, time %f sec, bw= %.3f MB/s\n", size/(1024*1024), 
       sec, size/(1024*1024) / sec );
     */

    pthread_exit(NULL);

  err_out_1:
    minfo->fail_flag = 1;
    sem_post(&minfo->sem);
    pthread_exit(NULL);
}

static pthread_t thr_migsrc;    // thread to work at mig-source node
static mig_info_t *p_mig = NULL;
/**
This is the source-node of a mig, and user has requested to start a mig. 
Start a new thread to connect to mig-target node,  loops over
the IB-comm routine to take care of all ib-events
**/
int ibcli_start_mig(mig_info_t * minfo)
{
    int ret;

    if (minfo->port == 0) {
        minfo->port = g_srv_tcpport;
    }
    dbg("will start thr_migsrc:  src=%s, tgt=%s, port=%d...\n", minfo->src, minfo->tgt, minfo->port);
    ret = pthread_create(&thr_migsrc, NULL, ibcli_mig_src, (void *) minfo);

    if (ret) {
        err("Fail to create mig-src-thread. errno=%d\n", errno);
        perror("Fail to create thr:: \n");
        p_mig = NULL;
    } else {
        // have created the thread to do mig. wait for it to init
        p_mig = minfo;
        sem_wait(&minfo->sem);  // wait for IB connection to be established
        if (minfo->fail_flag) {
            err("init migration failed...\n");
            return -1;
        }
    }

    return 0;
}

void ibcli_end_mig(mig_info_t * minfo)
{
    int i;

    dbg("will end mig. num_connection=%d\n", num_connection);

    if (atomic_read(&minfo->chunk_cnt) > 0) {
        // has pending chunks not finished, cannot terminate now..
        dbg("*****  cannot term before all chunks RRed, wait...\n");
        sem_wait(&minfo->chunk_comp_sem);
    }

    ib_connection_send_terminate_rqst(minfo->conn);

    pthread_join(thr_migsrc, NULL);

    /// release all ib-connection
    for (i = 0; i < num_connection; i++) {
        if (conn_array[i].status == connection_terminated)
            ib_connection_release(&conn_array[i]);

        conn_array[i].status = connection_uninit;
    }
    dbg("Has released all %d connections\n", num_connection);
    num_connection = 0;
    p_mig = NULL;
}

/*
process one data-chunk at client side
*/
void *ibcli_ioprocess(void *arg)
{
    int ret, i;

    char iofname[128];

    struct thread_pool *ownertp = (struct thread_pool *) arg;
    sem_t sem;
    //struct ib_packet* tmppkt;
    struct work_elem welem;     // a RDMA rqst 

    sem_init(&sem, 0, 0);
    sleep(1);                   // wait for the complete io-pool to be created

    pthread_t mytid = pthread_self();   // my-thread id
    int myid = -1;

    /// find my id
    for (i = 0; i < ownertp->num_threads; i++) {
        if (mytid == ownertp->thread[i]) {
            myid = i;
            break;
        }
    }
    /////////
    if (myid < 0) {             // Error!!
        error("Error::  mytid = %lu, fail to find id\n", mytid);
        goto err_out_1;
    } else {
        printf(" ibcli_iopro %lu, id = %d\n", mytid, myid);
    }

    //// main body...
    while (1) {
        if (crfs_mode != MODE_WRITEAGGRE) {
            err("CRFS-mode =%d, exit...\n", crfs_mode);
            break;
        }
        // wait to be waken up
        // get a RDMA rqst from the queue, copy this item to local-var
        workqueue_dequeue(ownertp->queue, &welem);
        //dbg("after wait for welem...\n");

        // terminate?
        if (welem.arg1 == arg_invalid || welem.arg2 == arg_invalid) {
            //printf(" iothread %d exit...\n", myid);
            break;
        }
        /// now get a full-chunk    
        ckpt_chunk_t *chunk = (ckpt_chunk_t *) welem.data;
        ckpt_file_t *cfile = chunk->ckpt_file;

        strncpy(iofname, cfile->filename, 128);
        iofname[128 - 1] = 0;

/* it turned out that, the perf bottleneck is at write-aggregation.
fuse implementation is low perf in writing...: use a LOT of mem internally, low throughput,
When tried 8 write processes 1GB each, and ignore the data chunks directly,
the aggre-write bw is only ~200MB/s !!!!!!!!!!!     */

        /**  printf("%s: [iot_%d]: aggre to buf %d, bypass...\n", __func__, myid, chunk->bufid);
        if( chunk->bufid >= 0 && chunk->curr_pos>0)
            ckpt_free_chunk( chunk );
        goto chunk_done; **/

        /// if this is src-node of a migration::
        if (mig_role == ROLE_MIG_SRC) { /// will perform RDMA to mig-tgt node
            dbg("[iot_%d]: send RR rqst: (%s) (%d@%d), lbuf %d, is-last-chunk=%d\n", myid, cfile->filename, chunk->curr_pos, chunk->offset, chunk->bufid, chunk->is_last_chunk);
            /// rqst server to RR this chunk    

            //// check the contents of the outgoing chunk...            
            //check_chunk_content( cfile, chunk, chunk->curr_pos );

            ib_connection_send_chunk_RR_rqst(p_mig->conn, cfile, chunk, &sem);

            /// wait for the RR to complete
            dbg("[iot_%d]: wait for sem...\n", myid);
            sem_wait(&sem);

            if (atomic_dec_and_test(&p_mig->chunk_cnt)) // the count has reached 0
            {
                //if( chunk->is_last_chunk )
                dbg("*****   will post chunk_comp_sem...\n");
                sem_post(&p_mig->chunk_comp_sem);   // signify: all chunks have been RRed to server
            }
            // at this point, the ib-loop has freed the chunk->bufid
            dbg("[iot_%d]: RR finished: (%s) (%d@%d), lbuf %d, is-last-chunk=%d\n", myid, cfile->filename, chunk->curr_pos, chunk->offset, chunk->bufid, chunk->is_last_chunk);
            //ckpt_free_chunk( chunk ); // free the buf-chunk associated with this chunk
        } else                  // in pure WA mode: only write to local file
        {
            //dbg("iothr %d: get buf=%d, chunk (%d@%d) for cfile: %s...\n", myid, chunk->bufid,
            //  chunk->curr_pos, chunk->offset, cfile->filename  );
            //pthread_mutex_lock( &cfile->io_mtx );
            if (chunk->curr_pos > 0)
                //ret = 0;
                ret = pwrite(cfile->fd, chunk->buf, chunk->curr_pos, chunk->offset);
            else
                ret = 0;
            //pthread_mutex_unlock( &cfile->io_mtx );
            if (chunk->bufid >= 0 && chunk->curr_pos > 0)
                ckpt_free_chunk(chunk); // free the buf-chunk associated with this chunk

            dbg("[iot_%d]: write (%s) (%d@%d): ret=%d, lbuf=%d, is-last-chunk=%d\n", myid, cfile->filename, chunk->curr_pos, chunk->offset, ret, chunk->bufid, chunk->is_last_chunk);
            if (ret < 0) {
                printf("[iot_%d] write (%s) (%lu@%lu) ret=%d, lbuf=%d, is-last-chunk=%d\n", myid, cfile->filename, chunk->curr_pos, chunk->offset, ret, chunk->bufid, chunk->is_last_chunk);
                perror("fail to pwrite  \n");
            }
        }                       // end of else( In pure WA mode )

// chunk_done:  
        if (cfile->can_release && !chunk->is_last_chunk && (chunk->offset + chunk->curr_pos == cfile->adv_size))    // this is the last chunk
        {
            chunk->is_last_chunk = 1;
        }

        if (chunk->is_last_chunk)   // no more data for this file. can close it now.
        {
            //dbg(" [iot_%d]: (%s) set has_last_chunk=1\n", myid,cfile->filename);
            //strncpy(iofname, cfile->filename, 128);
            //iofname[128-1] = 0;
            //pthread_mutex_lock( &cfile->mutex );
            cfile->has_last_chunk = 1;
            //pthread_mutex_unlock( &cfile->mutex );
            /// after setting "can_release", an iothr 
            /// may have cleaned the hash-tbl to remove this file
            hash_bucket_t *bkt = htbl_find_lock_bucket(g_ht_cfile, iofname, strlen(iofname));
            if (bkt) {
                dbg(" [iot_%d]: will call: hash_table_put_record()...\n", myid);
                ret = hash_table_put_record(g_ht_cfile, cfile, 1);
                htbl_unlock_bucket(bkt);
                //dbg(" [iot_%d]: release file %s ret %d...\n", myid, cfile->filename, ret);
            } else {
                dbg(" other thr has del cfile-record for (%s)\n", iofname);
            }
        }
        ///} // end of else( In pure WA mode )
    }

  err_out_1:
    sem_destroy(&sem);
    printf("ibcli-iothr %d exit...\n", myid);
    pthread_exit(NULL);
}

#endif
