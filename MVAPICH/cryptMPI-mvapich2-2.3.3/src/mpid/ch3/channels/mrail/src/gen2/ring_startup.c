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

#include "mpichconf.h"

#include <mpimem.h>
#include <netdb.h>
#include <string.h>

#include "rdma_impl.h"
#include "upmi.h"
#include "cm.h"
#include "ibv_param.h"

#define MPD_WINDOW 10

#define IBA_PMI_ATTRLEN (16)
#define IBA_PMI_VALLEN  (4096)

struct init_addr_inf {
    int    lid;
    int    qp_num[2];
    union  ibv_gid gid;
};

struct host_addr_inf {
    uint32_t    sr_qp_num;
    uint64_t    vc_addr;
};

struct addr_packet {
    int         rank;
    int         host_id;
    int         lid;
    int         rail;
    mv2_arch_hca_type    arch_hca_type;
    union  ibv_gid gid;
    struct host_addr_inf val[0];
};

static inline int addr_packet_size(int pg_size)
{
    return (sizeof(struct addr_packet) + 
            pg_size * sizeof(struct host_addr_inf));
}

static void *addr_packet_buffer(void *start, int index, int pg_size)
{
    return (void *)((uintptr_t)start + index * addr_packet_size(pg_size));
}

struct ring_packet {
    int     type;
    int     value;
};

static union ibv_gid get_local_gid(struct ibv_context * ctx, int port)
{
    union ibv_gid gid;

    ibv_query_gid(ctx, port, rdma_default_gid_index, &gid);

    return gid;
}

static uint16_t get_local_lid(struct ibv_context * ctx, int port)
{
    struct ibv_port_attr attr;

    if (ibv_query_port(ctx, port, &attr)) {
        return -1;
    }

    mv2_MPIDI_CH3I_RDMA_Process.lmc = attr.lmc;

    return attr.lid;
}

static inline int round_left(int current, int size)
{
    return current == 0 ? size - 1 : current - 1;
}

static inline int is_A_on_left_of_B(int a, int b, int rank, int size)
{
    int dist_a = (rank - a + size) % size;
    int dist_b = (rank - b + size) % size;
    return dist_a > dist_b;
}

/* Exchange address info with other processes in the job.
 * MPD provides the ability for processes within the job to
 * publish information which can then be querried by other
 * processes.  It also provides a simple barrier sync.
 */
static int _rdma_pmi_exchange_addresses(int pg_rank, int pg_size,
                                       void *localaddr, int addrlen, 
                                       void *alladdrs)
{
    int     ret, i, j, lhs, rhs, len_local, len_remote;
    char    attr_buff[IBA_PMI_ATTRLEN];
    char    val_buff[IBA_PMI_VALLEN];
    char    *temp_localaddr = (char *) localaddr;
    char    *temp_alladdrs = (char *) alladdrs;
    char    *kvsname = NULL;

    len_local = strlen(temp_localaddr);
    /* TODO: Double check the value of value */
    CHECK_UNEXP((len_local > mv2_pmi_max_vallen), "local address length is larger then string length");

    /* Be sure to use different keys for different processes */
    MPIU_Memset(attr_buff, 0, IBA_PMI_ATTRLEN * sizeof(char));
    snprintf(attr_buff, IBA_PMI_ATTRLEN, "MVAPICH2-%04d", pg_rank);

    /* put the kvs into PMI */
    MPIU_Strncpy(mv2_pmi_key, attr_buff, mv2_pmi_max_keylen);
    MPIU_Strncpy(mv2_pmi_val, temp_localaddr, mv2_pmi_max_vallen);
    MPIDI_PG_GetConnKVSname( &kvsname );
    ret = UPMI_KVS_PUT(kvsname, mv2_pmi_key, mv2_pmi_val);

    CHECK_UNEXP((ret != 0), "UPMI_KVS_PUT error \n");

    ret = UPMI_KVS_COMMIT(kvsname);
    CHECK_UNEXP((ret != 0), "UPMI_KVS_COMMIT error \n");

    /* Wait until all processes done the same */
    ret = UPMI_BARRIER();
    CHECK_UNEXP((ret != 0), "UPMI_BARRIER error \n");
    lhs = (pg_rank + pg_size - 1) % pg_size;
    rhs = (pg_rank + 1) % pg_size;

    for (i = 0; i < 2; i++) {
        /* get lhs and rhs processes' data */
        j = (i == 0) ? lhs : rhs;
        /* Use the key to extract the value */
        MPIU_Memset(attr_buff, 0, IBA_PMI_ATTRLEN * sizeof(char));
        MPIU_Memset(val_buff, 0, IBA_PMI_VALLEN * sizeof(char));
        snprintf(attr_buff, IBA_PMI_ATTRLEN, "MVAPICH2-%04d", j);
        MPIU_Strncpy(mv2_pmi_key, attr_buff, mv2_pmi_max_keylen);

        ret = UPMI_KVS_GET(kvsname, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
        CHECK_UNEXP((ret != 0), "UPMI_KVS_GET error \n");
        MPIU_Strncpy(val_buff, mv2_pmi_val, mv2_pmi_max_vallen);

        /* Simple sanity check before stashing it to the alladdrs */
        len_remote = strlen(val_buff);
        CHECK_UNEXP((len_remote < len_local), "remote length is smaller than local length");
        strncpy(temp_alladdrs, val_buff, len_local);
        temp_alladdrs += len_local;
    }

    /* this barrier is to prevent some process from overwriting values that
       has not been get yet */
    ret = UPMI_BARRIER();
    CHECK_UNEXP((ret != 0), "UPMI_BARRIER error \n");
    return 0;
}


static struct ibv_qp *create_qp(struct ibv_pd *pd, 
                                struct ibv_cq *scq, struct ibv_cq *rcq)
{
    struct ibv_qp_init_attr boot_attr;

    MPIU_Memset(&boot_attr, 0, sizeof boot_attr);
    boot_attr.cap.max_send_wr   = 128;
    boot_attr.cap.max_recv_wr   = 128;
    boot_attr.cap.max_send_sge  = rdma_default_max_sg_list;
    boot_attr.cap.max_recv_sge  = rdma_default_max_sg_list;
    boot_attr.cap.max_inline_data = rdma_max_inline_size;
    boot_attr.qp_type = IBV_QPT_RC;
    boot_attr.sq_sig_all = 0;

    boot_attr.send_cq = scq;
    boot_attr.recv_cq = rcq;

    return ibv_create_qp(pd, &boot_attr);
}

static int _find_active_port(struct ibv_context *context) 
{
    struct ibv_port_attr port_attr;
    int j;

    for (j = 1; j <= RDMA_DEFAULT_MAX_PORTS; ++ j) {
        if ((! ibv_query_port(context, j, &port_attr)) &&
             port_attr.state == IBV_PORT_ACTIVE) {
            return j;
        }
    }

    return -1;
}

static int _setup_ib_boot_ring(struct init_addr_inf * neighbor_addr,
                              struct mv2_MPIDI_CH3I_RDMA_Process_t *proc,
                              int port)
{
    struct ibv_qp_attr      qp_attr;
    uint32_t    qp_attr_mask = 0;
    int         i;
    int         ret;
    qp_attr.qp_state        = IBV_QPS_INIT;
    set_pkey_index(&qp_attr.pkey_index, 0, port);
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    qp_attr.port_num        = port;

    DEBUG_PRINT("default port %d, qpn %x\n", port,
            proc->boot_qp_hndl[0]->qp_num);

    ret = ibv_modify_qp(proc->boot_qp_hndl[0],&qp_attr,(IBV_QP_STATE
                        | IBV_QP_PKEY_INDEX
                        | IBV_QP_PORT
                        | IBV_QP_ACCESS_FLAGS));
    CHECK_RETURN(ret, "Could not modify boot qp to INIT");

    ret = ibv_modify_qp(proc->boot_qp_hndl[1],&qp_attr,(IBV_QP_STATE
                        | IBV_QP_PKEY_INDEX
                        | IBV_QP_PORT
                        | IBV_QP_ACCESS_FLAGS));
    CHECK_RETURN(ret, "Could not modify boot qp to INIT");

    /**********************  INIT --> RTR  ************************/
    MPIU_Memset(&qp_attr, 0, sizeof qp_attr);
    qp_attr.qp_state    =   IBV_QPS_RTR;
    qp_attr.rq_psn      =   rdma_default_psn;
    qp_attr.max_dest_rd_atomic  =   rdma_default_max_rdma_dst_ops;
    qp_attr.min_rnr_timer       =   rdma_default_min_rnr_timer;
    qp_attr.ah_attr.sl          =   rdma_default_service_level;
    qp_attr.ah_attr.static_rate =   rdma_default_static_rate;
    qp_attr.ah_attr.src_path_bits   =   rdma_default_src_path_bits;
    qp_attr.ah_attr.port_num    =   port;

    if (use_iboeth) {
        qp_attr.ah_attr.grh.dgid.global.subnet_prefix = 0;
        qp_attr.ah_attr.grh.dgid.global.interface_id = 0;
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.sgid_index = rdma_default_gid_index;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.ah_attr.grh.traffic_class = 0;
        qp_attr.ah_attr.is_global      = 1;
        qp_attr.ah_attr.dlid           = 0;
        qp_attr.path_mtu            = IBV_MTU_1024;
    } else {
        qp_attr.ah_attr.is_global   =   0;
        qp_attr.path_mtu    =   IBV_MTU_1024;
    }

    qp_attr_mask        |=  IBV_QP_STATE;
    qp_attr_mask        |=  IBV_QP_PATH_MTU;
    qp_attr_mask        |=  IBV_QP_RQ_PSN;
    qp_attr_mask        |=  IBV_QP_MAX_DEST_RD_ATOMIC;
    qp_attr_mask        |=  IBV_QP_MIN_RNR_TIMER;
    qp_attr_mask        |=  IBV_QP_AV;

    /* lhs */
    for (i = 0; i < 2; i++) {
        qp_attr.dest_qp_num     = neighbor_addr[i].qp_num[1 - i];
        if (use_iboeth) {
           qp_attr.ah_attr.grh.dgid = neighbor_addr[i].gid;
        } else {
           qp_attr.ah_attr.dlid    = neighbor_addr[i].lid;
        }
        qp_attr_mask            |=  IBV_QP_DEST_QPN;

        /* Path SL Lookup */
        if (!use_iboeth && (rdma_3dtorus_support || rdma_path_sl_query)) {
            struct ibv_context *context = proc->boot_context;
             struct ibv_pd *pd  = proc->boot_ptag;
            /* don't know our local LID yet, so set it to 0 to let
               mv2_get_path_rec_sl do the lookup */
            /*uint16_t lid = proc->lids[0][port];*/
            uint16_t lid = 0x0;
            uint16_t rem_lid   = qp_attr.ah_attr.dlid;
            uint32_t port_num  = qp_attr.ah_attr.port_num;
            qp_attr.ah_attr.sl = mv2_get_path_rec_sl(context, pd, port_num, lid,
                                                 rem_lid, rdma_3dtorus_support,
                                                 rdma_num_sa_query_retries);
        }

        ret = ibv_modify_qp(proc->boot_qp_hndl[i],&qp_attr, qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTR");

        DEBUG_PRINT("local QP=%x\n", proc->boot_qp_hndl[i]->qp_num);
    }

    /************** RTS *******************/
    MPIU_Memset(&qp_attr, 0, sizeof qp_attr);
    qp_attr.qp_state        = IBV_QPS_RTS;
    qp_attr.sq_psn          = rdma_default_psn;
    qp_attr.timeout         = rdma_default_time_out;
    qp_attr.retry_cnt       = rdma_default_retry_count;
    qp_attr.rnr_retry       = rdma_default_rnr_retry;
    qp_attr.max_rd_atomic   = rdma_default_qp_ous_rd_atom;

    qp_attr_mask = 0;
    qp_attr_mask =    IBV_QP_STATE              |
                      IBV_QP_TIMEOUT            |
                      IBV_QP_RETRY_CNT          |
                      IBV_QP_RNR_RETRY          |
                      IBV_QP_SQ_PSN             |
                      IBV_QP_MAX_QP_RD_ATOMIC;

    ret = ibv_modify_qp(proc->boot_qp_hndl[0],&qp_attr,qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTS");
    ret = ibv_modify_qp(proc->boot_qp_hndl[1],&qp_attr,qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTS");

    DEBUG_PRINT("Modified to RTS..Qp\n");
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME rdma_exchange_host_id
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_ring_exchange_host_id(MPIDI_PG_t * pg, int pg_rank, int pg_size)
{
    int mpi_errno = MPI_SUCCESS;
    int *hostid_all;

    hostid_all =  (int *) MPIU_Malloc(pg_size * sizeof(int));

    int my_hostid =  gethostid();
    hostid_all[pg_rank] = my_hostid;
    mpi_errno = rdma_ring_based_allgather(&my_hostid, sizeof my_hostid,
                                      pg_rank, hostid_all, pg_size, &mv2_MPIDI_CH3I_RDMA_Process);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    rdma_process_hostid(pg, hostid_all, pg_rank, pg_size );

fn_exit:
    MPIU_Free(hostid_all);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME ring_rdma_get_hca_context
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int ring_rdma_get_hca_context(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    proc->boot_ptag     = proc->ptag[0];
    proc->boot_device   = proc->ib_dev[0];
    proc->boot_context  = proc->nic_context[0];

    return 1;
}

#undef FUNCNAME
#define FUNCNAME rdma_setup_startup_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_setup_startup_ring(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc, int pg_rank,
                        int pg_size)
{
    struct init_addr_inf neighbor_addr[2];
    char ring_qp_out[128];
    char ring_qp_in[256];
    int bootstrap_len;
    union ibv_gid gid;
    int mpi_errno = MPI_SUCCESS;
    int port;
    char *value = NULL;

    if (!ring_rdma_get_hca_context(proc)) {
        MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "**fail %s", "cannot retrieve hca device");
    }
        
    if ((value = getenv("MV2_DEFAULT_PORT")) != NULL) {
        rdma_default_port = atoi(value);
    }
    if ((value = getenv("MV2_DEFAULT_GID_INDEX")) != NULL) {
        rdma_default_gid_index = atoi(value);
    }

    if (rdma_default_port < 0 || rdma_num_ports > 1) {
        /* Find active port if user has not asked us to use one */
        port = _find_active_port(proc->boot_context);
        if (port < 0) {
            MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto out, "**fail",
                    "**fail %s", "could not find active port");
        }
    } else {
        /* Use port specified by user */
        port = rdma_default_port;
    }

    proc->boot_cq_hndl = ibv_create_cq(proc->boot_context,
                                       rdma_default_max_cq_size,
                                       NULL, NULL, 0);
    if (!proc->boot_cq_hndl) {
        MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "**fail %s", "cannot create cq");
    }

    proc->boot_qp_hndl[0] = create_qp(proc->boot_ptag, proc->boot_cq_hndl,
                                      proc->boot_cq_hndl);
    if (!proc->boot_qp_hndl[0]) {
        MPIR_ERR_SETFATALANDSTMT2(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "%s%d", "Fail to create qp on rank ", pg_rank);
    }

    proc->boot_qp_hndl[1] = create_qp(proc->boot_ptag, proc->boot_cq_hndl,
                                      proc->boot_cq_hndl);
    if (!proc->boot_qp_hndl[1]) {
        MPIR_ERR_SETFATALANDSTMT2(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "%s%d", "Fail to create qp on rank ", pg_rank);
    }

    if (use_iboeth) {
        gid = get_local_gid(proc->boot_context, port);
        sprintf(ring_qp_out, "%016"SCNx64":%016"SCNx64":%08x:%08x:",
                 gid.global.subnet_prefix, gid.global.interface_id,
                 proc->boot_qp_hndl[0]->qp_num,
                 proc->boot_qp_hndl[1]->qp_num
               );
    
        DEBUG_PRINT("After setting GID: %"PRIx64":%"PRIx64", qp0: %x, qp1: %x\n",
                gid.global.subnet_prefix, gid.global.interface_id,
                proc->boot_qp_hndl[0]->qp_num,
                proc->boot_qp_hndl[1]->qp_num
                );
    } else {
        sprintf(ring_qp_out, "%08x:%08x:%08x:",
                 get_local_lid(proc->boot_context,
                               port),
                 proc->boot_qp_hndl[0]->qp_num,
                 proc->boot_qp_hndl[1]->qp_num
               );
    
        DEBUG_PRINT("After setting LID: %d, qp0: %x, qp1: %x\n",
                get_local_lid(proc->boot_context,
                              port),
                proc->boot_qp_hndl[0]->qp_num,
                proc->boot_qp_hndl[1]->qp_num
                );
    }

    bootstrap_len = strlen(ring_qp_out);
    _rdma_pmi_exchange_addresses(pg_rank, pg_size, ring_qp_out,
            bootstrap_len, ring_qp_in);

    if (use_iboeth) {
        sscanf(&ring_qp_in[0], "%016"SCNx64":%016"SCNx64":%08x:%08x:", 
               &neighbor_addr[0].gid.global.subnet_prefix,
               &neighbor_addr[0].gid.global.interface_id,
               &neighbor_addr[0].qp_num[0],
               &neighbor_addr[0].qp_num[1]);
        sscanf(&ring_qp_in[53], "%016"SCNx64":%016"SCNx64":%08x:%08x:",
               &neighbor_addr[1].gid.global.subnet_prefix,
               &neighbor_addr[1].gid.global.interface_id,
               &neighbor_addr[1].qp_num[0],
               &neighbor_addr[1].qp_num[1]);
        DEBUG_PRINT("After retrieving GID: %"PRIx64":%"PRIx64", qp0: %x, qp1: %x\n",
               neighbor_addr[0].gid.global.subnet_prefix,
               neighbor_addr[0].gid.global.interface_id,
               neighbor_addr[0].qp_num[0],
               neighbor_addr[0].qp_num[1]);
        DEBUG_PRINT("After retrieving GID: %"PRIx64":%"PRIx64", qp0: %x, qp1: %x\n",
               neighbor_addr[1].gid.global.subnet_prefix,
               neighbor_addr[1].gid.global.interface_id,
               neighbor_addr[1].qp_num[0],
               neighbor_addr[1].qp_num[1]);
    } else {
        sscanf(&ring_qp_in[0], "%08x:%08x:%08x:", 
               &neighbor_addr[0].lid, 
               &neighbor_addr[0].qp_num[0],
               &neighbor_addr[0].qp_num[1]);
        sscanf(&ring_qp_in[27], "%08x:%08x:%08x:",
               &neighbor_addr[1].lid, 
               &neighbor_addr[1].qp_num[0],
               &neighbor_addr[1].qp_num[1]);
    }

    mpi_errno = _setup_ib_boot_ring(neighbor_addr, proc, port);
    UPMI_BARRIER();

out:
    return mpi_errno;
}


#undef FUNCNAME
#define FUNCNAME rdma_cleanup_startup_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cleanup_startup_ring(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    int mpi_errno = MPI_SUCCESS;

    UPMI_BARRIER();
    
    if(ibv_destroy_qp(proc->boot_qp_hndl[0])) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy lhs QP");
    }

    if(ibv_destroy_qp(proc->boot_qp_hndl[1])) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy rhs QP");
    }

    if(ibv_destroy_cq(proc->boot_cq_hndl)) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy CQ");
    }

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME rdma_ring_based_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_ring_based_allgather(void *sbuf, int data_size,
        int pg_rank, void *rbuf, int pg_size,
        struct mv2_MPIDI_CH3I_RDMA_Process_t *proc)
{
    int i; 
    struct ibv_mr *addr_hndl = NULL;
    int mpi_errno = MPI_SUCCESS;

    addr_hndl = ibv_reg_mr(proc->boot_ptag, rbuf, data_size*pg_size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    if (addr_hndl == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                MPI_ERR_INTERN,
                "**fail",
                "**fail %s",
                "ibv_reg_mr failed for addr_hndl\n");
    }

    DEBUG_PRINT("val of addr_pool is: %p, handle: %08x\n",
            rbuf, addr_hndl->handle);

    /* Now start exchanging data*/
    {
        int recv_post_index = round_left(pg_rank,pg_size);
        int send_post_index = pg_rank;
        int recv_comp_index = pg_rank;
        int send_comp_index = -1;
        int credit = MPD_WINDOW/2;

        /* work entries related variables */
        struct ibv_recv_wr rr;
        struct ibv_sge sg_entry_r;
        struct ibv_recv_wr *bad_wr_r;
        struct ibv_send_wr sr;
        struct ibv_sge sg_entry_s;
        struct ibv_send_wr *bad_wr_s;

        /* completion related variables */
        struct ibv_wc rc;

        char* rbufProxy = (char*) rbuf;

        /* copy self data*/
        MPIU_Memcpy(rbufProxy+data_size*pg_rank, sbuf, data_size);

        /* post receive*/
        for (i = 0; i < MPD_WINDOW; i++) {
            if (recv_post_index == pg_rank)
                continue;
            rr.wr_id   = recv_post_index;
            rr.num_sge = 1;
            rr.sg_list = &(sg_entry_r);
            rr.next    = NULL;
            sg_entry_r.lkey = addr_hndl->lkey;
            sg_entry_r.addr = (uintptr_t)(rbufProxy+data_size*recv_post_index);
            sg_entry_r.length = data_size;

            if (ibv_post_recv(proc->boot_qp_hndl[0], &rr, &bad_wr_r)) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                        MPI_ERR_INTERN,
                        "**fail",
                        "**fail %s",
                        "Error posting recv!\n");
            }
            recv_post_index = round_left(recv_post_index,pg_size);
        }

        UPMI_BARRIER();

        /* sending and receiving*/
        while ((recv_comp_index != (pg_rank+1)%pg_size) ||
               (send_comp_index != (pg_rank+2)%pg_size+pg_size)) {
            int ne;
            /* Three conditions
             * 1: not complete sending
             * 2: has received the data
             * 3: has enough credit
             */
            if ((send_post_index != (pg_rank+1)%pg_size) &&
                ((recv_comp_index == send_post_index) ||
                is_A_on_left_of_B(recv_comp_index,
                     send_post_index,pg_rank,pg_size)) && credit > 0) {

                sr.opcode         = IBV_WR_SEND;
                sr.send_flags     = IBV_SEND_SIGNALED;
                sr.wr_id          = send_post_index+pg_size;
                sr.num_sge        = 1;
                sr.sg_list        = &sg_entry_s;
                sr.next           = NULL;
                sg_entry_s.addr   = (uintptr_t)
                                    (rbufProxy+data_size*send_post_index);
                sg_entry_s.length = data_size;
                sg_entry_s.lkey   = addr_hndl->lkey;

                if (ibv_post_send(proc->boot_qp_hndl[1], &sr, &bad_wr_s)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                            MPI_ERR_INTERN,
                            "**fail",
                            "**fail %s",
                            "Error posting send!\n");
                }

                send_post_index=round_left(send_post_index,pg_size);
                credit--;
            }

            ne = ibv_poll_cq(proc->boot_cq_hndl, 1, &rc);
            if (ne < 0) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                        MPI_ERR_INTERN,
                        "**fail",
                        "**fail %s",
                        "Poll CQ failed!\n");
            } else if (ne > 1) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                        MPI_ERR_INTERN,
                        "**fail",
                        "**fail %s",
                        "Got more than one\n");
            } else if (ne == 1) {
                if (rc.status != IBV_WC_SUCCESS) {
                    if (rc.status == IBV_WC_RETRY_EXC_ERR) {
                        DEBUG_PRINT("Got IBV_WC_RETRY_EXC_ERR\n");
                    }
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                            MPI_ERR_INTERN,
                            "**fail",
                            "**fail %s",
                            "Error code in polled desc!\n");
                }
                if (rc.wr_id < pg_size) {
                    /*recv completion*/
                    recv_comp_index = round_left(recv_comp_index,pg_size);
                    MPIU_Assert(recv_comp_index == rc.wr_id);
                    if (recv_post_index != pg_rank) {
                        rr.wr_id   = recv_post_index;
                        rr.num_sge = 1;
                        rr.sg_list = &(sg_entry_r);
                        rr.next    = NULL;
                        sg_entry_r.lkey = addr_hndl->lkey;
                        sg_entry_r.addr = (uintptr_t)
                                          (rbufProxy+data_size*recv_post_index);
                        sg_entry_r.length = data_size;

                        if(ibv_post_recv(proc->boot_qp_hndl[0], &rr, &bad_wr_r)) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                                    MPI_ERR_INTERN,
                                    "**fail",
                                    "**fail %s",
                                    "Error posting recv!\n");
                        }

                        recv_post_index = round_left(recv_post_index,pg_size);
                    }
                } else {
                    /*send completion*/
                    credit++;
                    send_comp_index = rc.wr_id;
                }
            }
        }
        /*Now all send and recv finished*/
    }

    ibv_dereg_mr(addr_hndl);
fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME _ring_boot_exchange
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int _ring_boot_exchange(struct ibv_mr * addr_hndl, void * addr_pool,
        struct mv2_MPIDI_CH3I_RDMA_Process_t *proc, MPIDI_PG_t *pg, int pg_rank,
        struct process_init_info *info)
{
    int i, ne, index_to_send, rail_index, pg_size;
    int hostid;
    struct hostent *hostent;
    char hostname[HOSTNAME_LEN + 1];
    int mpi_errno = MPI_SUCCESS;
    uint64_t last_send = 0;
    uint64_t last_send_comp = 0;

    struct addr_packet * send_packet;
    struct addr_packet * recv_packet;

    /* work entries related variables */
    struct ibv_recv_wr rr;
    struct ibv_sge sg_entry_r;
    struct ibv_recv_wr *bad_wr_r;
    struct ibv_send_wr sr;
    struct ibv_sge sg_entry_s;
    struct ibv_send_wr *bad_wr_s;

    /* completion related variables */
    struct ibv_wc rc;
    MPIDI_VC_t * vc;

    int result;

    /* Post the window of recvs: The first entry
     * is not posted since it is used for the
     * initial send
     */

    DEBUG_PRINT("Posting recvs\n");
    char* addr_poolProxy = (char*) addr_pool;
    pg_size = MPIDI_PG_Get_size(pg);
    
    for(i = 1; i < MPD_WINDOW; i++) {
        rr.wr_id   = i;
        rr.num_sge = 1;
        rr.sg_list = &(sg_entry_r);
        rr.next    = NULL;
        sg_entry_r.lkey = addr_hndl->lkey;
        sg_entry_r.addr = 
                (uintptr_t)addr_packet_buffer(addr_poolProxy, i, pg_size);
        sg_entry_r.length = addr_packet_size(pg_size);

        if(ibv_post_recv(proc->boot_qp_hndl[0], &rr, &bad_wr_r)) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                    MPI_ERR_INTERN,
                    "**fail",
                    "**fail %s",
                    "Error posting recv!\n");
        }
    }

    DEBUG_PRINT("done posting recvs\n");

    index_to_send = 0;

    /* get hostname stuff */

    result = gethostname(hostname, HOSTNAME_LEN);
    if (result!=0) {
        PRINT_ERROR_ERRNO("Could not get hostname.", errno);
        exit(EXIT_FAILURE);
    }
    hostent = gethostbyname(hostname);
    if (hostent == NULL) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER,
                "**gethostbyname", "**gethostbyname %s %d", 
                hstrerror(h_errno), h_errno );

    }
    hostid = (int) ((struct in_addr *) hostent->h_addr_list[0])->s_addr;

    /* send information for each rail */

    DEBUG_PRINT("rails: %d\n", rdma_num_rails);

    UPMI_BARRIER();

    for(rail_index = 0; rail_index < rdma_num_rails; rail_index++) {

        DEBUG_PRINT("doing rail %d\n", rail_index);

        send_packet          = addr_packet_buffer(addr_poolProxy, index_to_send,
                                                  pg_size);
        send_packet->rank    = pg_rank;
        send_packet->rail    = rail_index;
        send_packet->host_id = hostid;

        for(i = 0; i < pg_size; i++) {
            MPIDI_PG_Get_vc(pg, i, &vc); 
            if (!qp_required(vc, pg_rank, i)) {
                send_packet->val[i].sr_qp_num = -1;
                info->arch_hca_type[i] = mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
            } else {

                send_packet->lid     = vc->mrail.rails[rail_index].lid;
                send_packet->gid     = vc->mrail.rails[rail_index].gid;
                send_packet->val[i].sr_qp_num =
                    vc->mrail.rails[rail_index].qp_hndl->qp_num;
                send_packet->val[i].vc_addr  = (uintptr_t)vc;
                send_packet->arch_hca_type = mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
            }
        }

        DEBUG_PRINT("starting to do sends\n");
        for(i = 0; i < pg_size - 1; i++) {

            sr.opcode         = IBV_WR_SEND;
            sr.send_flags     = IBV_SEND_SIGNALED;
            sr.wr_id          = MPD_WINDOW + index_to_send;
            sr.num_sge        = 1;
            sr.sg_list        = &sg_entry_s;
            sr.next           = NULL;
            sg_entry_s.addr   = (uintptr_t)addr_packet_buffer(addr_poolProxy, 
                                                              index_to_send, 
                                                              pg_size);
            sg_entry_s.length = addr_packet_size(pg_size);
            sg_entry_s.lkey   = addr_hndl->lkey;

            /* keep track of the last send... */
            last_send         = sr.wr_id;

            if (ibv_post_send(proc->boot_qp_hndl[1], &sr, &bad_wr_s)) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                        MPI_ERR_INTERN,
                        "**fail",
                        "**fail %s",
                        "Error posting send!\n");
            }

            /* flag that keeps track if we are waiting
             * for a recv or more credits
             */

            while(1) {
                ne = ibv_poll_cq(proc->boot_cq_hndl, 1, &rc);
                if (ne < 0) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                            MPI_ERR_INTERN,
                            "**fail",
                            "**fail %s",
                            "Poll CQ failed!\n");
                } else if (ne > 1) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                            MPI_ERR_INTERN,
                            "**fail",
                            "**fail %s",
                            "Got more than one\n");
                } else if (ne == 1) {
                    if (rc.status != IBV_WC_SUCCESS) {
                        if(rc.status == IBV_WC_RETRY_EXC_ERR) {
                            DEBUG_PRINT("Got IBV_WC_RETRY_EXC_ERR\n");
                        }

                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                                MPI_ERR_INTERN,
                                "**fail",
                                "**fail %s",
                                "Error code in polled desc!\n");
                    }

                    if (rc.wr_id < MPD_WINDOW) {
                        /* completion of recv */

                        recv_packet = addr_packet_buffer(addr_poolProxy,
                                                         rc.wr_id, pg_size);

                        info->lid[recv_packet->rank][rail_index] =
                            recv_packet->lid;
                        info->gid[recv_packet->rank][rail_index] =
                            recv_packet->gid;
                        info->hostid[recv_packet->rank][rail_index] =
                            recv_packet->host_id;
                        info->arch_hca_type[recv_packet->rank] =
                            recv_packet->arch_hca_type;
                        info->vc_addr[recv_packet->rank] =
                                recv_packet->val[pg_rank].vc_addr;

                        MPIDI_PG_Get_vc(pg, recv_packet->rank, &vc);
                        vc->smp.hostid = recv_packet->host_id;

                        info->qp_num_rdma[recv_packet->rank][rail_index] =
                            recv_packet->val[pg_rank].sr_qp_num;

                        /* queue this for sending to the next
                         * hop in the ring
                         */
                        index_to_send = rc.wr_id;

                        break;
                    } else {
                        /* completion of send */
                        last_send_comp = rc.wr_id;

                        /* now post as recv */
                        rr.wr_id   = rc.wr_id - MPD_WINDOW;
                        rr.num_sge = 1;
                        rr.sg_list = &(sg_entry_r);
                        rr.next    = NULL;
                        sg_entry_r.lkey = addr_hndl->lkey;
                        sg_entry_r.addr = (uintptr_t)
                            addr_packet_buffer(addr_poolProxy, rr.wr_id, pg_size);
                        sg_entry_r.length = addr_packet_size(pg_size);
                        if(ibv_post_recv(proc->boot_qp_hndl[0], &rr, &bad_wr_r)) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                                    MPI_ERR_INTERN,
                                    "**fail",
                                    "**fail %s",
                                    "Error posting recv!\n");
                        }
                    }
                }
            }

        }
    } /* end for(rail_index... */

    /* Make sure all sends have completed */

    while(last_send_comp != last_send) {
        ne = ibv_poll_cq(proc->boot_cq_hndl, 1, &rc);
        if(ne == 1) {
            if (rc.status != IBV_WC_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP2(mpi_errno,
                        MPI_ERR_INTERN,
                        "**fail",
                        "**fail %s %d",
                        "Error code %d in polled desc!\n",
                        rc.status);
            }
            last_send_comp = rc.wr_id;
        }
    }

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;

}

#undef FUNCNAME
#define FUNCNAME rdma_ring_boot_exchange
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_ring_boot_exchange(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc,
                      MPIDI_PG_t *pg, int pg_rank, struct process_init_info *info)
{
    struct ibv_mr * addr_hndl;
    void * addr_pool;
    int pg_size = MPIDI_PG_Get_size(pg);
    int mpi_errno = MPI_SUCCESS;
    addr_pool = MPIU_Malloc(MPD_WINDOW * addr_packet_size(pg_size));
    addr_hndl = ibv_reg_mr(proc->boot_ptag,
            addr_pool, MPD_WINDOW * addr_packet_size(pg_size),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    if(addr_hndl == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                MPI_ERR_INTERN,
                "**fail",
                "**fail %s",
                "ibv_reg_mr failed for addr_hndl\n");
    }

    mpi_errno = _ring_boot_exchange(addr_hndl, addr_pool, proc, pg, pg_rank, info);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    ibv_dereg_mr(addr_hndl);
    MPIU_Free(addr_pool);

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;

}


