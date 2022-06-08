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

#include <infiniband/verbs.h>
#include <netdb.h>

#include "mpidimpl.h"
#include "upmi.h"

#include "ib_param.h"
#include "ib_hca.h"
#include "ib_cm.h"
#include "ib_errors.h"
#include "ib_process.h"
#include "ib_send.h"
#include "rdma_3dtorus.h"

/****************for ring_startup*****************************/
#define CHECK_UNEXP(ret, s)                           \
do {                                                  \
    if (ret) {                                        \
        fprintf(stderr, "[%s:%d]: %s\n",              \
                __FILE__,__LINE__, s);                \
    exit(1);                                          \
    }                                                 \
} while (0)

#define CHECK_RETURN(ret, s)                            \
do {                                                    \
    if (ret) {                                          \
    fprintf(stderr, "[%s:%d] error(%d): %s\n",          \
        __FILE__,__LINE__, ret, s);                     \
    exit(1);                                            \
    }                                                   \
}                                                       \
while (0)


#define MPD_WINDOW 10

#define IBA_PMI_ATTRLEN (16)
#define IBA_PMI_VALLEN  (4096)

int mv2_pmi_max_keylen=0;
int mv2_pmi_max_vallen=0;
char *mv2_pmi_key=NULL;
char *mv2_pmi_val=NULL;

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
/*****************above for ring_startup***********************/

MPID_nem_ib_conn_info_t conn_info;

static int get_pkey_index(uint16_t pkey, int hca_num, int port_num, uint16_t* index)
{
    uint16_t i = 0;
    struct ibv_device_attr dev_attr;

    if(ibv_query_device( hca_list[hca_num].nic_context, &dev_attr)) {

        ibv_error_abort(GEN_EXIT_ERR,
                "Error getting HCA attributes\n");
    }

    for (i=0; i < dev_attr.max_pkeys ; ++i) {
        uint16_t curr_pkey;
        ibv_query_pkey(hca_list[hca_num].nic_context,
                (uint8_t)port_num, (int)i ,&curr_pkey);
        if (pkey == (ntohs(curr_pkey) & PKEY_MASK)) {
            *index = i;
            return 1;
        }
    }

    return 0;
}


static void set_pkey_index(uint16_t * pkey_index, int hca_num, int port_num)
{
    if (rdma_default_pkey == RDMA_DEFAULT_PKEY)
    {
        *pkey_index = rdma_default_pkey_ix;
    }
    else if (!get_pkey_index(
        rdma_default_pkey,
        hca_num,
        port_num,
        pkey_index)
    )
    {
        ibv_error_abort(
            GEN_EXIT_ERR,
            "Can't find PKEY INDEX according to given PKEY\n"
        );
    }
}


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_init_connection
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * Init connections
 *
 * @param size the number of processes in the group.
 * @param rank rank of current process.
 */
int MPID_nem_ib_init_connection(int rank, int size)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_INIT_CONN_INFO);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_INIT_CONN_INFO);

    memset(&conn_info, 0, sizeof(MPID_nem_ib_conn_info_t));

    conn_info.size = size;
    conn_info.rank = rank;
    conn_info.connections = (MPID_nem_ib_connection_t *)MPIU_Malloc(size * sizeof(MPID_nem_ib_connection_t));
    memset(conn_info.connections, 0, size * sizeof(MPID_nem_ib_connection_t));

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_INIT_CONN_INFO);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME rdma_cleanup_startup_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cleanup_startup_ring()
{           
    int mpi_errno = MPI_SUCCESS;

    UPMI_BARRIER(); 
        
    if(ibv_destroy_qp(process_info.boot_qp_hndl[0])) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy lhs QP");
    }   
    
    if(ibv_destroy_qp(process_info.boot_qp_hndl[1])) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy rhs QP");
    }   
    
    if(ibv_destroy_cq(process_info.boot_cq_hndl)) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "**fail %s", "could not destroy CQ");
    }

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

/**
 * free conn_info
 */
int MPID_nem_ib_free_conn_info(int size) {
    int mpi_errno = MPI_SUCCESS;
    int i;

    for(i=0;i<size;i++){
        MPIU_Free(conn_info.init_info->qp_num_rdma[i]);
        MPIU_Free(conn_info.init_info->lid[i]);
        MPIU_Free(conn_info.init_info->gid[i]);
        MPIU_Free(conn_info.init_info->hostid[i]);
    }
    MPIU_Free(conn_info.init_info->lid);
    MPIU_Free(conn_info.init_info->gid);
    MPIU_Free(conn_info.init_info->hostid);
    MPIU_Free(conn_info.init_info->qp_num_rdma);
    MPIU_Free(conn_info.init_info->arch_hca_type);
    MPIU_Free(conn_info.init_info->vc_addr);

    MPIU_Free( conn_info.init_info );
    conn_info.init_info = NULL;

    if(size > 1) {
        if ((mpi_errno = rdma_cleanup_startup_ring())
            != MPI_SUCCESS) {
                MPIR_ERR_POP(mpi_errno);
        }
    }

fn_fail:
    return mpi_errno;
}

/**
 * allocate memory for process_init_info
 *
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_alloc_process_init_info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_alloc_process_init_info()
{
    int mpi_errno = MPI_SUCCESS;

    struct process_init_info *info;
    int i;
    int size = conn_info.size;
    int rails = rdma_num_rails;

    info = MPIU_Malloc(sizeof *info);
    if (!info) {
        return 1;
    }

    info->lid = (uint16_t **) MPIU_Malloc(size * sizeof(uint16_t *));
    info->gid = (union ibv_gid **)
                    MPIU_Malloc(size * sizeof(union ibv_gid *));
    info->hostid = (int **) MPIU_Malloc(size * sizeof(int *));
    info->qp_num_rdma = (uint32_t **)
                            MPIU_Malloc(size * sizeof(uint32_t *));
    info->arch_hca_type = (mv2_arch_hca_type *) MPIU_Malloc(size * sizeof(mv2_arch_hca_type));
    info->vc_addr  = (uint64_t *) MPIU_Malloc(size * sizeof(uint64_t));

    if (!info->lid
        || !info->gid
        || !info->hostid
        || !info->qp_num_rdma
        || !info->arch_hca_type) {
        return 1;
    }

    for (i = 0; i < size; ++i) {
        info->qp_num_rdma[i] = (uint32_t *)
                                    MPIU_Malloc(rails * sizeof(uint32_t));
        info->lid[i] = (uint16_t *) MPIU_Malloc(rails * sizeof(uint16_t));
        info->gid[i] = (union ibv_gid *)
                         MPIU_Malloc(rails * sizeof(union ibv_gid));
        info->hostid[i] = (int *) MPIU_Malloc(rails * sizeof(int));
        if (!info->lid[i]
                || !info->gid[i]
                || !info->hostid[i]
                || !info->qp_num_rdma[i]) {
             return 1;
        }
    }

    conn_info.init_info = info;
    return mpi_errno;
}

/**
 * The second step from priv.c/MPID_nem_ib_setup_conn()
 * create qps for all connections
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_setup_conn
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_setup_conn(MPIDI_PG_t *pg)
{
    /* Error codes */
    int mpi_errno = MPI_SUCCESS;
    int size = conn_info.size;
    int curRank;
    int rail_index;

    MPIDI_VC_t  *vc;

    /* Infiniband Verb Structures */
    struct ibv_qp_init_attr attr;
    struct ibv_qp_attr      qp_attr;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_SETUP_CONN);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_SETUP_CONN);


    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;

    for (curRank = 0; curRank < size; curRank++) {
        if (curRank == conn_info.rank) {
            continue;
        }
        
        /* TODO: currently we also setup connection for intra-node, 
         * because we still don't know whether it's a local process 
         * or not. Ideally, we should avoid setup connection for 
         * local processes. Then in MPID_nem_ib_finalize fucntion,
         * we can avoid extra steps of ibv_destroy_qp and so on.
         */
        MPIDI_PG_Get_vc_set_active(pg, curRank, &vc);

        conn_info.connections[curRank].rails = MPIU_Malloc
            (sizeof *conn_info.connections->rails * rdma_num_rails);
        conn_info.connections[curRank].srp.credits = MPIU_Malloc(sizeof *(conn_info.connections->srp.credits) * rdma_num_rails);
        if (!conn_info.connections[curRank].rails
                || !conn_info.connections[curRank].srp.credits) {
            MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                "**fail", "**fail %s", "Failed to allocate resources for "
                "multirails");
        }

        /* set to the first arch_hca_type ? */
        if (conn_info.init_info)
            conn_info.init_info->arch_hca_type[curRank] = process_info.arch_hca_type;

        for ( rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            int hca_index, port_index;
            hca_index  = rail_index / (rdma_num_rails / ib_hca_num_hcas);
            port_index = (rail_index / (rdma_num_rails / (ib_hca_num_hcas * ib_hca_num_ports))) % ib_hca_num_ports;

            memset(&attr, 0, sizeof attr);
            attr.cap.max_send_wr = rdma_default_max_send_wqe;

            /* has_srq */
            if (process_info.has_srq) {
                attr.cap.max_recv_wr = 0;
                attr.srq = hca_list[hca_index].srq_hndl;
            } else {
                attr.cap.max_recv_wr = rdma_default_max_recv_wqe;
            }

            attr.cap.max_send_sge = rdma_default_max_sg_list;
            attr.cap.max_recv_sge = rdma_default_max_sg_list;
            attr.cap.max_inline_data = rdma_max_inline_size;
            attr.send_cq = hca_list[hca_index].cq_hndl;
            attr.recv_cq = hca_list[hca_index].cq_hndl;
            attr.qp_type = IBV_QPT_RC;
            attr.sq_sig_all = 0;

            conn_info.connections[curRank].rails[rail_index].qp_hndl = ibv_create_qp(hca_list[hca_index].ptag, &attr);

            if (!conn_info.connections[curRank].rails[rail_index].qp_hndl) {
                MPIR_ERR_SETFATALANDSTMT2(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                    "**fail", "%s%d", "Failed to create qp for rank ", curRank);
            }

            if (conn_info.init_info) {
                conn_info.init_info->lid[curRank][rail_index] = hca_list[hca_index].lids[port_index];
                conn_info.init_info->gid[curRank][rail_index] = hca_list[hca_index].gids[port_index];
                conn_info.init_info->qp_num_rdma[curRank][rail_index] =
                        conn_info.connections[curRank].rails[rail_index].qp_hndl->qp_num;
                conn_info.init_info->vc_addr[curRank] = (uintptr_t)vc;
            }

            qp_attr.qp_state        = IBV_QPS_INIT;
            qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
            qp_attr.port_num = hca_list[hca_index].ports[port_index];
            set_pkey_index(&qp_attr.pkey_index, hca_index, qp_attr.port_num);
            if (ibv_modify_qp(conn_info.connections[curRank].rails[rail_index].qp_hndl, &qp_attr,
                                    IBV_QP_STATE              |
                                    IBV_QP_PKEY_INDEX         |
                                    IBV_QP_PORT               |
                                    IBV_QP_ACCESS_FLAGS)) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                    "**fail", "**fail %s", "Failed to modify QP to INIT");
            }

        }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_SETUP_CONN);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

/**
 * get_local_lid
 * @param ctx local nic_contect
 * @param port port number
 */
static inline uint16_t get_local_lid(struct ibv_context *ctx, int port)
{
    struct ibv_port_attr attr;

    if (ibv_query_port(ctx, port, &attr)) {
        return -1;
    }

    return attr.lid;
}

/**
 * pmi_exchange
 * MPIDI_PG_t *pg.ch->kvs_name is used for pmi exchange
 * maybe we need to have this function outside the cm.c file
 * currently pass pg in
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_pmi_exchange
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_pmi_exchange()
{
    int mpi_errno = MPI_SUCCESS;
    int error;
    unsigned int ui;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_PMI_EXCHANGE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_PMI_EXCHANGE);

    char *kvsname = NULL;
    char rdmakey[512];
    char rdmavalue[512];
    char *buf = NULL;

    int i, rail_index;

    /* For now, here exchange the information of each LID separately */
    for (i = 0; i < conn_info.size; i++) {
        if (conn_info.rank == i) {
            continue;
        }

        /* Generate the key and value pair */
        MPL_snprintf(rdmakey, 512, "%08x-%08x", conn_info.rank, i);
        buf = rdmavalue;

        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            sprintf(buf, "%08x", conn_info.init_info->lid[i][rail_index]);
            buf += 8;
        }

        /* hca_list[0].arch_hca_type
         * what if there are different kinds of arch_hca_type
         * for multiple HCAs?
         */
        sprintf(buf, "%016lx", conn_info.init_info->arch_hca_type[i]);
        buf += 16;
        /* still about vc_addr here. vc should have not be initialized right now*/
        sprintf(buf, "%016llx", (long long unsigned int)conn_info.init_info->vc_addr[i]);
        buf += 16;

        /* put the kvs into PMI */
        MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
        MPIU_Strncpy(mv2_pmi_val, rdmavalue, mv2_pmi_max_vallen);
	    MPIDI_PG_GetConnKVSname( &kvsname );
        DEBUG_PRINT(stderr, "rdmavalue %s\n", pmi_val);

        /*
        This function puts the key/value pair in the specified keyval space.  The
        value is not visible to other processes until 'UPMI_KVS_COMMIT()' is called.
        The function may complete locally.  After 'UPMI_KVS_COMMIT()' is called, the
        value may be retrieved by calling 'UPMI_KVS_GET()'.  All keys put to a keyval
        space must be unique to the keyval space.  You may not put more than once
        with the same key.
        */
        error = UPMI_KVS_PUT(kvsname, mv2_pmi_key, mv2_pmi_val);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_put", "**pmi_kvs_put %d", error);
        }

        /*
        This function commits all previous puts since the last 'UPMI_KVS_COMMIT()' into
        the specified keyval space. It is a process local operation.
        */
        error = UPMI_KVS_COMMIT(kvsname);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_commit", "**pmi_kvs_commit %d", error);
        }
    }
    error = UPMI_BARRIER();
    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**pmi_barrier", "**pmi_barrier %d", error);
    }

    /* Here, all the key and value pairs are put, now we can get them */
    for (i = 0; i < conn_info.size; i++) {
        rail_index = 0;
        if (conn_info.rank == i) {
            conn_info.init_info->lid[i][0] =
                get_local_lid(hca_list[0].nic_context,
                                rdma_default_port);
            conn_info.init_info->arch_hca_type[i] = process_info.arch_hca_type;
            continue;
        }

        /* Generate the key */
        MPL_snprintf(rdmakey, 512, "%08x-%08x", i, conn_info.rank);
        MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);

        error = UPMI_KVS_GET(kvsname, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_get", "**pmi_kvs_get %d", error);
        }

        MPIU_Strncpy(rdmavalue, mv2_pmi_val, mv2_pmi_max_vallen);
        buf = rdmavalue;

        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            sscanf(buf, "%08x", &ui);
            conn_info.init_info->lid[i][rail_index] = ui;
            buf += 8;
        }

        sscanf(buf, "%016lx", &conn_info.init_info->arch_hca_type[i]);
        buf += 16;
        sscanf(buf, "%016llx", (long long unsigned int *) &conn_info.init_info->vc_addr[i]);
        buf += 16;
    }

    /* This barrier is to prevent some process from
     * overwriting values that has not been get yet
     */
    error = UPMI_BARRIER();
    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**pmi_barrier", "**pmi_barrier %d", error);
    }

    /* STEP 2: Exchange qp_num and vc addr */
    for (i = 0; i < conn_info.size; i++) {
        if (conn_info.rank == i) {
            continue;
        }

        /* Generate the key and value pair */
        MPL_snprintf(rdmakey, 512, "1-%08x-%08x", conn_info.rank, i);
        buf = rdmavalue;

        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            sprintf(buf, "%08X", conn_info.init_info->qp_num_rdma[i][rail_index]);
            buf += 8;
            DEBUG_PRINT("target %d, put qp %d, num %08X \n", i,
                    rail_index, conn_info.init_info->qp_num_rdma[i][rail_index]);
            DEBUG_PRINT("[%d] %s(%d) put qp %08X \n", conn_info.rank, __FUNCTION__,
                    __LINE__, conn_info.init_info->qp_num_rdma[i][rail_index]);
        }

        DEBUG_PRINT("put rdma value %s\n", rdmavalue);
        /* Put the kvs into PMI */
        MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
        MPIU_Strncpy(mv2_pmi_val, rdmavalue, mv2_pmi_max_vallen);
	    MPIDI_PG_GetConnKVSname( &kvsname );

        error = UPMI_KVS_PUT(kvsname, mv2_pmi_key, mv2_pmi_val);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_put", "**pmi_kvs_put %d", error);
        }

        error = UPMI_KVS_COMMIT(kvsname);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_commit", "**pmi_kvs_commit %d", error);
        }
    }

    error = UPMI_BARRIER();
    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**pmi_barrier", "**pmi_barrier %d", error);
    }

    /* Here, all the key and value pairs are put, now we can get them */
    for (i = 0; i < conn_info.size; i++) {
        if (conn_info.rank == i) {
            continue;
        }

        /* Generate the key */
        MPL_snprintf(rdmakey, 512, "1-%08x-%08x", i, conn_info.rank);
        MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
        error = UPMI_KVS_GET(kvsname, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
        if (error != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_kvs_get", "**pmi_kvs_get %d", error);
        }
        MPIU_Strncpy(rdmavalue, mv2_pmi_val, mv2_pmi_max_vallen);

        buf = rdmavalue;
        DEBUG_PRINT("get rdmavalue %s\n", rdmavalue);
        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            sscanf(buf, "%08X", &conn_info.init_info->qp_num_rdma[i][rail_index]);
            buf += 8;
            DEBUG_PRINT("[%d] %s(%d) get qp %08X from %d\n", conn_info.rank,
                    __FUNCTION__, __LINE__,
                    conn_info.init_info->qp_num_rdma[i][rail_index], i);
        }
    }

    error = UPMI_BARRIER();
    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**pmi_barrier", "**pmi_barrier %d", error);
    }

    DEBUG_PRINT("After barrier\n");

    /* if mv2_MPIDI_CH3I_RDMA_Process.has_one_sided
     * skip one-sided for now
     */

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_PMI_EXCHANGE);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

/* rdma_param_handle_heterogeneity resets control parameters given the arch_hca_type 
 * from all ranks. Parameters may change:
 *      rdma_default_mtu
 *      rdma_iba_eager_threshold
 *      proc->has_srq
 *      rdma_credit_preserve
 *      rdma_max_inline_size
 *      rdma_default_qp_ous_rd_atom
 *      rdma_put_fallback_threshold
 *      rdma_get_fallback_threshold
 *      num_rdma_buffer
 *      rdma_vbuf_total_size
 */
void rdma_param_handle_heterogeneity(mv2_arch_hca_type arch_hca_type[], int pg_size)
{       
    mv2_arch_hca_type type;                           
    process_info.heterogeneity = 0;
    int i;  

    type = arch_hca_type[0];
    for (i = 0; i < pg_size; ++ i) {

        if(MV2_IS_ARCH_HCA_TYPE(arch_hca_type[i], 
                    MV2_ARCH_ANY, MV2_HCA_QLGIC_PATH_HT) || 
                MV2_IS_ARCH_HCA_TYPE(arch_hca_type[i], 
                    MV2_ARCH_ANY, MV2_HCA_QLGIC_QIB)){
            process_info.has_srq = 0;
            process_info.post_send = MPIDI_nem_ib_post_send;
            rdma_credit_preserve = 3;
            rdma_default_qp_ous_rd_atom = 1;
        }

        else if(MV2_IS_ARCH_HCA_TYPE(arch_hca_type[i], 
                    MV2_ARCH_ANY, MV2_HCA_MLX_PCI_X)){
            process_info.has_srq = 0;
            process_info.post_send = MPIDI_nem_ib_post_send;
            rdma_credit_preserve = 3;
        }

        else if(MV2_IS_ARCH_HCA_TYPE(arch_hca_type[i], 
                    MV2_ARCH_ANY, MV2_HCA_IBM_EHCA)){
            process_info.has_srq = 0;
            process_info.post_send = MPIDI_nem_ib_post_send;
            rdma_credit_preserve = 3;
            rdma_max_inline_size = -1;
        }

        if (arch_hca_type[i] != type)
            process_info.heterogeneity = 1;

        DEBUG_PRINT("rank %d, arch_hca_type %d\n", i, arch_hca_type[i]);
    }

    if (process_info.heterogeneity) {
        rdma_default_mtu = IBV_MTU_1024;
        rdma_vbuf_total_size = 8 * 1024;
        rdma_fp_buffer_size = 8 * 1024;
        rdma_iba_eager_threshold = rdma_vbuf_total_size;
        rdma_max_inline_size = (rdma_max_inline_size == -1) ? -1 : 64;
        rdma_put_fallback_threshold = 4 * 1024;
        rdma_get_fallback_threshold = 192 * 1024;
        num_rdma_buffer = 16;
    }
}

#undef FUNCNAME
#define FUNCNAME _ring_boot_exchange
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int _ring_boot_exchange(struct ibv_mr * addr_hndl, void * addr_pool,
            MPIDI_PG_t *pg, int pg_rank)
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

        if(ibv_post_recv(process_info.boot_qp_hndl[0], &rr, &bad_wr_r)) {
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
        MPL_error_printf("Could not get hostname\n");
        exit(1);
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
            if(i == pg_rank) {
                send_packet->val[i].sr_qp_num = -1;
                conn_info.init_info->arch_hca_type[i] = process_info.arch_hca_type;
            } else {
                send_packet->lid     = conn_info.init_info->lid[i][rail_index];
                send_packet->gid     = conn_info.init_info->gid[i][rail_index];
                send_packet->val[i].sr_qp_num =
                    conn_info.init_info->qp_num_rdma[i][rail_index];
                send_packet->val[i].vc_addr  = (uintptr_t)conn_info.init_info->vc_addr[i];
                send_packet->arch_hca_type = conn_info.init_info->arch_hca_type[i];
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

            if (ibv_post_send(process_info.boot_qp_hndl[1], &sr, &bad_wr_s)) {
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
                ne = ibv_poll_cq(process_info.boot_cq_hndl, 1, &rc);
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

                        conn_info.init_info->lid[recv_packet->rank][rail_index] =
                            recv_packet->lid;
                        conn_info.init_info->gid[recv_packet->rank][rail_index] =
                            recv_packet->gid;
                        conn_info.init_info->hostid[recv_packet->rank][rail_index] =
                            recv_packet->host_id;
                        conn_info.init_info->arch_hca_type[recv_packet->rank] =
                            recv_packet->arch_hca_type;
                        conn_info.init_info->vc_addr[recv_packet->rank] =
                                recv_packet->val[pg_rank].vc_addr;
                        /* smp codes
                        MPIDI_PG_Get_vc(pg, recv_packet->rank, &vc);
                        vc->smp.hostid = recv_packet->host_id;
                        */

                        conn_info.init_info->qp_num_rdma[recv_packet->rank][rail_index] =
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
                        if(ibv_post_recv(process_info.boot_qp_hndl[0], &rr, &bad_wr_r)) {
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
        ne = ibv_poll_cq(process_info.boot_cq_hndl, 1, &rc);
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
int rdma_ring_boot_exchange(MPIDI_PG_t *pg, int pg_rank)
{
    struct ibv_mr * addr_hndl;
    void * addr_pool;
    int pg_size = MPIDI_PG_Get_size(pg);

    int mpi_errno = MPI_SUCCESS;
    addr_pool = MPIU_Malloc(MPD_WINDOW * addr_packet_size(pg_size));
    addr_hndl = ibv_reg_mr(hca_list[0].ptag,
            addr_pool, MPD_WINDOW * addr_packet_size(pg_size),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        
    if(addr_hndl == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                MPI_ERR_INTERN,
                "**fail",
                "**fail %s",
                "ibv_reg_mr failed for addr_hndl\n");
    }

    mpi_errno = _ring_boot_exchange(addr_hndl, addr_pool, pg, pg_rank);
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

/**
 * Exchange conn
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_exchange_conn
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_exchange_conn(MPIDI_PG_t *pg, int rank)
{
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_EXCHANGE_CONN_INFO);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_EXCHANGE_CONN_INFO);

    /* Based on type of connection requested by the user, do either PMI exchg,
     * or UD ring based exchg
     * For now, lets just do PMI conn
     */

    /* Get port attributes */
    if(conn_info.size > 1) {
        if (process_info.has_ring_startup) {
            mpi_errno = rdma_ring_boot_exchange(pg, rank);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

        } else
        {
            if ((mpi_errno = MPID_nem_ib_pmi_exchange()) != MPI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**fail",
                        "**fail %s", "Failed to get HCA attrs");
            }
            rdma_param_handle_heterogeneity(conn_info.init_info->arch_hca_type, conn_info.size);
        }
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_EXCHANGE_CONN_INFO);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

/**
 * Enable all the queue pair connections
 */
#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_establish_conn
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_establish_conn()
{
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_ESTABLISH_CONN);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_ESTABLISH_CONN);

    struct ibv_qp_attr  qp_attr;
    uint32_t            qp_attr_mask = 0;
    static int          rdma_qos_sl = 0;

    int i;
    int size = conn_info.size;
    int rank = conn_info.rank;
    int rail_index;

    /**********************  INIT --> RTR  ************************/
    memset(&qp_attr, 0, sizeof qp_attr);
    qp_attr.qp_state    =   IBV_QPS_RTR;
    qp_attr.rq_psn      =   rdma_default_psn;
    qp_attr.max_dest_rd_atomic  =   rdma_default_max_rdma_dst_ops;
    qp_attr.min_rnr_timer       =   rdma_default_min_rnr_timer;
    if (rdma_use_qos) {
        qp_attr.ah_attr.sl          =   rdma_qos_sl;
        rdma_qos_sl = (rdma_qos_sl + 1) % rdma_qos_num_sls;
    } else {
        qp_attr.ah_attr.sl          =   rdma_default_service_level;
    }

    qp_attr.ah_attr.static_rate =   rdma_default_static_rate;
    qp_attr.ah_attr.src_path_bits   =   rdma_default_src_path_bits;

    qp_attr.ah_attr.is_global   =   0;
    qp_attr.path_mtu    =   rdma_default_mtu;

    qp_attr_mask        |=  IBV_QP_STATE;
    qp_attr_mask        |=  IBV_QP_PATH_MTU;
    qp_attr_mask        |=  IBV_QP_RQ_PSN;
    qp_attr_mask        |=  IBV_QP_MAX_DEST_RD_ATOMIC;
    qp_attr_mask        |=  IBV_QP_MIN_RNR_TIMER;
    qp_attr_mask        |=  IBV_QP_AV;

    for (i = 0; i < size; i++) {
        if (i == rank)
            continue;

        conn_info.connections[i].remote_vc_addr = conn_info.init_info->vc_addr[i];
        for (rail_index = 0; rail_index < rdma_num_rails; rail_index ++) {
            /* should we store port, qp_num_rdma and lids in conn_info
             * in order to simplify the calculation here?
             */
            int hca_index, port_index;
            hca_index  = rail_index / (rdma_num_rails / ib_hca_num_hcas);
            port_index = (rail_index / (rdma_num_rails / (ib_hca_num_hcas * ib_hca_num_ports))) % ib_hca_num_ports;

            if (!use_iboeth && (rdma_3dtorus_support || rdma_path_sl_query)) {
                /* Path SL Lookup */
                struct ibv_context *context = hca_list[hca_index].nic_context;
                struct ibv_pd *pd  = hca_list[hca_index].ptag;
                uint16_t lid       = hca_list[hca_index].lids[port_index];
                uint16_t rem_lid   = conn_info.init_info->lid[i][rail_index];
                uint32_t port_num  = hca_list[hca_index].ports[port_index];
                qp_attr.ah_attr.sl = mv2_get_path_rec_sl(context, pd, port_num,
                                            lid, rem_lid, rdma_3dtorus_support,
                                            rdma_num_sa_query_retries);
            }


            qp_attr.dest_qp_num = conn_info.init_info->qp_num_rdma[i][rail_index];
            qp_attr.ah_attr.port_num = hca_list[hca_index].ports[port_index];
            qp_attr.ah_attr.dlid = conn_info.init_info->lid[i][rail_index];
            qp_attr_mask    |=  IBV_QP_DEST_QPN;
            if (ibv_modify_qp(conn_info.connections[i].rails[rail_index].qp_hndl,
                       &qp_attr, qp_attr_mask)) {
                fprintf(stderr, "[%s:%d] Could not modify qp"
                    "to RTR\n",__FILE__, __LINE__);
                return 1;
            }
        }
    } /*for (i = 0; i < size; i++)*/

    /************** RTR --> RTS *******************/
    memset(&qp_attr, 0, sizeof qp_attr);
    qp_attr.qp_state        = IBV_QPS_RTS;
    qp_attr.sq_psn          = rdma_default_psn;
    qp_attr.timeout         = rdma_default_time_out;
    qp_attr.retry_cnt       = rdma_default_retry_count;
    qp_attr.rnr_retry       = rdma_default_rnr_retry;
    qp_attr.max_rd_atomic   = rdma_default_qp_ous_rd_atom;

    qp_attr_mask = 0;
    qp_attr_mask =      IBV_QP_STATE              |
                        IBV_QP_TIMEOUT            |
                        IBV_QP_RETRY_CNT          |
                        IBV_QP_RNR_RETRY          |
                        IBV_QP_SQ_PSN             |
                        IBV_QP_MAX_QP_RD_ATOMIC;

    /* MRAILI_Init_vc is added here */
    for (i = 0; i < size; i++) {
        if (i == rank)
            continue;

        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
            if (ibv_modify_qp(conn_info.connections[i].rails[rail_index].qp_hndl, &qp_attr,
                qp_attr_mask)) {
                fprintf(stderr, "[%s:%d] Could not modify rdma qp to RTS\n",
                    __FILE__, __LINE__);
                return 1;
            }
        }
    } /*for (i = 0; i < size; i++)*/

    DEBUG_PRINT("Done enabling connections\n");

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_ESTABLISH_CONN);
    return mpi_errno;
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

static int _setup_ib_boot_ring(struct init_addr_inf * neighbor_addr,
                              int port)
{
    struct ibv_qp_attr      qp_attr;
    uint32_t    qp_attr_mask = 0;
    int         i;
    int         ret;
    static int  rdma_qos_sl = 0;

    qp_attr.qp_state        = IBV_QPS_INIT;
    set_pkey_index(&qp_attr.pkey_index, 0, port);
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    qp_attr.port_num        = port;

    DEBUG_PRINT("default port %d, qpn %x\n", port,
            process_info.boot_qp_hndl[0]->qp_num);

    ret = ibv_modify_qp(process_info.boot_qp_hndl[0],&qp_attr,(IBV_QP_STATE
                        | IBV_QP_PKEY_INDEX
                        | IBV_QP_PORT
                        | IBV_QP_ACCESS_FLAGS));
    CHECK_RETURN(ret, "Could not modify boot qp to INIT");

    ret = ibv_modify_qp(process_info.boot_qp_hndl[1],&qp_attr,(IBV_QP_STATE
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
    if (rdma_use_qos) {
        qp_attr.ah_attr.sl          =   rdma_qos_sl;
        rdma_qos_sl = (rdma_qos_sl + 1) % rdma_qos_num_sls;
    } else {
        qp_attr.ah_attr.sl          =   rdma_default_service_level;
    }

    qp_attr.ah_attr.static_rate =   rdma_default_static_rate;
    qp_attr.ah_attr.src_path_bits   =   rdma_default_src_path_bits;
    qp_attr.ah_attr.port_num    =   port;

#if defined(USE_IBOETH)
    if (use_iboeth) {
        qp_attr.ah_attr.grh.dgid.global.subnet_prefix = 0;
        qp_attr.ah_attr.grh.dgid.global.interface_id = 0;
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.sgid_index = 0;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.ah_attr.grh.traffic_class = 0;
        qp_attr.ah_attr.is_global      = 1;
        qp_attr.ah_attr.dlid           = 0;
        qp_attr.path_mtu            = IBV_MTU_1024;
    } else 
#endif
    {
        qp_attr.ah_attr.is_global   =   0;
        qp_attr.path_mtu    =   rdma_default_mtu;
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
            struct ibv_context *context = hca_list[0].nic_context;
            struct ibv_pd *pd  = hca_list[0].ptag;
            /* don't know our local LID yet, so set it to 0 to let
               mv2_get_path_rec_sl do the lookup */
            /*uint16_t lid = proc->lids[0][port];*/
            uint16_t lid = 0x0;
            uint16_t rem_lid   = qp_attr.ah_attr.dlid;
            uint32_t port_num  = qp_attr.ah_attr.port_num;
            qp_attr.ah_attr.sl = mv2_get_path_rec_sl(context, pd, port_num,
                                            lid, rem_lid, rdma_3dtorus_support,
                                            rdma_num_sa_query_retries);
        }

        ret = ibv_modify_qp(process_info.boot_qp_hndl[i],&qp_attr, qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTR");

        DEBUG_PRINT("local QP=%x\n", process_info.boot_qp_hndl[i]->qp_num);
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

    ret = ibv_modify_qp(process_info.boot_qp_hndl[0],&qp_attr,qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTS");
    ret = ibv_modify_qp(process_info.boot_qp_hndl[1],&qp_attr,qp_attr_mask);
        CHECK_RETURN(ret, "Could not modify boot qp to RTS");

    DEBUG_PRINT("Modified to RTS..Qp\n");
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME rdma_setup_startup_ring
#undef FCNAME                                
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_setup_startup_ring(int pg_rank, int pg_size)
{
    struct init_addr_inf neighbor_addr[2];
    char ring_qp_out[128];
    char ring_qp_in[256];               
    int bootstrap_len;                  
    int mpi_errno = MPI_SUCCESS;
    int port;
        
    port = _find_active_port(hca_list[0].nic_context);
    if (port < 0) {
        MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto out, "**fail",
                "**fail %s", "could not find active port");
    }

    process_info.boot_cq_hndl = ibv_create_cq(hca_list[0].nic_context,
                                       rdma_default_max_cq_size,
                                       NULL, NULL, 0);
    if (!process_info.boot_cq_hndl) {  
        MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "**fail %s", "cannot create cq");
    }

    process_info.boot_qp_hndl[0] = create_qp(hca_list[0].ptag, process_info.boot_cq_hndl,
                                      process_info.boot_cq_hndl);
    if (!process_info.boot_qp_hndl[0]) {
        MPIR_ERR_SETFATALANDSTMT2(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "%s%d", "Fail to create qp on rank ", pg_rank);
    }

    process_info.boot_qp_hndl[1] = create_qp(hca_list[0].ptag, process_info.boot_cq_hndl,
                                      process_info.boot_cq_hndl);
    if (!process_info.boot_qp_hndl[1]) {
        MPIR_ERR_SETFATALANDSTMT2(mpi_errno, MPI_ERR_OTHER, goto out,
                "**fail", "%s%d", "Fail to create qp on rank ", pg_rank);
    }

#if defined(USE_IBOETH)
    if (use_iboeth) {
        gid = get_local_gid(mv2_MPIDI_CH3I_RDMA_Process.nic_context[0], port);
        sprintf(ring_qp_out, "%016llx:%016llx:%08x:%08x:",
                 gid.global.subnet_prefix, gid.global.interface_id,
                 proc->boot_qp_hndl[0]->qp_num,
                 proc->boot_qp_hndl[1]->qp_num
               );

        DEBUG_PRINT("After setting GID: %llx:%llx, qp0: %x, qp1: %x\n",
                gid.global.subnet_prefix, gid.global.interface_id,
                proc->boot_qp_hndl[0]->qp_num,
                proc->boot_qp_hndl[1]->qp_num
                );
    } else 
#endif
    {
        sprintf(ring_qp_out, "%08x:%08x:%08x:",
                 get_local_lid(hca_list[0].nic_context,
                               port),
                 process_info.boot_qp_hndl[0]->qp_num,
                 process_info.boot_qp_hndl[1]->qp_num
               );

        DEBUG_PRINT("After setting LID: %d, qp0: %x, qp1: %x\n",
                get_local_lid(hca_list[0].nic_context,
                              port),
                process_info.boot_qp_hndl[0]->qp_num,
                process_info.boot_qp_hndl[1]->qp_num
                );
    }

    bootstrap_len = strlen(ring_qp_out);
    _rdma_pmi_exchange_addresses(pg_rank, pg_size, ring_qp_out,
            bootstrap_len, ring_qp_in);

#if defined(USE_IBOETH)
    if (use_iboeth) {
        sscanf(&ring_qp_in[0], "%016llx:%016llx:%08x:%08x:",
               &neighbor_addr[0].gid.global.subnet_prefix,
               &neighbor_addr[0].gid.global.interface_id,
               &neighbor_addr[0].qp_num[0],
               &neighbor_addr[0].qp_num[1]);
        sscanf(&ring_qp_in[53], "%016llx:%016llx:%08x:%08x:",
               &neighbor_addr[1].gid.global.subnet_prefix,
               &neighbor_addr[1].gid.global.interface_id,
               &neighbor_addr[1].qp_num[0],
               &neighbor_addr[1].qp_num[1]);
        DEBUG_PRINT("After retrieving GID: %llx:%llx, qp0: %x, qp1: %x\n",
               neighbor_addr[0].gid.global.subnet_prefix,
               neighbor_addr[0].gid.global.interface_id,
               neighbor_addr[0].qp_num[0],
               neighbor_addr[0].qp_num[1]);
        DEBUG_PRINT("After retrieving GID: %llx:%llx, qp0: %x, qp1: %x\n",
               neighbor_addr[1].gid.global.subnet_prefix,
               neighbor_addr[1].gid.global.interface_id,
               neighbor_addr[1].qp_num[0],
               neighbor_addr[1].qp_num[1]);
    } else 
#endif
    {
        sscanf(&ring_qp_in[0], "%08x:%08x:%08x:",
               &neighbor_addr[0].lid,
               &neighbor_addr[0].qp_num[0],
               &neighbor_addr[0].qp_num[1]);
        sscanf(&ring_qp_in[27], "%08x:%08x:%08x:",
               &neighbor_addr[1].lid,
               &neighbor_addr[1].qp_num[0],
               &neighbor_addr[1].qp_num[1]);
    }

    mpi_errno = _setup_ib_boot_ring(neighbor_addr, port);
    UPMI_BARRIER();

out:
    return mpi_errno;

}

#undef FUNCNAME
#define FUNCNAME rdma_ring_based_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_ring_based_allgather(void *sbuf, int data_size,
        int pg_rank, void *rbuf, int pg_size)
{       
    struct ibv_mr * addr_hndl;                  
    int i;                                      
    int mpi_errno = MPI_SUCCESS;
    addr_hndl = ibv_reg_mr(hca_list[0].ptag,
            rbuf, data_size*pg_size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        
    if(addr_hndl == NULL) {
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
        for(i = 0; i < MPD_WINDOW; i++) {
            if (recv_post_index == pg_rank)
                continue;
            rr.wr_id   = recv_post_index;
            rr.num_sge = 1;
            rr.sg_list = &(sg_entry_r);
            rr.next    = NULL;
            sg_entry_r.lkey = addr_hndl->lkey;
            sg_entry_r.addr = (uintptr_t)(rbufProxy+data_size*recv_post_index);
            sg_entry_r.length = data_size;

            if(ibv_post_recv(process_info.boot_qp_hndl[0], &rr, &bad_wr_r)) {
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
        while (recv_comp_index !=
                (pg_rank+1)%pg_size ||
                send_comp_index !=
                (pg_rank+2)%pg_size+pg_size) {
            int ne;
            /* Three conditions
             * 1: not complete sending
             * 2: has received the data
             * 3: has enough credit
             */
            if (send_post_index != (pg_rank+1)%pg_size &&
                    (recv_comp_index == send_post_index ||
                 is_A_on_left_of_B(recv_comp_index,
                     send_post_index,pg_rank,pg_size)) && credit > 0) {

                sr.opcode         = IBV_WR_SEND;
                sr.send_flags     = IBV_SEND_SIGNALED;
                sr.wr_id          = send_post_index+pg_size;
                sr.num_sge        = 1;
                sr.sg_list        = &sg_entry_s;
                sr.next           = NULL;
                sg_entry_s.addr   = (uintptr_t)(rbufProxy+data_size*send_post_index);
                sg_entry_s.length = data_size;
                sg_entry_s.lkey   = addr_hndl->lkey;

                if (ibv_post_send(process_info.boot_qp_hndl[1], &sr, &bad_wr_s)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                            MPI_ERR_INTERN,
                            "**fail",
                            "**fail %s",
                            "Error posting send!\n");
                }

                send_post_index=round_left(send_post_index,pg_size);
                credit--;
            }

            ne = ibv_poll_cq(process_info.boot_cq_hndl, 1, &rc);
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
                        sg_entry_r.addr = (uintptr_t)(rbufProxy+data_size*recv_post_index);
                        sg_entry_r.length = data_size;

                        if(ibv_post_recv(process_info.boot_qp_hndl[0], &rr, &bad_wr_r)) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno,
                                    MPI_ERR_INTERN,
                                    "**fail",
                                    "**fail %s",
                                    "Error posting recv!\n");
                        }

                        recv_post_index = round_left(recv_post_index,pg_size);
                    }
                }
                else {
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
#define FUNCNAME MPID_nem_ib_setup_startup_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_setup_startup_ring(MPIDI_PG_t *pg, int rank)
{
    int mpi_errno = MPI_SUCCESS;
    int pg_size;
    uint64_t my_arch_hca_type;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_SETUP_STARTUP_RING);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_SETUP_STARTUP_RING);

    pg_size = MPIDI_PG_Get_size(pg);
    if(pg_size > 1) {
        /* IBoEth mode does not support loop back connections as of now. Ring
         * based connection setup uses QP's to exchange info between all
         * processes, whether they're on the same node or different nodes.
         */
        mpi_errno = rdma_setup_startup_ring(rank, pg_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        my_arch_hca_type = process_info.arch_hca_type;

        mpi_errno = rdma_ring_based_allgather(&my_arch_hca_type, sizeof(my_arch_hca_type),
                                        rank, conn_info.init_info->arch_hca_type, pg_size);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        /* Check heterogeneity */
        rdma_param_handle_heterogeneity(conn_info.init_info->arch_hca_type, pg_size);
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_SETUP_STARTUP_RING);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

int mv2_allocate_pmi_keyval(void)
{
    if (!mv2_pmi_max_keylen) {
        UPMI_KVS_GET_KEY_LENGTH_MAX(&mv2_pmi_max_keylen);
    }
    if (!mv2_pmi_max_vallen) {
        UPMI_KVS_GET_VALUE_LENGTH_MAX(&mv2_pmi_max_vallen);
    }

    mv2_pmi_key = MPIU_Malloc(mv2_pmi_max_keylen+1);
    mv2_pmi_val = MPIU_Malloc(mv2_pmi_max_vallen+1);

    if (mv2_pmi_key==NULL || mv2_pmi_val==NULL) {
        mv2_free_pmi_keyval();
        return -1; 
    }
    return 0;
}

void mv2_free_pmi_keyval(void)
{
    if (mv2_pmi_key!=NULL) {
        MPIU_Free(mv2_pmi_key);
        mv2_pmi_key = NULL;
    }

    if (mv2_pmi_val!=NULL) {
        MPIU_Free(mv2_pmi_val);
        mv2_pmi_val = NULL;
    }
}

