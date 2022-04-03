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

#include "mpichconf.h"
#include <mpimem.h>
#include "rdma_impl.h"
#include "upmi.h"
#include "vbuf.h"
#include "dreg.h"
#include "rdma_cm.h"
#include "cm.h"
#include "mv2_utils.h"
#ifdef RDMA_CM
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>

#define RDMA_MAX_PRIVATE_LENGTH     56
#define MV2_RDMA_CM_MIN_PORT_LIMIT  1024
#define MV2_RDMA_CM_MAX_PORT_LIMIT  65536

int *rdma_base_listen_port = NULL;
int *rdma_cm_host_list = NULL;
#ifdef _MULTI_SUBNET_SUPPORT_
static uint64_t *rdma_base_listen_sid = NULL;
union ibv_gid *rdma_cm_host_gid_list = NULL;
#endif
int *rdma_cm_local_ips;
int *rdma_cm_accept_count;
volatile int *rdma_cm_connect_count;
volatile int *rdma_cm_iwarp_msg_count;
volatile int rdma_cm_connected_count = 0;
volatile int rdma_cm_num_expected_connections = 0;
volatile int rdma_cm_finalized = 0;
int rdma_cm_arp_timeout = 2000;
int g_num_smp_peers = 0;

char *init_message_buf;        /* Used for message exchange in RNIC case */
struct ibv_mr *init_mr;
struct ibv_sge init_send_sge;
struct ibv_recv_wr init_rwr;
struct ibv_send_wr init_swr;
struct rdma_cm_id *tmpcmid;    
sem_t rdma_cm_addr;

/* Handle the connection events */
static int ib_cma_event_handler(struct rdma_cm_id *cma_id,
        struct rdma_cm_event *event);

/* Thread to poll and handle CM events */
void *cm_thread(void *arg);

/* Obtain the information of local RNIC IP from the mv2.conf file */
int rdma_cm_get_local_ip(int *num_interfaces);

/* create qp's for a ongoing connection request */
int rdma_cm_create_qp(MPIDI_VC_t *vc, int rail_index);

/* Initialize pd and cq associated with one rail */
int rdma_cm_init_pd_cq();

/* Get the rank of an active connect request */
int get_remote_rank(struct rdma_cm_id *cmid);

/* Get the rank of an active connect request */
int get_remote_rail(struct rdma_cm_id *cmid);

/* Get the rank of an active connect request */
int get_remote_qp_type(struct rdma_cm_id *cmid);

/* Exchange init messages for iWARP compliance */
int init_messages(int *hosts, int pg_rank, int pg_size);

/* RDMA_CM specific method implementations */

#undef FUNCNAME
#define FUNCNAME ib_cma_event_handler
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int static ib_cma_event_handler(struct rdma_cm_id *cma_id,
        struct rdma_cm_event *event)
{
    int mpi_errno = MPI_SUCCESS;
    int ret, rank, rail_index = 0;
    int connect_attempts = 0;
    int exp_factor = 1;
    int offset = 0, private_data_start_offset = 0;
    int pg_size, pg_rank, tmplen;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    MPIDI_VC_t  *vc, *gotvc;
    MPIDI_PG_t *pg_tmp;
    struct rdma_conn_param conn_param;

    MPIDI_STATE_DECL(MPIDI_STATE_IB_CMA_EVENT_HANDLER);
    MPIDI_FUNC_ENTER(MPIDI_STATE_IB_CMA_EVENT_HANDLER);

    UPMI_GET_RANK(&pg_rank);
    UPMI_GET_SIZE(&pg_size);

    switch (event->event) {

        case RDMA_CM_EVENT_ADDR_RESOLVED:
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"case RDMA_CM_ADDR_RESOLVED\n");
            if (cma_id == tmpcmid) {
                sem_post(&rdma_cm_addr);
                break;
            }

            do {
                ret = rdma_resolve_route(cma_id, rdma_cm_arp_timeout*exp_factor);
                if (ret) {
                    connect_attempts++;
                    exp_factor *= 2;
                    PRINT_DEBUG(DEBUG_CM_verbose>0, "connect_attempts = %d, exp_factor=%d, ret = %d,"
                        "wait_time = %d ms\n", connect_attempts, exp_factor, ret,
                        (rdma_cm_arp_timeout*exp_factor));
                }
            } while (ret && (connect_attempts < max_rdma_connect_attempts));
            if (ret) {
                MPIR_ERR_SETANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "rdma_resolve_route error %d after"
                    " %d attempts\n", ret, connect_attempts);
            }

        break;
        case RDMA_CM_EVENT_ROUTE_RESOLVED:

            /* VC pointer is stored in cm_id->context at cm_id creation */
            vc = (MPIDI_VC_t *) cma_id->context;
            rank = vc->pg_rank;
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"case RDMA_CM_EVENT_ROUTE_RESOLVED for %d\n", rank);
            rail_index = get_remote_rail(cma_id);

            /* In a multi-rail scenario, we hit this condition incorrectly
             * when establishing a loopback connection. This causes the latter
             * connections to get dropped incorrectly. This updated
             * condition is to catch this situation.
             */
            if ((vc->ch.state != MPIDI_CH3I_VC_STATE_CONNECTING_CLI) && (rank != pg_rank)) {
                /* Switched into server mode */
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"Switched into server mode for %d\n", rank);
                break;
            }
    
            if (rank < 0 || rail_index < 0) {
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"Unexpected error occured\n");
            }

            rdma_cm_create_qp(vc, rail_index);

            /* Connect to remote node */
            MPIU_Memset(&conn_param, 0, sizeof conn_param);
#ifdef _ENABLE_XRC_
            if (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity || USE_XRC)
#else
            if (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity)
#endif
            {
                conn_param.initiator_depth      = rdma_default_max_rdma_dst_ops;
                conn_param.responder_resources  = rdma_default_max_rdma_dst_ops;
            } else {
                conn_param.initiator_depth      = rdma_supported_max_rdma_dst_ops;
                conn_param.responder_resources  = rdma_supported_max_rdma_dst_ops;
            }
            conn_param.retry_count = rdma_default_rnr_retry;
            conn_param.rnr_retry_count = rdma_default_rnr_retry;

#ifdef _MULTI_SUBNET_SUPPORT_
            if (mv2_rdma_cm_multi_subnet_support) {
                /* Leave additional offset at the beginning of private_data if
                 * we are supporting routing */
                tmplen = 4 * sizeof(uint64_t) + 1;
                private_data_start_offset = 1;
            } else
#endif /* _MULTI_SUBNET_SUPPORT_ */
            {
                tmplen = 3 * sizeof(uint64_t) + 1;
                private_data_start_offset = 0;
            }
            if(tmplen > RDMA_MAX_PRIVATE_LENGTH) {
                PRINT_ERROR("Length of private data too long. Requested: %d. Supported: %d.",
                            tmplen, RDMA_MAX_PRIVATE_LENGTH);
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "Cannot use RDMA CM on rank %d", pg_rank);
            }

            PRINT_DEBUG(DEBUG_RDMACM_verbose,"allocating %d bytes for private_data\n", tmplen);
            conn_param.private_data = MPIU_Malloc(tmplen);

            if (!conn_param.private_data) {
                MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
            }
    
            do {
                offset = private_data_start_offset;
                conn_param.private_data_len = tmplen;
                ((uint64_t *) conn_param.private_data)[offset++] = pg_rank;
                ((uint64_t *) conn_param.private_data)[offset++] = rail_index;
                ((uint64_t *) conn_param.private_data)[offset] = (uint64_t) vc;
                offset = private_data_start_offset;
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"Sending connection request to [rank = %ld], "
                        " [rail = %ld] [vc = %lx]\n",
                        ((uint64_t *) conn_param.private_data)[offset],
                        ((uint64_t *) conn_param.private_data)[offset+1],
                        ((uint64_t *) conn_param.private_data)[offset+2]);

                ret = rdma_connect(cma_id, &conn_param);
                connect_attempts++;
                if (ret) {
                    usleep(rdma_cm_connect_retry_interval*exp_factor);
                    exp_factor *= 2;
                }
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"connect_attempts = %d, exp_factor=%d, ret = %d,"
                    "wait_time = %d\n", connect_attempts, exp_factor, ret,
                    (rdma_cm_connect_retry_interval*exp_factor));
            } while (ret && (connect_attempts < max_rdma_connect_attempts));

            if (ret) {
                MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "rdma_connect error %d after %d attempts\n",
                        ret, connect_attempts);
            }

            MPIU_Free(conn_param.private_data);

        break;
        case RDMA_CM_EVENT_CONNECT_REQUEST:
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"case RDMA_CM_EVENT_CONNECT_REQUEST\n");
#ifdef _MULTI_SUBNET_SUPPORT_
            if (mv2_rdma_cm_multi_subnet_support) {
                /* Leave additional offset at the beginning of private_data if
                 * we are supporting routing */
                tmplen = 2 * sizeof(uint64_t) + 1;
                private_data_start_offset = 1;
            } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
            {
                tmplen = 1 * sizeof(uint64_t) + 1;
                private_data_start_offset = 0;
            }

            offset = private_data_start_offset;
#ifndef OFED_VERSION_1_1        /* OFED 1.2 */
            if (!event->param.conn.private_data_len){
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                     "**fail %s", "Error obtaining remote data from event private data\n");
            }
            rank       = ((uint64_t *) event->param.conn.private_data)[offset++];
            rail_index = ((uint64_t *) event->param.conn.private_data)[offset++];
            gotvc      = (MPIDI_VC_t *) ((uint64_t *) 
                event->param.conn.private_data)[offset];
#else  /* OFED 1.1 */
            if (!event->private_data_len){
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                     "**fail %s", "Error obtaining remote data from event private data\n");
            }
            rank       = ((uint64_t *) event->private_data)[offset++];
            rail_index = ((uint64_t *) event->private_data)[offset++];
            gotvc      = (MPIDI_VC_t*) ((uint64_t *) event->private_data)[offset];
#endif

            PRINT_DEBUG(DEBUG_RDMACM_verbose,"Passive side recieved connect request: [%d] :[%d]" 
            " [vc: %p], vc->pg_rank = %d\n", rank, rail_index, gotvc, gotvc->pg_rank);
    
            MPIDI_PG_Find(MPIDI_Process.my_pg->id, &pg_tmp);
            if(pg_tmp == NULL) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                     "**fail %s", "Could not find PG in conn request\n");
            }

            MPIDI_PG_Get_vc(pg_tmp, rank, &vc);
            cma_id->context = vc;
            vc->mrail.remote_vc_addr = (uint64_t) gotvc;

            /* Both ranks are trying to connect. Clearing race condition */
            if (((vc->ch.state == MPIDI_CH3I_VC_STATE_CONNECTING_CLI) && (pg_rank > rank)) 
                || vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE 
                || vc->ch.state == MPIDI_CH3I_VC_STATE_IWARP_CLI_WAITING )
            {
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"Passive size rejecting connect request: "
                    "Crossing connection requests expected\n");
                ret = rdma_reject(cma_id, NULL, 0);
                if (ret) {
                    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "rdma_reject error: %d\n", ret);
                }
                break;
            }
    
            /* Accepting the connection */
            rdma_cm_accept_count[rank]++;
    
            if (proc->use_iwarp_mode)
                vc->ch.state = MPIDI_CH3I_VC_STATE_IWARP_SRV_WAITING;
            else
                vc->ch.state = MPIDI_CH3I_VC_STATE_CONNECTING_SRV;

            PRINT_DEBUG(DEBUG_CM_verbose>0, "Current state of %d is %s, moving to LOCAL_ACTIVE.\n",  vc->pg_rank, MPIDI_VC_GetStateString(vc->state));
            vc->state = MPIDI_VC_STATE_LOCAL_ACTIVE;

            vc->mrail.rails[rail_index].cm_ids = cma_id;
        
            /* Create qp */
            rdma_cm_create_qp(vc, rail_index);

            /* Posting a single buffer to cover for iWARP MPA requirement. */
            if (proc->use_iwarp_mode && !proc->has_srq)
            {
                PREPOST_VBUF_RECV(vc, rail_index);
            }
            if (rdma_cm_accept_count[rank] == rdma_num_rails)
            {
                MRAILI_Init_vc(vc);
            }
            offset = private_data_start_offset;
            /* Accept remote connection - passive connect */
            MPIU_Memset(&conn_param, 0, sizeof conn_param);
#ifdef _ENABLE_XRC_
            if (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity || USE_XRC)
#else
            if (mv2_MPIDI_CH3I_RDMA_Process.heterogeneity)
#endif
            {
                conn_param.initiator_depth      = rdma_default_max_rdma_dst_ops;
                conn_param.responder_resources  = rdma_default_max_rdma_dst_ops;
            } else {
                conn_param.initiator_depth      = rdma_supported_max_rdma_dst_ops;
                conn_param.responder_resources  = rdma_supported_max_rdma_dst_ops;
            }
            conn_param.retry_count = rdma_default_rnr_retry;
            conn_param.rnr_retry_count = rdma_default_rnr_retry;
            conn_param.private_data_len = tmplen;
            conn_param.private_data = MPIU_Malloc(conn_param.private_data_len);
            ((uint64_t *) conn_param.private_data)[offset] = (uint64_t) vc;
            ret = rdma_accept(cma_id, &conn_param);
            if (ret) {
                MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "rdma_accept error: %d\n", ret);
            }
            MPIU_Free(conn_param.private_data);

        break;
        case RDMA_CM_EVENT_ESTABLISHED:
            vc = (MPIDI_VC_t *) cma_id->context;
            rank = vc->pg_rank;
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"case RDMA_CM_EVENT_ESTABLISHED for rank %d\n", rank);
#ifdef _MULTI_SUBNET_SUPPORT_
            if (mv2_rdma_cm_multi_subnet_support) {
                /* Leave additional offset at the beginning of private_data if
                 * we are supporting routing */
                private_data_start_offset = 1;
            } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
            {
                private_data_start_offset = 0;
            }

#ifndef OFED_VERSION_1_1        /* OFED 1.2 */
            if (event->param.conn.private_data_len) 
                vc->mrail.remote_vc_addr = ((uint64_t *)
                    event->param.conn.private_data)[private_data_start_offset];
#else  /* OFED 1.1 */
            if (event->private_data_len) 
                vc->mrail.remote_vc_addr = ((uint64_t *) 
                    event->private_data)[private_data_start_offset];
#endif

            if (rank < 0) {        /* Overlapping connections */
                PRINT_DEBUG(DEBUG_RDMACM_verbose,"Got event for overlapping connections? "
                   " removing...\n");
                break;
            }

            rdma_cm_connect_count[rank]++;

            int i = 0;
            /* When establishing a loopback connection, we receive two
             * RDMA_CM_EVENT_ESTABLISHED events. The updated condition
             * below reflects this.
             */
            if (((vc->pg_rank == pg_rank) && (rdma_cm_connect_count[rank] == 2*rdma_num_rails)) ||
                ((vc->pg_rank != pg_rank) && (rdma_cm_connect_count[rank] == rdma_num_rails)))
            {
                if (vc->ch.state == MPIDI_CH3I_VC_STATE_CONNECTING_CLI) {
                    /* Server has init'ed before accepting */
                    MRAILI_Init_vc(vc);

                    /* Sending a noop for handling the iWARP requirement */
                    if (proc->use_iwarp_mode) {
                        vc->ch.state = MPIDI_CH3I_VC_STATE_IWARP_CLI_WAITING;
                        for (i = 0; i < rdma_num_rails; i++){
                            MRAILI_Send_noop(vc, i);
                            PRINT_DEBUG(DEBUG_RDMACM_verbose,"Sending noop to [%d]\n", rank);
                        }
                     }
                     else {
                         vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
                         vc->state = MPIDI_VC_STATE_ACTIVE;
                         MPIDI_CH3I_Process.new_conn_complete = 1;
                         PRINT_DEBUG(DEBUG_RDMACM_verbose,"Connection Complete - Client: %d->%d\n", 
                             pg_rank, rank);
                         if (mv2_use_eager_fast_send &&
                             !(SMP_INIT && (vc->smp.local_nodes >= 0))) {
                             if (likely(rdma_use_coalesce)) {
                                 vc->eager_fast_fn = mv2_eager_fast_coalesce_send;
                             } else {
                                 vc->eager_fast_fn = mv2_eager_fast_send;
                             }
                         }
                     }
                 }
                 else {         /* Server side */
                     if (!proc->use_iwarp_mode ||
                         (rdma_cm_iwarp_msg_count[vc->pg_rank] 
                         >= rdma_num_rails)) {

                         if ((vc->ch.state == 
                             MPIDI_CH3I_VC_STATE_IWARP_SRV_WAITING)
                             || (vc->ch.state == 
                             MPIDI_CH3I_VC_STATE_CONNECTING_SRV)) {

                             vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
                             vc->state = MPIDI_VC_STATE_ACTIVE;
                             MPIDI_CH3I_Process.new_conn_complete = 1;
                             for (i = 0; i < rdma_num_rails; i++){
                                 MRAILI_Send_noop(vc, i);
                             }
                             PRINT_DEBUG(DEBUG_RDMACM_verbose,"Connection Complete - Server: "
                             "%d->%d\n", pg_rank, rank);
                             if (mv2_use_eager_fast_send &&
                                 !(SMP_INIT && (vc->smp.local_nodes >= 0))) {
                                 if (likely(rdma_use_coalesce)) {
                                     vc->eager_fast_fn = mv2_eager_fast_coalesce_send;
                                 } else {
                                     vc->eager_fast_fn = mv2_eager_fast_send;
                                 }
                             }
                         }
                     }
                 }
                 rdma_cm_connected_count++;
             }

             /* All connections connected? Used only for non-on_demand case.
              * In multi-qp scenarios, we are seeing some random events from the
              * HCA which is causing the sem_post to get triggered multiple
              * times even before the sem_wait is called. This is causing the
              * process to exit rdma_cm_connect_all even before the connections
              * have been fully established. The cause for the generation of the
              * events is still unclear. The first two conditions chceking
              * if the counters are non-zero are to catch this corner case.
              */
             if (rdma_cm_connected_count && rdma_cm_num_expected_connections &&
                 (rdma_cm_connected_count == rdma_cm_num_expected_connections)) {
                 sem_post(&proc->rdma_cm);        
             }

        break;

        case RDMA_CM_EVENT_ADDR_ERROR:
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                "RDMA CM Address error: rdma cma event %d, error %d\n", 
                    event->event, event->status);
        break;
        case RDMA_CM_EVENT_ROUTE_ERROR:
            PRINT_DEBUG(DEBUG_CM_verbose>0,
                "RDMA CM Route error: rdma cma event %d, error %d, Retrying\n",
                    event->event, event->status);
            /* If rdma_resolve_route failed, retry */
	    ret = rdma_resolve_route(cma_id, rdma_cm_arp_timeout*exp_factor);
        break;
        case RDMA_CM_EVENT_CONNECT_ERROR:
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                "RDMA CM Connect error: rdma cma event %d, error %d\n", 
                    event->event, event->status);
        break;
        case RDMA_CM_EVENT_UNREACHABLE:
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
                "RDMA CM Unreachable error: rdma cma event %d, error %d\n",
                event->event, event->status);
        break;
#if 0
        /*
         * These events don't really need a case since they are currently no
         * ops.
         */
        case RDMA_CM_EVENT_REJECTED:
        PRINT_DEBUG(DEBUG_RDMACM_verbose,"RDMA CM Reject Event %d, error %d\n", event->event, 
            event->status);
        break;

        case RDMA_CM_EVENT_DISCONNECTED:
        break;

        case RDMA_CM_EVENT_TIMEWAIT_EXIT:
        PRINT_DEBUG(DEBUG_RDMACM_verbose,"caught RDMA_CM_EVENT_TIMEWAIT_EXIT \n");
        break;  

        case RDMA_CM_EVENT_DEVICE_REMOVAL:
#endif

        default:
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"%s: Caught unhandled rdma cm event - %s\n",
                __FUNCTION__, rdma_event_str(event->event));
        break;
    }
fn_fail:
    MPIDI_FUNC_EXIT(MPIDI_STATE_IB_CMA_EVENT_HANDLER);
    return ret;
}

#undef FUNCNAME
#define FUNCNAME cm_thread
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void *cm_thread(void *arg)
{
    struct rdma_cm_event *event = NULL;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;    
    int mpi_errno;

    while (1) {

        event = NULL;
        mpi_errno = rdma_get_cm_event(proc->cm_channel, &event);
        if (rdma_cm_finalized) {
            if (event != NULL) {
                rdma_ack_cm_event(event);
            }
            return NULL;
        }
        if (mpi_errno) {
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "rdma_get_cm_event err %d\n", mpi_errno);
        }

        PRINT_DEBUG(DEBUG_RDMACM_verbose,"rdma cm event[id: %p]: %d\n", event->id, event->event);
        {
         
            MPICM_lock();
            mpi_errno = ib_cma_event_handler(event->id, event);
            MPICM_unlock();
        }

        rdma_ack_cm_event(event);
    }
fn_fail:
    return NULL;
}

#undef FUNCNAME
#define FUNCNAME get_base_listen_port
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int get_base_listen_port(int pg_rank, int* port)
{
    int mpi_errno = MPI_SUCCESS;
    char* cMaxPort = getenv("MV2_RDMA_CM_MAX_PORT");
    int maxPort = MV2_RDMA_CM_MAX_PORT_LIMIT;
    MPIDI_STATE_DECL(MPID_STATE_GET_BASE_LISTEN_PORT);
    MPIDI_FUNC_ENTER(MPID_STATE_GET_BASE_LISTEN_PORT);

    if (cMaxPort)
    {
        maxPort = atoi(cMaxPort);

        if (maxPort > MV2_RDMA_CM_MAX_PORT_LIMIT || 
            maxPort < MV2_RDMA_CM_MIN_PORT_LIMIT)
        {
            MPIR_ERR_SETANDJUMP3(
                mpi_errno,
                MPI_ERR_OTHER,
                "**rdmacmmaxport",
                "**rdmacmmaxport %d %d %d",
                maxPort,
                MV2_RDMA_CM_MIN_PORT_LIMIT,
                MV2_RDMA_CM_MAX_PORT_LIMIT
            );
        }
    }

    char* cMinPort = getenv("MV2_RDMA_CM_MIN_PORT");
    int minPort = MV2_RDMA_CM_MIN_PORT_LIMIT;

    if (cMinPort)
    {
        minPort = atoi(cMinPort);

        if (minPort > MV2_RDMA_CM_MAX_PORT_LIMIT || 
            minPort < MV2_RDMA_CM_MIN_PORT_LIMIT)
        {
            MPIR_ERR_SETANDJUMP3(
                mpi_errno,
                MPI_ERR_OTHER,
                "**rdmacmminport",
                "**rdmacmminport %d %d %d",
                minPort,
                MV2_RDMA_CM_MIN_PORT_LIMIT,
                MV2_RDMA_CM_MAX_PORT_LIMIT
            );
        }
    }

    int portRange = MPIDI_PG_Get_size(MPIDI_Process.my_pg) - g_num_smp_peers;
    PRINT_DEBUG(DEBUG_RDMACM_verbose,"%s: portRange = %d\r\n", __FUNCTION__, portRange);

    if (maxPort - minPort < portRange)
    {
        MPIR_ERR_SETANDJUMP2(
            mpi_errno,
            MPI_ERR_OTHER,
            "**rdmacmportrange",
            "**rdmacmportrange %d %d",
            maxPort - minPort,
            portRange
        );
    }

    struct timeval seed;
    gettimeofday(&seed, NULL);
    char* envPort = getenv("MV2_RDMA_CM_PORT");
    int rdma_cm_default_port;

    if (envPort)
    {
        rdma_cm_default_port = atoi(envPort);

        if (rdma_cm_default_port == -1)
        {
            srand(seed.tv_usec);    /* Random seed for the port */
            rdma_cm_default_port = (rand() % (maxPort - minPort + 1)) + minPort;
        }
        else if (rdma_cm_default_port > maxPort || 
            rdma_cm_default_port <= minPort)
        {
            MPIR_ERR_SETANDJUMP1(
                mpi_errno,
                MPI_ERR_OTHER,
                "**rdmacminvalidport",
                "**rdmacminvalidport %d",
                atoi(envPort)
            );
        }
    }
    else
    {
        srand(seed.tv_usec);    /* Random seed for the port */
        rdma_cm_default_port = rand() % (maxPort - minPort + 1) + minPort;
    }

    *port = htons(rdma_cm_default_port);

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_GET_BASE_LISTEN_PORT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME bind_listen_port
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int bind_listen_port(int pg_rank, int pg_size)
{
    struct sockaddr_in sin;
    int ret, count = 0;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_BIND_LISTEN_PORT);
    MPIDI_FUNC_ENTER(MPID_STATE_BIND_LISTEN_PORT);

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        struct rdma_addrinfo *rdma_addr = NULL, hints;
        char rdma_cm_ipv6_addr[128];

        if (!inet_ntop(AF_INET6, mv2_MPIDI_CH3I_RDMA_Process.gids[0][0].raw,
                        rdma_cm_ipv6_addr, sizeof(rdma_cm_ipv6_addr))) {
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not convert local GID to IPv6 address\n");
        }

        MPIU_Memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_IB;
        hints.ai_port_space = RDMA_PS_TCP;
        hints.ai_flags = RAI_NUMERICHOST | RAI_FAMILY | RAI_PASSIVE;

        if (rdma_getaddrinfo(rdma_cm_ipv6_addr, NULL, &hints, &rdma_addr)) {
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not get transport independent address translation\n");
        }
        if (rdma_bind_addr(proc->cm_listen_id, rdma_addr->ai_src_addr)) {
            rdma_freeaddrinfo(rdma_addr);
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Binding to local IPv6 address failed\n");
        }
        rdma_base_listen_sid[pg_rank] = ((struct sockaddr_ib *)
                                         (&proc->cm_listen_id->route.addr.src_addr))->sib_sid;
        rdma_freeaddrinfo(rdma_addr);
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        mpi_errno = get_base_listen_port(pg_rank, &rdma_base_listen_port[pg_rank]);
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }

        MPIU_Memset(&sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_addr.s_addr = 0;
        sin.sin_port = rdma_base_listen_port[pg_rank];

        ret = rdma_bind_addr(proc->cm_listen_id, (struct sockaddr *) &sin);

        while (ret)
        {
            if ((mpi_errno = get_base_listen_port(pg_rank,
                            &rdma_base_listen_port[pg_rank])) != MPI_SUCCESS)
            {
                MPIR_ERR_POP(mpi_errno);
            }

            sin.sin_port = rdma_base_listen_port[pg_rank];
            ret = rdma_bind_addr(proc->cm_listen_id, (struct sockaddr *) &sin);
            PRINT_DEBUG(DEBUG_RDMACM_verbose,"[%d] Port bind failed - %d. retrying %d\n",
                        pg_rank, rdma_base_listen_port[pg_rank], count++);
            if (count > 1000){
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "rdma_bind_addr failed: %d\n", ret);
            }
        }
    }

    ret = rdma_listen(proc->cm_listen_id, 2 * (pg_size) * rdma_num_rails);
    if (ret) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**fail", "rdma_listen failed: %d\n", ret);
    }

    PRINT_DEBUG(DEBUG_RDMACM_verbose,"Listen port bind on %d\n", sin.sin_port);

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_BIND_LISTEN_PORT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME ib_init_rdma_cm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int ib_init_rdma_cm(struct mv2_MPIDI_CH3I_RDMA_Process_t *proc,
                int pg_rank, int pg_size)
{
    int i = 0, ret, num_interfaces;
    int mpi_errno = MPI_SUCCESS;
    char *value;
    MPIDI_STATE_DECL(MPID_STATE_IB_INIT_RDMA_CM);
    MPIDI_FUNC_ENTER(MPID_STATE_IB_INIT_RDMA_CM);

    if(sem_init(&(proc->rdma_cm), 0, 0)) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
            "sem_init", strerror(errno));
    }

    if(sem_init(&(rdma_cm_addr), 0, 0)) {
        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail", "%s: %s",
        "sem_init", strerror(errno));
    }

    if (!(proc->cm_channel = rdma_create_event_channel()))
    {
        MPIR_ERR_SETFATALANDJUMP1(
            mpi_errno,
            MPI_ERR_OTHER,
            "**fail",
            "**fail %s",
            "Cannot create rdma_create_event_channel."
        );
    }

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        rdma_base_listen_sid = (uint64_t*) MPIU_Malloc (pg_size * sizeof(uint64_t));
        if (!rdma_base_listen_sid) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        rdma_base_listen_port = (int *) MPIU_Malloc (pg_size * sizeof(int));
        if (!rdma_base_listen_port) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }
    }
    rdma_cm_connect_count = (int *) MPIU_Malloc (pg_size * sizeof(int));
    rdma_cm_accept_count = (int *) MPIU_Malloc (pg_size * sizeof(int));
    rdma_cm_iwarp_msg_count = (int *) MPIU_Malloc (pg_size * sizeof(int));

    if (!rdma_cm_connect_count
        || !rdma_cm_accept_count
        || !rdma_cm_iwarp_msg_count) {

        MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
    }
    for (i = 0; i < pg_size; i++) {
        rdma_cm_connect_count[i] = 0;
        rdma_cm_accept_count[i] = 0;
        rdma_cm_iwarp_msg_count[i] = 0;
    }

    for (i = 0; i < rdma_num_hcas; i++){
        proc->cq_hndl[i] = NULL;
        proc->send_cq_hndl[i] = NULL;
        proc->recv_cq_hndl[i] = NULL;
    }

    if ((value = getenv("MV2_RDMA_CM_ARP_TIMEOUT")) != NULL) {
        rdma_cm_arp_timeout = atoi(value);
        if (rdma_cm_arp_timeout < 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                 "**fail %s", "Invalid rdma cm arp timeout value specified\n");
        }
    }

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        int j = 0;
        for (i = 0; i < rdma_num_hcas; i++) {
            for (j = 1; j <= rdma_num_ports; j++) {
                if (ibv_query_gid(mv2_MPIDI_CH3I_RDMA_Process.nic_context[i], j,
                                    rdma_default_gid_index,
                                    &mv2_MPIDI_CH3I_RDMA_Process.gids[i][j-1])) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                            "**fail",
                            "Failed to retrieve gid on rank %d",
                            pg_rank);
                }
            }
        }
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        /* Init. list of local IPs to use */
        mpi_errno = rdma_cm_get_local_ip(&num_interfaces);

        if (num_interfaces < rdma_num_hcas * rdma_num_ports){
            MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "**fail %s",
                    "Not enough interfaces (ip addresses) "
                    "specified in /etc/mv2.conf\n");
        }
    }

    /* Create the listen cm_id */
    ret = rdma_create_id(proc->cm_channel, 
        &proc->cm_listen_id, proc, RDMA_PS_TCP);
    if (ret) {
        MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                "**fail", "Could not create listen cm_id, "
                "rdma_create_id error: %d\n", ret);
    }

    /* Create the connection management thread */
    pthread_create(&proc->cmthread, NULL, cm_thread, NULL);

    /* Find a base port, relay it to the peers and listen */
    if((mpi_errno = bind_listen_port(pg_rank, pg_size)) != MPI_SUCCESS)
    {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Create CQ and PD */
    rdma_cm_init_pd_cq();

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_IB_INIT_RDMA_CM);
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_connect_all
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_connect_all(int pg_rank, MPIDI_PG_t *pg)
{
    int i, j, k, rail_index, pg_size;
    MPIDI_VC_t  *vc;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    int max_num_ips = rdma_num_hcas * rdma_num_ports;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_CONNECT_ALL);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_CONNECT_ALL);

    if (!proc->use_rdma_cm_on_demand){
        pg_size = MPIDI_PG_Get_size(pg);
        /* Identify number of connections to wait for */
        for (i = 0; i < pg_size; i++) {
            MPIDI_PG_Get_vc(pg, i, &vc);
            if (qp_required(vc, pg_rank, i)) {
                rdma_cm_num_expected_connections++;
            }
        }

        /* Initiate active connect requests */
        for (i = 0; i <= pg_rank; i++){
            MPIDI_PG_Get_vc(pg, i, &vc);
            if (qp_required(vc, pg_rank, i)) {
                vc->ch.state = MPIDI_CH3I_VC_STATE_CONNECTING_CLI;
                vc->state = MPIDI_VC_STATE_LOCAL_ACTIVE;
                PRINT_DEBUG(DEBUG_CM_verbose>0, "Current state of %d is %s.\n",
                            vc->pg_rank, MPIDI_VC_GetStateString(vc->state));

                /* Initiate all needed qp connections */
                for (j = 0; j < rdma_num_hcas*rdma_num_ports; j++){
                    for (k = 0; k < rdma_num_qp_per_port; k++){
                        rail_index = j * rdma_num_qp_per_port + k;
                        mpi_errno = rdma_cm_connect_to_server(vc, 
                            (i*max_num_ips + j), rail_index);
                        if (mpi_errno) MPIR_ERR_POP (mpi_errno);
                    }
                }
            }
        }
    
        /* Wait for all connections to complete */
        if (rdma_cm_num_expected_connections > 0)
            sem_wait(&proc->rdma_cm);

        /* RDMA CM Connection Setup Complete */
        PRINT_DEBUG(DEBUG_RDMACM_verbose,"RDMA CM based connection setup complete\n");
    }

fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_CONNECT_ALL);
    return mpi_errno;
fn_fail:
   goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_get_contexts
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_get_contexts() {
    int mpi_errno = MPI_SUCCESS;
    int i = 0, ret;
    struct sockaddr_in sin;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_GET_CONTEXTS);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_GET_CONTEXTS);

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        int j = 0;
        for (i = 0; i < rdma_num_hcas; ++i) {
            for (j = 0; j < rdma_num_ports; ++j) {
                struct rdma_addrinfo *rdma_addr = NULL, hints;
                char rdma_cm_ipv6_addr[128];

                ret = rdma_create_id(proc->cm_channel, &tmpcmid, proc, RDMA_PS_TCP);
                if (ret) {
                    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "rdma_create_id error %d\n", ret);
                }

                if (!inet_ntop(AF_INET6, proc->gids[i][j].raw,
                                rdma_cm_ipv6_addr, sizeof(rdma_cm_ipv6_addr))) {
                    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not convert local GID to IPv6 address\n");
                }

                MPIU_Memset(&hints, 0, sizeof(hints));
                hints.ai_family = AF_IB;
                hints.ai_port_space = RDMA_PS_TCP;
                hints.ai_flags = RAI_NUMERICHOST | RAI_FAMILY;

                if (rdma_getaddrinfo(rdma_cm_ipv6_addr, NULL, &hints, &rdma_addr)) {
                    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not get transport independent address translation\n");
                }
                ((struct sockaddr_ib *) (rdma_addr->ai_dst_addr))->sib_sid =
                 ((struct sockaddr_ib *) (&tmpcmid->route.addr.dst_addr))->sib_sid;

                ret = rdma_resolve_addr(tmpcmid, NULL, rdma_addr->ai_dst_addr,
                                        rdma_cm_arp_timeout);
                rdma_freeaddrinfo(rdma_addr);
                if (ret) {
                    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "rdma_resolve_addr error %d\n", ret);
                }

                sem_wait(&rdma_cm_addr);

                proc->nic_context[i] = tmpcmid->verbs;

                rdma_destroy_id(tmpcmid);
                tmpcmid = NULL;
            }
        }
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        for (i = 0; i < rdma_num_hcas*rdma_num_ports; i++) {
            ret = rdma_create_id(proc->cm_channel, &tmpcmid, proc, RDMA_PS_TCP);
            if (ret) {
                MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "rdma_create_id error %d\n", ret);
            }

            MPIU_Memset(&sin, 0, sizeof(sin));
            sin.sin_family = AF_INET;
            sin.sin_addr.s_addr = rdma_cm_local_ips[i];
            ret = rdma_resolve_addr(tmpcmid, NULL, (struct sockaddr *) &sin,
                    rdma_cm_arp_timeout);

            if (ret) {
                MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "rdma_resolve_addr error %d\n", ret);
            }

            sem_wait(&rdma_cm_addr);

            proc->nic_context[i] = tmpcmid->verbs;

            rdma_destroy_id(tmpcmid);
            tmpcmid = NULL;
        }
    }

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_GET_CONTEXTS);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_create_qp
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_create_qp(MPIDI_VC_t *vc, int rail_index)
{
    struct ibv_qp_init_attr init_attr;
    int hca_index, ret;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;
    struct rdma_cm_id *cmid;
    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_CREATE_QP);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_CREATE_QP);

    hca_index = rail_index / (rdma_num_ports * rdma_num_qp_per_port);

    /* Create CM_ID */
    cmid = vc->mrail.rails[rail_index].cm_ids;

    {
        MPIU_Memset(&init_attr, 0, sizeof(init_attr));
        init_attr.cap.max_recv_sge = rdma_default_max_sg_list;
        init_attr.cap.max_send_sge = rdma_default_max_sg_list;
        init_attr.cap.max_inline_data = rdma_max_inline_size;
    
        init_attr.cap.max_send_wr = rdma_default_max_send_wqe;
        if (rdma_iwarp_use_multiple_cq &&
            MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type) &&
            (proc->cluster_size != VERY_SMALL_CLUSTER)) {
            init_attr.send_cq = proc->send_cq_hndl[hca_index];
            init_attr.recv_cq = proc->recv_cq_hndl[hca_index];
        } else {
            init_attr.send_cq = proc->cq_hndl[hca_index];
            init_attr.recv_cq = proc->cq_hndl[hca_index];
        }
        init_attr.qp_type = IBV_QPT_RC;
        init_attr.sq_sig_all = 0;
    }

    /* SRQ based? */
    if (proc->has_srq) {
        init_attr.cap.max_recv_wr = 0;
        init_attr.srq = proc->srq_hndl[hca_index];
    } else {
        init_attr.cap.max_recv_wr = rdma_default_max_recv_wqe;
    }

    ret = rdma_create_qp(cmid, proc->ptag[hca_index], &init_attr);
    if (ret){
        ibv_va_error_abort(IBV_RETURN_ERR,
                "Error creating qp on hca %d using rdma_cm."
                " %d [cmid: %p, pd: %p, send_cq: %p, recv_cq: %p] \n",
                hca_index, ret, cmid, proc->ptag[hca_index],
                proc->send_cq_hndl[hca_index],
                proc->recv_cq_hndl[hca_index]);
    }

    /* Save required handles */
    vc->mrail.rails[rail_index].qp_hndl = cmid->qp;
    if (rdma_iwarp_use_multiple_cq &&
        MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type) &&
        (proc->cluster_size != VERY_SMALL_CLUSTER)) {
       vc->mrail.rails[rail_index].cq_hndl = NULL;
       vc->mrail.rails[rail_index].send_cq_hndl = proc->send_cq_hndl[hca_index];
       vc->mrail.rails[rail_index].recv_cq_hndl = proc->recv_cq_hndl[hca_index];
    } else {
       vc->mrail.rails[rail_index].cq_hndl = proc->cq_hndl[hca_index];
       vc->mrail.rails[rail_index].send_cq_hndl = NULL;
       vc->mrail.rails[rail_index].recv_cq_hndl = NULL;
    }

    vc->mrail.rails[rail_index].nic_context = cmid->verbs;
    vc->mrail.rails[rail_index].hca_index = hca_index;
    vc->mrail.rails[rail_index].port = 1;

    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_CREATE_QP);
    return ret;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_exchange_hostid
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_exchange_hostid(MPIDI_PG_t *pg, int pg_rank, int pg_size)
{
    int *hostid_all;
    int i, mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_EXCHANGE_HOSTID);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_EXCHANGE_HOSTID);

    hostid_all = (int *) MPIU_Malloc (pg_size * sizeof(int));
    if (!hostid_all){
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
    }
    
    memset(mv2_pmi_key, 0, mv2_pmi_max_keylen);
    MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "HOST-%d", pg_rank);

    hostid_all[pg_rank] = gethostid();
    sprintf(mv2_pmi_val, "%d", hostid_all[pg_rank] );

    mpi_errno = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_put",
                                    "**pmi_kvs_put %d", mpi_errno);
    }

    mpi_errno = UPMI_KVS_COMMIT(pg->ch.kvs_name);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_commit",
                                    "**pmi_kvs_commit %d", mpi_errno);
    }

    mpi_errno = UPMI_BARRIER();
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_barrier",
                "**pmi_barrier %d", mpi_errno);
    }

    for (i = 0; i < pg_size; i++){    
        if(i != pg_rank) {
            MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "HOST-%d", i);
            mpi_errno = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
            if (mpi_errno != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_get",
                        "**pmi_kvs_get %d", mpi_errno);
            }
            sscanf(mv2_pmi_val, "%d", &hostid_all[i]);
        }
    }

    rdma_process_hostid(pg, hostid_all, pg_rank, pg_size);

fn_fail:
    MPIU_Free(hostid_all);

    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_EXCHANGE_HOSTID);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_get_hostnames
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_get_hostnames(int pg_rank, MPIDI_PG_t *pg)
{
    int *hosts = NULL;
    int i = 0, j = 0;
    int mpi_errno = MPI_SUCCESS;
    char *temp = NULL;
    int length = 0;
    char rank[16];
    char *buffer = NULL;
    int pg_size = MPIDI_PG_Get_size(pg);
    int max_num_ips = rdma_num_hcas * rdma_num_ports; 

    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_GET_HOSTNAMES);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_GET_HOSTNAMES);

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        rdma_cm_host_gid_list = (union ibv_gid*)
                    MPIU_Malloc(pg_size * max_num_ips * sizeof(union ibv_gid));

        length = 128*rdma_num_hcas*rdma_num_ports + 16;
        buffer = MPIU_Malloc(sizeof(char)*length);
        if (!rdma_cm_host_gid_list || !buffer) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        hosts = (int *) MPIU_Malloc (pg_size * max_num_ips * sizeof(int));
        rdma_cm_host_list = hosts;

        length = 32*rdma_num_hcas*rdma_num_ports;
        buffer = MPIU_Malloc(sizeof(char)*length);
        if (!hosts || !buffer){
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }
    }
    
    sprintf(rank, "ip-%d", pg_rank);
#ifdef _MULTI_SUBNET_SUPPORT_
    int k = 0;
    if (mv2_rdma_cm_multi_subnet_support) {
        sprintf(buffer, "%016lx", rdma_base_listen_sid[pg_rank]);
        for(i = 0; i < rdma_num_hcas; i++) {
            for(j = 1; j <= rdma_num_ports; j++) {
                sprintf(buffer+strlen(buffer), "-%016" SCNx64 ":%016" SCNx64,
                        mv2_MPIDI_CH3I_RDMA_Process.gids[i][j-1].global.subnet_prefix,
                        mv2_MPIDI_CH3I_RDMA_Process.gids[i][j-1].global.interface_id);
                MPIU_Memcpy(&rdma_cm_host_gid_list[pg_rank*max_num_ips + k],
                            &mv2_MPIDI_CH3I_RDMA_Process.gids[i][j-1],
                            sizeof(union ibv_gid));
                k++;
            }
        }
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        sprintf(buffer, "%d", rdma_base_listen_port[pg_rank]);
        for(i=0; i<max_num_ips; i++) {
            sprintf( buffer+strlen(buffer), "-%d", rdma_cm_local_ips[i]);
            rdma_cm_host_list[pg_rank*max_num_ips + i] = rdma_cm_local_ips[i];
        }
    }

    PRINT_DEBUG(DEBUG_RDMACM_verbose,"[%d] message to be sent: %s\n", pg_rank, buffer);

    MPIU_Strncpy(mv2_pmi_key, rank, 16);
    MPIU_Strncpy(mv2_pmi_val, buffer, length);
    mpi_errno = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_put",
                                    "**pmi_kvs_put %d", mpi_errno);
    }

    mpi_errno = UPMI_KVS_COMMIT(pg->ch.kvs_name);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_commit",
                                    "**pmi_kvs_commit %d", mpi_errno);
    }

    mpi_errno = UPMI_BARRIER();
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_barrier",
                "**pmi_barrier %d", mpi_errno);
    }

    for (i = 0; i < pg_size; i++) {
        if (i != pg_rank) {
            sprintf(rank, "ip-%d", i);
            MPIU_Strncpy(mv2_pmi_key, rank, 16);
            mpi_errno = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
            if (mpi_errno != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_get",
                        "**pmi_kvs_get %d", mpi_errno);
            }
            MPIU_Strncpy(buffer, mv2_pmi_val, length);

#ifdef _MULTI_SUBNET_SUPPORT_
            if (mv2_rdma_cm_multi_subnet_support) {
                sscanf(buffer, "%016lx", &rdma_base_listen_sid[i]);
            } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
            {
                sscanf(buffer, "%d", &rdma_base_listen_port[i]);
            }
            temp = buffer;
            for(j=0; j<max_num_ips; j++)
            {
                temp = strchr(temp,'-') + 1;
#ifdef _MULTI_SUBNET_SUPPORT_
                if (mv2_rdma_cm_multi_subnet_support) {
                    sscanf(temp, "%016" SCNx64 ":%016" SCNx64,
                            &rdma_cm_host_gid_list[i*max_num_ips + j].global.subnet_prefix,
                            &rdma_cm_host_gid_list[i*max_num_ips + j].global.interface_id);
                } else
#endif /* _MULTI_SUBNET_SUPPORT_ */
                {
                    sscanf(temp, "%d", &rdma_cm_host_list[i*max_num_ips + j]);
                }
            }
        }
    }

    /* Find smp processes */
    if (rdma_use_smp) {
        for (i = 0; i < pg_size; i++){
        if (pg_rank == i)
            continue;
#ifdef _MULTI_SUBNET_SUPPORT_
            if (mv2_rdma_cm_multi_subnet_support) {
                if (!memcmp(&rdma_cm_host_gid_list[i*max_num_ips],
                            &rdma_cm_host_gid_list[pg_rank * max_num_ips],
                            sizeof(union ibv_gid))) {
                    ++g_num_smp_peers;
                }
            } else
#endif /* _MULTI_SUBNET_SUPPORT_ */
            {
                if (hosts[i * max_num_ips] == hosts[pg_rank * max_num_ips])
                    ++g_num_smp_peers;
            }
        }
    }
    PRINT_DEBUG(DEBUG_RDMACM_verbose,"Number of SMP peers for %d is %d\n", pg_rank, 
        g_num_smp_peers);

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_GET_HOSTNAMES);
    MPIU_Free(buffer);

    return mpi_errno;
}

/*
 * Iterate over available interfaces
 * and determine verbs capable ones
 * Exclude Loopback & down interfaces
 */
#undef FUNCNAME
#define FUNCNAME rdma_cm_get_verbs_ip
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_get_verbs_ip(int *num_interfaces)
{
    int index = 0;
    int mpi_errno = MPI_SUCCESS;
    char *value = getenv("MV2_IBA_HCA");
 
    *num_interfaces = num_ip_enabled_devices;
    rdma_cm_local_ips = MPIU_Malloc((rdma_num_hcas*rdma_num_ports) * sizeof(int));
    for(index = 0; index < (*num_interfaces); index++){
        if (value && ip_address_enabled_devices[index].device_name) {
            if(strstr(value, ip_address_enabled_devices[index].device_name) == NULL) {
                continue;
            }
        }
        PRINT_DEBUG(DEBUG_RDMACM_verbose, "Assigning ip of: device %s, ip address %s\n", ip_address_enabled_devices[index].device_name,ip_address_enabled_devices[index].ip_address);
        rdma_cm_local_ips[index] = inet_addr(ip_address_enabled_devices[index].ip_address);
    }


    return mpi_errno;
}

/* Gets the ip address in network byte order */
/*
 * TODO add error handling
 */
#undef FUNCNAME
#define FUNCNAME rdma_cm_get_local_ip
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_get_local_ip(int *num_interfaces)
{
    FILE *fp_port;
    char ip[32];
    char fname[512];
    int i = 0;
    char *value;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_RDMA_CM_GET_LOCAL_IP);
    MPIDI_FUNC_ENTER(MPID_STATE_RDMA_CM_GET_LOCAL_IP);

    value = getenv("MV2_RDMA_CM_CONF_FILE_PATH");

    if (value == NULL) {
        sprintf(fname, "/etc/mv2.conf");
    } else {
        strncpy(fname, value, strlen(value));
        sprintf(fname + strlen(value), "/mv2.conf");
    }

    fp_port = fopen(fname, "r");

    if (NULL == fp_port) {
        PRINT_DEBUG(DEBUG_RDMACM_verbose, "Can't open file: %s, "
                    "trying to determine verbs capable IP\n", fname);
        mpi_errno = rdma_cm_get_verbs_ip(&i);
    } else {
        rdma_cm_local_ips = MPIU_Malloc(rdma_num_hcas*rdma_num_ports*sizeof(int));

        while ((fscanf(fp_port, "%s\n", ip)) != EOF && (i < (rdma_num_hcas*rdma_num_ports))){
            rdma_cm_local_ips[i] = inet_addr(ip);
            i++;
        }
        fclose(fp_port);
    }

    *num_interfaces = i;

    MPIDI_FUNC_EXIT(MPID_STATE_RDMA_CM_GET_LOCAL_IP);
    return mpi_errno;
}

int rdma_cm_connect_to_server(MPIDI_VC_t *vc, int offset, int rail_index){
    int mpi_errno = MPI_SUCCESS;
    struct sockaddr_in sin;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    /* store VC used for connection in the context, 
     * so we get back vc at event callbacks 
     */
    mpi_errno = rdma_create_id(proc->cm_channel, 
        &(vc->mrail.rails[rail_index].cm_ids), vc, RDMA_PS_TCP);
    if (mpi_errno) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "rdma_create_id error %d\n", mpi_errno);
    }

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        struct rdma_addrinfo *src_rdma_addr = NULL, *dst_rdma_addr = NULL, hints;
        char rdma_cm_ipv6_src_addr[128], rdma_cm_ipv6_dst_addr[128];

        /* Address info for source */
        if (!inet_ntop(AF_INET6, proc->gids[0][0].raw,
                        rdma_cm_ipv6_src_addr, sizeof(rdma_cm_ipv6_src_addr))) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not convert local GID to IPv6 address\n");
        }
        MPIU_Memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_IB;
        hints.ai_port_space = RDMA_PS_TCP;
        hints.ai_flags = RAI_NUMERICHOST | RAI_FAMILY | RAI_PASSIVE;

        if (rdma_getaddrinfo(rdma_cm_ipv6_src_addr, NULL, &hints, &src_rdma_addr)) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not get transport independent address translation\n");
        }

        /* Address info for destination */
        if (!inet_ntop(AF_INET6, rdma_cm_host_gid_list[offset].raw,
                        rdma_cm_ipv6_dst_addr, sizeof(rdma_cm_ipv6_dst_addr))) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not convert local GID to IPv6 address\n");
        }
        MPIU_Memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_IB;
        hints.ai_port_space = RDMA_PS_TCP;
        hints.ai_flags = RAI_NUMERICHOST | RAI_FAMILY;
        hints.ai_src_len  = src_rdma_addr->ai_src_len;
        hints.ai_src_addr = src_rdma_addr->ai_src_addr;

        if (rdma_getaddrinfo(rdma_cm_ipv6_dst_addr, NULL, &hints, &dst_rdma_addr)) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                            "Could not get transport independent address translation\n");
        }
        ((struct sockaddr_ib *) (dst_rdma_addr->ai_dst_addr))->sib_sid =
                                            rdma_base_listen_sid[offset];

        mpi_errno = rdma_resolve_addr(vc->mrail.rails[rail_index].cm_ids,
                                dst_rdma_addr->ai_src_addr,
                                dst_rdma_addr->ai_dst_addr,
                                rdma_cm_arp_timeout);
        rdma_freeaddrinfo(src_rdma_addr);
        rdma_freeaddrinfo(dst_rdma_addr);
        PRINT_DEBUG(DEBUG_RDMACM_verbose,"Active connect initiated for %d"
                    " [ip: %s:%016lx] [rail %d]\n", vc->pg_rank, rdma_cm_ipv6_dst_addr,
                    rdma_base_listen_sid[vc->pg_rank], rail_index);
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        /* Resolve addr */
        MPIU_Memset(&sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_addr.s_addr = rdma_cm_host_list[offset];
        sin.sin_port = rdma_base_listen_port[vc->pg_rank];

        mpi_errno = rdma_resolve_addr(vc->mrail.rails[rail_index].cm_ids, NULL,
                                (struct sockaddr *) &sin, rdma_cm_arp_timeout);
        PRINT_DEBUG(DEBUG_RDMACM_verbose,"Active connect initiated for %d"
                    " [ip: %d:%d] [rail %d]\n", vc->pg_rank, rdma_cm_host_list[offset],
                    rdma_base_listen_port[vc->pg_rank], rail_index);
    }
    if (mpi_errno) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "rdma_resolve_addr error %d\n", mpi_errno);
    }

fn_fail:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME rdma_cm_init_pd_cq
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int rdma_cm_init_pd_cq()
{
    mv2_MPIDI_CH3I_RDMA_Process_t* proc = &mv2_MPIDI_CH3I_RDMA_Process;
    int i = 0;
    int pg_rank;
    int mpi_errno = MPI_SUCCESS;

    UPMI_GET_RANK(&pg_rank);
    rdma_cm_get_contexts();

    for (; i < rdma_num_hcas; ++i)
    {
        /* Allocate the protection domain for the HCA */
        proc->ptag[i] = ibv_alloc_pd(proc->nic_context[i]);

        if (!proc->ptag[i]) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                "Failed to allocate pd %d\n", i);
        }

        /* Allocate the completion queue handle for the HCA */
        if(rdma_use_blocking)
        {
            proc->comp_channel[i] =
               ibv_create_comp_channel(proc->nic_context[i]);

            if (!proc->comp_channel[i]) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "**fail %s", "Create comp channel failed\n");
            }

            if (rdma_iwarp_use_multiple_cq &&
                MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type) &&
                (proc->cluster_size != VERY_SMALL_CLUSTER)) {
                /* Allocate the completion queue handle for the HCA */
                /* Trac #423 */
                proc->send_cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    NULL,
                    0);
    
                if (!proc->send_cq_hndl[i]) {
                   /*Falling back to smaller cq size if creation failed*/ 
                    if(rdma_default_max_cq_size > RDMA_DEFAULT_IWARP_CQ_SIZE) {
                       rdma_default_max_cq_size = RDMA_DEFAULT_IWARP_CQ_SIZE;
                       proc->send_cq_hndl[i] = ibv_create_cq(
                                proc->nic_context[i],
                                rdma_default_max_cq_size,
                                NULL,
                                NULL,
                                0);
                      if (!proc->send_cq_hndl[i]) {
                          MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s", "Error allocating CQ");
                      }
                   } else {
                       MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                               "**fail %s", "Error allocating CQ");
                   }
                }
    
                if (ibv_req_notify_cq(proc->send_cq_hndl[i], 0)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "**fail %s", "Request notify for CQ failed\n");
                }

                proc->recv_cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    NULL,
                    0);
    
                if (!proc->recv_cq_hndl[i]) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "**fail %s", "Error allocating CQ");
                }

                if (ibv_req_notify_cq(proc->recv_cq_hndl[i], 0)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                     "**fail %s", "Request notify for CQ failed\n");
                }
            } else {
                proc->cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    proc->comp_channel[i],
                    0);
    
                if (!proc->cq_hndl[i]) {
                    /*Falling back to smaller cq size if creation failed*/
                    if((rdma_default_max_cq_size > RDMA_DEFAULT_IWARP_CQ_SIZE) 
                             && MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type)) {
                        rdma_default_max_cq_size = RDMA_DEFAULT_IWARP_CQ_SIZE;
                        proc->send_cq_hndl[i] = ibv_create_cq(
                                proc->nic_context[i],
                                rdma_default_max_cq_size,
                                NULL,
                                NULL,
                                0);
                        if (!proc->send_cq_hndl[i]) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                "**fail %s", "Error allocating CQ");
                        }
                    } else {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "**fail %s", "Error allocating CQ");
                    }
                }
    
                if (ibv_req_notify_cq(proc->cq_hndl[i], 0)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "**fail %s", "Request notify for CQ failed\n");
                }
            }
        }
        else
        {
            if (rdma_iwarp_use_multiple_cq &&
                MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type) &&
                (proc->cluster_size != VERY_SMALL_CLUSTER)) {
                /* Allocate the completion queue handle for the HCA */
                /* Trac #423*/
                proc->send_cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    NULL,
                    0);
    
                if (!proc->send_cq_hndl[i]) {
                    /*Falling back to smaller cq size if creation failed*/
                    if(rdma_default_max_cq_size > RDMA_DEFAULT_IWARP_CQ_SIZE) {
                        rdma_default_max_cq_size = RDMA_DEFAULT_IWARP_CQ_SIZE;
                        proc->send_cq_hndl[i] = ibv_create_cq(
                                proc->nic_context[i],
                                rdma_default_max_cq_size,
                                NULL,
                                NULL,
                                0);
                        if (!proc->send_cq_hndl[i]) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                    "**fail %s", "Error allocating CQ");
                        }
                    } else {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                "**fail %s", "Error allocating CQ");
                    }
                }
    
                proc->recv_cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    NULL,
                    0);
    
                if (!proc->recv_cq_hndl[i]) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                            "**fail %s", "Error allocating CQ");
                }
            } else {
                proc->cq_hndl[i] = ibv_create_cq(
                    proc->nic_context[i],
                    rdma_default_max_cq_size,
                    NULL,
                    NULL,
                    0);
    
                if (!proc->cq_hndl[i]) {
                    /*Falling back to smaller cq size if creation failed*/
                    if((rdma_default_max_cq_size > RDMA_DEFAULT_IWARP_CQ_SIZE)
                        && MV2_IS_CHELSIO_IWARP_CARD(proc->hca_type)) {
                        rdma_default_max_cq_size = RDMA_DEFAULT_IWARP_CQ_SIZE;
                        proc->cq_hndl[i] = ibv_create_cq(
                                proc->nic_context[i],
                                rdma_default_max_cq_size,
                                NULL,
                                NULL,
                                0);
                        if (!proc->cq_hndl[i]) {
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                    "**fail %s", "Error allocating CQ");
                        }
                    } else {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                "**fail %s", "Error allocating CQ");
                    }
                }
            }
        }

        if (proc->has_srq && !proc->srq_hndl[i])
        {
            proc->srq_hndl[i] = create_srq(proc, i);
        }

        PRINT_DEBUG(DEBUG_RDMACM_verbose,"[%d][rail %d] proc->ptag %p, "
            "proc->cq_hndl %p, proc->srq_hndl %p\n",
            pg_rank, i, proc->ptag[i], proc->cq_hndl[i], proc->srq_hndl[i]);
    }

fn_fail:
    return mpi_errno;
}

int get_remote_rank(struct rdma_cm_id *cmid)
{
    return -1;
}

int get_remote_rail(struct rdma_cm_id *cmid) 
{
    int pg_size, pg_rank, i, rail_index = 0;
    MPIDI_VC_t  *vc = (MPIDI_VC_t *) cmid->context;

    UPMI_GET_SIZE(&pg_size);
    UPMI_GET_RANK(&pg_rank);

    for (i = 0; i < pg_size; i++){
        for (rail_index = 0; rail_index < rdma_num_rails; rail_index++){
            if (cmid == vc->mrail.rails[rail_index].cm_ids)
            return rail_index;
        }
    }
    return -1;
}

void ib_finalize_rdma_cm(int pg_rank, MPIDI_PG_t *pg)
{
    int i, rail_index = 0, pg_size;
    MPIDI_VC_t  *vc;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        MPIU_Free(rdma_base_listen_sid);
    } else
#endif /*_MULTI_SUBNET_SUPPORT_*/
    {
        MPIU_Free(rdma_base_listen_port);
    }
    MPIU_Free(rdma_cm_accept_count);
    MPIU_Free(rdma_cm_connect_count);
    MPIU_Free(rdma_cm_iwarp_msg_count);
    MPIU_Free(rdma_cm_local_ips);
    pg_size = MPIDI_PG_Get_size(pg);

    {

        for (i = 0; i < pg_size; i++) {

            MPIDI_PG_Get_vc(pg, i, &vc); 
            if ((qp_required(vc, pg_rank, i)) &&
                (vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE)) {
                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++){
                    if (vc->mrail.rails[rail_index].cm_ids != NULL) {
                        rdma_disconnect(vc->mrail.rails[rail_index].cm_ids);
                        rdma_destroy_qp(vc->mrail.rails[rail_index].cm_ids);
                    }
                    vc->mrail.rails[rail_index].cm_ids = NULL;
                }
            }
        }
    
        for (i = 0; i < rdma_num_hcas; i++) {
            if (mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i]) {
                ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i]);
                mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i] = NULL;
            }

            if (mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i]) {
                ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i]);
                mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i] = NULL;
            }

            if (mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i]) {
                ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i]);
                mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i] = NULL;
            }

            if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
                /* Signal thread if waiting */
                pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                        srq_post_mutex_lock[i]);
                mv2_MPIDI_CH3I_RDMA_Process.is_finalizing = 1;
                pthread_cond_signal(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
                pthread_mutex_unlock(&mv2_MPIDI_CH3I_RDMA_Process.
                        srq_post_mutex_lock[i]);

                /* wait for async thread to finish processing */
                pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                        async_mutex_lock[i]);

                /* destroy mutex and cond and cancel thread */
                pthread_cond_destroy(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
                pthread_mutex_destroy(&mv2_MPIDI_CH3I_RDMA_Process.
                        srq_post_mutex_lock[i]);

                if (mv2_MPIDI_CH3I_RDMA_Process.async_thread[i]) {
                    pthread_cancel(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i]);
                    pthread_join(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i], NULL);
                    mv2_MPIDI_CH3I_RDMA_Process.async_thread[i] = 0;
                }
                if (mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i]){
                    ibv_destroy_srq(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i]);
                    mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i] = NULL;
                }
            }
            if(rdma_use_blocking) {
                ibv_destroy_comp_channel(
                    mv2_MPIDI_CH3I_RDMA_Process.comp_channel[i]);
                mv2_MPIDI_CH3I_RDMA_Process.comp_channel[i] = NULL;
            }
            deallocate_vbufs(i);
        }

        for (i = 0; i < pg_size; i++){
            MPIDI_PG_Get_vc(pg, i, &vc);
            if ((qp_required(vc, pg_rank, i)) &&
                (vc->ch.state == MPIDI_CH3I_VC_STATE_IDLE)) {
                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++){
                    if (vc->mrail.rails[rail_index].cm_ids != NULL) {
                        rdma_destroy_id(vc->mrail.rails[rail_index].cm_ids);
                        vc->mrail.rails[rail_index].cm_ids = NULL;
                    }
                }
            }
            MPIDI_CH3_VC_Destroy(vc);
        }
    }

    if (proc->cm_listen_id) {
        rdma_destroy_id(proc->cm_listen_id);
        rdma_cm_finalized = 1;
        rdma_destroy_event_channel(mv2_MPIDI_CH3I_RDMA_Process.cm_channel);

        pthread_cancel(proc->cmthread);
        pthread_join(proc->cmthread, NULL);
        proc->cm_listen_id = NULL;
    }

    PRINT_DEBUG(DEBUG_RDMACM_verbose,"RDMA CM resources finalized\n");
}


#endif /* RDMA_CM */
