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

#if defined(_MCST_SUPPORT_)
#include <infiniband/verbs.h>
#include <infiniband/umad.h>
#include <infiniband/mad.h>
#if defined(RDMA_CM)
#include <infiniband/ib.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <rdma/rdma_cma.h>
#endif /* #if defined RDMA_CM */
#include <string.h>
#include <errno.h>
#include "rdma_impl.h"
#include "ibv_mcast.h"
#include "vbuf.h"
#include "debug_utils.h"

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);


extern mv2_MPIDI_CH3I_RDMA_Process_t mv2_MPIDI_CH3I_RDMA_Process;
/* TODO : replace with hash table */
MPID_Comm *comm_table[MV2_MCAST_MAX_COMMS];
mcast_context_t *mcast_ctx = NULL;

MPID_Comm *mv2_mcast_find_comm(int comm_id)
{
    return comm_table[comm_id];
}

static void mv2_mcast_register_comm(MPID_Comm * comm_ptr, int comm_id)
{
    MPIU_Assert(comm_table[comm_id] == NULL);
    comm_table[comm_id] = comm_ptr;
    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "register comm_id :%d comm:%p\n",
                comm_id, comm_ptr);
}

static void mv2_mcast_unregister_comm(int comm_id)
{
    if (comm_table[comm_id] == NULL) {
        return;
    }
    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "unregister comm_id :%d comm:%p\n",
                comm_id, comm_table[comm_id]);
    comm_table[comm_id] = NULL;
}

void mv2_mcast_add_comm_init_req(int comm_id, int comm_size)
{
    mcast_init_elem_t *init_elem;
    int mask_size;

    /* initialize the element */
    init_elem = (mcast_init_elem_t *) MPIU_Malloc(sizeof(mcast_init_elem_t));
    init_elem->next = NULL;
    init_elem->status = MCAST_COMM_INACTIVE;
    init_elem->comm_id = comm_id;
    init_elem->init_timer = mv2_get_time_us();
    init_elem->init_retries = 0;
    init_elem->acks_pending = comm_size - 1;
    mask_size = ((sizeof(unsigned char) * 8 - 1) + (comm_size - 1)) /
        (sizeof(unsigned char) * 8);
    init_elem->init_ack_mask = (unsigned char *) MPIU_Malloc(mask_size);
    MPIU_Memset(init_elem->init_ack_mask, 0, mask_size);

    /* insert at the beginig  */
    init_elem->next = mcast_ctx->init_list;
    mcast_ctx->init_list = init_elem;
    init_elem->status = MCAST_COMM_INITIALIZING;
    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Start MCAST Comm init comm_id:%d\n",
                comm_id);
}

void mv2_mcast_remove_comm_init_req(mcast_init_elem_t * init_elem)
{

    mcast_init_elem_t *curr = NULL, *prev = NULL;
    for (curr = mcast_ctx->init_list; curr != NULL; prev = curr, curr = curr->next) {
        if (curr == init_elem) {
            if (prev == NULL) {
                mcast_ctx->init_list = curr->next;
            } else {
                prev->next = curr->next;
            }
            break;
        }
    }
    MPIU_Assert(curr);
    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "End MCAST Comm init comm_id:%d\n",
                init_elem->comm_id);

    MPIU_Free(init_elem->init_ack_mask);
    MPIU_Free(curr);
}

static inline void vbuf_init_mcast_send(vbuf * v, unsigned long len, int hca_num,
                                        mcast_info_t * minfo)
{
    MPIU_Assert(v != NULL);

    v->desc.u.sr.next = NULL;
    v->desc.u.sr.send_flags = IBV_SEND_SIGNALED;
    if (len <= rdma_max_inline_size) {
        v->desc.u.sr.send_flags |= IBV_SEND_INLINE;
    }
    v->desc.u.sr.opcode = IBV_WR_SEND;
    v->desc.u.sr.wr_id = (uintptr_t) v;
    v->desc.u.sr.num_sge = 1;
    v->desc.u.sr.sg_list = &(v->desc.sg_entry);
    v->desc.u.sr.wr.ud.ah = minfo->ah;
    v->desc.u.sr.wr.ud.remote_qpn = MCAST_QP;
    v->desc.u.sr.wr.ud.remote_qkey = MCAST_DEF_QKEY;
    v->desc.sg_entry.length = len;
    v->desc.sg_entry.lkey = v->region->mem_handle[hca_num]->lkey;
    v->desc.sg_entry.addr = (uintptr_t) (v->buffer);

    v->padding = NORMAL_VBUF_FLAG;
    v->rail = hca_num;
}

static void mv2_mcast_send_comm_init(mcast_init_elem_t * elem, int rail)
{
    MPID_Comm *comm_ptr;
    vbuf *v;
    mcast_info_t *minfo;

    comm_ptr = mv2_mcast_find_comm(elem->comm_id);
    if (comm_ptr == NULL) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                    "Didn't find comm ptr. remove init info element \n");
        mv2_mcast_remove_comm_init_req(elem);
        return;
    }
    PRINT_DEBUG(DEBUG_MCST_verbose > 1, "sending on rail %d\n",rail);
    minfo = &((bcast_info_t *) comm_ptr->dev.ch.bcast_info)->minfo;

    MV2_GET_AND_INIT_UD_VBUF(v);
    MPIDI_CH3_Pkt_mcast_init_t *p = (MPIDI_CH3_Pkt_mcast_init_t *) v->pheader;
    p->type = MPIDI_CH3_PKT_MCST_INIT;
    p->rail = rail;
    p->psn = 0;
    p->comm_id = elem->comm_id;
    p->src_rank = comm_ptr->rank;

    vbuf_init_mcast_send(v, sizeof(MPIDI_CH3_Pkt_mcast_init_t), rail, minfo);

    IBV_POST_MCAST_SEND(v, mcast_ctx);
    PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                "MCAST send init elem:%p comm_id:%d\n", elem, elem->comm_id);
}

void mv2_mcast_handle_init_ack(MPIDI_CH3_Pkt_mcast_init_ack_t * p)
{
    int comm_id, leader_rank;
    mcast_init_elem_t *curr = mcast_ctx->init_list;

    PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                "received mcast init ack from:%d \n", p->src_rank);
    comm_id = p->comm_id;
    leader_rank = p->src_rank;

    while (curr) {
        if (curr->comm_id == comm_id) {
            break;
        }
        curr = curr->next;
    }

    if (curr == NULL) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                    "Mcast init ack received. comm not found id:%d\n", comm_id);
        return;
    }

    curr->init_timer = mv2_get_time_us();

    if (MV2_CHAR_ISBITSET(curr->init_ack_mask, leader_rank)) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Mcast init ack is already received \n");
        return;
    }

    MV2_CHAR_SETBIT(curr->init_ack_mask, leader_rank);
    curr->acks_pending--;
    if (curr->acks_pending <= 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Mcast init recieved all acks \n");
        curr->status = MCAST_COMM_ACTIVE;
    }
}

void mv2_mcast_process_comm_init_req(mcast_init_elem_t * list)
{
    mcast_init_elem_t *curr = NULL, *next = NULL;
    curr = next = list;
    long now = mv2_get_time_us();
    while (curr) {
        next = curr->next;
        if (curr->status == MCAST_COMM_INITIALIZING &&
            now > (curr->init_timer + mcast_comm_init_timeout)) {
            curr->init_retries++;
            if (curr->init_retries > mcast_comm_init_retries) {
                PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                            "MCAST process Comm init failed comm_id:%d retries :%d\n",
                            curr->comm_id, curr->init_retries);
                curr->status = MCAST_COMM_FAILED;
                curr = next;
                continue;
            }

            curr->init_timer = now;
            PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                        "MCAST process Comm init comm_id:%d\n", curr->comm_id);
            mv2_mcast_send_comm_init(curr, mcast_ctx->selected_rail);
        }
        curr = next;
    }
}

int mv2_mcast_progress_comm_ready(MPID_Comm * comm_ptr)
{
    mcast_info_t *minfo = NULL;
    mcast_init_elem_t *curr = NULL;
    mcast_comm_status_t status;
    int comm_id = 0;

    minfo = &(((bcast_info_t *) (comm_ptr->dev.ch.bcast_info))->minfo);
    comm_id = minfo->grp_info.comm_id;
    curr = mcast_ctx->init_list;
    while (curr) {
        if (curr->comm_id == comm_id) {
            break;
        }
        curr = curr->next;
    }
    MPIU_Assert(curr);
    if (curr == NULL) {
        return 0;
    }
    while (curr->status == MCAST_COMM_INITIALIZING) {
        MPIDI_CH3I_Progress(FALSE, NULL);
    }
    status = curr->status;
    mv2_mcast_remove_comm_init_req(curr);
    return (status == MCAST_COMM_ACTIVE) ? 1 : 0;
}

static inline int mv2_prepare_mcast_mad(void *umad_buf, uint8_t method,
                                        ib_portid_t * dport, union ibv_gid *port_gid,
                                        union ibv_gid *mgid, uint64_t trid,
                                        uint64_t comp_mask)
{
    uint8_t mad_data[IB_SA_DATA_SIZE];
    ib_rpc_t rpc;

    memset(mad_data, 0, IB_SA_DATA_SIZE);
    mad_set_array(mad_data, 0, IB_SA_MCM_MGID_F, mgid->raw);
    mad_set_array(mad_data, 0, IB_SA_MCM_PORTGID_F, port_gid->raw);
    mad_set_field(mad_data, 0, IB_SA_MCM_JOIN_STATE_F, 1);
    mad_set_field(mad_data, 0, IB_SA_MCM_PROXY_JOIN_F, 1);
    mad_set_field(mad_data, 0, IB_SA_MCM_QKEY_F, MCAST_DEF_QKEY);
    mad_set_field(mad_data, 0, IB_SA_MCM_PKEY_F, 0xffff);

    memset(&rpc, 0, sizeof(rpc));
    rpc.mgtclass = IB_SA_CLASS;
    rpc.attr.id = IB_SA_ATTR_MCRECORD;
    rpc.datasz = IB_SA_DATA_SIZE;
    rpc.dataoffs = IB_SA_DATA_OFFS;
    rpc.method = method;
    rpc.attr.mod = 0;
    rpc.mask = comp_mask;
    rpc.trid = trid;

    mad_build_pkt(umad_buf, &rpc, dport, NULL, mad_data);

    return MCAST_SUCCESS;
}

static inline int mv2_recv_umad_response(int mad_portid, char *umad_buf,
                                         int trid_index, uint64_t trid,
                                         uint64_t trid_max, unsigned char *recv_mask)
{
    char *mad = NULL;
    uint64_t expected_trid, recv_trid;
    int ret = MCAST_SUCCESS;
    int len = IB_MAD_SIZE;
    int recv_retry_count = 0;
    int status = 0;

    /* wait for first response to get mgid */
    mad = umad_get_mad(umad_buf);
    expected_trid = trid + trid_index;
    while (1) {
        ret = umad_recv(mad_portid, umad_buf, &len, 5000);
        if (ret < 0) {
            if (errno == ENOSPC) {
                // TODO : handle this error case
                PRINT_DEBUG(DEBUG_MCST_verbose > 1,"recv buffer is not sufficitient \n");
            }
            goto fn_fail;
        }

        status = mad_get_field(mad, 0, IB_MAD_STATUS_F);
        if (status == 0) {
            /* recv success */
            int recv_index;
            recv_trid = (uint32_t) mad_get_field64(mad, 0, IB_MAD_TRID_F);
            MPIU_Assert(recv_trid >= trid && recv_trid <= trid_max);
            recv_index = recv_trid - trid + 1;
            MV2_CHAR_SETBIT(recv_mask, recv_index);
            if (recv_trid == expected_trid) {
                break;
            }
            PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Out of order recv index:%d\n",
                        recv_index);
        } else {
            /* retry recv again */
            recv_retry_count++;
            if (recv_retry_count > MCAST_MAX_UMAD_RETRIES) {
                PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                            "Umad recv failed retry_count:%d \n", recv_retry_count);
                goto fn_fail;
            }
        }
    }
    return MCAST_SUCCESS;
  fn_fail:
    return MCAST_FAILURE;
}

static inline int mv2_op_mcast_group(mcast_init_info_t * all_init_info,
                                     int num_ranks, int method, union ibv_gid *mgid,
                                     uint16_t * mlid)
{
    char *ib_dev;
    int mad_portid = -1;
    int sa_agentid = -1;
    int umad_buf_size;
    char *umad_send_buf = NULL, *umad_recv_buf = NULL;
    char *mad = NULL;
    int ret = MCAST_SUCCESS;
    int status = 0, mask_size, i, recv_index = 0;
    union ibv_gid port_gid;
    unsigned char *recv_mask = NULL;
    ib_portid_t dport;
    struct ibv_port_attr port_attr;
    uint64_t trid, create_mask, join_mask, delete_mask, comp_mask;

    if (method == IB_MAD_METHOD_SET) {
        create_mask = SUBN_ADM_COMPMASK_MGID | SUBN_ADM_COMPMASK_PORT_GID |
            SUBN_ADM_COMPMASK_QKEY | SUBN_ADM_COMPMASK_PKEY |
            SUBN_ADM_COMPMASK_TCLASS | SUBN_ADM_COMPMASK_SL | SUBN_ADM_COMPMASK_FLOW
            | SUBN_ADM_COMPMASK_JOIN_STATE;
        join_mask =
            SUBN_ADM_COMPMASK_MGID | SUBN_ADM_COMPMASK_PORT_GID |
            SUBN_ADM_COMPMASK_JOIN_STATE;
    } else if (method == IB_MAD_METHOD_DELETE) {
        delete_mask = SUBN_ADM_COMPMASK_MGID | SUBN_ADM_COMPMASK_PORT_GID |
            SUBN_ADM_COMPMASK_JOIN_STATE;

    }


    if (umad_init() < 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"UMAD Init failed\n");
        goto fn_fail;
    }

    ib_dev = (char *) ibv_get_device_name(mv2_MPIDI_CH3I_RDMA_Process.ib_dev[0]);

    mad_portid = umad_open_port(ib_dev, mv2_MPIDI_CH3I_RDMA_Process.ports[0][0]);
    if (mad_portid < 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"UMAD open port id failed\n");
        goto fn_fail;
    }

    memset(&dport, 0, sizeof(ib_portid_t));
    if (ibv_query_port(mv2_MPIDI_CH3I_RDMA_Process.nic_context[0],
                       mv2_MPIDI_CH3I_RDMA_Process.ports[0][0], &port_attr)) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"ibv_query_port failed \n");
        return MCAST_FAILURE;
    }
    dport.lid = port_attr.sm_lid;
    dport.qp = 1;
    dport.qkey = IB_DEFAULT_QP1_QKEY;

    sa_agentid = umad_register(mad_portid, IB_SA_CLASS, SA_CLASS_VERSION, 0, 0);
    if (sa_agentid < 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1," UMAD register failed \n");
        goto fn_fail;
    }

    umad_buf_size = umad_size() + IB_MAD_SIZE;
    umad_send_buf = umad_alloc(1, umad_buf_size);
    if (!umad_send_buf) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"UMAD send buffer allocation failed\n");
        goto fn_fail;
    }
    if ((umad_recv_buf = umad_alloc(1, umad_buf_size)) == NULL) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"UMAD recv buffer allocation failed\n");
        goto fn_fail;
    }

    MPIU_Memset(umad_recv_buf, 0, umad_buf_size);

    trid = (uint32_t) rand();
    mask_size = ((sizeof(unsigned char) * 8 - 1) + num_ranks) /
        (sizeof(unsigned char) * 8);
    recv_mask = (unsigned char *) MPIU_Malloc(mask_size);
    MPIU_Memset(recv_mask, 0, mask_size);

    for (i = 0; i < num_ranks; i++) {

        MPIU_Memset(umad_send_buf, 0, umad_buf_size);
        MPIU_Memcpy(port_gid.raw, &all_init_info[i], 16);
        if (method == IB_MAD_METHOD_SET) {
            comp_mask = (i == 0) ? create_mask : join_mask;
        } else {
            comp_mask = delete_mask;
        }
        mv2_prepare_mcast_mad(umad_send_buf, method,
                              &dport, &port_gid, mgid, trid + i, comp_mask);

        if (umad_send(mad_portid, sa_agentid, umad_send_buf,
                      IB_MAD_SIZE, 100, 5) < 0) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 1,"UMAD Send failed \n");
            goto fn_fail;
        }

        if (i == 0 && method == IB_MAD_METHOD_SET) {
            ret = mv2_recv_umad_response(mad_portid, umad_recv_buf, i,
                                         trid, trid + num_ranks, recv_mask);
            if (ret != MCAST_SUCCESS) {
                goto cleanup;
            }
            /* extract mgid, mlid from first response */
            mad = umad_get_mad(umad_recv_buf);
            status = mad_get_field(mad, 0, IB_MAD_STATUS_F);
            if (status == 0) {
                /* recv success */
                if (method == IB_MAD_METHOD_SET) {
                    *mlid =
                        (uint16_t) mad_get_field(mad, IB_SA_DATA_OFFS,
                                                 IB_SA_MCM_MLID_F);
                    mad_get_array(mad, IB_SA_DATA_OFFS, IB_SA_MCM_MGID_F, mgid->raw);
                    PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                                "multicast join returned mlid:%8x mgid:0x%16lx\n ",
                                *mlid, mgid->global.interface_id);
                }
            }
            recv_index++;
        }
    }

    /* recv all responses */

    for (; recv_index < num_ranks; recv_index++) {
        /* check if response is already recieved */
        if (MV2_CHAR_ISBITSET(recv_mask, recv_index + 1)) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 2, "already received index:%d\n",
                        recv_index + 1);
            continue;
        }
        ret = mv2_recv_umad_response(mad_portid, umad_recv_buf,
                                     recv_index, trid, trid + num_ranks, recv_mask);
        if (ret != MCAST_SUCCESS) {
            goto cleanup;
        }
        /* extract mgid, mlid from first response */
        mad = umad_get_mad(umad_recv_buf);
        status = mad_get_field(mad, 0, IB_MAD_STATUS_F);
        if (status == 0) {
            /* recv success */
            if (method == IB_MAD_METHOD_SET) {
                *mlid =
                    (uint16_t) mad_get_field(mad, IB_SA_DATA_OFFS, IB_SA_MCM_MLID_F);
                mad_get_array(mad, IB_SA_DATA_OFFS, IB_SA_MCM_MGID_F, mgid->raw);
                PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                            "multicast join returned mlid:%8x mgid:0x%8lx\n ",
                            *mlid, mgid->global.interface_id);
            }
        }
    }
    ret = MCAST_SUCCESS;

  cleanup:
    if (umad_send_buf) {
        umad_free(umad_send_buf);
    }

    if (umad_recv_buf) {
        umad_free(umad_recv_buf);
    }

    if (mad_portid >= 0) {
        if (sa_agentid >= 0) {
            if (umad_unregister(mad_portid, sa_agentid)) {
                PRINT_DEBUG(DEBUG_MCST_verbose > 1,"Failed to UMAD deregister agent for MADS\n");
                ret = MCAST_FAILURE;
            }
        }
        if (umad_close_port(mad_portid)) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 1,"failed to close UMAD port \n");
            ret = MCAST_FAILURE;
        }
    }
    if (recv_mask) {
        MPIU_Free(recv_mask);
    }

    return ret;

  fn_fail:
    ret = MCAST_FAILURE;
    goto cleanup;
}

static inline int mv2_mcast_detach_ud_qp(mcast_info_t * minfo)
{
    /* Detach qp from mcast group */
    if (ibv_detach_mcast
        (mcast_ctx->ud_ctx->qp, &minfo->grp_info.mgid, minfo->grp_info.mlid) < 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"MCAST QP detach failied \n");
        return MCAST_FAILURE;
    }

    return MCAST_SUCCESS;
}

static inline int mv2_mcast_post_ud_recv_buffers(int num_bufs, mv2_ud_ctx_t * ud_ctx)
{
    int i = 0, ret = 0;
    vbuf *v = NULL;
    struct ibv_recv_wr *bad_wr = NULL;

    if (num_bufs > mcast_max_ud_recv_wqe) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"Try to post %d to MCAST UD recv buffers, max %d\n",
                    num_bufs, mcast_max_ud_recv_wqe);
    }

    for (i = 0; i < num_bufs; ++i) {
        MV2_GET_AND_INIT_UD_VBUF(v);
        if (v == NULL) {
            break;
        }

        vbuf_init_ud_recv(v, rdma_default_ud_mtu, mcast_ctx->selected_rail);
        if (ud_ctx->qp->srq) {
            ret = ibv_post_srq_recv(ud_ctx->qp->srq, &v->desc.u.rr, &bad_wr);
        } else {
            ret = ibv_post_recv(ud_ctx->qp, &v->desc.u.rr, &bad_wr);
        }
        if (ret) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 1, "Failed to post recv buffer \n");
            MRAILI_Release_vbuf(v);
            break;
        }
    }

    PRINT_DEBUG(DEBUG_MCST_verbose > 4,
                "Posted %d buffers of size:%d to MCAST QP, rail %d \n", num_bufs,
                rdma_default_ud_mtu, mcast_ctx->selected_rail);

    return i;

}

mv2_ud_ctx_t * mv2_mcast_prepare_ud_ctx()
{
    mv2_ud_qp_info_t qp_info;
    mv2_ud_ctx_t *ud_ctx;
    char *val;


    qp_info.send_cq = qp_info.recv_cq = mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[mcast_ctx->selected_rail];
    qp_info.sq_psn = rdma_default_psn;
    qp_info.pd = mv2_MPIDI_CH3I_RDMA_Process.ptag[mcast_ctx->selected_rail];
    qp_info.cap.max_send_sge = rdma_default_max_sg_list;
    qp_info.cap.max_recv_sge = rdma_default_max_sg_list;
    qp_info.cap.max_send_wr = 2 * mcast_window_size;
    qp_info.cap.max_recv_wr = mcast_max_ud_recv_wqe;
    qp_info.srq = NULL;
    if ((val = getenv("MV2_USE_UD_SRQ")) != NULL && atoi(val)) {
        qp_info.srq = create_srq(&mv2_MPIDI_CH3I_RDMA_Process, mcast_ctx->selected_rail);
    }
    qp_info.cap.max_inline_data = rdma_max_inline_size;
    ud_ctx = mv2_ud_create_ctx(&qp_info, mcast_ctx->selected_rail);
    if (!ud_ctx) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"MCAST UD QP creation failed");
        return NULL;
    }

    ud_ctx->send_wqes_avail = 2 * mcast_window_size;
    MESSAGE_QUEUE_INIT(&ud_ctx->ext_send_queue);
    ud_ctx->hca_num = mcast_ctx->selected_rail;
    ud_ctx->num_recvs_posted = 0;
    ud_ctx->credit_preserve = (mcast_max_ud_recv_wqe * 3) / 4;
    ud_ctx->ext_sendq_count = 0;

    ud_ctx->num_recvs_posted +=
        mv2_mcast_post_ud_recv_buffers(mcast_max_ud_recv_wqe, ud_ctx);

    return ud_ctx;
}

static inline int mv2_mcast_attach_ud_qp(mcast_info_t * minfo)
{
    char buf[40];
    struct ibv_ah_attr *ah_attr = NULL;
    /* create address handle */

    ah_attr = &minfo->ah_attr;
    MPIU_Memset(ah_attr, 0, sizeof(struct ibv_ah_attr));
    ah_attr->dlid = minfo->grp_info.mlid;
    ah_attr->sl = rdma_default_service_level;
    ah_attr->is_global = 1;
    memcpy(ah_attr->grh.dgid.raw, minfo->grp_info.mgid.raw, 16);
    ah_attr->port_num = rdma_default_port;
    minfo->ah = ibv_create_ah(mv2_MPIDI_CH3I_RDMA_Process.ptag[0], ah_attr);
    if (!minfo->ah) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "MCAST Address handle\n");
        return MCAST_FAILURE;
    }

    /* attach qp to multicast group */
    if (ibv_attach_mcast
        (mcast_ctx->ud_ctx->qp, &minfo->grp_info.mgid, minfo->grp_info.mlid) < 0) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "MCAST QP attach failed\n");
        return MCAST_FAILURE;
    }

    inet_ntop(AF_INET6, minfo->grp_info.mgid.raw, buf, 40);
    PRINT_DEBUG(DEBUG_MCST_verbose>1,"attached qp mgid %s \n",buf);

    return MCAST_SUCCESS;
}

#if defined(RDMA_CM)
int mv2_rdma_cm_mcst_get_addr_info(char *dst, struct sockaddr *addr)
{
    struct addrinfo *res;
    int ret;

    ret = getaddrinfo(dst, NULL, NULL, &res);
    if (ret) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: getaddrinfo failed (%s) - invalid hostname or IP"
                " address for rdma_cm multicast\n", gai_strerror(ret));
        return ret;
    }
    PRINT_DEBUG(DEBUG_MCST_verbose,"addr len %d, addr: %s ends\n",res->ai_addrlen,dst); 
    MPIU_Memcpy(addr, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);

    return ret;
}

static int mv2_rdma_cm_mcst_addr_handler(mcast_info_t *node)
{
    int ret = 0;
    
    ret = rdma_join_multicast(node->cma_id, node->dst_addr, node);     
    PRINT_DEBUG(DEBUG_MCST_verbose>1,"Join multicast status %d\n",ret);
    if (ret) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: Failure joining multicast group for rdma_cm"
                " multicast\n");
    }   

    return ret;
}

static int mv2_rdma_cm_mcst_join_handler(mcast_info_t *minfo,
            struct rdma_ud_param *param)
{
    char buf[40];
    mcast_ctx->remote_qpn = param->qp_num;
    mcast_ctx->remote_qkey = param->qkey;
    inet_ntop(AF_INET6, param->ah_attr.grh.dgid.raw, buf, 40);
    MPIU_Memcpy(minfo->grp_info.mgid.raw,param->ah_attr.grh.dgid.raw, sizeof(param->ah_attr.grh.dgid.raw));
    minfo->grp_info.mlid = param->ah_attr.dlid;
    return 0;
}


static int mv2_rdma_cm_mcst_cma_handler(struct rdma_cm_id *cma_id, struct rdma_cm_event *event, mcast_info_t * minfo)
{
    int ret = 0;
    
    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        ret = mv2_rdma_cm_mcst_addr_handler((mcast_info_t *)cma_id->context);
        break;
    case RDMA_CM_EVENT_MULTICAST_JOIN:
        ret = mv2_rdma_cm_mcst_join_handler((mcast_info_t *)cma_id->context, &event->param.ud);
        break;
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
    case RDMA_CM_EVENT_MULTICAST_ERROR:
        PRINT_DEBUG(DEBUG_MCST_verbose,"Event: %s, Error: %d\n",
               rdma_event_str(event->event), event->status);
       
        ret = event->status;
        break;
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
        /* Cleanup will occur after test completes. */
        break;
    default:
        break; 
    }
    return ret;
}

static int mv2_rdma_cm_mcst_get_dst_addr(char *dst, struct sockaddr *addr)
{
	struct sockaddr_ib *sib;
	sib = (struct sockaddr_ib *) addr;
	MPIU_Memset(sib, 0, sizeof *sib);
	sib->sib_family = AF_IB;
	inet_pton(AF_INET6, dst, &sib->sib_addr);
	return 0;
}

static inline int mv2_rdma_cm_join_mcast_group(mcast_info_t * minfo,
                                       mcast_init_info_t * all_init_info,
                                       int rank)
{
    char buf[40];
    int ret = 0;
    int num_expected_events = 0;
    struct rdma_cm_event *event = NULL;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    if (rank == 0) {
        minfo->grp_info.mlid = 0;
        minfo->grp_info.status = 0;
        minfo->grp_info.comm_id = 0;
        MPIU_Memset(minfo->grp_info.mgid.raw, 0, 16);
    }

    minfo->dst_addr = (struct sockaddr *) &(minfo->dst_in);
    minfo->channel = rdma_create_event_channel();

    if (!minfo->channel) {
	    minfo->grp_info.status = 0;
	    return MCAST_FAILURE;
    }

    ret = rdma_create_id(minfo->channel, &minfo->cma_id,
		    minfo, RDMA_PS_UDP);
    if (ret) {
	    minfo->grp_info.status = 0;
	    return MCAST_FAILURE;
    }
    
    ret = rdma_bind_addr(minfo->cma_id,mcast_ctx->src_addr);
    PRINT_DEBUG(DEBUG_MCST_verbose, "bind addr %d\n",ret);


    if (rank == 0) {
        mv2_rdma_cm_mcst_get_addr_info("0.0.0.0", (struct sockaddr *) &minfo->dst_in);
        ret = rdma_resolve_addr(minfo->cma_id,
                mcast_ctx->src_addr, minfo->dst_addr,
                2000);
        if (ret) {
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: rdma_resolve_addr failed (ret = %d) when"
                    " trying to resolve address for rdma cm multicast\n", ret);
            return MCAST_FAILURE;
        }
        PRINT_DEBUG(DEBUG_MCST_verbose,"Resolve addr success\n");
    } else {
        inet_ntop(AF_INET6, minfo->grp_info.mgid.raw, buf, 40);
        PRINT_DEBUG(DEBUG_MCST_verbose, "MGID RAW %s \n",buf);
        mv2_rdma_cm_mcst_get_dst_addr(buf, (struct sockaddr *) &minfo->dst_in);
        mv2_rdma_cm_mcst_addr_handler(minfo); 
    }

    if (rank != 0) {
        num_expected_events = 1;
    } else {
        num_expected_events = 2;
    }
    while (!ret && num_expected_events > 0) {
        ret = rdma_get_cm_event(minfo->channel, &event);
        if (!ret) {
            ret = mv2_rdma_cm_mcst_cma_handler(event->id, event, minfo); 
            rdma_ack_cm_event(event);
            num_expected_events--;
        }
    }
    minfo->grp_info.status = 1;
    inet_ntop(AF_INET6, minfo->grp_info.mgid.raw, buf, 40);
    PRINT_DEBUG(DEBUG_MCST_verbose, "Joined dgid: %s mlid 0x%x\n", buf,
                minfo->grp_info.mlid);
    return MCAST_SUCCESS;
}

static inline int mv2_rdma_cm_leave_mcast_group(mcast_info_t * minfo)
{
    int ret = 0;
    ret = rdma_leave_multicast(minfo->cma_id, minfo->dst_addr);  
    if (ret) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: rdma_leave_multicast failed for rdma_cm"
                " multicast\n");
    }
    return ret;
}
#endif /* #if defined (RDMA_CM) */

static inline int mv2_join_mcast_group(mcast_info_t * minfo,
                                       mcast_init_info_t * all_init_info,
                                       int num_ranks)
{
    minfo->grp_info.mlid = 0;
    minfo->grp_info.comm_id = 0;
    MPIU_Memset(minfo->grp_info.mgid.raw, 0, 16);
    if (mv2_op_mcast_group(all_init_info, num_ranks, IB_MAD_METHOD_SET, 
             &minfo->grp_info.mgid, &minfo->grp_info.mlid) != MCAST_SUCCESS) {
        minfo->grp_info.status = 0;
        return MCAST_FAILURE;
    }
    minfo->grp_info.status = 1;
    return MCAST_SUCCESS;
}

static inline int mv2_leave_mcast_group(mcast_info_t * minfo,
                                        mcast_init_info_t * all_init_info,
                                        int num_ranks)
{
    return mv2_op_mcast_group(all_init_info, num_ranks, IB_MAD_METHOD_DELETE,
                              &(minfo->grp_info.mgid), NULL);
}

static inline void mv2_mcast_remove_sendwin(message_queue_t * q, vbuf * v)
{
    MPIU_Assert(q->head == v);
    q->head = v->mcast_sendwin_msg.next;
    q->count--;
    if (q->head == NULL) {
        q->tail = NULL;
        MPIU_Assert(q->count == 0);
    }
    v->mcast_sendwin_msg.next = NULL;
}

static inline void mv2_mcast_add_sendwin(message_queue_t * q, vbuf * v)
{
    v->mcast_sendwin_msg.next = v->mcast_sendwin_msg.prev = NULL;

    if (q->head == NULL) {
        q->head = v;
    } else {
        (q->tail)->mcast_sendwin_msg.next = v;
    }

    q->tail = v;
    q->count++;
}

static inline void mv2_mcast_remove_recvwin(message_queue_t * q, vbuf * v)
{
    MPIU_Assert(q->head == v);
    q->head = v->mcast_recvwin_msg.next;
    q->count--;
    if (q->head == NULL) {
        q->tail = NULL;
        MPIU_Assert(q->count == 0);
    } else {
        q->head->mcast_recvwin_msg.prev = NULL;
    }

    v->mcast_recvwin_msg.next = NULL;
}

static inline void mv2_mcast_add_recvwin(message_queue_t * q, vbuf * v)
{
    vbuf *next = NULL, *prev = NULL;
    MPIDI_CH3_Pkt_mcast_t *p, *p1;
    p = (MPIDI_CH3_Pkt_mcast_t *) v->pheader;

    /* check if it duplicate */
    next = q->head;
    while (next) {
        p1 = (MPIDI_CH3_Pkt_mcast_t *) next->pheader;
        if (p1->psn >= p->psn) {
            break;
        }
        prev = next;
        next = next->mcast_recvwin_msg.next;
    }

    if (next && p1->psn == p->psn) {
        return;
    }

#if !defined(NDEBUG)
    /* drop packet if it is to test reliability */
    if (mcast_drop_packet_rate &&
            (rand() % mcast_drop_packet_rate) == 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "mcast pkt dropped :%d\n",
                p->psn);
        MRAILI_Release_vbuf(v);
        return;
    }
#endif

    v->mcast_recvwin_msg.next = v->mcast_recvwin_msg.prev = NULL;

    if (q->head == NULL) {
        q->head = v;
    } else {
        if (prev == NULL) {
            v->mcast_recvwin_msg.next = q->head;
            q->head->mcast_recvwin_msg.prev = v;
            q->head = v;
        } else {
            v->mcast_recvwin_msg.next = next;
            v->mcast_recvwin_msg.prev = prev;
            prev->mcast_recvwin_msg.next = v;
            if (next) {
                next->mcast_recvwin_msg.prev = v;
            }
        }
    }

    q->count++;
}

static inline void mv2_mcast_send_init_ack(int comm_id, int root)
{
    int mpi_errno = MPI_SUCCESS;
    int leader_rank;
    MPIDI_VC_t *vc;
    MPID_Request *sreq = NULL;
    MPIDI_CH3_Pkt_mcast_init_ack_t pkt;
    MPID_Comm *comm_ptr;

    MPIDI_Pkt_init(&pkt, MPIDI_CH3_PKT_MCST_INIT_ACK);
    pkt.psn = 0;
    pkt.comm_id = comm_id;
    comm_ptr = mv2_mcast_find_comm(comm_id);
    MPIU_Assert(comm_ptr);
    PMPI_Comm_rank(comm_ptr->dev.ch.leader_comm, &leader_rank);
    pkt.src_rank = leader_rank;
    MPIDI_Comm_get_vc(comm_ptr, root, &vc);

    mpi_errno = MPIDI_CH3_iStartMsg(vc, &pkt, sizeof(pkt), &sreq);
    if (mpi_errno != MPI_SUCCESS) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1,"Error in sending multicast NACK\n");
    }

    if (sreq != NULL) {
        MPID_Request_release(sreq);
    }

    PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                "sent init ack comm_id:%d  to :%d\n", comm_id, root);
}

static inline void mv2_mcast_send_nack(uint32_t psn, int comm_id, int root)
{
    int mpi_errno = MPI_SUCCESS;
    vbuf *v;
    mcast_info_t *minfo;
    MPIDI_VC_t *vc;
    MPID_Request *sreq = NULL;
    MPIDI_CH3_Pkt_mcast_nack_t pkt;
    MPID_Comm *comm_ptr;
    bcast_info_t *bcast_info;

    MPIDI_Pkt_init(&pkt, MPIDI_CH3_PKT_MCST_NACK);
    pkt.psn = psn;
    pkt.comm_id = comm_id;
    pkt.root = root;
    comm_ptr = mv2_mcast_find_comm(comm_id);
    pkt.src_rank = comm_ptr->rank;

    if (mcast_use_mcast_nack) {
        bcast_info = (bcast_info_t *) comm_ptr->dev.ch.bcast_info;
        minfo = &bcast_info->minfo;

        long now = mv2_get_time_us();
        if ((now - bcast_info->nack_time) < mcast_retry_timeout) {
            /* received multicast NACK recently from other */
            return;
        }
        MV2_GET_AND_INIT_UD_VBUF(v);
        bcast_info->nack_time = mv2_get_time_us();
        MPIU_Memcpy(v->pheader, (const void *) &pkt, sizeof(MPIDI_CH3_Pkt_mcast_nack_t));

        vbuf_init_mcast_send(v, sizeof(MPIDI_CH3_Pkt_mcast_nack_t), mcast_ctx->selected_rail, minfo);
        IBV_POST_MCAST_SEND(v, mcast_ctx);

        PRINT_DEBUG(DEBUG_MCST_verbose > 2, 
                            "sending mcast nack psn:%d to :%d head:%d tail:%d\n", 
                            psn, root, bcast_info->win_head, bcast_info->win_tail);

    } else {
        MPIDI_Comm_get_vc(comm_ptr, root, &vc);
        mpi_errno = MPIDI_CH3_iStartMsg(vc, &pkt, sizeof(pkt), &sreq);
        if (mpi_errno != MPI_SUCCESS) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 1,"Error in sending multicast NACK\n");
        }

        if (sreq != NULL) {
            MPID_Request_release(sreq);
        }
    
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "sending rc nack psn:%d to :%d\n", psn, root);
    }
}

static inline void mv2_mcast_resend_window(bcast_info_t * bcast_info, uint32_t psn)
{
    int num_resends = 0;
    vbuf *v;

    bcast_info->resend_time = mv2_get_time_us();
    v = bcast_info->send_window.head;
    while (v) {

        MPIU_Assert(v != NULL);

        v->retry_count++;
        v->pending_send_polls++;
        v->timestamp = mv2_get_time_us();
        v->flags &= ~(UD_VBUF_FREE_PENIDING);

        IBV_POST_MCAST_SEND(v, mcast_ctx);

        v = v->mcast_sendwin_msg.next;
        num_resends++;
    }
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, 
                "mcast resend window psn:%d num_resends:%d \n", psn, num_resends);
}

static inline void mv2_mcast_resend(bcast_info_t * bcast_info, uint32_t psn)
{
    long now;
    uint16_t resend_backoff;
    vbuf *v;
    MPIDI_CH3_Pkt_mcast_t *p;

    /* TODO:  Presently sending only requested NACK.
     ** is it efficient to resend whole window instead ?
     */

    v = bcast_info->send_window.head;
    while (v) {
        p = (MPIDI_CH3_Pkt_mcast_t *) v->pheader;
        if (psn == p->psn) {
            break;
        }
        v = v->mcast_sendwin_msg.next;
    }

    MPIU_Assert(v != NULL);
    /* check if it already resend */
    if (v->retry_count < 2) {
        resend_backoff = 1;
    } else {
        LOG2(v->retry_count, resend_backoff);
    }

    now = mv2_get_time_us();
    if (now < v->timestamp + MIN((mcast_retry_timeout *
                                  resend_backoff), mcast_max_retry_timeout)) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                    "mcast resend psn:%u is resent recenty now:%ld \n", psn, now);
        return;
    }

    v->retry_count++;
    v->timestamp = mv2_get_time_us();
    v->pending_send_polls++;
    
    IBV_POST_MCAST_SEND(v, mcast_ctx);

    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "mcast resend psn:%d \n", psn);
}


int mv2_mcast_init_bcast_info(bcast_info_t ** bcast_info)
{
    *bcast_info = (bcast_info_t *) MPIU_Malloc(sizeof(bcast_info_t));
    /* init bcast info */
    MESSAGE_QUEUE_INIT(&((*bcast_info)->send_window));
    MESSAGE_QUEUE_INIT(&((*bcast_info)->recv_window));
    (*bcast_info)->in_recv_progress = 0;
    (*bcast_info)->nack_time = 0;
    (*bcast_info)->resend_time = 0;
    (*bcast_info)->win_head = 1;
    (*bcast_info)->win_tail = 0;

    return MCAST_SUCCESS;
}

int mv2_setup_multicast(mcast_info_t * minfo, MPID_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS, i;
    MPID_Comm *leader_ptr;
    mcast_init_info_t init_info;
    mcast_init_info_t *all_init_info = NULL;
    minfo->init_info = NULL;
    int leader_rank, leader_comm_size, num_ip_enabled_nodes = 0;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int comm_id, mcast_status = MCAST_SUCCESS;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_ptr);

    PMPI_Comm_size(comm_ptr->dev.ch.leader_comm, &leader_comm_size);
    PMPI_Comm_rank(comm_ptr->dev.ch.leader_comm, &leader_rank);
    
    if (leader_rank == 0) {
        all_init_info = MPIU_Malloc(leader_comm_size * sizeof(mcast_init_info_t));
        if (all_init_info == NULL) {
            return MCAST_FAILURE;
        }
    }

    MPIU_Memcpy(init_info.gid, mv2_MPIDI_CH3I_RDMA_Process.gids[0][0].raw, 16);

    MPIR_Gather_impl(init_info.gid, sizeof(mcast_init_info_t), MPI_BYTE,
                                 all_init_info, sizeof(mcast_init_info_t), MPI_BYTE,
                                 0, leader_ptr, &errflag);
#if defined(RDMA_CM)
    if (num_ip_enabled_devices > 0) {
        num_ip_enabled_nodes = 1;
    }
    mpi_errno =  MPIR_Allreduce_impl(MPI_IN_PLACE, &num_ip_enabled_nodes,
                                    1, MPI_INT, MPI_SUM, leader_ptr, &errflag);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    if (num_ip_enabled_nodes < leader_comm_size) {
        PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: One or more nodes do not have any IP enabled"
                " HCAs. Disabling rdma_cm based multicast\n");
        rdma_use_rdma_cm_mcast = 0;
    }
#endif /*defined(RDMA_CM)*/
        
    if (leader_rank == 0) {
        minfo->init_info = all_init_info;
        for (i = 0; i < leader_comm_size; i++) {
            PRINT_DEBUG_GID(((union ibv_gid *) &all_init_info[i]))
        }

#if defined(RDMA_CM)
        if(rdma_use_rdma_cm_mcast == 1) {
            mcast_status = mv2_rdma_cm_join_mcast_group(minfo, all_init_info, leader_rank);
        }
        else
#endif
        {
            mcast_status = mv2_join_mcast_group(minfo, all_init_info, leader_comm_size);
        }

        if (mcast_status == MCAST_SUCCESS) {
            /* assign a unique comm_id */
            while (1) {
                comm_id = rand() % MV2_MCAST_MAX_COMMS;
                if (mv2_mcast_find_comm(comm_id) == NULL) {
                    minfo->grp_info.comm_id = comm_id;
                    break;
                }
            }
            PRINT_DEBUG(DEBUG_MCST_verbose>1,"leader comm size %d\n",leader_comm_size);
            mv2_mcast_add_comm_init_req(comm_id, leader_comm_size);
        } else {
            PRINT_INFO((MPIDI_Process.my_pg_rank == 0), "[Warning]: mv2_rdma_cm_join_mcast_group failed\n");
        }
    }
    /* Brodcast mcast info */
    mpi_errno = MPIR_Bcast_impl(&minfo->grp_info, sizeof(mcast_grp_info_t),
                                MPI_BYTE, 0, leader_ptr, &errflag);
    if (mpi_errno) {
        goto fn_fail;
    }

    /* check multicast group setup status */
    if (!minfo->grp_info.status) {
        /* setup failed */
        return MCAST_FAILURE;
    }
#if defined(RDMA_CM)
    if (leader_rank != 0 && rdma_use_rdma_cm_mcast) {
        mv2_rdma_cm_join_mcast_group(minfo, all_init_info, leader_rank);
    }
#endif

    PRINT_DEBUG(DEBUG_MCST_verbose > 1,
                "Mcast grp_info: status:%d comm_id:%d mlid:%8x  mgid:%8lx \n",
                minfo->grp_info.status, minfo->grp_info.comm_id,
                minfo->grp_info.mlid, minfo->grp_info.mgid.global.interface_id);

    mv2_mcast_register_comm(comm_ptr, minfo->grp_info.comm_id);

    /* TODO: does root needs to gather attach status from all non-roots? */
    if (mv2_mcast_attach_ud_qp(minfo) != MCAST_SUCCESS) {
        goto fn_fail;
    }
    return MCAST_SUCCESS;

fn_fail:
#if defined(RDMA_CM)
    if (rdma_use_rdma_cm_mcast != 0) {
        mv2_rdma_cm_leave_mcast_group(minfo);
    }
    else
#endif
    {    
        if (leader_rank == 0) {
            mv2_leave_mcast_group(minfo, minfo->init_info, leader_comm_size);
        }
    }

    return MCAST_FAILURE;
}

int mv2_cleanup_multicast(mcast_info_t * minfo, MPID_Comm * comm_ptr)
{
    int leader_rank, leader_comm_size, cleanup_status = MCAST_SUCCESS;

    PMPI_Comm_size(comm_ptr->dev.ch.leader_comm, &leader_comm_size);
    PMPI_Comm_rank(comm_ptr->dev.ch.leader_comm, &leader_rank);

    mv2_mcast_unregister_comm(minfo->grp_info.comm_id);
    mv2_mcast_detach_ud_qp(minfo);


#if defined(RDMA_CM)
    if (rdma_use_rdma_cm_mcast != 0) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Leaving RDMA CM multicast group\n"); 
        if (mv2_rdma_cm_leave_mcast_group(minfo)) {
            cleanup_status = MCAST_FAILURE;
        }
        rdma_destroy_id(minfo->cma_id); 
        minfo->cma_id = NULL;
        rdma_destroy_event_channel(minfo->channel); 
        minfo->channel = NULL;
    }
    else
#endif
    {
        if (leader_rank == 0) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 2, "Leaving multicast group\n");
            if (mv2_leave_mcast_group(minfo, minfo->init_info, leader_comm_size) !=
                    MCAST_SUCCESS) {
                cleanup_status = MCAST_FAILURE;
            }
        }
    }

    if (leader_rank == 0 && minfo->init_info) {
        MPIU_Free(minfo->init_info);
        minfo->init_info = NULL;
    }

    return cleanup_status;
}

void mv2_mcast_flush_sendwin(message_queue_t * q)
{
    vbuf *v;

    while (q->head) {
        v = q->head;
        mv2_mcast_remove_sendwin(q, v);
        if (v->flags & UD_VBUF_FREE_PENIDING)  {
            v->flags &= ~(UD_VBUF_FREE_PENIDING);
            release_vbuf(v);
        }
        v->flags &= ~(UD_VBUF_SEND_INPROGRESS);
    }
}

void mv2_mcast_send(bcast_info_t * bcast_info, char *buf, int len)
{
    vbuf *v;
    MPID_Comm *comm_ptr;
    mcast_info_t *minfo = &bcast_info->minfo;

    MV2_GET_AND_INIT_UD_VBUF(v);
    MPIDI_CH3_Pkt_mcast_t *p = (MPIDI_CH3_Pkt_mcast_t *) v->pheader;
    p->type = MPIDI_CH3_PKT_MCST;
    p->rail = mcast_ctx->selected_rail;
    p->psn = bcast_info->win_head;
    p->comm_id = minfo->grp_info.comm_id;
    comm_ptr = mv2_mcast_find_comm(p->comm_id);
    p->src_rank = comm_ptr->rank;

    memcpy((char *) p + sizeof(MPIDI_CH3_Pkt_mcast_t), buf, len);

    mv2_mcast_add_sendwin(&bcast_info->send_window, v);
    v->flags |= (UD_VBUF_SEND_INPROGRESS | UD_VBUF_MCAST_MSG);
    v->timestamp = mv2_get_time_us();
    v->retry_count = 1;
    v->pending_send_polls = 1;

    vbuf_init_mcast_send(v, sizeof(MPIDI_CH3_Pkt_mcast_t) + len, mcast_ctx->selected_rail, minfo);

    IBV_POST_MCAST_SEND(v, mcast_ctx);
    PRINT_DEBUG(DEBUG_MCST_verbose > 3, "mcast send psn:%u len:%lu\n",
                p->psn, sizeof(MPIDI_CH3_Pkt_mcast_t) + len);
}

void mv2_process_mcast_msg(vbuf * v)
{
    MPIDI_CH3_Pkt_mcast_t *p;
    MPID_Comm *comm_ptr = NULL;
    bcast_info_t *bcast_info;
    p = (MPIDI_CH3_Pkt_mcast_t *) v->pheader;

    mv2_ud_ctx_t *ud_ctx = mcast_ctx->ud_ctx;
    --ud_ctx->num_recvs_posted;
    if (ud_ctx->num_recvs_posted < ud_ctx->credit_preserve) {
        ud_ctx->num_recvs_posted +=
            mv2_mcast_post_ud_recv_buffers((mcast_max_ud_recv_wqe -
                                            ud_ctx->num_recvs_posted), ud_ctx);
    }

    comm_ptr = mv2_mcast_find_comm(p->comm_id);
    if (comm_ptr == NULL) {
        MRAILI_Release_vbuf(v);
        return;
    }

    if (mcast_skip_loopback && p->src_rank == comm_ptr->rank) {
        /* multicast loopback message */
        MRAILI_Release_vbuf(v);
        return;
    }

    bcast_info = (bcast_info_t *) comm_ptr->dev.ch.bcast_info;

    switch (p->type) {
        case MPIDI_CH3_PKT_MCST_NACK:
            if (comm_ptr->rank == p->root) {
                mv2_mcast_handle_nack((void *) p);
            } else {
                PRINT_DEBUG(DEBUG_MCST_verbose > 3, 
                        "Nack recieved at non root. psn:%d root:%d\n", p->psn, p->root);
                bcast_info->nack_time = mv2_get_time_us();
            }
            MRAILI_Release_vbuf(v);
            break;
        case MPIDI_CH3_PKT_MCST_INIT:
            PRINT_DEBUG(DEBUG_MCST_verbose > 2,
                        "mcast init recieved comm_id:%d\n", p->comm_id);
            mv2_mcast_send_init_ack(p->comm_id, p->src_rank);
            MRAILI_Release_vbuf(v);
            break;
        case MPIDI_CH3_PKT_MCST:
             PRINT_DEBUG(DEBUG_MCST_verbose > 3, "Received psn:%d head:%d tail:%d size %lu \n",
                                    p->psn, bcast_info->win_head, bcast_info->win_tail, v->content_size);
            /* check out of window */
            if (!INCL_BETWEEN(p->psn, bcast_info->win_head, 
                        bcast_info->win_head + mcast_window_size)) {
                PRINT_DEBUG(DEBUG_MCST_verbose > 3, "Dropped psn:%d head:%d tail:%d \n", 
                        p->psn, bcast_info->win_head, bcast_info->win_tail);
                MRAILI_Release_vbuf(v);
                break;
            }

            mv2_mcast_add_recvwin(&bcast_info->recv_window, v);

            if (bcast_info->in_recv_progress && p->psn == bcast_info->win_head) {
                MPIDI_CH3I_progress_completion_count++;
            }
            break;
        default:
            PRINT_DEBUG(DEBUG_MCST_verbose > 1,"unknown mcast pkt received\n");
    }

}

void mv2_mcast_handle_nack(MPIDI_CH3_Pkt_mcast_nack_t * p)
{
    uint32_t psn;
    MPID_Comm *comm_ptr = NULL;
    bcast_info_t *bcast_info;
    PRINT_DEBUG(DEBUG_MCST_verbose > 2, "nack received psn:%d\n", p->psn);
    psn = p->psn;
    comm_ptr = mv2_mcast_find_comm(p->comm_id);
    bcast_info = (bcast_info_t *) comm_ptr->dev.ch.bcast_info;
    if (!EXCL_BETWEEN(psn, bcast_info->win_tail, bcast_info->win_head)) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 2, "psn:%d is not in window (%d - %d)\n",
                    psn, bcast_info->win_tail, bcast_info->win_head);
        return;
    }

    if (mcast_use_mcast_nack) {
        mv2_mcast_resend_window(bcast_info, psn);
    } else {
        mv2_mcast_resend(bcast_info, psn);
    }
}

void mv2_mcast_recv(bcast_info_t * bcast_info, char *buf, int len, int root)
{
    vbuf *v;
    uint32_t nspin = 1, resend_count = 1;
    MPIDI_CH3_Pkt_mcast_t *p;
    long next_retry_timeout = mcast_retry_timeout + mv2_get_time_us();
    long timestamp;
    int retry_backoff;

    bcast_info->in_recv_progress = 1;


    while (1) {
        if (bcast_info->recv_window.head) {
            v = bcast_info->recv_window.head;
            p = (MPIDI_CH3_Pkt_mcast_t *) v->pheader;

            if (p->psn == bcast_info->win_head) {
                break;
            }
        }
        nspin++;
        if (nspin % mcast_nspin_threshold == 0) {
            timestamp = mv2_get_time_us();
            if (timestamp > next_retry_timeout && mcast_enable_rel) {
                mv2_mcast_send_nack(bcast_info->win_head,
                                    bcast_info->minfo.grp_info.comm_id, root);
                resend_count++;
                LOG2(resend_count, retry_backoff);
                next_retry_timeout = timestamp +
                    MIN((mcast_retry_timeout * retry_backoff),
                        mcast_max_retry_timeout);
            }

        }
        MPIDI_CH3I_Progress(FALSE, NULL);
    }

    bcast_info->in_recv_progress = 0;

    mv2_mcast_remove_recvwin(&bcast_info->recv_window, v);

    PRINT_DEBUG(DEBUG_MCST_verbose > 3,
                "mcast recv size:%zu psn:%d len:%d\n", v->content_size, p->psn, len);
    if (len != (v->content_size - sizeof(MPIDI_CH3_Pkt_mcast_t))) {
        PRINT_DEBUG(DEBUG_MCST_verbose > 1, "mismatch in size\n");
    }
    MPIU_Assert(len == (v->content_size - sizeof(MPIDI_CH3_Pkt_mcast_t)));
    memcpy(buf, (char *) v->pheader + sizeof(MPIDI_CH3_Pkt_mcast_t), len);

    MRAILI_Release_vbuf(v);

}
#endif
