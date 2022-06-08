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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "mpichconf.h"
#include "mpimem.h"
#include "mv2_ud.h"
#include "debug_utils.h"
#include "rdma_impl.h"

extern mv2_MPIDI_CH3I_RDMA_Process_t mv2_MPIDI_CH3I_RDMA_Process;

/* create UD context */
struct ibv_qp * mv2_ud_create_qp(mv2_ud_qp_info_t *qp_info, int hca_index)
{
    struct ibv_qp *qp;
    struct ibv_qp_init_attr init_attr;
 
    memset(&init_attr, 0, sizeof(struct ibv_qp_init_attr));
    init_attr.send_cq = qp_info->send_cq;
    init_attr.recv_cq = qp_info->recv_cq;
    init_attr.cap.max_send_wr = qp_info->cap.max_send_wr;
    
    if (qp_info->srq) {
        init_attr.srq = qp_info->srq;
        init_attr.cap.max_recv_wr = 0;
    } else {    
        init_attr.cap.max_recv_wr = qp_info->cap.max_recv_wr;
    }

    init_attr.cap.max_send_sge = qp_info->cap.max_send_sge;
    init_attr.cap.max_recv_sge = qp_info->cap.max_recv_sge;
    init_attr.cap.max_inline_data = qp_info->cap.max_inline_data;
    init_attr.qp_type = IBV_QPT_UD;

    qp = ibv_create_qp(qp_info->pd, &init_attr);
    if(!qp)
    {
        fprintf(stderr,"error in creating UD qp\n");
        return NULL;
    }
    
    if (mv2_ud_qp_transition(qp, hca_index)) {
        return NULL;
    }

    PRINT_DEBUG(DEBUG_UD_verbose>0," UD QP:%p qpn:%d \n",qp, qp->qp_num);

    return qp;
}

int mv2_ud_qp_transition(struct ibv_qp *qp, int hca_index)
{
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(struct ibv_qp_attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = mv2_MPIDI_CH3I_RDMA_Process.ports[hca_index][0];
    attr.qkey = 0;

    if (ibv_modify_qp(qp, &attr,
                IBV_QP_STATE |
                IBV_QP_PKEY_INDEX |
                IBV_QP_PORT | IBV_QP_QKEY)) {
            fprintf(stderr,"Failed to modify QP to INIT\n");
            return 1;
    }    
        
    memset(&attr, 0, sizeof(struct ibv_qp_attr));

    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
            fprintf(stderr, "Failed to modify QP to RTR\n");
            return 1;
    }   

    memset(&attr, 0, sizeof(struct ibv_qp_attr));

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = rdma_default_psn;
    if (ibv_modify_qp(qp, &attr,
                IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        fprintf(stderr, "Failed to modify QP to RTS\n");
        return 1;
    }

    return 0;

}

mv2_ud_ctx_t* mv2_ud_create_ctx (mv2_ud_qp_info_t *qp_info, int hca_index)
{
    mv2_ud_ctx_t *ctx;

    ctx = MPIU_Malloc( sizeof(mv2_ud_ctx_t) );
    if (!ctx){
        fprintf( stderr, "%s:no memory!\n", __func__ );
        return NULL;
    }
    memset( ctx, 0, sizeof(mv2_ud_ctx_t) );

    ctx->qp = mv2_ud_create_qp(qp_info, hca_index);
    if(!ctx->qp) {
        fprintf(stderr, "Error in creating UD QP\n");
        return NULL;
    }

    return ctx;
}

/* create ud vc */
int mv2_ud_set_vc_info (mv2_ud_vc_info_t *ud_vc_info, mv2_ud_exch_info_t *rem_info, union ibv_gid gid, struct ibv_pd *pd, int port)
{
    struct ibv_ah_attr ah_attr;

    MPIU_Memset(&ah_attr, 0, sizeof(ah_attr));

    PRINT_DEBUG(DEBUG_UD_verbose>0,"lid:%d, qpn:%d, port: %d\n",
                rem_info->lid,rem_info->qpn, port);
    
    if (use_iboeth) {
        ah_attr.grh.dgid.global.subnet_prefix = 0;
        ah_attr.grh.dgid.global.interface_id = 0;
        ah_attr.grh.flow_label = 0;
        ah_attr.grh.sgid_index = rdma_default_gid_index;
        ah_attr.grh.hop_limit = 1;
        ah_attr.grh.traffic_class = 0;
        ah_attr.is_global = 1;
        ah_attr.dlid = 0;
        ah_attr.grh.dgid = gid;
    } else {
        ah_attr.is_global = 0;
        ah_attr.dlid = rem_info->lid;
        ah_attr.sl = 0;
    }

    ah_attr.src_path_bits = 0;
    ah_attr.port_num = port;

    ud_vc_info->ah = ibv_create_ah(pd, &ah_attr);
    if(!(ud_vc_info->ah)){    
        fprintf(stderr, "Error in creating address handle\n");
        PRINT_DEBUG(DEBUG_UD_verbose>0,"Error in creating address handle: lid:%d, qpn:%d, port: %d\n",
                    rem_info->lid,rem_info->qpn, port);
        return -1;
    }
    ud_vc_info->lid = rem_info->lid;
    ud_vc_info->qpn = rem_info->qpn;
    return 0;
}

/* destroy ud context */
void mv2_ud_destroy_ctx (mv2_ud_ctx_t *ctx)
{
    if (ctx->qp) {
        ibv_destroy_qp(ctx->qp);
    }
    MPIU_Free(ctx);
}

