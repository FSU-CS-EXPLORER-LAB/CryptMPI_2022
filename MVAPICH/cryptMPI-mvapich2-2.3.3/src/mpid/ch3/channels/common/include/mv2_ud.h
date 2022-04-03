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

#ifndef _MV2_UD_H_
#define _MV2_UD_H_

#include <vbuf.h>
#include <infiniband/verbs.h>

#define LOG2(_v, _r)                            \
do {                                            \
    (_r) = ((_v) & 0xFF00) ? 8 : 0;             \
    if ( (_v) & ( 0x0F << (_r + 4 ))) (_r)+=4;  \
    if ( (_v) & ( 0x03 << (_r + 2 ))) (_r)+=2;  \
    if ( (_v) & ( 0x01 << (_r + 1 ))) (_r)+=1;  \
} while(0)

/*
** We should check if the ackno had been handled before.
** We process this only if ackno had advanced.
** There are 2 cases to consider:
** 1. ackno_handled < seqnolast (normal case)
** 2. ackno_handled > seqnolast (wraparound case)
*/
#define INCL_BETWEEN(_val, _start, _end)                            \
    (((_start > _end) && (_val >= _start || _val <= _end)) ||       \
     ((_end > _start) && (_val >= _start && _val <= _end)) ||       \
     ((_end == _start) && (_end == _val)))

#define EXCL_BETWEEN(_val, _start, _end)                            \
    (((_start > _end) && (_val > _start || _val < _end)) ||         \
     ((_end > _start) && (_val > _start && _val < _end)))



#define UD_ACK_PROGRESS_TIMEOUT (((mv2_get_time_us() - rdma_ud_last_check) > rdma_ud_progress_timeout))

#define MV2_UD_RESET_CREDITS(_vc, _v)  {    \
    if (_v->transport == IB_TRANSPORT_UD) { \
        _vc->mrail.rely.ack_pending = 0;      \
    }                                       \
}   

#define MV2_UD_ACK_CREDIT_CHECK(_vc, _v)   {                            \
    if (_v->transport == IB_TRANSPORT_UD) {                             \
        if (++(_vc->mrail.rely.ack_pending) > rdma_ud_max_ack_pending) {  \
            mv2_send_explicit_ack(_vc);                                 \
        }                                                               \
    }                                                                   \
}

#define MAX_SEQ_NUM (UINT16_MAX)
#define MESSAGE_QUEUE_INIT(q)   \
{                               \
    (q)->head = NULL;           \
    (q)->tail = NULL;           \
    (q)->count = 0 ;            \
}

typedef struct message_queue_t
{
    struct vbuf *head;
    struct vbuf *tail;
    uint16_t count;
} message_queue_t;

typedef struct mv2_ud_ctx_t mv2_ud_ctx_t;
/* ud context */
struct mv2_ud_ctx_t{
    int hca_num;
    int send_wqes_avail;
    int num_recvs_posted;
    int default_mtu_sz;
    int credit_preserve;
    struct ibv_qp *qp;
    message_queue_t ext_send_queue;
    uint64_t ext_sendq_count;
};

typedef struct mv2_ud_qp_info {
    struct ibv_cq      *send_cq;
    struct ibv_cq      *recv_cq;
    struct ibv_srq     *srq;
    struct ibv_pd      *pd; 
    struct ibv_qp_cap  cap; 
    uint32_t           sq_psn;
} mv2_ud_qp_info_t;

/* ud vc info */
typedef struct _mv2_ud_vc_info_t {
    struct ibv_ah *ah;
    uint32_t qpn;
    uint16_t lid;
} mv2_ud_vc_info_t;

typedef struct _mv2_ud_reliability_info_t {
    uint16_t ack_pending;
    message_queue_t send_window;
    message_queue_t ext_window;
    message_queue_t recv_window;
    unsigned long long total_messages;

    /* profiling counters */
    uint64_t cntl_acks;
    uint64_t resend_count;
    uint64_t ext_win_send_count;
} mv2_ud_reliability_info_t;

/* ud exhange info */
typedef struct _mv2_ud_exch_info_t
{
    uint16_t lid;
    uint32_t qpn;
}mv2_ud_exch_info_t;

typedef struct _mv2_rndv_qp_t {
    uint32_t seqnum;
    uint16_t index;
    uint16_t hca_num;
    
    struct ibv_qp *ud_qp;
    struct ibv_cq *ud_cq;

    void *next;
    void *prev;
}mv2_rndv_qp_t;

typedef struct _mv2_ud_zcopy_info_t {
    /* Rndv QP pool */
    mv2_rndv_qp_t *rndv_qp_pool;
    mv2_rndv_qp_t *rndv_qp_pool_free_head;
    int no_free_rndv_qp;
    char *grh_buf;
    void *grh_mr;
    
    struct ibv_cq **rndv_ud_cqs;
    mv2_ud_ctx_t **rndv_ud_qps;
}mv2_ud_zcopy_info_t;

/* create UD context */
mv2_ud_ctx_t* mv2_ud_create_ctx (mv2_ud_qp_info_t *qp_info, int hca_index);

int mv2_ud_qp_transition(struct ibv_qp *qp, int hca_index);

void mv2_ud_zcopy_poll_cq(mv2_ud_zcopy_info_t *zcopy_info, mv2_ud_ctx_t *ud_ctx,
                                vbuf *resend_buf, int hca_index, int *found);
/* create UD QP */
struct ibv_qp *mv2_ud_create_qp (mv2_ud_qp_info_t *qp_info, int hca_index);

/* create ud vc */
int mv2_ud_set_vc_info (mv2_ud_vc_info_t *,
        mv2_ud_exch_info_t *rem_info, union ibv_gid gid,
        struct ibv_pd *pd, int rdma_default_port);

/* destroy ud context */
void mv2_ud_destroy_ctx (mv2_ud_ctx_t *ctx);
void mv2_ud_resend(vbuf *v);

#endif /* #ifndef _MV2_UD_H_ */

