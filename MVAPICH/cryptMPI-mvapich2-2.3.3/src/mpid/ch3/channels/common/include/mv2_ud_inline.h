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
#include <mv2_ud.h>
#include "vbuf.h"
#include <debug_utils.h>

enum {
    MSG_QUEUED_RECVWIN,
    MSG_IN_RECVWIN
};
#define IBV_UD_POST_SR(_v, _ud_vc, _ud_ctx) {                       \
    int __ret;                                                      \
    if(((_v)->desc.sg_entry.length <= rdma_max_inline_size))        \
    {                                                               \
        (_v)->desc.u.sr.send_flags = (enum ibv_send_flags)          \
                (IBV_SEND_SIGNALED | IBV_SEND_INLINE);              \
    } else {                                                        \
        (_v)->desc.u.sr.send_flags = IBV_SEND_SIGNALED ;            \
    }                                                               \
    (_v)->desc.u.sr.wr.ud.ah = (_ud_vc).ah;                         \
    (_v)->desc.u.sr.wr.ud.remote_qpn =  (_ud_vc).qpn;               \
    if ((_ud_ctx)->send_wqes_avail <=0 ||                           \
            (NULL != (_ud_ctx)->ext_send_queue.head)) {             \
        mv2_ud_ext_sendq_queue(&(_ud_ctx)->ext_send_queue, _v);     \
    } else {                                                        \
        (_ud_ctx)->send_wqes_avail--;                               \
        __ret = ibv_post_send((_ud_ctx->qp),                        \
                &((_v)->desc.u.sr),&((_v)->desc.y.bad_sr));         \
        if(__ret) {                                                 \
            fprintf(stderr, "failed to send\n ");                   \
            ibv_error_abort(-1, "ud send failed");                  \
        }                                                           \
    }                                                               \
}

#define SEND_WINDOW_CHECK(_ud_vc, _v)   {                               \
    if ((_ud_vc)->send_window.count > rdma_default_ud_sendwin_size      \
                || ((_ud_vc)->ext_window.head != NULL                   \
                 && (_ud_vc)->ext_window.head != (_v))) {               \
        mv2_ud_ext_window_add(&(_ud_vc)->ext_window, v);                \
        PRINT_DEBUG(DEBUG_UD_verbose>1,"msg(%p) queued to ext window size:%d\n",_v,  \
                        (_ud_vc)->ext_window.count);                    \
        return 0;                                                       \
    }                                                                   \
}

static inline void mv2_ud_ext_sendq_queue(message_queue_t *q, vbuf *v)
{
    v->desc.next = NULL;
    if (q->head == NULL) {
        q->head = v;
    } else {
        q->tail->desc.next = v;
    }
    q->tail = v;
    q->count++;
    PRINT_DEBUG(DEBUG_UD_verbose>1,"queued to ext send queue, queue len:%d seqnum:%d\n", q->count, v->seqnum);
}

static inline void mv2_ud_ext_window_add(message_queue_t *q, vbuf *v)
{
    v->extwin_msg.next = v->extwin_msg.prev = NULL;
    if (q->head == NULL) {
        q->head = v;
    } else {
        (q->tail)->extwin_msg.next = v;
    }
    q->tail = v;
    q->count++;
}

static inline void mv2_ud_send_window_add(message_queue_t *q, vbuf *v)
{
    v->sendwin_msg.next = v->sendwin_msg.prev = NULL;
    v->in_sendwin = 1;

    if(q->head == NULL) {
        q->head = v;
    } else {
        (q->tail)->sendwin_msg.next = v;
    }

    q->tail = v;
    q->count++;
}

static inline void mv2_ud_send_window_remove(message_queue_t *q, vbuf *v)
{
    MPIU_Assert (q->head == v);
    v->in_sendwin = 0;
    q->head = v->sendwin_msg.next;
    q->count--;
    if (q->head == NULL ) {
        q->tail = NULL;
        MPIU_Assert(q->count == 0);
    }

    v->sendwin_msg.next = NULL;
}    

static inline void mv2_ud_unack_queue_add(message_queue_t *q, vbuf *v)
{
    v->unack_msg.next = NULL;

    if (q->head == NULL) {
        q->head = v;
        v->unack_msg.prev = NULL;
    } else {
        (q->tail)->unack_msg.next = v;
        v->unack_msg.prev = q->tail;
    }

    q->tail = v;
    q->count++;
}

static inline void mv2_ud_unack_queue_remove(message_queue_t *q, vbuf *v)
{
    vbuf *next = v->unack_msg.next;
    vbuf *prev = v->unack_msg.prev;

    if (prev == NULL) {
        q->head = next;
    } else {
        prev->unack_msg.next = next;
    }

    if (next == NULL) {
        q->tail = prev;
    } else {
        next->unack_msg.prev = prev;
    }
    v->unack_msg.next = v->unack_msg.prev = NULL;
    q->count--;
}

static inline void mv2_ud_track_send(mv2_ud_reliability_info_t *ud_vc, message_queue_t *unack_queue, vbuf *v)
{
    rdma_ud_last_check = mv2_get_time_us();
    if (v->transport == IB_TRANSPORT_RC) {
        return;
    }
    if(v->in_sendwin) {
        return;
    }
    v->timestamp = mv2_get_time_us();
    /* Add vbuf to the send window */
    mv2_ud_send_window_add(&(ud_vc->send_window), v);
    /* Add vbuf to global unack queue */
    mv2_ud_unack_queue_add(unack_queue, v);
}

static inline void mv2_ud_unackq_traverse(message_queue_t *q)
{
    int r;
    long  delay;
    double timestamp = mv2_get_time_us();
    vbuf *cur = q->head;

    while(cur) {
        //TODO:: if (cur->left_to_send == 0 || cur->retry_always) {
        delay = timestamp - cur->timestamp;
        if (cur->retry_count > 1) {
            LOG2(cur->retry_count, r);
        } else {
            r = 1;
        }
        if ((delay > (rdma_ud_retry_timeout * r)) 
               || (delay > rdma_ud_max_retry_timeout)) {
            mv2_ud_resend(cur);
            PRINT_DEBUG(DEBUG_UD_verbose>1,"resend seqnum:%d retry : %d \n",cur->seqnum, cur->retry_count);
            cur->timestamp = timestamp;
            timestamp = mv2_get_time_us();
        }
        cur = cur->unack_msg.next;
    } 
}

static inline int mv2_ud_recv_window_add(message_queue_t *q, vbuf *v, int recv_win_start)
{

    PRINT_DEBUG(DEBUG_UD_verbose>1,"recv window add recv_win_start:%d rece'd seqnum:%d\n", recv_win_start, v->seqnum);

    v->recvwin_msg.next = v->recvwin_msg.prev = NULL;
    if(q->head == NULL) {
        q->head = q->tail = v;
    } else {
        vbuf *cur_buf = q->head;
        if (v->seqnum > recv_win_start) {
            if (cur_buf->seqnum < recv_win_start) {
            } else {
                while (NULL != cur_buf && cur_buf->seqnum < v->seqnum 
                        && cur_buf->seqnum > recv_win_start) {
                    cur_buf = cur_buf->recvwin_msg.next;
                }
            }
        } else {
            if (cur_buf->seqnum > recv_win_start) {
                while (NULL != cur_buf && ((cur_buf->seqnum >= recv_win_start)
                            || (cur_buf->seqnum < v->seqnum))) { 
                    cur_buf = cur_buf->recvwin_msg.next;
                }
            } else {
                while (NULL != cur_buf && cur_buf->seqnum < v->seqnum) {
                    cur_buf = cur_buf->recvwin_msg.next;
                }
            }
        }

        if (NULL != cur_buf) {
            if (cur_buf->seqnum == v->seqnum) {
                return MSG_IN_RECVWIN;
            }

            v->recvwin_msg.next = cur_buf;
            v->recvwin_msg.prev = cur_buf->recvwin_msg.prev;

            if (cur_buf == q->head) {
                q->head = v;
            } else {
                ((vbuf *)(cur_buf->recvwin_msg.prev))->recvwin_msg.next = v;
            }
            cur_buf->recvwin_msg.prev = v;
        } else {
            v->recvwin_msg.next = NULL;
            v->recvwin_msg.prev = q->tail;
            q->tail->recvwin_msg.next = v;
            q->tail = v;
        }
        q->count++;
    }
    return MSG_QUEUED_RECVWIN; 
}

static inline void mv2_ud_recv_window_remove(message_queue_t *q)
{
    vbuf *next = (q->head)->recvwin_msg.next;
    q->head = next;
    if (next != NULL) {
        next->recvwin_msg.prev = NULL;
    } else {
        q->head = q->tail = NULL;
    }
    q->count--;
}

