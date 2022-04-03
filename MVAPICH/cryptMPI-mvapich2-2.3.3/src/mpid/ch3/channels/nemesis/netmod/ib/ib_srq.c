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

#define _GNU_SOURCE
#include <mpiimpl.h>
#include "ib_process.h"
#include "ib_errors.h"
#include "ib_srq.h"
#include "upmi.h"

MPID_nem_ib_srq_info_t srq_info;

static pthread_spinlock_t g_apm_lock;

int power_two(int x)
{
    int pow = 1;

    while (x) {
        pow = pow * 2;
        x--;
    }

    return pow;
}

#undef FUNCNAME
#define FUNCNAME lock_apm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static void lock_apm()
{
    pthread_spin_lock(&g_apm_lock);
    return;
}

#undef FUNCNAME
#define FUNCNAME unlock_apm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static void unlock_apm()
{
    pthread_spin_unlock(&g_apm_lock);
    return;
}

/* This function is used for implmeneting  "Alternate Path Specification"
 * and "Path Loading Request Module", (SMTPS 2007 Paper) */

/* Description:
 * sl: service level, which can be changed once the QoS is enabled
 *
 * alt_timeout: alternate timeout, which can be increased to maximum, which
 * may make the failover little slow
 *
 * path_mig_state: Path migration state, since we are combining the modules
 * this should be the path migration state of the QP
 *
 * pkey_index: Index of the partition
 *
 * static_rate: typically "0" is the safest, which corresponds to the
 * maximum value of the static rate
 *
 * Alternate Path can be specified in multiple ways:
 * Using Same Port, but different LMC (common case)
 * rdma_num_qp_per_port refers to the number of QPs per port. The QPs
 * will use the first "rdma_num_qp_per_port" set of paths, and their
 * alternate paths are specified by a displacement of rdma_num_qp_per_port.
 * As an example, with rdma_num_qp_per_port = 4, first QP will use path0 and
 * path4, second QP will use path1 and path5 and so on, as the primary and
 * alternate path respectively.
 *
 * Finally, Since this function also implements Path Loading Rquest Module,
 * it should modify the QP with the specification of the alternate path */

 int reload_alternate_path(struct ibv_qp *qp)
{

    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    enum ibv_qp_attr_mask attr_mask;

    lock_apm();

    /* For Sanity */
    memset(&attr, 0, sizeof attr);
    memset(&init_attr, 0, sizeof init_attr);

    attr_mask = 0;

    if (ibv_query_qp(qp, &attr,
                attr_mask, &init_attr)) {
        ibv_error_abort(GEN_EXIT_ERR, "Failed to query QP\n");
    }

    /* This value should change with enabling of QoS */
    attr.alt_ah_attr.sl =  attr.ah_attr.sl;
    attr.alt_ah_attr.static_rate = attr.ah_attr.static_rate;
    attr.alt_ah_attr.port_num =  attr.ah_attr.port_num;
    attr.alt_ah_attr.is_global =  attr.ah_attr.is_global;
    attr.alt_timeout = attr.timeout;
    attr.alt_port_num = attr.port_num;
    attr.alt_ah_attr.src_path_bits =
        (attr.ah_attr.src_path_bits + rdma_num_qp_per_port) %
        power_two(process_info.lmc);
    attr.alt_ah_attr.dlid =
        attr.ah_attr.dlid - attr.ah_attr.src_path_bits
        + attr.alt_ah_attr.src_path_bits;
    attr.path_mig_state = IBV_MIG_REARM;
    attr_mask = 0;
    attr_mask |= IBV_QP_ALT_PATH;
    attr_mask |= IBV_QP_PATH_MIG_STATE;

    if (ibv_modify_qp(qp, &attr, attr_mask))
    {
        ibv_error_abort(GEN_EXIT_ERR, "Failed to modify QP\n");
    }

    unlock_apm();

    return 0;
}

void async_thread(void *context)
{
    struct ibv_async_event event;
    struct ibv_srq_attr srq_attr;
    int post_new, i, hca_num = -1;
#ifdef _ENABLE_XRC_
    int xrc_event = 0;
#endif

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

    while (1) {
        if (ibv_get_async_event((struct ibv_context *) context, &event)) {
            fprintf(stderr, "Error getting event!\n");
        }

        for(i = 0; i < ib_hca_num_hcas; i++) {
            if(hca_list[i].nic_context == context) {
                hca_num = i;
            }
        }

        pthread_mutex_lock(&srq_info.async_mutex_lock[hca_num]);
#ifdef _ENABLE_XRC_
        if (event.event_type & IBV_XRC_QP_EVENT_FLAG) {
            event.event_type ^= IBV_XRC_QP_EVENT_FLAG;
            xrc_event = 1;
        }
#endif

        switch (event.event_type) {
            /* Fatal */
            case IBV_EVENT_CQ_ERR:
            case IBV_EVENT_QP_FATAL:
            case IBV_EVENT_QP_REQ_ERR:
            case IBV_EVENT_QP_ACCESS_ERR:
                ibv_va_error_abort(GEN_EXIT_ERR, "Got FATAL event %d\n",
                        event.event_type);
                break;
            case IBV_EVENT_PATH_MIG_ERR:
#ifdef DEBUG
                if(process_info.has_apm) {
                    DEBUG_PRINT("Path Migration Failed\n");
                }
#endif /* ifdef DEBUG */
                ibv_va_error_abort(GEN_EXIT_ERR, "Got FATAL event %d\n",
                        event.event_type);
                break;
            case IBV_EVENT_PATH_MIG:
                if(process_info.has_apm && !apm_tester){
                    DEBUG_PRINT("Path Migration Successful\n");
                    reload_alternate_path((&event)->element.qp);
                }

                if(!process_info.has_apm) {
                    ibv_va_error_abort(GEN_EXIT_ERR, "Got FATAL event %d\n",
                            event.event_type);
                }

                break;

            case IBV_EVENT_DEVICE_FATAL:
            case IBV_EVENT_SRQ_ERR:
                ibv_va_error_abort(GEN_EXIT_ERR, "Got FATAL event %d\n",
                        event.event_type);
                break;

            case IBV_EVENT_COMM_EST:
            case IBV_EVENT_PORT_ACTIVE:
            case IBV_EVENT_SQ_DRAINED:
            case IBV_EVENT_PORT_ERR:
            case IBV_EVENT_LID_CHANGE:
            case IBV_EVENT_PKEY_CHANGE:
            case IBV_EVENT_SM_CHANGE:
            case IBV_EVENT_QP_LAST_WQE_REACHED:
                break;

            case IBV_EVENT_SRQ_LIMIT_REACHED:

                pthread_spin_lock(&srq_info.srq_post_spin_lock);

                if(-1 == hca_num) {
                    /* Was not able to find the context,
                     * error condition */
                    ibv_error_abort(GEN_EXIT_ERR,
                            "Couldn't find out SRQ context\n");
                }

                /* dynamically re-size the srq to be larger */
                mv2_srq_fill_size *= 2;
                if (mv2_srq_fill_size > mv2_srq_alloc_size) {
                    mv2_srq_fill_size = mv2_srq_alloc_size;
                }

                rdma_credit_preserve = (mv2_srq_fill_size > 200) ?
                     (mv2_srq_fill_size - 100) : (mv2_srq_fill_size / 2);

                /* Need to post more to the SRQ */
                post_new = srq_info.posted_bufs[hca_num];

                srq_info.posted_bufs[hca_num] +=
                    MPIDI_nem_ib_post_srq_buffers(mv2_srq_fill_size -
                            mv2_srq_limit, hca_num);

                post_new = srq_info.posted_bufs[hca_num] -
                    post_new;

                pthread_spin_unlock(&srq_info.
                        srq_post_spin_lock);

                if(!post_new) {
                    pthread_mutex_lock(
                            &srq_info.
                            srq_post_mutex_lock[hca_num]);

                    ++srq_info.srq_zero_post_counter[hca_num];

                    while(srq_info.
                            srq_zero_post_counter[hca_num] >= 1) {
                        /* Cannot post to SRQ, since all WQEs
                         * might be waiting in CQ to be pulled out */
                        pthread_cond_wait(
                                &srq_info.
                                srq_post_cond[hca_num],
                                &srq_info.
                                srq_post_mutex_lock[hca_num]);
                    }
                    pthread_mutex_unlock(&srq_info.
                            srq_post_mutex_lock[hca_num]);
                } else {
                    /* Was able to post some, so erase old counter */
                    if(srq_info.
                            srq_zero_post_counter[hca_num]) {
                        srq_info.
                            srq_zero_post_counter[hca_num] = 0;
                }
                }

                pthread_spin_lock(&srq_info.srq_post_spin_lock);

                srq_attr.max_wr = mv2_srq_fill_size;
                srq_attr.max_sge = 1;
                srq_attr.srq_limit = mv2_srq_limit;

                if (ibv_modify_srq(hca_list[hca_num].srq_hndl,
                            &srq_attr, IBV_SRQ_LIMIT)) {
                    ibv_va_error_abort(GEN_EXIT_ERR,
                            "Couldn't modify SRQ limit (%u) after posting %d\n",
                            mv2_srq_limit, post_new);
                }

                pthread_spin_unlock(&srq_info.srq_post_spin_lock);

                break;
            default:
                fprintf(stderr,
                        "Got unknown event %d ... continuing ...\n",
                        event.event_type);
        }
#ifdef _ENABLE_XRC_
        if (xrc_event) {
            event.event_type |= IBV_XRC_QP_EVENT_FLAG;
            xrc_event = 0;
        }
#endif
        ibv_ack_async_event(&event);
        pthread_mutex_unlock(&srq_info.async_mutex_lock[hca_num]);
    }
}

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_allocate_srq
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_nem_ib_allocate_srq()
{
    int mpi_errno = MPI_SUCCESS;
    int hca_num = 0;

        pthread_spin_init(&srq_info.srq_post_spin_lock, 0);
        pthread_spin_lock(&srq_info.srq_post_spin_lock);

        for (; hca_num < ib_hca_num_hcas; ++hca_num)
        {
            pthread_mutex_init(&srq_info.srq_post_mutex_lock[hca_num], 0);
            pthread_cond_init(&srq_info.srq_post_cond[hca_num], 0);
            srq_info.srq_zero_post_counter[hca_num] = 0;
            srq_info.posted_bufs[hca_num] =
                MPIDI_nem_ib_post_srq_buffers(mv2_srq_fill_size, hca_num);

            {
                struct ibv_srq_attr srq_attr;
                srq_attr.max_wr = mv2_srq_alloc_size;
                srq_attr.max_sge = 1;
                srq_attr.srq_limit = mv2_srq_limit;

                if (ibv_modify_srq(
                    hca_list[hca_num].srq_hndl,
                    &srq_attr,
                    IBV_SRQ_LIMIT))
                {
                    ibv_error_abort(IBV_RETURN_ERR, "Couldn't modify SRQ limit\n");
                }

                /* Start the async thread which watches for SRQ limit events */
                pthread_create(
                    &srq_info.async_thread[hca_num],
                    NULL,
                    (void *) async_thread,
                    (void *) hca_list[hca_num].nic_context);
            }
        }

        pthread_spin_unlock(&srq_info.srq_post_spin_lock);
        return mpi_errno;
}

int MPIDI_nem_ib_post_srq_buffers(int num_bufs,
        int hca_num)
{
    int i = 0;
    vbuf* v = NULL;
    struct ibv_recv_wr* bad_wr = NULL;

    if (num_bufs > mv2_srq_fill_size)
    {
        ibv_va_error_abort(
            GEN_ASSERT_ERR,
            "Try to post %d to SRQ, max %d\n",
            num_bufs,
            mv2_srq_fill_size);
    }

    for (; i < num_bufs; ++i)
    {
        if ((v = get_vbuf()) == NULL)
        {
            break;
        }

        vbuf_init_recv(
            v,
            VBUF_BUFFER_SIZE,
            hca_num * ib_hca_num_ports * rdma_num_qp_per_port);

        if (ibv_post_srq_recv(hca_list[hca_num].srq_hndl, &v->desc.u.rr, &bad_wr))
        {
            MRAILI_Release_vbuf(v);
            break;
        }
    }

    return i;
}

