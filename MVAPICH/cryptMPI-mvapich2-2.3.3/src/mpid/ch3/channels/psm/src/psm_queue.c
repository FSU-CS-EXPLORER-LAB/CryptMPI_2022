/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include "mpiimpl.h"
#include "psmpriv.h"
#include "psm_vbuf.h"

MPID_Request psmcomphead;
pthread_spinlock_t reqlock;
pthread_spinlock_t psmlock;
pthread_spinlock_t psmlock_progress;

static void psm_dump_debug();

#undef FUNCNAME
#define FUNCNAME psm_queue_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void psm_queue_init()
{
    pthread_spin_init(&reqlock, 0);
    pthread_spin_init(&psmlock, 0);
    pthread_spin_init(&psmlock_progress, 0);
    psmcomphead.psmcompnext = &psmcomphead;
    psmcomphead.psmcompprev = &psmcomphead;
}
        
/* request complete:  add to completed entry queue,
                      update MPI_Status
                      set cc_ptr
                      release ref count on req
                      update pending send/recv ops
*/
#undef FUNCNAME
#define FUNCNAME psm_complete_req
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_complete_req(MPID_Request *req, PSM_MQ_STATUS_T psmstat)
{
    int count = 0;
    int mpi_errno = MPI_SUCCESS;

    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        pthread_spin_lock(&reqlock);
    }

    req->psm_flags |= PSM_COMPQ_PENDING;
    req->psmcompnext = &psmcomphead;
    req->psmcompprev = psmcomphead.psmcompprev;
    req->psmcompnext->psmcompprev = req;
    req->psmcompprev->psmcompnext = req;

    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        pthread_spin_unlock(&reqlock);
    }

    assert(psmstat.context == req);
    if((&req->status) != MPI_STATUS_IGNORE) {
        psm_update_mpistatus(&(req->status), psmstat, 1);
    }
    MPID_cc_decr(req->cc_ptr, &count);
    if (!count) {
        if (req->request_completed_cb != NULL) {
            mpi_errno = req->request_completed_cb(req);
            if (mpi_errno != MPI_SUCCESS) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

fn_exit:
    return mpi_errno;
fn_fail:
    PRINT_DEBUG(DEBUG_1SC_verbose>1, "request error\n");
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME psm_do_cancel
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_do_cancel(MPID_Request *req)
{
    PSM_ERROR_T psmerr;
    int mpi_errno = MPI_SUCCESS;
    PSM_MQ_STATUS_T status;

    if(req->psm_flags & PSM_SEND_CANCEL) {
        printf("send cancel unsupported\n");
        req->psm_flags &= ~PSM_SEND_CANCEL;
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**psmsendcancel");
    }

    if(req->psm_flags & PSM_RECV_CANCEL) {
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "recv cancel\n");
        req->psm_flags &= ~PSM_RECV_CANCEL;
        _psm_enter_;
        psmerr = PSM_MQ_CANCEL(&(req->mqreq));
        _psm_exit_;
        if(unlikely(psmerr != PSM_OK)) {
            MPIR_ERR_POP(mpi_errno);
        } else {
            psmerr = PSM_TEST(&(req->mqreq), &status);
            if (psmerr == PSM_OK) {
                MPIR_STATUS_SET_CANCEL_BIT(req->status, TRUE);
                MPIR_STATUS_SET_COUNT(req->status, 0);
            } else {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

fn_fail:
    return mpi_errno;    
}

#undef FUNCNAME
#define FUNCNAME psm_process_completion
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_process_completion(MPID_Request *req, PSM_MQ_STATUS_T gblstatus)
{
    int mpi_errno = MPI_SUCCESS;

    /* request is a 1-sided pre-posted receive */
    if(req->psm_flags & PSM_1SIDED_PREPOST) {
        --psm_tot_pposted_recvs;
        mpi_errno = psm_1sided_input(req, gblstatus.nbytes);
        if(mpi_errno)   MPIR_ERR_POP(mpi_errno);
        PRINT_DEBUG(DEBUG_1SC_verbose>1, "1-sided pre-posted completed\n");
        goto fn_exit;
    }

    /* request is a RNDV receive for a GET */
    if(req->psm_flags & (PSM_RNDVRECV_GET_REQ | PSM_GETACCUM_GET_RNDV_REQ)) {
        mpi_errno = psm_getresp_rndv_complete(req, gblstatus.nbytes);
        goto fn_exit;
    }

    /* request is a GET-Response */
    if(req->psm_flags & PSM_GETRESP_REQ) {
        mpi_errno = psm_getresp_complete(req);
        if(mpi_errno)   MPIR_ERR_POP(mpi_errno);
        goto fn_exit;
    }

    /* request is a FOP-Response */
    if(req->psm_flags & PSM_FOPRESP_REQ) {
        mpi_errno = psm_fopresp_complete(req);
        if(mpi_errno)   MPIR_ERR_POP(mpi_errno);
        goto fn_exit;
    }

    /* request is a GET-Accum-response */
    if(req->psm_flags & PSM_GETACCUMRESP_REQ) {
        mpi_errno = psm_getaccumresp_complete(req);
        if(mpi_errno)   MPIR_ERR_POP(mpi_errno);
        goto fn_exit;
    }

    /* request is a RNDV send */
    if(req->psm_flags & PSM_RNDVSEND_REQ) {
        mpi_errno = psm_complete_req(req, gblstatus);
        MPID_Request_release(req);
        goto fn_exit;
    }

    /* request was a GET packet */
    if(req->psm_flags & PSM_GETPKT_REQ) {
        /* completion of GET req is not complete until the
           GET-RESP comes back. */
        goto fn_exit;
    }

    /* request was a PUT/ACCUM RNDV receive */
    if(req->psm_flags & (PSM_RNDVRECV_ACCUM_REQ | PSM_RNDVRECV_PUT_REQ
        | PSM_GETACCUM_RNDV_REQ)) {
        mpi_errno = psm_complete_rndvrecv(req, gblstatus.nbytes);
        if(mpi_errno)   MPIR_ERR_POP(mpi_errno);
        goto fn_exit;
    }

    mpi_errno = psm_complete_req(req, gblstatus);
    MPID_Request_release(req);

fn_exit:
fn_fail:    
    return mpi_errno;
}

/* try to complete a specified request */
#undef FUNCNAME
#define FUNCNAME psm_try_complete
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_try_complete(MPID_Request *req)
{
    int mpi_errno = MPI_SUCCESS;

    while(!MPID_cc_is_complete(req->cc_ptr))
      mpi_errno = psm_progress_wait(TRUE);

    return mpi_errno;
}

/* progress engine:
        peek into PSM. If no completion for 10 spins, get out.
            if MT yield CPU and release global lock
        if we got completion, do mq_test, release PSM lock,
        run completion handler, re-acquire PSM lock and
        back into ipeek.
*/    

#undef FUNCNAME
#define FUNCNAME psm_progress_wait
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_progress_wait(int blocking)
{
    PSM_ERROR_T psmerr;
    PSM_MQ_STATUS_T gblstatus;
    PSM_MQ_REQ_T gblpsmreq;
    int i, made_progress;
    register MPID_Request *req;
    int mpi_errno = MPI_SUCCESS;
    int yield_count = ipath_progress_yield_count;

    _psm_progress_enter_;
    do {


        psmerr = PSM_IPEEK(psmdev_cw.mq, &gblpsmreq, NULL);

        if(psmerr == PSM_OK) {
            psmerr = PSM_TEST(&gblpsmreq, &gblstatus);
            _psm_progress_exit_;
            req = (MPID_Request *) gblstatus.context;
#if PSM_VERNO >= PSM_2_1_VERSION
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "got %llu of %llu bytes from %d\n",
                    gblstatus.nbytes, gblstatus.msg_length, gblstatus.msg_tag.tag1);
#else
            PRINT_DEBUG(DEBUG_CHM_verbose>1, "got bytes from %d\n", (int)(gblstatus.msg_tag & SRC_RANK_MASK));
#endif

            mpi_errno = psm_process_completion(req, gblstatus);
            if(mpi_errno != MPI_SUCCESS) {
                MPIR_ERR_POP(mpi_errno);
            }
            goto out_2;
        }
        else {
            if ((MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) &&
                (--yield_count == 0)) {
                goto out;
            }
        }

        made_progress = FALSE;
        for (i = 0; i < MAX_PROGRESS_HOOKS; i++) {
            if (progress_hooks[i].active == TRUE) {
                MPIU_Assert(progress_hooks[i].func_ptr != NULL);
                mpi_errno = progress_hooks[i].func_ptr(&made_progress);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                if (made_progress) {
                    _psm_progress_exit_;
                    goto out_2; 
                }
            }
        }
    } while (blocking);

out:
    _psm_progress_exit_;
    if(unlikely(ipath_debug_enable)) {
        psm_dump_debug();
    }

    if (MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        psm_pe_yield();
    }

out_2:
fn_fail:
    return mpi_errno;
}

/* remove request from completed queue. called from request release
   in CH3. lock needed for MT */
#undef FUNCNAME
#define FUNCNAME psm_dequeue_compreq
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void psm_dequeue_compreq(MPID_Request *req)
{
    PRINT_DEBUG(DEBUG_CHM_verbose>1, "Request release\n");
    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        pthread_spin_lock(&reqlock);
    }
       
    assert(req != (&psmcomphead));
    assert(req->psmcompnext);
    assert(req->psmcompprev);
    req->psmcompnext->psmcompprev = req->psmcompprev;
    req->psmcompprev->psmcompnext = req->psmcompnext;    
    
    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        pthread_spin_unlock(&reqlock);
    }
}

#undef FUNCNAME
#define FUNCNAME psm_probe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
PSM_ERROR_T psm_probe(int src, int tag, int context, MPI_Status *stat)
{
    #if PSM_VERNO >= PSM_2_1_VERSION
        psm2_mq_tag_t rtag, rtagsel;
    #else
        uint64_t rtag, rtagsel;
    #endif
    PSM_MQ_STATUS_T gblstatus;
    PSM_ERROR_T psmerr;

    #if PSM_VERNO >= PSM_2_1_VERSION
        rtagsel.tag0 = MQ_TAGSEL_ALL;
        rtagsel.tag1 = MQ_TAGSEL_ALL;
        rtagsel.tag2 = MQ_TAGSEL_ALL;
        if(unlikely(tag == MPI_ANY_TAG))
            rtagsel.tag0 = MQ_TAGSEL_ANY_TAG;
        if(unlikely(src == MPI_ANY_SOURCE))
            rtagsel.tag1 = MQ_TAGSEL_ANY_SOURCE;
    #else
        rtag = 0;
        rtagsel = MQ_TAGSEL_ALL;
        if(unlikely(src == MPI_ANY_SOURCE))
            rtagsel = rtagsel & MQ_TAGSEL_ANY_SOURCE;
        if(unlikely(tag == MPI_ANY_TAG))
            rtagsel = rtagsel & MQ_TAGSEL_ANY_TAG;
    #endif

    MAKE_PSM_SELECTOR(rtag, context, tag, src);
    
    _psm_enter_;
    psmerr = PSM_IPROBE(psmdev_cw.mq, rtag, rtagsel, &gblstatus);
    _psm_exit_;
    if(psmerr == PSM_OK) {
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "one psm probe completed\n");
        if(stat != MPI_STATUS_IGNORE) {
            psm_update_mpistatus(stat, gblstatus, 0);
        }
    }

    return psmerr;    
}

#undef FUNCNAME
#define FUNCNAME psm_mprobe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
PSM_ERROR_T psm_mprobe(int src, int tag, int context,
        MPID_Request *req, MPI_Status *stat)
{
    PSM_ERROR_T psmerr = PSM_OK;
#if PSM_VERNO >= PSM_2_1_VERSION
    psm2_mq_tag_t rtag, rtagsel;
    PSM_MQ_STATUS_T gblstatus;

    rtagsel.tag0 = MQ_TAGSEL_ALL;
    rtagsel.tag1 = MQ_TAGSEL_ALL;
    rtagsel.tag2 = MQ_TAGSEL_ALL;
    if(unlikely(tag == MPI_ANY_TAG))
        rtagsel.tag0 = MQ_TAGSEL_ANY_TAG;
    if(unlikely(src == MPI_ANY_SOURCE))
        rtagsel.tag1 = MQ_TAGSEL_ANY_SOURCE;

    MAKE_PSM_SELECTOR(rtag, context, tag, src);

    _psm_enter_;
    PRINT_DEBUG(DEBUG_CHM_verbose>1, "calling psm improbe2\n");
    psmerr = PSM_IMPROBE(psmdev_cw.mq, rtag, rtagsel, req->mqreq, &gblstatus);
    _psm_exit_;
    if(psmerr == PSM_OK) {
        PRINT_DEBUG(DEBUG_CHM_verbose>1, "one psm matched probe completed\n");
        if (gblstatus.error_code == PSM_OK) {
            req->status.MPI_TAG = gblstatus.msg_tag.tag0;
            req->status.MPI_SOURCE = gblstatus.msg_tag.tag1;
            MPIR_STATUS_SET_COUNT(req->status, gblstatus.nbytes);
            if(stat != MPI_STATUS_IGNORE) {
                MPIR_Request_extract_status(req, stat);
            }
        }
    }
#endif

    return psmerr;
}

int psm_no_lock(pthread_spinlock_t *lock) 
{  
    /* no-lock needed */
    return 0;
}

/* in multi-threaded mode, the mpich2 global lock has to be released
   by the progress engine for other threads to enter MPID layer */
#undef FUNCNAME
#define FUNCNAME psm_pe_yield
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void psm_pe_yield()
{
    pthread_mutex_unlock((pthread_mutex_t *)&(MPIR_ThreadInfo.global_mutex));
    sched_yield();
    pthread_mutex_lock((pthread_mutex_t *)&(MPIR_ThreadInfo.global_mutex));
}

#undef FUNCNAME
#define FUNCNAME psm_update_mpistatus
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
void psm_update_mpistatus(MPI_Status *stat, PSM_MQ_STATUS_T psmst, int append)
{
    MPIDI_msg_sz_t old_nbytes = 0;

    switch(psmst.error_code) {
        case PSM_OK:    
            stat->MPI_ERROR = MPI_SUCCESS;
            break;
        case PSM_MQ_TRUNCATION:
            stat->MPI_ERROR = MPI_ERR_TRUNCATE;
            break;
        default:
            break;
    }

    #if PSM_VERNO >= PSM_2_1_VERSION
        stat->MPI_TAG = psmst.msg_tag.tag0;
        stat->MPI_SOURCE = psmst.msg_tag.tag1;
    #else
        stat->MPI_TAG = (psmst.msg_tag >> SRC_RANK_BITS) & TAG_MASK;
        stat->MPI_SOURCE = psmst.msg_tag & SRC_RANK_MASK;
    #endif

    if (append) {
        old_nbytes = MPIR_STATUS_GET_COUNT(*stat);
    }
    MPIR_STATUS_SET_COUNT(*stat, (psmst.nbytes + old_nbytes));
}

/* if PSM_DEBUG is enabled, we will dump some counters */

static void psm_dump_debug()
{
    static time_t  timedump;
    struct tm *ts;
    char buf[80];
    time_t last;
    int rank;
    extern uint32_t ipath_dump_frequency;

    last = time(NULL);
    
    if((last - timedump) < ipath_dump_frequency) 
        return;
    UPMI_GET_RANK(&rank);

    fprintf(stderr, "[%d]------- PSM COUNTERS---------\n", rank);
    fprintf(stderr, "[%d] Total SENDS\t\t%d\n", rank, psm_tot_sends);
    fprintf(stderr, "[%d] Total RECVS\t\t%d\n", rank, psm_tot_recvs);
    fprintf(stderr, "[%d] Total pre-posted receives\t\t%d\n", rank, psm_tot_pposted_recvs);
    fprintf(stderr, "[%d] Total eager PUTS\t\t%d\n", rank, psm_tot_eager_puts);
    fprintf(stderr, "[%d] Total eager GETS\t\t%d\n", rank, psm_tot_eager_gets);
    fprintf(stderr, "[%d] Total rendezvous PUTS\t\t%d\n", rank, psm_tot_rndv_puts);
    fprintf(stderr, "[%d] Total rendezvous GETS\t\t%d\n", rank, psm_tot_rndv_gets);
    fprintf(stderr, "[%d] Total ACCUMULATES\t\t%d\n", rank, psm_tot_accs);
    ts = localtime(&last);
    strftime(buf, sizeof(buf), "%a %Y-%m-%d %H:%M:%S %Z", ts);
    fprintf(stderr, "[%d] ------Time of dump %s-----\n", rank, buf);
    timedump = last;
}
