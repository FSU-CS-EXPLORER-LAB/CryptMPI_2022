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

#ifndef _PSMPRIV_H
#define _PSMPRIV_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include "mpichconf.h"
#ifdef HAVE_LIBPSM2
    #include <psm2.h>
    #include <psm2_mq.h>
    /* PSM2 lib changed the names of PSM_* defines and env vars to PSM2_*.
     * Since PSM2 does not define PSM_* defines, it is easier to add aliases
     * from PSM_ to PSM2_ then have extra #ifdef PSM_VERNO blocks. */
    #define PSM_VERNO                   PSM2_VERNO
    #define PSM_VERNO_MAJOR             PSM2_VERNO_MAJOR
    #define PSM_VERNO_MINOR             PSM2_VERNO_MINOR
    #define PSM_OK                      PSM2_OK
    #define PSM_EP_CLOSE_GRACEFUL       PSM2_EP_CLOSE_GRACEFUL
    #define PSM_EP_OPEN_AFFINITY_SKIP   PSM2_EP_OPEN_AFFINITY_SKIP
    #define PSM_MQ_NO_COMPLETIONS       PSM2_MQ_NO_COMPLETIONS
    #define PSM_MQ_ORDERMASK_ALL        PSM2_MQ_ORDERMASK_ALL
    #define PSM_MQ_TRUNCATION           PSM2_MQ_TRUNCATION
    #define PSM_MQ_RNDV_IPATH_SZ        PSM2_MQ_RNDV_HFI_SZ
    #define PSM_MQ_RNDV_SHM_SZ          PSM2_MQ_RNDV_SHM_SZ
    #define PSM_MQ_FLAG_SENDSYNC        PSM2_MQ_FLAG_SENDSYNC
    /* PSM2 claims a max transfer limit of 4GB 
     * Experiments show issues for inter-node messages
     * Limiting to 2GB for now */
    #define DEFAULT_IPATH_MAX_TRANSFER_SIZE (2L*1024*1024*1024)
    #define DEFAULT_IPATH_RNDV_THRESH   (64*1000)
    #define DEFAULT_PSM_HFI_RNDV_THRESH (64*1000)
    #define DEFAULT_PSM_SHM_RNDV_THRESH (32*1024)
#elif HAVE_LIBPSM_INFINIPATH
    #include <psm.h>
    #include <psm_mq.h>
    /* Currently PSM has a max transfer limit of 1GB */
    #define DEFAULT_IPATH_MAX_TRANSFER_SIZE (1L*1024*1024*1024)
    #define DEFAULT_IPATH_RNDV_THRESH   (64*1000)
    #define DEFAULT_PSM_HFI_RNDV_THRESH (64*1000)
    #define DEFAULT_PSM_SHM_RNDV_THRESH (32*1024)
#endif

#define PSM_2_1_VERSION         0x0201
#define PSM_2_VERSION_MAJOR     0x02

#include <string.h>
#include <pthread.h>
#include "mpidimpl.h"
#include "upmi.h"
#include <debug_utils.h>

#define MPID_PSM_UUID           "uuid"      /* pmi key for uuid */
#define WRBUFSZ                 1024        /* scratch buffer */
#define ROOT                    0           
#define TIMEOUT                 50          /* connect timeout */
#define MQ_FLAGS_NONE           0 

#define MPIDI_PSM_DEFAULT_ON_DEMAND_THRESHOLD   64

/* tag selection macros, taken from mvapich-psm code */
#if PSM_VERNO >= PSM_2_1_VERSION
    /* PSM2 uses two instances of the new type psm2_mq_tag_t to hold the
     * PSM tag and the tag selector. */
    #define MQ_TAGSEL_ALL           0xffffffff
    #define MQ_TAGSEL_ANY_TAG       0x00000000
    #define MQ_TAGSEL_ANY_SOURCE    0x00000000
#else
    /* PSM1 uses a single 64-bit number to hold the PSM tag. */
    #define MQ_TAGSEL_ALL           0xffffffffffffffff
    #define TAG_BITS                32
    #define TAG_MASK                ~(MQ_TAGSEL_ALL << TAG_BITS)
    #define SRC_RANK_BITS           16
    #define SRC_RANK_MASK           ~(MQ_TAGSEL_ALL << SRC_RANK_BITS)
    #define MQ_TAGSEL_ANY_TAG       ~(TAG_MASK << SRC_RANK_BITS)
    #define MQ_TAGSEL_ANY_SOURCE    (MQ_TAGSEL_ALL << SRC_RANK_BITS)
#endif
#define SEC_IN_NS               1000000000ULL

#define PSM_ERR_ABORT(args...) do {                                          \
    int __rank; UPMI_GET_RANK(&__rank);                                       \
    fprintf(stderr, "[Rank %d][%s: line %d]", __rank ,__FILE__, __LINE__);   \
    fprintf(stderr, args);                                                   \
    fprintf(stderr, "\n");                                                   \
    fflush(stderr);                                                          \
}while (0)

#if PSM_VERNO >= PSM_2_1_VERSION
/* For PSM2, there is a performance enchancement that uses the MPI rank to
 * place messages in separate hash buckets for faster message matching.
 * In order to take advantage of this feature, the PSM2 tag must be in the
 * order: user tag, rank, context id. PSM2 expects the PSM tag to be in
 * this order to extract the rank id. */
    #define MAKE_PSM_SELECTOR(out, cid, tag, rank) do { \
        out.tag0 = tag;                                 \
        out.tag1 = rank;                                \
        out.tag2 = cid;                                 \
    } while(0)
#else
    #define MAKE_PSM_SELECTOR(out, cid, tag, rank) do { \
        out = cid;                                      \
        out = out << TAG_BITS;                          \
        out = out | (tag & TAG_MASK);                   \
        out = out << SRC_RANK_BITS;                     \
        out = out | (rank & SRC_RANK_MASK);             \
    } while(0)
#endif

/* Macros for adding abstractions for psm_mq_send, psm_mq_isend, psm_mq_irecv, psm_mq_iprobe.
 * These macros exist to avoid having "PSM_VERSION >= PSM_2_1_VERSION" multiple times in the code.
 * These macros call in to the correct PSM function for the PSM version used, i.e.: psm_mq_send or psm_mq_send2
 */
#if PSM_VERNO >= PSM_2_1_VERSION
    /* Functions added/edited in PSM 2 */
    #define PSM_SEND(mq, epaddr, flags, stag, buf, buflen)                      psm2_mq_send2(mq, epaddr, flags, &stag, buf, buflen)
    #define PSM_ISEND(mq, epaddr, flags, stag, buf, buflen, req, mqreq)         psm2_mq_isend2(mq, epaddr, flags, &stag, buf, buflen, req, mqreq)
    #define PSM_ISEND_PTR(mq, epaddr, flags, stag, buf, buflen, req, mqreq)     psm2_mq_isend2(mq, epaddr, flags, stag, buf, buflen, req, mqreq)
    #define PSM_LARGE_ISEND(rptr, dest, buf, buflen, stag, flags)               psm_large_msg_isend_pkt(rptr, dest, buf, buflen, &stag, flags)
    #define PSM_IRECV(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)        psm2_mq_irecv2(mq, PSM2_MQ_ANY_ADDR, &rtag, &rtagsel, flags, buf, buflen, req, mqreq)
    #define PSM_IRECV_PTR(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)    psm2_mq_irecv2(mq, PSM2_MQ_ANY_ADDR, rtag, rtagsel, flags, buf, buflen, req, mqreq)
    #define PSM_LARGE_IRECV(buf, buflen, request, rtag, rtagsel)                psm_post_large_msg_irecv(buf, buflen, request, &rtag, &rtagsel)
    #define PSM_IPROBE(mq, rtag, rtagsel, status)                               psm2_mq_iprobe2(mq, PSM2_MQ_ANY_ADDR, &rtag, &rtagsel, status)
    #define PSM_IMPROBE(mq, rtag, rtagsel, mqreq, status)                       psm2_mq_improbe2(mq, PSM2_MQ_ANY_ADDR, &rtag, &rtagsel, &mqreq, status)
    #define PSM_IMRECV(mq, buf, buflen, req, mqreq)                             psm2_mq_imrecv(mq, MQ_FLAGS_NONE, buf, buflen, req, mqreq)
    #define PSM_TEST(req, status)                                               psm2_mq_test2(req,status)
    #define PSM_IPEEK(mq, req, stat)                                            psm2_mq_ipeek2(mq, req, stat)
    #define PSM_WAIT(mq, status)                                                psm2_mq_wait2(mq, status)

    /* Symbols that have a renamed API in PSM2 but same functionality  */
    #define PSM_POLL                                                            psm2_poll
    #define PSM_MQ_STATUS_T                                                     psm2_mq_status2_t
    #define PSM_MQ_INIT                                                         psm2_mq_init
    #define PSM_MQ_FINALIZE                                                     psm2_mq_finalize
    #define PSM_MQ_CANCEL                                                       psm2_mq_cancel
    #define PSM_MQ_T                                                            psm2_mq_t
    #define PSM_EP_T                                                            psm2_ep_t
    #define PSM_EPID_T                                                          psm2_epid_t
    #define PSM_UUID_T                                                          psm2_uuid_t
    #define PSM_INIT                                                            psm2_init
    #define PSM_EPADDR_T                                                        psm2_epaddr_t
    #define PSM_ERROR_TOKEN_T                                                   psm2_error_token_t
    #define PSM_EP_OPEN_OPTS                                                    psm2_ep_open_opts
    #define PSM_MQ_GETOPT                                                       psm2_mq_getopt
    #define PSM_MQ_SETOPT                                                       psm2_mq_setopt
    #define PSM_FINALIZE                                                        psm2_finalize
    #define PSM_ERROR_T                                                         psm2_error_t
    #define PSM_ERROR_REGISTER_HANDLER                                          psm2_error_register_handler
    #define PSM_ERROR_GET_STRING                                                psm2_error_get_string
    #define PSM_EP_OPEN_OPTS_GET_DEFAULTS                                       psm2_ep_open_opts_get_defaults
    #define PSM_EP_OPEN                                                         psm2_ep_open
    #define PSM_EP_CONNECT                                                      psm2_ep_connect
    #define PSM_EP_CLOSE                                                        psm2_ep_close
    #define PSM_UUID_GENERATE                                                   psm2_uuid_generate
    #ifndef PSM_MQ_REQ_T
       #define PSM_MQ_REQ_T                                                     psm2_mq_req_t
    #endif
#else
    #define PSM_SEND(mq, epaddr, flags, stag, buf, buflen)                      psm_mq_send(mq, epaddr, flags, stag, buf, buflen)
    #define PSM_ISEND(mq, epaddr, flags, stag, buf, buflen, req, mqreq)         psm_mq_isend(mq, epaddr, flags, stag, buf, buflen, req, mqreq)
    #define PSM_ISEND_PTR(mq, epaddr, flags, stag, buf, buflen, req, mqreq)     psm_mq_isend(mq, epaddr, flags, stag, buf, buflen, req, mqreq)
    #define PSM_LARGE_ISEND(rptr, dest, buf, buflen, stag, flags)               psm_large_msg_isend_pkt(rptr, dest, buf, buflen, stag, flags)
    #define PSM_IRECV(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)        psm_mq_irecv(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)
    #define PSM_IRECV_PTR(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)    psm_mq_irecv(mq, rtag, rtagsel, flags, buf, buflen, req, mqreq)
    #define PSM_LARGE_IRECV(buf, buflen, request, rtag, rtagsel)                psm_post_large_msg_irecv(buf, buflen, request, rtag, rtagsel)
    #define PSM_IPROBE(mq, rtag, rtagsel, status)                               psm_mq_iprobe(mq, rtag, rtagsel, status)
    #define PSM_TEST(req, status)                                               psm_mq_test(req,status)
    #define PSM_IPEEK(mq, req, stat)                                            psm_mq_ipeek(mq, req, stat)
    #define PSM_WAIT(mq, status)                                                psm_mq_wait(mq, status)
    /* Matched Probe/Recv Not Supported */
    #define PSM_IMPROBE(mq, rtag, rtagsel, mqreq, status)
    #define PSM_IMRECV(mq, buf, buflen, req, mqreq)

    #define PSM_POLL                                                            psm_poll
    #define PSM_MQ_STATUS_T                                                     psm_mq_status_t
    #define PSM_MQ_INIT                                                         psm_mq_init
    #define PSM_MQ_FINALIZE                                                     psm_mq_finalize
    #define PSM_MQ_CANCEL                                                       psm_mq_cancel
    #define PSM_MQ_T                                                            psm_mq_t
    #define PSM_EP_T                                                            psm_ep_t
    #define PSM_EPID_T                                                          psm_epid_t
    #define PSM_UUID_T                                                          psm_uuid_t
    #define PSM_EPADDR_T                                                        psm_epaddr_t
    #define PSM_ERROR_TOKEN_T                                                   psm_error_token_t
    #define PSM_EP_OPEN_OPTS                                                    psm_ep_open_opts
    #define PSM_MQ_GETOPT                                                       psm_mq_getopt
    #define PSM_MQ_SETOPT                                                       psm_mq_setopt
    #define PSM_INIT                                                            psm_init
    #define PSM_FINALIZE                                                        psm_finalize
    #define PSM_ERROR_T                                                         psm_error_t
    #define PSM_ERROR_REGISTER_HANDLER                                          psm_error_register_handler
    #define PSM_ERROR_GET_STRING                                                psm_error_get_string
    #define PSM_EP_OPEN_OPTS_GET_DEFAULTS                                       psm_ep_open_opts_get_defaults
    #define PSM_EP_OPEN                                                         psm_ep_open
    #define PSM_EP_CONNECT                                                      psm_ep_connect
    #define PSM_EP_CLOSE                                                        psm_ep_close
    #define PSM_UUID_GENERATE                                                   psm_uuid_generate
    #ifndef PSM_MQ_REQ_T
       #define PSM_MQ_REQ_T                                                     psm_mq_req_t
    #endif
#endif

#define CAN_BLK_PSM(_len) ((MPIR_ThreadInfo.thread_provided != MPI_THREAD_MULTIPLE) &&  \
                             (_len < ipath_rndv_thresh))

int psm_no_lock(pthread_spinlock_t *);
int (*psm_lock_fn)(pthread_spinlock_t *);
int (*psm_unlock_fn)(pthread_spinlock_t *);
int (*psm_progress_lock_fn)(pthread_spinlock_t *);
int (*psm_progress_unlock_fn)(pthread_spinlock_t *);

#define MAX_PROGRESS_HOOKS 4
typedef int (*progress_func_ptr_t) (int* made_progress);

typedef struct progress_hook_slot {
    progress_func_ptr_t func_ptr;
    int active;
} progress_hook_slot_t;

progress_hook_slot_t progress_hooks[MAX_PROGRESS_HOOKS];

#define _psm_enter_  psm_lock_fn(&psmlock)
#define _psm_exit_   psm_unlock_fn(&psmlock)

#define _psm_progress_enter_  psm_progress_lock_fn(&psmlock_progress)
#define _psm_progress_exit_   psm_progress_unlock_fn(&psmlock_progress)

#define PSM_COUNTERS    9 

struct psmdev_info_t {
    /* PSM1 structures renamed in PSM2 have the sam functionality for
     * backwards compatibility.  */
    PSM_EP_T        ep;
    PSM_MQ_T        mq;
    PSM_EPADDR_T    *epaddrs;
    PSM_EPID_T      epid;
    int             pg_rank;
    int             pg_size;
    uint16_t        cnt[PSM_COUNTERS];
};

#define psm_tot_sends           psmdev_cw.cnt[0]
#define psm_tot_recvs           psmdev_cw.cnt[1]
#define psm_tot_pposted_recvs   psmdev_cw.cnt[2]
#define psm_tot_eager_puts      psmdev_cw.cnt[3]
#define psm_tot_eager_accs      psmdev_cw.cnt[4]
#define psm_tot_rndv_puts       psmdev_cw.cnt[5]
#define psm_tot_eager_gets      psmdev_cw.cnt[6]
#define psm_tot_rndv_gets       psmdev_cw.cnt[7]
#define psm_tot_accs            psmdev_cw.cnt[8]

#define PSM_ADDR_RESOLVED(peer) (psmdev_cw.epaddrs[peer] != NULL)

/* externs */
extern struct psmdev_info_t psmdev_cw;
extern uint32_t             ipath_rndv_thresh;
extern uint32_t             hfi_rndv_thresh;
extern uint32_t             shm_rndv_thresh;
extern uint8_t              ipath_debug_enable;
extern uint8_t                 ipath_enable_func_lock;
extern uint32_t                ipath_progress_yield_count;
extern pthread_spinlock_t   psmlock;
extern pthread_spinlock_t   psmlock_progress;
extern size_t ipath_max_transfer_size;



typedef enum{
    PACK_RMA_STREAM = 0,
    PACK_NON_STREAM     
}psm_pack_type; 

void psm_queue_init();
int psm_dofinalize();
int psm_do_cancel(MPID_Request *req);
PSM_ERROR_T psm_probe(int src, int tag, int context, MPI_Status *stat);
PSM_ERROR_T psm_mprobe(int src, int tag, int context, MPID_Request *req, MPI_Status *stat);
void psm_init_1sided();
int psm_doinit(int has_parent, MPIDI_PG_t *pg, int pg_rank);   
int psm_connect_peer(int peer);
int psm_istartmsgv(MPIDI_VC_t *vc, MPL_IOV *iov, int iov_n, MPID_Request **rptr);
/* PSM2 uses psm2_mq_tag_t instead of a uint64_t. */
#if PSM_VERNO >= PSM_2_1_VERSION
    int psm_post_large_msg_irecv(void *buf, MPIDI_msg_sz_t buflen, MPID_Request **request, psm2_mq_tag_t *rtag, psm2_mq_tag_t *rtagsel);
#else
    int psm_post_large_msg_irecv(void *buf, MPIDI_msg_sz_t buflen, MPID_Request **request, uint64_t rtag, uint64_t rtagsel);
#endif
int psm_recv(int rank, int tag, int context_id, void *buf, MPIDI_msg_sz_t buflen,
             MPI_Status *stat, MPID_Request **req);
int psm_isendv(MPIDI_VC_t *vc, MPL_IOV *iov, int iov_n, MPID_Request *rptr);
int psm_irecv(int src, int tag, int context_id, void *buf, MPIDI_msg_sz_t buflen,
        MPID_Request *req);
int psm_imrecv(void *buf, MPIDI_msg_sz_t buflen, MPID_Request *req);
int psm_istartmsg(MPIDI_VC_t *vc, void *upkt, MPIDI_msg_sz_t pkt_sz, MPID_Request **rptr);
int psm_send_noncontig(MPIDI_VC_t *vc, MPID_Request *sreq, 
                       MPIDI_Message_match match);
int MPIDI_CH3_iRecv(int rank, int tag, int cid, void *buf, MPIDI_msg_sz_t buflen, MPID_Request *req);
int MPIDI_CH3_Recv(int rank, int tag, int cid, void *buf, MPIDI_msg_sz_t buflen, MPI_Status *stat, MPID_Request **req);
int MPIDI_CH3_iMrecv(void *buf, MPIDI_msg_sz_t buflen, MPID_Request *req);

void psm_pe_yield();
int psm_try_complete(MPID_Request *req);
int psm_progress_wait(int blocking);
int psm_map_error(PSM_ERROR_T psmerr);
MPID_Request *psm_create_req();
void psm_update_mpistatus(MPI_Status *, PSM_MQ_STATUS_T, int);
PSM_ERROR_T psm_isend_pkt(MPID_Request *req, MPIDI_Message_match m,
                        int dest, void *buf, MPIDI_msg_sz_t buflen);
int psm_1sided_input(MPID_Request *req, MPIDI_msg_sz_t inlen);
int psm_1sided_putpkt(MPIDI_CH3_Pkt_put_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr);
int psm_1sided_atomicpkt(MPIDI_CH3_Pkt_t *pkt, MPL_IOV *iov, int iov_n,
                       int rank, int srank, MPID_Request **rptr);
int psm_1sided_accumpkt(MPIDI_CH3_Pkt_accum_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr);
int psm_1sided_getaccumpkt(MPIDI_CH3_Pkt_get_accum_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr);
int psm_1sided_getresppkt(MPIDI_CH3_Pkt_get_resp_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr);
int psm_1sided_getaccumresppkt(MPIDI_CH3_Pkt_get_accum_resp_t *pkt, MPL_IOV *iov, int iov_n,
                       MPID_Request **rptr);
int psm_getresp_complete(MPID_Request *req); 
int psm_fopresp_complete(MPID_Request *req); 
int psm_getaccumresp_complete(MPID_Request *req); 
int psm_1sided_getpkt(MPIDI_CH3_Pkt_get_t *pkt, MPL_IOV *iov, int iov_n,
        MPID_Request **rptr);
int psm_1sc_get_rndvrecv(MPID_Request *savreq, MPIDI_CH3_Pkt_t *pkt, int from_rank);
int psm_dt_1scop(MPID_Request *req, char *buf, int len);
int psm_complete_rndvrecv(MPID_Request *req, MPIDI_msg_sz_t inlen);
/* PSM2 uses psm2_mq_tag_t instead of a uint64_t. */
#if PSM_VERNO >= PSM_2_1_VERSION
    PSM_ERROR_T psm_large_msg_isend_pkt(MPID_Request **rptr, int dest, void *buf, MPIDI_msg_sz_t buflen, psm2_mq_tag_t *stag, uint32_t flags);
#else
    PSM_ERROR_T psm_large_msg_isend_pkt(MPID_Request **rptr, int dest, void *buf, MPIDI_msg_sz_t buflen, uint64_t stag, uint32_t flags);
#endif
PSM_ERROR_T psm_send_pkt(MPID_Request **rptr, MPIDI_Message_match m,
                 int dest, void *buf, MPIDI_msg_sz_t buflen);
int psm_send_1sided_ctrlpkt(MPID_Request **rptr, int dest, void *buf, 
                            MPIDI_msg_sz_t buflen, int src, int create_req);
int psm_getresp_rndv_complete(MPID_Request *req, MPIDI_msg_sz_t inlen);
int psm_do_unpack(int count, MPI_Datatype datatype, MPID_Comm *comm, 
                  void *pkbuf, int pksz, void *inbuf, MPIDI_msg_sz_t data_sz);
int psm_do_pack(int count, MPI_Datatype datatype, MPID_Comm *comm, MPID_Request
                *sreq, const void *buf, MPIDI_msg_sz_t offset, MPIDI_msg_sz_t data_sz,
                psm_pack_type type);
void psm_do_ncrecv_complete(MPID_Request *req);
void psm_dequeue_compreq(MPID_Request *req);
void psm_prepost_1sc();
void psm_release_prepost_1sc();
int MPIDI_CH3_Probe(int source, int tag, int context, MPI_Status *stat,
                    int *complete, int blk);
int MPIDI_CH3_Mprobe(int source, int tag, int context, MPID_Request *req,
                    MPI_Status *stat, int *complete, int blk);
int MPID_Probe(int source, int tag, MPID_Comm * comm, int context_offset, 
	       MPI_Status * status);
int MPID_Mprobe(int source, int tag, MPID_Comm * comm, int context_offset,
	       MPID_Request **req, MPI_Status * status);
int MPID_Improbe(int source, int tag, MPID_Comm * comm, int context_offset,
	       int *flag, MPID_Request **req, MPI_Status * status);
int MPIDI_CH3I_comm_create(MPID_Comm *comm, void *param);
int MPIDI_CH3I_comm_destroy(MPID_Comm *comm, void *param);

extern int mv2_pmi_max_keylen;
extern int mv2_pmi_max_vallen;
extern char *mv2_pmi_key;
extern char *mv2_pmi_val;
extern int mv2_use_pmi_ibarrier;

int mv2_allocate_pmi_keyval(void);
void mv2_free_pmi_keyval(void);

int psm_get_rndvtag();
#endif 
