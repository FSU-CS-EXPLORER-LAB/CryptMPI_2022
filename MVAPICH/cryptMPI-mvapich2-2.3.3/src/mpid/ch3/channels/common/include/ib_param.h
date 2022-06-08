/*!\file */
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

#ifndef _IBV_PARAM_H
#define _IBV_PARAM_H

#if defined(HAVE_LIBIBVERBS)
#include <infiniband/verbs.h>
#endif
#include "mv2_arch_hca_detect.h"

#define MAX_NUM_HCAS                    (4)

/**
 *  Number of queue pairs per port.
 */
extern int rdma_num_qp_per_port;
extern int rdma_num_hcas;
extern int                  rdma_iwarp_use_multiple_cq;

/**
 *  Number of rails.
 */
extern int rdma_num_rails;

extern unsigned long        rdma_default_max_cq_size;
extern int                  rdma_default_port;
extern unsigned long        rdma_default_max_send_wqe;
extern unsigned long        rdma_default_max_recv_wqe;
extern uint32_t             rdma_default_max_sg_list;
extern uint16_t             rdma_default_pkey_ix;
extern uint16_t             rdma_default_pkey;
extern uint8_t              rdma_default_qp_ous_rd_atom;
extern uint8_t              rdma_default_max_rdma_dst_ops;
extern enum ibv_mtu         rdma_default_mtu;
extern uint32_t             rdma_default_psn;
extern uint8_t              rdma_default_min_rnr_timer;
extern uint8_t              rdma_default_service_level;
extern uint8_t              rdma_default_static_rate;
extern uint8_t              rdma_default_src_path_bits;
extern uint8_t              rdma_default_time_out;
extern uint8_t              rdma_default_retry_count;
extern uint8_t              rdma_default_rnr_retry;
extern int                  rdma_default_put_get_list_size;
extern float                rdma_credit_update_threshold;
extern int                  num_rdma_buffer;
extern int                  rdma_iba_eager_threshold;
extern char                 rdma_iba_hca[32];
extern unsigned int         rdma_ndreg_entries;
extern int                  rdma_vbuf_max;
extern int                  rdma_vbuf_pool_size;
extern int                  rdma_vbuf_secondary_pool_size;
extern int                  rdma_initial_prepost_depth;
extern int                  rdma_prepost_depth;
extern int                  rdma_prepost_threshold;
extern int                  rdma_prepost_noop_extra;
extern int                  rdma_initial_credits;
extern int                  rdma_prepost_rendezvous_extra;
extern int                  rdma_dynamic_credit_threshold;
extern int                  rdma_credit_notify_threshold;
extern int                  rdma_credit_preserve;
extern int                  rdma_rq_size;
extern unsigned long        rdma_dreg_cache_limit;
extern int                  rdma_rndv_protocol;
extern int                  smp_rndv_protocol;
extern int                  rdma_rndv_immediate;
extern int                  rdma_r3_threshold;
extern int                  rdma_r3_threshold_nocache;
extern int                  rdma_max_r3_pending_data;
extern int                  rdma_vbuf_total_size;
extern int                  rdma_max_inline_size;

extern uint32_t             mv2_srq_alloc_size;
extern uint32_t             mv2_srq_fill_size;
extern uint32_t             mv2_srq_limit;
extern uint32_t             mv2_max_r3_oust_send;

extern int                  rdma_polling_set_threshold;
extern int                  rdma_polling_set_limit;
extern int                  rdma_fp_buffer_size;
extern int                  rdma_fp_sendconn_accepted;
extern int                  rdma_pending_conn_request;
extern int		            rdma_eager_limit;
extern int                  rdma_rndv_ext_sendq_size;

extern int                  rdma_pin_pool_size;
extern int                  rdma_put_fallback_threshold;
extern int                  rdma_get_fallback_threshold;
extern int                  rdma_iba_eager_threshold;
extern long                 rdma_eagersize_1sc;
extern int                  rdma_qos_num_sls;
extern int                  rdma_use_qos;
extern int                  rdma_3dtorus_support;
extern int                  rdma_path_sl_query;
extern int                  rdma_num_sa_query_retries;

/* HSAM Definitions */

extern  int                 striping_threshold;
extern  int                 sm_scheduling;
extern  int                 alpha;
extern  int                 stripe_factor;
extern  int                 apm_tester;

extern int                  rdma_coalesce_threshold;
extern int                  rdma_use_coalesce;

extern int                  rdma_use_blocking;
extern unsigned long        rdma_blocking_spin_count_threshold;
extern int                  rdma_use_smp;
extern int                  use_iboeth;
extern int                  mv2_show_env_info;



#define PKEY_MASK 0x7fff /* the last bit is reserved */
#define RDMA_PIN_POOL_SIZE              (2*1024*1024)
#define RDMA_DEFAULT_MAX_CQ_SIZE        (40000)
#define RDMA_DEFAULT_IWARP_CQ_SIZE      (8192)
#define RDMA_DEFAULT_PORT               (-1)
#define RDMA_DEFAULT_MAX_PORTS          (2)
#define RDMA_DEFAULT_MAX_SEND_WQE       (64)
#define RDMA_DEFAULT_MAX_RECV_WQE       (128)
#define RDMA_DEFAULT_MAX_SG_LIST        (1)
#define RDMA_DEFAULT_PKEY_IX            (0)
#define RDMA_DEFAULT_PKEY               (0x0)
#define RDMA_DEFAULT_MAX_RDMA_DST_OPS   (4)
#define RDMA_DEFAULT_PSN                (0)
#define RDMA_DEFAULT_MIN_RNR_TIMER      (12)
#define RDMA_DEFAULT_SERVICE_LEVEL      (0)
#define RDMA_DEFAULT_STATIC_RATE        (0)
#define RDMA_DEFAULT_SRC_PATH_BITS      (0)
#define RDMA_DEFAULT_TIME_OUT          (20)
#define RDMA_DEFAULT_RETRY_COUNT        (7)
#define RDMA_DEFAULT_RNR_RETRY          (7)
#define RDMA_DEFAULT_PUT_GET_LIST_SIZE  (200)
#define RDMA_DEFAULT_POLLING_SET_LIMIT  (64)
#define RDMA_IBA_NULL_HCA               "nohca"
#define RDMA_DEFAULT_NUM_SA_QUERY_RETRIES   (20)

/* This is a overprovision of resource, do not use in critical structures */
#define MAX_NUM_SUBRAILS                (MAX_NUM_HCAS*  \
                                         MAX_NUM_PORTS* \
                                         MAX_NUM_QP_PER_PORT)

#define RDMA_QOS_MAX_NUM_SLS            (15)
#define RDMA_QOS_DEFAULT_NUM_SLS        (8)
#define RDMA_NDREG_ENTRIES              (1100)
#define RDMA_NDREG_ENTRIES_MAX          (4096)
#define RDMA_VBUF_POOL_SIZE             (512)
#define RDMA_VBUF_SECONDARY_POOL_SIZE   (256)
#define RDMA_PREPOST_DEPTH              (64)
#define RDMA_INITIAL_PREPOST_DEPTH      (10)
#define RDMA_LOW_WQE_THRESHOLD          (10)
#define RDMA_MAX_RDMA_SIZE              (4194304)
#define RDMA_FP_DEFAULT_BUF_SIZE        (4096)

#define RDMA_IWARP_DEFAULT_MULTIPLE_CQ_THRESHOLD (32)

/* Inline not supported for PPC */
#define HOSTNAME_LEN                    (255)
#define MAX_WIN_NUM                     (16)
#define RDMA_MAX_REGISTERED_PAGES       (0)

/*#define MIN(a,b) ((a)<(b)?(a):(b))*/

#define NUM_BOOTSTRAP_BARRIERS  2

/* Statistically sending a stripe below this may not lead
 * to benefit */
#define STRIPING_THRESHOLD              8 * 1024

#define ROUND_ROBIN                     0
#define USE_FIRST                       1
#define EVEN_STRIPING                   2
#define ADAPTIVE_STRIPING               3
#define PROCESS_BINDING                 4
#define PARTIAL_ADAPTIVE                5
#define BEST_ADAPTIVE                   6

#define DYNAMIC_TOTAL_WEIGHT            (3* 1024)


/**
 * Get the parameters from users.
 */
int MPID_nem_ib_get_user_params();

/**
 * Set the default parameter values.
 */
int MPID_nem_ib_set_default_params();

/**
 * Get the control parameters.
 */
int MPID_nem_ib_get_control_params();
int MPID_nem_ib_get_control_params_after_hcainit();

/**
 * Print values of important environment variables.
 */
void mv2_print_env_info();

mv2_arch_hca_type MV2_get_arch_hca_type();
int mv2_get_heterogeneity();

#endif /* _RDMA_PARAM_H */
