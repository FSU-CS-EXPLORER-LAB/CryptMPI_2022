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

#ifndef IB_PROCESS_H
#define IB_PROCESS_H

#include <infiniband/verbs.h>

#include "ib_hca.h"
#include "ib_cm.h"
#include "ib_param.h"


struct MPIDI_VC;

/**
 * Information about current process.
 */
typedef struct  {
    /**
     * Process group.
     * MPIDI_PG_t defined in mpidimpl.h. Number of processes
     * in the process group: pg->size.
     */
	struct MPIDI_PG *pg;

    /** Rank in process group */
    int rank;

    int             **hostid;

    uint64_t        *vc_addr;

    int num_rails;
    /** Linked list of connection to other processes. */
    MPID_nem_ib_connection_t *connections;

    int rails;

    uint8_t                     heterogeneity;
    uint8_t                     has_srq;
    uint8_t                     has_hsam;
    uint8_t                     has_apm;
    uint8_t                     has_adaptive_fast_path;
    uint8_t                     has_ring_startup;
    uint8_t                     has_lazy_mem_unregister;
    uint8_t                     has_one_sided;
    int                         maxtransfersize;
    uint8_t                     lmc;
    int                         global_used_send_cq;
    int                         global_used_recv_cq;

    int    (*post_send)(struct MPIDI_VC * vc, vbuf * v, int rail);


    int                         polling_group_size;
    struct MPIDI_VC             **polling_set;

    /* data structure for ring based startup */
    struct ibv_cq               *boot_cq_hndl;
    struct ibv_qp               *boot_qp_hndl[2];
    int                         boot_tb[2][2];

    mv2_hca_type        hca_type;
    mv2_arch_hca_type   arch_hca_type;

} MPID_nem_ib_process_info_t;


/**
 * Process information.
 * \see MPID_nem_ib_process_info_t
 */
extern MPID_nem_ib_process_info_t process_info;


/**
 *  Initialize the process information.
 */
int MPID_nem_ib_init_process_info(int pg_rank, struct MPIDI_PG *pg_p);


#endif /* IB_PROCESS_H */
