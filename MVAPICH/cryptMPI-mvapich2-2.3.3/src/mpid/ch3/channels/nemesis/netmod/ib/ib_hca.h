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

#ifndef IB_HCA_H
#define IB_HCA_H

#include <infiniband/verbs.h>

#include "ib_param.h"
#include <mv2_arch_hca_detect.h>

/* --------------------------------------------- *
 * Constants and limits
 * --------------------------------------------- */

/**
 * Max number of devices.
 */
#define MAX_NUM_HCAS                    (4)

/**
 * Max number of ports.
 */
#define MAX_NUM_PORTS                   (2)

/**
 * Max number queues per port.
 */
#define MAX_NUM_QP_PER_PORT             (4)

/* --------------------------------------------- *
 * Configuration parameters (ex. from environment
 * variables).
 * --------------------------------------------- */

/* Support multiple QPs/port, multiple ports, multiple HCAs and combinations */

/**
 *  Number of HCSs. The user can set this value with the environment
 *  variable MV2_NUM_HCAS.
 *  (Was rdma_num_hcas).
 */
extern int ib_hca_num_hcas;

/**
 *  Number of ports.
 *  (Was rdma_num_ports).
 */
extern int ib_hca_num_ports;




/* --------------------------------------------- *
 * Public data structures
 * --------------------------------------------- */

/*
FROM priv.h

    struct ibv_context          *nic_context[MAX_NUM_HCAS];
    struct ibv_device           *ib_dev[MAX_NUM_HCAS];
    struct ibv_pd               *ptag[MAX_NUM_HCAS];
    struct ibv_cq               *cq_hndl[MAX_NUM_HCAS];
    struct ibv_cq               *send_cq_hndl[MAX_NUM_HCAS];
    struct ibv_cq               *recv_cq_hndl[MAX_NUM_HCAS];
    struct ibv_comp_channel     *comp_channel[MAX_NUM_HCAS];
*/

/**
 * Information about one HCA.
 */
typedef struct {
    mv2_hca_type hca_type;

    struct ibv_context          *nic_context;
    struct ibv_device           *ib_dev;

    /** Infiniband protection tag */
    struct ibv_pd               *ptag;

    struct ibv_cq               *cq_hndl;
    struct ibv_cq               *send_cq_hndl;
    struct ibv_cq               *recv_cq_hndl;
    struct ibv_comp_channel     *comp_channel;


    /** Record lid and port information for connection establish later. */
    int ports[MAX_NUM_PORTS];
    int lids[MAX_NUM_PORTS];
    union ibv_gid gids[MAX_NUM_PORTS];

    struct ibv_srq *srq_hndl;

    /*
       int                     num_cqs;
       struct ibv_pd           *pd;
       struct ibv_cq           **cq;
       struct ibv_device       *device;
       struct ibv_context      *context;
       struct ibv_port_attr    *port_attr;
       struct ibv_port_attr    *default_port_attr;
       struct ibv_device_attr  device_attr;
       pthread_t async_thread;
     */
} MPID_nem_ib_nem_hca;

/* --------------------------------------------- *
 * Function declarations.
 * --------------------------------------------- */

/**
 * The list of the HCAs found in the system.
 */
extern MPID_nem_ib_nem_hca hca_list[MAX_NUM_HCAS];


/* --------------------------------------------- *
 * Function declarations.
 * --------------------------------------------- */

/**
 * Opens the HCA and allocates protection domain for it.
 */
int MPID_nem_ib_init_hca();

/**
 * the first step in original MPID_nem_ib_setup_conn() function
 * open hca, create ptags  and create cqs
 */
int MPID_nem_ib_open_ports();



#endif /* IB_PRIV_H */
