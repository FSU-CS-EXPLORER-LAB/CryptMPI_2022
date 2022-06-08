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

#ifndef RDMA_3DTORUS_H
#define RDMA_3DTORUS_H
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <sys/time.h>
#include <mpidi_ch3_impl.h>

#define IB_MGMT_BASE_VERSION        1
#define IB_MGMT_CLASS_SUBN_ADM      0x03
#define IB_MGMT_METHOD_GET          0x01
#define IB_SA_TID_GET_PATH_REC_0    0xCA000000UL
#define IB_SA_TID_GET_PATH_REC_1    0xBEEF0000UL

struct ib_gid_half {
    uint8_t bytes[8];
};

struct ib_gid {
    union {
        uint8_t bytes[16];
        uint16_t words[8];
        uint32_t dwords[4];
        struct ib_gid_half half[2];
    } u;
};

struct ib_smp_class_specific {
    uint8_t hop_pointer;
    uint8_t hop_count;
} __attribute__ (( packed ));

union ib_mad_class_specific {
    uint16_t raw;
    struct ib_smp_class_specific smp;
} __attribute__ (( packed ));

struct ib_mad_hdr {
    uint8_t base_version;
    uint8_t mgmt_class;
    uint8_t class_version;
    uint8_t method;
    uint16_t status;
    union ib_mad_class_specific class_specific;
    uint32_t tid[2];
    uint16_t attr_id;
    uint8_t reserved[2];
    uint32_t attr_mod;
} __attribute__ (( packed ));

struct ib_rmpp_hdr {
    uint32_t raw[3];
} __attribute__ (( packed ));

struct ib_sa_hdr {
    uint32_t sm_key[2];
    uint16_t reserved;
    uint16_t attrib_offset;
    uint32_t comp_mask[2];
} __attribute__ (( packed ));

#define IB_SA_ATTR_PATH_REC                     0x35

struct ib_path_record {
    uint32_t reserved0[2];
    struct ib_gid dgid;
    struct ib_gid sgid;
    uint16_t dlid;
    uint16_t slid;
    uint32_t hop_limit__flow_label__raw_traffic;
    uint32_t pkey__numb_path__reversible__tclass;
    uint8_t reserved1;
    uint8_t reserved__sl;
    uint8_t mtu_selector__mtu;
    uint8_t rate_selector__rate;
    uint32_t preference__packet_lifetime__packet_lifetime_selector;
    uint32_t reserved2[35];
} __attribute__ (( packed ));

#define IB_SA_PATH_REC_DLID                     (1<<4)
#define IB_SA_PATH_REC_SLID                     (1<<5)

union ib_sa_data {
    struct ib_path_record path_record;
} __attribute__ (( packed ));

struct ib_mad_sa {
    struct ib_mad_hdr mad_hdr;
    struct ib_rmpp_hdr rmpp_hdr;
    struct ib_sa_hdr sa_hdr;
    union ib_sa_data sa_data;
} __attribute__ (( packed ));

#define IB_SA_QPN               1
#define IB_GLOBAL_QKEY          0x80010000UL

struct openib_sa_qp_cache_t {
    /* There will be a MR with the one send and receive buffer together */
    /* The send buffer is first, the receive buffer is second */
    /* The receive buffer in a UD queue pair needs room for the 40 byte GRH */
    /* The buffers are first in the structure for page alignment */
    char   send_recv_buffer[sizeof(struct ib_mad_sa) * 2 + 40];
    struct openib_sa_qp_cache_t *next;
    struct ibv_context *context;
    char  *device_name;
    uint32_t port_num;
    struct ibv_qp *qp;
    struct ibv_ah *ah;
    struct ibv_cq *cq;
    struct ibv_mr *mr;
    struct ibv_recv_wr rwr;
    struct ibv_sge rsge;
    char   sl_values[65536];
};

int mv2_release_3d_torus_resources();
int mv2_get_path_rec_sl(struct ibv_context *context_arg, struct ibv_pd *hca_pd,
                        uint32_t port_num, uint16_t lid, uint16_t rem_lid,
                        int network_is_3dtorus, int num_sa_retries);

#endif /* RDMA_3DTORUS_H */
