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

#ifndef IB_SRQ_H
#define IB_SRQ_H

#define _GNU_SOURCE

/**
 * srq information structure.
 */
typedef struct  {
    pthread_spinlock_t          srq_post_spin_lock;
    pthread_mutex_t             srq_post_mutex_lock[MAX_NUM_HCAS];
    pthread_mutex_t             async_mutex_lock[MAX_NUM_HCAS];
    pthread_cond_t              srq_post_cond[MAX_NUM_HCAS];
    uint32_t                    srq_zero_post_counter[MAX_NUM_HCAS];
    pthread_t                   async_thread[MAX_NUM_HCAS];
    uint32_t                    posted_bufs[MAX_NUM_HCAS];
} MPID_nem_ib_srq_info_t;

extern MPID_nem_ib_srq_info_t srq_info;

int MPID_nem_ib_allocate_srq();
int MPIDI_nem_ib_post_srq_buffers(int num_bufs,
        int hca_num);
#endif
