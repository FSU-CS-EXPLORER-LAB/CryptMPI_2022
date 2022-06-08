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

#ifndef WFE_MPIRUN_H
#define WFE_MPIRUN_H 1

#include <netinet/in.h>

struct wfe_params {
    int s;
    struct sockaddr_in * sockaddr;
    unsigned sockaddr_len;
};

extern void start_wfe_thread (struct wfe_params *);
extern void stop_wfe_thread (void);

#endif
