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

/*
 * limic_internal.h
 *
 * LiMIC2:  Linux Kernel Module for High-Performance MPI Intra-Node
 *          Communication
 *
 * Author:  Hyun-Wook Jin <jinh@konkuk.ac.kr>
 *          System Software Laboratory
 *          Department of Computer Science and Engineering
 *          Konkuk University
 *
 * History: Jul 15 2007 Launch
 *
 *          Feb 27 2009 Modified by Karthik Gopalakrishnan (gopalakk@cse.ohio-state.edu)
 *                                  Jonathan Perkins       (perkinjo@cse.ohio-state.edu)
 */

#ifndef _LIMIC_INTERNAL_H_INCLUDED_
#define _LIMIC_INTERNAL_H_INCLUDED_

#include <stdio.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define LIMIC_LIBRARY_MAJOR 0
#define LIMIC_LIBRARY_MINOR 7

/* /dev file name */
#define DEV_NAME  "limic"
#define DEV_CLASS "limic"

#define LIMIC_TX      0x1c01
#define LIMIC_RX      0x1c02
#define LIMIC_VERSION 0x1c03
#define LIMIC_TXW     0x1c04

typedef struct limic_request {
    void *buf;       /* user buffer */
    size_t len;         /* buffer length */
    limic_user *lu;  /* shandle or rhandle */
} limic_request;

#endif

