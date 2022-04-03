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

#ifndef __IB_BUF__
#define __IB_BUF__

struct ib_buffer *create_ib_buffer(int size, int slot_size, char *name);

void free_ib_buffer(struct ib_buffer *buf);

int get_buf_slot(struct ib_buffer *buf, void **addr, int expect);

int free_buf_slot(struct ib_buffer *buf, int slot, int expect);

void *ib_buffer_slot_addr(struct ib_buffer *buf, int slot);

void dump_ib_buffer(struct ib_buffer *buf);

#endif                          // __IB_BUF__
