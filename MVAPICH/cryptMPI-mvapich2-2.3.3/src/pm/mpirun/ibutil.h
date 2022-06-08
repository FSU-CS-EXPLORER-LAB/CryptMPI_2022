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

#ifndef __IB_UTIL__
#define __IB_UTIL__

char *ibv_wr_opcode_string(int opcode);

char *ibv_wc_opcode_string(int opcode);

//char* ibv_mtu_string( enum ibv_mtu  mtu );

const char *ibv_port_state_string(enum ibv_port_state state);

const char *ibv_port_phy_state_string(uint8_t phys_state);

const char *ibv_atomic_cap_string(enum ibv_atomic_cap atom_cap);

const char *ibv_mtu_string(enum ibv_mtu max_mtu);

const char *ibv_width_string(uint8_t width);

const char *ibv_speed_string(uint8_t speed);

const char *ibv_vl_string(uint8_t vl_num);

const char *ibv_wc_status_string(int status);

void dump_wc(struct ibv_wc *wc);

void dump_send_wr(struct ibv_send_wr *wr);

void dump_ibv_device_attr(struct ibv_device_attr *attr);

void dump_ibv_port_attr(struct ibv_port_attr *attr);

double tv2sec(struct timeval *start, struct timeval *end);

const char *int_to_binary(int x);
#endif                          // __IB_UTIL__
