/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpid_nem_impl.h"

/* forward declaration of funcs structs defined in network modules */
extern MPID_nem_netmod_funcs_t MPIDI_nem_tcp_funcs;

int MPID_nem_num_netmods = 1;
MPID_nem_netmod_funcs_t *MPID_nem_netmod_funcs[1] = { &MPIDI_nem_tcp_funcs };
char MPID_nem_netmod_strings[1][MPID_NEM_MAX_NETMOD_STRING_LEN] = { "tcp" };
