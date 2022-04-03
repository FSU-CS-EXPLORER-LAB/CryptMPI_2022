/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

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

#define _GNU_SOURCE
#include "mpidimpl.h"
#include "ib_impl.h"
#include "ib_device.h"
#include "ib_cm.h"
#include "ib_utils.h"

#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_vc_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPID_nem_ib_vc_terminate (MPIDI_VC_t *vc)
{
    return MPIDI_CH3U_Handle_connection(vc, MPIDI_VC_EVENT_TERMINATED);
}
