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

#include "mpidi_ch3_impl.h"

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Cancel_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Cancel_send(MPIDI_VC_t * vc, MPID_Request * sreq,
                          int *cancelled)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_CANCEL_SEND);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_CANCEL_SEND);
    *cancelled = FALSE;
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_CANCEL_SEND);
    return MPI_SUCCESS;
}
