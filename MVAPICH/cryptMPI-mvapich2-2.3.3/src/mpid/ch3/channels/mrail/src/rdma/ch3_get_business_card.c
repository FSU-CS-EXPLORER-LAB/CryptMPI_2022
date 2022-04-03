/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#include "mpidi_ch3_impl.h"

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Get_business_card
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Get_business_card(int rank, char *value, int length)
{
    return MPI_SUCCESS;
}
