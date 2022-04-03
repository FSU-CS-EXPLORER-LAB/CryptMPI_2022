/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpidi_ch3_impl.h"

/*
 * MPIDI_CH3_Open_port()
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Open_port
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Open_port(char *port_name)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_OPEN_PORT);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_OPEN_PORT);


    mpi_errno =
        MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME,
                             __LINE__, MPI_ERR_OTHER, "**notimpl", 0);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_OPEN_PORT);
    return mpi_errno;
}
