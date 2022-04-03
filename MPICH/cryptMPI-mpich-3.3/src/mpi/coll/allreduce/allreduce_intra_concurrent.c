/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"



#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_intra_concurrent
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_intra_concurrent(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    return MPIR_Allreduce_intra_smp(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);;
}
