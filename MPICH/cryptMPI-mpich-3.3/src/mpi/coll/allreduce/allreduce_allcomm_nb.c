/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2017 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

#undef FUNCNAME
#define FUNCNAME MPIR_Allreduce_allcomm_nb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Allreduce_allcomm_nb(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                              MPI_Op op, MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
#if ALLREDUCE_PRINT_FUN
   if (PRINT_FUN_NAME || DEBUG_INIT_FILE){
	char hostname[100];
    gethostname(hostname, MAX_HOSTNAME_LEN);
    printf("[allreduce rank = %d host = %s count = %d] Func: MPIR_Allreduce_allcomm_nb\n",
    comm_ptr->rank,hostname,count);fflush(stdout);}
#endif  	
    int mpi_errno = MPI_SUCCESS;
    MPI_Request req = MPI_REQUEST_NULL;
    MPIR_Request *req_ptr = NULL;

    /* just call the nonblocking version and wait on it */
    mpi_errno = MPIR_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm_ptr, &req_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
    if (req_ptr)
        req = req_ptr->handle;

    mpi_errno = MPIR_Wait(&req, MPI_STATUS_IGNORE);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
