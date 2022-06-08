/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpidimpl.h"

int (*MPIDI_Anysource_improbe_fn)(int tag, MPID_Comm * comm, int context_offset,
                                  int *flag, MPID_Request **message,
                                  MPI_Status * status) = NULL;

#undef FUNCNAME
#define FUNCNAME MPID_Improbe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Improbe(int source, int tag, MPID_Comm *comm, int context_offset,
                 int *flag, MPID_Request **message, MPI_Status *status)
{
    int mpi_errno = MPI_SUCCESS;
    int context_id = comm->recvcontext_id + context_offset;

    *message = NULL;

    MV2_INC_NUM_UNEXP_RECV();
    if (source == MPI_PROC_NULL)
    {
        MPIR_Status_set_procnull(status);
        *flag = TRUE;
        *message = NULL; /* should be interpreted as MPI_MESSAGE_NO_PROC */
        goto fn_exit;
    }

    /* Check to make sure the communicator hasn't already been revoked */
    if (comm->revoked) {
        MPIR_ERR_SETANDJUMP(mpi_errno,MPIX_ERR_REVOKED,"**revoked");
    }

#if defined (CHANNEL_PSM)
    #if PSM_VERNO < PSM_2_1_VERSION
    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
            "Operation not supported for QLogic PSM (CH3:PSM) channel\n");
    #endif

	int complete = FALSE;
    MPID_Request *rreq = MPID_Request_create();
    MPIU_Object_set_ref(rreq, 2);
    rreq->kind = MPID_REQUEST_MPROBE;
    MPIR_Comm_add_ref(comm);
    rreq->comm = comm;

    MPID_Progress_poke();
    mpi_errno = MPIDI_CH3_Mprobe(source, tag, context_id,
                                rreq, status, &complete,
                                PSM_NONBLOCKING);
    if(mpi_errno) MPIR_ERR_POP(mpi_errno);

    if (complete) {
        *flag = TRUE;
        *message = rreq;
    } else {
        *flag = FALSE;
        MPID_Request_release(rreq);
    }
    goto fn_exit;
#endif

#ifdef ENABLE_COMM_OVERRIDES
    if (MPIDI_Anysource_improbe_fn) {
        if (source == MPI_ANY_SOURCE) {
            /* if it's anysource, check shm, then check the network.
               If still not found, call progress, and check again. */

            /* check shm*/
            MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
            *message = MPIDI_CH3U_Recvq_FDU_matchonly(source, tag, context_id, comm, flag);
            MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
            if (!*flag) {
                /* not found, check network */
                mpi_errno = MPIDI_Anysource_improbe_fn(tag, comm, context_offset, flag, message, status);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                if (!*flag) {
                    /* still not found, make some progress*/
                    mpi_errno = MPIDI_CH3_Progress_poke();
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    /* check shm again */
                    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
                    *message = MPIDI_CH3U_Recvq_FDU_matchonly(source, tag, context_id, comm, flag);
                    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
                    if (!*flag) {
                        /* check network again */
                        mpi_errno = MPIDI_Anysource_improbe_fn(tag, comm, context_offset, flag, message, status);
                        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    }
                }
            }
            goto fn_exit;
        }
        else {
            /* it's not anysource, check if the netmod has overridden it */
            MPIDI_VC_t * vc;
            MPIDI_Comm_get_vc_set_active(comm, source, &vc);
            if (vc->comm_ops && vc->comm_ops->improbe) {
                mpi_errno = vc->comm_ops->improbe(vc, source, tag, comm, context_offset, flag, message, status);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                goto fn_exit;
            }
            /* fall-through to shm case */
        }
    }
#endif

    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    *message = MPIDI_CH3U_Recvq_FDU_matchonly(source, tag, context_id, comm, flag);
    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);

    if (!*flag) {
        /* Always try to advance progress before returning failure
           from the improbe test. */
        /* FIXME: It would be helpful to know if the Progress_poke
           operation causes any change in state; we could then avoid
           a second test of the receive queue if we knew that nothing
           had changed */
        mpi_errno = MPID_Progress_poke();
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
        *message = MPIDI_CH3U_Recvq_FDU_matchonly(source, tag, context_id, comm, flag);
        MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_MSGQ_MUTEX);
    }

    if (*flag && *message) {
        (*message)->kind = MPID_REQUEST_MPROBE;
        MPIR_Request_extract_status((*message), status);
    }

fn_exit:
    MV2_DEC_NUM_UNEXP_RECV();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

