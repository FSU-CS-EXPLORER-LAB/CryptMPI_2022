/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#include "psmpriv.h"
#include <pthread.h>


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_Init(int has_parent, MPIDI_PG_t *pg, int pg_rank) 
{
    return (psm_doinit(has_parent, pg, pg_rank));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iSendv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_iSendv(MPIDI_VC_t *vc, MPID_Request *req, MPL_IOV *iov, int iov_n)
{
    return (psm_isendv(vc, iov, iov_n, req));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iStartMsgv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_iStartMsgv(MPIDI_VC_t *vc, MPL_IOV *iov, int iov_n, MPID_Request **req)
{
   return (psm_istartmsgv(vc, iov, iov_n, req));     
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iStartMsg
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_iStartMsg(MPIDI_VC_t *vc, void *pkt, MPIDI_msg_sz_t pkt_sz, MPID_Request **req)
{
    return (psm_istartmsg(vc, pkt, pkt_sz, req));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iSend
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_iSend(MPIDI_VC_t *vc, MPID_Request *req, void *pkt, MPIDI_msg_sz_t pkt_sz)
{
    return MPI_SUCCESS; 
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Connection_terminate
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_Connection_terminate(MPIDI_VC_t *vc)
{
    return MPI_SUCCESS; 
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Connect_to_root
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_Connect_to_root(const char *port, MPIDI_VC_t **vc)
{
    return MPI_SUCCESS; 
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_VC_GetStateString
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

const char * MPIDI_CH3_VC_GetStateString(MPIDI_VC_t *vc)
{
    return NULL;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PG_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_PG_Init(MPIDI_PG_t *pg)
{
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PG_Destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_PG_Destroy(MPIDI_PG_t *pg)
{
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_InitCompleted
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_InitCompleted()
{
    
    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        if (!ipath_enable_func_lock) {
            psm_lock_fn = psm_no_lock;
            psm_unlock_fn = psm_no_lock;
        } else {
            psm_lock_fn = pthread_spin_lock;
            psm_unlock_fn = pthread_spin_unlock;
        }
        psm_progress_lock_fn = pthread_spin_lock;
        psm_progress_unlock_fn = pthread_spin_unlock;
    } else {
        psm_lock_fn = psm_unlock_fn = psm_progress_lock_fn = psm_progress_unlock_fn = psm_no_lock;
    }
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PortFnsInit  
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_PortFnsInit(MPIDI_PortFns *fns)
{
    return MPI_SUCCESS; 
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Get_business_card
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_Get_business_card(int myRank, char *port, int len)
{
    return MPI_SUCCESS; 
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_VC_Destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_VC_Destroy(MPIDI_VC_t *vc)    
{
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_VC_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_VC_Init(MPIDI_VC_t *vc)
{
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Progress_start
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline void MPIDI_CH3_Progress_start(MPID_Progress_state *pstate)
{
  _psm_enter_;
    PSM_POLL(psmdev_cw.ep);
  _psm_exit_;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Progress_end
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPIDI_CH3_Progress_end(MPID_Progress_state *pstate)
{
  _psm_enter_;
    PSM_POLL(psmdev_cw.ep);
  _psm_exit_;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Progress_wakeup
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

void MPIDI_CH3I_Progress_wakeup(void)
{

}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Progress_wait
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_Progress_wait(MPID_Progress_state *state)
{
    return(psm_progress_wait(TRUE));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_Finalize(void)
{
    return(psm_dofinalize());
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Progress_test
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_Progress_test(void)
{
    return (psm_progress_wait(FALSE));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Progress_poke
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_Progress_poke()
{
    return (psm_progress_wait(FALSE));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_Recv(int rank, int tag, int cid, void *buf, MPIDI_msg_sz_t buflen,
                          MPI_Status *stat, MPID_Request **req)
{
    return (psm_recv(rank, tag, cid, buf, buflen, stat, req));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iRecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

inline int MPIDI_CH3_iRecv(int rank, int tag, int cid, void *buf, MPIDI_msg_sz_t buflen,
        MPID_Request *req)
{
    return (psm_irecv(rank, tag, cid, buf, buflen, req));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_iMrecv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

int MPIDI_CH3_iMrecv(void *buf, MPIDI_msg_sz_t buflen, MPID_Request *req)
{
    return (psm_imrecv(buf, buflen, req));
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_SendNonContig
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_SendNonContig(MPIDI_VC_t *vc, MPID_Request *sreq,
                            MPIDI_Message_match match, int blocking)
{
    fprintf(stderr, "SHOULD NOT BE USED\n");
    fflush(stderr);
    return MPI_SUCCESS; 
}
    
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Probe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Probe(int source, int tag, int context, MPI_Status *stat,
                    int *complete, int blk)
{
    int mpi_errno = MPI_SUCCESS, i;
    PSM_ERROR_T psmerr;
    uint32_t ipath_spinlimit = 
      (MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) ? 100 : 1000;
   
    /* if not blocking, do probe once */
    if(blk == PSM_NONBLOCKING) {
        psmerr = psm_probe(source, tag, context, stat);
        if(psmerr == PSM_OK) {
            *complete = TRUE;
        } else if(psmerr == PSM_MQ_NO_COMPLETIONS) {
            *complete = FALSE;
        } else {
            MPIR_ERR_POP(mpi_errno);
        }
        goto fn_exit;
    }
 
    /* if blocking, probe SPINLIMIT times */
spin:
    for(i = 0; i < ipath_spinlimit; i++) {
        psmerr = psm_probe(source, tag, context, stat);
        if(psmerr == PSM_OK) {
            *complete = TRUE;
            goto fn_exit;
        } else if(psmerr != PSM_MQ_NO_COMPLETIONS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /* if we're MT yield global lock */
    
    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        psm_pe_yield();
    }
    goto spin;
    
fn_fail:    
fn_exit:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Mprobe
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Mprobe(int source, int tag, int context,
                    MPID_Request *req, MPI_Status *stat,
                    int *complete, int blk)
{
    int mpi_errno = MPI_SUCCESS, i;
    PSM_ERROR_T psmerr;
    uint32_t ipath_spinlimit =
      (MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) ? 100 : 1000;

    /* if not blocking, do probe once */
    if(blk == PSM_NONBLOCKING) {
        psmerr = psm_mprobe(source, tag, context, req, stat);
        if(psmerr == PSM_OK) {
            *complete = TRUE;
        } else if(psmerr == PSM_MQ_NO_COMPLETIONS) {
            *complete = FALSE;
        } else {
            MPIR_ERR_POP(mpi_errno);
        }
        goto fn_exit;
    }

    /* if blocking, probe SPINLIMIT times */
spin:
    for(i = 0; i < ipath_spinlimit; i++) {
        psmerr = psm_mprobe(source, tag, context, req, stat);
        if(psmerr == PSM_OK) {
            *complete = TRUE;
            goto fn_exit;
        } else if(psmerr != PSM_MQ_NO_COMPLETIONS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    /* if we're MT yield global lock */

    if(MPIR_ThreadInfo.thread_provided == MPI_THREAD_MULTIPLE) {
        psm_pe_yield();
    }
    goto spin;

fn_fail:
fn_exit:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Progress_register_hook
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Progress_register_hook(int (*progress_fn)(int*), int *id)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PROGRESS_REGISTER_HOOK);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PROGRESS_REGISTER_HOOK);
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);

    for (i = 0; i < MAX_PROGRESS_HOOKS; i++) {
        if (progress_hooks[i].func_ptr == NULL) {
            progress_hooks[i].func_ptr = progress_fn;
            progress_hooks[i].active = FALSE;
            break;
        }
    }

    if (i >= MAX_PROGRESS_HOOKS) {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                     "MPIDI_CH3I_Progress_register_hook", __LINE__,
                     MPI_ERR_INTERN, "**progresshookstoomany", 0 );
        goto fn_fail;
    }

    (*id) = i;

  fn_exit:
    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PROGRESS_REGISTER_HOOK);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Progress_deregister_hook
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Progress_deregister_hook(int id)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PROGRESS_DEREGISTER_HOOK);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PROGRESS_DEREGISTER_HOOK);
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);

    MPIU_Assert(id >= 0 && id < MAX_PROGRESS_HOOKS && progress_hooks[id].func_ptr != NULL);

    progress_hooks[id].func_ptr = NULL;
    progress_hooks[id].active = FALSE;

    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PROGRESS_DEREGISTER_HOOK);
    return mpi_errno;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Progress_activate_hook
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Progress_activate_hook(int id)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PROGRESS_ACTIVATE_HOOK);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PROGRESS_ACTIVATE_HOOK);
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);

    MPIU_Assert(id >= 0 && id < MAX_PROGRESS_HOOKS &&
                progress_hooks[id].active == FALSE && progress_hooks[id].func_ptr != NULL);
    progress_hooks[id].active = TRUE;

    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PROGRESS_ACTIVATE_HOOK);
    return mpi_errno;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Progress_deactivate_hook
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Progress_deactivate_hook(int id)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_PROGRESS_DEACTIVATE_HOOK);

    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_PROGRESS_DEACTIVATE_HOOK);
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);

    MPIU_Assert(id >= 0 && id < MAX_PROGRESS_HOOKS &&
                progress_hooks[id].active == TRUE && progress_hooks[id].func_ptr != NULL);
    progress_hooks[id].active = FALSE;

    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_PROGRESS_DEACTIVATE_HOOK);
    return mpi_errno;
}

