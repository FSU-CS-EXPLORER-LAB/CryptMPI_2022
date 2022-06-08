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

#ifndef _MPIDI_CH3_IMPL_H
#define _MPIDI_CH3_IMPL_H

#include "mpidimpl.h"
#include "mpl_utlist.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

/* Shared memory window atomic/accumulate mutex implementation */

#define MPIDI_CH3I_SHM_MUTEX_LOCK(win_ptr)                                         \
    do {                                                                           \
        int pt_err = pthread_mutex_lock((win_ptr)->shm_mutex);                     \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_lock",   \
                "**pthread_lock %s", strerror(pt_err));                            \
    } while (0)

#define MPIDI_CH3I_SHM_MUTEX_UNLOCK(win_ptr)                                       \
    do {                                                                           \
        int pt_err = pthread_mutex_unlock((win_ptr)->shm_mutex);                   \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_unlock", \
                "**pthread_unlock %s", strerror(pt_err));                          \
    } while (0)

#define MPIDI_CH3I_SHM_MUTEX_INIT(win_ptr)                                         \
    do {                                                                           \
        int pt_err;                                                                \
        pthread_mutexattr_t attr;                                                  \
                                                                                   \
        pt_err = pthread_mutexattr_init(&attr);                                    \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",  \
                "**pthread_mutex %s", strerror(pt_err));                           \
        pt_err = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);      \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",  \
                "**pthread_mutex %s", strerror(pt_err));                           \
        pt_err = pthread_mutex_init((win_ptr)->shm_mutex, &attr);                  \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",  \
                "**pthread_mutex %s", strerror(pt_err));                           \
        pt_err = pthread_mutexattr_destroy(&attr);                                 \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",  \
                "**pthread_mutex %s", strerror(pt_err));                           \
    } while (0);

#define MPIDI_CH3I_SHM_MUTEX_DESTROY(win_ptr)                                      \
    do {                                                                           \
        int pt_err = pthread_mutex_destroy((win_ptr)->shm_mutex);                  \
        MPIR_ERR_CHKANDJUMP1(pt_err, mpi_errno, MPI_ERR_OTHER, "**pthread_mutex",  \
                "**pthread_mutex %s", strerror(pt_err));                           \
    } while (0)

int MPIDI_CH3I_Progress_init(void);
int MPIDI_CH3I_Progress_finalize(void);
int MPIDI_CH3I_Progress_register_hook(int (*progress_fn)(int*), int *id);
int MPIDI_CH3I_Progress_deregister_hook(int id);
int MPIDI_CH3I_Progress_activate_hook(int id);
int MPIDI_CH3I_Progress_deactivate_hook(int id);


/* Starting of shared window list */

typedef struct MPIDI_SHM_Win {
    struct MPIDI_SHM_Win *prev;
    struct MPIDI_SHM_Win *next;
    MPID_Win *win;
} MPIDI_SHM_Win_t;

typedef MPIDI_SHM_Win_t *MPIDI_SHM_Wins_list_t;

extern MPIDI_SHM_Wins_list_t shm_wins_list;

#define MPIDI_SHM_Wins_next_and_continue(elem) {elem = elem->next; continue;}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHM_Wins_append
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_CH3I_SHM_Wins_append(MPIDI_SHM_Wins_list_t * list, MPID_Win * win)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_SHM_Win_t *tmp_ptr;
    MPIU_CHKPMEM_DECL(1);

    /* FIXME: We should use a pool allocator here */
    MPIU_CHKPMEM_MALLOC(tmp_ptr, MPIDI_SHM_Win_t *, sizeof(MPIDI_SHM_Win_t),
                        mpi_errno, "SHM window entry");

    tmp_ptr->next = NULL;
    tmp_ptr->win = win;

    MPL_DL_APPEND(*list, tmp_ptr);

  fn_exit:
    MPIU_CHKPMEM_COMMIT();
    return mpi_errno;
  fn_fail:
    MPIU_CHKPMEM_REAP();
    goto fn_exit;
}

/* Unlink an element from the SHM window list
 *
 * @param IN    list      Pointer to the SHM window list
 * @param IN    elem      Pointer to the element to be unlinked
 */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHM_Wins_unlink
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline void MPIDI_CH3I_SHM_Wins_unlink(MPIDI_SHM_Wins_list_t * list, MPID_Win * shm_win)
{
    MPIDI_SHM_Win_t *elem = NULL;
    MPIDI_SHM_Win_t *tmp_elem = NULL;

    MPL_LL_SEARCH_SCALAR(*list, elem, win, shm_win);
    if (elem != NULL) {
        tmp_elem = elem;
        MPL_DL_DELETE(*list, elem);
        MPIU_Free(tmp_elem);
    }
}

#endif
