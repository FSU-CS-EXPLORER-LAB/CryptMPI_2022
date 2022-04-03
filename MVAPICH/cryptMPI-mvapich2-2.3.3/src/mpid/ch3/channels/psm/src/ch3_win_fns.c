/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpidimpl.h"
#include "mpiinfo.h"
#include "mpidrma.h"
#include "mpiu_os_wrappers_pre.h"
#include "mpiu_shm_wrappers.h"
#include "coll_shmem.h"
#include "mpidi_ch3_impl.h"

#include "bcast_tuning.h"

#undef FUNCNAME

/* FIXME: get this from OS */
#define MPIDI_CH3_PAGESIZE ((MPI_Aint)4096)
#define MPIDI_CH3_PAGESIZE_MASK (~(MPIDI_CH3_PAGESIZE-1))
#define MPIDI_CH3_ROUND_UP_PAGESIZE(x) ((((MPI_Aint)x)+(~MPIDI_CH3_PAGESIZE_MASK)) & MPIDI_CH3_PAGESIZE_MASK)

MPIR_T_PVAR_DOUBLE_TIMER_DECL_EXTERN(RMA, rma_wincreate_allgather);

MPIDI_SHM_Wins_list_t shm_wins_list;

static int MPIDI_CH3I_Win_init(MPI_Aint size, int disp_unit, int create_flavor, int model,
                               MPID_Info * info, MPID_Comm * comm_ptr, MPID_Win ** win_ptr);

static int MPIDI_CH3I_Win_allocate_shm(MPI_Aint size, int disp_unit, MPID_Info * info,
                                       MPID_Comm * comm_ptr, void *base_ptr, MPID_Win ** win_ptr);

static int MPIDI_CH3I_Win_detect_shm(MPID_Win ** win_ptr);

static int MPIDI_CH3I_Win_gather_info(void *base, MPI_Aint size, int disp_unit, MPID_Info * info,
                                      MPID_Comm * comm_ptr, MPID_Win ** win_ptr);

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Win_shared_query
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_SHM_Win_shared_query(MPID_Win * win_ptr, int target_rank, MPI_Aint * size,
                                   int *disp_unit, void *baseptr)
{
    int comm_size;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_WIN_SHARED_QUERY);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3_WIN_SHARED_QUERY);

    comm_size = win_ptr->comm_ptr->local_size;

    if (FALSE == win_ptr->shm_allocated || comm_size <= 1) {
        mpi_errno = MPIDI_CH3U_Win_shared_query(win_ptr, target_rank, size, disp_unit, baseptr);
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
        goto fn_exit;
    }

    /* Scan the sizes to locate the first process that allocated a nonzero
     * amount of space */
    if (target_rank == MPI_PROC_NULL) {
        int i;

        /* Default, if no processes have size > 0. */
        *size = 0;
        *disp_unit = 0;
        *((void **) baseptr) = NULL;

        for (i = 0; i < comm_size; i++) {
            if (win_ptr->basic_info_table[i].size > 0) {
                int local_i = win_ptr->comm_ptr->intranode_table[i];
                MPIU_Assert(local_i >= 0 && local_i < win_ptr->comm_ptr->node_comm->local_size);
                *size = win_ptr->basic_info_table[i].size;
                *disp_unit = win_ptr->basic_info_table[i].disp_unit;
                *((void **) baseptr) = win_ptr->shm_base_addrs[local_i];
                break;
            }
        }

    }
    else {
        int local_target_rank = win_ptr->comm_ptr->intranode_table[target_rank];
        MPIU_Assert(local_target_rank >= 0 &&
                    local_target_rank < win_ptr->comm_ptr->node_comm->local_size);
        *size = win_ptr->basic_info_table[target_rank].size;
        *disp_unit = win_ptr->basic_info_table[target_rank].disp_unit;
        *((void **) baseptr) = win_ptr->shm_base_addrs[local_target_rank];
    }

  fn_exit:
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3_WIN_SHARED_QUERY);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_SHM_Win_free
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_SHM_Win_free(MPID_Win ** win_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_SHM_WIN_FREE);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3_SHM_WIN_FREE);

    if ((*win_ptr)->comm_ptr->local_size <= 1 || (*win_ptr)->node_comm_ptr == NULL) {
        goto fn_exit;
    }

    /* Free shared memory region */
    if ((*win_ptr)->shm_allocated) {
        /* free shm_base_addrs that's only used for shared memory windows */
        MPIU_Free((*win_ptr)->shm_base_addrs);

        /* Only allocate and allocate_shared allocate new shared segments */
        if (((*win_ptr)->create_flavor == MPI_WIN_FLAVOR_SHARED ||
             (*win_ptr)->create_flavor == MPI_WIN_FLAVOR_ALLOCATE) &&
            (*win_ptr)->shm_segment_len > 0) {
            /* detach from shared memory segment */
            mpi_errno =
                MPIU_SHMW_Seg_detach((*win_ptr)->shm_segment_handle,
                                     (char **) &(*win_ptr)->shm_base_addr,
                                     (*win_ptr)->shm_segment_len);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            MPIU_SHMW_Hnd_finalize(&(*win_ptr)->shm_segment_handle);
        }
    }

    /* Free shared process mutex memory region */
    /* Only allocate and allocate_shared allocate new shared mutex.
     * FIXME: it causes unnecessary synchronization when using the same mutex.  */
    if (((*win_ptr)->create_flavor == MPI_WIN_FLAVOR_SHARED ||
         (*win_ptr)->create_flavor == MPI_WIN_FLAVOR_ALLOCATE) &&
        (*win_ptr)->shm_mutex && (*win_ptr)->shm_segment_len > 0) {

        /* When allocating shared memory region segment, we need comm of processes
         * that are on the same node as this process (node_comm).
         * If node_comm == NULL, this process is the only one on this node, therefore
         * we use comm_self as node comm. */

        MPID_Comm *node_comm_ptr = (*win_ptr)->node_comm_ptr;
        MPIU_Assert(node_comm_ptr != NULL);
        if (node_comm_ptr->rank == 0) {
            MPIDI_CH3I_SHM_MUTEX_DESTROY(*win_ptr);
        }

        /* detach from shared memory segment */
        mpi_errno =
            MPIU_SHMW_Seg_detach((*win_ptr)->shm_mutex_segment_handle,
                                 (char **) &(*win_ptr)->shm_mutex, sizeof(MPIDI_CH3I_SHM_MUTEX));
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIU_SHMW_Hnd_finalize(&(*win_ptr)->shm_mutex_segment_handle);
    }

    /* Free shared memory region for window info */
    if ((*win_ptr)->info_shm_base_addr != NULL) {
        mpi_errno = MPIU_SHMW_Seg_detach((*win_ptr)->info_shm_segment_handle,
                                         (char **) &(*win_ptr)->info_shm_base_addr,
                                         (*win_ptr)->info_shm_segment_len);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);

        MPIU_SHMW_Hnd_finalize(&(*win_ptr)->info_shm_segment_handle);

        (*win_ptr)->basic_info_table = NULL;
    }

    /* Unlink from global SHM window list if it is original shared window */
    if ((*win_ptr)->create_flavor == MPI_WIN_FLAVOR_SHARED ||
        (*win_ptr)->create_flavor == MPI_WIN_FLAVOR_ALLOCATE) {
        MPIDI_CH3I_SHM_Wins_unlink(&shm_wins_list, (*win_ptr));
    }

  fn_exit:
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3_SHM_WIN_FREE);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Win_fns_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Win_fns_init(MPIDI_CH3U_Win_fns_t *win_fns)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_WIN_FNS_INIT);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3_WIN_FNS_INIT);

    win_fns->allocate_shm = MPIDI_CH3I_Win_allocate_shm;
    win_fns->detect_shm = MPIDI_CH3I_Win_detect_shm;
    win_fns->gather_info = MPIDI_CH3I_Win_gather_info;
    win_fns->shared_query = MPIDI_CH3_SHM_Win_shared_query;

    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3_WIN_FNS_INIT);

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Win_hooks_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Win_hooks_init(MPIDI_CH3U_Win_hooks_t * win_hooks)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_WIN_HOOKS_INIT);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3_WIN_HOOKS_INIT);

    win_hooks->win_init = MPIDI_CH3I_Win_init;
    win_hooks->win_free = MPIDI_CH3_SHM_Win_free;

    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3_WIN_HOOKS_INIT);

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Win_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3I_Win_init(MPI_Aint size, int disp_unit, int create_flavor, int model,
                               MPID_Info * info, MPID_Comm * comm_ptr, MPID_Win ** win_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WIN_INIT);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WIN_INIT);

    (*win_ptr)->shm_base_addr = NULL;
    (*win_ptr)->shm_segment_len = 0;
    (*win_ptr)->shm_segment_handle = 0;
    (*win_ptr)->shm_mutex = NULL;
    (*win_ptr)->shm_mutex_segment_handle = 0;

    (*win_ptr)->info_shm_base_addr = NULL;
    (*win_ptr)->info_shm_segment_len = 0;
    (*win_ptr)->info_shm_segment_handle = 0;

    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WIN_INIT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_SHM_Wins_match
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3I_SHM_Wins_match(MPID_Win ** win_ptr, MPID_Win ** matched_win,
                                     MPI_Aint ** base_shm_offs_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int i, comm_size;
    int node_size, node_rank, shm_node_size;
    int group_diff;
    int base_diff;

    MPID_Comm *node_comm_ptr = NULL, *shm_node_comm_ptr = NULL;
    int *node_ranks = NULL, *node_ranks_in_shm_node = NULL;
    MPID_Group *node_group_ptr = NULL, *shm_node_group_ptr = NULL;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPI_Aint *base_shm_offs;

    MPIDI_SHM_Win_t *elem = shm_wins_list;

    MPIU_CHKLMEM_DECL(2);
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_SHM_WINS_MATCH);
    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_SHM_WINS_MATCH);

    *matched_win = NULL;
    base_shm_offs = *base_shm_offs_ptr;
    node_comm_ptr = (*win_ptr)->comm_ptr->node_comm;
    MPIU_Assert(node_comm_ptr != NULL);
    node_size = node_comm_ptr->local_size;
    node_rank = node_comm_ptr->rank;

    comm_size = (*win_ptr)->comm_ptr->local_size;

    MPIU_CHKLMEM_MALLOC(node_ranks, int *, node_size * sizeof(int), mpi_errno, "node_ranks");
    MPIU_CHKLMEM_MALLOC(node_ranks_in_shm_node, int *, node_size * sizeof(int),
                        mpi_errno, "node_ranks_in_shm_comm");

    for (i = 0; i < node_size; i++) {
        node_ranks[i] = i;
    }

    mpi_errno = MPIR_Comm_group_impl(node_comm_ptr, &node_group_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    while (elem != NULL) {
        MPID_Win *shm_win = elem->win;
        if (!shm_win)
            MPIDI_SHM_Wins_next_and_continue(elem);

        /* Compare node_comm.
         *
         * Only support shm if new node_comm is equal to or a subset of shm node_comm.
         * Shm node_comm == a subset of node_comm is not supported, because it means
         * some processes of node_comm cannot be shared, but RMA operation simply checks
         * the node_id of a target process for distinguishing shm target.  */
        shm_node_comm_ptr = shm_win->comm_ptr->node_comm;
        shm_node_size = shm_node_comm_ptr->local_size;

        if (node_size > shm_node_size)
            MPIDI_SHM_Wins_next_and_continue(elem);

        mpi_errno = MPIR_Comm_group_impl(shm_win->comm_ptr, &shm_node_group_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno = MPIR_Group_translate_ranks_impl(node_group_ptr, node_size,
                                                    node_ranks, shm_node_group_ptr,
                                                    node_ranks_in_shm_node);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno = MPIR_Group_free_impl(shm_node_group_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        shm_node_group_ptr = NULL;

        group_diff = 0;
        for (i = 0; i < node_size; i++) {
            /* not exist in shm_comm->node_comm */
            if (node_ranks_in_shm_node[i] == MPI_UNDEFINED) {
                group_diff = 1;
                break;
            }
        }
        if (group_diff)
            MPIDI_SHM_Wins_next_and_continue(elem);

        /* Gather the offset of base_addr from all local processes. Match only
         * when all of them are included in the shm segment in current shm_win.
         *
         * Note that this collective call must be called after checking the
         * group match in order to guarantee all the local processes can perform
         * this call. */
        base_shm_offs[node_rank] = (MPI_Aint) ((*win_ptr)->base)
            - (MPI_Aint) (shm_win->shm_base_addr);
        mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                        base_shm_offs, 1, MPI_AINT, node_comm_ptr, &errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        base_diff = 0;
        for (i = 0; i < comm_size; ++i) {
            int i_node_rank = (*win_ptr)->comm_ptr->intranode_table[i];
            if (i_node_rank >= 0) {
                MPIU_Assert(i_node_rank < node_size);

                if (base_shm_offs[i_node_rank] < 0 ||
                    base_shm_offs[i_node_rank] + (*win_ptr)->basic_info_table[i].size >
                    shm_win->shm_segment_len) {
                    base_diff = 1;
                    break;
                }
            }
        }

        if (base_diff)
            MPIDI_SHM_Wins_next_and_continue(elem);

        /* Found the first matched shm_win */
        *matched_win = shm_win;
        break;
    }

  fn_exit:
    if (node_group_ptr != NULL)
        mpi_errno = MPIR_Group_free_impl(node_group_ptr);
    /* Only free it here when group_translate_ranks fails. */
    if (shm_node_group_ptr != NULL)
        mpi_errno = MPIR_Group_free_impl(shm_node_group_ptr);

    MPIU_CHKLMEM_FREEALL();
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_SHM_WINS_MATCH);
    return mpi_errno;
    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Win_detect_shm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3I_Win_detect_shm(MPID_Win ** win_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_Win *shm_win_ptr = NULL;
    int i, node_size;
    MPI_Aint *base_shm_offs;

    MPIU_CHKPMEM_DECL(1);
    MPIU_CHKLMEM_DECL(1);
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WIN_DETECT_SHM);
    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WIN_DETECT_SHM);

    if ((*win_ptr)->comm_ptr->node_comm == NULL) {
        goto fn_exit;
    }

    node_size = (*win_ptr)->comm_ptr->node_comm->local_size;

    MPIU_CHKLMEM_MALLOC(base_shm_offs, MPI_Aint *, node_size * sizeof(MPI_Aint),
                        mpi_errno, "base_shm_offs");

    /* Return the first matched shared window.
     * It is noted that the shared windows including all local processes are
     * stored in every local process in the same order, hence the first matched
     * shared window on every local process should be the same. */
    mpi_errno = MPIDI_CH3I_SHM_Wins_match(win_ptr, &shm_win_ptr, &base_shm_offs);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
    if (shm_win_ptr == NULL)
        goto fn_exit;

    (*win_ptr)->shm_allocated = TRUE;
    MPIU_CHKPMEM_MALLOC((*win_ptr)->shm_base_addrs, void **,
                        node_size * sizeof(void *), mpi_errno, "(*win_ptr)->shm_base_addrs");

    /* Compute the base address of shm buffer on each process.
     * shm_base_addrs[i] = my_shm_base_addr + off[i] */
    for (i = 0; i < node_size; i++) {
        (*win_ptr)->shm_base_addrs[i] =
            (void *) ((MPI_Aint) shm_win_ptr->shm_base_addr + base_shm_offs[i]);
    }

    /* TODO: should we use the same mutex or create a new one ?
     * It causes unnecessary synchronization.*/
    (*win_ptr)->shm_mutex = shm_win_ptr->shm_mutex;

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WIN_DETECT_SHM);
    return mpi_errno;
    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    MPIU_CHKPMEM_REAP();
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Win_gather_info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3I_Win_gather_info(void *base, MPI_Aint size, int disp_unit, MPID_Info * info,
                                      MPID_Comm * comm_ptr, MPID_Win ** win_ptr)
{
    MPID_Comm *node_comm_ptr = NULL;
    int node_rank, node_size;
    int comm_rank, comm_size;
    MPI_Aint *tmp_buf = NULL;
    MPIDI_VC_t *vc = NULL;
    int i, k;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int mpi_errno = MPI_SUCCESS;
    MPIU_CHKLMEM_DECL(1);
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WIN_GATHER_INFO);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WIN_GATHER_INFO);

    comm_size = (*win_ptr)->comm_ptr->local_size;
    comm_rank = (*win_ptr)->comm_ptr->rank;

    if (comm_size <= 1) {
        mpi_errno = MPIDI_CH3U_Win_gather_info(base, size, disp_unit, info, comm_ptr, win_ptr);
        goto fn_exit;
    }

    /* tell everyone, what is my COMM_WORLD rank */
    (*win_ptr)->rank_mapping[comm_rank] = MPIDI_Process.my_pg_rank;
    MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
           (*win_ptr)->rank_mapping, sizeof(uint32_t), MPI_BYTE,
          comm_ptr, &errflag);
        
   
    if (!mv2_enable_shmem_collectives && (*win_ptr)->shm_coll_comm_ref == -1) {
        MPIDI_PG_Get_vc(MPIDI_Process.my_pg, MPIDI_Process.my_pg_rank, &vc);
        /* Shared memory for collectives */
        mpi_errno = MPIDI_CH3I_SHMEM_COLL_init(MPIDI_Process.my_pg,
                vc->smp.local_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* local barrier */
        mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* Memory Mapping shared files for collectives*/
        mpi_errno = MPIDI_CH3I_SHMEM_COLL_Mmap(MPIDI_Process.my_pg,
                vc->smp.local_rank);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* local barrier */
        mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        /* Unlink mapped files so that they get cleaned up when
         * process exits */
        MPIDI_CH3I_SHMEM_COLL_Unlink();
        (*win_ptr)->shm_coll_comm_ref = 1;
    } else if ((*win_ptr)->shm_coll_comm_ref > 0) {
        (*win_ptr)->shm_coll_comm_ref++;
    }

    if ((*win_ptr)->node_comm_ptr == NULL) {
        if((*win_ptr)->comm_ptr->dev.ch.shmem_coll_ok == 0) {
            mpi_errno = create_2level_comm((*win_ptr)->comm_ptr->handle, 
                    (*win_ptr)->comm_ptr->local_size, (*win_ptr)->comm_ptr->rank);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }           

        MPI_Comm shmem_comm;
        shmem_comm = (*win_ptr)->comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, node_comm_ptr);

        /* Fall back to no_shm function if shmem_comm is not created
         * successfully*/
        if (node_comm_ptr == NULL) {
            mpi_errno = MPIDI_CH3U_Win_gather_info(base, size, disp_unit, info, comm_ptr, win_ptr);
            (*win_ptr)->shm_allocated = FALSE;
            goto fn_exit;
        }

        MPIU_Assert(node_comm_ptr != NULL);

        (*win_ptr)->node_comm_ptr = node_comm_ptr;
    } else {
        node_comm_ptr = (*win_ptr)->node_comm_ptr;
    }

    node_size = node_comm_ptr->local_size;
    node_rank = node_comm_ptr->rank;

    (*win_ptr)->node_comm_size = node_size;
    (*win_ptr)->info_shm_segment_len = comm_size * sizeof(MPIDI_Win_basic_info_t);

    mpi_errno = MPIU_SHMW_Hnd_init(&(*win_ptr)->info_shm_segment_handle);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    if (node_rank == 0) {
        char *serialized_hnd_ptr = NULL;

        /* create shared memory region for all processes in win and map. */
        mpi_errno = MPIU_SHMW_Seg_create_and_attach((*win_ptr)->info_shm_segment_handle,
                                                    (*win_ptr)->info_shm_segment_len,
                                                    (char **) &(*win_ptr)->info_shm_base_addr, 0);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* serialize handle and broadcast it to the other processes in win */
        mpi_errno =
            MPIU_SHMW_Hnd_get_serialized_by_ref((*win_ptr)->info_shm_segment_handle,
                                                &serialized_hnd_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno =
            MPIR_Shmem_Bcast_MV2(serialized_hnd_ptr, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                    &errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        /* wait for other processes to attach to win */
        mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        /* unlink shared memory region so it gets deleted when all processes exit */
        mpi_errno = MPIU_SHMW_Seg_remove((*win_ptr)->info_shm_segment_handle);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }
    else {
        char serialized_hnd[MPIU_SHMW_GHND_SZ] = { 0 };

        /* get serialized handle from rank 0 and deserialize it */
        mpi_errno =
            MPIR_Shmem_Bcast_MV2(serialized_hnd, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                    &errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        mpi_errno = MPIU_SHMW_Hnd_deserialize((*win_ptr)->info_shm_segment_handle, serialized_hnd,
                                              strlen(serialized_hnd));
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* attach to shared memory region created by rank 0 */
        mpi_errno =
            MPIU_SHMW_Seg_attach((*win_ptr)->info_shm_segment_handle,
                                 (*win_ptr)->info_shm_segment_len,
                                 (char **) &(*win_ptr)->info_shm_base_addr, 0);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    }

    (*win_ptr)->basic_info_table = (MPIDI_Win_basic_info_t *) ((*win_ptr)->info_shm_base_addr);

    MPIU_CHKLMEM_MALLOC(tmp_buf, MPI_Aint *, 4 * comm_size * sizeof(MPI_Aint),
                        mpi_errno, "tmp_buf");

    tmp_buf[4 * comm_rank] = MPIU_PtrToAint(base);
    tmp_buf[4 * comm_rank + 1] = size;
    tmp_buf[4 * comm_rank + 2] = (MPI_Aint) disp_unit;
    tmp_buf[4 * comm_rank + 3] = (MPI_Aint) (*win_ptr)->handle;

    mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tmp_buf, 4, MPI_AINT,
                                    (*win_ptr)->comm_ptr, &errflag);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    if (node_rank == 0) {
        /* only node_rank == 0 writes results to basic_info_table on shared memory region. */
        k = 0;
        for (i = 0; i < comm_size; i++) {
            (*win_ptr)->basic_info_table[i].base_addr = MPIU_AintToPtr(tmp_buf[k++]);
            (*win_ptr)->basic_info_table[i].size = tmp_buf[k++];
            (*win_ptr)->basic_info_table[i].disp_unit = (int) tmp_buf[k++];
            (*win_ptr)->basic_info_table[i].win_handle = (MPI_Win) tmp_buf[k++];
        }
    }

    /* Make sure that all local processes see the results written by node_rank == 0 */
    mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    /* call psm to pre-post receive buffers for Puts */
    psm_prepost_1sc();
    MPIR_Barrier_impl((*win_ptr)->comm_ptr, &errflag);

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WIN_GATHER_INFO);
    return mpi_errno;
    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Win_allocate_shm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIDI_CH3I_Win_allocate_shm(MPI_Aint size, int disp_unit, MPID_Info * info,
                                       MPID_Comm * comm_ptr, void *base_ptr, MPID_Win ** win_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    void **base_pp = (void **) base_ptr;
    int i, node_size, node_rank;
    MPID_Comm *node_comm_ptr = NULL;
    MPI_Aint *node_sizes;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int noncontig = FALSE;
    int comm_size;
    MPIDI_VC_t *vc = NULL;
    MPIU_CHKPMEM_DECL(1);
    MPIU_CHKLMEM_DECL(1);
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_WIN_ALLOCATE_SHM);

    MPIDI_RMA_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_WIN_ALLOCATE_SHM);

    comm_size = (*win_ptr)->comm_ptr->local_size;

    if (comm_size <= 1) {
        mpi_errno =
            MPIDI_CH3U_Win_allocate_no_shm(size, disp_unit, info, comm_ptr, base_ptr, win_ptr);
        goto fn_exit;
    }

    /* see if we can allocate all windows contiguously */
    noncontig = (*win_ptr)->info_args.alloc_shared_noncontig;

    (*win_ptr)->shm_allocated = TRUE;

    /* When allocating shared memory region segment, we need comm of processes
     * that are on the same node as this process (node_comm).
     * If node_comm == NULL, this process is the only one on this node, therefore
     * we use comm_self as node comm. */

    if (likely(!(*win_ptr)->node_comm_ptr)) {
        if (!mv2_enable_shmem_collectives && (*win_ptr)->shm_coll_comm_ref == -1) {
            MPIDI_PG_Get_vc(MPIDI_Process.my_pg, MPIDI_Process.my_pg_rank, &vc);
            /* Shared memory for collectives */
            mpi_errno = MPIDI_CH3I_SHMEM_COLL_init(MPIDI_Process.my_pg,
                    vc->smp.local_rank);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* local barrier */
            mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* Memory Mapping shared files for collectives*/
            mpi_errno = MPIDI_CH3I_SHMEM_COLL_Mmap(MPIDI_Process.my_pg,
                    vc->smp.local_rank);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* local barrier */
            mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* Unlink mapped files so that they get cleaned up when
             * process exits */
            MPIDI_CH3I_SHMEM_COLL_Unlink();
            (*win_ptr)->shm_coll_comm_ref = 1;
        } else if ((*win_ptr)->shm_coll_comm_ref > 0) {
            (*win_ptr)->shm_coll_comm_ref++;
        }

        if((*win_ptr)->comm_ptr->dev.ch.shmem_coll_ok == 0) {
            mpi_errno = create_2level_comm((*win_ptr)->comm_ptr->handle,
                    (*win_ptr)->comm_ptr->local_size, (*win_ptr)->comm_ptr->rank);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }

        MPI_Comm shmem_comm;
        shmem_comm = (*win_ptr)->comm_ptr->dev.ch.shmem_comm;
        MPID_Comm_get_ptr(shmem_comm, node_comm_ptr);


        /* Fall back to no_shm function if shmem_comm is not created successfully*/
        if (node_comm_ptr == NULL) {
            mpi_errno =
                MPIDI_CH3U_Win_allocate_no_shm(size, disp_unit, info, comm_ptr, base_ptr, win_ptr);
            (*win_ptr)->shm_allocated = FALSE;
            goto fn_exit;
        }

        MPIU_Assert(node_comm_ptr != NULL);

        node_size = node_comm_ptr->local_size;
        node_rank = node_comm_ptr->rank;
        (*win_ptr)->node_comm_ptr = node_comm_ptr;
    } else {
        node_comm_ptr = (*win_ptr)->node_comm_ptr;
    }

    MPIR_T_PVAR_TIMER_START(RMA, rma_wincreate_allgather);
    /* allocate memory for the base addresses, disp_units, and
     * completion counters of all processes */
    MPIU_CHKPMEM_MALLOC((*win_ptr)->shm_base_addrs, void **,
                        node_size * sizeof(void *), mpi_errno, "(*win_ptr)->shm_base_addrs");

    /* get the sizes of the windows and window objectsof
     * all processes.  allocate temp. buffer for communication */
    MPIU_CHKLMEM_MALLOC(node_sizes, MPI_Aint *, node_size * sizeof(MPI_Aint), mpi_errno,
                        "node_sizes");

    /* FIXME: This needs to be fixed for heterogeneous systems */
    node_sizes[node_rank] = (MPI_Aint) size;

    mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            node_sizes, sizeof(MPI_Aint), MPI_BYTE,
            node_comm_ptr, &errflag);

    MPIR_T_PVAR_TIMER_END(RMA, rma_wincreate_allgather);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
    MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    (*win_ptr)->shm_segment_len = 0;

    for (i = 0; i < node_size; i++) {
        if (noncontig)
            /* Round up to next page size */
            (*win_ptr)->shm_segment_len += MPIDI_CH3_ROUND_UP_PAGESIZE(node_sizes[i]);
        else
            (*win_ptr)->shm_segment_len += node_sizes[i];
    }

    if ((*win_ptr)->shm_segment_len == 0) {
        (*win_ptr)->base = NULL;
    }

    else {
        mpi_errno = MPIU_SHMW_Hnd_init(&(*win_ptr)->shm_segment_handle);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        if (node_rank == 0) {
            char *serialized_hnd_ptr = NULL;

            /* create shared memory region for all processes in win and map */
            mpi_errno =
                MPIU_SHMW_Seg_create_and_attach((*win_ptr)->shm_segment_handle,
                                                (*win_ptr)->shm_segment_len,
                                                (char **) &(*win_ptr)->shm_base_addr, 0);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* serialize handle and broadcast it to the other processes in win */
            mpi_errno =
                MPIU_SHMW_Hnd_get_serialized_by_ref((*win_ptr)->shm_segment_handle,
                                                    &serialized_hnd_ptr);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            mpi_errno =
                MPIR_Shmem_Bcast_MV2(serialized_hnd_ptr, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                        &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            /* wait for other processes to attach to win */
            mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            /* unlink shared memory region so it gets deleted when all processes exit */
            mpi_errno = MPIU_SHMW_Seg_remove((*win_ptr)->shm_segment_handle);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

        }
        else {
            char serialized_hnd[MPIU_SHMW_GHND_SZ] = { 0 };

            /* get serialized handle from rank 0 and deserialize it */
            mpi_errno =
                MPIR_Shmem_Bcast_MV2(serialized_hnd, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                        &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            mpi_errno =
                MPIU_SHMW_Hnd_deserialize((*win_ptr)->shm_segment_handle, serialized_hnd,
                                          strlen(serialized_hnd));
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* attach to shared memory region created by rank 0 */
            mpi_errno =
                MPIU_SHMW_Seg_attach((*win_ptr)->shm_segment_handle, (*win_ptr)->shm_segment_len,
                                     (char **) &(*win_ptr)->shm_base_addr, 0);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
        }

        /* Allocated the interprocess mutex segment. */
        mpi_errno = MPIU_SHMW_Hnd_init(&(*win_ptr)->shm_mutex_segment_handle);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        if (node_rank == 0) {
            char *serialized_hnd_ptr = NULL;

            /* create shared memory region for all processes in win and map */
            mpi_errno =
                MPIU_SHMW_Seg_create_and_attach((*win_ptr)->shm_mutex_segment_handle,
                                                sizeof(MPIDI_CH3I_SHM_MUTEX),
                                                (char **) &(*win_ptr)->shm_mutex, 0);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            MPIDI_CH3I_SHM_MUTEX_INIT(*win_ptr);

            /* serialize handle and broadcast it to the other processes in win */
            mpi_errno =
                MPIU_SHMW_Hnd_get_serialized_by_ref((*win_ptr)->shm_mutex_segment_handle,
                                                    &serialized_hnd_ptr);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            mpi_errno =
                MPIR_Shmem_Bcast_MV2(serialized_hnd_ptr, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                        &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            /* wait for other processes to attach to win */
            mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            /* unlink shared memory region so it gets deleted when all processes exit */
            mpi_errno = MPIU_SHMW_Seg_remove((*win_ptr)->shm_mutex_segment_handle);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
        else {
            char serialized_hnd[MPIU_SHMW_GHND_SZ] = { 0 };

            /* get serialized handle from rank 0 and deserialize it */
            mpi_errno =
                MPIR_Shmem_Bcast_MV2(serialized_hnd, MPIU_SHMW_GHND_SZ, MPI_CHAR, 0, node_comm_ptr,
                        &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

            mpi_errno =
                MPIU_SHMW_Hnd_deserialize((*win_ptr)->shm_mutex_segment_handle, serialized_hnd,
                                          strlen(serialized_hnd));
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            /* attach to shared memory region created by rank 0 */
            mpi_errno =
                MPIU_SHMW_Seg_attach((*win_ptr)->shm_mutex_segment_handle,
                                     sizeof(MPIDI_CH3I_SHM_MUTEX), (char **) &(*win_ptr)->shm_mutex,
                                     0);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            mpi_errno = MPIR_Barrier_impl(node_comm_ptr, &errflag);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
        }

        /* compute the base addresses of each process within the shared memory segment */
        {
            char *cur_base;
            int cur_rank;

            cur_base = (*win_ptr)->shm_base_addr;
            cur_rank = 0;
            ((*win_ptr)->shm_base_addrs)[0] = (*win_ptr)->shm_base_addr;
            for (i = 1; i < node_size; ++i) {
                if (node_sizes[i]) {
                    /* For the base addresses, we track the previous
                     * process that has allocated non-zero bytes of shared
                     * memory.  We can not simply use "i-1" for the
                     * previous process because rank "i-1" might not have
                     * allocated any memory. */
                    if (noncontig) {
                        ((*win_ptr)->shm_base_addrs)[i] =
                            cur_base + MPIDI_CH3_ROUND_UP_PAGESIZE(node_sizes[cur_rank]);
                    }
                    else {
                        ((*win_ptr)->shm_base_addrs)[i] = cur_base + node_sizes[cur_rank];
                    }
                    cur_base = ((*win_ptr)->shm_base_addrs)[i];
                    cur_rank = i;
                }
                else {
                    ((*win_ptr)->shm_base_addrs)[i] = NULL;
                }
            }
        }

        (*win_ptr)->base = (*win_ptr)->shm_base_addrs[node_rank];
    }

    *base_pp = (*win_ptr)->base;

    /* gather window information among processes via shared memory region. */
    mpi_errno = MPIDI_CH3I_Win_gather_info((*base_pp), size, disp_unit, info, comm_ptr, win_ptr);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    /* Cache SHM windows */
    MPIDI_CH3I_SHM_Wins_append(&shm_wins_list, (*win_ptr));

  fn_exit:
    MPIU_CHKLMEM_FREEALL();
    MPIDI_RMA_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_WIN_ALLOCATE_SHM);
    return mpi_errno;
    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    MPIU_CHKPMEM_REAP();
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
