/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"
#include "mpicomm.h"
#include "mpir_info.h"  /* MPIR_Info_free */

#include "utlist.h"
#include "uthash.h"

/* This is the utility file for comm that contains the basic comm items
   and storage management */
#ifndef MPID_COMM_PREALLOC
#define MPID_COMM_PREALLOC 8
#endif

/* Preallocated comm objects */
/* initialized in initthread.c */
MPIR_Comm MPIR_Comm_builtin[MPIR_COMM_N_BUILTIN] = { {0}
};
MPIR_Comm MPIR_Comm_direct[MPID_COMM_PREALLOC] = { {0}
};

MPIR_Object_alloc_t MPIR_Comm_mem = {
    0,
    0,
    0,
    0,
    MPIR_COMM,
    sizeof(MPIR_Comm),
    MPIR_Comm_direct,
    MPID_COMM_PREALLOC
};

/* Communicator creation functions */
struct MPIR_Commops *MPIR_Comm_fns = NULL;
struct MPIR_Comm_hint_fn_elt {
    char name[MPI_MAX_INFO_KEY];
    MPIR_Comm_hint_fn_t fn;
    void *state;
    UT_hash_handle hh;
};
static struct MPIR_Comm_hint_fn_elt *MPID_hint_fns = NULL;

/* FIXME :
   Reusing context ids can lead to a race condition if (as is desirable)
   MPI_Comm_free does not include a barrier.  Consider the following:
   Process A frees the communicator.
   Process A creates a new communicator, reusing the just released id
   Process B sends a message to A on the old communicator.
   Process A receives the message, and believes that it belongs to the
   new communicator.
   Process B then cancels the message, and frees the communicator.

   The likelihood of this happening can be reduced by introducing a gap
   between when a context id is released and when it is reused.  An alternative
   is to use an explicit message (in the implementation of MPI_Comm_free)
   to indicate that a communicator is being freed; this will often require
   less communication than a barrier in MPI_Comm_free, and will ensure that
   no messages are later sent to the same communicator (we may also want to
   have a similar check when building fault-tolerant versions of MPI).
 */

/* Zeroes most non-handle fields in a communicator, as well as initializing any
 * other special fields, such as a per-object mutex.  Also defaults the
 * reference count to 1, under the assumption that the caller holds a reference
 * to it.
 *
 * !!! The resulting struct is _not_ ready for communication !!! */
int MPII_Comm_init(MPIR_Comm * comm_p)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Object_set_ref(comm_p, 1);

    /* initialize local and remote sizes to -1 to allow other parts of
     * the stack to detect errors more easily */
    comm_p->local_size = -1;
    comm_p->remote_size = -1;

    /* Clear many items (empty means to use the default; some of these
     * may be overridden within the upper-level communicator initialization) */
    comm_p->errhandler = NULL;
    comm_p->attributes = NULL;
    comm_p->remote_group = NULL;
    comm_p->local_group = NULL;
    comm_p->topo_fns = NULL;
    comm_p->name[0] = '\0';
    comm_p->info = NULL;

    comm_p->hierarchy_kind = MPIR_COMM_HIERARCHY_KIND__FLAT;
    comm_p->node_comm = NULL;
    comm_p->node_roots_comm = NULL;
    comm_p->intranode_table = NULL;
    comm_p->internode_table = NULL;

    /* abstractions bleed a bit here... :(*/
    comm_p->next_sched_tag = MPIR_FIRST_NBC_TAG;

    /* Initialize the revoked flag as false */
    comm_p->revoked = 0;
    comm_p->mapper_head = NULL;
    comm_p->mapper_tail = NULL;

#if MPICH_THREAD_GRANULARITY == MPICH_THREAD_GRANULARITY__POBJ
    {
        int thr_err;
        MPID_Thread_mutex_create(&MPIR_THREAD_POBJ_COMM_MUTEX(comm_p), &thr_err);
        MPIR_Assert(thr_err == 0);
    }
#endif
    /* Fields not set include context_id, remote and local size, and
     * kind, since different communicator construction routines need
     * different values */
    return mpi_errno;
}


/*
    Create a communicator structure and perform basic initialization
    (mostly clearing fields and updating the reference count).
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Comm_create
#undef FCNAME
#define FCNAME "MPIR_Comm_create"
int MPIR_Comm_create(MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *newptr;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_CREATE);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_CREATE);

    newptr = (MPIR_Comm *) MPIR_Handle_obj_alloc(&MPIR_Comm_mem);
    MPIR_ERR_CHKANDJUMP(!newptr, mpi_errno, MPI_ERR_OTHER, "**nomem");

    *newcomm_ptr = newptr;

    mpi_errno = MPII_Comm_init(newptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Insert this new communicator into the list of known communicators.
     * Make this conditional on debugger support to match the test in
     * MPIR_Comm_release . */
    MPII_COMML_REMEMBER(newptr);

  fn_fail:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_CREATE);

    return mpi_errno;
}

/* Create a local intra communicator from the local group of the
   specified intercomm. */
/* FIXME this is an alternative constructor that doesn't use MPIR_Comm_create! */
#undef FUNCNAME
#define FUNCNAME MPII_Setup_intercomm_localcomm
#undef FCNAME
#define FCNAME "MPII_Setup_intercomm_localcomm"
int MPII_Setup_intercomm_localcomm(MPIR_Comm * intercomm_ptr)
{
    MPIR_Comm *localcomm_ptr;
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_SETUP_INTERCOMM_LOCALCOMM);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_SETUP_INTERCOMM_LOCALCOMM);

    localcomm_ptr = (MPIR_Comm *) MPIR_Handle_obj_alloc(&MPIR_Comm_mem);
    MPIR_ERR_CHKANDJUMP(!localcomm_ptr, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* get sensible default values for most fields (usually zeros) */
    mpi_errno = MPII_Comm_init(localcomm_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* use the parent intercomm's recv ctx as the basis for our ctx */
    localcomm_ptr->recvcontext_id =
        MPIR_CONTEXT_SET_FIELD(IS_LOCALCOMM, intercomm_ptr->recvcontext_id, 1);
    localcomm_ptr->context_id = localcomm_ptr->recvcontext_id;

    MPL_DBG_MSG_FMT(MPIR_DBG_COMM, TYPICAL,
                    (MPL_DBG_FDEST,
                     "setup_intercomm_localcomm ic=%p ic->context_id=%d ic->recvcontext_id=%d lc->recvcontext_id=%d",
                     intercomm_ptr, intercomm_ptr->context_id, intercomm_ptr->recvcontext_id,
                     localcomm_ptr->recvcontext_id));

    /* Save the kind of the communicator */
    localcomm_ptr->comm_kind = MPIR_COMM_KIND__INTRACOMM;

    /* Set the sizes and ranks */
    localcomm_ptr->remote_size = intercomm_ptr->local_size;
    localcomm_ptr->local_size = intercomm_ptr->local_size;
    localcomm_ptr->pof2 = intercomm_ptr->pof2;
    localcomm_ptr->rank = intercomm_ptr->rank;

    MPIR_Comm_map_dup(localcomm_ptr, intercomm_ptr, MPIR_COMM_MAP_DIR__L2L);

    /* TODO More advanced version: if the group is available, dup it by
     * increasing the reference count instead of recreating it later */
    /* FIXME  : No local functions for the topology routines */

    intercomm_ptr->local_comm = localcomm_ptr;

    /* sets up the SMP-aware sub-communicators and tables */
    mpi_errno = MPIR_Comm_commit(localcomm_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_fail:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_SETUP_INTERCOMM_LOCALCOMM);

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Comm_map_irregular
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_map_irregular(MPIR_Comm * newcomm, MPIR_Comm * src_comm,
                            int *src_mapping, int src_mapping_size,
                            MPIR_Comm_map_dir_t dir, MPIR_Comm_map_t ** map)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm_map_t *mapper;
    MPIR_CHKPMEM_DECL(3);
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_MAP_TYPE__IRREGULAR);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_MAP_TYPE__IRREGULAR);

    MPIR_CHKPMEM_MALLOC(mapper, MPIR_Comm_map_t *, sizeof(MPIR_Comm_map_t), mpi_errno, "mapper",
                        MPL_MEM_COMM);

    mapper->type = MPIR_COMM_MAP_TYPE__IRREGULAR;
    mapper->src_comm = src_comm;
    mapper->dir = dir;
    mapper->src_mapping_size = src_mapping_size;

    if (src_mapping) {
        mapper->src_mapping = src_mapping;
        mapper->free_mapping = 0;
    } else {
        MPIR_CHKPMEM_MALLOC(mapper->src_mapping, int *,
                            src_mapping_size * sizeof(int), mpi_errno, "mapper mapping",
                            MPL_MEM_COMM);
        mapper->free_mapping = 1;
    }

    mapper->next = NULL;

    LL_APPEND(newcomm->mapper_head, newcomm->mapper_tail, mapper);

    if (map)
        *map = mapper;

  fn_exit:
    MPIR_CHKPMEM_COMMIT();
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_MAP_TYPE__IRREGULAR);
    return mpi_errno;
  fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Comm_map_dup
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_map_dup(MPIR_Comm * newcomm, MPIR_Comm * src_comm, MPIR_Comm_map_dir_t dir)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm_map_t *mapper;
    MPIR_CHKPMEM_DECL(1);
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_MAP_TYPE__DUP);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_MAP_TYPE__DUP);

    MPIR_CHKPMEM_MALLOC(mapper, MPIR_Comm_map_t *, sizeof(MPIR_Comm_map_t), mpi_errno, "mapper",
                        MPL_MEM_COMM);

    mapper->type = MPIR_COMM_MAP_TYPE__DUP;
    mapper->src_comm = src_comm;
    mapper->dir = dir;

    mapper->next = NULL;

    LL_APPEND(newcomm->mapper_head, newcomm->mapper_tail, mapper);

  fn_exit:
    MPIR_CHKPMEM_COMMIT();
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_MAP_TYPE__DUP);
    return mpi_errno;
  fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIR_Comm_map_free
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_map_free(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm_map_t *mapper, *tmp;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_MAP_FREE);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_MAP_FREE);

    for (mapper = comm->mapper_head; mapper;) {
        tmp = mapper->next;
        if (mapper->type == MPIR_COMM_MAP_TYPE__IRREGULAR && mapper->free_mapping)
            MPL_free(mapper->src_mapping);
        MPL_free(mapper);
        mapper = tmp;
    }
    comm->mapper_head = NULL;

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_MAP_FREE);
    return mpi_errno;
}

/* Provides a hook for the top level functions to perform some manipulation on a
   communicator just before it is given to the application level.

   For example, we create sub-communicators for SMP-aware collectives at this
   step. */
#undef FUNCNAME
#define FUNCNAME MPIR_Comm_commit
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_commit(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    int num_local = -1, num_external = -1;
    int local_rank = -1, external_rank = -1;
    int *local_procs = NULL, *external_procs = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_COMMIT);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_COMMIT);

    /* It's OK to relax these assertions, but we should do so very
     * intentionally.  For now this function is the only place that we create
     * our hierarchy of communicators */
    MPIR_Assert(comm->node_comm == NULL);
    MPIR_Assert(comm->node_roots_comm == NULL);

    /* Notify device of communicator creation */
    if (comm != MPIR_Process.comm_world) {
        mpi_errno = MPID_Comm_create_hook(comm);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIR_Comm_map_free(comm);
    }

    /* Create collectives-specific infrastructure */
    mpi_errno = MPIR_Coll_comm_init(comm);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    MPIR_Comm_map_free(comm);

    comm->pof2 = MPL_pof2(comm->local_size);

    if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM && !MPIR_CONTEXT_READ_FIELD(SUBCOMM, comm->context_id)) {  /*make sure this is not a subcomm */

        mpi_errno = MPIR_Find_local_and_external(comm,
                                                 &num_local, &local_rank, &local_procs,
                                                 &num_external, &external_rank, &external_procs,
                                                 &comm->intranode_table, &comm->internode_table, &comm->node_sizes);
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno) {
            if (MPIR_Err_is_fatal(mpi_errno))
                MPIR_ERR_POP(mpi_errno);

            /* Non-fatal errors simply mean that this communicator will not have
             * any node awareness.  Node-aware collectives are an optimization. */
            MPL_DBG_MSG_P(MPIR_DBG_COMM, VERBOSE,
                          "MPIR_Find_local_and_external failed for comm_ptr=%p", comm);
            if (comm->intranode_table)
                MPL_free(comm->intranode_table);
            if (comm->internode_table)
                MPL_free(comm->internode_table);
            if (comm->node_sizes)
                MPL_free(comm->node_sizes);

            mpi_errno = MPI_SUCCESS;
            goto fn_exit;
        }
        /* --END ERROR HANDLING-- */

        /* defensive checks */
        MPIR_Assert(num_local > 0);
        MPIR_Assert(num_local > 1 || external_rank >= 0);
        MPIR_Assert(external_rank < 0 || external_procs != NULL);

        /* if the node_roots_comm and comm would be the same size, then creating
         * the second communicator is useless and wasteful. */
        if (num_external == comm->remote_size) {
            MPIR_Assert(num_local == 1);
            goto fn_exit;
        }

        /* we don't need a local comm if this process is the only one on this node */
        if (num_local > 1) {
            mpi_errno = MPIR_Comm_create(&comm->node_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            comm->node_comm->context_id = comm->context_id + MPIR_CONTEXT_INTRANODE_OFFSET;
            comm->node_comm->recvcontext_id = comm->node_comm->context_id;
            comm->node_comm->rank = local_rank;
            comm->node_comm->comm_kind = MPIR_COMM_KIND__INTRACOMM;
            comm->node_comm->hierarchy_kind = MPIR_COMM_HIERARCHY_KIND__NODE;
            comm->node_comm->local_comm = NULL;
            MPL_DBG_MSG_D(MPIR_DBG_COMM, VERBOSE, "Create node_comm=%p\n", comm->node_comm);

            comm->node_comm->local_size = num_local;
            comm->node_comm->pof2 = MPL_pof2(comm->node_comm->local_size);
            comm->node_comm->remote_size = num_local;

            MPIR_Comm_map_irregular(comm->node_comm, comm, local_procs,
                                    num_local, MPIR_COMM_MAP_DIR__L2L, NULL);

            /* Notify device of communicator creation */
            mpi_errno = MPID_Comm_create_hook(comm->node_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            /* don't call MPIR_Comm_commit here */

            /* Create collectives-specific infrastructure */
            mpi_errno = MPIR_Coll_comm_init(comm->node_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            MPIR_Comm_map_free(comm->node_comm);
        }


        /* this process may not be a member of the node_roots_comm */
        if (local_rank == 0) {
            mpi_errno = MPIR_Comm_create(&comm->node_roots_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            comm->node_roots_comm->context_id = comm->context_id + MPIR_CONTEXT_INTERNODE_OFFSET;
            comm->node_roots_comm->recvcontext_id = comm->node_roots_comm->context_id;
            comm->node_roots_comm->rank = external_rank;
            comm->node_roots_comm->comm_kind = MPIR_COMM_KIND__INTRACOMM;
            comm->node_roots_comm->hierarchy_kind = MPIR_COMM_HIERARCHY_KIND__NODE_ROOTS;
            comm->node_roots_comm->local_comm = NULL;
            MPL_DBG_MSG_D(MPIR_DBG_COMM, VERBOSE, "Create node_roots_comm=%p\n",
                          comm->node_roots_comm);

            comm->node_roots_comm->local_size = num_external;
            comm->node_roots_comm->pof2 = MPL_pof2(comm->node_roots_comm->local_size);
            comm->node_roots_comm->remote_size = num_external;

            MPIR_Comm_map_irregular(comm->node_roots_comm, comm,
                                    external_procs, num_external, MPIR_COMM_MAP_DIR__L2L, NULL);

            /* Notify device of communicator creation */
            mpi_errno = MPID_Comm_create_hook(comm->node_roots_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
            /* don't call MPIR_Comm_commit here */

            /* Create collectives-specific infrastructure */
            mpi_errno = MPIR_Coll_comm_init(comm->node_roots_comm);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);

            MPIR_Comm_map_free(comm->node_roots_comm);
        }

        comm->hierarchy_kind = MPIR_COMM_HIERARCHY_KIND__PARENT;
    }

  fn_exit:
    if (comm == MPIR_Process.comm_world) {
        mpi_errno = MPID_Comm_create_hook(comm);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIR_Comm_map_free(comm);
    }

    if (external_procs != NULL)
        MPL_free(external_procs);
    if (local_procs != NULL)
        MPL_free(local_procs);

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_COMMIT);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Returns true if the given communicator is aware of node topology information,
   false otherwise.  Such information could be used to implement more efficient
   collective communication, for example. */
int MPIR_Comm_is_node_aware(MPIR_Comm * comm)
{
    return (comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__PARENT);
}

/* Returns true if the communicator is node-aware and processes in all the nodes
   are consecutive. For example, if node 0 contains "0, 1, 2, 3", node 1
   contains "4, 5, 6", and node 2 contains "7", we shall return true. */
int MPII_Comm_is_node_consecutive(MPIR_Comm * comm)
{
    int i = 0, curr_nodeidx = 0;
    int *internode_table = comm->internode_table;

    if (!MPIR_Comm_is_node_aware(comm))
        return 0;

    for (; i < comm->local_size; i++) {
        if (internode_table[i] == curr_nodeidx + 1)
            curr_nodeidx++;
        else if (internode_table[i] != curr_nodeidx)
            return 0;
    }

    return 1;
}

/*
 * Copy a communicator, including creating a new context and copying the
 * virtual connection tables and clearing the various fields.
 * Does *not* copy attributes.  If size is < the size of the local group
 * in the input communicator, copy only the first size elements.
 * If this process is not a member, return a null pointer in outcomm_ptr.
 * This is only supported in the case where the communicator is in
 * Intracomm (not an Intercomm).  Note that this is all that is required
 * for cart_create and graph_create.
 *
 * Used by cart_create, graph_create, and dup_create
 */
#undef FUNCNAME
#define FUNCNAME MPII_Comm_copy
#undef FCNAME
#define FCNAME "MPII_Comm_copy"
int MPII_Comm_copy(MPIR_Comm * comm_ptr, int size, MPIR_Comm ** outcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Context_id_t new_context_id, new_recvcontext_id;
    MPIR_Comm *newcomm_ptr = NULL;
    MPIR_Comm_map_t *map;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_COPY);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_COPY);

    /* Get a new context first.  We need this to be collective over the
     * input communicator */
    /* If there is a context id cache in oldcomm, use it here.  Otherwise,
     * use the appropriate algorithm to get a new context id.  Be careful
     * of intercomms here */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        mpi_errno = MPIR_Get_intercomm_contextid(comm_ptr, &new_context_id, &new_recvcontext_id);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else {
        mpi_errno = MPIR_Get_contextid_sparse(comm_ptr, &new_context_id, FALSE);
        new_recvcontext_id = new_context_id;
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_Assert(new_context_id != 0);
    }

    /* This is the local size, not the remote size, in the case of
     * an intercomm */
    if (comm_ptr->rank >= size) {
        *outcomm_ptr = 0;
        /* always free the recvcontext ID, never the "send" ID */
        MPIR_Free_contextid(new_recvcontext_id);
        goto fn_exit;
    }

    /* We're left with the processes that will have a non-null communicator.
     * Create the object, initialize the data, and return the result */

    mpi_errno = MPIR_Comm_create(&newcomm_ptr);
    if (mpi_errno)
        goto fn_fail;

    newcomm_ptr->context_id = new_context_id;
    newcomm_ptr->recvcontext_id = new_recvcontext_id;

    /* Save the kind of the communicator */
    newcomm_ptr->comm_kind = comm_ptr->comm_kind;
    newcomm_ptr->local_comm = 0;

    /* There are two cases here - size is the same as the old communicator,
     * or it is smaller.  If the size is the same, we can just add a reference.
     * Otherwise, we need to create a new network address mapping.  Note that this is the
     * test that matches the test on rank above. */
    if (size == comm_ptr->local_size) {
        /* Duplicate the network address mapping */
        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__L2L);
        else
            MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__R2R);
    } else {
        int i;

        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
            MPIR_Comm_map_irregular(newcomm_ptr, comm_ptr, NULL, size, MPIR_COMM_MAP_DIR__L2L,
                                    &map);
        else
            MPIR_Comm_map_irregular(newcomm_ptr, comm_ptr, NULL, size, MPIR_COMM_MAP_DIR__R2R,
                                    &map);
        for (i = 0; i < size; i++) {
            /* For rank i in the new communicator, find the corresponding
             * rank in the input communicator */
            map->src_mapping[i] = i;
        }
    }

    /* If it is an intercomm, duplicate the local network address references */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__L2L);
    }

    /* Set the sizes and ranks */
    newcomm_ptr->rank = comm_ptr->rank;
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        newcomm_ptr->local_size = comm_ptr->local_size;
        newcomm_ptr->remote_size = comm_ptr->remote_size;
        newcomm_ptr->is_low_group = comm_ptr->is_low_group;
    } else {
        newcomm_ptr->local_size = size;
        newcomm_ptr->remote_size = size;
    }
    newcomm_ptr->pof2 = MPL_pof2(newcomm_ptr->local_size);

    /* Inherit the error handler (if any) */
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));
    newcomm_ptr->errhandler = comm_ptr->errhandler;
    if (comm_ptr->errhandler) {
        MPIR_Errhandler_add_ref(comm_ptr->errhandler);
    }
    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));

    mpi_errno = MPIR_Comm_commit(newcomm_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Start with no attributes on this communicator */
    newcomm_ptr->attributes = 0;

    /* Copy over the info hints from the original communicator. */
    mpi_errno = MPIR_Info_dup_impl(comm_ptr->info, &(newcomm_ptr->info));
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPII_Comm_apply_hints(newcomm_ptr, newcomm_ptr->info);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    *outcomm_ptr = newcomm_ptr;

  fn_fail:
  fn_exit:

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_COPY);

    return mpi_errno;
}

/* Copy a communicator, including copying the virtual connection tables and
 * clearing the various fields.  Does *not* allocate a context ID or commit the
 * communicator.  Does *not* copy attributes.
 *
 * Used by comm_idup.
 */
#undef FUNCNAME
#define FUNCNAME MPII_Comm_copy_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPII_Comm_copy_data(MPIR_Comm * comm_ptr, MPIR_Comm ** outcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *newcomm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_COPY_DATA);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_COPY_DATA);

    mpi_errno = MPIR_Comm_create(&newcomm_ptr);
    if (mpi_errno)
        goto fn_fail;

    /* use a large garbage value to ensure errors are caught more easily */
    newcomm_ptr->context_id = 32767;
    newcomm_ptr->recvcontext_id = 32767;

    /* Save the kind of the communicator */
    newcomm_ptr->comm_kind = comm_ptr->comm_kind;
    newcomm_ptr->local_comm = 0;

    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM)
        MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__L2L);
    else
        MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__R2R);

    /* If it is an intercomm, duplicate the network address mapping */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        MPIR_Comm_map_dup(newcomm_ptr, comm_ptr, MPIR_COMM_MAP_DIR__L2L);
    }

    /* Set the sizes and ranks */
    newcomm_ptr->rank = comm_ptr->rank;
    newcomm_ptr->local_size = comm_ptr->local_size;
    newcomm_ptr->pof2 = comm_ptr->pof2;
    newcomm_ptr->remote_size = comm_ptr->remote_size;
    newcomm_ptr->is_low_group = comm_ptr->is_low_group; /* only relevant for intercomms */

    /* Inherit the error handler (if any) */
    MPID_THREAD_CS_ENTER(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));
    newcomm_ptr->errhandler = comm_ptr->errhandler;
    if (comm_ptr->errhandler) {
        MPIR_Errhandler_add_ref(comm_ptr->errhandler);
    }
    MPID_THREAD_CS_EXIT(POBJ, MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr));

    /* Start with no attributes on this communicator */
    newcomm_ptr->attributes = 0;
    *outcomm_ptr = newcomm_ptr;

  fn_fail:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_COPY_DATA);
    return mpi_errno;
}

/* Common body between MPIR_Comm_release and MPIR_comm_release_always.  This
 * helper function frees the actual MPIR_Comm structure and any associated
 * storage.  It also releases any references to other objects.
 * This function should only be called when the communicator's reference count
 * has dropped to 0.
 *
 * !!! This routine should *never* be called outside of MPIR_Comm_release{,_always} !!!
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Comm_delete_internal
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_delete_internal(MPIR_Comm * comm_ptr)
{
    int in_use;
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_COMM_DELETE_INTERNAL);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_COMM_DELETE_INTERNAL);

    MPIR_Assert(MPIR_Object_get_ref(comm_ptr) == 0);    /* sanity check */

    /* Remove the attributes, executing the attribute delete routine.
     * Do this only if the attribute functions are defined.
     * This must be done first, because if freeing the attributes
     * returns an error, the communicator is not freed */
    if (MPIR_Process.attr_free && comm_ptr->attributes) {
        /* Temporarily add a reference to this communicator because
         * the attr_free code requires a valid communicator */
        MPIR_Object_add_ref(comm_ptr);
        mpi_errno = MPIR_Process.attr_free(comm_ptr->handle, &comm_ptr->attributes);
        /* Release the temporary reference added before the call to
         * attr_free */
        MPIR_Object_release_ref(comm_ptr, &in_use);
    }

    /* If the attribute delete functions return failure, the
     * communicator must not be freed.  That is the reason for the
     * test on mpi_errno here. */
    if (mpi_errno == MPI_SUCCESS) {
        /* If this communicator is our parent, and we're disconnecting
         * from the parent, mark that fact */
        if (MPIR_Process.comm_parent == comm_ptr)
            MPIR_Process.comm_parent = NULL;

        /* Cleanup collectives-specific infrastructure */
        mpi_errno = MPII_Coll_comm_cleanup(comm_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Notify the device that the communicator is about to be
         * destroyed */
        mpi_errno = MPID_Comm_free_hook(comm_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Free info hints */
        if (comm_ptr->info != NULL) {
            MPIR_Info_free(comm_ptr->info);
        }

        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM && comm_ptr->local_comm)
            MPIR_Comm_release(comm_ptr->local_comm);

        /* Free the local and remote groups, if they exist */
        if (comm_ptr->local_group)
            MPIR_Group_release(comm_ptr->local_group);
        if (comm_ptr->remote_group)
            MPIR_Group_release(comm_ptr->remote_group);

        /* free the intra/inter-node communicators, if they exist */
        if (comm_ptr->node_comm)
            MPIR_Comm_release(comm_ptr->node_comm);
        if (comm_ptr->node_roots_comm)
            MPIR_Comm_release(comm_ptr->node_roots_comm);
        if (comm_ptr->intranode_table != NULL)
            MPL_free(comm_ptr->intranode_table);
        if (comm_ptr->internode_table != NULL)
            MPL_free(comm_ptr->internode_table);

        /* Free the context value.  This should come after freeing the
         * intra/inter-node communicators since those free calls won't
         * release this context ID and releasing this before then could lead
         * to races once we make threading finer grained. */
        /* This must be the recvcontext_id (i.e. not the (send)context_id)
         * because in the case of intercommunicators the send context ID is
         * allocated out of the remote group's bit vector, not ours. */
        MPIR_Free_contextid(comm_ptr->recvcontext_id);

#if MPICH_THREAD_GRANULARITY == MPICH_THREAD_GRANULARITY__POBJ
        {
            int thr_err;
            MPID_Thread_mutex_destroy(&MPIR_THREAD_POBJ_COMM_MUTEX(comm_ptr), &thr_err);
            MPIR_Assert(thr_err == 0);
        }
#endif
        /* We need to release the error handler */
        if (comm_ptr->errhandler &&
            !(HANDLE_GET_KIND(comm_ptr->errhandler->handle) == HANDLE_KIND_BUILTIN)) {
            int errhInuse;
            MPIR_Errhandler_release_ref(comm_ptr->errhandler, &errhInuse);
            if (!errhInuse) {
                MPIR_Handle_obj_free(&MPIR_Errhandler_mem, comm_ptr->errhandler);
            }
        }

        /* Remove from the list of active communicators if
         * we are supporting message-queue debugging.  We make this
         * conditional on having debugger support since the
         * operation is not constant-time */
        MPII_COMML_FORGET(comm_ptr);

        /* Check for predefined communicators - these should not
         * be freed */
        if (!(HANDLE_GET_KIND(comm_ptr->handle) == HANDLE_KIND_BUILTIN))
            MPIR_Handle_obj_free(&MPIR_Comm_mem, comm_ptr);
    } else {
        /* If the user attribute free function returns an error,
         * then do not free the communicator */
        MPIR_Comm_add_ref(comm_ptr);
    }

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_COMM_DELETE_INTERNAL);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Release a reference to a communicator.  If there are no pending
   references, delete the communicator and recover all storage and
   context ids.  This version of the function always manipulates the reference
   counts, even for predefined objects. */
#undef FUNCNAME
#define FUNCNAME MPIR_Comm_release_always
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_release_always(MPIR_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int in_use;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_RELEASE_ALWAYS);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_RELEASE_ALWAYS);

    /* we want to short-circuit any optimization that avoids reference counting
     * predefined communicators, such as MPI_COMM_WORLD or MPI_COMM_SELF. */
    MPIR_Object_release_ref_always(comm_ptr, &in_use);
    if (!in_use) {
        mpi_errno = MPIR_Comm_delete_internal(comm_ptr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_RELEASE_ALWAYS);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Apply all known info hints in the specified info chain to the given
 * communicator. */
#undef FUNCNAME
#define FUNCNAME MPII_Comm_apply_hints
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPII_Comm_apply_hints(MPIR_Comm * comm_ptr, MPIR_Info * info_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Info *hint = NULL;
    char hint_name[MPI_MAX_INFO_KEY] = { 0 };
    struct MPIR_Comm_hint_fn_elt *hint_fn = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_APPLY_HINTS);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_APPLY_HINTS);

    LL_FOREACH(info_ptr, hint) {
        /* Have we hit the default, empty info hint? */
        if (hint->key == NULL)
            continue;

        MPL_strncpy(hint_name, hint->key, MPI_MAX_INFO_KEY);

        HASH_FIND_STR(MPID_hint_fns, hint_name, hint_fn);

        /* Skip hints that MPICH doesn't recognize. */
        if (hint_fn) {
            mpi_errno = hint_fn->fn(comm_ptr, hint, hint_fn->state);
            if (mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }
    }

  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_APPLY_HINTS);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME free_hint_handles
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int free_hint_handles(void *ignore)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPIR_Comm_hint_fn_elt *curr_hint = NULL, *tmp = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_FREE_HINT_HANDLES);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_FREE_HINT_HANDLES);

    if (MPID_hint_fns) {
        HASH_ITER(hh, MPID_hint_fns, curr_hint, tmp) {
            HASH_DEL(MPID_hint_fns, curr_hint);
            MPL_free(curr_hint);
        }
    }

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_FREE_HINT_HANDLES);
    return mpi_errno;
}

/* The hint logic is stored in a uthash, with hint name as key and
 * the function responsible for applying the hint as the value. */
#undef FUNCNAME
#define FUNCNAME MPIR_Comm_register_hint
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Comm_register_hint(const char *hint_key, MPIR_Comm_hint_fn_t fn, void *state)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPIR_Comm_hint_fn_elt *hint_elt = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPIR_COMM_REGISTER_HINT);

    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPIR_COMM_REGISTER_HINT);

    if (MPID_hint_fns == NULL) {
        MPIR_Add_finalize(free_hint_handles, NULL, MPIR_FINALIZE_CALLBACK_PRIO - 1);
    }

    hint_elt = MPL_malloc(sizeof(struct MPIR_Comm_hint_fn_elt), MPL_MEM_COMM);
    MPL_strncpy(hint_elt->name, hint_key, MPI_MAX_INFO_KEY);
    hint_elt->state = state;
    hint_elt->fn = fn;

    HASH_ADD_STR(MPID_hint_fns, name, hint_elt, MPL_MEM_COMM);

    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPIR_COMM_REGISTER_HINT);
    return mpi_errno;
}
