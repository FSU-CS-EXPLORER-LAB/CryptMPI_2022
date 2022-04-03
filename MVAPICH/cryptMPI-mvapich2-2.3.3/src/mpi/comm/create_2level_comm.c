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

#include "mpiimpl.h"
#include <mpimem.h>
#include "mpidimpl.h"
#include "mpicomm.h"
#include "coll_shmem.h"
#include <pthread.h>
#include <unistd.h>
#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#endif

#if defined (_SHARP_SUPPORT_)
#include "ibv_sharp.h"
extern int mv2_sharp_tuned_msg_size;
#endif

MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_2level_comm_requests);
MPIR_T_PVAR_ULONG2_COUNTER_DECL_EXTERN(MV2, mv2_num_2level_comm_success);

#if defined(_SMP_LIMIC_)    
int mv2_limic_comm_count = 0; 
#endif

static pthread_mutex_t comm_lock  = PTHREAD_MUTEX_INITIALIZER;
extern int mv2_g_shmem_coll_blocks;

#define MAX_NUM_THREADS 1024
pthread_t thread_reg[MAX_NUM_THREADS];

void clear_2level_comm (MPID_Comm* comm_ptr)
{
    comm_ptr->dev.ch.allgather_comm_ok = 0;
    comm_ptr->dev.ch.shmem_coll_ok = 0;
    comm_ptr->dev.ch.leader_map  = NULL;
    comm_ptr->dev.ch.leader_rank = NULL;
    comm_ptr->dev.ch.node_disps  = NULL;
    comm_ptr->dev.ch.rank_list   = NULL;
    comm_ptr->dev.ch.rank_list_index = -1;
    comm_ptr->dev.ch.shmem_comm = MPI_COMM_NULL; 
    comm_ptr->dev.ch.leader_comm = MPI_COMM_NULL;
    comm_ptr->dev.ch.allgather_comm = MPI_COMM_NULL;
    comm_ptr->dev.ch.intra_node_done = 0;
#if defined(_MCST_SUPPORT_)
    comm_ptr->dev.ch.is_mcast_ok = 0;
#endif

#if defined(_SHARP_SUPPORT_)
    comm_ptr->dev.ch.is_sharp_ok = 0;
#endif
}

int free_limic_comm (MPID_Comm* shmem_comm_ptr)
{
    int intra_comm_rank = -1;
    int mpi_errno=MPI_SUCCESS;

    MPID_Comm *intra_sock_comm_ptr=NULL; 
    MPID_Comm *intra_sock_leader_comm_ptr=NULL; 

    MPID_Comm_get_ptr(shmem_comm_ptr->dev.ch.intra_sock_comm, intra_sock_comm_ptr);
    MPID_Comm_get_ptr(shmem_comm_ptr->dev.ch.intra_sock_leader_comm, intra_sock_leader_comm_ptr);

    if (intra_sock_comm_ptr != NULL) { 
        PMPI_Comm_rank(shmem_comm_ptr->dev.ch.intra_sock_comm, &intra_comm_rank);
        if(intra_comm_rank == 0) {
            if(shmem_comm_ptr->dev.ch.socket_size !=NULL) {
                MPIU_Free(shmem_comm_ptr->dev.ch.socket_size);
            }
        }
    }
    if (intra_sock_comm_ptr != NULL) { 
        mpi_errno = MPIR_Comm_release(intra_sock_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
    if (intra_sock_leader_comm_ptr != NULL)  { 

        //If shmem coll ok is set to 1 for a socket leader communicator of size 1 (for correctness), set it back to 0
        //to prevent segmentation faults while releasing the communicator
        if(intra_sock_leader_comm_ptr->local_size == 1 && intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok == 1)
        {
            intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok = 0;
        }
        mpi_errno = MPIR_Comm_release(intra_sock_leader_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
    shmem_comm_ptr->dev.ch.socket_size = NULL;
    shmem_comm_ptr->dev.ch.is_socket_uniform = 0;
    shmem_comm_ptr->dev.ch.use_intra_sock_comm= 0;
    shmem_comm_ptr->dev.ch.intra_sock_comm = MPI_COMM_NULL;
    shmem_comm_ptr->dev.ch.intra_sock_leader_comm = MPI_COMM_NULL;        
    fn_exit:
       return mpi_errno;
    fn_fail:
       goto fn_exit;
}

int free_intra_sock_comm (MPID_Comm* comm_ptr)
{
    int mpi_errno;
    MPID_Comm *shmem_comm_ptr = NULL;
    MPID_Comm *global_sock_leader_comm_ptr = NULL;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_comm_ptr );
    MPID_Comm_get_ptr(comm_ptr->dev.ch.global_sock_leader_comm, global_sock_leader_comm_ptr);

    if (global_sock_leader_comm_ptr != NULL)  {
        mpi_errno = MPIR_Comm_release(global_sock_leader_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) {
            goto fn_fail;
        }
    } 

    /*Reuse limic comm free since some variables are common between these comms */
    mpi_errno = free_limic_comm(shmem_comm_ptr);

    fn_exit : return mpi_errno;
    fn_fail : goto fn_exit;
}

int free_2level_comm (MPID_Comm* comm_ptr)
{
    MPID_Comm *shmem_comm_ptr=NULL; 
    MPID_Comm *leader_comm_ptr=NULL;
    MPID_Comm *allgather_comm_ptr=NULL;

    int local_rank=0;
    int mpi_errno=MPI_SUCCESS;

    if (comm_ptr->dev.ch.leader_map != NULL)  { 
        MPIU_Free(comm_ptr->dev.ch.leader_map);  
    }
    if (comm_ptr->dev.ch.leader_rank != NULL) { 
        MPIU_Free(comm_ptr->dev.ch.leader_rank); 
    }
    if (comm_ptr->dev.ch.rank_list != NULL) {
        MPIU_Free(comm_ptr->dev.ch.rank_list);
    }
 
    MPID_Comm_get_ptr((comm_ptr->dev.ch.shmem_comm), shmem_comm_ptr );
    MPID_Comm_get_ptr((comm_ptr->dev.ch.leader_comm), leader_comm_ptr );
    
    if(comm_ptr->dev.ch.allgather_comm_ok == 1)  { 
       MPID_Comm_get_ptr((comm_ptr->dev.ch.allgather_comm), allgather_comm_ptr );
       MPIU_Free(comm_ptr->dev.ch.allgather_new_ranks); 
    }

    local_rank = shmem_comm_ptr->rank;

#if defined (_SHARP_SUPPORT_)
    if (mv2_enable_sharp_coll) {
        mpi_errno = mv2_free_sharp_handlers(comm_ptr->dev.ch.sharp_coll_info); 
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
#endif /* if defined (_SHARP_SUPPORT_) */
    
    if(local_rank == 0 && shmem_comm_ptr->dev.ch.shmem_comm_rank >= 0) { 
        lock_shmem_region();
        MPIDI_CH3I_SHMEM_Coll_Block_Clear_Status(shmem_comm_ptr->dev.ch.shmem_comm_rank); 
        unlock_shmem_region();
    }

#if defined(_SMP_LIMIC_)    
    if(shmem_comm_ptr->dev.ch.use_intra_sock_comm == 1) { 
        free_limic_comm(shmem_comm_ptr);
    } 
#endif /* #if defined(_SMP_LIMIC_) */ 

    if(comm_ptr->dev.ch.use_intra_sock_comm == 1) {
        free_intra_sock_comm(comm_ptr);
    }

#if defined(_MCST_SUPPORT_)
    if (local_rank == 0 && comm_ptr->dev.ch.is_mcast_ok) {
        mv2_cleanup_multicast(&((bcast_info_t *) comm_ptr->dev.ch.bcast_info)->minfo, comm_ptr);
    }
    MPIU_Free(comm_ptr->dev.ch.bcast_info);
#endif

    if (comm_ptr->dev.ch.shmem_info) {
        mv2_shm_coll_cleanup((shmem_info_t *)comm_ptr->dev.ch.shmem_info);
        MPIU_Free(comm_ptr->dev.ch.shmem_info);
    }

    if(local_rank == 0) { 
        if(comm_ptr->dev.ch.node_sizes != NULL) { 
            MPIU_Free(comm_ptr->dev.ch.node_sizes); 
        } 
    } 
    if(comm_ptr->dev.ch.node_disps != NULL) {
        MPIU_Free(comm_ptr->dev.ch.node_disps);
    }
    if (local_rank == 0 && leader_comm_ptr != NULL) { 
        mpi_errno = MPIR_Comm_release(leader_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
    if (shmem_comm_ptr != NULL)  { 
        /* Decrease the reference number of shmem_group, which 
         * was increased in create_2level_comm->PMPI_Group_incl */
        if (shmem_comm_ptr->local_group != NULL) {
            MPIR_Group_release(shmem_comm_ptr->local_group);
        }
        mpi_errno = MPIR_Comm_release(shmem_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
    if (allgather_comm_ptr != NULL)  { 
        mpi_errno = MPIR_Comm_release(allgather_comm_ptr);
        if (mpi_errno != MPI_SUCCESS) { 
            goto fn_fail;
        } 
    }
    clear_2level_comm(comm_ptr);
    fn_exit:
       return mpi_errno;
    fn_fail:
       goto fn_exit;
}

inline void MPIR_pof2_comm(MPID_Comm * comm_ptr, int size, int my_rank)
{
    int v = 1, old_v = 1;

    //*  Check if comm is a pof2 or not */
    comm_ptr->dev.ch.is_pof2 = (size & (size - 1)) ? 0 : 1;

    /* retrieve the greatest power of two < size of comm */
    if (comm_ptr->dev.ch.is_pof2) {
        comm_ptr->dev.ch.gpof2 = size;
    } else {
        while (v < size) {
            old_v = v;
            v = v << 1;
        }
        comm_ptr->dev.ch.gpof2 = old_v;
    }
}

#if defined(_SMP_LIMIC_)
int create_intra_node_multi_level_comm(MPID_Comm* comm_ptr) {

    static const char FCNAME[] = "create_intra_node_multi_level_comm";
    int socket_bound=-1;
    int numCoresSocket = 0;
    int numSocketsNode = 0;
    int* intra_socket_leader_map=NULL;
    int* intra_sock_leader_group=NULL;
    int intra_comm_rank=0, intra_comm_size=0;
    int intra_leader_comm_size=0, intra_leader_comm_rank=0;
    int ok_to_create_intra_sock_comm=0, i=0;
    int input_flag = 0, output_flag = 0;
    int my_local_size, my_local_id;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    MPI_Group subgroup2;
    MPID_Group *group_ptr=NULL;
    MPID_Comm *shmem_ptr=NULL;
    MPID_Comm *intra_sock_leader_commptr=NULL; 
    MPID_Comm *intra_sock_commptr=NULL; 
    
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr);
    mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.shmem_comm, &my_local_size);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    if(g_use_limic2_coll) {
        MPI_Group comm_group1; 
        lock_shmem_region();
        mv2_limic_comm_count = get_mv2_limic_comm_count();
        unlock_shmem_region();

        if (mv2_limic_comm_count <= mv2_max_limic_comms){
            socket_bound = get_socket_bound(); 
            numSocketsNode = numofSocketsPerNode();
            numCoresSocket = numOfCoresPerSocket(socket_bound);
            int *intra_socket_map = MPIU_Malloc(sizeof(int)*my_local_size);
            if (NULL == intra_socket_map){
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);

            }

            memset(intra_socket_map, -1, sizeof(int)*my_local_size);

            mpi_errno = MPIR_Allgather_impl(&socket_bound, 1, MPI_INT,
                    intra_socket_map, 1, MPI_INT, shmem_ptr, &errflag);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /*Check if all the proceses are not in same socket.
             * We create socket communicators only when 2 or 
             * more sockets are present*/ 
            for(i=1;i<my_local_size;i++) {
                if(intra_socket_map[0] != intra_socket_map[i]) {
                    ok_to_create_intra_sock_comm=1;
                    break;
                }
            }
            MPIU_Free(intra_socket_map);
        }

        if(ok_to_create_intra_sock_comm != 0) {
            /*Create communicator for intra socket communication*/
            mpi_errno = PMPI_Comm_split(comm_ptr->dev.ch.shmem_comm, socket_bound, 
                    my_local_id, &(shmem_ptr->dev.ch.intra_sock_comm));
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            int intra_socket_leader_id=-1;
            int intra_socket_leader_cnt=0;
            PMPI_Comm_rank(shmem_ptr->dev.ch.intra_sock_comm, &intra_comm_rank);
            PMPI_Comm_size(shmem_ptr->dev.ch.intra_sock_comm, &intra_comm_size);

            if(intra_comm_rank == 0)
            {
                intra_socket_leader_id=1;
            }

            /*Creating intra-socket leader group*/
            intra_socket_leader_map = MPIU_Malloc(sizeof(int)*my_local_size);
            if (NULL == intra_socket_leader_map){
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);

            }
            /*initialize the intra_socket_leader_map*/
            for(i=0;i<my_local_size;i++)     //TODO: Replace with memset
                intra_socket_leader_map[i] = -1;

            mpi_errno = MPIR_Allgather_impl(&intra_socket_leader_id, 1, MPI_INT,
                    intra_socket_leader_map, 1, MPI_INT, shmem_ptr, &errflag);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            for(i=0;i<my_local_size;i++) {
                if(intra_socket_leader_map[i] == 1)
                    intra_socket_leader_cnt++;
            }

            intra_sock_leader_group = MPIU_Malloc(sizeof(int) *  
                    intra_socket_leader_cnt);
            if (NULL == intra_sock_leader_group){
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }

            /*Assuming homogeneous system, where every socket has same number
             * of cores */
            int j=0;
            for(i=0;i<my_local_size;i++) { 
                if(intra_socket_leader_map[i] == 1)
                    /*i here actually is the my_local_id for which
                    intra_sock_rank == 0*/
                    intra_sock_leader_group[j++] = i; 
                    }

            /*Resuing comm_group and subgroup1 variables for creation of 
             * intra socket leader comm*/
            mpi_errno = PMPI_Comm_group(comm_ptr->dev.ch.shmem_comm, &comm_group1);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            mpi_errno = PMPI_Group_incl(comm_group1, intra_socket_leader_cnt, 
                    intra_sock_leader_group, &subgroup2);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /*Creating intra_sock_leader communicator*/
            mpi_errno = PMPI_Comm_create(comm_ptr->dev.ch.shmem_comm, subgroup2, 
                    &(shmem_ptr->dev.ch.intra_sock_leader_comm));
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            if(intra_comm_rank == 0 ) {
                mpi_errno = PMPI_Comm_rank(shmem_ptr->dev.ch.intra_sock_leader_comm, 
                        &intra_leader_comm_rank);
                if(mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                mpi_errno = PMPI_Comm_size(shmem_ptr->dev.ch.intra_sock_leader_comm, 
                        &intra_leader_comm_size);
                if(mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }

            /*Check if all the data in sockets are of uniform size*/
            if(intra_comm_rank == 0) { 
                int array_index=0;

                shmem_ptr->dev.ch.socket_size = MPIU_Malloc(sizeof(int)*
                        intra_leader_comm_size);
                mpi_errno = PMPI_Allgather(&intra_comm_size, 1, MPI_INT,
                        shmem_ptr->dev.ch.socket_size, 1, MPI_INT, 
                        shmem_ptr->dev.ch.intra_sock_leader_comm);
                if(mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                shmem_ptr->dev.ch.is_socket_uniform = 1; 
                for(array_index=0; array_index < intra_leader_comm_size; 
                        array_index++) { 
                    if(shmem_ptr->dev.ch.socket_size[0] != 
                            shmem_ptr->dev.ch.socket_size[array_index]) {
                        shmem_ptr->dev.ch.is_socket_uniform = 0; 
                        break;
                    }
                }
            }

            MPID_Group_get_ptr( subgroup2, group_ptr );
            if(group_ptr != NULL) { 
                mpi_errno = PMPI_Group_free(&subgroup2);
                if(mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
            mpi_errno=PMPI_Group_free(&comm_group1);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }

        if ((mv2_limic_comm_count <= mv2_max_limic_comms)
                && (ok_to_create_intra_sock_comm != 0)
                && my_local_id == 0 ) { 
            /*update num of sockets within a node*/
            lock_shmem_region();

            UpdateNumCoresPerSock(numCoresSocket);
            UpdateNumSocketsPerNode(numSocketsNode); 
            /*only 1 intra sock leader comm*/
            increment_mv2_limic_comm_count();
            /*many per node intra sock comm*/
            for(i=0;i<intra_leader_comm_size;i++)
                increment_mv2_limic_comm_count();

            mv2_limic_comm_count = get_mv2_limic_comm_count();
            unlock_shmem_region();

        }

        mpi_errno = MPIR_Bcast_impl (&mv2_limic_comm_count, 1, MPI_INT, 
                0, shmem_ptr, &errflag);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        if ((mv2_limic_comm_count <= mv2_max_limic_comms) &&
                (g_use_limic2_coll) && (ok_to_create_intra_sock_comm != 0)){
            MPID_Comm_get_ptr(shmem_ptr->dev.ch.intra_sock_leader_comm, 
                    intra_sock_leader_commptr );
            MPID_Comm_get_ptr(shmem_ptr->dev.ch.intra_sock_comm, 
                    intra_sock_commptr );
            if(intra_comm_rank == 0) {
                intra_sock_leader_commptr->dev.ch.shmem_comm_rank = 
                    mv2_limic_comm_count-1; 
            } 
            intra_sock_commptr->dev.ch.shmem_comm_rank = 
                mv2_limic_comm_count-(socket_bound + 2); 
            input_flag = 1;
        } else{
            input_flag = 0;
        }

        mpi_errno = MPIR_Allreduce_impl(&input_flag, &output_flag, 1, 
                MPI_INT, MPI_LAND, comm_ptr, &errflag);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        if (output_flag == 1){
            /*Enable using the intra-sock communicators*/
            shmem_ptr->dev.ch.use_intra_sock_comm=1;
        } else{
            /*Disable using the intra-sock communicators*/
            shmem_ptr->dev.ch.use_intra_sock_comm=0;

            if((g_use_limic2_coll) && (ok_to_create_intra_sock_comm != 0)) {
                MPID_Group_get_ptr( subgroup2, group_ptr );
                if(group_ptr != NULL) { 
                    mpi_errno = PMPI_Group_free(&subgroup2);
                    if(mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
                MPID_Group_get_ptr( comm_group1, group_ptr );
                if(group_ptr != NULL) { 
                    mpi_errno = PMPI_Group_free(&comm_group1);
                    if(mpi_errno) {
                        MPIR_ERR_POP(mpi_errno);
                    }
                }
                free_limic_comm(shmem_ptr);
            }
        } 
    }
    fn_fail:
    if(intra_socket_leader_map != NULL)
        MPIU_Free(intra_socket_leader_map);
    if(intra_sock_leader_group != NULL)        
        MPIU_Free(intra_sock_leader_group);
    return (mpi_errno);
}
#endif

int create_intra_sock_comm(MPI_Comm comm)
{    
    static const char FCNAME[] = "create_intra_sock_comm";
    int socket_bound = -1;
    int numCoresSocket = 0;
    int numSocketsNode = 0;
    int* intra_socket_leader_map=NULL;
    int* intra_sock_leader_group=NULL;
    int intra_comm_rank=0, intra_comm_size=0;
    int intra_leader_comm_size=0, intra_leader_comm_rank=0;
    int ok_to_create_intra_sock_comm=1, i=0;
    int output_flag = 0;
    int my_local_size, my_local_id;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int global_size               = 0;
    int global_socket_leader_cnt  = 0;
    int* global_sock_leader_group  = NULL;
    int* global_socket_leader_map  = NULL;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(comm, comm_ptr);
    MPI_Group subgroup2;
    MPID_Group *group_ptr=NULL;
    MPID_Comm *shmem_ptr=NULL;
    MPID_Comm *intra_sock_commptr=NULL;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr);
    mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    comm_ptr->dev.ch.use_intra_sock_comm = 0;
    mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.shmem_comm, &my_local_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (mv2_enable_socket_aware_collectives) {
        MPI_Group comm_group1;
        int is_uniform;
        int err = get_socket_bound_info(&socket_bound, &numSocketsNode, &numCoresSocket, &is_uniform);
        int output_err = 0;
        mpi_errno = MPIR_Allreduce_impl(&err, &output_err, 1,
                                            MPI_INT, MPI_LOR, comm_ptr,
                                            &errflag);
        if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
        }

        if (output_err != 0) {

            PRINT_INFO(comm_ptr->rank == 0, "Failed to get correct process to socket binding info."
                                            "Proceeding by disabling socket aware collectives support.");
            return MPI_SUCCESS;
        }

        comm_ptr->dev.ch.my_sock_id = socket_bound;
        int intra_socket_leader_id=-1;
        int intra_socket_leader_cnt=0;
        if (ok_to_create_intra_sock_comm) {
            /*Create communicator for intra socket communication*/
            mpi_errno = PMPI_Comm_split(comm_ptr->dev.ch.shmem_comm, socket_bound,
                    my_local_id, &(shmem_ptr->dev.ch.intra_sock_comm));
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            PMPI_Comm_rank(shmem_ptr->dev.ch.intra_sock_comm, &intra_comm_rank);
            PMPI_Comm_size(shmem_ptr->dev.ch.intra_sock_comm, &intra_comm_size);
            /* allocate a shared region for each socket */
            int mv2_shmem_coll_blk_stat = -1;
            int input_flag = 0;
            if (intra_comm_rank == 0){
                lock_shmem_region();
                mv2_shmem_coll_blk_stat = MPIDI_CH3I_SHMEM_Coll_get_free_block();
                if (mv2_shmem_coll_blk_stat >= 0) {
                    input_flag = 1;
                }
                unlock_shmem_region();
            } else {
                input_flag = 1;
            }

            mpi_errno = MPIR_Allreduce_impl(&input_flag, &output_flag, 1,
                                            MPI_INT, MPI_LAND, comm_ptr,
                                            &errflag);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            if (!output_flag) {
                /* None of the shmem-coll-blocks are available. We cannot support
                   shared-memory collectives for this communicator */
                PRINT_DEBUG(DEBUG_SHM_verbose > 1,"Not enough shared memory regions."
                                                  " Cannot support socket aware collectives\n");
                if (intra_comm_rank == 0 && mv2_shmem_coll_blk_stat >= 0) {
                    /*release the slot if it is aquired */
                    lock_shmem_region();
                    MPIDI_CH3I_SHMEM_Coll_Block_Clear_Status(mv2_shmem_coll_blk_stat);
                    unlock_shmem_region();
                }
                goto fn_fail;
            }

            MPID_Comm_get_ptr(shmem_ptr->dev.ch.intra_sock_comm,
                    intra_sock_commptr );

            mpi_errno = MPIR_Bcast_impl (&mv2_shmem_coll_blk_stat, 1,
                    MPI_INT, 0, intra_sock_commptr, &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            intra_sock_commptr->dev.ch.shmem_comm_rank = mv2_shmem_coll_blk_stat;
            if (intra_comm_rank == 0) {
                intra_socket_leader_id=1;
            }

            /*Creating intra-socket leader group*/
            intra_socket_leader_map = MPIU_Malloc(sizeof(int)*my_local_size);
            if (NULL == intra_socket_leader_map){
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }

            /*initialize the intra_socket_leader_map*/
            memset(intra_socket_leader_map, -1, global_size);

            mpi_errno = MPIR_Allgather_impl(&intra_socket_leader_id, 1, MPI_INT,
                    intra_socket_leader_map, 1, MPI_INT, shmem_ptr, &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            for (i=0; i<my_local_size; i++) {
                if (intra_socket_leader_map[i] == 1)
                    intra_socket_leader_cnt++;
            }

            intra_sock_leader_group = MPIU_Malloc(sizeof(int) *
                    intra_socket_leader_cnt);
            if (NULL == intra_sock_leader_group) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }

            /*Assuming homogeneous system, where every socket has same number
              of cores */
            int j=0;
            for (i=0;i<my_local_size;i++) {
                if (intra_socket_leader_map[i] == 1) { 
                    /*i here actually is the my_local_id for which
                       intra_sock_rank == 0*/
                    intra_sock_leader_group[j++] = i;
                }
            }

            /*Resuing comm_group and subgroup1 variables for creation of 
              intra socket leader comm*/
            mpi_errno = PMPI_Comm_group(comm_ptr->dev.ch.shmem_comm, &comm_group1);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            mpi_errno = PMPI_Group_incl(comm_group1, intra_socket_leader_cnt,
                    intra_sock_leader_group, &subgroup2);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /*Creating intra_sock_leader communicator*/
            mpi_errno = PMPI_Comm_create(comm_ptr->dev.ch.shmem_comm, subgroup2,
                    &(shmem_ptr->dev.ch.intra_sock_leader_comm));
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            if (intra_comm_rank == 0) {
                mpi_errno = PMPI_Comm_rank(shmem_ptr->dev.ch.intra_sock_leader_comm,
                        &intra_leader_comm_rank);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                mpi_errno = PMPI_Comm_size(shmem_ptr->dev.ch.intra_sock_leader_comm,
                        &intra_leader_comm_size);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }

            /*Check if all the data in sockets are of uniform size*/
            if (intra_comm_rank == 0) {
                int array_index=0;

                shmem_ptr->dev.ch.socket_size = MPIU_Malloc(sizeof(int)*
                        intra_leader_comm_size);
                mpi_errno = PMPI_Allgather(&intra_comm_size, 1, MPI_INT,
                        shmem_ptr->dev.ch.socket_size, 1, MPI_INT,
                        shmem_ptr->dev.ch.intra_sock_leader_comm);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                shmem_ptr->dev.ch.is_socket_uniform = 1;
                for (array_index=0; array_index < intra_leader_comm_size;
                        array_index++) {
                    if (shmem_ptr->dev.ch.socket_size[0] !=
                            shmem_ptr->dev.ch.socket_size[array_index]) {
                        shmem_ptr->dev.ch.is_socket_uniform = 0;
                        break;
                    }
                }
            }

            MPID_Group_get_ptr( subgroup2, group_ptr );
            if (group_ptr != NULL) {
                mpi_errno = PMPI_Group_free(&subgroup2);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
            mpi_errno=PMPI_Group_free(&comm_group1);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /* create comm between all the sock leaders across all nodes */
            mpi_errno = PMPI_Comm_size(comm, &global_size);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            global_socket_leader_map = MPIU_Malloc(sizeof(int)*global_size);
            if (NULL == global_socket_leader_map) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }
            /* initialize the global_socket_leader_map */
            memset(global_socket_leader_map, -1, global_size);
            mpi_errno = MPIR_Allgather_impl(&intra_socket_leader_id, 1, MPI_INT,
                    global_socket_leader_map, 1, MPI_INT, comm_ptr, &errflag);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }


            for (i=0;i<global_size;i++) {
                if (global_socket_leader_map[i] == 1)
                    global_socket_leader_cnt++;
            }

            global_sock_leader_group = MPIU_Malloc(sizeof(int) *
                    global_socket_leader_cnt);
            if (NULL == global_sock_leader_group){
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                        FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                        "memory allocation failed", strerror(errno));
                MPIR_ERR_POP(mpi_errno);
            }

            /* Create the list of global sock leaders ranks */
            j=0;
            for (i=0;i<global_size;i++) {
                if (global_socket_leader_map[i] == 1)
                    global_sock_leader_group[j++] = i;
            }

            mpi_errno = PMPI_Comm_group(comm, &comm_group1);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            mpi_errno = PMPI_Group_incl(comm_group1, global_socket_leader_cnt,
                                        global_sock_leader_group, &subgroup2);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            /*Creating global_sock_leader communicator*/
            mpi_errno = PMPI_Comm_create(comm, subgroup2,
                                        &(comm_ptr->dev.ch.global_sock_leader_comm));
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            MPID_Group_get_ptr( subgroup2, group_ptr );
            if (group_ptr != NULL) {
                mpi_errno = PMPI_Group_free(&subgroup2);
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
            mpi_errno=PMPI_Group_free(&comm_group1);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }

            comm_ptr->dev.ch.use_intra_sock_comm=1;

            MPID_Comm *global_sock_leader_ptr;
            MPID_Comm *intra_sock_comm_ptr=NULL;
            MPID_Comm *intra_sock_leader_comm_ptr=NULL;

            MPID_Comm_get_ptr(shmem_ptr->dev.ch.intra_sock_comm, 
                              intra_sock_comm_ptr);
            MPID_Comm_get_ptr(shmem_ptr->dev.ch.intra_sock_leader_comm, 
                              intra_sock_leader_comm_ptr);
            MPID_Comm_get_ptr(comm_ptr->dev.ch.global_sock_leader_comm, 
                              global_sock_leader_ptr);
#if defined (_SHARP_SUPPORT_)
            if (global_sock_leader_ptr != NULL) {
                global_sock_leader_ptr->dev.ch.sharp_coll_info = NULL;
            }
            if (intra_sock_comm_ptr != NULL) {
                intra_sock_comm_ptr->dev.ch.sharp_coll_info = NULL;
            }
            if (intra_sock_leader_comm_ptr != NULL) {
                intra_sock_leader_comm_ptr->dev.ch.sharp_coll_info = NULL;
            }
#endif

            if (intra_comm_rank == 0) {

                if (intra_sock_leader_comm_ptr != NULL &&
                    intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok == 0) {
                    intra_sock_leader_comm_ptr->dev.ch.tried_to_create_leader_shmem = 1;
                    mpi_errno = create_2level_comm(shmem_ptr->dev.ch.intra_sock_leader_comm,
                                                    intra_sock_leader_comm_ptr->local_size, 
                                                    intra_sock_leader_comm_ptr->rank);
                    if (mpi_errno == MPI_SUCCESS && intra_sock_leader_comm_ptr->local_size == 1) {
                        intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok = 1;
                    }
                }

                int local_leader_shmem_status = intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok;
                int global_leader_shmem_status = 0;
                int allred_flag = mv2_use_socket_aware_allreduce;
                mv2_use_socket_aware_allreduce = 0;
                mpi_errno = MPIR_Allreduce_impl(&local_leader_shmem_status, &global_leader_shmem_status, 1,
                        MPI_INT, MPI_LAND, global_sock_leader_ptr, &errflag);
                mv2_use_socket_aware_allreduce = allred_flag;
                if (mpi_errno) {
                    MPIR_ERR_POP(mpi_errno);
                }
                if (global_leader_shmem_status == 0) {
                    intra_sock_leader_comm_ptr->dev.ch.shmem_coll_ok = 0;    
                }
            }
        }
    }

fn_exit:
    if (intra_socket_leader_map != NULL) {
        MPIU_Free(intra_socket_leader_map);
    }
    if (intra_sock_leader_group != NULL) {
        MPIU_Free(intra_sock_leader_group);
    }
    if (global_socket_leader_map != NULL) {
        MPIU_Free(global_socket_leader_map);
    }
    if (global_sock_leader_group != NULL) {
        MPIU_Free(global_sock_leader_group);
    }
    return mpi_errno;
fn_fail:
    free_intra_sock_comm(comm_ptr);
    goto fn_exit;
}

int create_allgather_comm(MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag)
{
    static const char FCNAME[] = "create_allgather_comm";
    int mpi_errno = MPI_SUCCESS; 
    int is_contig =1, check_leader =1, check_size=1, is_local_ok=0,is_block=0;
    int PPN, i=0;
    int leader_rank = -1, leader_comm_size = -1;
    int size = comm_ptr->local_size; 
    int my_rank = comm_ptr->rank; 
    int my_local_id = -1, my_local_size = -1;
    int grp_index=0, leader=0; 
    MPID_Comm *shmem_ptr=NULL; 
    MPID_Comm *leader_ptr=NULL; 
    MPI_Group allgather_group, comm_group; 
    comm_ptr->dev.ch.allgather_comm=MPI_COMM_NULL; 
    comm_ptr->dev.ch.allgather_new_ranks=NULL;

    if(comm_ptr->dev.ch.leader_comm != MPI_COMM_NULL) { 
        MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_ptr); 
        mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.leader_comm, &leader_rank); 
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.leader_comm, &leader_comm_size);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } 
    if(comm_ptr->dev.ch.shmem_comm != MPI_COMM_NULL){ 
        MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr); 
        mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.shmem_comm, &my_local_size);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } 

    int* shmem_group = MPIU_Malloc(sizeof(int) * size);
    if (NULL == shmem_group){
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }              
    
    MPIDI_VC_t* vc = NULL;
    for (i = 0; i < size ; ++i){
       MPIDI_Comm_get_vc(comm_ptr, i, &vc);
#if CHANNEL_NEMESIS_IB
       if (my_rank == i || vc->ch.is_local)
#else
       if (my_rank == i || vc->smp.local_rank >= 0)
#endif
       {
           shmem_group[grp_index++] = i;
       }   
    }  
    leader = shmem_group[0]; 

    mpi_errno = PMPI_Comm_group(comm_ptr->handle, &comm_group);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno=MPIR_Bcast_impl(&leader_rank, 1, MPI_INT, 0, shmem_ptr, errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    } 

    for (i=1; i < my_local_size; i++ ){
        if (shmem_group[i] != shmem_group[i-1]+1){
            is_contig =0; 
            break;
        }
    }

    if (leader != (my_local_size*leader_rank)){
        check_leader=0;
    }

    if (my_local_size != (size/comm_ptr->dev.ch.leader_group_size)){
        check_size=0;
    }

    is_local_ok = is_contig && check_leader && check_size;

    mpi_errno = MPIR_Allreduce_impl(&is_local_ok, &is_block, 1, 
                                    MPI_INT, MPI_LAND, comm_ptr, errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (is_block) {
        int counter=0,j;
        comm_ptr->dev.ch.allgather_new_ranks = MPIU_Malloc(sizeof(int)*size);
        if (NULL == comm_ptr->dev.ch.allgather_new_ranks){
                mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, 
                                              FCNAME, __LINE__, MPI_ERR_OTHER, 
                                              "**nomem", 0 );
                return mpi_errno;
        }
   
        PPN = my_local_size;
        
        for (j=0; j < PPN; j++){
            for (i=0; i < comm_ptr->dev.ch.leader_group_size; i++){
                comm_ptr->dev.ch.allgather_new_ranks[counter] = j + i*PPN;
                counter++;
            }
        }

        mpi_errno = PMPI_Group_incl(comm_group, size, 
                                    comm_ptr->dev.ch.allgather_new_ranks, 
                                    &allgather_group);
        if(mpi_errno) {
             MPIR_ERR_POP(mpi_errno);
        }  
        mpi_errno = PMPI_Comm_create(comm_ptr->handle, allgather_group, 
                                     &(comm_ptr->dev.ch.allgather_comm));
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        }
        comm_ptr->dev.ch.allgather_comm_ok = 1;
        
        mpi_errno=PMPI_Group_free(&allgather_group);
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        }

    } else {
        /* Set this to -1 so that we never get back in here again
         * for this cyclic comm */
        comm_ptr->dev.ch.allgather_comm_ok = -1;
    }

    /* Gives the mapping to any process's leader in comm */
    comm_ptr->dev.ch.rank_list = MPIU_Malloc(sizeof(int) * size);
    if (NULL == comm_ptr->dev.ch.rank_list){
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }

    /* gather full rank list on leader processes, the rank list is ordered
     * by node based on leader rank, and then by rank within the node according
     * to the shmem_group list */
    if (my_local_id == 0) {
        /* execute allgather or allgatherv across leaders */
        if (comm_ptr->dev.ch.is_uniform != 1) {
            /* allocate memory for displacements and counts */
            int* displs = MPIU_Malloc(sizeof(int) * leader_comm_size);
            int* counts = MPIU_Malloc(sizeof(int) * leader_comm_size);
            if (!displs || !counts) {
                mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                        MPIR_ERR_RECOVERABLE,
                        FCNAME, __LINE__,
                        MPI_ERR_OTHER,
                        "**nomem", 0);
                return mpi_errno;
            }

            /* get pointer to array of node sizes */
            int* node_sizes = comm_ptr->dev.ch.node_sizes;

            /* compute values for displacements and counts arrays */
            displs[0] = 0;
            counts[0] = node_sizes[0];
            for (i = 1; i < leader_comm_size; i++) {
                displs[i] = displs[i - 1] + node_sizes[i - 1];
                counts[i] = node_sizes[i];
            }

            /* execute the allgatherv to collect full rank list */
            mpi_errno = MPIR_Allgatherv_impl(
                shmem_group, my_local_size, MPI_INT,
                comm_ptr->dev.ch.rank_list, counts, displs, MPI_INT,
                leader_ptr, errflag
            );

            /* free displacements and counts arrays */
            MPIU_Free(displs);
            MPIU_Free(counts);
        } else {
            /* execute the allgather to collect full rank list */
            mpi_errno = MPIR_Allgather_impl(
                shmem_group, my_local_size, MPI_INT,
                comm_ptr->dev.ch.rank_list, my_local_size, MPI_INT,
                leader_ptr, errflag
            );
        }

        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* broadcast rank list to other ranks on this node */
    mpi_errno = MPIR_Bcast_impl(comm_ptr->dev.ch.rank_list, size, MPI_INT, 0, shmem_ptr, errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* lookup and record our index within the rank list */
    for (i = 0; i < size; i++) {
        if (my_rank == comm_ptr->dev.ch.rank_list[i]) {
            /* found ourself in the list, record the index */
            comm_ptr->dev.ch.rank_list_index = i;
            break;
        }
    }

    mpi_errno=PMPI_Group_free(&comm_group);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit: 
    MPIU_Free(shmem_group);
    return mpi_errno;
fn_fail: 
    goto fn_exit; 
}

#if defined (_SHARP_SUPPORT_)
int create_sharp_comm(MPI_Comm comm, int size, int my_rank)
{
    static const char FCNAME[] = "create_sharp_comm";
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm* comm_ptr = NULL;
    int leader_group_size = 0, my_local_id = -1;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    if (size <= 1) {
        return mpi_errno;
    }

    MPID_Comm_get_ptr(comm, comm_ptr);
    mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }
    leader_group_size = comm_ptr->dev.ch.leader_group_size;
    comm_ptr->dev.ch.sharp_coll_info = NULL;

    if (comm == MPI_COMM_WORLD && mv2_enable_sharp_coll
        && (leader_group_size > 1) && (comm_ptr->dev.ch.is_sharp_ok == 0)) {
        sharp_info_t * sharp_coll_info = NULL;        

        comm_ptr->dev.ch.sharp_coll_info = 
            (sharp_info_t *)MPIU_Malloc(sizeof(sharp_info_t));
        sharp_coll_info = comm_ptr->dev.ch.sharp_coll_info; 
        sharp_coll_info->sharp_comm_module = NULL;
        sharp_coll_info->sharp_conf = MPIU_Malloc(sizeof(sharp_conf_t));
        
        sharp_coll_log_early_init();  
        mpi_errno = mv2_setup_sharp_env(sharp_coll_info->sharp_conf, MPI_COMM_WORLD);
        if (mpi_errno) {
           MPIR_ERR_POP(mpi_errno);  
        }
        
        /* Initialize sharp */
        if (mv2_enable_sharp_coll == 2) {
            /* Flat algorithm in which every process uses SHArP */
            mpi_errno = mv2_sharp_coll_init(sharp_coll_info->sharp_conf, my_local_id);
        } else if (mv2_enable_sharp_coll == 1) {
            /* Two-level hierarchical algorithm in which, one process at each
             * node uses SHArP for inter-node communication */
            mpi_errno = mv2_sharp_coll_init(sharp_coll_info->sharp_conf, 0);
        } else {
            PRINT_ERROR("Invalid value for MV2_ENABLE_SHARP\n");
            mpi_errno = MPI_ERR_OTHER;
        }
        int can_support_sharp = 0;
        can_support_sharp = (mpi_errno == SHARP_COLL_SUCCESS) ? 1 : 0;


        int global_sharp_init_ok = 0;
        mpi_errno = MPIR_Allreduce_impl(&can_support_sharp, &global_sharp_init_ok,  1, 
                                        MPI_INT, MPI_LAND, comm_ptr, &errflag);
        if (mpi_errno) {
           MPIR_ERR_POP(mpi_errno);  
        }

        if (global_sharp_init_ok == 0) {
           mv2_free_sharp_handlers(comm_ptr->dev.ch.sharp_coll_info); 
           /* avoid using sharp and fall back to other designs */
           comm_ptr->dev.ch.sharp_coll_info = NULL;
           comm_ptr->dev.ch.is_sharp_ok = -1; /* we set it to -1 so that we do not get back to here anymore */
           mpi_errno = MPI_SUCCESS;
           PRINT_DEBUG(DEBUG_Sharp_verbose, "Falling back from Sharp  \n");
           goto sharp_fall_back;
        } 

        sharp_coll_info->sharp_comm_module = MPIU_Malloc(sizeof(coll_sharp_module_t));
        MPIU_Memset(sharp_coll_info->sharp_comm_module, 0, sizeof(coll_sharp_module_t));
        /* create sharp module which contains sharp communicator */
        if (mv2_enable_sharp_coll == 2) {
            sharp_coll_info->sharp_comm_module->comm = MPI_COMM_WORLD; 
            mpi_errno = mv2_sharp_coll_comm_init(sharp_coll_info->sharp_comm_module);
            if (mpi_errno) {
               MPIR_ERR_POP(mpi_errno);  
            } 
        } else if (my_local_id == 0) {
            sharp_coll_info->sharp_comm_module->comm = comm_ptr->dev.ch.leader_comm;
            mpi_errno = mv2_sharp_coll_comm_init(sharp_coll_info->sharp_comm_module);
            if (mpi_errno) {
               MPIR_ERR_POP(mpi_errno);  
            } 
        }
        comm_ptr->dev.ch.is_sharp_ok = 1;

        PRINT_DEBUG(DEBUG_Sharp_verbose, "Sharp was initialized successfully \n");

        /* If the user does not set the MV2_SHARP_MAX_MSG_SIZE then try to tune
         * mv2_sharp_tuned_msg_size variable based on node count */
        if (mv2_enable_sharp_coll == 1 &&
            (getenv("MV2_SHARP_MAX_MSG_SIZE")) == NULL) {
            if (leader_group_size == 2) {
                    mv2_sharp_tuned_msg_size = 256;
            } else if (leader_group_size <= 4) {
                mv2_sharp_tuned_msg_size = 512;
            } else {
                /* in all other cases set max msg size to
                 * MV2_DEFAULT_SHARP_MAX_MSG_SIZE */
                mv2_sharp_tuned_msg_size = MV2_DEFAULT_SHARP_MAX_MSG_SIZE;
            }
        }
    }
sharp_fall_back:

    fn_exit:
       return (mpi_errno);
    fn_fail: 
       MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
       mv2_free_sharp_handlers(comm_ptr->dev.ch.sharp_coll_info);
       comm_ptr->dev.ch.sharp_coll_info = NULL;
       goto fn_exit; 
}
#endif /*(_SHARP_SUPPORT_)*/

#if defined(_MCST_SUPPORT_)
int create_mcast_comm (MPI_Comm comm, int size, int my_rank)
{
    static const char FCNAME[] = "create_mcast_comm";
    int mpi_errno = MPI_SUCCESS;
    int mcast_setup_success = 0;
    int leader_group_size = 0, my_local_id = -1;
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm *shmem_ptr = NULL;
    MPID_Comm *leader_ptr = NULL;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;

    if (size <= 1) {
        return mpi_errno;
    }

    MPID_Comm_get_ptr(comm, comm_ptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr);
    MPID_Comm_get_ptr(comm_ptr->dev.ch.leader_comm, leader_ptr);
    
    mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    leader_group_size = comm_ptr->dev.ch.leader_group_size;

    comm_ptr->dev.ch.is_mcast_ok = 0;
    bcast_info_t **bcast_info = (bcast_info_t **)&comm_ptr->dev.ch.bcast_info;
    if (leader_group_size >= mcast_num_nodes_threshold && rdma_enable_mcast) {
        mv2_mcast_init_bcast_info(bcast_info);
        if (my_local_id == 0) {
            if (mv2_setup_multicast(&(*bcast_info)->minfo, comm_ptr) == MCAST_SUCCESS) {
                mcast_setup_success = 1;
            }   
        }

        int leader_rank;
        int status = 0;
        int mcast_status[2] = {0, 0}; /* status, comm_id */
        if(comm_ptr->dev.ch.leader_comm != MPI_COMM_NULL) { 
            PMPI_Comm_rank(comm_ptr->dev.ch.leader_comm, &leader_rank); 
            if (leader_rank == 0 && mcast_setup_success) {
                /* Wait for comm ready */
                status = mv2_mcast_progress_comm_ready(comm_ptr);
            }
        } 

        if (my_local_id == 0) {
            mpi_errno = MPIR_Bcast_impl(&status, 1, MPI_INT, 0, leader_ptr, &errflag);
            if (mpi_errno) {
                goto fn_fail;
            }
            mcast_status[0] = status;
            mcast_status[1] = ((bcast_info_t *) comm_ptr->dev.ch.bcast_info)->minfo.grp_info.comm_id;
            if (!status) {
                mv2_cleanup_multicast(&((bcast_info_t *) comm_ptr->dev.ch.bcast_info)->minfo, comm_ptr);
            }
        }

        mpi_errno = MPIR_Bcast_impl (mcast_status, 2, MPI_INT, 0, shmem_ptr, &errflag);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }

        comm_ptr->dev.ch.is_mcast_ok = mcast_status[0];

        if (my_rank == 0) {
            PRINT_DEBUG(DEBUG_MCST_verbose > 1, 
                    "multicast setup status:%d\n", comm_ptr->dev.ch.is_mcast_ok);
            if (comm_ptr->dev.ch.is_mcast_ok == 0) {
                PRINT_INFO (1, "Warning: Multicast group setup failed. Not using any multicast features\n");
            }
        }

        if (comm_ptr->dev.ch.is_mcast_ok == 0 && comm == MPI_COMM_WORLD) {
            /* if mcast setup failed on comm world because of any reason, it is
            ** most likely is going to fail on other communicators. Hence, disable
            ** the mcast feaure */
            rdma_enable_mcast = 0;
            mv2_ud_destroy_ctx(mcast_ctx->ud_ctx);
            MPIU_Free(mcast_ctx);
            PRINT_DEBUG(DEBUG_MCST_verbose,"mcast setup failed on comm world, disabling mcast\n");
        }
	
    }
    fn_exit:
       return (mpi_errno);
    fn_fail: 
       MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
       goto fn_exit; 
}
#endif /*(_MCST_SUPPORT_)*/

int create_2level_comm (MPI_Comm comm, int size, int my_rank)
{
    static const char FCNAME[] = "create_2level_comm";
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm* comm_ptr = NULL;
    MPI_Group subgroup1, comm_group; 
    MPID_Group *group_ptr=NULL;
    int leader_comm_size, my_local_size, my_local_id;
    int input_flag = 0, output_flag = 0;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int leader_group_size = 0;
    int mv2_shmem_coll_blk_stat = 0;
    int iter;
    MPID_Node_id_t node_id;
    int blocked = 0;
    int up = 0;
    int down = 0;
    int prev = -1;
    int shmem_size;

    MPIU_THREADPRIV_DECL;
    MPIU_THREADPRIV_GET;

    MPID_Comm_get_ptr( comm, comm_ptr );
    if (size <= 1) {
        return mpi_errno;
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_2level_comm_requests, 1);

    /* Find out if ranks are block ordered locally */
    for (iter = 0; iter < size; iter++) {
        MPID_Get_node_id(comm_ptr, iter, &node_id);
        if ((node_id != -1) && (prev == -1)) {
            up++;
        }
        if ((node_id == -1) && (prev == 1)) {
            down++;
        }
        prev = node_id;
    }
    blocked = (up > 1) ? 0 : 1;
    
    int* shmem_group = MPIU_Malloc(sizeof(int) * size);
    if (NULL == shmem_group){
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }

    /* Creating local shmem group */
    int i = 0;
    int local_rank = 0;
    int grp_index = 0;
    comm_ptr->dev.ch.leader_comm=MPI_COMM_NULL;
    comm_ptr->dev.ch.shmem_comm=MPI_COMM_NULL;

    MPIDI_VC_t* vc = NULL;
    for (i = 0; i < size ; ++i){
       MPIDI_Comm_get_vc(comm_ptr, i, &vc);
#ifdef CHANNEL_NEMESIS_IB
       if (my_rank == i || vc->ch.is_local)
#else
       if (my_rank == i || vc->smp.local_rank >= 0)
#endif
        {
           shmem_group[grp_index] = i;
           if (my_rank == i){
               local_rank = grp_index;
           }
           ++grp_index;
       }  
    }
    shmem_size = grp_index;

    if (local_rank == 0){
        lock_shmem_region();
        mv2_shmem_coll_blk_stat = MPIDI_CH3I_SHMEM_Coll_get_free_block();
        if (mv2_shmem_coll_blk_stat >= 0) {
            input_flag = 1;
        }
        unlock_shmem_region();
    } else {
        input_flag = 1;
    }

    mpi_errno = MPIR_Allreduce_impl(&input_flag, &output_flag, 1, 
            MPI_INT, MPI_LAND, comm_ptr, 
            &errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    } 

    mpi_errno = MPIR_Allreduce_impl(&blocked, &(comm_ptr->dev.ch.is_blocked), 1, 
            MPI_INT, MPI_LAND, comm_ptr, 
            &errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (!output_flag) {
        /* None of the shmem-coll-blocks are available. We cannot support
         * shared-memory collectives for this communicator */
        if (local_rank == 0 && mv2_shmem_coll_blk_stat >= 0) {
            /*relese the slot if it is aquired */ 
            lock_shmem_region();
            MPIDI_CH3I_SHMEM_Coll_Block_Clear_Status(mv2_shmem_coll_blk_stat); 
            unlock_shmem_region();
        }
        comm_ptr->dev.ch.shmem_coll_ok = -1; 
        goto fn_exit;
    }

    MPIR_T_PVAR_COUNTER_INC(MV2, mv2_num_2level_comm_success, 1);

    /* Creating leader group */
    int leader = 0;
    leader = shmem_group[0];

    /* Gives the mapping to any process's leader in comm */
    comm_ptr->dev.ch.leader_map = MPIU_Malloc(sizeof(int) * size);
    if (NULL == comm_ptr->dev.ch.leader_map){
        mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }
    
    mpi_errno = MPIR_Allgather_impl (&leader, 1, MPI_INT , comm_ptr->dev.ch.leader_map,
                                        1, MPI_INT, comm_ptr, &errflag);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    int* leader_group = MPIU_Malloc(sizeof(int) * size);
    if (NULL == leader_group){
       mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }

    /* Gives the mapping from leader's rank in comm to 
     * leader's rank in leader_comm */
    comm_ptr->dev.ch.leader_rank = MPIU_Malloc(sizeof(int) * size);
    if (NULL == comm_ptr->dev.ch.leader_rank){
       mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }

    for (i=0; i < size ; ++i){
         comm_ptr->dev.ch.leader_rank[i] = -1;
    }
    int* group = comm_ptr->dev.ch.leader_map;
    grp_index = 0;
    for (i=0; i < size ; ++i){
        if (comm_ptr->dev.ch.leader_rank[(group[i])] == -1){
            comm_ptr->dev.ch.leader_rank[(group[i])] = grp_index;
            leader_group[grp_index++] = group[i];
           
        }
    }
    leader_group_size = grp_index;
    comm_ptr->dev.ch.leader_group_size = leader_group_size;

    mpi_errno = PMPI_Comm_group(comm, &comm_group);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = PMPI_Group_incl(comm_group, leader_group_size, leader_group, &subgroup1);
     if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = PMPI_Comm_create(comm, subgroup1, &(comm_ptr->dev.ch.leader_comm));
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    MPID_Comm *leader_ptr;
    MPID_Comm_get_ptr( comm_ptr->dev.ch.leader_comm, leader_ptr );
    if(leader_ptr != NULL) { 
        /* Set leader_ptr's shmem_coll_ok so that we dont call create_2level_comm on
         * it again */ 
        leader_ptr->dev.ch.shmem_coll_ok = -1; 
    } 
       
    MPIU_Free(leader_group);
    MPID_Group_get_ptr( subgroup1, group_ptr );
    if(group_ptr != NULL) { 
       mpi_errno = PMPI_Group_free(&subgroup1);
       if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
       }
    }

    mpi_errno = PMPI_Group_incl(comm_group, shmem_size, shmem_group, &subgroup1);
     if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = PMPI_Comm_create(comm, subgroup1, &(comm_ptr->dev.ch.shmem_comm));
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }
    
    /*
    mpi_errno = PMPI_Comm_split(comm, leader, local_rank, &(comm_ptr->dev.ch.shmem_comm));
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }
    */

    MPID_Comm *shmem_ptr;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_ptr);
    if(shmem_ptr != NULL) { 
        /* Set shmem_ptr's shmem_coll_ok so that we dont call create_2level_comm on
         * it again */ 
        shmem_ptr->dev.ch.shmem_coll_ok = -1; 
    } 
    
    mpi_errno = PMPI_Comm_rank(comm_ptr->dev.ch.shmem_comm, &my_local_id);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }
    mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.shmem_comm, &my_local_size);
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }
    comm_ptr->dev.ch.intra_node_done = 0;

    if(my_local_id == 0) { 
           int array_index=0;
           mpi_errno = PMPI_Comm_size(comm_ptr->dev.ch.leader_comm, &leader_comm_size);
           if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
           }

           comm_ptr->dev.ch.node_sizes = MPIU_Malloc(sizeof(int)*leader_comm_size);
           mpi_errno = PMPI_Allgather(&my_local_size, 1, MPI_INT,
				 comm_ptr->dev.ch.node_sizes, 1, MPI_INT, comm_ptr->dev.ch.leader_comm);
           if(mpi_errno) {
              MPIR_ERR_POP(mpi_errno);
           }

           /* allocate memory for displacements into rank_list */
           comm_ptr->dev.ch.node_disps = MPIU_Malloc(sizeof(int)*leader_comm_size);

           /* compute values for displacements and counts arrays */
           int* sizes = comm_ptr->dev.ch.node_sizes;
           int* disps = comm_ptr->dev.ch.node_disps;
           disps[0] = 0;
           for (i = 1; i < leader_comm_size; i++) {
               disps[i] = disps[i - 1] + sizes[i - 1];
           }

           comm_ptr->dev.ch.is_uniform = 1; 
           for(array_index=0; array_index < leader_comm_size; array_index++) { 
                if(comm_ptr->dev.ch.node_sizes[0] != comm_ptr->dev.ch.node_sizes[array_index]) {
                     comm_ptr->dev.ch.is_uniform = 0; 
                     break;
                }
           }
     }

    comm_ptr->dev.ch.is_global_block = 0; 
    /* We need to check to see if the ranks are block or not. Each node leader
     * gets the global ranks of all of its children processes. It scans through
     * this array to see if the ranks are in block order. The node-leaders then
     * do an allreduce to see if all the other nodes are also in block order.
     * This is followed by an intra-node bcast to let the children processes
     * know of the result of this step */ 
    if(my_local_id == 0) {
        int is_local_block = 1; 
        int index = 1; 
        
        while( index < my_local_size) { 
            if((shmem_group[index] - 1) != 
               shmem_group[index - 1]) { 
                is_local_block = 0; 
                break; 
            }
            index++;  
        }  

        mpi_errno = MPIR_Allreduce_impl(&(is_local_block), 
                                  &(comm_ptr->dev.ch.is_global_block), 1, 
                                  MPI_INT, MPI_LAND, leader_ptr, &errflag);
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        } 
        mpi_errno = MPIR_Bcast_impl(&(comm_ptr->dev.ch.is_global_block),1, MPI_INT, 0,
                               shmem_ptr, &errflag); 
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        } 
    } else { 
        mpi_errno = MPIR_Bcast_impl(&(comm_ptr->dev.ch.is_global_block),1, MPI_INT, 0,
                               shmem_ptr, &errflag); 
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        } 
    }      

    /* bcast uniformity info to node local processes for tuning selection
       later */
    mpi_errno = MPIR_Bcast_impl(&(comm_ptr->dev.ch.is_uniform),1, MPI_INT, 0,
                           shmem_ptr, &errflag); 
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    } 

    comm_ptr->dev.ch.allgather_comm_ok = 0;

    mpi_errno=PMPI_Group_free(&comm_group);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
                             
    shmem_ptr->dev.ch.shmem_coll_ok = 0;
    /* To prevent Bcast taking the knomial_2level_bcast route */
    mpi_errno = MPIR_Bcast_impl (&mv2_shmem_coll_blk_stat, 1, 
            MPI_INT, 0, shmem_ptr, &errflag);
    if(mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    MPIU_Assert(mv2_shmem_coll_blk_stat >= 0);

    shmem_ptr->dev.ch.shmem_comm_rank = mv2_shmem_coll_blk_stat; 

    if (mv2_use_slot_shmem_coll) {
        comm_ptr->dev.ch.shmem_info = mv2_shm_coll_init(mv2_shmem_coll_blk_stat, my_local_id, 
                                                    my_local_size, comm_ptr); 
        if (comm_ptr->dev.ch.shmem_info == NULL) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                    FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                    "collective shmem allocation failed", strerror(errno));
            MPIR_ERR_POP(mpi_errno);
        }
        shmem_ptr->dev.ch.shmem_info = comm_ptr->dev.ch.shmem_info;
    }
    comm_ptr->dev.ch.shmem_coll_ok = 1;

#if defined(_SMP_LIMIC_)
    if(comm_ptr->dev.ch.shmem_coll_ok == 1) {
        mpi_errno = create_intra_node_multi_level_comm(comm_ptr);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif  /* #if defined(_SMP_LIMIC_) */ 

    if(mv2_enable_socket_aware_collectives)
    {
        //tried_to_create_leader_shmem exists to ensure socket-aware comms aren't recursively created 
        //unnecessarily (which would cause memory leaks)
        if(comm_ptr->dev.ch.shmem_coll_ok == 1 && comm_ptr->dev.ch.tried_to_create_leader_shmem == 0) {
            mpi_errno = create_intra_sock_comm(comm);
            if(mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

    
    /************************** Added by Mehran *********************************/
    /** We are going to determine if:
     * 1- local sizes are equal? i. e. each node has p ranks
     **/
    if(my_local_id == 0){
    int* local_sizes = MPIU_Malloc(sizeof(int) * leader_comm_size);
    if (NULL == local_sizes){
       mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s: %s",
                   "memory allocation failed", strerror(errno));
                   MPIR_ERR_POP(mpi_errno);
    }
        mpi_errno = MPIR_Allgather_impl(&my_local_size, 1, MPI_INT,
                    local_sizes, 1, MPI_INT, leader_ptr, &errflag);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
        int idx;
        comm_ptr->dev.ch.equal_local_sizes=1;
        for(idx=0; idx<leader_comm_size-1; ++idx){
            if(local_sizes[idx]!= local_sizes[idx+1]){
                comm_ptr->dev.ch.equal_local_sizes=0;
                break; 
            }
        }

        mpi_errno = MPIR_Bcast_impl(&(comm_ptr->dev.ch.equal_local_sizes),1, MPI_INT, 0,
                               shmem_ptr, &errflag); 
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        }
	MPIU_Free(local_sizes);
    }else{
        mpi_errno = MPIR_Bcast_impl(&(comm_ptr->dev.ch.equal_local_sizes),1, MPI_INT, 0,
                               shmem_ptr, &errflag); 
        if(mpi_errno) {
           MPIR_ERR_POP(mpi_errno);
        } 
    }
    /*****************************************************************************/

    fn_exit:
       MPIU_Free(shmem_group);
       return (mpi_errno);
    fn_fail: 
       MPIDU_ERR_CHECK_MULTIPLE_THREADS_EXIT( comm_ptr );
       goto fn_exit; 
}

int init_thread_reg(void)
{
    int j = 0;

    for (; j < MAX_NUM_THREADS; ++j)
    {
        thread_reg[j] = -1;
    }

    return 1;
}

int check_split_comm(pthread_t my_id)
{
    int j = 0;
    pthread_mutex_lock(&comm_lock);

    for (; j < MAX_NUM_THREADS; ++j)
    {
        if (pthread_equal(thread_reg[j], my_id))
        {
            pthread_mutex_unlock(&comm_lock);
            return 0;
        }
    }

    pthread_mutex_unlock(&comm_lock);
    return 1;
}

int disable_split_comm(pthread_t my_id)
{
    int j = 0;
    int found = 0;
    int mpi_errno=MPI_SUCCESS; 
    static const char FCNAME[] = "disable_split_comm";
    pthread_mutex_lock(&comm_lock);

    for (; j < MAX_NUM_THREADS; ++j)
    {
        if (thread_reg[j] == -1)
        {
            thread_reg[j] = my_id;
            found = 1;
            break;
        }
    }

    pthread_mutex_unlock(&comm_lock);

    if (found == 0)
    {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                           FCNAME, __LINE__, MPI_ERR_OTHER,
                                           "**fail", "**fail %s", "max_num_threads created");
        return mpi_errno;

    }

    return 1;
}


int enable_split_comm(pthread_t my_id)
{
    int j = 0;
    int found = 0;
    int mpi_errno=MPI_SUCCESS; 
    static const char FCNAME[] = "enable_split_comm";
    pthread_mutex_lock(&comm_lock);

    for (; j < MAX_NUM_THREADS; ++j)
    {
        if (pthread_equal(thread_reg[j], my_id))
        {
            thread_reg[j] = -1;
            found = 1;
            break;
        }
    }

    pthread_mutex_unlock(&comm_lock);

    if (found == 0)
    {
        mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
                                           FCNAME, __LINE__, MPI_ERR_OTHER,
                                           "**fail", "**fail %s", "max_num_threads created");
        return mpi_errno;
    }

    return 1;
}
