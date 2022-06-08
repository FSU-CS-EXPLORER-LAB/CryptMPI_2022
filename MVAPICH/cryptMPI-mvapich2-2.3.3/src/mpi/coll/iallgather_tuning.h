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

#ifndef _IALLGATHER_TUNING_
#define _IALLGATHER_TUNING_

#include "coll_shmem.h"
#if defined(CHANNEL_MRAIL)
#include "ibv_param.h"
#endif                          /* #if defined(CHANNEL_MRAIL) */

#define NMATCH (3+1)

/* Note: Several members of the structures used are meant to be used
   sometime in the future */

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Iallgather_function) (const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s);
    int zcopy_factor;
} mv2_iallgather_tuning_element;

typedef struct {
    int numproc;
    int iallgather_segment_size;
    int is_two_level_iallgather[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_iallgather_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_iallgather_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_iallgather_tuning_table;

//extern int mv2_use_pipelined_allgather;
//extern int mv2_pipelined_knomial_factor; 
//extern int mv2_pipelined_zcpy_knomial_factor; 
//extern int zcpy_knomial_factor;
extern int iallgather_segment_size;
extern int mv2_size_iallgather_tuning_table;
extern mv2_iallgather_tuning_table *mv2_iallgather_thresholds_table;

/* Architecture detection tuning */
int MV2_set_iallgather_tuning_table(int heterogeneity);

/* Function to clean free memory allocated by iallgather tuning table*/
void MV2_cleanup_iallgather_tuning_table();

// Consider removing
/* Function used inside ch3_shmem_coll.c to tune iallgather thresholds */
int MV2_internode_Iallgather_is_define(char *mv2_user_iallgather_inter, char *mv2_user_iallgather_intra); 
int MV2_intranode_Iallgather_is_define(char *mv2_user_iallgather_intra);

extern int MPIR_Iallgather_ring(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s);
extern int MPIR_Iallgather_bruck(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s);
extern int MPIR_Iallgather_rec_dbl(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPID_Comm *comm_ptr, MPID_Sched_t s);
#endif
