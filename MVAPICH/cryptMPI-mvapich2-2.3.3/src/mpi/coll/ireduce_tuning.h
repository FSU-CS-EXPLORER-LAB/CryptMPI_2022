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

#ifndef _IREDUCE_TUNING_
#define _IREDUCE_TUNING_

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
    int (*MV2_pt_Ireduce_function) (const void *sendbuf, void *recvbuf, int count,
				     MPI_Datatype datatype, MPI_Op op, int root,
				     MPID_Comm *comm_ptr, MPID_Sched_t s);
    int zcpy_knomial_factor;
} mv2_ireduce_tuning_element;

typedef struct {
    int numproc;
    int ireduce_segment_size;
    int intra_node_knomial_factor;
    int inter_node_knomial_factor;
    int is_two_level_ireduce[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_ireduce_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_ireduce_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_ireduce_tuning_table;

//extern int mv2_use_pipelined_reduce;
//extern int mv2_pipelined_knomial_factor; 
//extern int mv2_pipelined_zcpy_knomial_factor; 
//extern int zcpy_knomial_factor;
extern int ireduce_segment_size;
extern int mv2_size_ireduce_tuning_table;
extern mv2_ireduce_tuning_table *mv2_ireduce_thresholds_table;

/* Architecture detection tuning */
int MV2_set_ireduce_tuning_table(int heterogeneity);


/* Function to clean free memory allocated by ireduce tuning table*/
void MV2_cleanup_ireduce_tuning_table();

// Consider removing
/* Function used inside ch3_shmem_coll.c to tune ireduce thresholds */
int MV2_internode_Ireduce_is_define(char *mv2_user_ireduce_inter, char *mv2_user_ireduce_intra); 
int MV2_intranode_Ireduce_is_define(char *mv2_user_ireduce_intra);

extern int MPIR_Ireduce_binomial(const void *sendbuf, void *recvbuf, int count,
				 MPI_Datatype datatype, MPI_Op op, int root,
				 MPID_Comm *comm_ptr, MPID_Sched_t s);
extern int MPIR_Ireduce_redscat_gather(const void *sendbuf, void *recvbuf, int count,
			    MPI_Datatype datatype, MPI_Op op, int root,
			    MPID_Comm *comm_ptr, MPID_Sched_t s);
#endif
