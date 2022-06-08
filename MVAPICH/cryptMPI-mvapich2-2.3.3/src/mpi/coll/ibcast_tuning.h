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

#ifndef _IBCAST_TUNING_
#define _IBCAST_TUNING_

#include "coll_shmem.h"
#if defined(CHANNEL_MRAIL)
#include "ibv_param.h"
#endif                          /* #if defined(CHANNEL_MRAIL) */

#define NMATCH (3+1)

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Ibcast_function) (void *buffer, int count, MPI_Datatype datatype,
				   int root, MPID_Comm *comm_ptr, MPID_Sched_t s);
    int zcpy_knomial_factor;
} mv2_ibcast_tuning_element;

typedef struct {
    int numproc;
    int ibcast_segment_size;
    int intra_node_knomial_factor;
    int inter_node_knomial_factor;
    int is_two_level_ibcast[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_ibcast_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_ibcast_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_ibcast_tuning_table;

//extern int mv2_use_pipelined_bcast;
//extern int mv2_pipelined_knomial_factor; 
//extern int mv2_pipelined_zcpy_knomial_factor; 
//extern int zcpy_knomial_factor;
extern int ibcast_segment_size;
extern int mv2_size_ibcast_tuning_table;
extern mv2_ibcast_tuning_table *mv2_ibcast_thresholds_table;

/* Architecture detection tuning */
int MV2_set_ibcast_tuning_table(int heterogeneity);

/* Function to clean free memory allocated by ibcast tuning table*/
void MV2_cleanup_ibcast_tuning_table();

// Consider removing
/* Function used inside ch3_shmem_coll.c to tune ibcast thresholds */
int MV2_internode_Ibcast_is_define(char *mv2_user_ibcast_inter, char *mv2_user_ibcast_intra); 
int MV2_intranode_Ibcast_is_define(char *mv2_user_ibcast_intra);

extern int MPIR_Ibcast_binomial(void *buffer, int count, MPI_Datatype datatype,
				int root, MPID_Comm *comm_ptr, MPID_Sched_t s);

extern int MPIR_Ibcast_scatter_rec_dbl_allgather(void *buffer, int count,
						 MPI_Datatype datatype, int root,
						 MPID_Comm *comm_ptr, MPID_Sched_t s);

extern int MPIR_Ibcast_scatter_ring_allgather(void *buffer, int count,
					      MPI_Datatype datatype, int root,
					      MPID_Comm *comm_ptr, MPID_Sched_t s);
#endif
