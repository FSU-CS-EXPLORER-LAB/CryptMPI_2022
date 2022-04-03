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

#ifndef _REDUCE_TUNING_
#define _REDUCE_TUNING_

#include "coll_shmem.h"

#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Reduce tuning flags
 * flat binomial: MV2_INTER_REDUCE_TUNING=1
 * flat knomial:  MV2_INTER_REDUCE_TUNING=2 
 *                MV2_USE_INTER_KNOMIAL_REDUCE_FACTOR=?
 * flat reduce-scatter-gather(rsa): MV2_INTER_REDUCE_TUNING=5
 * 2-level: MV2_INTER_REDUCE_TUNING=? MV2_INTRA_REDUCE_TUNING=? 
 *          MV2_USE_INTRA_KNOMIAL_REDUCE_FACTOR=? 
 *          MV2_USE_INTER_KNOMIAL_REDUCE_FACTOR=?
 *          MV2_INTER_REDUCE_TUNING_TWO_LEVEL=1
 *          where intra-reduce flag takes 1(binomial) 2(knomial) 4(shm) 5(rsa) 
 */

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Reduce_function)(const void *sendbuf,
                                 void *recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 int root,
                                 MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_reduce_tuning_element;

typedef struct {
    int numproc; 
    int inter_k_degree;
    int intra_k_degree;
    int is_two_level_reduce[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_reduce_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_reduce_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_reduce_tuning_table;

extern int mv2_size_reduce_tuning_table;
extern mv2_reduce_tuning_table *mv2_reduce_thresholds_table;
extern int mv2_use_old_reduce;

/*Entries related to indexed tuning table*/
typedef struct {
    int msg_sz;
    int (*MV2_pt_Reduce_function)(const void *sendbuf,
                                 void *recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 int root,
                                 MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_reduce_indexed_tuning_element;

typedef struct {
    int numproc; 
    int inter_k_degree;
    int intra_k_degree;
    int is_two_level_reduce[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_reduce_indexed_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_reduce_indexed_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_reduce_indexed_tuning_table;

/* Indicates number of processes per node */
extern int *mv2_reduce_indexed_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_reduce_indexed_num_ppn_conf;
extern int *mv2_size_reduce_indexed_tuning_table;
extern mv2_reduce_indexed_tuning_table **mv2_reduce_indexed_thresholds_table;

extern int MPIR_Reduce_binomial_MV2(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          int root,
                                          MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_intra_knomial_wrapper_MV2(const void *sendbuf,
                                   void *recvbuf,
                                   int count,
                                   MPI_Datatype datatype,
                                   MPI_Op op,
                                   int root,
                                   MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_inter_knomial_wrapper_MV2(const void *sendbuf,
                                   void *recvbuf,
                                   int count,
                                   MPI_Datatype datatype,
                                   MPI_Op op,
                                   int root,
                                   MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_shmem_MV2(const void *sendbuf,
                                 void *recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 int root,
                                 MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_redscat_gather_MV2(const void *sendbuf,
                                          void *recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          int root,
                                          MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
  
extern int MPIR_Reduce_two_level_helper_MV2(const void *sendbuf,
                                            void *recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            int root,
                                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_Zcpy_MV2(const void *sendbuf,
				void *recvbuf,
				int count,
				MPI_Datatype datatype,
				MPI_Op op,
				int root,
				MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

/* Architecture detection tuning */
int MV2_set_reduce_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by reduce tuning table*/
void MV2_cleanup_reduce_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune reduce thresholds */
int MV2_internode_Reduce_is_define(char *mv2_user_reduce_inter, char
                                  *mv2_user_reduce_intra);
int MV2_intranode_Reduce_is_define(char *mv2_user_reduce_intra);
                                           

#endif


