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

#ifndef _SCATTER_TUNING_
#define _SCATTER_TUNING_

#include "coll_shmem.h"

#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Scatter tuning flag
 * Binomial: MV2_INTER_SCATTER_TUNING=1
 * Direct: MV2_INTER_SCATTER_TUNING=2
 * Mcast: MV2_INTER_SCATTER_TUNING=5
 *        MV2_USE_MCAST_SCATTER=1 MV2_USE_MCAST_PIPELINE_SHM=1 MV2_USE_MCAST=1 
 * 2-level inter-binomial-intra-binomial: 
 *        MV2_INTER_SCATTER_TUNING=3 MV2_INTRA_SCATTER_TUNING=1 
 * 2-level inter-binomial-intra-direct:
 *        MV2_INTER_SCATTER_TUNING=3 MV2_INTRA_SCATTER_TUNING=2 
 * 2-level inter-direct-intra-binomial: 
 *        MV2_INTER_SCATTER_TUNING=4 MV2_INTRA_SCATTER_TUNING=1
 * 2-level inter-direct-intra-direct: 
 *        MV2_INTER_SCATTER_TUNING=4 MV2_INTRA_SCATTER_TUNING=2
 */

/* Indicates number of processes per node */
extern int *mv2_scatter_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_scatter_num_ppn_conf;

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Scatter_function)(const void *sendbuf,
                                   int sendcnt,
                                   MPI_Datatype sendtype,
                                   void *recvbuf,
                                   int recvcnt,
                                   MPI_Datatype recvtype,
                                   int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_scatter_tuning_element;

typedef struct {
    int numproc;
    int size_inter_table;
    mv2_scatter_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_scatter_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_scatter_tuning_table;

extern int *mv2_size_scatter_tuning_table;
extern mv2_scatter_tuning_table **mv2_scatter_thresholds_table;
extern int mv2_use_old_scatter;

/*Entries related to indexed tuning table*/
typedef struct {
    int msg_sz;
    int (*MV2_pt_Scatter_function)(const void *sendbuf,
                                   int sendcnt,
                                   MPI_Datatype sendtype,
                                   void *recvbuf,
                                   int recvcnt,
                                   MPI_Datatype recvtype,
                                   int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_scatter_indexed_tuning_element;

typedef struct {
    int numproc;
    int size_inter_table;
    mv2_scatter_indexed_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_scatter_indexed_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_scatter_indexed_tuning_table;

/* Indicates number of processes per node */
extern int *mv2_scatter_indexed_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_scatter_indexed_num_ppn_conf;
extern int *mv2_size_scatter_indexed_tuning_table;
extern mv2_scatter_indexed_tuning_table **mv2_scatter_indexed_thresholds_table;

extern int MPIR_Scatter_mcst_MV2(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


extern int MPIR_Scatter_mcst_wrap_MV2(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Scatter_MV2_Binomial(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


extern int MPIR_Scatter_MV2_Direct(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Scatter_MV2_Direct_Blk(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


extern int MPIR_Scatter_MV2_two_level_Binomial(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


extern int MPIR_Scatter_MV2_two_level_Direct(const void *sendbuf,
                              int sendcnt,
                              MPI_Datatype sendtype,
                              void *recvbuf,
                              int recvcnt,
                              MPI_Datatype recvtype,
                              int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

/* Architecture detection tuning */
int MV2_set_scatter_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by scatter tuning table*/
void MV2_cleanup_scatter_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune scatter thresholds */
int MV2_internode_Scatter_is_define(char *mv2_user_scatter_inter, char
                                  *mv2_user_scatter_intra);
int MV2_intranode_Scatter_is_define(char *mv2_user_scatter_intra);
                                           

#endif


