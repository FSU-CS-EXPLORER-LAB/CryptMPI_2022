/* Copyright (c) 2001-2019, The Ohio State University. All rights
 *  * reserved.
 *   *
 *    * This file is part of the MVAPICH2 software package developed by the
 *     * team members of The Ohio State University's Network-Based Computing
 *      * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *       *
 *        * For detailed copyright and licensing information, please refer to the
 *         * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *          *
 *           */

#ifndef _ALLTOALLV_TUNING_
#define _ALLTOALLV_TUNING_

#include "coll_shmem.h"

#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Alltoallv tuning flags
  0: Alltoallv_intra_MV2
  1: Alltoallv_MV2 
*/

/* Indicates number of processes per node */
extern int *mv2_alltoallv_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_alltoallv_num_ppn_conf;

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Alltoallv_function) (const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_alltoallv_tuning_element;

typedef struct {
    int numproc;
    int size_table;
    mv2_alltoallv_tuning_element algo_table[MV2_MAX_NB_THRESHOLDS];
    mv2_alltoallv_tuning_element in_place_algo_table[MV2_MAX_NB_THRESHOLDS];
} mv2_alltoallv_tuning_table;

extern int *mv2_size_alltoallv_tuning_table;
extern mv2_alltoallv_tuning_table **mv2_alltoallv_thresholds_table;
extern int mv2_use_old_alltoallv;

typedef struct {
    int msg_sz;
    int (*MV2_pt_Alltoallv_function) (const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_alltoallv_indexed_tuning_element;

typedef struct {
    int numproc;
    int in_place_algo_table[MV2_MAX_NB_THRESHOLDS];
    int size_table;
    mv2_alltoallv_indexed_tuning_element algo_table[MV2_MAX_NB_THRESHOLDS];
} mv2_alltoallv_indexed_tuning_table;

/* Indicates number of processes per node */
extern int *mv2_alltoallv_indexed_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_alltoallv_indexed_num_ppn_conf;
extern int *mv2_size_alltoallv_indexed_tuning_table;
extern mv2_alltoallv_indexed_tuning_table **mv2_alltoallv_indexed_thresholds_table;

extern int MPIR_Alltoallv_index_tuned_intra_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Alltoallv_intra_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Alltoallv_intra_scatter_MV2(const void *sendbuf,
                             const int *sendcnts,
                             const int *sdispls,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcnts,
                             const int *rdispls,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


/* Architecture detection tuning */
int MV2_set_alltoallv_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by bcast tuning table*/
void MV2_cleanup_alltoallv_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune bcast thresholds */
int MV2_Alltoallv_is_define(char *mv2_user_alltoallv);


#endif

