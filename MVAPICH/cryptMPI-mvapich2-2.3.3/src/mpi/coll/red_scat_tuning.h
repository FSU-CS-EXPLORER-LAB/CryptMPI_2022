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

#ifndef _RED_SCAT_TUNING_
#define _RED_SCAT_TUNING_

#include "coll_shmem.h"
#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Red_scat tuning flags 
 * RED_SCAT_BASIC:    MV2_INTER_RED_SCAT_TUNING=1
 * RED_SCAT_REC_HALF: MV2_INTER_RED_SCAT_TUNING=2
 * RED_SCAT_PAIRWISE: MV2_INTER_RED_SCAT_TUNING=3
 *
 * Regular expression example:
 *   MV2_INTER_RED_SCAT_TUNING=2:0-1024,1:1024-8192,3:8192-+
 *   meaning: use RED_SCAT_REC_HALF for 0 byte to 1024 bytes
 *            use RED_SCAT_BASIC for 1024 byte to 8192 bytes
 *            use RED_SCAT_PAIRWISE since 8192 bytes
 *           
 */ 

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Red_scat_function)(const void *sendbuf,
                                    void *recvbuf,
                                    const int *recvcnts,
                                    MPI_Datatype datatype,
                                    MPI_Op op,
                                    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_red_scat_tuning_element;

typedef struct {
    int numproc; 
    int size_inter_table;
    mv2_red_scat_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
} mv2_red_scat_tuning_table;

extern int mv2_size_red_scat_tuning_table;
extern mv2_red_scat_tuning_table *mv2_red_scat_thresholds_table;

extern int MPIR_Reduce_Scatter_Basic_MV2(const void *sendbuf,
                                         void *recvbuf,
                                         const int *recvcnts,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_scatter_Rec_Halving_MV2(const void *sendbuf,
                                         void *recvbuf,
                                         const int *recvcnts,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_scatter_Pair_Wise_MV2(const void *sendbuf,
                                         void *recvbuf,
                                         const int *recvcnts,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_scatter_ring(const void *sendbuf,
                                         void *recvbuf,
                                         const int *recvcnts,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Reduce_scatter_ring_2lvl(const void *sendbuf,
                                         void *recvbuf,
                                         const int *recvcnts,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

/* Architecture detection tuning */
int MV2_set_red_scat_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by red_scat tuning table*/
void MV2_cleanup_red_scat_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune red_scat thresholds */
int MV2_internode_Red_scat_is_define(char *mv2_user_red_scat_inter);
#endif
