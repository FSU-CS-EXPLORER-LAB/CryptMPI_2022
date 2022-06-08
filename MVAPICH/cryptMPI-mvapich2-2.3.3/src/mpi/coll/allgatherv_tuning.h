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

#ifndef _ALLGATHERV_TUNING_
#define _ALLGATHERV_TUNING_

#include "coll_shmem.h"

#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Allgatherv tuning flags
 * recursive doubling: MV2_INTER_ALLGATHERV_TUNING=1 
 * bruck:              MV2_INTER_ALLGATHERV_TUNING=2
 * ring:               MV2_INTER_ALLGATHERV_TUNING=3
 * ring_cyclic:        MV2_INTER_ALLGATHERV_TUNING=4
 * Regular expression example:
 *   MV2_INTER_ALLGATHERV_TUNING=2:0-1024,1:1024-8192,3:8192-+
 *   meaning: use bruck for 0 byte to 1024 bytes
 *            use recursive doubling for 1024 byte to 8192 bytes
 *            use ring since 8192 bytes
 */

char *mv2_user_allgatherv_inter;

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Allgatherv_function)(const void *sendbuf,
                                      int sendcount,
                                      MPI_Datatype sendtype,
                                      void *recvbuf,
                                      const int *recvcounts,
                                      const int *displs,
                                      MPI_Datatype recvtype,
                                      MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
} mv2_allgatherv_tuning_element;

typedef struct {
    int numproc; 
    int size_inter_table;
    mv2_allgatherv_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
} mv2_allgatherv_tuning_table;

extern int mv2_size_allgatherv_tuning_table;
extern mv2_allgatherv_tuning_table *mv2_allgatherv_thresholds_table;

extern int MPIR_Allgatherv_Rec_Doubling_MV2(const void *sendbuf,
                                            int sendcount,
                                            MPI_Datatype sendtype,
                                            void *recvbuf,
                                            const int *recvcounts,
                                            const int *displs,
                                            MPI_Datatype recvtype,
                                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Allgatherv_Bruck_MV2(const void *sendbuf,
                                     int sendcount,
                                     MPI_Datatype sendtype,
                                     void *recvbuf,
                                     const int *recvcounts,
                                     const int *displs,
                                     MPI_Datatype recvtype,
                                     MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Allgatherv_Ring_MV2(const void *sendbuf,
                                    int sendcount,
                                    MPI_Datatype sendtype,
                                    void *recvbuf,
                                    const int *recvcounts,
                                    const int *displs,
                                    MPI_Datatype recvtype,
                                    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

extern int MPIR_Allgatherv_Ring_Cyclic_MV2(const void *sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             const int *recvcounts,
                             const int *displs,
                             MPI_Datatype recvtype,
                             MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

/* Architecture detection tuning */
int MV2_set_allgatherv_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by allgatherv tuning table*/
void MV2_cleanup_allgatherv_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune allgatherv thresholds */
int MV2_internode_Allgatherv_is_define(char *mv2_user_allgatherv_inter);
#endif
