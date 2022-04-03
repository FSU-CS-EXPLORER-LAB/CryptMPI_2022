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

#ifndef _GATHER_TUNING_
#define _GATHER_TUNING_

#include "coll_shmem.h"
#if defined(CHANNEL_MRAIL)
#include "ibv_param.h"
#endif /* #if defined(CHANNEL_MRAIL) */
#define MV2_DEFAULT_SHMEM_BCAST_LEADERS    4096
#define MV2_GATHER_DIRECT_SYSTEM_SIZE_SMALL      384
#define MV2_GATHER_DIRECT_SYSTEM_SIZE_MEDIUM     1024
#define MPIR_GATHER_BINOMIAL_MEDIUM_MSG 16384

#define NMATCH (3+1)

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Gather_function)(const void *sendbuf, int sendcnt,
                                  MPI_Datatype sendtype, void *recvbuf, int recvcnt,
                                  MPI_Datatype recvtype, int root, MPID_Comm * comm_ptr,
                                  MPIR_Errflag_t *errflag);
} mv2_gather_tuning_element;

#if defined(_SMP_LIMIC_)

typedef enum limic_gather_algo
{
    USE_GATHER_PT_PT_BINOMIAL      = 1,
    USE_GATHER_PT_PT_DIRECT        = 2,
    USE_GATHER_PT_LINEAR_BINOMIAL  = 3,
    USE_GATHER_PT_LINEAR_DIRECT    = 4,
    USE_GATHER_LINEAR_PT_BINOMIAL  = 5,
    USE_GATHER_LINEAR_PT_DIRECT    = 6,
    USE_GATHER_LINEAR_LINEAR       = 7,
    USE_GATHER_SINGLE_LEADER       = 8

} limic_gather_algo;

typedef struct {
    int min;
    int max;
    limic_gather_algo scheme;
} list_limic_scheme;

#endif /*#if defined(_SMP_LIMIC_)*/
typedef struct {
    int numproc;
    int size_inter_table;
    mv2_gather_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_gather_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
#if defined(_SMP_LIMIC_)
    int nb_limic_scheme;
    list_limic_scheme limic_gather_scheme[MV2_MAX_NB_THRESHOLDS];
#endif /*#if defined(_SMP_LIMIC_)*/
} mv2_gather_tuning_table;

extern int mv2_size_gather_tuning_table;
extern mv2_gather_tuning_table * mv2_gather_thresholds_table;

extern int mv2_user_gather_switch_point;
extern int mv2_use_two_level_gather;
extern int mv2_gather_direct_system_size_small;
extern int mv2_gather_direct_system_size_medium;
extern int mv2_use_direct_gather;

/*Entries related to indexed tuning table*/

typedef struct {
  int msg_sz;
    int (*MV2_pt_Gather_function)(const void *sendbuf, int sendcnt,
                                  MPI_Datatype sendtype, void *recvbuf, int recvcnt,
                                  MPI_Datatype recvtype, int root, MPID_Comm * comm_ptr,
                                  MPIR_Errflag_t *errflag);
} mv2_gather_indexed_tuning_element;

typedef struct {
    int numproc;
    int size_inter_table;
    mv2_gather_indexed_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
    int size_intra_table;
    mv2_gather_indexed_tuning_element intra_node[MV2_MAX_NB_THRESHOLDS];
} mv2_gather_indexed_tuning_table;

/* Indicates number of processes per node */
extern int *mv2_gather_indexed_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_gather_indexed_num_ppn_conf;
extern int *mv2_size_gather_indexed_tuning_table;
extern mv2_gather_indexed_tuning_table **mv2_gather_indexed_thresholds_table;

extern int MPIR_Gather_MV2_Direct_Blk(const void *sendbuf, int sendcnt,
                                      MPI_Datatype sendtype, void *recvbuf, int recvcnt,
                                      MPI_Datatype recvtype, int root, MPID_Comm * comm_ptr,
                                      MPIR_Errflag_t *errflag);

extern int MPIR_Gather_MV2_Direct(const void *sendbuf, int sendcnt,
                                      MPI_Datatype sendtype, void *recvbuf, int recvcnt,
                                      MPI_Datatype recvtype, int root, MPID_Comm * comm_ptr,
                                      MPIR_Errflag_t *errflag);
extern int MPIR_Gather_MV2_two_level_Direct(const void *sendbuf, int sendcnt,
            MPI_Datatype sendtype, void *recvbuf, int recvcnt,
            MPI_Datatype recvtype, int root, MPID_Comm * comm_ptr,
            MPIR_Errflag_t *errflag);

#if defined(_SMP_LIMIC_)
extern int MPIR_Intra_node_LIMIC_Gather_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                                     void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                                     int root, MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);
#endif /*#if defined(_SMP_LIMIC_)*/

/* Architecture detection tuning */
int MV2_set_gather_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);
void MV2_cleanup_gather_tuning_table(); 

/* Function used inside ch3_shmem_coll.c to tune gather thresholds */
int MV2_internode_Gather_is_define(char *mv2_user_gather_inter, char *mv2_user_gather_intra);
int MV2_intranode_Gather_is_define(char *mv2_user_gather_intra);
int MV2_intranode_multi_lvl_Gather_is_define(char *mv2_user_gather_inter,
                                             char *mv2_user_gather_intra,
                                             char *mv2_user_gather_intra_multi_lvl);

void MV2_user_gather_switch_point_is_define(int mv2_user_gather_switch_point);

#endif
