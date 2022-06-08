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

#ifndef _ALLGATHER_TUNING_
#define _ALLGATHER_TUNING_

#include "coll_shmem.h"
#ifdef _OSU_MVAPICH_
#   include "ib_param.h"
#endif /* _OSU_MVAPICH_ */

#ifdef CHANNEL_MRAIL_GEN2
#   include "ibv_param.h"
#endif

#define NMATCH (3+1)

/* Allgather tuning flags 
 * recursive doubling with allgather_comm: MV2_INTER_ALLGATHER_TUNING=1 
 * recursive doubling: MV2_INTER_ALLGATHER_TUNING=2
 * bruck: MV2_INTER_ALLGATHER_TUNING=3
 * ring: MV2_INTER_ALLGATHER_TUNING=4
 * 2-level recursive doubling:  MV2_INTER_ALLGATHER_TUNING=2 
 *                              MV2_INTER_ALLGATHER_TUNING_TWO_LEVEL=1
 * 2-level bruck: MV2_INTER_ALLGATHER_TUNING=3
 *                MV2_INTER_ALLGATHER_TUNING_TWO_LEVEL=1
 * 2-level ring:  MV2_INTER_ALLGATHER_TUNING=4
 *                MV2_INTER_ALLGATHER_TUNING_TWO_LEVEL=1
 */
/* Indicates number of processes per node */
extern int *mv2_allgather_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_allgather_num_ppn_conf;

typedef struct {
    int min;
    int max;
    int (*MV2_pt_Allgather_function)(const void *sendbuf,
                                 int sendcount,
                                 MPI_Datatype sendtype,
                                 void *recvbuf,
                                 int recvcount,
                                 MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                 MPIR_Errflag_t *errflag);
} mv2_allgather_tuning_element;

typedef struct {
    int numproc; 
    int two_level[MV2_MAX_NB_THRESHOLDS];
    int size_inter_table;
    mv2_allgather_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
} mv2_allgather_tuning_table;

extern int *mv2_size_allgather_tuning_table;
extern mv2_allgather_tuning_table **mv2_allgather_thresholds_table;
extern int mv2_use_old_allgather;

typedef struct {
    int msg_sz;
    int (*MV2_pt_Allgather_function)(const void *sendbuf,
                                 int sendcount,
                                 MPI_Datatype sendtype,
                                 void *recvbuf,
                                 int recvcount,
                                 MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                 MPIR_Errflag_t *errflag);
} mv2_allgather_indexed_tuning_element;

typedef struct {
    int numproc; 
    int size_inter_table;
    mv2_allgather_indexed_tuning_element inter_leader[MV2_MAX_NB_THRESHOLDS];
} mv2_allgather_indexed_tuning_table;

/* Indicates number of processes per node */
extern int *mv2_allgather_indexed_table_ppn_conf;
/* Indicates total number of configurations */
extern int mv2_allgather_indexed_num_ppn_conf;
extern int *mv2_size_allgather_indexed_tuning_table;
extern mv2_allgather_indexed_tuning_table **mv2_allgather_indexed_thresholds_table;

extern int MPIR_Allgather_RD_Allgather_Comm_MV2(const void *sendbuf,
											    int sendcount,
											    MPI_Datatype sendtype,
											    void *recvbuf,
											    int recvcount,
											    MPI_Datatype recvtype, MPID_Comm * comm_ptr,
											    MPIR_Errflag_t *errflag);

extern int MPIR_Allgather_RD_MV2(const void *sendbuf,
                                 int sendcount,
                                 MPI_Datatype sendtype,
                                 void *recvbuf,
                                 int recvcount,
                                 MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                 MPIR_Errflag_t *errflag);

extern int MPIR_Allgather_Bruck_MV2(const void *sendbuf,
                                    int sendcount,
                                    MPI_Datatype sendtype,
                                    void *recvbuf,
                                    int recvcount,
                                    MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                    MPIR_Errflag_t *errflag);

extern int MPIR_Allgather_Ring_MV2(const void *sendbuf,
                                   int sendcount,
                                   MPI_Datatype sendtype,
                                   void *recvbuf,
                                   int recvcount,
                                   MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                                   MPIR_Errflag_t *errflag);

extern int MPIR_2lvl_Allgather_MV2(const void *sendbuf,
	     					       int sendcnt,
	     					       MPI_Datatype sendtype,
	     					       void *recvbuf,
	     					       int recvcnt,
	     					       MPI_Datatype recvtype,
	     					       MPID_Comm * comm_ptr,
	     					       MPIR_Errflag_t *errflag);


int MPIR_Concurrent_Allgather_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


int MPIR_Allgather_Direct_MV2(const void *sendbuf,
							  int sendcnt,
							  MPI_Datatype sendtype,
							  void *recvbuf,
							  int recvcnt,
							  MPI_Datatype recvtype,
							  MPID_Comm * comm_ptr,
	                          MPIR_Errflag_t *errflag);

int MPIR_Allgather_DirectSpread_MV2(const void *sendbuf,
									int sendcnt,
									MPI_Datatype sendtype,
									void *recvbuf,
									int recvcnt,
									MPI_Datatype recvtype,
									MPID_Comm * comm_ptr,
									MPIR_Errflag_t *errflag);

int MPIR_Allgather_gather_bcast_MV2(const void *sendbuf,
									int sendcount,
									MPI_Datatype sendtype,
									void *recvbuf,
									int recvcount,
									MPI_Datatype recvtype,
									MPID_Comm * comm_ptr,
									MPIR_Errflag_t *errflag);

int MPIR_2lvl_Allgather_nonblocked_MV2(const void *sendbuf,
								       int sendcnt,
								       MPI_Datatype sendtype,
								       void *recvbuf,
								       int recvcnt,
								       MPI_Datatype recvtype,
								       MPID_Comm * comm_ptr,
								       MPIR_Errflag_t *errflag);

int MPIR_2lvl_Allgather_Ring_nonblocked_MV2(const void *sendbuf,
											int sendcount,
											MPI_Datatype sendtype,
											void *recvbuf, int recvcount,
											MPI_Datatype recvtype,
											MPID_Comm * comm_ptr,
											MPIR_Errflag_t *errflag);

int MPIR_2lvl_Allgather_Multileader_Ring_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_2lvl_Allgather_Multileader_RD_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);


int MPIR_2lvl_Allgather_Multileader_RD_nonblocked_MV2(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
          void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_2lvl_SharedMem_Allgather_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_2lvl_Concurrent_Multileader_SharedMem_Allgather_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_Allgather_Encrypted_RDB_MV2(const void *sendbuf,
                          int sendcount,
                          MPI_Datatype sendtype,
                          void *recvbuf,
                          int recvcount,
                          MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                          MPIR_Errflag_t *errflag);

int MPIR_2lvl_Allgather_Encrypted_RDB_MV2(const void *sendbuf,int sendcnt, MPI_Datatype sendtype,
                            void *recvbuf, int recvcnt,MPI_Datatype recvtype,
                            MPID_Comm * comm_ptr, MPIR_Errflag_t *errflag);

int MPIR_Allgather_NaivePlus_RDB_MV2(const void *sendbuf,
                          int sendcount,
                          MPI_Datatype sendtype,
                          void *recvbuf,
                          int recvcount,
                          MPI_Datatype recvtype, MPID_Comm * comm_ptr,
                          MPIR_Errflag_t *errflag);
						  
int MPIR_2lvl_Allgather_Direct_MV2(const void *sendbuf,
							       int sendcnt,
								   MPI_Datatype sendtype,
							       void *recvbuf,
							       int recvcnt,
							       MPI_Datatype recvtype,
							       MPID_Comm *comm_ptr,
							       MPIR_Errflag_t *errflag);


int MPIR_2lvl_Allgather_Ring_MV2(const void *sendbuf,
								int sendcnt,
								MPI_Datatype sendtype,
								void *recvbuf,
								int recvcnt, MPI_Datatype recvtype,
								MPID_Comm *comm_ptr,
								MPIR_Errflag_t *errflag);

/* Architecture detection tuning */
int MV2_set_allgather_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca);

/* Function to clean free memory allocated by allgather tuning table*/
void MV2_cleanup_allgather_tuning_table();

/* Function used inside ch3_shmem_coll.c to tune allgather thresholds */
int MV2_internode_Allgather_is_define(char *mv2_user_allgather_inter);

#endif


