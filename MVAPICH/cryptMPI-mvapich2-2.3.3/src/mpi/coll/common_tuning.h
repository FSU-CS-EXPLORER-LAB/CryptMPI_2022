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

#ifndef _COMMON_TUNING_
#define _COMMON_TUNING_

#define MV2_COLL_TUNING_SETUP_TABLE(_cname)                     \
    int *mv2_##_cname##_table_ppn_conf = NULL;                  \
    int mv2_##_cname##_num_ppn_conf = 1;                        \
    int *mv2_size_##_cname##_tuning_table = NULL;               \
    mv2_##_cname##_tuning_table                                 \
        **mv2_##_cname##_thresholds_table = NULL;               \
    int *mv2_##_cname##_indexed_table_ppn_conf = NULL;          \
    int mv2_##_cname##_indexed_num_ppn_conf = 1;                \
    int *mv2_size_##_cname##_indexed_tuning_table = NULL;       \
    mv2_##_cname##_indexed_tuning_table                         \
        **mv2_##_cname##_indexed_thresholds_table = NULL;

#define MV2_COLL_TUNING_START_TABLE(_cname, _nconf)                             \
{                                                                               \
    colls_arch_hca[_cname].arch_type = table_arch_tmp;                     \
    colls_arch_hca[_cname].hca_type = table_hca_tmp;                       \
    int idx = -1, nconf = _nconf;                                               \
    mv2_##_cname##_indexed_num_ppn_conf = nconf;                                \
    mv2_##_cname##_indexed_thresholds_table = MPIU_Malloc(                      \
        sizeof(mv2_##_cname##_indexed_tuning_table *) * nconf);                 \
    table_ptrs = MPIU_Malloc(                                                   \
        sizeof(mv2_##_cname##_indexed_tuning_table *) * nconf);                 \
    mv2_size_##_cname##_indexed_tuning_table = MPIU_Malloc(                     \
            sizeof(int) * nconf);                                               \
    mv2_##_cname##_indexed_table_ppn_conf = MPIU_Malloc(                        \
            sizeof(int) * nconf);

#define MV2_COLL_TUNING_ADD_CONF(_cname, _ppn, _size, _name)                    \
  ++idx;                                                                        \
  mv2_##_cname##_indexed_tuning_table tmp_##_cname##_ppn[] = _name;             \
  mv2_##_cname##_indexed_table_ppn_conf[idx] = _ppn;                            \
  mv2_size_##_cname##_indexed_tuning_table[idx] = _size;                        \
  table_ptrs[idx] = tmp_##_cname##_ppn;                                         \
    

#if defined(_SMP_CMA_)
#define MV2_COLL_TUNING_ADD_CONF_CMA(_cname, _ppn, _size, _name)                \
  mv2_##_cname##_indexed_tuning_table tmp_cma_##_cname##_ppn[] = _name;         \
  if (g_smp_use_cma) {                                                          \
    mv2_##_cname##_indexed_table_ppn_conf[idx] = _ppn;                          \
    mv2_size_##_cname##_indexed_tuning_table[idx] = _size;                      \
    table_ptrs[idx] = tmp_cma_##_cname##_ppn;                                   \
  }
#else
#define MV2_COLL_TUNING_ADD_CONF_CMA(_cname, _ppn, _size, _name)
#endif

#define MV2_COLL_TUNING_FINISH_TABLE(_cname)                        \
    agg_table_sum = 0;                                              \
    for (i = 0; i < nconf; i++) {                                   \
        agg_table_sum +=                                            \
            mv2_size_##_cname##_indexed_tuning_table[i];            \
    }                                                               \
    mv2_##_cname##_indexed_thresholds_table[0] = MPIU_Malloc(       \
        sizeof (mv2_##_cname##_indexed_tuning_table) *              \
        agg_table_sum);                                             \
    MPIU_Memcpy(mv2_##_cname##_indexed_thresholds_table[0],         \
        table_ptrs[0],                                              \
        sizeof(mv2_##_cname##_indexed_tuning_table) *               \
        mv2_size_##_cname##_indexed_tuning_table[0]);               \
    for (i = 1; i < nconf; i++) {                                   \
        mv2_##_cname##_indexed_thresholds_table[i] =                \
            mv2_##_cname##_indexed_thresholds_table[i - 1]          \
            + mv2_size_##_cname##_indexed_tuning_table[i - 1];      \
        MPIU_Memcpy(mv2_##_cname##_indexed_thresholds_table[i],     \
            table_ptrs[i],                                          \
            sizeof(mv2_##_cname##_indexed_tuning_table) *           \
            mv2_size_##_cname##_indexed_tuning_table[i]);           \
    }                                                               \
    MPIU_Free(table_ptrs);                                          \
    return 0;                                                       \
}

#define FIND_PPN_INDEX(_cname, _locsize,_confindx,_parflag)             \
{                                                                       \
    int i=0;                                                            \
    do {                                                                \
        if (_locsize == mv2_##_cname##_indexed_table_ppn_conf[i]) {     \
            _confindx=i;                                                \
            _parflag=1;                                                 \
            break;                                                      \
        } else if (i < mv2_##_cname##_indexed_num_ppn_conf-1) {         \
            if (_locsize > mv2_##_cname##_indexed_table_ppn_conf[i] &&  \
                _locsize < mv2_##_cname##_indexed_table_ppn_conf[i+1]) { \
                _confindx=i+1;                                          \
                _parflag=1;                                             \
                break;                                                  \
            }                                                           \
        } else if (i == mv2_##_cname##_indexed_num_ppn_conf-1) {        \
            if (_locsize > mv2_##_cname##_indexed_table_ppn_conf[i]) {  \
                _confindx=i;                                            \
                _parflag=1;                                             \
                break;                                                  \
            }                                                           \
        }                                                               \
        i++;                                                            \
    } while(i < mv2_##_cname##_indexed_num_ppn_conf);                   \
}                                                                       

/* defined enum for right hand side values used in mv2 collective algorithms
 selection. */
enum mv2_bcast_tuning
{
    SHMEM_BCAST_INTRA,
    KNOMIAL_BCAST_INTRA,
    BCAST_BIONOMIAL_INTRA,
    BCAST_SCATTER_DOUBLING_ALLGATHER_FLAT,
    BCAST_SCATTER_DOUBLING_ALLGATHER,
    BCAST_SCATTER_RING_ALLGATEHR_FLAT,
    BCAST_SCATTER_RING_ALLGATHER,
    BCAST_SCATTER_RING_ALLGATHER_SHM, 
    KNOMIAL_BCAST_INTER_NODE_WRAPPER,
    PIPELINED_BCAST,
    PIPELINED_BCAST_ZCPY
};

enum mv2_reduce_tuning 
{
    REDUCE_BINOMIAL = 1,
    REDUCE_INTER_KNOMIAL,
    REDUCE_INTRA_KNOMIAL,
    REDUCE_SHMEM,
    REDUCE_RDSC_GATHER,
    REDUCE_ZCPY
};

enum mv2_gather_tuning 
{
    GATHER_INTRA,
    GATHER_INTER,
    GATHER_MV2_DIRECT,
    GATHER_MV2_TWO_LEVEL_DIRECT,
    /* added by abu naser */
    GATHER_CHS,
    GATHER_RR
    /* end of add */
};

enum mv2_allreduce_tuning 
{
    ALLREDUCE_P2P_RD = 1,
    ALLREDUCE_P2P_RS,
    ALLREDUCE_MCAST_2LEVEL,
    ALLREDUCE_MCAST_RSA,
    ALLREDUCE_SHMEM_REDUCE,
    ALLREDUCE_P2P_REDUCE,
    ALLREDUCE_RED_SCAT_ALLGA_COLL
};

enum mv2_scatter_tuning 
{
    SCATTER_BINOMIAL = 1,
    SCATTER_DIRECT,
    SCATTER_TWO_LEVEL_BINOMIAL,
    SCATTER_TWO_LEVEL_DIRECT,
    SCATTER_MCAST,
    /* added by abu naser */
    SCATTER_CHS,
    SCATTER_RR,
    SCATTER_RR_UNEN,
    SCATTER_HBCAST,
    SCATTTER_C
    /* end of add */

};

enum mv2_allgather_tuning 
{
    ALLGATHER_RD_ALLGATHER_COMM = 1,
    ALLGATHER_RD,
    ALLGATHER_BRUCK,
    ALLGATHER_RING,
	ALLGATHER_DIRECT,
	ALLGATHER_DIRECTSPREAD,
	ALLGATHER_GATHER_BCAST,
	ALLGATHER_2LVL_NONBLOCKED,
	ALLGATHER_2LVL_RING_NONBLOCKED,
	ALLGATHER_2LVL_DIRECT,
	ALLGATHER_2LVL_RING,
     /****** Added by Mehran *****/
    ALLGATHER_2LVL_MULTILEADER_RING, //12
    ALLGATHER_2LVL_MULTILEADER_RD,
    ALLGATHER_2LVL_SHMEM,
    ALLGATHER_ENC_RDB,
    ALLGATHER_2LVL_ENC_RDB,
    ALLGATHER_NP_RDB,
    ALLGATHER_2LVL_SHMEM_CONCURRENT_ENCRYPTION,
    CONCURRENT_ALLGATHER,//19
    ALLGATHER_CONCURRENT_MULTILEADER_SHMEM
};

enum mv2_alltoall_tuning 
{
    ALLTOALL_BRUCK_MV2,
    ALLTOALL_RD_MV2,
    ALLTOALL_SCATTER_DEST_MV2,
    ALLTOALL_PAIRWISE_MV2,
    ALLTOALL_INPLACE_MV2
};

enum mv2_alltoallv_tuning
{
    ALLTOALLV_INTRA_SCATTER_MV2,
    ALLTOALLV_INTRA_MV2,
    ALLTOALLV_MV2
};
#endif
