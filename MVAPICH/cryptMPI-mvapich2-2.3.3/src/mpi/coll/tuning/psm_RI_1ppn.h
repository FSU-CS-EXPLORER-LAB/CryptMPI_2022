/*
 * Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#define PSM__RI__1PPN {		\
      {\
        2,\
        8192,\
        4,\
        4,\
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},\
        20,\
        {\
          {1, &MPIR_Pipelined_Bcast_MV2, -1},\
          {2, &MPIR_Pipelined_Bcast_MV2, -1},\
          {4, &MPIR_Pipelined_Bcast_MV2, -1},\
          {8, &MPIR_Pipelined_Bcast_MV2, -1},\
          {16, &MPIR_Bcast_binomial_MV2, -1},\
          {32, &MPIR_Bcast_binomial_MV2, -1},\
          {64, &MPIR_Pipelined_Bcast_MV2, -1},\
          {128, &MPIR_Bcast_binomial_MV2, -1},\
          {256, &MPIR_Pipelined_Bcast_MV2, -1},\
          {512, &MPIR_Pipelined_Bcast_MV2, -1},\
          {1024, &MPIR_Pipelined_Bcast_MV2, -1},\
          {2048, &MPIR_Pipelined_Bcast_MV2, -1},\
          {4096, &MPIR_Pipelined_Bcast_MV2, -1},\
          {8192, &MPIR_Bcast_binomial_MV2, -1},\
          {16384, &MPIR_Bcast_binomial_MV2, -1},\
          {32768, &MPIR_Pipelined_Bcast_MV2, -1},\
          {65536, &MPIR_Pipelined_Bcast_MV2, -1},\
          {131072, &MPIR_Pipelined_Bcast_MV2, -1},\
          {262144, &MPIR_Pipelined_Bcast_MV2, -1},\
          {524288, &MPIR_Pipelined_Bcast_MV2, -1},\
          {1048576, &MPIR_Pipelined_Bcast_MV2, -1}\
        },\
        20,\
        {\
          {1, &MPIR_Shmem_Bcast_MV2, -1},\
          {2, &MPIR_Shmem_Bcast_MV2, -1},\
          {4, &MPIR_Shmem_Bcast_MV2, -1},\
          {8, &MPIR_Shmem_Bcast_MV2, -1},\
          {16, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},\
          {64, &MPIR_Shmem_Bcast_MV2, -1},\
          {128, &MPIR_Knomial_Bcast_intra_node_MV2, 4},\
          {256, &MPIR_Shmem_Bcast_MV2, -1},\
          {512, &MPIR_Shmem_Bcast_MV2, -1},\
          {1024, &MPIR_Shmem_Bcast_MV2, -1},\
          {2048, &MPIR_Shmem_Bcast_MV2, -1},\
          {4096, &MPIR_Shmem_Bcast_MV2, -1},\
          {8192, &MPIR_Knomial_Bcast_intra_node_MV2, 4},\
          {16384, &MPIR_Knomial_Bcast_intra_node_MV2, 4},\
          {32768, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {65536, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {131072, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {262144, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {524288, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {1048576, &MPIR_Shmem_Bcast_MV2, -1}\
        }\
      },\
      {\
        4,\
        8192,\
        4,\
        4,\
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},\
        20,\
        {\
          {1, &MPIR_Bcast_binomial_MV2, -1},\
          {2, &MPIR_Bcast_binomial_MV2, -1},\
          {4, &MPIR_Bcast_binomial_MV2, -1},\
          {8, &MPIR_Bcast_binomial_MV2, -1},\
          {16, &MPIR_Bcast_binomial_MV2, -1},\
          {32, &MPIR_Pipelined_Bcast_MV2, -1},\
          {64, &MPIR_Bcast_binomial_MV2, -1},\
          {128, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {256, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {512, &MPIR_Bcast_binomial_MV2, -1},\
          {1024, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {4096, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},\
          {16384, &MPIR_Pipelined_Bcast_MV2, -1},\
          {32768, &MPIR_Pipelined_Bcast_MV2, -1},\
          {65536, &MPIR_Pipelined_Bcast_MV2, -1},\
          {131072, &MPIR_Pipelined_Bcast_MV2, -1},\
          {262144, &MPIR_Pipelined_Bcast_MV2, -1},\
          {524288, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},\
          {1048576, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}\
        },\
        20,\
        {\
          {1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},\
          {2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},\
          {4, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {8, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {16, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {32, &MPIR_Shmem_Bcast_MV2, -1},\
          {64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {128, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {256, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {512, &MPIR_Knomial_Bcast_intra_node_MV2, 2},\
          {1024, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {2048, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {4096, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {8192, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {32768, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {65536, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {131072, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {262144, &MPIR_Knomial_Bcast_intra_node_MV2, -1},\
          {524288, &MPIR_Shmem_Bcast_MV2, -1},\
          {1048576, &MPIR_Shmem_Bcast_MV2, -1}\
        }\
      }\
};
