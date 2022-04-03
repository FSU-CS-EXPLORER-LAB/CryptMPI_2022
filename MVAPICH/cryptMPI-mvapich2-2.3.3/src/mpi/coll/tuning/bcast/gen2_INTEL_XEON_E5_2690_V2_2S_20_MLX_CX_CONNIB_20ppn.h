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

#define GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN { \
	{		\
	20,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	20,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{2, &MPIR_Bcast_binomial_MV2, -1},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4096, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Bcast_binomial_MV2, -1},		\
	{32768, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{65536, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{131072, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{262144, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{524288, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{1048576, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}		\
	},		\
	20,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{2, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8, &MPIR_Shmem_Bcast_MV2, -1},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4096, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Shmem_Bcast_MV2, -1},		\
	{32768, &MPIR_Shmem_Bcast_MV2, -1},		\
	{65536, &MPIR_Shmem_Bcast_MV2, -1},		\
	{131072, &MPIR_Shmem_Bcast_MV2, -1},		\
	{262144, &MPIR_Shmem_Bcast_MV2, -1},		\
	{524288, &MPIR_Shmem_Bcast_MV2, -1},		\
	{1048576, &MPIR_Shmem_Bcast_MV2, -1}		\
	}		\
	},		\
	{		\
	40,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	20,		\
	{		\
	{1, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	20,		\
	{		\
	{1, &MPIR_Shmem_Bcast_MV2, -1},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		\
	{		\
	80,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	20,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	20,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		\
	{		\
	160,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	20,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	20,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	}		\
};
