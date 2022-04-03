#define GEN2_CMA__AMD_EPYC_ROME__1PPN {		\
	{		\
	2,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{64, &MPIR_Bcast_binomial_MV2, -1},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{256, &MPIR_Bcast_binomial_MV2, -1},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{1024, &MPIR_Bcast_binomial_MV2, -1},		\
	{2048, &MPIR_Bcast_binomial_MV2, -1},		\
	{4096, &MPIR_Bcast_binomial_MV2, -1},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{524288, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{1048576, &MPIR_Bcast_binomial_MV2, -1}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{64, &MPIR_Shmem_Bcast_MV2, -1},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{256, &MPIR_Shmem_Bcast_MV2, -1},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{1024, &MPIR_Shmem_Bcast_MV2, -1},		\
	{2048, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4096, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, -1}		\
	}		\
	}		 \
}
