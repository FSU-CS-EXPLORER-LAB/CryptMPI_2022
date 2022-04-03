#define GEN2_CMA__RI2__28PPN {		\
	{		\
	28,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{32768, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},		\
	{65536, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},		\
	{131072, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},		\
	{524288, &MPIR_Bcast_binomial_MV2, -1},		\
	{1048576, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Shmem_Bcast_MV2, -1},		\
	{65536, &MPIR_Shmem_Bcast_MV2, -1},		\
	{131072, &MPIR_Shmem_Bcast_MV2, -1},		\
	{262144, &MPIR_Shmem_Bcast_MV2, -1},		\
	{524288, &MPIR_Shmem_Bcast_MV2, -1},		\
	{1048576, &MPIR_Shmem_Bcast_MV2, -1}		\
	}		\
	},		 \
	{		\
	56,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{2, &MPIR_Bcast_binomial_MV2, -1},		\
	{4, &MPIR_Bcast_binomial_MV2, -1},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{32, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{64, &MPIR_Bcast_binomial_MV2, -1},		\
	{128, &MPIR_Bcast_binomial_MV2, -1},		\
	{256, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{512, &MPIR_Bcast_binomial_MV2, -1},		\
	{1024, &MPIR_Bcast_binomial_MV2, -1},		\
	{2048, &MPIR_Bcast_binomial_MV2, -1},		\
	{4096, &MPIR_Bcast_binomial_MV2, -1},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16384, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Shmem_Bcast_MV2, -1},		\
	{2, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16, &MPIR_Shmem_Bcast_MV2, -1},		\
	{32, &MPIR_Shmem_Bcast_MV2, -1},		\
	{64, &MPIR_Shmem_Bcast_MV2, -1},		\
	{128, &MPIR_Shmem_Bcast_MV2, -1},		\
	{256, &MPIR_Shmem_Bcast_MV2, -1},		\
	{512, &MPIR_Shmem_Bcast_MV2, -1},		\
	{1024, &MPIR_Shmem_Bcast_MV2, -1},		\
	{2048, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4096, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		 \
	{		\
	112,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16384, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		 \
	{		\
	224,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
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
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
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
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		 \
	{		\
	448,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{128, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{128, &MPIR_Shmem_Bcast_MV2, -1},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	},		 \
	{		\
	896,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},		\
	{16384, &MPIR_Pipelined_Bcast_MV2, -1},		\
	{32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},		\
	{1048576, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{32, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{64, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{128, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{256, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{512, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{1024, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{2048, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{4096, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{8192, &MPIR_Knomial_Bcast_intra_node_MV2, 8},		\
	{16384, &MPIR_Knomial_Bcast_intra_node_MV2, -1},		\
	{32768, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{65536, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{262144, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},		\
	{1048576, &MPIR_Knomial_Bcast_intra_node_MV2, 2}		\
	}		\
	}		 \
}
