#define GEN2__RI__4PPN {		\
	{		\
	4,		\
	8192,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1},		\
	21,		\
	{		\
	{1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},		\
	{2, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{4, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{8, &MPIR_Bcast_binomial_MV2, -1},		\
	{16, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{32, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{64, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{128, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{256, &MPIR_Bcast_binomial_MV2, -1},		\
	{512, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{1024, &MPIR_Bcast_binomial_MV2, -1},		\
	{2048, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{4096, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},		\
	{8192, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{16384, &MPIR_Bcast_binomial_MV2, -1},		\
	{32768, &MPIR_Bcast_binomial_MV2, -1},		\
	{65536, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{131072, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{262144, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{524288, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},		\
	{1048576, &MPIR_Bcast_binomial_MV2, -1}		\
	},		\
	21,		\
	{		\
	{1, &MPIR_Knomial_Bcast_intra_node_MV2, 4},		\
	{2, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8, &MPIR_Shmem_Bcast_MV2, -1},		\
	{16, &MPIR_Shmem_Bcast_MV2, -1},		\
	{32, &MPIR_Shmem_Bcast_MV2, -1},		\
	{64, &MPIR_Shmem_Bcast_MV2, -1},		\
	{128, &MPIR_Shmem_Bcast_MV2, -1},		\
	{256, &MPIR_Shmem_Bcast_MV2, -1},		\
	{512, &MPIR_Shmem_Bcast_MV2, -1},		\
	{1024, &MPIR_Shmem_Bcast_MV2, -1},		\
	{2048, &MPIR_Shmem_Bcast_MV2, -1},		\
	{4096, &MPIR_Shmem_Bcast_MV2, -1},		\
	{8192, &MPIR_Shmem_Bcast_MV2, -1},		\
	{16384, &MPIR_Shmem_Bcast_MV2, -1},		\
	{32768, &MPIR_Shmem_Bcast_MV2, -1},		\
	{65536, &MPIR_Shmem_Bcast_MV2, -1},		\
	{131072, &MPIR_Shmem_Bcast_MV2, -1},		\
	{262144, &MPIR_Shmem_Bcast_MV2, -1},		\
	{524288, &MPIR_Shmem_Bcast_MV2, -1},		\
	{1048576, &MPIR_Shmem_Bcast_MV2, -1}		\
	}		\
	}		 \
}
