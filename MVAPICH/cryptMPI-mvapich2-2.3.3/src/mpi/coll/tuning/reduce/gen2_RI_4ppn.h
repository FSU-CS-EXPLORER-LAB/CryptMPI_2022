#define GEN2__RI__4PPN {		\
	{		\
	4,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1},		\
	19,		\
	{		\
	{4, &MPIR_Reduce_redscat_gather_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{32, &MPIR_Reduce_redscat_gather_MV2},		\
	{64, &MPIR_Reduce_redscat_gather_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_binomial_MV2},		\
	{1024, &MPIR_Reduce_redscat_gather_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_redscat_gather_MV2},		\
	{8192, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{16384, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{32768, &MPIR_Reduce_redscat_gather_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2},		\
	{524288, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{1048576, &MPIR_Reduce_binomial_MV2}		\
	},		\
	19,		\
	{		\
	{4, &MPIR_Reduce_shmem_MV2},		\
	{8, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{16, &MPIR_Reduce_shmem_MV2},		\
	{32, &MPIR_Reduce_shmem_MV2},		\
	{64, &MPIR_Reduce_shmem_MV2},		\
	{128, &MPIR_Reduce_shmem_MV2},		\
	{256, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_shmem_MV2},		\
	{1024, &MPIR_Reduce_shmem_MV2},		\
	{2048, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{4096, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_shmem_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{524288, &MPIR_Reduce_shmem_MV2},		\
	{1048576, &MPIR_Reduce_shmem_MV2}		\
	}		\
	}		 \
}
