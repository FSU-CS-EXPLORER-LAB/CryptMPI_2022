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

#define GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN { \
	{		\
	2,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_redscat_gather_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_redscat_gather_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_binomial_MV2},		\
	{8192, &MPIR_Reduce_redscat_gather_MV2},		\
	{16384, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{32768, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{65536, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{131072, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{262144, &MPIR_Reduce_inter_knomial_wrapper_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_shmem_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_shmem_MV2},		\
	{1024, &MPIR_Reduce_shmem_MV2},		\
	{2048, &MPIR_Reduce_shmem_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_shmem_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	},		\
	{		\
	4,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{4, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_redscat_gather_MV2},		\
	{32, &MPIR_Reduce_binomial_MV2},		\
	{64, &MPIR_Reduce_redscat_gather_MV2},		\
	{128, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_redscat_gather_MV2},		\
	{1024, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_redscat_gather_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_binomial_MV2},		\
	{16384, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{32768, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_binomial_MV2},		\
	{16, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{32, &MPIR_Reduce_binomial_MV2},		\
	{64, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_binomial_MV2},		\
	{1024, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_shmem_MV2},		\
	{131072, &MPIR_Reduce_shmem_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	},		\
	{		\
	8,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_redscat_gather_MV2},		\
	{4, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_redscat_gather_MV2},		\
	{32, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_redscat_gather_MV2},		\
	{1024, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_redscat_gather_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_binomial_MV2},		\
	{16384, &MPIR_Reduce_binomial_MV2},		\
	{32768, &MPIR_Reduce_binomial_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{8, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{16, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{32, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_shmem_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_shmem_MV2},		\
	{131072, &MPIR_Reduce_shmem_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	},		\
	{		\
	16,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_redscat_gather_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_binomial_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_redscat_gather_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{1024, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_redscat_gather_MV2},		\
	{16384, &MPIR_Reduce_binomial_MV2},		\
	{32768, &MPIR_Reduce_redscat_gather_MV2},		\
	{65536, &MPIR_Reduce_redscat_gather_MV2},		\
	{131072, &MPIR_Reduce_redscat_gather_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_binomial_MV2},		\
	{2, &MPIR_Reduce_shmem_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_binomial_MV2},		\
	{16, &MPIR_Reduce_shmem_MV2},		\
	{32, &MPIR_Reduce_shmem_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_shmem_MV2},		\
	{256, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_binomial_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	},		\
	{		\
	32,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_redscat_gather_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_redscat_gather_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_redscat_gather_MV2},		\
	{512, &MPIR_Reduce_redscat_gather_MV2},		\
	{1024, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_redscat_gather_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_binomial_MV2},		\
	{16384, &MPIR_Reduce_binomial_MV2},		\
	{32768, &MPIR_Reduce_redscat_gather_MV2},		\
	{65536, &MPIR_Reduce_redscat_gather_MV2},		\
	{131072, &MPIR_Reduce_redscat_gather_MV2},		\
	{262144, &MPIR_Reduce_redscat_gather_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_binomial_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_shmem_MV2},		\
	{512, &MPIR_Reduce_binomial_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	}		\
	},		\
	{		\
	64,		\
	4,		\
	4,		\
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{4, &MPIR_Reduce_redscat_gather_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_redscat_gather_MV2},		\
	{32, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_redscat_gather_MV2},		\
	{128, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_binomial_MV2},		\
	{16384, &MPIR_Reduce_binomial_MV2},		\
	{32768, &MPIR_Reduce_binomial_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_shmem_MV2},		\
	{4, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{8, &MPIR_Reduce_binomial_MV2},		\
	{16, &MPIR_Reduce_binomial_MV2},		\
	{32, &MPIR_Reduce_binomial_MV2},		\
	{64, &MPIR_Reduce_binomial_MV2},		\
	{128, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{512, &MPIR_Reduce_shmem_MV2},		\
	{1024, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{2048, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_shmem_MV2},		\
	{131072, &MPIR_Reduce_shmem_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	},		\
	{		\
	128,		\
	4,		\
	4,		\
	{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8, &MPIR_Reduce_redscat_gather_MV2},		\
	{16, &MPIR_Reduce_redscat_gather_MV2},		\
	{32, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{64, &MPIR_Reduce_redscat_gather_MV2},		\
	{128, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{256, &MPIR_Reduce_redscat_gather_MV2},		\
	{512, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{1024, &MPIR_Reduce_redscat_gather_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_inter_knomial_wrapper_MV2},		\
	{8192, &MPIR_Reduce_binomial_MV2},		\
	{16384, &MPIR_Reduce_binomial_MV2},		\
	{32768, &MPIR_Reduce_binomial_MV2},		\
	{65536, &MPIR_Reduce_binomial_MV2},		\
	{131072, &MPIR_Reduce_binomial_MV2},		\
	{262144, &MPIR_Reduce_binomial_MV2}		\
	},		\
	18,		\
	{		\
	{1, &MPIR_Reduce_shmem_MV2},		\
	{2, &MPIR_Reduce_binomial_MV2},		\
	{4, &MPIR_Reduce_binomial_MV2},		\
	{8, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{16, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{32, &MPIR_Reduce_binomial_MV2},		\
	{64, &MPIR_Reduce_intra_knomial_wrapper_MV2},		\
	{128, &MPIR_Reduce_binomial_MV2},		\
	{256, &MPIR_Reduce_binomial_MV2},		\
	{512, &MPIR_Reduce_binomial_MV2},		\
	{1024, &MPIR_Reduce_binomial_MV2},		\
	{2048, &MPIR_Reduce_binomial_MV2},		\
	{4096, &MPIR_Reduce_shmem_MV2},		\
	{8192, &MPIR_Reduce_shmem_MV2},		\
	{16384, &MPIR_Reduce_shmem_MV2},		\
	{32768, &MPIR_Reduce_shmem_MV2},		\
	{65536, &MPIR_Reduce_shmem_MV2},		\
	{131072, &MPIR_Reduce_shmem_MV2},		\
	{262144, &MPIR_Reduce_shmem_MV2}		\
	}		\
	}		\
};
