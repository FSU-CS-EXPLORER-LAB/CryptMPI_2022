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

#define PSM__RI__2PPN {		\
	  {		\
	    2,		\
	    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},		\
	    20,		\
	    {		\
	      {1, &MPIR_Alltoall_pairwise_MV2},		\
	      {2, &MPIR_Alltoall_pairwise_MV2},		\
	      {4, &MPIR_Alltoall_pairwise_MV2},		\
	      {8, &MPIR_Alltoall_pairwise_MV2},		\
	      {16, &MPIR_Alltoall_pairwise_MV2},		\
	      {32, &MPIR_Alltoall_pairwise_MV2},		\
	      {64, &MPIR_Alltoall_pairwise_MV2},		\
	      {128, &MPIR_Alltoall_pairwise_MV2},		\
	      {256, &MPIR_Alltoall_pairwise_MV2},		\
	      {512, &MPIR_Alltoall_pairwise_MV2},		\
	      {1024, &MPIR_Alltoall_pairwise_MV2},		\
	      {2048, &MPIR_Alltoall_pairwise_MV2},		\
	      {4096, &MPIR_Alltoall_pairwise_MV2},		\
	      {8192, &MPIR_Alltoall_pairwise_MV2},		\
	      {16384, &MPIR_Alltoall_pairwise_MV2},		\
	      {32768, &MPIR_Alltoall_pairwise_MV2},		\
	      {65536, &MPIR_Alltoall_pairwise_MV2},		\
	      {131072, &MPIR_Alltoall_pairwise_MV2},		\
	      {262144, &MPIR_Alltoall_pairwise_MV2},		\
	      {524288, &MPIR_Alltoall_pairwise_MV2},		\
	      {1048576, &MPIR_Alltoall_pairwise_MV2}		\
	    }		\
	  },		\
	  {		\
	    4,		\
	    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},		\
	    20,		\
	    {		\
	      {1, &MPIR_Alltoall_RD_MV2},		\
	      {2, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {4, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {8, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {16, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {32, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {64, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {128, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {256, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {512, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {1024, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {2048, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {4096, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {8192, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {16384, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {32768, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {65536, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {131072, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {262144, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {524288, &MPIR_Alltoall_Scatter_dest_MV2},		\
	      {1048576, &MPIR_Alltoall_Scatter_dest_MV2}		\
	    }		\
	  }		\
};		
