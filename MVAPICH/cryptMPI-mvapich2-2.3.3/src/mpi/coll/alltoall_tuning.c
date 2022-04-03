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

#include <regex.h>
#include "common_tuning.h"
#include "alltoall_tuning.h"
#include "tuning/alltoall_arch_tuning.h"
#include "mv2_arch_hca_detect.h"
/* array used to tune alltoall */


int *mv2_alltoall_table_ppn_conf = NULL;
int mv2_alltoall_num_ppn_conf = 1;
int *mv2_size_alltoall_tuning_table = NULL;
mv2_alltoall_tuning_table **mv2_alltoall_thresholds_table = NULL;

int *mv2_alltoall_indexed_table_ppn_conf = NULL;
int mv2_alltoall_indexed_num_ppn_conf = 1;
int *mv2_size_alltoall_indexed_tuning_table = NULL;
mv2_alltoall_indexed_tuning_table **mv2_alltoall_indexed_thresholds_table = NULL;

int MV2_set_alltoall_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
  int agg_table_sum = 0;
  int i;

  
    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoall_tuning) {
      mv2_alltoall_indexed_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Lonestar Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 12;
	mv2_size_alltoall_indexed_tuning_table[2] = 6;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_12ppn[] =
	  GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_12ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      if ((MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                   MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20, MV2_HCA_MLX_CX_CONNIB) ||
          MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                   MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20, MV2_HCA_MLX_CX_CONNIB)) && !heterogeneity) {
	/*PSG Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 3;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 20;
	mv2_size_alltoall_indexed_tuning_table[2] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_20ppn[] =
	  GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_20ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Trestles Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 3;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 32;
	mv2_size_alltoall_indexed_tuning_table[2] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_32ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_32ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Gordon Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 8;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Yellowstone Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[0] = 3;
	  table_ptrs[0] = mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[0] = 2;
	  table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
#endif
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[1] = 3;
	  table_ptrs[1] = mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[1] = 2;
	  table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
#endif
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
        mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn[] =
          GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
        mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
          GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[2] = 4;
	  table_ptrs[2] = mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[2] = 5;
	  table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Stampede Table*/
	mv2_alltoall_indexed_num_ppn_conf = 4;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[0] = 4;
	  table_ptrs[0] = mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
#endif
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[1] = 4;
	  table_ptrs[1] = mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
#endif
 
    mv2_alltoall_indexed_table_ppn_conf[2] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_4ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_4ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[2] = 1;
	  table_ptrs[2] = mv2_tmp_cma_alltoall_indexed_thresholds_table_4ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[2] = 1;
	  table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_4ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_4ppn;
#endif     

	mv2_alltoall_indexed_table_ppn_conf[3] = 16;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[3] = 5;
	  table_ptrs[3] = mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[3] = 7;
	  table_ptrs[3] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[3] = 7;
	table_ptrs[3] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }

      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		  MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
	    /*RI2 Table*/
    MV2_COLL_TUNING_START_TABLE  (alltoall, 6)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, GEN2__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  4, GEN2_CMA__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, GEN2__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  5, GEN2_CMA__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, GEN2__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  5, GEN2_CMA__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, GEN2__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, GEN2__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 5, GEN2_CMA__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 28, 5, GEN2__RI2__28PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 28, 5, GEN2_CMA__RI2__28PPN)
    MV2_COLL_TUNING_FINISH_TABLE (alltoall)
      }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7551_64, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* AMD EPYC table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 7)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  3, GEN2__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  3, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  4, GEN2__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  4, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  4, GEN2__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  4, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  4, GEN2__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  4, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 4, GEN2__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 4, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 4, GEN2__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 32, 3, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 64, 4, GEN2__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 64, 4, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7742_128, MV2_HCA_ANY) && !heterogeneity) {
      /* AMD EPYC rome table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 7)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  1, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  2, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  2, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  2, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 2, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 2, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 64, 2, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* Haswell HDR nodes on NOWLAB */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 6)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Frontera */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 8)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  4, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  4, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  4, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  4, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 4, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 4, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 28, 3, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 28, 3, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 3, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 32, 3, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Mayer */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 8)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_32, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Catalyst */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 7)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /* ARM system at Hartree Center */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 5)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  2, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 24,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__24PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER8, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Ray Table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 3)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  4, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER9, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Sierra Table: Use table for Ray temporarily */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 9)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, GEN2__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  4, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, GEN2__IBM_POWER9_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, GEN2__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, GEN2__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, GEN2__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 22, 5, GEN2__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 22, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 5, GEN2__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 32, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 44, 5, GEN2__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 44, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*RI Table*/
    MV2_COLL_TUNING_START_TABLE  (alltoall, 4)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  2, GEN2__RI__1PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  2, GEN2__RI__2PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  1, GEN2__RI__4PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  8, GEN2__RI__8PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  1, GEN2_CMA__RI__4PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__RI__8PPN)
    MV2_COLL_TUNING_FINISH_TABLE (alltoall)
      }




      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Comet Table*/
	mv2_alltoall_indexed_num_ppn_conf = 1;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 24;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_24ppn[] =
	  GEN2__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
	/*
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_24ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_cma_alltoall_indexed_thresholds_table_24ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_24ppn;
	}
#else
	*/
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_24ppn;
	/*
#endif
	*/
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		  MV2_ARCH_ANY, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
	/*RI2 Table*/
    MV2_COLL_TUNING_START_TABLE  (alltoall, 6)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, GEN2__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  4, GEN2_CMA__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, GEN2__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  5, GEN2_CMA__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, GEN2__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  5, GEN2_CMA__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, GEN2__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, GEN2__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 5, GEN2_CMA__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 28, 5, GEN2__RI2__28PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 28, 5, GEN2_CMA__RI2__28PPN)
    MV2_COLL_TUNING_FINISH_TABLE (alltoall) 
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		  MV2_ARCH_ANY, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
	/*RI2 Table*/
    MV2_COLL_TUNING_START_TABLE  (alltoall, 6)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, GEN2__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 1,  4, GEN2_CMA__RI2__1PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, GEN2__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 2,  5, GEN2_CMA__RI2__2PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, GEN2__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 4,  5, GEN2_CMA__RI2__4PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, GEN2__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 8,  5, GEN2_CMA__RI2__8PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, GEN2__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 16, 5, GEN2_CMA__RI2__16PPN)
    MV2_COLL_TUNING_ADD_CONF     (alltoall, 28, 5, GEN2__RI2__28PPN)
    MV2_COLL_TUNING_ADD_CONF_CMA (alltoall, 28, 5, GEN2_CMA__RI2__28PPN)
    MV2_COLL_TUNING_FINISH_TABLE (alltoall) 
      }
      else {
	/*Stampede Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[0] = 4;
	  table_ptrs[0] = mv2_tmp_cma_alltoall_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
#endif
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[1] = 4;
	  table_ptrs[1] = mv2_tmp_cma_alltoall_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
#endif
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	mv2_alltoall_indexed_tuning_table mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_alltoall_indexed_tuning_table[2] = 5;
	  table_ptrs[2] = mv2_tmp_cma_alltoall_indexed_thresholds_table_16ppn;
	}
	else {
	  mv2_size_alltoall_indexed_tuning_table[2] = 7;
	  table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
	}
#else
	mv2_size_alltoall_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
#elif defined (CHANNEL_NEMESIS_IB)
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Lonestar Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 8;
	mv2_size_alltoall_indexed_tuning_table[2] = 3;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_8ppn[] =
	  NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_8ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Trestles Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 3;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 32;
	mv2_size_alltoall_indexed_tuning_table[2] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_32ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_32ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Gordon Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 4;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Yellowstone Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 2;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Stampede Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 7;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*RI Table*/
        mv2_alltoall_indexed_num_ppn_conf = 3;
        mv2_alltoall_indexed_thresholds_table
          = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
                        * mv2_alltoall_indexed_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
                                 * mv2_alltoall_indexed_num_ppn_conf);
        mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							     mv2_alltoall_indexed_num_ppn_conf);
        mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));

        mv2_alltoall_indexed_table_ppn_conf[0] = 1;
        mv2_size_alltoall_indexed_tuning_table[0] = 2;
        mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
          NEMESIS__RI__1PPN;
	  table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;

        mv2_alltoall_indexed_table_ppn_conf[1] = 2;
        mv2_size_alltoall_indexed_tuning_table[1] = 2;
        mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
          NEMESIS__RI__2PPN;
	  table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;

        mv2_alltoall_indexed_table_ppn_conf[2] = 8;
        mv2_size_alltoall_indexed_tuning_table[2] = 7;
        mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_8ppn[] =
          NEMESIS__RI__8PPN;
	  table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_8ppn;

        agg_table_sum = 0;
        for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
          agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
        }
        mv2_alltoall_indexed_thresholds_table[0] =
          MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
        MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_alltoall_indexed_tuning_table)
                     * mv2_size_alltoall_indexed_tuning_table[0]));
        for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
          mv2_alltoall_indexed_thresholds_table[i] =
            mv2_alltoall_indexed_thresholds_table[i - 1]
            + mv2_size_alltoall_indexed_tuning_table[i - 1];
          MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_alltoall_indexed_tuning_table)
                       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
        return 0;
      }
      else {
	/*Stampede Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 7;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
#endif
#else /* !CHANNEL_PSM */
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_QLGIC_QIB) && !heterogeneity) {
      /*Sierra Table*/
      mv2_alltoall_indexed_num_ppn_conf = 2;
      mv2_alltoall_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
		      * mv2_alltoall_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			       * mv2_alltoall_indexed_num_ppn_conf);
      mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							   mv2_alltoall_indexed_num_ppn_conf);
      mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
      mv2_alltoall_indexed_table_ppn_conf[0] = 1;
      mv2_size_alltoall_indexed_tuning_table[0] = 5;
      mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__1PPN;
      table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
      mv2_alltoall_indexed_table_ppn_conf[1] = 12;
      mv2_size_alltoall_indexed_tuning_table[1] = 6;
      mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_12ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__12PPN;
      table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
      }
      mv2_alltoall_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
      MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_alltoall_indexed_tuning_table)
		   * mv2_size_alltoall_indexed_tuning_table[0]));
      for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	mv2_alltoall_indexed_thresholds_table[i] =
	  mv2_alltoall_indexed_thresholds_table[i - 1]
	  + mv2_size_alltoall_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /*Bridges Table, we are using Bebop tables here*/
      MV2_COLL_TUNING_START_TABLE  (alltoall, 5)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16,  4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 36, 4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 5)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16,  4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 36, 4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-KNL Table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 6)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 32, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 64, 6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_PLATINUM_8170_2S_52, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-Skylake Table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 9)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 2,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 24, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__24PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 26, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__26PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 48, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__48PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 52, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__52PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
    else {
      /* default alltoall psm tables: Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (alltoall, 5)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 16,  4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (alltoall, 36, 4, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (alltoall)
    }
#endif /* !CHANNEL_PSM */
    {
	/*Stampede Table*/
	mv2_alltoall_indexed_num_ppn_conf = 3;
	mv2_alltoall_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
			* mv2_alltoall_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_indexed_tuning_table *)
				 * mv2_alltoall_indexed_num_ppn_conf);
	mv2_size_alltoall_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_alltoall_indexed_num_ppn_conf);
	mv2_alltoall_indexed_table_ppn_conf = MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      
	mv2_alltoall_indexed_table_ppn_conf[0] = 1;
	mv2_size_alltoall_indexed_tuning_table[0] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	table_ptrs[0] = mv2_tmp_alltoall_indexed_thresholds_table_1ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[1] = 2;
	mv2_size_alltoall_indexed_tuning_table[1] = 5;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	table_ptrs[1] = mv2_tmp_alltoall_indexed_thresholds_table_2ppn;
      
	mv2_alltoall_indexed_table_ppn_conf[2] = 16;
	mv2_size_alltoall_indexed_tuning_table[2] = 7;
	mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	table_ptrs[2] = mv2_tmp_alltoall_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_indexed_tuning_table[i];
	}
	mv2_alltoall_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_indexed_tuning_table));
	MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_alltoall_indexed_tuning_table)
		     * mv2_size_alltoall_indexed_tuning_table[0]));
	for (i = 1; i < mv2_alltoall_indexed_num_ppn_conf; i++) {
	  mv2_alltoall_indexed_thresholds_table[i] =
	    mv2_alltoall_indexed_thresholds_table[i - 1]
	    + mv2_size_alltoall_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_alltoall_indexed_tuning_table)
		       * mv2_size_alltoall_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
    }
    else {
      mv2_alltoall_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_alltoall_num_ppn_conf = 1;
        mv2_alltoall_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			* mv2_alltoall_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				 * mv2_alltoall_num_ppn_conf);
        mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						     mv2_alltoall_num_ppn_conf);
        mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
        mv2_alltoall_table_ppn_conf[0] = 12;
        mv2_size_alltoall_tuning_table[0] = 6;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_12ppn[] = {
	  {12,
	   2, 
	   {{0, 65536, &MPIR_Alltoall_Scatter_dest_MV2},
	    {65536, -1,  &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{32768, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {24,
	   3,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, 131072, &MPIR_Alltoall_Scatter_dest_MV2},
	    {131072, -1,  &MPIR_Alltoall_pairwise_MV2},
	   },
                
	   {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {48,
	   3,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, 131072, &MPIR_Alltoall_Scatter_dest_MV2},
	    {131072, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{32768, 131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {96,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {192,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {384,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
        };
        table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_12ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_tuning_table[i];
        }
        mv2_alltoall_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
        MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_alltoall_tuning_table)
                     * mv2_size_alltoall_tuning_table[0]));
        for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
	  mv2_alltoall_thresholds_table[i] =
            mv2_alltoall_thresholds_table[i - 1]
            + mv2_size_alltoall_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_alltoall_tuning_table)
                       * mv2_size_alltoall_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				      MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
        mv2_alltoall_num_ppn_conf = 3;
        mv2_alltoall_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			* mv2_alltoall_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				 * mv2_alltoall_num_ppn_conf);
        mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						     mv2_alltoall_num_ppn_conf);
        mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
        mv2_alltoall_table_ppn_conf[0] = 1;
        mv2_size_alltoall_tuning_table[0] = 6;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_1ppn[] = {
	  {2,
	   1, 
	   {{0, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {4,
	   2,
	   {{0, 262144, &MPIR_Alltoall_Scatter_dest_MV2},
	    {262144, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
                
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {8,
	   2,
	   {{0, 8, &MPIR_Alltoall_RD_MV2},
	    {8, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {16,
	   3,
	   {{0, 64, &MPIR_Alltoall_RD_MV2},
	    {64, 512, &MPIR_Alltoall_bruck_MV2},
	    {512, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0,-1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {32,
	   3,
	   {{0, 32, &MPIR_Alltoall_RD_MV2},
	    {32, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {64,
	   3,
	   {{0, 8, &MPIR_Alltoall_RD_MV2},
	    {8, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
        };
        table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_1ppn;
        mv2_alltoall_table_ppn_conf[1] = 2;
        mv2_size_alltoall_tuning_table[1] = 6;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_2ppn[] = {
	  {4,
	   2,
	   {{0, 32, &MPIR_Alltoall_RD_MV2},
	    {32, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
                
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {8,
	   2,
	   {{0, 64, &MPIR_Alltoall_RD_MV2},
	    {64, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
                
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {16,
	   3,
	   {{0, 64, &MPIR_Alltoall_RD_MV2},
	    {64, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0,-1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {32,
	   3,
	   {{0, 16, &MPIR_Alltoall_RD_MV2},
	    {16, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {64,
	   3,
	   {{0, 8, &MPIR_Alltoall_RD_MV2},
	    {8, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },

	  {128,
	   3,
	   {{0, 4, &MPIR_Alltoall_RD_MV2},
	    {4, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{0, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
        };
        table_ptrs[1] = mv2_tmp_alltoall_thresholds_table_2ppn;
        mv2_alltoall_table_ppn_conf[2] = 16;
        mv2_size_alltoall_tuning_table[2] = 7;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_16ppn[] = {
	  {16,
	   2, 
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1,  &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{32768, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {32,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
                
	   {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {64,
	   3,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, 16384, &MPIR_Alltoall_Scatter_dest_MV2},
	    {16384, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{32768, 131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {128,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {256,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {512,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
	  {1024,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
        };
        table_ptrs[2] = mv2_tmp_alltoall_thresholds_table_16ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_tuning_table[i];
        }
        mv2_alltoall_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
        MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_alltoall_tuning_table)
                     * mv2_size_alltoall_tuning_table[0]));
        for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
	  mv2_alltoall_thresholds_table[i] =
            mv2_alltoall_thresholds_table[i - 1]
            + mv2_size_alltoall_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_alltoall_tuning_table)
                       * mv2_size_alltoall_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      } else {
        mv2_alltoall_num_ppn_conf = 1;
        mv2_alltoall_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			* mv2_alltoall_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				 * mv2_alltoall_num_ppn_conf);
        mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						     mv2_alltoall_num_ppn_conf);
        mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
        mv2_alltoall_table_ppn_conf[0] = 8;
        mv2_size_alltoall_tuning_table[0] = 7;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_8ppn[] = {
	  {8,
	   1, 
	   {{0, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {16,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
                 
	   {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {32,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{16384, 262144, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {64,
	   3,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, 8192, &MPIR_Alltoall_Scatter_dest_MV2},
	    {8192, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {128,
	   3,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, 4096, &MPIR_Alltoall_Scatter_dest_MV2},
	    {16384, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {256,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {512,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
        };
        table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_tuning_table[i];
        }
        mv2_alltoall_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
        MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_alltoall_tuning_table)
                     * mv2_size_alltoall_tuning_table[0]));
        for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
	  mv2_alltoall_thresholds_table[i] =
            mv2_alltoall_thresholds_table[i - 1]
            + mv2_size_alltoall_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_alltoall_tuning_table)
                       * mv2_size_alltoall_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      }
#elif defined (CHANNEL_NEMESIS_IB)
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	  mv2_alltoall_num_ppn_conf = 1;
	  mv2_alltoall_thresholds_table
            = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			  * mv2_alltoall_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				   * mv2_alltoall_num_ppn_conf);
	  mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						       mv2_alltoall_num_ppn_conf);
	  mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
	  mv2_alltoall_table_ppn_conf[0] = 12;
	  mv2_size_alltoall_tuning_table[0] = 6;
	  mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_12ppn[] = {
            {12,
	     2, 
	     {{0, 65536, &MPIR_Alltoall_Scatter_dest_MV2},
	      {65536, -1,  &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{32768, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {24,
	     3,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, 131072, &MPIR_Alltoall_Scatter_dest_MV2},
	      {131072, -1,  &MPIR_Alltoall_pairwise_MV2},
	     },
                
	     {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {48,
	     3,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, 131072, &MPIR_Alltoall_Scatter_dest_MV2},
	      {131072, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{32768, 131072, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {96,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384,65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {192,
	     2,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {384,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
	  };
	  table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_12ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_alltoall_tuning_table[i];
	  }
	  mv2_alltoall_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_alltoall_tuning_table)
		       * mv2_size_alltoall_tuning_table[0]));
	  for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
            mv2_alltoall_thresholds_table[i] =
	      mv2_alltoall_thresholds_table[i - 1]
	      + mv2_size_alltoall_tuning_table[i - 1];
            MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_alltoall_tuning_table)
			 * mv2_size_alltoall_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	} else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
					MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	  mv2_alltoall_num_ppn_conf = 3;
	  mv2_alltoall_thresholds_table
            = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			  * mv2_alltoall_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				   * mv2_alltoall_num_ppn_conf);
	  mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						       mv2_alltoall_num_ppn_conf);
	  mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
	  mv2_alltoall_table_ppn_conf[0] = 1;
	  mv2_size_alltoall_tuning_table[0] = 6;
	  mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_1ppn[] = {
            {2,
	     1, 
	     {{0, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {4,
	     2,
	     {{0, 262144, &MPIR_Alltoall_Scatter_dest_MV2},
	      {262144, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
                
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {8,
	     2,
	     {{0, 8, &MPIR_Alltoall_RD_MV2},
	      {8, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {16,
	     3,
	     {{0, 64, &MPIR_Alltoall_RD_MV2},
	      {64, 512, &MPIR_Alltoall_bruck_MV2},
	      {512, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0,-1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {32,
	     3,
	     {{0, 32, &MPIR_Alltoall_RD_MV2},
	      {32, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {64,
	     3,
	     {{0, 8, &MPIR_Alltoall_RD_MV2},
	      {8, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
	  };
	  table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_1ppn;
	  mv2_alltoall_table_ppn_conf[1] = 2;
	  mv2_size_alltoall_tuning_table[1] = 6;
	  mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_2ppn[] = {
            {4,
	     2,
	     {{0, 32, &MPIR_Alltoall_RD_MV2},
	      {32, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
                
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {8,
	     2,
	     {{0, 64, &MPIR_Alltoall_RD_MV2},
	      {64, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
                
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {16,
	     3,
	     {{0, 64, &MPIR_Alltoall_RD_MV2},
	      {64, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0,-1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {32,
	     3,
	     {{0, 16, &MPIR_Alltoall_RD_MV2},
	      {16, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {64,
	     3,
	     {{0, 8, &MPIR_Alltoall_RD_MV2},
	      {8, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },

            {128,
	     3,
	     {{0, 4, &MPIR_Alltoall_RD_MV2},
	      {4, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{0, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
	  };
	  table_ptrs[1] = mv2_tmp_alltoall_thresholds_table_2ppn;
	  mv2_alltoall_table_ppn_conf[2] = 16;
	  mv2_size_alltoall_tuning_table[2] = 7;
	  mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_16ppn[] = {
            {16,
	     2, 
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1,  &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{32768, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {32,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
                
	     {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {64,
	     3,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, 16384, &MPIR_Alltoall_Scatter_dest_MV2},
	      {16384, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{32768, 131072, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {128,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384,65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {256,
	     2,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
            {512,
	     2,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
            {1024,
	     2,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, 65536, &MPIR_Alltoall_inplace_MV2},
	     },
            },
  
	  };
	  table_ptrs[2] = mv2_tmp_alltoall_thresholds_table_16ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_alltoall_tuning_table[i];
	  }
	  mv2_alltoall_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_alltoall_tuning_table)
		       * mv2_size_alltoall_tuning_table[0]));
	  for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
            mv2_alltoall_thresholds_table[i] =
	      mv2_alltoall_thresholds_table[i - 1]
	      + mv2_size_alltoall_tuning_table[i - 1];
            MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_alltoall_tuning_table)
			 * mv2_size_alltoall_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	} else {
	  mv2_alltoall_num_ppn_conf = 1;
	  mv2_alltoall_thresholds_table
            = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			  * mv2_alltoall_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				   * mv2_alltoall_num_ppn_conf);
	  mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						       mv2_alltoall_num_ppn_conf);
	  mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
	  mv2_alltoall_table_ppn_conf[0] = 8;
	  mv2_size_alltoall_tuning_table[0] = 7;
	  mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_8ppn[] = {
	    {8,
	     1, 
	     {{0, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {16,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
                 
	     {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {32,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	     },
  
	     {{16384, 262144, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {64,
	     3,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, 8192, &MPIR_Alltoall_Scatter_dest_MV2},
	      {8192, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {128,
	     3,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, 4096, &MPIR_Alltoall_Scatter_dest_MV2},
	      {16384, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {256,
	     2,
	     {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	      {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
  
	    {512,
	     2,
	     {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	      {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	     },
  
	     {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	     },
	    },
	  };
	  table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_8ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_alltoall_tuning_table[i];
	  }
	  mv2_alltoall_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_alltoall_tuning_table)
		       * mv2_size_alltoall_tuning_table[0]));
	  for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
            mv2_alltoall_thresholds_table[i] =
	      mv2_alltoall_thresholds_table[i - 1]
	      + mv2_size_alltoall_tuning_table[i - 1];
            MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_alltoall_tuning_table)
			 * mv2_size_alltoall_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
#endif
#endif /* !CHANNEL_PSM */
      {
        mv2_alltoall_num_ppn_conf = 1;
        mv2_alltoall_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
			* mv2_alltoall_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_alltoall_tuning_table *)
				 * mv2_alltoall_num_ppn_conf);
        mv2_size_alltoall_tuning_table = MPIU_Malloc(sizeof(int) *
						     mv2_alltoall_num_ppn_conf);
        mv2_alltoall_table_ppn_conf = MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
        mv2_alltoall_table_ppn_conf[0] = 8;
        mv2_size_alltoall_tuning_table[0] = 7;
        mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table_8ppn[] = {
	  {8,
	   1, 
	   {{0, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {16,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
                 
	   {{65536, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {32,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_Scatter_dest_MV2},
	   },
  
	   {{16384, 262144, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {64,
	   3,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, 8192, &MPIR_Alltoall_Scatter_dest_MV2},
	    {8192, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {128,
	   3,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, 4096, &MPIR_Alltoall_Scatter_dest_MV2},
	    {16384, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {256,
	   2,
	   {{0, 2048, &MPIR_Alltoall_bruck_MV2},
	    {2048, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384,131072, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
  
	  {512,
	   2,
	   {{0, 1024, &MPIR_Alltoall_bruck_MV2},
	    {1024, -1, &MPIR_Alltoall_pairwise_MV2},
	   },
  
	   {{16384, -1, &MPIR_Alltoall_inplace_MV2},
	   },
	  },
        };
        table_ptrs[0] = mv2_tmp_alltoall_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_alltoall_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_alltoall_tuning_table[i];
        }
        mv2_alltoall_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_alltoall_tuning_table));
        MPIU_Memcpy(mv2_alltoall_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_alltoall_tuning_table)
                     * mv2_size_alltoall_tuning_table[0]));
        for (i = 1; i < mv2_alltoall_num_ppn_conf; i++) {
	  mv2_alltoall_thresholds_table[i] =
            mv2_alltoall_thresholds_table[i - 1]
            + mv2_size_alltoall_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_alltoall_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_alltoall_tuning_table)
                       * mv2_size_alltoall_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      }
    }
    return 0;
}

void MV2_cleanup_alltoall_tuning_table()
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_alltoall_tuning) {
    MPIU_Free(mv2_alltoall_indexed_thresholds_table[0]);
    MPIU_Free(mv2_alltoall_indexed_table_ppn_conf);
    MPIU_Free(mv2_size_alltoall_indexed_tuning_table);
    if (mv2_alltoall_indexed_thresholds_table != NULL) {
      MPIU_Free(mv2_alltoall_indexed_thresholds_table);
    }
  }
  else {
    MPIU_Free(mv2_alltoall_thresholds_table[0]);
    MPIU_Free(mv2_alltoall_table_ppn_conf);
    MPIU_Free(mv2_size_alltoall_tuning_table);
    if (mv2_alltoall_thresholds_table != NULL) {
        MPIU_Free(mv2_alltoall_thresholds_table);
    }
  }
}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}

int MV2_Alltoall_is_define(char *mv2_user_alltoall)
{
    int i;
    int nb_element = count_sep(mv2_user_alltoall) + 1;
    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoall_tuning) {
      mv2_alltoall_indexed_num_ppn_conf = 1;

      if (mv2_size_alltoall_indexed_tuning_table == NULL) {
        mv2_size_alltoall_indexed_tuning_table =
	  MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      }
      mv2_size_alltoall_indexed_tuning_table[0] = 1;
    
      if (mv2_alltoall_indexed_table_ppn_conf == NULL) {
        mv2_alltoall_indexed_table_ppn_conf =
	  MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf * sizeof(int));
      }
      mv2_alltoall_indexed_table_ppn_conf[0] = -1;

      mv2_alltoall_indexed_tuning_table mv2_tmp_alltoall_indexed_thresholds_table[1];

      /* If one alltoall_indexed tuning table is already defined */
      if (mv2_alltoall_indexed_thresholds_table != NULL) {
        if (mv2_alltoall_indexed_thresholds_table[0] != NULL) {
            MPIU_Free(mv2_alltoall_indexed_thresholds_table[0]);
        }
        MPIU_Free(mv2_alltoall_indexed_thresholds_table);
      }

      /* We realloc the space for the new alltoall_indexed tuning table */
      mv2_alltoall_indexed_thresholds_table =
	MPIU_Malloc(mv2_alltoall_indexed_num_ppn_conf *
		    sizeof(mv2_alltoall_indexed_tuning_table *));
      mv2_alltoall_indexed_thresholds_table[0] =
	MPIU_Malloc(mv2_size_alltoall_indexed_tuning_table[0] *
		    sizeof(mv2_alltoall_indexed_tuning_table));

      if (nb_element == 1) {
        mv2_tmp_alltoall_indexed_thresholds_table[0].numproc = 1;
        mv2_tmp_alltoall_indexed_thresholds_table[0].size_table = 1;
        mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].msg_sz = 1;
        mv2_tmp_alltoall_indexed_thresholds_table[0].in_place_algo_table[0] = 0;
        switch (atoi(mv2_user_alltoall)) {
        case ALLTOALL_BRUCK_MV2:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_bruck_MV2;
	  break;
        case ALLTOALL_RD_MV2:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_RD_MV2;
	  break;
        case ALLTOALL_SCATTER_DEST_MV2:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_Scatter_dest_MV2;
	  break;
        case ALLTOALL_PAIRWISE_MV2:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_pairwise_MV2;
	  break;
        case ALLTOALL_INPLACE_MV2:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_inplace_MV2;
	  break;
        default:
	  mv2_tmp_alltoall_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_bruck_MV2;
        }
      }
      MPIU_Memcpy(mv2_alltoall_indexed_thresholds_table[0], mv2_tmp_alltoall_indexed_thresholds_table, sizeof
		  (mv2_alltoall_indexed_tuning_table));
    }
    else {
      mv2_alltoall_num_ppn_conf = 1;

      if (mv2_size_alltoall_tuning_table == NULL) {
        mv2_size_alltoall_tuning_table =
	  MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
      }
      mv2_size_alltoall_tuning_table[0] = 1;
    
      if (mv2_alltoall_table_ppn_conf == NULL) {
        mv2_alltoall_table_ppn_conf =
	  MPIU_Malloc(mv2_alltoall_num_ppn_conf * sizeof(int));
      }
      mv2_alltoall_table_ppn_conf[0] = -1;

      mv2_alltoall_tuning_table mv2_tmp_alltoall_thresholds_table[1];

      /* If one alltoall tuning table is already defined */
      if (mv2_alltoall_thresholds_table != NULL) {
        MPIU_Free(mv2_alltoall_thresholds_table);
      }

      /* We realloc the space for the new alltoall tuning table */
      mv2_alltoall_thresholds_table =
	MPIU_Malloc(mv2_alltoall_num_ppn_conf *
		    sizeof(mv2_alltoall_tuning_table *));
      mv2_alltoall_thresholds_table[0] =
	MPIU_Malloc(mv2_size_alltoall_tuning_table[0] *
		    sizeof(mv2_alltoall_tuning_table));

      if (nb_element == 1) {
        mv2_tmp_alltoall_thresholds_table[0].numproc = 1;
        mv2_tmp_alltoall_thresholds_table[0].size_table = 1;
        mv2_tmp_alltoall_thresholds_table[0].algo_table[0].min = 0;
        mv2_tmp_alltoall_thresholds_table[0].algo_table[0].max = -1;
        switch (atoi(mv2_user_alltoall)) {
        case ALLTOALL_BRUCK_MV2:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_bruck_MV2;
	  break;
        case ALLTOALL_RD_MV2:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_RD_MV2;
	  break;
        case ALLTOALL_SCATTER_DEST_MV2:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_Scatter_dest_MV2;
	  break;
        case ALLTOALL_PAIRWISE_MV2:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_pairwise_MV2;
	  break;
        case ALLTOALL_INPLACE_MV2:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_inplace_MV2;
	  break;
        default:
	  mv2_tmp_alltoall_thresholds_table[0].algo_table[0].MV2_pt_Alltoall_function =
	    &MPIR_Alltoall_bruck_MV2;
        }
      } else {
        char *dup, *p, *save_p;
        regmatch_t match[NMATCH];
        regex_t preg;
        const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

        if (!(dup = MPIU_Strdup(mv2_user_alltoall))) {
	  fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_alltoall);
	  return 1;
        }

        if (regcomp(&preg, regexp, REG_EXTENDED)) {
	  fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_alltoall);
	  MPIU_Free(dup);
	  return 2;
        }

        mv2_tmp_alltoall_thresholds_table[0].numproc = 1;
        mv2_tmp_alltoall_thresholds_table[0].size_table = nb_element;
        i = 0;
        for (p = strtok_r(dup, ",", &save_p); p; p = strtok_r(NULL, ",", &save_p)) {
	  if (regexec(&preg, p, NMATCH, match, 0)) {
	    fprintf(stderr, "failed to match on `%s'\n", p);
	    regfree(&preg);
	    MPIU_Free(dup);
	    return 2;
	  }
	  /* given () start at 1 */
	  switch (atoi(p + match[1].rm_so)) {
	  case ALLTOALL_BRUCK_MV2:
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].MV2_pt_Alltoall_function =
	      &MPIR_Alltoall_bruck_MV2;
	    break;
	  case ALLTOALL_RD_MV2:
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].MV2_pt_Alltoall_function =
	      &MPIR_Alltoall_RD_MV2;
	    break;
	  case ALLTOALL_SCATTER_DEST_MV2:
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].MV2_pt_Alltoall_function =
	      &MPIR_Alltoall_Scatter_dest_MV2;
	    break;
	  case ALLTOALL_PAIRWISE_MV2:
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].MV2_pt_Alltoall_function =
	      &MPIR_Alltoall_pairwise_MV2;
	    break;
	  case ALLTOALL_INPLACE_MV2:
	    mv2_tmp_alltoall_thresholds_table[0].in_place_algo_table[i].
	      MV2_pt_Alltoall_function = &MPIR_Alltoall_inplace_MV2;
	    break;
	  default:
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].MV2_pt_Alltoall_function =
	      &MPIR_Alltoall_bruck_MV2;
	  }
	  if(atoi(p + match[1].rm_so) <= ALLTOALL_PAIRWISE_MV2) { 
	    mv2_tmp_alltoall_thresholds_table[0].algo_table[i].min = atoi(p +
									  match[2].rm_so);
	    if (p[match[3].rm_so] == '+') {
	      mv2_tmp_alltoall_thresholds_table[0].algo_table[i].max = -1;
	    } else {
	      mv2_tmp_alltoall_thresholds_table[0].algo_table[i].max =
		atoi(p + match[3].rm_so);
	    }
	  } else {  
	    int j=0; 
	    mv2_tmp_alltoall_thresholds_table[0].in_place_algo_table[j].min = atoi(p +
										   match[2].rm_so);
	    if (p[match[3].rm_so] == '+') {
	      mv2_tmp_alltoall_thresholds_table[0].in_place_algo_table[j].max = -1;
	    } else {
	      mv2_tmp_alltoall_thresholds_table[0].in_place_algo_table[j].max =
		atoi(p + match[3].rm_so);
	    }
	  } 
	  i++;
        }
        MPIU_Free(dup);
        regfree(&preg);
      }
      MPIU_Memcpy(mv2_alltoall_thresholds_table[0], mv2_tmp_alltoall_thresholds_table, sizeof
		  (mv2_alltoall_tuning_table));
    }
    return 0;
}
