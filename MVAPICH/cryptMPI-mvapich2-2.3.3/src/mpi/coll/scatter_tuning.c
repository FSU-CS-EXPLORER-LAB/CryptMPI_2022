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
#include "scatter_tuning.h"
#include "tuning/scatter_arch_tuning.h"
#include "mv2_arch_hca_detect.h"


int *mv2_scatter_table_ppn_conf = NULL;
int mv2_scatter_num_ppn_conf = 1;
int *mv2_size_scatter_tuning_table = NULL;
mv2_scatter_tuning_table **mv2_scatter_thresholds_table = NULL;


int *mv2_scatter_indexed_table_ppn_conf = NULL;
int mv2_scatter_indexed_num_ppn_conf = 1;
int *mv2_size_scatter_indexed_tuning_table = NULL;
mv2_scatter_indexed_tuning_table **mv2_scatter_indexed_thresholds_table = NULL;

int MV2_set_scatter_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
    int agg_table_sum = 0;
    int i;
    
    if (mv2_use_indexed_tuning || mv2_use_indexed_scatter_tuning) {
        mv2_scatter_indexed_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	  /* Lonestar Table*/
	  mv2_scatter_indexed_num_ppn_conf = 3;
	  mv2_scatter_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			  * mv2_scatter_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				   * mv2_scatter_indexed_num_ppn_conf);
	  mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							      mv2_scatter_indexed_num_ppn_conf);
	  mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_scatter_indexed_table_ppn_conf[0] = 1;
	  mv2_size_scatter_indexed_tuning_table[0] = 4;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[1] = 2;
	  mv2_size_scatter_indexed_tuning_table[1] = 3;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[2] = 12;
	  mv2_size_scatter_indexed_tuning_table[2] = 6;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_12ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN
	  table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_12ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	  }
	  mv2_scatter_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    mv2_scatter_indexed_thresholds_table[i] =
	      mv2_scatter_indexed_thresholds_table[i - 1]
	      + mv2_size_scatter_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_indexed_tuning_table)
			 * mv2_size_scatter_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
    if ((MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                 MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20, MV2_HCA_MLX_CX_CONNIB) ||
        MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                 MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20, MV2_HCA_MLX_CX_CONNIB)) && !heterogeneity) {
	  /* PSG Table*/
	  mv2_scatter_indexed_num_ppn_conf = 3;
	  mv2_scatter_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			  * mv2_scatter_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				   * mv2_scatter_indexed_num_ppn_conf);
	  mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							      mv2_scatter_indexed_num_ppn_conf);
	  mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_scatter_indexed_table_ppn_conf[0] = 1;
	  mv2_size_scatter_indexed_tuning_table[0] = 3;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__1PPN;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[1] = 2;
	  mv2_size_scatter_indexed_tuning_table[1] = 4;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__2PPN;
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[2] = 20;
	  mv2_size_scatter_indexed_tuning_table[2] = 4;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_20ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN;
	  table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_20ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	  }
	  mv2_scatter_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    mv2_scatter_indexed_thresholds_table[i] =
	      mv2_scatter_indexed_thresholds_table[i - 1]
	      + mv2_size_scatter_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_indexed_tuning_table)
			 * mv2_size_scatter_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12, MV2_HCA_MLX_CX_CONNIB) && !heterogeneity) {
	  /* Wilkes Table*/
	  mv2_scatter_indexed_num_ppn_conf = 3;
	  mv2_scatter_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			  * mv2_scatter_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				   * mv2_scatter_indexed_num_ppn_conf);
	  mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							      mv2_scatter_indexed_num_ppn_conf);
	  mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_scatter_indexed_table_ppn_conf[0] = 1;
	  mv2_size_scatter_indexed_tuning_table[0] = 6;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__1PPN
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[1] = 2;
	  mv2_size_scatter_indexed_tuning_table[1] = 6;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__2PPN
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	  mv2_scatter_indexed_table_ppn_conf[2] = 12;
	  mv2_size_scatter_indexed_tuning_table[2] = 6;
	  mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_12ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__12PPN
	  table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_12ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	  }
	  mv2_scatter_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	    mv2_scatter_indexed_thresholds_table[i] =
	      mv2_scatter_indexed_thresholds_table[i - 1]
	      + mv2_size_scatter_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_indexed_tuning_table)
			 * mv2_size_scatter_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Trestles Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 4;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 3;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 32;
#if defined(_SMP_CMA_)
      mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_32ppn[] =
        GEN2_CMA__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_32ppn[] =
        GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      if (g_smp_use_cma) {
	mv2_size_scatter_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_scatter_indexed_thresholds_table_32ppn;
      }
      else {
	mv2_size_scatter_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_32ppn;
      }
#else
	mv2_size_scatter_indexed_tuning_table[2] = 4;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_32ppn[] =
	  GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_32ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Gordon Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[0] = 6;
	  table_ptrs[0] = mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[0] = 6;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[1] = 7;
	  table_ptrs[1] = mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[1] = 7;
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[1] = 7;
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
      mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
      mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      }
#else
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Yellowstone Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[0] = 3;
	  table_ptrs[0] = mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[0] = 2;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[0] = 2;
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[1] = 3;
	  table_ptrs[1] = mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[1] = 2;
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[1] = 2;
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
      mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
      mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_scatter_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_scatter_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      }
#else
	mv2_size_scatter_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Stampede Table*/
	mv2_scatter_indexed_num_ppn_conf = 4;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
#endif

 	mv2_scatter_indexed_table_ppn_conf[2] = 4;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_4ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_4ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[2] = 1;
	  table_ptrs[2] = mv2_tmp_cma_scatter_indexed_thresholds_table_4ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[2] = 1;
	  table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_4ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_4ppn;
#endif     

	mv2_scatter_indexed_table_ppn_conf[3] = 16;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[3] = 6;
	  table_ptrs[3] = mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[3] = 6;
	  table_ptrs[3] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[3] = 6;
	table_ptrs[3] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }

      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		 MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
        /*RI2 Table*/
        MV2_COLL_TUNING_START_TABLE  (scatter, 6)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  5, GEN2__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  5, GEN2_CMA__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  6, GEN2__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  6, GEN2_CMA__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  5, GEN2_CMA__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, GEN2__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  5, GEN2_CMA__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, GEN2__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 5, GEN2_CMA__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 28, 6, GEN2__RI2__28PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 28, 6, GEN2_CMA__RI2__28PPN)
        MV2_COLL_TUNING_FINISH_TABLE (scatter)
      }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7551_64, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* AMD EPYC table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 7)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  3, GEN2__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  3, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  4, GEN2__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  4, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  4, GEN2__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  4, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  4, GEN2__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  4, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 4, GEN2__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 4, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 4, GEN2__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 32, 4, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 64, 4, GEN2__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 64, 4, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7742_128, MV2_HCA_ANY) && !heterogeneity) {
      /* AMD EPYC rome table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 7)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  1, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  2, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  2, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  2, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 2, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 2, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 64, 2, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* Haswell HDR nodes on NOWLAB */
      MV2_COLL_TUNING_START_TABLE  (scatter, 6)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Frontera */
      MV2_COLL_TUNING_START_TABLE  (scatter, 8)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 28, 4, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 28, 4, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 5, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 32, 5, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Mayer */
      MV2_COLL_TUNING_START_TABLE  (scatter, 8)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_32, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Catalyst */
      MV2_COLL_TUNING_START_TABLE  (scatter, 7)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /* ARM system at Hartree Center */
      MV2_COLL_TUNING_START_TABLE  (scatter, 5)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  2, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 24,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__24PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER8, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Ray Table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 3)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  4, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER9, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Sierra Table: Use table for Ray temporarily */
      MV2_COLL_TUNING_START_TABLE  (scatter, 9)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  4, GEN2__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  4, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, GEN2__IBM_POWER9_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, GEN2__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, GEN2__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 22, 5, GEN2__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 22, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 5, GEN2__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 32, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 44, 5, GEN2__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 44, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        /*RI Table*/
        MV2_COLL_TUNING_START_TABLE  (scatter, 4)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  6, GEN2__RI__1PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  6, GEN2_CMA__RI__1PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  7, GEN2__RI__2PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  7, GEN2_CMA__RI__2PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  1, GEN2__RI__4PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  1, GEN2_CMA__RI__4PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  8, GEN2__RI__8PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  8, GEN2_CMA__RI__8PPN)
        MV2_COLL_TUNING_FINISH_TABLE (scatter)
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Comet Table*/
	mv2_scatter_indexed_num_ppn_conf = 1;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 24;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_24ppn[] =
	    GEN2__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
	/*
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_24ppn[] =
	    GEN2_CMA__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[0] = 6;
	  table_ptrs[0] = mv2_tmp_cma_scatter_indexed_thresholds_table_24ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[0] = 6;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_24ppn;
	}
#else
	*/
	mv2_size_scatter_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_24ppn;
	/*
#endif
	*/
	
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		 MV2_ARCH_ANY, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
        MV2_COLL_TUNING_START_TABLE  (scatter, 6)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  5, GEN2__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  5, GEN2_CMA__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  6, GEN2__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  6, GEN2_CMA__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  5, GEN2_CMA__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, GEN2__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  5, GEN2_CMA__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, GEN2__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 5, GEN2_CMA__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 28, 6, GEN2__RI2__28PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 28, 6, GEN2_CMA__RI2__28PPN)
        MV2_COLL_TUNING_FINISH_TABLE (scatter)
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		 MV2_ARCH_ANY, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
        MV2_COLL_TUNING_START_TABLE  (scatter, 6)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  5, GEN2__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 1,  5, GEN2_CMA__RI2__1PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  6, GEN2__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 2,  6, GEN2_CMA__RI2__2PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, GEN2__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 4,  5, GEN2_CMA__RI2__4PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, GEN2__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 8,  5, GEN2_CMA__RI2__8PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, GEN2__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 16, 5, GEN2_CMA__RI2__16PPN)
        MV2_COLL_TUNING_ADD_CONF     (scatter, 28, 6, GEN2__RI2__28PPN)
        MV2_COLL_TUNING_ADD_CONF_CMA (scatter, 28, 6, GEN2_CMA__RI2__28PPN)
        MV2_COLL_TUNING_FINISH_TABLE (scatter)
      }
      else {
	/*Stampede Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_cma_scatter_indexed_thresholds_table_1ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[0] = 5;
	  table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_cma_scatter_indexed_thresholds_table_2ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[1] = 6;
	  table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
#endif
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_scatter_indexed_tuning_table mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
	if (g_smp_use_cma) {
	  mv2_size_scatter_indexed_tuning_table[2] = 6;
	  table_ptrs[2] = mv2_tmp_cma_scatter_indexed_thresholds_table_16ppn;
	}
	else {
	  mv2_size_scatter_indexed_tuning_table[2] = 6;
	  table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
	}
#else
	mv2_size_scatter_indexed_tuning_table[2] = 6;
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
#endif
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
#elif defined (CHANNEL_NEMESIS_IB)
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Trestles Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 4;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 3;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 32;
	mv2_size_scatter_indexed_tuning_table[2] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_32ppn[] =
	  NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_32ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	/*Gordon Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_size_scatter_indexed_tuning_table[2] = 4;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_16PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Yellowstone Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 2;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_size_scatter_indexed_tuning_table[2] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
	/*Stampede Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        /*RI Table*/
        mv2_scatter_indexed_num_ppn_conf = 3;
        mv2_scatter_indexed_thresholds_table
          = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
                        * mv2_scatter_indexed_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
                                 * mv2_scatter_indexed_num_ppn_conf);
        mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
                                                          mv2_scatter_indexed_num_ppn_conf);
        mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));

        mv2_scatter_indexed_table_ppn_conf[0] = 1;
        mv2_size_scatter_indexed_tuning_table[0] = 2;
        mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
          NEMESIS__RI__1PPN
        table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;

        mv2_scatter_indexed_table_ppn_conf[1] = 2;
        mv2_size_scatter_indexed_tuning_table[1] = 2;
        mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
          NEMESIS__RI__2PPN
        table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;

        mv2_scatter_indexed_table_ppn_conf[2] = 8;
        mv2_size_scatter_indexed_tuning_table[2] = 8;
        mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_8ppn[] =
          NEMESIS__RI__8PPN
        table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_8ppn;

        agg_table_sum = 0;
        for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
          agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
        }
        mv2_scatter_indexed_thresholds_table[0] =
          MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
        MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_scatter_indexed_tuning_table)
                     * mv2_size_scatter_indexed_tuning_table[0]));
        for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
          mv2_scatter_indexed_thresholds_table[i] =
            mv2_scatter_indexed_thresholds_table[i - 1]
            + mv2_size_scatter_indexed_tuning_table[i - 1];
          MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_scatter_indexed_tuning_table)
                       * mv2_size_scatter_indexed_tuning_table[i]));
	}
        MPIU_Free(table_ptrs);
        return 0;
      }
      else {
	/*Stampede Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
#endif
#else /* !CHANNEL_PSM */
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_QLGIC_QIB) && !heterogeneity) {
      /*Sierra Table*/
      mv2_scatter_indexed_num_ppn_conf = 2;
      mv2_scatter_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
		      * mv2_scatter_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			       * mv2_scatter_indexed_num_ppn_conf);
      mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
      mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
      mv2_scatter_indexed_table_ppn_conf[0] = 1;
      mv2_size_scatter_indexed_tuning_table[0] = 5;
      mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__1PPN;
      table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
      mv2_scatter_indexed_table_ppn_conf[1] = 12;
      mv2_size_scatter_indexed_tuning_table[1] = 6;
      mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_12ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__12PPN;
      table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
      }
      mv2_scatter_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
      MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_scatter_indexed_tuning_table)
		   * mv2_size_scatter_indexed_tuning_table[0]));
      for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	mv2_scatter_indexed_thresholds_table[i] =
	  mv2_scatter_indexed_thresholds_table[i - 1]
	  + mv2_size_scatter_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /*Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (scatter, 6)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 5)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 36, 5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-KNL Table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 6)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 32, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 64, 4, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_PLATINUM_8170_2S_52, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-Skylake Table */
      MV2_COLL_TUNING_START_TABLE  (scatter, 9)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 24, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__24PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 26, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__26PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 48, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__48PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 52, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__52PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
    else {
      /*default psm table: Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (scatter, 6)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (scatter, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (scatter)
    }
#endif /* !CHANNEL_PSM */
     {
	/*Stampede Table*/
	mv2_scatter_indexed_num_ppn_conf = 3;
	mv2_scatter_indexed_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
			* mv2_scatter_indexed_num_ppn_conf);
	table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_indexed_tuning_table *)
				 * mv2_scatter_indexed_num_ppn_conf);
	mv2_size_scatter_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							  mv2_scatter_indexed_num_ppn_conf);
	mv2_scatter_indexed_table_ppn_conf = MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
      
	mv2_scatter_indexed_table_ppn_conf[0] = 1;
	mv2_size_scatter_indexed_tuning_table[0] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_1ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
	table_ptrs[0] = mv2_tmp_scatter_indexed_thresholds_table_1ppn;
      
	mv2_scatter_indexed_table_ppn_conf[1] = 2;
	mv2_size_scatter_indexed_tuning_table[1] = 5;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_2ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
	table_ptrs[1] = mv2_tmp_scatter_indexed_thresholds_table_2ppn;
      
	mv2_scatter_indexed_table_ppn_conf[2] = 16;
	mv2_size_scatter_indexed_tuning_table[2] = 7;
	mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table_16ppn[] =
	  NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
	table_ptrs[2] = mv2_tmp_scatter_indexed_thresholds_table_16ppn;
      
	agg_table_sum = 0;
	for (i = 0; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_indexed_tuning_table[i];
	}
	mv2_scatter_indexed_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_indexed_tuning_table));
	MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_indexed_tuning_table)
		     * mv2_size_scatter_indexed_tuning_table[0]));
	for (i = 1; i < mv2_scatter_indexed_num_ppn_conf; i++) {
	  mv2_scatter_indexed_thresholds_table[i] =
	    mv2_scatter_indexed_thresholds_table[i - 1]
	    + mv2_size_scatter_indexed_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[i], table_ptrs[i],
		      (sizeof(mv2_scatter_indexed_tuning_table)
		       * mv2_size_scatter_indexed_tuning_table[i]));
	}
	MPIU_Free(table_ptrs);
	return 0;
      }
    }
    else {
      mv2_scatter_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_scatter_num_ppn_conf = 1;
        mv2_scatter_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			* mv2_scatter_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
				 * mv2_scatter_num_ppn_conf);
        mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
						    mv2_scatter_num_ppn_conf);
        mv2_scatter_table_ppn_conf = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
        mv2_scatter_table_ppn_conf[0] = 12;
        mv2_size_scatter_tuning_table[0] = 6;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_12ppn[] = {
	  {
	    12,
	    2,
	    { 
	      {0, 512, &MPIR_Scatter_MV2_Binomial}, 
	      {512, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    24,
	    2,
	    {
	      {0, 512, &MPIR_Scatter_MV2_Binomial}, 
	      {512, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    48,
	    2,
	    {
	      {0, 512, &MPIR_Scatter_MV2_Binomial},
	      {512, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    96,
	    3,
	    {
	      {0, 256, &MPIR_Scatter_MV2_two_level_Direct},
	      {256, 8192, &MPIR_Scatter_MV2_two_level_Direct},
	      {8192, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    2,
	    {
	      { 0, 256, &MPIR_Scatter_MV2_Binomial},
	      { 256, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    192,
	    3,
	    {
	      {1, 2, &MPIR_Scatter_MV2_Binomial},
	      {2, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Binomial},
	    },
	  },

	  {
	    384,
	    3,
	    {
	      {1, 32, &MPIR_Scatter_MV2_Binomial},
	      {32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Binomial},
	    },  
	  },  
        };
        table_ptrs[0] = mv2_tmp_scatter_thresholds_table_12ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_tuning_table[i];
        }
        mv2_scatter_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
        MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_scatter_tuning_table)
                     * mv2_size_scatter_tuning_table[0]));
        for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
	  mv2_scatter_thresholds_table[i] =
            mv2_scatter_thresholds_table[i - 1]
            + mv2_size_scatter_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_scatter_tuning_table)
                       * mv2_size_scatter_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
        /*Stampede,*/
        mv2_scatter_num_ppn_conf = 3;
        mv2_scatter_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			* mv2_scatter_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
                                 * mv2_scatter_num_ppn_conf);
        mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
						    mv2_scatter_num_ppn_conf);
        mv2_scatter_table_ppn_conf 
	  = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
        mv2_scatter_table_ppn_conf[0] = 1;
        mv2_size_scatter_tuning_table[0] = 6;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_1ppn[] = {
	  {2,
	   1, 
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },

	  {4,
	   1, 
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	  },
  
	  {8,
	   1, 
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	  },
  
	  {16,
	   1, 
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	  },
  
	  {32,
	   1, 
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	  },
  
	  {64,
	   2, 
	   {
	     {0, 32, &MPIR_Scatter_MV2_Binomial},
	     {32, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
        };
        table_ptrs[0] = mv2_tmp_scatter_thresholds_table_1ppn;
        mv2_scatter_table_ppn_conf[1] = 2;
        mv2_size_scatter_tuning_table[1] = 6;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_2ppn[] = {
	  {4,
	   2, 
	   {
	     {0, 4096, &MPIR_Scatter_MV2_Binomial},
	     {4096, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Direct},
	   },
	  },
  
	  {8,
	   2, 
	   {
	     {0, 512, &MPIR_Scatter_MV2_two_level_Direct},
	     {512, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
  
	  {16,
	   2, 
	   {
	     {0, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	     {2048, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
  
	  {32,
	   2, 
	   {
	     {0, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	     {2048, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
  
	  {64,
	   2, 
	   {
	     {0, 8192, &MPIR_Scatter_MV2_two_level_Direct},
	     {8192, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
  
	  {128,
	   4, 
	   {
	     {0, 16, &MPIR_Scatter_MV2_Binomial},
	     {16, 128, &MPIR_Scatter_MV2_two_level_Binomial},
	     {128, 16384, &MPIR_Scatter_MV2_two_level_Direct},
	     {16384, -1, &MPIR_Scatter_MV2_Direct},
	   },
	   1,
	   {
	     {0, 128, &MPIR_Scatter_MV2_Direct},
	     {128, -1, &MPIR_Scatter_MV2_Binomial},
	   },
	  },
        };
        table_ptrs[1] = mv2_tmp_scatter_thresholds_table_2ppn;
        mv2_scatter_table_ppn_conf[2] = 16;
        mv2_size_scatter_tuning_table[2] = 8;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_16ppn[] = {
	  {
	    16,
	    2,
	    { 
	      {0, 256, &MPIR_Scatter_MV2_Binomial}, 
	      {256, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    32,
	    2,
	    {
	      {0, 512, &MPIR_Scatter_MV2_Binomial}, 
	      {512, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    64,
	    2,
	    {
	      {0, 1024, &MPIR_Scatter_MV2_two_level_Direct},
	      {1024, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    128,
	    4,
	    {
	      {0, 16, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 16, &MPIR_Scatter_MV2_two_level_Direct},
	      {16, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    256,
	    4,
	    {
	      {0, 16, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 16, &MPIR_Scatter_MV2_two_level_Direct},
	      {16, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, -1,  &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    512,
	    4,
	    {
	      {0, 16, &MPIR_Scatter_mcst_wrap_MV2},
	      {16, 16, &MPIR_Scatter_MV2_two_level_Direct},
	      {16, 4096, &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Binomial},
	    }, 
	  },  
	  {
	    1024,
	    5,
	    {
	      {0, 16, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 16,  &MPIR_Scatter_MV2_Binomial},
	      {16, 32, &MPIR_Scatter_MV2_Binomial},
	      {32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1,
	    {
	      { 0, -1, &MPIR_Scatter_MV2_Binomial},
	    },  
	  },  
	  {
	    2048,
	    7,
	    {
	      {0, 16, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 16,  &MPIR_Scatter_MV2_two_level_Binomial},
	      {16, 128, &MPIR_Scatter_MV2_two_level_Binomial},
	      {128, 1024, &MPIR_Scatter_MV2_two_level_Direct},
	      {1024, 16384, &MPIR_Scatter_MV2_two_level_Direct},
	      {16384, 65536, &MPIR_Scatter_MV2_Direct},
	      {65536, -1, &MPIR_Scatter_MV2_two_level_Direct},
	    },
	    6,
	    {
	      {0, 16, &MPIR_Scatter_MV2_Binomial},
	      {16, 128, &MPIR_Scatter_MV2_Binomial},
	      {128, 1024, &MPIR_Scatter_MV2_Binomial},
	      {1024, 16384, &MPIR_Scatter_MV2_Direct},
	      {16384, 65536, &MPIR_Scatter_MV2_Direct},
	      {65536, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
        };
        table_ptrs[2] = mv2_tmp_scatter_thresholds_table_16ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_tuning_table[i];
        }
        mv2_scatter_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
        MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_tuning_table)
                     * mv2_size_scatter_tuning_table[0]));
        for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
	  mv2_scatter_thresholds_table[i] =
            mv2_scatter_thresholds_table[i - 1]
            + mv2_size_scatter_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_scatter_tuning_table)
                       * mv2_size_scatter_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        /*Trestles*/
        mv2_scatter_num_ppn_conf = 1;
        mv2_scatter_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			* mv2_scatter_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
                                 * mv2_scatter_num_ppn_conf);
        mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
						    mv2_scatter_num_ppn_conf);
        mv2_scatter_table_ppn_conf 
	  = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
        mv2_scatter_table_ppn_conf[0] = 32;
        mv2_size_scatter_tuning_table[0] = 6;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_32ppn[] = {
	  {
	    32,
	    2,
	    {
	      {0, 32, &MPIR_Scatter_MV2_Binomial},
	      {32, -1, &MPIR_Scatter_MV2_two_level_Direct},
	    },
	    2,
	    {
	      {0, 32, &MPIR_Scatter_MV2_Binomial},
	      {32, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
	  {
	    64,
	    3,
	    {
	      {0, 64, &MPIR_Scatter_MV2_Binomial},
	      {64, 1024, &MPIR_Scatter_MV2_two_level_Binomial},
	      {1024, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    3,
	    {
	      {0, 64, &MPIR_Scatter_MV2_Direct},
	      {64, 1024, &MPIR_Scatter_MV2_Direct},
	      {1024, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
	  {
	    128,
	    3,
	    {
	      {0, 64, &MPIR_Scatter_MV2_two_level_Direct},
	      {64, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    3,
	    {
	      {0, 64, &MPIR_Scatter_MV2_Binomial},
	      {64, 2048, &MPIR_Scatter_MV2_Direct},
	      {2048, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
	  {
	    256,
	    3,
	    {
	      {0, 128, &MPIR_Scatter_MV2_two_level_Direct},
	      {128, 8192,  &MPIR_Scatter_MV2_two_level_Direct},
	      {8192, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    3,
	    {
	      {0, 128, &MPIR_Scatter_MV2_Binomial},
	      {128, 8192,  &MPIR_Scatter_MV2_Direct},
	      {8192, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
	  {
	    512,
	    3,
	    {
	      {0, 256, &MPIR_Scatter_MV2_two_level_Direct},
	      {256, 16384,  &MPIR_Scatter_MV2_two_level_Direct},
	      {16384, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    3,
	    {
	      {0, 256, &MPIR_Scatter_MV2_Binomial},
	      {256, 16384,  &MPIR_Scatter_MV2_Direct},
	      {16384, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
	  {
	    1024,
	    3,
	    {
	      {0, 16, &MPIR_Scatter_MV2_two_level_Binomial},
	      {16, 16384,  &MPIR_Scatter_MV2_two_level_Direct},
	      {16384, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    3,
	    {
	      {0, 16, &MPIR_Scatter_MV2_Binomial},
	      {16, 16384,  &MPIR_Scatter_MV2_Direct},
	      {16384, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  }, 
        };
        table_ptrs[0] = mv2_tmp_scatter_thresholds_table_32ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_tuning_table[i];
        }
        mv2_scatter_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
        MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_tuning_table)
                     * mv2_size_scatter_tuning_table[0]));
        for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
	  mv2_scatter_thresholds_table[i] =
            mv2_scatter_thresholds_table[i - 1]
            + mv2_size_scatter_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_scatter_tuning_table)
                       * mv2_size_scatter_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      } else
#elif defined (CHANNEL_NEMESIS_IB)
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
	  mv2_scatter_num_ppn_conf = 1;
	  mv2_scatter_thresholds_table
            = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			  * mv2_scatter_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
				   * mv2_scatter_num_ppn_conf);
	  mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_scatter_num_ppn_conf);
	  mv2_scatter_table_ppn_conf = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
	  mv2_scatter_table_ppn_conf[0] = 12;
	  mv2_size_scatter_tuning_table[0] = 6;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_12ppn[] = {
            {
	      12,
	      2,
	      { 
		{0, 512, &MPIR_Scatter_MV2_Binomial}, 
		{512, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      24,
	      2,
	      {
		{0, 512, &MPIR_Scatter_MV2_Binomial}, 
		{512, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      48,
	      2,
	      {
		{0, 512, &MPIR_Scatter_MV2_Binomial},
		{512, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      96,
	      3,
	      {
		{0, 256, &MPIR_Scatter_MV2_two_level_Direct},
		{256, 8192, &MPIR_Scatter_MV2_two_level_Direct},
		{8192, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      2,
	      {
		{ 0, 256, &MPIR_Scatter_MV2_Binomial},
		{ 256, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      192,
	      3,
	      {
		{1, 2, &MPIR_Scatter_MV2_Binomial},
		{2, 2048, &MPIR_Scatter_MV2_two_level_Direct},
		{2048, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Binomial},
	      },
            },

            {
	      384,
	      3,
	      {
		{1, 32, &MPIR_Scatter_MV2_Binomial},
		{32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
		{4096, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Binomial},
	      },  
            },  
	  };
	  table_ptrs[0] = mv2_tmp_scatter_thresholds_table_12ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_scatter_tuning_table[i];
	  }
	  mv2_scatter_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
	  MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_tuning_table)
		       * mv2_size_scatter_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
            mv2_scatter_thresholds_table[i] =
	      mv2_scatter_thresholds_table[i - 1]
	      + mv2_size_scatter_tuning_table[i - 1];
            MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_tuning_table)
			 * mv2_size_scatter_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
	else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				      MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
	  /*Stampede,*/
	  mv2_scatter_num_ppn_conf = 3;
	  mv2_scatter_thresholds_table
            = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			  * mv2_scatter_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
				   * mv2_scatter_num_ppn_conf);
	  mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_scatter_num_ppn_conf);
	  mv2_scatter_table_ppn_conf 
            = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
	  mv2_scatter_table_ppn_conf[0] = 1;
	  mv2_size_scatter_tuning_table[0] = 6;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_1ppn[] = {
            {2,
	     1, 
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },

            {4,
	     1, 
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
            },
  
            {8,
	     1, 
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
            },
  
            {16,
	     1, 
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
            },
  
            {32,
	     1, 
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
            },
  
            {64,
	     2, 
	     {
	       {0, 32, &MPIR_Scatter_MV2_Binomial},
	       {32, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
	  };
	  table_ptrs[0] = mv2_tmp_scatter_thresholds_table_1ppn;
	  mv2_scatter_table_ppn_conf[1] = 2;
	  mv2_size_scatter_tuning_table[1] = 6;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_2ppn[] = {
            {4,
	     2, 
	     {
	       {0, 4096, &MPIR_Scatter_MV2_Binomial},
	       {4096, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Direct},
	     },
            },
  
            {8,
	     2, 
	     {
	       {0, 512, &MPIR_Scatter_MV2_two_level_Direct},
	       {512, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
  
            {16,
	     2, 
	     {
	       {0, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	       {2048, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
  
            {32,
	     2, 
	     {
	       {0, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	       {2048, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
  
            {64,
	     2, 
	     {
	       {0, 8192, &MPIR_Scatter_MV2_two_level_Direct},
	       {8192, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
  
            {128,
	     4, 
	     {
	       {0, 16, &MPIR_Scatter_MV2_Binomial},
	       {16, 128, &MPIR_Scatter_MV2_two_level_Binomial},
	       {128, 16384, &MPIR_Scatter_MV2_two_level_Direct},
	       {16384, -1, &MPIR_Scatter_MV2_Direct},
	     },
	     1,
	     {
	       {0, 128, &MPIR_Scatter_MV2_Direct},
	       {128, -1, &MPIR_Scatter_MV2_Binomial},
	     },
            },
	  };
	  table_ptrs[1] = mv2_tmp_scatter_thresholds_table_2ppn;
	  mv2_scatter_table_ppn_conf[2] = 16;
	  mv2_size_scatter_tuning_table[2] = 8;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_16ppn[] = {
            {
	      16,
	      2,
	      { 
		{0, 256, &MPIR_Scatter_MV2_Binomial}, 
		{256, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      32,
	      2,
	      {
		{0, 512, &MPIR_Scatter_MV2_Binomial}, 
		{512, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      64,
	      2,
	      {
		{0, 1024, &MPIR_Scatter_MV2_two_level_Direct},
		{1024, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      128,
	      4,
	      {
		{0, 16, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 16, &MPIR_Scatter_MV2_two_level_Direct},
		{16, 2048, &MPIR_Scatter_MV2_two_level_Direct},
		{2048, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      256,
	      4,
	      {
		{0, 16, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 16, &MPIR_Scatter_MV2_two_level_Direct},
		{16, 2048, &MPIR_Scatter_MV2_two_level_Direct},
		{2048, -1,  &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      512,
	      4,
	      {
		{0, 16, &MPIR_Scatter_mcst_wrap_MV2},
		{16, 16, &MPIR_Scatter_MV2_two_level_Direct},
		{16, 4096, &MPIR_Scatter_MV2_two_level_Direct},
		{4096, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Binomial},
	      }, 
            },  
            {
	      1024,
	      5,
	      {
		{0, 16, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 16,  &MPIR_Scatter_MV2_Binomial},
		{16, 32, &MPIR_Scatter_MV2_Binomial},
		{32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
		{4096, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1,
	      {
		{ 0, -1, &MPIR_Scatter_MV2_Binomial},
	      },  
            },  
            {
	      2048,
	      7,
	      {
		{0, 16, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 16,  &MPIR_Scatter_MV2_two_level_Binomial},
		{16, 128, &MPIR_Scatter_MV2_two_level_Binomial},
		{128, 1024, &MPIR_Scatter_MV2_two_level_Direct},
		{1024, 16384, &MPIR_Scatter_MV2_two_level_Direct},
		{16384, 65536, &MPIR_Scatter_MV2_Direct},
		{65536, -1, &MPIR_Scatter_MV2_two_level_Direct},
	      },
	      6,
	      {
		{0, 16, &MPIR_Scatter_MV2_Binomial},
		{16, 128, &MPIR_Scatter_MV2_Binomial},
		{128, 1024, &MPIR_Scatter_MV2_Binomial},
		{1024, 16384, &MPIR_Scatter_MV2_Direct},
		{16384, 65536, &MPIR_Scatter_MV2_Direct},
		{65536, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
	  };
	  table_ptrs[2] = mv2_tmp_scatter_thresholds_table_16ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_scatter_tuning_table[i];
	  }
	  mv2_scatter_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
	  MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_tuning_table)
		       * mv2_size_scatter_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
            mv2_scatter_thresholds_table[i] =
	      mv2_scatter_thresholds_table[i - 1]
	      + mv2_size_scatter_tuning_table[i - 1];
            MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_tuning_table)
			 * mv2_size_scatter_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
	else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				      MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
	  /*Trestles*/
	  mv2_scatter_num_ppn_conf = 1;
	  mv2_scatter_thresholds_table
            = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			  * mv2_scatter_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
				   * mv2_scatter_num_ppn_conf);
	  mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_scatter_num_ppn_conf);
	  mv2_scatter_table_ppn_conf 
            = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
	  mv2_scatter_table_ppn_conf[0] = 32;
	  mv2_size_scatter_tuning_table[0] = 6;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_32ppn[] = {
            {
	      32,
	      2,
	      {
		{0, 32, &MPIR_Scatter_MV2_Binomial},
		{32, -1, &MPIR_Scatter_MV2_two_level_Direct},
	      },
	      2,
	      {
		{0, 32, &MPIR_Scatter_MV2_Binomial},
		{32, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
            {
	      64,
	      3,
	      {
		{0, 64, &MPIR_Scatter_MV2_Binomial},
		{64, 1024, &MPIR_Scatter_MV2_two_level_Binomial},
		{1024, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      3,
	      {
		{0, 64, &MPIR_Scatter_MV2_Direct},
		{64, 1024, &MPIR_Scatter_MV2_Direct},
		{1024, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
            {
	      128,
	      3,
	      {
		{0, 64, &MPIR_Scatter_MV2_two_level_Direct},
		{64, 2048, &MPIR_Scatter_MV2_two_level_Direct},
		{2048, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      3,
	      {
		{0, 64, &MPIR_Scatter_MV2_Binomial},
		{64, 2048, &MPIR_Scatter_MV2_Direct},
		{2048, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
            {
	      256,
	      3,
	      {
		{0, 128, &MPIR_Scatter_MV2_two_level_Direct},
		{128, 8192,  &MPIR_Scatter_MV2_two_level_Direct},
		{8192, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      3,
	      {
		{0, 128, &MPIR_Scatter_MV2_Binomial},
		{128, 8192,  &MPIR_Scatter_MV2_Direct},
		{8192, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
            {
	      512,
	      3,
	      {
		{0, 256, &MPIR_Scatter_MV2_two_level_Direct},
		{256, 16384,  &MPIR_Scatter_MV2_two_level_Direct},
		{16384, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      3,
	      {
		{0, 256, &MPIR_Scatter_MV2_Binomial},
		{256, 16384,  &MPIR_Scatter_MV2_Direct},
		{16384, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
            {
	      1024,
	      3,
	      {
		{0, 16, &MPIR_Scatter_MV2_two_level_Binomial},
		{16, 16384,  &MPIR_Scatter_MV2_two_level_Direct},
		{16384, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      3,
	      {
		{0, 16, &MPIR_Scatter_MV2_Binomial},
		{16, 16384,  &MPIR_Scatter_MV2_Direct},
		{16384, -1, &MPIR_Scatter_MV2_Direct},
	      },
            }, 
	  };
	  table_ptrs[0] = mv2_tmp_scatter_thresholds_table_32ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_scatter_tuning_table[i];
	  }
	  mv2_scatter_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
	  MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_tuning_table)
		       * mv2_size_scatter_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
            mv2_scatter_thresholds_table[i] =
	      mv2_scatter_thresholds_table[i - 1]
	      + mv2_size_scatter_tuning_table[i - 1];
            MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_tuning_table)
			 * mv2_size_scatter_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	} else {
	  mv2_scatter_num_ppn_conf = 1;
	  mv2_scatter_thresholds_table
            = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			  * mv2_scatter_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
				   * mv2_scatter_num_ppn_conf);
	  mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_scatter_num_ppn_conf);
	  mv2_scatter_table_ppn_conf 
            = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
	  mv2_scatter_table_ppn_conf[0] = 8;
	  mv2_size_scatter_tuning_table[0] = 7;
	  mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_8ppn[] = {
            {
	      8,
	      2,
	      { 
		{0, 256, &MPIR_Scatter_MV2_Binomial}, 
		{256, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      16,
	      2,
	      {
		{0, 512, &MPIR_Scatter_MV2_Binomial}, 
		{512, -1, &MPIR_Scatter_MV2_Direct},
	      },
	      1, 
	      { 
		{ 0, -1, &MPIR_Scatter_MV2_Direct},
	      },
            },

            {
	      32,
	      3,
	      {
		{0, 256, &MPIR_Scatter_MV2_two_level_Direct},
		{256, 2048, &MPIR_Scatter_MV2_two_level_Direct},
		{2048, -1,  &MPIR_Scatter_MV2_Direct},
	      },
	      2,
	      {
		{ 0, 256, &MPIR_Scatter_MV2_Direct},
		{ 256, -1, &MPIR_Scatter_MV2_Binomial},
	      },
            },

            {
	      64,
	      6,
	      {
		{0, 32, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 32, &MPIR_Scatter_MV2_two_level_Direct},
		{32, 256, &MPIR_Scatter_MV2_two_level_Direct},
		{256, 2048,  &MPIR_Scatter_MV2_two_level_Direct},
		{2048, 65536, &MPIR_Scatter_MV2_Direct},
		{65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	      },
	      2,
	      {
		{ 0, 256, &MPIR_Scatter_MV2_Direct},
		{ 256, -1, &MPIR_Scatter_MV2_Binomial},
	      },
            },

            {
	      128,
	      5,
	      {
		{0, 64, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 64, &MPIR_Scatter_MV2_Binomial},
		{64, 4096, &MPIR_Scatter_MV2_two_level_Direct},
		{4096, 65536, &MPIR_Scatter_MV2_Direct},
		{65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	      },
	      2,
	      {
		{ 0, 1024, &MPIR_Scatter_MV2_Direct},
		{ 1024, -1, &MPIR_Scatter_MV2_Binomial},
	      },
            },

            {
	      256,
	      6,
	      {
		{0, 64, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 64, &MPIR_Scatter_MV2_Binomial},
		{64, 256,  &MPIR_Scatter_MV2_two_level_Direct},
		{256, 4096,  &MPIR_Scatter_MV2_two_level_Direct},
		{4096, 65536, &MPIR_Scatter_MV2_Direct},
		{65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	      },
	      2,
	      {
		{ 0, 256, &MPIR_Scatter_MV2_Binomial},
		{ 256, -1, &MPIR_Scatter_MV2_Binomial},
	      },  
            },  

            {
	      512,
	      5,
	      {
		{0, 32, &MPIR_Scatter_mcst_wrap_MV2},
		{0, 32, &MPIR_Scatter_MV2_Binomial},
		{32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
		{4096, 32768, &MPIR_Scatter_MV2_Direct},
		{32768, -1, &MPIR_Scatter_MV2_two_level_Direct},
	      },
	      2,
	      {
		{ 0, 1024, &MPIR_Scatter_MV2_Binomial},
		{ 1024, -1, &MPIR_Scatter_MV2_Binomial},
	      },
            },
	  };
	  table_ptrs[0] = mv2_tmp_scatter_thresholds_table_8ppn;
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_scatter_tuning_table[i];
	  }
	  mv2_scatter_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
	  MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_scatter_tuning_table)
		       * mv2_size_scatter_tuning_table[0]));
	  for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
            mv2_scatter_thresholds_table[i] =
	      mv2_scatter_thresholds_table[i - 1]
	      + mv2_size_scatter_tuning_table[i - 1];
            MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_scatter_tuning_table)
			 * mv2_size_scatter_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
#endif
#endif /* !CHANNEL_PSM */
      {
        mv2_scatter_num_ppn_conf = 1;
        mv2_scatter_thresholds_table
	  = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
			* mv2_scatter_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_scatter_tuning_table *)
                                 * mv2_scatter_num_ppn_conf);
        mv2_size_scatter_tuning_table = MPIU_Malloc(sizeof(int) *
						    mv2_scatter_num_ppn_conf);
        mv2_scatter_table_ppn_conf 
	  = MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
        mv2_scatter_table_ppn_conf[0] = 8;
        mv2_size_scatter_tuning_table[0] = 7;
        mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table_8ppn[] = {
	  {
	    8,
	    2,
	    { 
	      {0, 256, &MPIR_Scatter_MV2_Binomial}, 
	      {256, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    16,
	    2,
	    {
	      {0, 512, &MPIR_Scatter_MV2_Binomial}, 
	      {512, -1, &MPIR_Scatter_MV2_Direct},
	    },
	    1, 
	    { 
	      { 0, -1, &MPIR_Scatter_MV2_Direct},
	    },
	  },

	  {
	    32,
	    3,
	    {
	      {0, 256, &MPIR_Scatter_MV2_two_level_Direct},
	      {256, 2048, &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, -1,  &MPIR_Scatter_MV2_Direct},
	    },
	    2,
	    {
	      { 0, 256, &MPIR_Scatter_MV2_Direct},
	      { 256, -1, &MPIR_Scatter_MV2_Binomial},
	    },
	  },

	  {
	    64,
	    6,
	    {
	      {0, 32, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 32, &MPIR_Scatter_MV2_two_level_Direct},
	      {32, 256, &MPIR_Scatter_MV2_two_level_Direct},
	      {256, 2048,  &MPIR_Scatter_MV2_two_level_Direct},
	      {2048, 65536, &MPIR_Scatter_MV2_Direct},
	      {65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	    },
	    2,
	    {
	      { 0, 256, &MPIR_Scatter_MV2_Direct},
	      { 256, -1, &MPIR_Scatter_MV2_Binomial},
	    },
	  },

	  {
	    128,
	    5,
	    {
	      {0, 64, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 64, &MPIR_Scatter_MV2_Binomial},
	      {64, 4096, &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, 65536, &MPIR_Scatter_MV2_Direct},
	      {65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	    },
	    2,
	    {
	      { 0, 1024, &MPIR_Scatter_MV2_Direct},
	      { 1024, -1, &MPIR_Scatter_MV2_Binomial},
	    },
	  },

	  {
	    256,
	    6,
	    {
	      {0, 64, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 64, &MPIR_Scatter_MV2_Binomial},
	      {64, 256,  &MPIR_Scatter_MV2_two_level_Direct},
	      {256, 4096,  &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, 65536, &MPIR_Scatter_MV2_Direct},
	      {65536, -1, &MPIR_Scatter_MV2_Direct_Blk},
	    },
	    2,
	    {
	      { 0, 256, &MPIR_Scatter_MV2_Binomial},
	      { 256, -1, &MPIR_Scatter_MV2_Binomial},
	    },  
	  },  

	  {
	    512,
	    5,
	    {
	      {0, 32, &MPIR_Scatter_mcst_wrap_MV2},
	      {0, 32, &MPIR_Scatter_MV2_Binomial},
	      {32, 4096, &MPIR_Scatter_MV2_two_level_Direct},
	      {4096, 32768, &MPIR_Scatter_MV2_Direct},
	      {32768, -1, &MPIR_Scatter_MV2_two_level_Direct},
	    },
	    2,
	    {
	      { 0, 1024, &MPIR_Scatter_MV2_Binomial},
	      { 1024, -1, &MPIR_Scatter_MV2_Binomial},
	    },
	  },
        };
        table_ptrs[0] = mv2_tmp_scatter_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_scatter_num_ppn_conf; i++) {
	  agg_table_sum += mv2_size_scatter_tuning_table[i];
        }
        mv2_scatter_thresholds_table[0] =
	  MPIU_Malloc(agg_table_sum * sizeof (mv2_scatter_tuning_table));
        MPIU_Memcpy(mv2_scatter_thresholds_table[0], table_ptrs[0],
		    (sizeof(mv2_scatter_tuning_table)
                     * mv2_size_scatter_tuning_table[0]));
        for (i = 1; i < mv2_scatter_num_ppn_conf; i++) {
	  mv2_scatter_thresholds_table[i] =
            mv2_scatter_thresholds_table[i - 1]
            + mv2_size_scatter_tuning_table[i - 1];
	  MPIU_Memcpy(mv2_scatter_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_scatter_tuning_table)
                       * mv2_size_scatter_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
      }
    }
}

void MV2_cleanup_scatter_tuning_table()
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_scatter_tuning) {
    MPIU_Free(mv2_scatter_indexed_thresholds_table[0]);
    MPIU_Free(mv2_scatter_indexed_table_ppn_conf);
    MPIU_Free(mv2_size_scatter_indexed_tuning_table);
    if (mv2_scatter_indexed_thresholds_table != NULL) {
      MPIU_Free(mv2_scatter_indexed_thresholds_table);
    }
  }
  else {
    MPIU_Free(mv2_scatter_thresholds_table[0]);
    MPIU_Free(mv2_scatter_table_ppn_conf);
    MPIU_Free(mv2_size_scatter_tuning_table);
    if (mv2_scatter_thresholds_table != NULL) {
        MPIU_Free(mv2_scatter_thresholds_table);
    }
  }
}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}


int MV2_internode_Scatter_is_define(char *mv2_user_scatter_inter, char
                                    *mv2_user_scatter_intra)
{
  int i = 0;
  int nb_element = count_sep(mv2_user_scatter_inter) + 1;

  if (mv2_use_indexed_tuning || mv2_use_indexed_scatter_tuning) {

    /* If one scatter tuning table is already defined */
    if (mv2_scatter_indexed_thresholds_table != NULL) {
      if (mv2_scatter_indexed_thresholds_table[0] != NULL) {
          MPIU_Free(mv2_scatter_indexed_thresholds_table[0]);
      }
      MPIU_Free(mv2_scatter_indexed_thresholds_table);
    }

    mv2_scatter_indexed_tuning_table mv2_tmp_scatter_indexed_thresholds_table[1];
    mv2_scatter_indexed_num_ppn_conf = 1;
    if (mv2_size_scatter_indexed_tuning_table == NULL) {
      mv2_size_scatter_indexed_tuning_table =
	MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
    }
    mv2_size_scatter_indexed_tuning_table[0] = 1;

    if (mv2_scatter_indexed_table_ppn_conf == NULL) {
      mv2_scatter_indexed_table_ppn_conf =
	MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf * sizeof(int));
    }
    /* -1 indicates user defined algorithm */
    mv2_scatter_indexed_table_ppn_conf[0] = -1;

    /* We realloc the space for the new scatter tuning table */
    mv2_scatter_indexed_thresholds_table =
      MPIU_Malloc(mv2_scatter_indexed_num_ppn_conf *
		  sizeof(mv2_scatter_indexed_tuning_table *));
    mv2_scatter_indexed_thresholds_table[0] =
      MPIU_Malloc(mv2_size_scatter_indexed_tuning_table[0] *
		  sizeof(mv2_scatter_indexed_tuning_table));

    if (nb_element == 1) {
      mv2_tmp_scatter_indexed_thresholds_table[0].numproc = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].size_inter_table = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].size_intra_table = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].msg_sz = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].msg_sz = 1;
	
      switch (atoi(mv2_user_scatter_inter)) {
      case SCATTER_BINOMIAL:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Binomial;
	break;
      case SCATTER_DIRECT:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
	break;
      case SCATTER_TWO_LEVEL_BINOMIAL:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_two_level_Binomial;
	break;
      case SCATTER_TWO_LEVEL_DIRECT:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_two_level_Direct;
	break;
#if defined(_MCST_SUPPORT_)
      case SCATTER_MCAST:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_mcst_wrap_MV2;
	break;
#endif /* #if defined(_MCST_SUPPORT_) */
	  /******************** added by abu naser ************/
	case SCATTER_CHS:
	  mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_CHS;
	  break;
	case SCATTER_RR:
	  mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem_intra_RR;
	  break;
	case SCATTER_RR_UNEN:
	  mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC;
	  break;
	case SCATTER_HBCAST:
	  mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_HBcast;
	  break; 
	case SCATTTER_C:
	  mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem;
	  break;          
	/***************** end of add *************************/ 	
      default:
	mv2_tmp_scatter_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Binomial;
      }
    }
    mv2_tmp_scatter_indexed_thresholds_table[0].size_intra_table = 1;

    if (mv2_user_scatter_intra != NULL) {
      i = 0;
      nb_element = count_sep(mv2_user_scatter_intra) + 1;        
      if (nb_element == 1) {
	mv2_tmp_scatter_indexed_thresholds_table[0].size_intra_table = 1;
	mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].msg_sz = 1;
    
	switch (atoi(mv2_user_scatter_intra)) {
	case SCATTER_DIRECT:
	  mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct;
	  break;
	case SCATTER_BINOMIAL:
	  mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Binomial;
	  break;
	default:
	  mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct;
	}
      }
    }
    else {
      mv2_tmp_scatter_indexed_thresholds_table[0].size_intra_table = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].msg_sz = 1;
      mv2_tmp_scatter_indexed_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	&MPIR_Scatter_MV2_Direct;
    }
    MPIU_Memcpy(mv2_scatter_indexed_thresholds_table[0], mv2_tmp_scatter_indexed_thresholds_table, sizeof
		(mv2_scatter_indexed_tuning_table));
  }
  else
    {
      mv2_scatter_tuning_table mv2_tmp_scatter_thresholds_table[1];
      mv2_scatter_num_ppn_conf = 1;
      if (mv2_size_scatter_tuning_table == NULL) {
	mv2_size_scatter_tuning_table =
	  MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
      }
      mv2_size_scatter_tuning_table[0] = 1;

      if (mv2_scatter_table_ppn_conf == NULL) {
	mv2_scatter_table_ppn_conf =
	  MPIU_Malloc(mv2_scatter_num_ppn_conf * sizeof(int));
      }
      mv2_scatter_table_ppn_conf[0] = -1;

      /* If one scatter tuning table is already defined */
      if (mv2_scatter_thresholds_table != NULL) {
	MPIU_Free(mv2_scatter_thresholds_table);
      }
      /* We realloc the space for the new scatter tuning table */
      mv2_scatter_thresholds_table =
	MPIU_Malloc(mv2_scatter_num_ppn_conf *
		    sizeof(mv2_scatter_tuning_table *));
      mv2_scatter_thresholds_table[0] =
	MPIU_Malloc(mv2_size_scatter_tuning_table[0] *
		    sizeof(mv2_scatter_tuning_table));

      if (nb_element == 1) {
	mv2_tmp_scatter_thresholds_table[0].numproc = 1;
	mv2_tmp_scatter_thresholds_table[0].size_inter_table = 1;
	mv2_tmp_scatter_thresholds_table[0].size_intra_table = 1;
	mv2_tmp_scatter_thresholds_table[0].inter_leader[0].min = 0;
	mv2_tmp_scatter_thresholds_table[0].inter_leader[0].max = -1;
	mv2_tmp_scatter_thresholds_table[0].intra_node[0].min = 0;
	mv2_tmp_scatter_thresholds_table[0].intra_node[0].max = -1;
    
	switch (atoi(mv2_user_scatter_inter)) {
	case SCATTER_BINOMIAL:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Binomial;
	  break;
	case SCATTER_DIRECT:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct;
	  break;
	case SCATTER_TWO_LEVEL_BINOMIAL:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_two_level_Binomial;
	  break;
	case SCATTER_TWO_LEVEL_DIRECT:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_two_level_Direct;
	  break;
#if defined(_MCST_SUPPORT_)
	case SCATTER_MCAST:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_mcst_wrap_MV2;
	  break;
#endif /* #if defined(_MCST_SUPPORT_) */
    /******************** added by abu naser ************/
	case SCATTER_CHS:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_CHS;
	  break;
	case SCATTER_RR:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem_intra_RR;
	  break;
	case SCATTER_RR_UNEN:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC;
	  break;
	case SCATTER_HBCAST:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_HBcast;
	  break; 
	case SCATTTER_C:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct_no_shmem;
	  break;        
	/***************** end of add *************************/  
	default:
	  mv2_tmp_scatter_thresholds_table[0].inter_leader[0].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Binomial;
	}
        
      }
      else {
	char *dup, *p, *save_p;
	regmatch_t match[NMATCH];
	regex_t preg;
	const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

	if (!(dup = MPIU_Strdup(mv2_user_scatter_inter))) {
	  fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_scatter_inter);
	  return 1;
	}

	if (regcomp(&preg, regexp, REG_EXTENDED)) {
	  fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_scatter_inter);
	  MPIU_Free(dup);
	  return 2;
	}

	mv2_tmp_scatter_thresholds_table[0].numproc = 1;
	mv2_tmp_scatter_thresholds_table[0].size_inter_table = nb_element;
	mv2_tmp_scatter_thresholds_table[0].size_intra_table = 2;
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

	  case SCATTER_BINOMIAL:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Binomial;
	    break;
	  case SCATTER_DIRECT:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Direct;
	    break;
	  case SCATTER_TWO_LEVEL_BINOMIAL:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_two_level_Binomial;
	    break;
	  case SCATTER_TWO_LEVEL_DIRECT:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_two_level_Direct;
	    break;
#if defined(_MCST_SUPPORT_)
	  case SCATTER_MCAST:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_mcst_wrap_MV2;
	    break;
#endif /* #if defined(_MCST_SUPPORT_) */
	  default:
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Binomial;
	  }

	  mv2_tmp_scatter_thresholds_table[0].inter_leader[i].min = atoi(p +
									 match[2].rm_so);
	  if (p[match[3].rm_so] == '+') {
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].max = -1;
	  } else {
	    mv2_tmp_scatter_thresholds_table[0].inter_leader[i].max =
	      atoi(p + match[3].rm_so);
	  }
	  i++;
	}
	MPIU_Free(dup);
	regfree(&preg);
      }
      mv2_tmp_scatter_thresholds_table[0].size_intra_table = 2;

      if (mv2_user_scatter_intra != NULL) {
	i = 0;
	nb_element = count_sep(mv2_user_scatter_intra) + 1;        
	if (nb_element == 1) {
	  mv2_tmp_scatter_thresholds_table[0].size_intra_table = 1;
	  mv2_tmp_scatter_thresholds_table[0].intra_node[0].min = 0;
	  mv2_tmp_scatter_thresholds_table[0].intra_node[0].max = -1;
    
	  switch (atoi(mv2_user_scatter_intra)) {
	  case SCATTER_DIRECT:
	    mv2_tmp_scatter_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Direct;
	    break;
	  case SCATTER_BINOMIAL:
	    mv2_tmp_scatter_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Binomial;
	    break;
	  default:
	    mv2_tmp_scatter_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	      &MPIR_Scatter_MV2_Direct;
	  }
        
	}
	else {
	  char *dup, *p, *save_p;
	  regmatch_t match[NMATCH];
	  regex_t preg;
	  const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

	  if (!(dup = MPIU_Strdup(mv2_user_scatter_intra))) {
	    fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_scatter_intra);
	    return 1;
	  }

	  if (regcomp(&preg, regexp, REG_EXTENDED)) {
	    fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_scatter_intra);
	    MPIU_Free(dup);
	    return 2;
	  }

	  mv2_tmp_scatter_thresholds_table[0].numproc = 1;
	  mv2_tmp_scatter_thresholds_table[0].size_intra_table = 2;
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

	    case SCATTER_DIRECT:
	      mv2_tmp_scatter_thresholds_table[0].intra_node[i].MV2_pt_Scatter_function =
		&MPIR_Scatter_MV2_Direct;
	      break;
	    case SCATTER_BINOMIAL:
	      mv2_tmp_scatter_thresholds_table[0].intra_node[i].MV2_pt_Scatter_function =
		&MPIR_Scatter_MV2_Binomial;
	      break;
	    default:
	      mv2_tmp_scatter_thresholds_table[0].intra_node[i].MV2_pt_Scatter_function =
		&MPIR_Scatter_MV2_Direct;
	    }

	    mv2_tmp_scatter_thresholds_table[0].intra_node[i].min = atoi(p +
									 match[2].rm_so);
	    if (p[match[3].rm_so] == '+') {
	      mv2_tmp_scatter_thresholds_table[0].intra_node[i].max = -1;
	    } else {
	      mv2_tmp_scatter_thresholds_table[0].intra_node[i].max =
		atoi(p + match[3].rm_so);
	    }
	    i++;
	  }
	  MPIU_Free(dup);
	  regfree(&preg);
	}
      }
      else {
	mv2_tmp_scatter_thresholds_table[0].size_intra_table = 1;
	mv2_tmp_scatter_thresholds_table[0].intra_node[0].min = 0;
	mv2_tmp_scatter_thresholds_table[0].intra_node[0].max = -1;
	mv2_tmp_scatter_thresholds_table[0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
      }
      MPIU_Memcpy(mv2_scatter_thresholds_table[0], mv2_tmp_scatter_thresholds_table, sizeof
		  (mv2_scatter_tuning_table));
    }

  return 0;
}

int MV2_intranode_Scatter_is_define(char *mv2_user_scatter_intra)
{
  int i = 0;
  int nb_element = count_sep(mv2_user_scatter_intra) + 1;

  if (mv2_use_indexed_tuning || mv2_use_indexed_scatter_tuning) {
    if (nb_element == 1) {
      mv2_scatter_indexed_thresholds_table[0][0].size_intra_table = 1;
      mv2_scatter_indexed_thresholds_table[0][0].intra_node[0].msg_sz = 1;
    
      switch (atoi(mv2_user_scatter_intra)) {
      case SCATTER_DIRECT:
	mv2_scatter_indexed_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
	break;
      case SCATTER_BINOMIAL:
	mv2_scatter_indexed_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Binomial;
	break;
      default:
	mv2_scatter_indexed_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
      }
    }
  }
  else {
    if (nb_element == 1) {
      mv2_scatter_thresholds_table[0][0].size_intra_table = 1;
      mv2_scatter_thresholds_table[0][0].intra_node[0].min = 0;
      mv2_scatter_thresholds_table[0][0].intra_node[0].max = -1;
    
      switch (atoi(mv2_user_scatter_intra)) {
      case SCATTER_DIRECT:
	mv2_scatter_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
	break;
      case SCATTER_BINOMIAL:
	mv2_scatter_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Binomial;
	break;
      default:
	mv2_scatter_thresholds_table[0][0].intra_node[0].MV2_pt_Scatter_function =
	  &MPIR_Scatter_MV2_Direct;
      }
        
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_scatter_intra))) {
	fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_scatter_intra);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_scatter_intra);
	MPIU_Free(dup);
	return 2;
      }

      mv2_scatter_thresholds_table[0][0].numproc = 1;
      mv2_scatter_thresholds_table[0][0].size_intra_table = 2;
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

	case SCATTER_DIRECT:
	  mv2_scatter_thresholds_table[0][0].intra_node[i].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct;
	  break;
	case SCATTER_BINOMIAL:
	  mv2_scatter_thresholds_table[0][0].intra_node[i].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Binomial;
	  break;
	default:
	  mv2_scatter_thresholds_table[0][0].intra_node[i].MV2_pt_Scatter_function =
	    &MPIR_Scatter_MV2_Direct;
	}

	mv2_scatter_thresholds_table[0][0].intra_node[i].min = atoi(p +
								    match[2].rm_so);
	if (p[match[3].rm_so] == '+') {
	  mv2_scatter_thresholds_table[0][0].intra_node[i].max = -1;
	} else {
	  mv2_scatter_thresholds_table[0][0].intra_node[i].max =
	    atoi(p + match[3].rm_so);
	}
	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
  }
  return 0;
}
