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
#include "bcast_tuning.h"
#include "tuning/bcast_arch_tuning.h"
#include "mv2_arch_hca_detect.h"
/* array used to tune bcast */

int mv2_size_bcast_tuning_table = 0;
mv2_bcast_tuning_table *mv2_bcast_thresholds_table = NULL;

int *mv2_bcast_indexed_table_ppn_conf = NULL;
int mv2_bcast_indexed_num_ppn_conf = 1;
int *mv2_size_bcast_indexed_tuning_table = NULL;
mv2_bcast_indexed_tuning_table **mv2_bcast_indexed_thresholds_table = NULL;

int MV2_set_bcast_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
  
    if (mv2_use_indexed_tuning || mv2_use_indexed_bcast_tuning) {
      int agg_table_sum = 0;
      int i;
      mv2_bcast_indexed_tuning_table **table_ptrs = NULL;
#if defined(CHANNEL_MRAIL_GEN2)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /* Lonestar Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 3;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 12;
      mv2_size_bcast_indexed_tuning_table[2] = 6;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_12ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    if ((MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                 MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20, MV2_HCA_MLX_CX_CONNIB) ||
        MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                 MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20, MV2_HCA_MLX_CX_CONNIB)) && !heterogeneity) {
      /* PSG Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 3;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__1PPN;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__2PPN;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 20;
      mv2_size_bcast_indexed_tuning_table[2] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_20ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN;
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_20ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12, MV2_HCA_MLX_CX_CONNIB) && !heterogeneity) {
      /* Wilkes Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 6;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 6;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 12;
      mv2_size_bcast_indexed_tuning_table[2] = 6;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_12ppn[] =
	GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__12PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 3;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 32;
#if defined(_SMP_CMA_)
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_32ppn[] =
        GEN2_CMA__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_32ppn[] =
        GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_bcast_indexed_thresholds_table_32ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_32ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[2] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_32ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_32ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[0] = 6;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[1] = 7;
	table_ptrs[1] = mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[1] = 7;
	table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[1] = 7;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[2] = 7;
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[0] = 3;
	table_ptrs[0] = mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[0] = 2;
	table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[0] = 2;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[1] = 3;
	table_ptrs[1] = mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[1] = 2;
	table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[1] = 2;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[2] = 5;
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_bcast_indexed_num_ppn_conf = 4;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
#endif


       mv2_bcast_indexed_table_ppn_conf[2] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_4ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_4ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_cma_bcast_indexed_thresholds_table_4ppn;
  }
    else {
       mv2_size_bcast_indexed_tuning_table[2] = 1;
       table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_4ppn;
   }
#else
    mv2_size_bcast_indexed_tuning_table[2] = 1;
    table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_4ppn;

#endif     


      mv2_bcast_indexed_table_ppn_conf[3] = 16;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[3] = 9;
	table_ptrs[3] = mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[3] = 6;
	table_ptrs[3] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[3] = 6;
      table_ptrs[3] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /*RI2 Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  5, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  6, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  6, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28, 6, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 28, 6, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7551_64, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* AMD EPYC table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 7)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  3, GEN2__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  3, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  4, GEN2__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  4, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  4, GEN2__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  4, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  4, GEN2__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  4, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 4, GEN2__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 4, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 4, GEN2__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 32, 4, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 64, 4, GEN2__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 64, 4, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7742_128, MV2_HCA_ANY) && !heterogeneity) {
      /* AMD EPYC rome table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 7)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  1, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  2, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  2, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  2, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 2, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 2, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 64, 2, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* Haswell HDR nodes on NOWLAB */
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Frontera */
      MV2_COLL_TUNING_START_TABLE  (bcast, 8)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  5, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  7, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  7, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  7, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  7, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  7, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  7, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 7, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 7, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28, 7, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 28, 7, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 7, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 32, 7, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 56, 7, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 56, 7, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Mayer */
      MV2_COLL_TUNING_START_TABLE  (bcast, 8)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_32, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Catalyst */
      MV2_COLL_TUNING_START_TABLE  (bcast, 7)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /* ARM system at Hartree Center */
      MV2_COLL_TUNING_START_TABLE  (bcast, 5)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  2, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 24,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__24PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER8, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Ray Table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 3)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  4, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER9, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Sierra Table: Use table for Ray temporarily */
      MV2_COLL_TUNING_START_TABLE  (bcast, 9)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  4, GEN2__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  4, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  5, GEN2__IBM_POWER9_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, GEN2__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, GEN2__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, GEN2__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 22, 5, GEN2__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 22, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 5, GEN2__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 32, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 44, 5, GEN2__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 44, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*RI Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 4)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  6, GEN2__RI__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  6, GEN2_CMA__RI__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  7, GEN2__RI__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  7, GEN2_CMA__RI__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  1, GEN2__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  1, GEN2_CMA__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  8, GEN2__RI__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  8, GEN2_CMA__RI__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Comet Table*/
      mv2_bcast_indexed_num_ppn_conf = 1;
      mv2_bcast_indexed_thresholds_table
        = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
                      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
                               * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
                                                        mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));

      mv2_bcast_indexed_table_ppn_conf[0] = 24;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_24ppn[] =
        GEN2__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
      /*
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_24ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
        mv2_size_bcast_indexed_tuning_table[0] = 6;
        table_ptrs[0] = mv2_tmp_cma_bcast_indexed_thresholds_table_24ppn;
      }
      else {
        mv2_size_bcast_indexed_tuning_table[0] = 6;
        table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_24ppn;
      }
#else
      */
      mv2_size_bcast_indexed_tuning_table[0] = 6;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_24ppn;
      /*
#endif
      */

      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
        agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
        MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
                  (sizeof(mv2_bcast_indexed_tuning_table)
                   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
        mv2_bcast_indexed_thresholds_table[i] =
          mv2_bcast_indexed_thresholds_table[i - 1]
          + mv2_size_bcast_indexed_tuning_table[i - 1];
        MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
                    (sizeof(mv2_bcast_indexed_tuning_table)
                     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_ANY, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /*RI2 Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  5, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  6, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  6, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28, 6, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 28, 6, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_ANY, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /*RI2 Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 1,  5, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  6, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 2,  6, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28, 6, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (bcast, 28, 6, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else {
      /*Stampede Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_cma_bcast_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_cma_bcast_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
#endif
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_bcast_indexed_tuning_table mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_bcast_indexed_tuning_table[2] = 9;
	table_ptrs[2] = mv2_tmp_cma_bcast_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_bcast_indexed_tuning_table[2] = 6;
	table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_bcast_indexed_tuning_table[2] = 6;
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#elif defined(CHANNEL_NEMESIS_IB)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 32;
      mv2_size_bcast_indexed_tuning_table[2] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_32ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_32ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_size_bcast_indexed_tuning_table[2] = 4;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_16PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 2;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_size_bcast_indexed_tuning_table[2] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_size_bcast_indexed_tuning_table[2] = 7;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else  {
      /*Stampede Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_size_bcast_indexed_tuning_table[2] = 7;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#elif defined(CHANNEL_PSM)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_QLGIC_QIB) && !heterogeneity) {
      /*Sierra Table*/
      mv2_bcast_indexed_num_ppn_conf = 2;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__1PPN;
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 12;
      mv2_size_bcast_indexed_tuning_table[1] = 6;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_32ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__12PPN;
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_32ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /*Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 5)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 36, 5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-KNL Table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 32, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 64, 4, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_PLATINUM_8170_2S_52, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-Skylake Table */
      MV2_COLL_TUNING_START_TABLE  (bcast, 7)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 24, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__24PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 48, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__48PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
    else {
      /*default psm table: Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (bcast, 6)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (bcast, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (bcast)
    }
#else
    {
      /*Stampede Table*/
      mv2_bcast_indexed_num_ppn_conf = 3;
      mv2_bcast_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
		      * mv2_bcast_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_bcast_indexed_tuning_table *)
			       * mv2_bcast_indexed_num_ppn_conf);
      mv2_size_bcast_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_bcast_indexed_num_ppn_conf);
      mv2_bcast_indexed_table_ppn_conf = MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
      
      mv2_bcast_indexed_table_ppn_conf[0] = 1;
      mv2_size_bcast_indexed_tuning_table[0] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_bcast_indexed_thresholds_table_1ppn;
      
      mv2_bcast_indexed_table_ppn_conf[1] = 2;
      mv2_size_bcast_indexed_tuning_table[1] = 5;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_bcast_indexed_thresholds_table_2ppn;
      
      mv2_bcast_indexed_table_ppn_conf[2] = 16;
      mv2_size_bcast_indexed_tuning_table[2] = 7;
      mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_bcast_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_bcast_indexed_tuning_table[i];
      }
      mv2_bcast_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_bcast_indexed_tuning_table));
      MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_bcast_indexed_tuning_table)
		   * mv2_size_bcast_indexed_tuning_table[0]));
      for (i = 1; i < mv2_bcast_indexed_num_ppn_conf; i++) {
	mv2_bcast_indexed_thresholds_table[i] =
	  mv2_bcast_indexed_thresholds_table[i - 1]
	  + mv2_size_bcast_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_bcast_indexed_tuning_table)
		     * mv2_size_bcast_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#endif
    }
    else {
#if defined(CHANNEL_MRAIL_GEN2)
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        /*Trestles Table*/
        mv2_size_bcast_tuning_table=6;
        mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
                                                 sizeof (mv2_bcast_tuning_table));

	mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[]={
	  {
	    32,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 0, 1, 1, 1},
	    9,
	    {
	      {0, 64, &MPIR_Bcast_binomial_MV2, -1},
	      {64, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {128, 16384, &MPIR_Bcast_binomial_MV2, -1},
	      {16384, 32768, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {32768, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {65536, 131072, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {131072, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	      {262144, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Bcast_binomial_MV2, -1}
	    },
	    9,
	    {
	      {0, 64, &MPIR_Shmem_Bcast_MV2, -1},
	      {64, 128, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {128, 16384, &MPIR_Shmem_Bcast_MV2, -1},
	      {16384, 32768, &MPIR_Shmem_Bcast_MV2, -1},
	      {32768, 65536, &MPIR_Knomial_Bcast_intra_node_MV2, 8},
	      {65536, 131072, NULL, -1},
	      {131072, 262144, &MPIR_Shmem_Bcast_MV2, -1},
	      {262144, 524288, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    64,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0},
	    11,
	    {
	      {0, 8, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {8, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {16, 32, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {32, 64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {64, 256, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {256, 512, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {512, 4096, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {4096, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {131072, 262144, &MPIR_Bcast_binomial_MV2, -1},
	      {262144, 524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	    },
	    11,
	    {
	      {0, 8, &MPIR_Shmem_Bcast_MV2, -1},
	      {8, 16, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {16, 32, &MPIR_Shmem_Bcast_MV2, -1},
	      {32, 64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {64, 256, &MPIR_Shmem_Bcast_MV2, -1},
	      {256, 512, NULL, -1},
	      {512, 4096, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {4096, 131072, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {131072, 262144, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {262144, 524288, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {524288, -1, NULL, -1}
	    }
	  },
	  {
	    128,
	    8192, 4, 4,
	    {0, 0, 1, 1, 0, 1, 1, 1, 1, 1},
	    10,
	    {
	      {0, 32, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {32, 64, &MPIR_Bcast_binomial_MV2, -1},
	      {64, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {128, 256, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {256, 2048, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {2048, 4096, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {4096, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {65536, 262144, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {262144, 524288, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {524288, -1, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1}
	    },
	    10,
	    {
	      {0, 32, NULL, -1},
	      {32, 64, NULL, -1},
	      {64, 128, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {128, 256, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {256, 2048, NULL, -1},
	      {2048, 4096, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {4096, 65536, &MPIR_Knomial_Bcast_intra_node_MV2, 4},
	      {65536, 262144, &MPIR_Shmem_Bcast_MV2, -1},
	      {262144, 524288, &MPIR_Shmem_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    256,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1},
	    15,
	    {
	      {0, 1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {1, 2, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	      {2, 4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {4, 8, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {8, 32, &MPIR_Pipelined_Bcast_MV2, -1},
	      {32, 64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {64, 128, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {128, 256, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {256, 512, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	      {512, 2048, &MPIR_Pipelined_Bcast_MV2, -1},
	      {2048, 4096, &MPIR_Bcast_binomial_MV2, -1},
	      {4096, 65536, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {65536, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
	      {131072, 262144, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {262144, -1, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1}
	    },
	    15,
	    {
	      {0, 1, &MPIR_Knomial_Bcast_intra_node_MV2, 8},
	      {1, 2, &MPIR_Shmem_Bcast_MV2, -1},
	      {2, 4, &MPIR_Knomial_Bcast_intra_node_MV2, 8},
	      {4, 8, &MPIR_Shmem_Bcast_MV2, -1},
	      {8, 32, &MPIR_Shmem_Bcast_MV2, -1},
	      {32, 64, &MPIR_Knomial_Bcast_intra_node_MV2, 8},
	      {64, 128, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {128, 256, &MPIR_Shmem_Bcast_MV2, -1},
	      {256, 512, &MPIR_Shmem_Bcast_MV2, -1},
	      {512, 2048, &MPIR_Shmem_Bcast_MV2, -1},
	      {2048, 4096, NULL, -1},
	      {4096, 65536, &MPIR_Shmem_Bcast_MV2, -1},
	      {65536, 131072, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {131072, 262144, &MPIR_Knomial_Bcast_intra_node_MV2, -1},
	      {262144, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    512,
	    8192, 4, 4,
	    {1, 0, 0, 0, 1, 0, 1, 1, 0, 1},
	    10,
	    {
	      {0, 2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {2, 4, &MPIR_Bcast_binomial_MV2, -1},
	      {4, 8, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {8, 32, &MPIR_Bcast_binomial_MV2, -1},
	      {32, 64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {64, 128, &MPIR_Bcast_binomial_MV2, -1},
	      {128, 1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {1024, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	      {262144, 524288, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	      {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	    },
	    10,
	    {
	      {0, 2, &MPIR_Knomial_Bcast_intra_node_MV2, 4},
	      {2, 4, NULL, -1},
	      {4, 8, NULL, -1},
	      {8, 32, NULL, -1},
	      {32, 64, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {64, 128, NULL, -1},
	      {128, 1024, &MPIR_Knomial_Bcast_intra_node_MV2, 2},
	      {1024, 262144, &MPIR_Shmem_Bcast_MV2, -1},
	      {262144, 524288, NULL, -1},
	      {524288, -1, &MPIR_Knomial_Bcast_intra_node_MV2, -1}
	    }
	  },
	  {
	    1024,
	    8192, 4, 4,
	    {1, 1, 0},
	    3, {{0, 32768, &MPIR_Bcast_binomial_MV2, -1},
		{32768, 131072, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
		{131072, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	    },
	    1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  }
        };
        MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
                    mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	return 0;
      }else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				     MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){

        /*Gordon Table*/
        mv2_size_bcast_tuning_table=6;
        mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
                                                 sizeof (mv2_bcast_tuning_table));

	mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[]={
	  {
	    16,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	    14,
	    {
	      {0, 1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {1, 2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {2, 4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {4, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {16, 64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {64, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {128, 1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {1024, 2048, &MPIR_Bcast_binomial_MV2, -1},
	      {2048, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {8192, 16384, &MPIR_Bcast_binomial_MV2, -1},
	      {16384, 32768, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	      {32768, 65536, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
	      {65536, 524288, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	      {524288, -1, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1}
	    },
	    14,
	    {
	      {0, 1, &MPIR_Shmem_Bcast_MV2, 8},
	      {1, 2, &MPIR_Shmem_Bcast_MV2, 2},
	      {2, 4, &MPIR_Shmem_Bcast_MV2, 8},
	      {4, 16, &MPIR_Shmem_Bcast_MV2, 4},
	      {16, 64, &MPIR_Shmem_Bcast_MV2, 2},
	      {64, 128, &MPIR_Shmem_Bcast_MV2, 8},
	      {128, 1024, &MPIR_Shmem_Bcast_MV2, 2},
	      {1024, 2048, &MPIR_Shmem_Bcast_MV2, -1},
	      {2048, 8192, &MPIR_Shmem_Bcast_MV2, 2},
	      {8192, 16384, &MPIR_Shmem_Bcast_MV2, -1},
	      {16384, 32768, &MPIR_Shmem_Bcast_MV2, -1},
	      {32768, 65536, &MPIR_Shmem_Bcast_MV2, -1},
	      {65536, 524288, &MPIR_Shmem_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    32,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	    10,
	    {
	      {0, 4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {4, 8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {8, 32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {32, 64, &MPIR_Bcast_binomial_MV2, -1},
	      {64, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {128, 256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {256, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {2048, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {4096, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {65536, -1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4}
	    },
	    10,
	    {
	      {0, 4, &MPIR_Shmem_Bcast_MV2, 8},
	      {4, 8, &MPIR_Shmem_Bcast_MV2, 2},
	      {8, 32, &MPIR_Shmem_Bcast_MV2, 8},
	      {32, 64, &MPIR_Shmem_Bcast_MV2, -1},
	      {64, 128, &MPIR_Shmem_Bcast_MV2, 2},
	      {128, 256, &MPIR_Shmem_Bcast_MV2, 4},
	      {256, 2048, &MPIR_Shmem_Bcast_MV2, 8},
	      {2048, 4096, &MPIR_Shmem_Bcast_MV2, 2},
	      {4096, 65536, &MPIR_Shmem_Bcast_MV2, 8},
	      {65536, -1, &MPIR_Shmem_Bcast_MV2, 4}
	    }
	  },
	  {
	    64,
	    8192, 4, 4,
	    {1, 1, 1, 1},
	    4,
	    {
	      {0, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {8192, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {131072, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	    },
	    4,
	    {
	      {0, 8192, &MPIR_Shmem_Bcast_MV2, 8},
	      {8192, 131072, &MPIR_Shmem_Bcast_MV2, 2},
	      {131072, 524288, &MPIR_Shmem_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    128,
	    8192, 4, 4,
	    {1, 1, 1, 0},
	    4,
	    {
	      {0, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {8192, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {32768, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
	      {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	    },
	    4,
	    {
	      {0, 8192, &MPIR_Shmem_Bcast_MV2, 8},
	      {8192, 32768, &MPIR_Shmem_Bcast_MV2, 2},
	      {32768, 524288, &MPIR_Shmem_Bcast_MV2, -1},
	      {524288, -1, NULL, -1}
	    }
	  },
	  {
	    256,
	    8192, 4, 4,
	    {1, 1, 1, 1},
	    4,
	    {
	      {0, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {16384, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {32768, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
	      {131072, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	    },
	    4,
	    {
	      {0, 16384, &MPIR_Shmem_Bcast_MV2, 4},
	      {16384, 32768, &MPIR_Shmem_Bcast_MV2, 2},
	      {32768, 131072, &MPIR_Shmem_Bcast_MV2, -1},
	      {131072, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    512,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1},
	    5,
	    {
	      {0, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {2048, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {8192, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {32768, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
	      {131072, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	    },
	    5,
	    {
	      {0, 2048, &MPIR_Shmem_Bcast_MV2, 8},
	      {2048, 8192, &MPIR_Shmem_Bcast_MV2, 4},
	      {8192, 32768, &MPIR_Shmem_Bcast_MV2, 2},
	      {32768, 131072, &MPIR_Shmem_Bcast_MV2, -1},
	      {131072, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  },
	  {
	    1024,
	    8192, 4, 4,
	    {1, 1, 1, 1, 1, 1, 1},
	    7,
	    {
	      {0, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
	      {2048, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
	      {8192, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {16384, 32768, &MPIR_Pipelined_Bcast_MV2, -1},
	      {32768, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
	      {65536, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
	      {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	    },
	    7,
	    {
	      {0, 2048, &MPIR_Shmem_Bcast_MV2, 8},
	      {2048, 8192, &MPIR_Shmem_Bcast_MV2, 4},
	      {8192, 16384, &MPIR_Shmem_Bcast_MV2, 2},
	      {16384, 32768, &MPIR_Shmem_Bcast_MV2, -1},
	      {32768, 65536, &MPIR_Shmem_Bcast_MV2, 2},
	      {65536, 262144, &MPIR_Shmem_Bcast_MV2, -1},
	      {262144, -1, &MPIR_Shmem_Bcast_MV2, -1}
	    }
	  }
	};

        MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
                    mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	return 0;
      }
      else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
        /*Stampede,*/
        mv2_size_bcast_tuning_table=8;
        mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
                                                 sizeof (mv2_bcast_tuning_table));

	mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[]={
	  {
            16,
            8192, 4, 4,
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            11,
            {
              {0, 8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {8, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16, 1024, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {1024, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {8192, 16384, &MPIR_Bcast_binomial_MV2, -1},
              {16384, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {32768, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {65536, 131072, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
              {131072, 262144, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
              {262144, 524288, &MPIR_Bcast_scatter_doubling_allgather_MV2, -1},
              {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
            },
            11,
            {
              {0, 8, &MPIR_Shmem_Bcast_MV2, 2},
              {8, 16, &MPIR_Shmem_Bcast_MV2, 4},
              {16, 1024, &MPIR_Shmem_Bcast_MV2, 2},
              {1024, 8192, &MPIR_Shmem_Bcast_MV2, 4},
              {8192, 16384, &MPIR_Shmem_Bcast_MV2, -1},
              {16384, 32768, &MPIR_Shmem_Bcast_MV2, 4},
              {32768, 65536, &MPIR_Shmem_Bcast_MV2, 2},
              {65536, 131072, &MPIR_Shmem_Bcast_MV2, -1},
              {131072, 262144, &MPIR_Shmem_Bcast_MV2, -1},
              {262144, 524288, &MPIR_Shmem_Bcast_MV2, -1},
              {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
            }
	  },
	  {
            32,
            8192, 4, 4,
            {1, 1, 1, 1, 1, 1, 1, 1},
            8,
            {
              {0, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {128, 256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {256, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {32768, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {65536, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {131072, 262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {262144, 524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {524288, -1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8}
            },
            8,
            {
              {0, 128, &MPIR_Shmem_Bcast_MV2, 2},
              {128, 256, &MPIR_Shmem_Bcast_MV2, 4},
              {256, 32768, &MPIR_Shmem_Bcast_MV2, 2},
              {32768, 65536, &MPIR_Shmem_Bcast_MV2, 4},
              {65536, 131072, &MPIR_Shmem_Bcast_MV2, 2},
              {131072, 262144, &MPIR_Shmem_Bcast_MV2, 8},
              {262144, 524288, &MPIR_Shmem_Bcast_MV2, 2},
              {524288, -1, &MPIR_Shmem_Bcast_MV2, 8}
            }
	  },
	  {
            64,
            8192, 4, 4,
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            9,
            {
              {0, 2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {2, 4, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {4, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16, 32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {32, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {128, 256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {256, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {4096, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {32768, -1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}
            },
            9,
            {
              {0, 2, &MPIR_Shmem_Bcast_MV2, 4},
              {2, 4, &MPIR_Shmem_Bcast_MV2, 8},
              {4, 16, &MPIR_Shmem_Bcast_MV2, 4},
              {16, 32, &MPIR_Shmem_Bcast_MV2, 8},
              {32, 128, &MPIR_Shmem_Bcast_MV2, 4},
              {128, 256, &MPIR_Shmem_Bcast_MV2, 8},
              {256, 4096, &MPIR_Shmem_Bcast_MV2, 4},
              {4096, 32768, &MPIR_Shmem_Bcast_MV2, 8},
              {32768, -1, &MPIR_Shmem_Bcast_MV2, 2}
            }
	  },
	  {
            128,
            8192, 4, 4,
            {1, 1, 1, 0},
            4,
            {
              {0, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {8192, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16384, 524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
            },
            4,
            {
              {0, 8192, &MPIR_Shmem_Bcast_MV2, 8},
              {8192, 16384, &MPIR_Shmem_Bcast_MV2, 4},
              {16384, 524288, &MPIR_Shmem_Bcast_MV2, 2},
              {524288, -1, NULL, -1}
            }
	  },
	  {
            256,
            8192, 4, 4,
            {1, 1, 1, 1, 1},
            5,
            {
              {0, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16384, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {131072, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
              {262144, 524288, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
            },
            5,
            {
              {0, 16384, &MPIR_Shmem_Bcast_MV2, 4},
              {16384, 131072, &MPIR_Shmem_Bcast_MV2, 2},
              {131072, 262144, &MPIR_Shmem_Bcast_MV2, -1},
              {262144, 524288, &MPIR_Shmem_Bcast_MV2, 2},
              {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
            }
	  },
	  {
            512,
            8192, 4, 4,
            {1, 1, 1, 1, 1},
            5,
            {
              {0, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {4096, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16384, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {131072, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
              {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
            },
            5,
            {
              {0, 4096, &MPIR_Shmem_Bcast_MV2, 8},
              {4096, 16384, &MPIR_Shmem_Bcast_MV2, 4},
              {16384, 131072, &MPIR_Shmem_Bcast_MV2, 2},
              {131072, 262144, &MPIR_Shmem_Bcast_MV2, -1},
              {262144, -1, &MPIR_Shmem_Bcast_MV2, -1}
            }
	  },
	  {
            1024,
            8192, 4, 4,
            {1, 1, 1, 1, 1},
            5,
            {
              {0, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {8192, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16384, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {65536, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
              {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
            },
            5,
            {
              {0, 8192, &MPIR_Shmem_Bcast_MV2, 8},
              {8192, 16384, &MPIR_Shmem_Bcast_MV2, 4},
              {16384, 65536, &MPIR_Shmem_Bcast_MV2, 2},
              {65536, 524288, &MPIR_Shmem_Bcast_MV2, -1},
              {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
            }
	  },
	  {
            2048,
            8192, 4, 4,
            {1, 1, 1, 1, 1, 1, 1},
            7,
            {
              {0, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {16, 32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {32, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
              {4096, 16384, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
              {16384, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
              {32768, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
              {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
            },
            7,
            {
              {0, 16, &MPIR_Shmem_Bcast_MV2, 8},
              {16, 32, &MPIR_Shmem_Bcast_MV2, 4},
              {32, 4096, &MPIR_Shmem_Bcast_MV2, 8},
              {4096, 16384, &MPIR_Shmem_Bcast_MV2, 4},
              {16384, 32768, &MPIR_Shmem_Bcast_MV2, 2},
              {32768, 524288, &MPIR_Shmem_Bcast_MV2, -1},
              {524288, -1, &MPIR_Shmem_Bcast_MV2, -1}
            }
	  }
	};

        MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
                    mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	return 0;
      } else 

	{
	  /*RI*/
	  mv2_size_bcast_tuning_table = 7;
	  mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
						   sizeof (mv2_bcast_tuning_table));
	  mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[] = {
            {8,
             8192, 4, 4,
             {1},
             1, {{0, -1, &MPIR_Bcast_binomial_MV2, -1}},
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {
	      16,
	      8192, 4, 4,
	      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	      14,
	      {
		{0, 1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{1, 2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{2, 8, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{8, 16, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{16, 32, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{32, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{128, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{2048, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{4096, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{8192, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{32768, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{65536, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{131072, 262144, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{262144, -1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8}
	      },
	      14,
	      {
		{0, 1, &MPIR_Shmem_Bcast_MV2, 8},
		{1, 2, &MPIR_Shmem_Bcast_MV2, 2},
		{2, 8, &MPIR_Shmem_Bcast_MV2, 8},
		{8, 16, &MPIR_Shmem_Bcast_MV2, 2},
		{16, 32, &MPIR_Shmem_Bcast_MV2, 8},
		{32, 128, &MPIR_Shmem_Bcast_MV2, 2},
		{128, 2048, &MPIR_Shmem_Bcast_MV2, 8},
		{2048, 4096, &MPIR_Shmem_Bcast_MV2, 2},
		{4096, 8192, &MPIR_Shmem_Bcast_MV2, 8},
		{8192, 32768, &MPIR_Shmem_Bcast_MV2, 4},
		{32768, 65536, &MPIR_Shmem_Bcast_MV2, 2},
		{65536, 131072, &MPIR_Shmem_Bcast_MV2, 4},
		{131072, 262144, &MPIR_Shmem_Bcast_MV2, 2},
		{262144, -1, &MPIR_Shmem_Bcast_MV2, 8}
	      }
            },
            {
	      32,
	      8192, 4, 4,
	      {1, 1, 1, 1},
	      4,
	      {
		{0, 64, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{64, 256, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{256, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{8192, -1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2}
	      },
	      4,
	      {
		{0, 64, &MPIR_Shmem_Bcast_MV2, 8},
		{64, 256, &MPIR_Shmem_Bcast_MV2, 4},
		{256, 8192, &MPIR_Shmem_Bcast_MV2, 8},
		{8192, -1, &MPIR_Shmem_Bcast_MV2, 2}
	      }
            },
            {
	      64,
	      8192, 4, 4,
	      {1, 1, 1, 0},
	      4,
	      {
		{0, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{2048, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{8192, 131072, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{131072, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	      },
	      4,
	      {
		{0, 2048, &MPIR_Shmem_Bcast_MV2, 8},
		{2048, 8192, &MPIR_Shmem_Bcast_MV2, 4},
		{8192, 131072, &MPIR_Shmem_Bcast_MV2, 2},
		{131072, -1, NULL, -1}
	      }
            },
            {
	      128,
	      8192, 4, 4,
	      {1, 1, 1, 1, 1, 1, 1, 1, 0},
	      9,
	      {
		{0, 1, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{1, 2, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{2, 128, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{128, 512, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{512, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{8192, 65536, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{65536, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
		{131072, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
		{262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	      },
	      9,
	      {
		{0, 1, &MPIR_Shmem_Bcast_MV2, 4},
		{1, 2, &MPIR_Shmem_Bcast_MV2, 8},
		{2, 128, &MPIR_Shmem_Bcast_MV2, 4},
		{128, 512, &MPIR_Shmem_Bcast_MV2, 8},
		{512, 8192, &MPIR_Shmem_Bcast_MV2, 4},
		{8192, 65536, &MPIR_Shmem_Bcast_MV2, 2},
		{65536, 131072, &MPIR_Shmem_Bcast_MV2, -1},
		{131072, 262144, &MPIR_Shmem_Bcast_MV2, -1},
		{262144, -1, NULL, -1}
	      }
            },
            {
	      256,
	      8192, 4, 4,
	      {1, 1, 1, 1, 1, 0},
	      6,
	      {
		{0, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{4096, 8192, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{8192, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{32768, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
		{131072, 524288, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
		{524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	      },
	      6,
	      {
		{0, 4096, &MPIR_Shmem_Bcast_MV2, 8},
		{4096, 8192, &MPIR_Shmem_Bcast_MV2, 4},
		{8192, 32768, &MPIR_Shmem_Bcast_MV2, 2},
		{32768, 131072, &MPIR_Shmem_Bcast_MV2, -1},
		{131072, 524288, &MPIR_Shmem_Bcast_MV2, -1},
		{524288, -1, NULL, -1}
	      }
            },
            {
	      512,
	      8192, 4, 4,
	      {1, 1, 1, 1, 1},
	      5,
	      {
		{0, 2048, &MPIR_Pipelined_Bcast_Zcpy_MV2, 8},
		{2048, 4096, &MPIR_Pipelined_Bcast_Zcpy_MV2, 4},
		{4096, 32768, &MPIR_Pipelined_Bcast_Zcpy_MV2, 2},
		{32768, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
		{262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	      },
	      5,
	      {
		{0, 2048, &MPIR_Shmem_Bcast_MV2, 8},
		{2048, 4096, &MPIR_Shmem_Bcast_MV2, 4},
		{4096, 32768, &MPIR_Shmem_Bcast_MV2, 2},
		{32768, 262144, &MPIR_Shmem_Bcast_MV2, -1},
		{262144, -1, &MPIR_Shmem_Bcast_MV2, -1}
	      }
            }
	  };
    
	  MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
		      mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	  return 0;
	}
#elif defined(CHANNEL_NEMESIS_IB)
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
	  mv2_size_bcast_tuning_table=6;
	  mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
						   sizeof (mv2_bcast_tuning_table));

	  mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[]={
            {32,
             8192, 4, 4,
             {1, 0},
             2, {{0, 16384, &MPIR_Bcast_binomial_MV2, -1}, 
                 {16384, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}},
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {64,
             8192, 4, 4,
             {1, 0},
             2, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1}, 
                 {8192, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}},
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {128,
             8192, 4, 4,
             {1, 1, 0},
             3, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 65536, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {65536, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {256,
             8192, 4, 4,
             {1, 1, 0},
             3, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 65536, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1},
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {512,
             8192, 4, 4,
             {1, 1, 0},
             3, {{0, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {1024,
             8192, 4, 4,
             {1, 1, 0},
             3, {{0, 32768, &MPIR_Bcast_binomial_MV2, -1},
                 {32768, 131072, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {131072, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            }
	  };
	  MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
		      mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	  return 0;
	} else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
					MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
	  /*Stampede,*/
	  mv2_size_bcast_tuning_table=8;
	  mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
						   sizeof (mv2_bcast_tuning_table));

	  mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[]={
            {16,
	     8192, 4, 4,
             {1, 1},
             2, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, -1, &MPIR_Pipelined_Bcast_MV2, -1},
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {32,
	     8192, 4, 4,
             {1, 1, 1},
             3, {{0, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
                 {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {64,
	     8192, 4, 4,
             {1, 1, 1, 0},
             4, {{0, 4096, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {4096, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {128,
	     8192, 4, 4,
	     {1, 1, 1, 0},
             4, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {2048, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {16384, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
                 {52488, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {256,
	     8192, 4, 4,
             {1, 1, 1, 0},
             4, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {2048, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
                 {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {512,
	     8192, 4, 4,
             {1, 1, 1, 1},
             4, {{0,  4096, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {4096, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {16384, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
		 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {1024,
             8192, 4, 4,
             {1, 1, 1, 1},
             4, {{0,  4096, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {4096, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {16384, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {2048,
             8192, 4, 4,
             {1, 1, 1},
             3, {{0,  8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {8192, 16384, &MPIR_Bcast_binomial_MV2, -1},
                 {16384, -1, &MPIR_Pipelined_Bcast_MV2, -1}
	     },
             3, {{0, 8192, &MPIR_Shmem_Bcast_MV2, -1},
                 {8192, 16384, &MPIR_Shmem_Bcast_MV2, -1},
                 {16384, -1, &MPIR_Shmem_Bcast_MV2, -1}
	     }
            }

	  };

	  MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
		      mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	  return 0;
	} else {
	  mv2_size_bcast_tuning_table = 7;
	  mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
						   sizeof (mv2_bcast_tuning_table));
	  mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[] = {
            {8,
             8192, 4, 4,
             {1},
             1, {{0, -1, &MPIR_Bcast_binomial_MV2, -1}},
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {16,
             8192, 4, 4,
             {1, 1},
             2, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, -1, &MPIR_Pipelined_Bcast_MV2, -1},
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {32,
             8192, 4, 4,
             {1, 1, 1},
             3, {{0, 8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {8192, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
                 {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {64,
             8192, 4, 4,
             {1, 1, 1, 0},
             4, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {2048, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {128,
             8192, 4, 4,
             {1, 1, 1, 0},
             4, {{0, 8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {8192, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
                 {131072, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {256,
             8192, 4, 4,
             {1, 1, 1, 1, 0},
             5, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {2048, 8192, &MPIR_Bcast_binomial_MV2, -1},
                 {8192, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
                 {131072, 524288, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
                 {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            },
            {512,
             8192, 4, 4,
             {1, 1, 1},
             3, {{0,  8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
                 {8192, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
                 {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	     },
             1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
            }
	  };
    
	  MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
		      mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	  return 0;
	}
#else
      {
        mv2_size_bcast_tuning_table = 7;
        mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
						 sizeof (mv2_bcast_tuning_table));
        mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[] = {
	  {8,
	   8192, 4, 4,
	   {1},
	   1, {{0, -1, &MPIR_Bcast_binomial_MV2, -1}},
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {16,
	   8192, 4, 4,
	   {1, 1},
	   2, {{0, 8192, &MPIR_Bcast_binomial_MV2, -1},
	       {8192, -1, &MPIR_Pipelined_Bcast_MV2, -1},
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {32,
	   8192, 4, 4,
	   {1, 1, 1},
	   3, {{0, 8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	       {8192, 524288, &MPIR_Pipelined_Bcast_MV2, -1},
	       {524288, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {64,
	   8192, 4, 4,
	   {1, 1, 1, 0},
	   4, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	       {2048, 8192, &MPIR_Bcast_binomial_MV2, -1},
	       {8192, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
	       {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {128,
	   8192, 4, 4,
	   {1, 1, 1, 0},
	   4, {{0, 8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	       {8192, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
	       {131072, 262144, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	       {262144, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {256,
	   8192, 4, 4,
	   {1, 1, 1, 1, 0},
	   5, {{0, 2048, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	       {2048, 8192, &MPIR_Bcast_binomial_MV2, -1},
	       {8192, 131072, &MPIR_Pipelined_Bcast_MV2, -1},
	       {131072, 524288, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1},
	       {524288, -1, &MPIR_Bcast_scatter_ring_allgather_MV2, -1}
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  },
	  {512,
	   8192, 4, 4,
	   {1, 1, 1},
	   3, {{0,  8192, &MPIR_Knomial_Bcast_inter_node_wrapper_MV2, -1},
	       {8192, 262144, &MPIR_Pipelined_Bcast_MV2, -1},
	       {262144, -1, &MPIR_Bcast_scatter_ring_allgather_shm_MV2, -1}
	   },
	   1, {{0, -1, &MPIR_Shmem_Bcast_MV2, -1}}
	  }
	};
    
	MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table,
		    mv2_size_bcast_tuning_table * sizeof (mv2_bcast_tuning_table));
	return 0;
      }
#endif
    }
    return 0;
}

void MV2_cleanup_bcast_tuning_table()
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_bcast_tuning) {
    MPIU_Free(mv2_bcast_indexed_thresholds_table[0]);
    MPIU_Free(mv2_bcast_indexed_table_ppn_conf);
    MPIU_Free(mv2_size_bcast_indexed_tuning_table);
    if (mv2_bcast_indexed_thresholds_table != NULL) {
      MPIU_Free(mv2_bcast_indexed_thresholds_table);
    }
  }
  else {
    if (mv2_bcast_thresholds_table != NULL) {
      MPIU_Free(mv2_bcast_thresholds_table);
    }
  }
}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}

int MV2_internode_Bcast_is_define(char *mv2_user_bcast_inter, char *mv2_user_bcast_intra)
{

  if (mv2_use_indexed_tuning || mv2_use_indexed_bcast_tuning) {
    int i=0;
    int nb_element = count_sep(mv2_user_bcast_inter) + 1;

    /* If one bcast tuning table is already defined */
    if (mv2_bcast_indexed_thresholds_table != NULL) {
        if (mv2_bcast_indexed_thresholds_table[0] != NULL) {
            MPIU_Free(mv2_bcast_indexed_thresholds_table[0]);
        }
	    MPIU_Free(mv2_bcast_indexed_thresholds_table);
    }

    mv2_bcast_indexed_tuning_table mv2_tmp_bcast_indexed_thresholds_table[1];
    mv2_bcast_indexed_num_ppn_conf = 1;
    if (mv2_size_bcast_indexed_tuning_table == NULL) {
        mv2_size_bcast_indexed_tuning_table =
	  MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
    }
    else {
	MPIU_Free(mv2_size_bcast_indexed_tuning_table);
        mv2_size_bcast_indexed_tuning_table =
	  MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
    }
    mv2_size_bcast_indexed_tuning_table[0] = 1;

    if (mv2_bcast_indexed_table_ppn_conf == NULL) {
        mv2_bcast_indexed_table_ppn_conf =
	  MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf * sizeof(int));
    }
    /* -1 indicates user defined algorithm */
    mv2_bcast_indexed_table_ppn_conf[0] = -1;

    /* We realloc the space for the new bcast tuning table */
    mv2_bcast_indexed_thresholds_table =
      MPIU_Malloc(mv2_bcast_indexed_num_ppn_conf *
		  sizeof(mv2_bcast_indexed_tuning_table *));
    mv2_bcast_indexed_thresholds_table[0] =
      MPIU_Malloc(mv2_size_bcast_indexed_tuning_table[0] *
		  sizeof(mv2_bcast_indexed_tuning_table));

    if (nb_element == 1) {
      mv2_tmp_bcast_indexed_thresholds_table[0].numproc = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].bcast_segment_size = bcast_segment_size;
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_node_knomial_factor = mv2_inter_node_knomial_factor;
      mv2_tmp_bcast_indexed_thresholds_table[0].intra_node_knomial_factor = mv2_intra_node_knomial_factor;
      mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].size_inter_table = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].msg_sz = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].zcpy_pipelined_knomial_factor = -1;
      mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].msg_sz = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].zcpy_pipelined_knomial_factor = -1;
      switch (atoi(mv2_user_bcast_inter)) {
      case 1:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 2:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 3:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_doubling_allgather_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 4:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_doubling_allgather_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 5:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 6:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 7:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_shm_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 8:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Knomial_Bcast_inter_node_wrapper_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 9:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Pipelined_Bcast_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
#ifdef CHANNEL_MRAIL_GEN2
      case 10:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Pipelined_Bcast_Zcpy_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
	if (mv2_pipelined_zcpy_knomial_factor == -1) {
	    mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].zcpy_pipelined_knomial_factor =
		zcpy_knomial_factor;
	}
	else {
	    mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].zcpy_pipelined_knomial_factor =
		mv2_pipelined_zcpy_knomial_factor;
	}
	break;
#endif

    case 12:
    mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
      &MPIR_Concurrent_Bcast_MV2;
      mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
      break;
      
/********************* Added by Cong *********************/
        case 13:
    mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
      &MPIR_Bcast_ML_Shmem_MV2;
      mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 1;
      break;


      default:
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[0] = 0;
      }
      if (mv2_user_bcast_intra == NULL) {
	mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Shmem_Bcast_MV2;
      } else {
	if (atoi(mv2_user_bcast_intra) == 1) {
	  mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	    &MPIR_Knomial_Bcast_intra_node_MV2;
	} else {
	  mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	    &MPIR_Shmem_Bcast_MV2;
	}
      }
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_bcast_inter))) {
	fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_bcast_inter);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_bcast_inter);
	MPIU_Free(dup);
	return 2;
      }

      mv2_tmp_bcast_indexed_thresholds_table[0].numproc = 1;
      mv2_tmp_bcast_indexed_thresholds_table[0].bcast_segment_size = bcast_segment_size;
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_node_knomial_factor = mv2_inter_node_knomial_factor;
      mv2_tmp_bcast_indexed_thresholds_table[0].intra_node_knomial_factor = mv2_intra_node_knomial_factor;
      mv2_tmp_bcast_indexed_thresholds_table[0].size_inter_table = nb_element;
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
	case 1:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 2:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 3:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_doubling_allgather_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 4:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_doubling_allgather_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 5:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 6:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 7:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_shm_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 8:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Knomial_Bcast_inter_node_wrapper_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 9:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Pipelined_Bcast_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
#ifdef CHANNEL_MRAIL_GEN2
	case 10:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Pipelined_Bcast_Zcpy_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
#endif
    case 12:
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
      &MPIR_Concurrent_Bcast_MV2;
      mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
      break;
/********************* Added by Cong *********************/
          case 13:
      mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
      &MPIR_Bcast_ML_Shmem_MV2;
      mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 1;
      break;

	default:
	  mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_indexed_thresholds_table[0].is_two_level_bcast[i] = 0;
	}
	mv2_tmp_bcast_indexed_thresholds_table[0].inter_leader[i].msg_sz = 1;

	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
    mv2_tmp_bcast_indexed_thresholds_table[0].size_intra_table = 1;
    if (mv2_user_bcast_intra == NULL) {
      mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	&MPIR_Shmem_Bcast_MV2;
    } else {
      if (atoi(mv2_user_bcast_intra) == 1) {
	mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Knomial_Bcast_intra_node_MV2;
      } else {
	mv2_tmp_bcast_indexed_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Shmem_Bcast_MV2;
      }
    }
    MPIU_Memcpy(mv2_bcast_indexed_thresholds_table[0], mv2_tmp_bcast_indexed_thresholds_table, sizeof
                (mv2_bcast_indexed_tuning_table));
  }
  else {
    int i;
    int nb_element = count_sep(mv2_user_bcast_inter) + 1;

    /* If one bcast tuning table is already defined */
    if (mv2_bcast_thresholds_table != NULL) {
      MPIU_Free(mv2_bcast_thresholds_table);
    }

    mv2_bcast_tuning_table mv2_tmp_bcast_thresholds_table[1];
    mv2_size_bcast_tuning_table = 1;

    /* We realloc the space for the new bcast tuning table */
    mv2_bcast_thresholds_table = MPIU_Malloc(mv2_size_bcast_tuning_table *
                                             sizeof (mv2_bcast_tuning_table));

    if (nb_element == 1) {
      mv2_tmp_bcast_thresholds_table[0].numproc = 1;
      mv2_tmp_bcast_thresholds_table[0].bcast_segment_size = bcast_segment_size;
      mv2_tmp_bcast_thresholds_table[0].inter_node_knomial_factor = mv2_inter_node_knomial_factor;
      mv2_tmp_bcast_thresholds_table[0].intra_node_knomial_factor = mv2_intra_node_knomial_factor;
      mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
      mv2_tmp_bcast_thresholds_table[0].size_inter_table = 1;
      mv2_tmp_bcast_thresholds_table[0].inter_leader[0].min = 0;
      mv2_tmp_bcast_thresholds_table[0].inter_leader[0].max = -1;
      switch (atoi(mv2_user_bcast_inter)) {
      case 1:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 2:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 3:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_doubling_allgather_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 4:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_doubling_allgather_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 5:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 0;
	break;
      case 6:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 7:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_scatter_ring_allgather_shm_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 8:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Knomial_Bcast_inter_node_wrapper_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
      case 9:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Pipelined_Bcast_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
#ifdef CHANNEL_MRAIL_GEN2
      case 10:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Pipelined_Bcast_Zcpy_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
	break;
#endif

  case 12:
      mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
      &MPIR_Concurrent_Bcast_MV2;
      mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
      break;

/********************* Added by Cong *********************/
        case 13:
      mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
      &MPIR_Bcast_ML_Shmem_MV2;
      mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 1;
      break;

      default:
	mv2_tmp_bcast_thresholds_table[0].inter_leader[0].MV2_pt_Bcast_function =
	  &MPIR_Bcast_binomial_MV2;
	mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[0] = 0;
      }
      if (mv2_user_bcast_intra == NULL) {
	mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Shmem_Bcast_MV2;
      } else {
	if (atoi(mv2_user_bcast_intra) == 1) {
	  mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	    &MPIR_Knomial_Bcast_intra_node_MV2;
	} else {
	  mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	    &MPIR_Shmem_Bcast_MV2;
	}

      }
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_bcast_inter))) {
	fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_bcast_inter);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_bcast_inter);
	MPIU_Free(dup);
	return 2;
      }

      mv2_tmp_bcast_thresholds_table[0].numproc = 1;
      mv2_tmp_bcast_thresholds_table[0].bcast_segment_size = bcast_segment_size;
      mv2_tmp_bcast_thresholds_table[0].inter_node_knomial_factor = mv2_inter_node_knomial_factor;
      mv2_tmp_bcast_thresholds_table[0].intra_node_knomial_factor = mv2_intra_node_knomial_factor;
      mv2_tmp_bcast_thresholds_table[0].size_inter_table = nb_element;
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
	case 1:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 2:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 3:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_doubling_allgather_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 4:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_doubling_allgather_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 5:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 0;
	  break;
	case 6:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 7:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_scatter_ring_allgather_shm_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 8:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Knomial_Bcast_inter_node_wrapper_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
	case 9:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Pipelined_Bcast_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
#ifdef CHANNEL_MRAIL_GEN2
	case 10:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Pipelined_Bcast_Zcpy_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
	  break;
#endif

  case 12:
      mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
      &MPIR_Concurrent_Bcast_MV2;
      mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
      break;

/********************* Added by Cong *********************/
    case 13:
      mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
      &MPIR_Bcast_ML_Shmem_MV2;
      mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 1;
      break;

	default:
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].MV2_pt_Bcast_function =
	    &MPIR_Bcast_binomial_MV2;
	  mv2_tmp_bcast_thresholds_table[0].is_two_level_bcast[i] = 0;
	}
	mv2_tmp_bcast_thresholds_table[0].inter_leader[i].min = atoi(p +
								     match[2].rm_so);
	if (p[match[3].rm_so] == '+') {
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].max = -1;
	} else {
	  mv2_tmp_bcast_thresholds_table[0].inter_leader[i].max =
	    atoi(p + match[3].rm_so);
	}

	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
    mv2_tmp_bcast_thresholds_table[0].size_intra_table = 1;
    if (mv2_user_bcast_intra == NULL) {
      mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	&MPIR_Shmem_Bcast_MV2;
    } else {
      if (atoi(mv2_user_bcast_intra) == 1) {
	mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Knomial_Bcast_intra_node_MV2;
      } else {
	mv2_tmp_bcast_thresholds_table[0].intra_node[0].MV2_pt_Bcast_function =
	  &MPIR_Shmem_Bcast_MV2;
      }
    }
    MPIU_Memcpy(mv2_bcast_thresholds_table, mv2_tmp_bcast_thresholds_table, sizeof
                (mv2_bcast_tuning_table));
  }
  return 0;
}

int MV2_intranode_Bcast_is_define(char *mv2_user_bcast_intra)
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_bcast_tuning) {
    //nop
  }
  else {
    int i, j;
    for (i = 0; i < mv2_size_bcast_tuning_table; i++) {
      for (j = 0; j < mv2_bcast_thresholds_table[i].size_intra_table; j++) {
	if (atoi(mv2_user_bcast_intra) == 1) {
	  mv2_bcast_thresholds_table[i].intra_node[j].MV2_pt_Bcast_function =
	    &MPIR_Knomial_Bcast_intra_node_MV2;
	} else {
	  mv2_bcast_thresholds_table[i].intra_node[j].MV2_pt_Bcast_function =
	    &MPIR_Shmem_Bcast_MV2;
	}
      }
    }
  }
    return 0;
}
