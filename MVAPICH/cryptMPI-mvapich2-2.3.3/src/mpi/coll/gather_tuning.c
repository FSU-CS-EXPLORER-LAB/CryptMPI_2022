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
#include "gather_tuning.h"
#include "tuning/gather_arch_tuning.h"
#include "mv2_arch_hca_detect.h"
/* array used to tune gather */
int mv2_size_gather_tuning_table=7;
mv2_gather_tuning_table * mv2_gather_thresholds_table=NULL; 

#if defined(_SMP_LIMIC_)
extern int use_limic_gather;
#endif /*#if defined(_SMP_LIMIC_)*/

int *mv2_gather_indexed_table_ppn_conf = NULL;
int mv2_gather_indexed_num_ppn_conf = 1;
int *mv2_size_gather_indexed_tuning_table = NULL;
mv2_gather_indexed_tuning_table **mv2_gather_indexed_thresholds_table = NULL;

int MV2_set_gather_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
  
  if (mv2_use_indexed_tuning || mv2_use_indexed_gather_tuning) {
    int agg_table_sum = 0;
    int i;
    mv2_gather_indexed_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
	  /* Lonestar Table*/
	  mv2_gather_indexed_num_ppn_conf = 3;
	  mv2_gather_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			  * mv2_gather_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
				   * mv2_gather_indexed_num_ppn_conf);
	  mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							     mv2_gather_indexed_num_ppn_conf);
	  mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_gather_indexed_table_ppn_conf[0] = 1;
	  mv2_size_gather_indexed_tuning_table[0] = 4;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN
	  table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
	  mv2_gather_indexed_table_ppn_conf[1] = 2;
	  mv2_size_gather_indexed_tuning_table[1] = 3;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN
	  table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
	  mv2_gather_indexed_table_ppn_conf[2] = 12;
	  mv2_size_gather_indexed_tuning_table[2] = 6;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_12ppn[] =
	    GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN
	  table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_12ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
	  }
	  mv2_gather_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
	  MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_gather_indexed_tuning_table)
		       * mv2_size_gather_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    mv2_gather_indexed_thresholds_table[i] =
	      mv2_gather_indexed_thresholds_table[i - 1]
	      + mv2_size_gather_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_gather_indexed_tuning_table)
			 * mv2_size_gather_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
    if ((MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20, MV2_HCA_MLX_CX_CONNIB) ||
        MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20, MV2_HCA_MLX_CX_CONNIB)) && !heterogeneity) {
	  /* PSG Table*/
	  mv2_gather_indexed_num_ppn_conf = 3;
	  mv2_gather_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			  * mv2_gather_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
				   * mv2_gather_indexed_num_ppn_conf);
	  mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							     mv2_gather_indexed_num_ppn_conf);
	  mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_gather_indexed_table_ppn_conf[0] = 1;
	  mv2_size_gather_indexed_tuning_table[0] = 3;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__1PPN;
	  table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
	  mv2_gather_indexed_table_ppn_conf[1] = 2;
	  mv2_size_gather_indexed_tuning_table[1] = 4;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__2PPN;
	  table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
	  mv2_gather_indexed_table_ppn_conf[2] = 20;
	  mv2_size_gather_indexed_tuning_table[2] = 4;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_20ppn[] =
	    GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN;
	  table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_20ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
	  }
	  mv2_gather_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
	  MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_gather_indexed_tuning_table)
		       * mv2_size_gather_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    mv2_gather_indexed_thresholds_table[i] =
	      mv2_gather_indexed_thresholds_table[i - 1]
	      + mv2_size_gather_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_gather_indexed_tuning_table)
			 * mv2_size_gather_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
	if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12, MV2_HCA_MLX_CX_CONNIB) && !heterogeneity) {
	  /* Wilkes Table*/
	  mv2_gather_indexed_num_ppn_conf = 3;
	  mv2_gather_indexed_thresholds_table
	    = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			  * mv2_gather_indexed_num_ppn_conf);
	  table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
				   * mv2_gather_indexed_num_ppn_conf);
	  mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							     mv2_gather_indexed_num_ppn_conf);
	  mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
	  mv2_gather_indexed_table_ppn_conf[0] = 1;
	  mv2_size_gather_indexed_tuning_table[0] = 6;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__1PPN
	  table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
	  mv2_gather_indexed_table_ppn_conf[1] = 2;
	  mv2_size_gather_indexed_tuning_table[1] = 6;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__2PPN
	  table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
	  mv2_gather_indexed_table_ppn_conf[2] = 12;
	  mv2_size_gather_indexed_tuning_table[2] = 6;
	  mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_12ppn[] =
	    GEN2__INTEL_XEON_E5_2630_V2_2S_12__MLX_CX_CONNIB__12PPN
	  table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_12ppn;
      
	  agg_table_sum = 0;
	  for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
	  }
	  mv2_gather_indexed_thresholds_table[0] =
	    MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
	  MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		      (sizeof(mv2_gather_indexed_tuning_table)
		       * mv2_size_gather_indexed_tuning_table[0]));
	  for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	    mv2_gather_indexed_thresholds_table[i] =
	      mv2_gather_indexed_thresholds_table[i - 1]
	      + mv2_size_gather_indexed_tuning_table[i - 1];
	    MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
			(sizeof(mv2_gather_indexed_tuning_table)
			 * mv2_size_gather_indexed_tuning_table[i]));
	  }
	  MPIU_Free(table_ptrs);
	  return 0;
	}
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 4;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 3;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 32;
#if defined(_SMP_CMA_)
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_32ppn[] =
        GEN2_CMA__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_32ppn[] =
        GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_gather_indexed_thresholds_table_32ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_32ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[2] = 4;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_32ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_32ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_cma_gather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[0] = 6;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[1] = 7;
	table_ptrs[1] = mv2_tmp_cma_gather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[1] = 7;
	table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[1] = 7;
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_cma_gather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[2] = 7;
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[0] = 3;
	table_ptrs[0] = mv2_tmp_cma_gather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[0] = 2;
	table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[0] = 2;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[1] = 3;
	table_ptrs[1] = mv2_tmp_cma_gather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[1] = 2;
	table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[1] = 2;
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_gather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[2] = 5;
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_gather_indexed_num_ppn_conf = 4;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_cma_gather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_cma_gather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
#endif

       mv2_gather_indexed_table_ppn_conf[2] = 4;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_4ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_4ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_cma_gather_indexed_thresholds_table_4ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_4ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[2] = 1;
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_4ppn;
#endif     

      mv2_gather_indexed_table_ppn_conf[3] = 16;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[3] = 6;
	table_ptrs[3] = mv2_tmp_cma_gather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[3] = 6;
	table_ptrs[3] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[3] = 6;
      table_ptrs[3] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }

    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /*RI2 Table*/
force_default_tables:
      MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  4, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  5, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28, 5, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 28, 5, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7551_64, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* AMD EPYC table */
      MV2_COLL_TUNING_START_TABLE  (gather, 7)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  3, GEN2__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  3, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  4, GEN2__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  4, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  4, GEN2__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  4, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  4, GEN2__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  4, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 4, GEN2__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 4, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 4, GEN2__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 32, 4, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 64, 4, GEN2__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 64, 4, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7742_128, MV2_HCA_ANY) && !heterogeneity) {
      /* AMD EPYC rome table */
      MV2_COLL_TUNING_START_TABLE  (gather, 7)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  1, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  2, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  2, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  2, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 2, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 2, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 64, 2, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* Haswell HDR nodes on NOWLAB */
      MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Frontera */
      MV2_COLL_TUNING_START_TABLE  (gather, 8)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28, 4, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 28, 4, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 5, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 32, 5, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Mayer */
      MV2_COLL_TUNING_START_TABLE  (gather, 8)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_32, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Catalyst */
      MV2_COLL_TUNING_START_TABLE  (gather, 7)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /* ARM system at Hartree Center */
      MV2_COLL_TUNING_START_TABLE  (gather, 5)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  2, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 24,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__24PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER8, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Ray Table */
      int pg_size = MPIDI_PG_Get_size(MPIDI_Process.my_pg);
      if (pg_size > 64) goto force_default_tables;

      MV2_COLL_TUNING_START_TABLE  (gather, 3)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  4, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  2, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  2, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER9, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Sierra Table: Use table for Ray temporarily */
      int pg_size = MPIDI_PG_Get_size(MPIDI_Process.my_pg);
      if (pg_size > 64) goto force_default_tables;

      MV2_COLL_TUNING_START_TABLE  (gather, 9)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, GEN2__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  4, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, GEN2__IBM_POWER9_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, GEN2__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, GEN2__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 5, GEN2__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 22, 5, GEN2__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 22, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 5, GEN2__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 32, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 44, 5, GEN2__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 44, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				 MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*RI Table*/
      MV2_COLL_TUNING_START_TABLE  (gather, 4)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  6, GEN2__RI__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  6, GEN2_CMA__RI__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  7, GEN2__RI__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  7, GEN2_CMA__RI__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  1, GEN2__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  1, GEN2_CMA__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  8, GEN2__RI__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  8, GEN2_CMA__RI__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Comet Table*/
      mv2_gather_indexed_num_ppn_conf = 1;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 24;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_24ppn[] =
	  GEN2__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
      /*
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_24ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_cma_gather_indexed_thresholds_table_24ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_24ppn;
      }
#else
      */
      mv2_size_gather_indexed_tuning_table[0] = 6;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_24ppn;
      /*
#endif
      */
            
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }    

    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_ANY, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /*RI2 Table*/
     MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  4, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  5, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28, 5, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 28, 5, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather) 
    }  	   
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
		MV2_ARCH_ANY, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /*RI2 Table*/
     MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 1,  4, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 2,  5, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 16, 5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28, 5, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (gather, 28, 5, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather) 
    }  	   
    else {
      /*Stampede Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_cma_gather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_cma_gather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
#endif
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_gather_indexed_tuning_table mv2_tmp_cma_gather_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_gather_indexed_tuning_table[2] = 6;
	table_ptrs[2] = mv2_tmp_cma_gather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_gather_indexed_tuning_table[2] = 6;
	table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_gather_indexed_tuning_table[2] = 6;
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#elif defined (CHANNEL_NEMESIS_IB)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 4;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 3;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 32;
      mv2_size_gather_indexed_tuning_table[2] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_32ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_32ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_size_gather_indexed_tuning_table[2] = 4;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_16PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_size_gather_indexed_tuning_table[2] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_size_gather_indexed_tuning_table[2] = 7;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*RI Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
        = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
                      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
                               * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
                                                         mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));

      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
        NEMESIS__RI__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;

      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 2;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
        NEMESIS__RI__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;

      mv2_gather_indexed_table_ppn_conf[2] = 8;
      mv2_size_gather_indexed_tuning_table[2] = 8;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_8ppn[] =
        NEMESIS__RI__8PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_8ppn;

      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
        agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
        MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
                  (sizeof(mv2_gather_indexed_tuning_table)
                   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
        mv2_gather_indexed_thresholds_table[i] =
          mv2_gather_indexed_thresholds_table[i - 1]
          + mv2_size_gather_indexed_tuning_table[i - 1];
        MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
                    (sizeof(mv2_gather_indexed_tuning_table)
                     * mv2_size_gather_indexed_tuning_table[i]));
	}
      MPIU_Free(table_ptrs);
      return 0;
    }
    else {
      /*Stampede Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_size_gather_indexed_tuning_table[2] = 7;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#endif
#else /* !CHANNEL_PSM */
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_QLGIC_QIB) && !heterogeneity) {
      /*Sierra Table*/
      mv2_gather_indexed_num_ppn_conf = 2;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__1PPN;
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 12;
      mv2_size_gather_indexed_tuning_table[1] = 6;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_12ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__12PPN;
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /*Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (gather, 5)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 36, 5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-KNL Table */
      MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 32, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 64, 4, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                             MV2_ARCH_INTEL_PLATINUM_8170_2S_52, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Diamond-Skylake Table */
      MV2_COLL_TUNING_START_TABLE  (gather, 8)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  3, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 26, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__26PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 48, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__48PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 52, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__52PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
    else {
      /*default psm table: Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (gather, 6)
      MV2_COLL_TUNING_ADD_CONF     (gather, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (gather, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (gather)
    }
#endif /* !CHANNEL_PSM */
    {
      /*Stampede Table*/
      mv2_gather_indexed_num_ppn_conf = 3;
      mv2_gather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
		      * mv2_gather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_gather_indexed_tuning_table *)
			       * mv2_gather_indexed_num_ppn_conf);
      mv2_size_gather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							 mv2_gather_indexed_num_ppn_conf);
      mv2_gather_indexed_table_ppn_conf = MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_gather_indexed_table_ppn_conf[0] = 1;
      mv2_size_gather_indexed_tuning_table[0] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN
      table_ptrs[0] = mv2_tmp_gather_indexed_thresholds_table_1ppn;
      
      mv2_gather_indexed_table_ppn_conf[1] = 2;
      mv2_size_gather_indexed_tuning_table[1] = 5;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN
      table_ptrs[1] = mv2_tmp_gather_indexed_thresholds_table_2ppn;
      
      mv2_gather_indexed_table_ppn_conf[2] = 16;
      mv2_size_gather_indexed_tuning_table[2] = 7;
      mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN
      table_ptrs[2] = mv2_tmp_gather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_gather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_gather_indexed_tuning_table[i];
      }
      mv2_gather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_gather_indexed_tuning_table));
      MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_gather_indexed_tuning_table)
		   * mv2_size_gather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_gather_indexed_num_ppn_conf; i++) {
	mv2_gather_indexed_thresholds_table[i] =
	  mv2_gather_indexed_thresholds_table[i - 1]
	  + mv2_size_gather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_gather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_gather_indexed_tuning_table)
		     * mv2_size_gather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
  }
  else {

#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
      mv2_size_gather_tuning_table=6;
      mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
      if((g_use_limic2_coll) && (use_limic_gather)) {
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {32,
	   3,{{0, 4096, &MPIR_Gather_MV2_Direct},
	      {4096, 1048576, &MPIR_Gather_intra},
	      {1048576, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   2,{{0, 131072, 0},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  },
	  {64,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
	      {512, 4096, &MPIR_Gather_MV2_Direct},
	      {4096, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 512, &MPIR_Gather_MV2_Direct},
	      {512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   3,{{0, 4096, 0},
	      {4096, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  },
	  {128,
	   3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
	      {1024, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 1024, &MPIR_Gather_MV2_Direct},
	      {1024, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   3,{{0, 8192, 0},
	      {8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  },
	  {256,
	   3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
	      {1024, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 1024, &MPIR_Gather_MV2_Direct},
	      {1024, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   3,{{0, 8192, 0},
	      {8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  },
	  {512,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
	      {512, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 512, &MPIR_Gather_MV2_Direct},
	      {512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   3,{{0, 8192, 0},
	      {8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  },
	  {1024,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
	      {512, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 512, &MPIR_Gather_MV2_Direct},
	      {512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	   3,{{0, 8192, 0},
	      {8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
	      {131072, -1, USE_GATHER_SINGLE_LEADER}}
	  }                   
	};
	MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		    mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	return 0;

      } else if (g_smp_use_limic2 ) {
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {32,                         
	   2,{{0, 4096, &MPIR_Gather_MV2_Direct},
	      {4096, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {64,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 4096, &MPIR_Gather_MV2_Direct},
	      {4096, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  },
	  {128,
	   3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
	      {1024, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   2,{{0, 8192, &MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {256,
	   3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
	      {1024, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  },
	  {512,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  },
	  {1024,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 8192,&MPIR_Gather_MV2_Direct},
	      {8192, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  }
	};
	MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		    mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	return 0;
      } 
      else                       
#endif /*if defined(_SMP_LIMIC_) */
	{
	  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	    {32,                         
	     2,{{0, 1048576, &MPIR_Gather_MV2_Direct},
		{1048576, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	    {64,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		{1048576,-1, &MPIR_Gather_intra}}},
	    {128,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		{1024, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		{1048576,-1, &MPIR_Gather_intra}}},
	    {256,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		{1024, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		{1048576,-1, &MPIR_Gather_intra}}},
	    {512,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}}, 
	    {1024,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}} 
	  };
	  MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		      mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	  return 0;
	}

    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){

      mv2_size_gather_tuning_table=8;
      mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
      mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	{12,
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, 0}}
	},
	{24,
	 2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, {512, -1,
							  &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{48,                  
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{96,
	 2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, {2048, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{192,
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{384,
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{768,
	 2,{{0, 64, &MPIR_Gather_intra}, {64, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{1024,
	 2,{{0, 32, &MPIR_Gather_intra}, {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}, 
	 1,{{0, -1, 0}}
	}
      };

#else /*#if defined(_SMP_LIMIC_)*/
      mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	{12,
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	{24,
	 2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, {512, -1,
							  &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{48,                  
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{96,
	 2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, {2048, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{192,
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{384,
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							   &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{768,
	 2,{{0, 64, &MPIR_Gather_intra}, {64, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{1024,
	 2,{{0, 32, &MPIR_Gather_intra}, {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}
	}
      };

#endif        
      MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		  mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
      return 0;
    } else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				   MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
      mv2_size_gather_tuning_table=7;
      mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
      mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	{16,
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, 0}}
	},
	{24,
	 2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512,-1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1,  &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{32,
	 2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
	    {1024,-1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{128,
	 2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, 
	    {2048,-1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, 0}}
	},
	{256,
	 2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}, 
	 1,{{0, -1, 0}}
	},
	{512,
	 3,{{0, 32, &MPIR_Gather_intra}, 
	    {32, 8196, &MPIR_Gather_MV2_two_level_Direct},
	    {8196, -1, &MPIR_Gather_MV2_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}, 
	 1,{{0, -1, 0}}
	},
	{1024,
	 2,{{0, 32, &MPIR_Gather_intra}, 
	    {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}}, 
	 1,{{0, -1, 0}}
	}
      };
#else /*#if defined(_SMP_LIMIC_)*/
      mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	{16,
	 2,{{0, 524288, &MPIR_Gather_MV2_Direct},
	    {524288, -1, &MPIR_Gather_intra}},
	 1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	{32,
	 3,{{0, 16384, &MPIR_Gather_MV2_Direct}, 
	    {16384, 131072, &MPIR_Gather_intra},
	    {131072, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{64,
	 3,{{0, 256, &MPIR_Gather_MV2_two_level_Direct}, 
	    {256, 16384, &MPIR_Gather_MV2_Direct},
	    {256, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{128,
	 3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512, 16384, &MPIR_Gather_MV2_Direct},
	    {16384, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{256,
	 3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512, 16384, &MPIR_Gather_MV2_Direct},
	    {16384, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{512,
	 3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512, 16384, &MPIR_Gather_MV2_Direct},
	    {8196, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
	{1024,
	 3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	    {512, 16384, &MPIR_Gather_MV2_Direct},
	    {8196, -1, &MPIR_Gather_MV2_two_level_Direct}},
	 1,{{0, -1, &MPIR_Gather_intra}}},
      };
#endif

      MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		  mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
      return 0;
    } else
#elif defined (CHANNEL_NEMESIS_IB)
      if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			       MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
	mv2_size_gather_tuning_table=6;
	mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						  sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
	if((g_use_limic2_coll) && (use_limic_gather)) {
	  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	    {32,
	     3,{{0, 4096, &MPIR_Gather_MV2_Direct},
		{4096, 1048576, &MPIR_Gather_intra},
		{1048576, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     1,{{0, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     2,{{0, 131072, 0},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    },
	    {64,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
		{512, 4096, &MPIR_Gather_MV2_Direct},
		{4096, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 512, &MPIR_Gather_MV2_Direct},
		{512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     3,{{0, 4096, 0},
		{4096, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    },
	    {128,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
		{1024, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 1024, &MPIR_Gather_MV2_Direct},
		{1024, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     3,{{0, 8192, 0},
		{8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    },
	    {256,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
		{1024, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 1024, &MPIR_Gather_MV2_Direct},
		{1024, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     3,{{0, 8192, 0},
		{8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    },
	    {512,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
		{512, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 512, &MPIR_Gather_MV2_Direct},
		{512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     3,{{0, 8192, 0},
		{8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    },
	    {1024,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct},
		{512, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 512, &MPIR_Gather_MV2_Direct},
		{512, -1, &MPIR_Intra_node_LIMIC_Gather_MV2}},
	     3,{{0, 8192, 0},
		{8192, 131072, USE_GATHER_LINEAR_PT_BINOMIAL},
		{131072, -1, USE_GATHER_SINGLE_LEADER}}
	    }                   
	  };
	  MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		      mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	  return 0;

	} else if (g_smp_use_limic2 ) {
	  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	    {32,                         
	     2,{{0, 4096, &MPIR_Gather_MV2_Direct},
		{4096, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, 0}}
	    },
	    {64,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 4096, &MPIR_Gather_MV2_Direct},
		{4096, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, 0}}
	    },
	    {128,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct},
		{1024, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     2,{{0, 8192, &MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, 0}}
	    },
	    {256,
	     3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		{1024, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, 0}}
	    },
	    {512,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, 0}}
	    },
	    {1024,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 8192,&MPIR_Gather_MV2_Direct},
		{8192, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, 0}}
	    }
	  };
	  MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		      mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	  return 0;
	} 
	else                       
#endif /*if defined(_SMP_LIMIC_) */
	  {
	    mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	      {32,                         
	       2,{{0, 1048576, &MPIR_Gather_MV2_Direct},
		  {1048576, -1, &MPIR_Gather_intra}},
	       1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	      {64,
	       3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		  {512, 8192, &MPIR_Gather_MV2_Direct},
		  {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	       2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		  {1048576,-1, &MPIR_Gather_intra}}},
	      {128,
	       3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		  {1024, 8192, &MPIR_Gather_MV2_Direct},
		  {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	       2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		  {1048576,-1, &MPIR_Gather_intra}}},
	      {256,
	       3,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		  {1024, 8192, &MPIR_Gather_MV2_Direct},
		  {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	       2,{{0, 1048576, &MPIR_Gather_MV2_Direct}, 
		  {1048576,-1, &MPIR_Gather_intra}}},
	      {512,
	       3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		  {512, 8192, &MPIR_Gather_MV2_Direct},
		  {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	       1,{{0, -1, &MPIR_Gather_MV2_Direct}}}, 
	      {1024,
	       3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		  {512, 8192, &MPIR_Gather_MV2_Direct},
		  {8192, -1, &MPIR_Gather_MV2_two_level_Direct}},
	       1,{{0, -1, &MPIR_Gather_MV2_Direct}}} 
	    };
	    MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
			mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	    return 0;
	  }

      } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				      MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){

	mv2_size_gather_tuning_table=8;
	mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						  sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {12,
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  },
	  {24,
	   2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, {512, -1,
							    &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {48,                  
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {96,
	   2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, {2048, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {192,
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {384,
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {768,
	   2,{{0, 64, &MPIR_Gather_intra}, {64, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {1024,
	   2,{{0, 32, &MPIR_Gather_intra}, {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}, 
	   1,{{0, -1, 0}}
	  }
	};

#else /*#if defined(_SMP_LIMIC_)*/
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {12,
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	  {24,
	   2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, {512, -1,
							    &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {48,                  
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {96,
	   2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, {2048, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {192,
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {384,
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, {1024, -1,
							     &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {768,
	   2,{{0, 64, &MPIR_Gather_intra}, {64, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {1024,
	   2,{{0, 32, &MPIR_Gather_intra}, {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}
	  }
	};

#endif        
	MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		    mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	return 0;
      } else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				     MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
	mv2_size_gather_tuning_table=7;
	mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						  sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {16,
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, 0}}
	  },
	  {24,
	   2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512,-1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1,  &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {32,
	   2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
	      {1024,-1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {128,
	   2,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, 
	      {2048,-1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, 0}}
	  },
	  {256,
	   2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}, 
	   1,{{0, -1, 0}}
	  },
	  {512,
	   3,{{0, 32, &MPIR_Gather_intra}, 
	      {32, 8196, &MPIR_Gather_MV2_two_level_Direct},
	      {8196, -1, &MPIR_Gather_MV2_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}, 
	   1,{{0, -1, 0}}
	  },
	  {1024,
	   2,{{0, 32, &MPIR_Gather_intra}, 
	      {32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}}, 
	   1,{{0, -1, 0}}
	  }
	};
#else /*#if defined(_SMP_LIMIC_)*/
	mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	  {16,
	   2,{{0, 524288, &MPIR_Gather_MV2_Direct},
	      {524288, -1, &MPIR_Gather_intra}},
	   1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	  {32,
	   3,{{0, 16384, &MPIR_Gather_MV2_Direct}, 
	      {16384, 131072, &MPIR_Gather_intra},
	      {131072, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {64,
	   3,{{0, 256, &MPIR_Gather_MV2_two_level_Direct}, 
	      {256, 16384, &MPIR_Gather_MV2_Direct},
	      {256, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {128,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 16384, &MPIR_Gather_MV2_Direct},
	      {16384, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {256,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 16384, &MPIR_Gather_MV2_Direct},
	      {16384, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {512,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 16384, &MPIR_Gather_MV2_Direct},
	      {8196, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	  {1024,
	   3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
	      {512, 16384, &MPIR_Gather_MV2_Direct},
	      {8196, -1, &MPIR_Gather_MV2_two_level_Direct}},
	   1,{{0, -1, &MPIR_Gather_intra}}},
	};
#endif

	MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		    mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	return 0;
      } else
#endif
#endif /* !CHANNEL_PSM */
	{ 
	  mv2_size_gather_tuning_table=7;
	  mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table*
						    sizeof (mv2_gather_tuning_table)); 
#if defined(_SMP_LIMIC_)
	  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	    {16,
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, 0}}
	    },
	    {24,
	     2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512,-1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1,  &MPIR_Gather_intra}},
	     1,{{0, -1, 0}}
	    },
	    {32,
	     2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		{1024,-1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, 0}}
	    },
	    {128,
	     3,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, 
		{2048,65536, &MPIR_Gather_MV2_Direct},
		{65536,-1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}},
	     1,{{0, -1, 0}}
	    },
	    {256,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 65536, &MPIR_Gather_MV2_Direct},
		{65536, -1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}}, 
	     1,{{0, -1, 0}}
	    },
	    {512,
	     4,{{0, 32, &MPIR_Gather_intra}, 
		{32, 8196, &MPIR_Gather_MV2_two_level_Direct},
		{8196, 65536, &MPIR_Gather_MV2_Direct},
		{65536, -1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}}, 
	     1,{{0, -1, 0}}
	    },
	    {1024,
	     2,{{0, 32, &MPIR_Gather_intra}, 
		{32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}, 
	     1,{{0, -1, 0}}
	    }
	  };
#else /*#if defined(_SMP_LIMIC_)*/
	  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[]={
	    {16,
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}},
	    {24,
	     2,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, -1,&MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, &MPIR_Gather_intra}}},
	    {32,
	     2,{{0, 1024, &MPIR_Gather_MV2_two_level_Direct}, 
		{1024, -1,&MPIR_Gather_MV2_Direct}},
	     1,{{0, -1, &MPIR_Gather_intra}}},
	    {128,
	     3,{{0, 2048, &MPIR_Gather_MV2_two_level_Direct}, 
		{2048, 65536, &MPIR_Gather_MV2_Direct},
		{65536, -1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}}},
	    {256,
	     3,{{0, 512, &MPIR_Gather_MV2_two_level_Direct}, 
		{512, 65536, &MPIR_Gather_MV2_Direct},
		{65536, -1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}}},
	    {512,
	     4,{{0, 32, &MPIR_Gather_intra}, 
		{32, 8196, &MPIR_Gather_MV2_two_level_Direct},
		{8196, 65536, &MPIR_Gather_MV2_Direct},
		{65536, -1, &MPIR_Gather_MV2_Direct_Blk}},
	     1,{{0, -1, &MPIR_Gather_intra}}},
	    {1024,
	     2,{{0, 32, &MPIR_Gather_intra}, 
		{32, -1, &MPIR_Gather_MV2_two_level_Direct}},
	     1,{{0, -1, &MPIR_Gather_MV2_Direct}}
	    }
	  };
#endif

	  MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table,
		      mv2_size_gather_tuning_table * sizeof (mv2_gather_tuning_table));
	  return 0;
	}
  }
  return 0;
}

void MV2_cleanup_gather_tuning_table()
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_gather_tuning) {
    MPIU_Free(mv2_gather_indexed_thresholds_table[0]);
    MPIU_Free(mv2_gather_indexed_table_ppn_conf);
    MPIU_Free(mv2_size_gather_indexed_tuning_table);
    if (mv2_gather_indexed_thresholds_table != NULL) {
      MPIU_Free(mv2_gather_indexed_thresholds_table);
    }
  }
  else {
    if(mv2_gather_thresholds_table != NULL) { 
      MPIU_Free(mv2_gather_thresholds_table); 
    }
  }
} 

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
  return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}

#if defined(_SMP_LIMIC_)
int  MV2_intranode_multi_lvl_Gather_is_define(char *mv2_user_gather_inter,
                                              char *mv2_user_gather_intra, 
                                              char *mv2_user_gather_intra_multi_lvl)
{

  int i;
  int nb_element = count_sep(mv2_user_gather_inter) + 1;

  // If one gather tuning table is already defined 
  if(mv2_gather_thresholds_table != NULL) {
    MPIU_Free(mv2_gather_thresholds_table);
  }


  mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[1];
  mv2_size_gather_tuning_table = 1;
    
  // We realloc the space for the new gather tuning table
  mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table *
					    sizeof (mv2_gather_tuning_table));

  if (nb_element == 1) {
    mv2_tmp_gather_thresholds_table[0].numproc = 1;
    mv2_tmp_gather_thresholds_table[0].size_inter_table = 1;
    mv2_tmp_gather_thresholds_table[0].inter_leader[0].min = 0;
    mv2_tmp_gather_thresholds_table[0].inter_leader[0].max = -1;
    switch (atoi(mv2_user_gather_inter)) {
    case 1:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_intra;
      break;
    case 2:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct;
      break;
    case 3:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_two_level_Direct;
      break;
	/* added by abu naser */
	case 4:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CHS;
      break;
	case 5:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CNoSM_RR;
      break;  
	/* end of add */  
    default:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct;
    }
  } else {
    char *dup, *p, *save_p;
    regmatch_t match[NMATCH];
    regex_t preg;
    const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

    if (!(dup = MPIU_Strdup(mv2_user_gather_inter))) {
      fprintf(stderr, "failed to duplicate `%s'\n",
	      mv2_user_gather_inter);
      return 1;
    }

    if (regcomp(&preg, regexp, REG_EXTENDED)) {
      fprintf(stderr, "failed to compile regexp `%s'\n",
	      mv2_user_gather_inter);
      MPIU_Free(dup);
      return 2;
    }

    mv2_tmp_gather_thresholds_table[0].numproc = 1;
    mv2_tmp_gather_thresholds_table[0].size_inter_table = nb_element;
    i = 0;
    for (p = strtok_r(dup, ",", &save_p); p;
	 p = strtok_r(NULL, ",", &save_p)) {
      if (regexec(&preg, p, NMATCH, match, 0)) {
	fprintf(stderr, "failed to match on `%s'\n", p);
	regfree(&preg);
	MPIU_Free(dup);
	return 2;
      }
      /* given () start at 1 */
      switch (atoi(p + match[1].rm_so)) {
      case 1:
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	  &MPIR_Gather_intra;
	break;
      case 2:
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
	break;
      case 3:
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_two_level_Direct;
	break;
      default:
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      }
      mv2_tmp_gather_thresholds_table[0].inter_leader[i].min = atoi(p + match[2].rm_so);

      if (p[match[3].rm_so] == '+') {
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].max = -1;
      } else {
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].max =
	  atoi(p + match[3].rm_so);
      }

      i++;
    }
    MPIU_Free(dup);
    regfree(&preg);
  }
    
  mv2_tmp_gather_thresholds_table[0].size_intra_table = 1;
  if (mv2_user_gather_intra == NULL) {
    int multi_lvl_scheme = atoi(mv2_user_gather_intra_multi_lvl);

    if(multi_lvl_scheme >=1 && multi_lvl_scheme <=8){

      mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	&MPIR_Intra_node_LIMIC_Gather_MV2;
    } else {
      /*If mv2_user_gather_intra_multi_lvl == 0 or anyother value
       * we do use the any limic schemes. so use the default gather
       * algorithms*/
      mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	&MPIR_Gather_intra;
    }
  } else {
    /*Tuning for intra-node schemes*/
    int nb_intra_element = count_sep(mv2_user_gather_intra) + 1;

    // If one gather tuning table is already defined 
    if(mv2_gather_thresholds_table != NULL) {
      MPIU_Free(mv2_gather_thresholds_table);
    }


    //mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[1];
    mv2_size_gather_tuning_table = 1;

    // We realloc the space for the new gather tuning table
    mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table *
					      sizeof (mv2_gather_tuning_table));

    if (nb_intra_element == 1) {
      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].intra_node[0].min = 0;
      mv2_tmp_gather_thresholds_table[0].intra_node[0].max = -1;
      switch (atoi(mv2_user_gather_intra)) {
      case 2:
	mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Intra_node_LIMIC_Gather_MV2;
	break;
      case 0: 
      case 1:
	/*0- Direct algo*/
	/*1- Binomial algo*/

	/*For limic gather schemes, we only use
	 * MPIR_Intra_node_LIMIC_Gather_MV2 for
	 * intra node communication. So all the other 
	 * intra node algo are just place holders*/
      default:          
	mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Intra_node_LIMIC_Gather_MV2;
      }
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_gather_intra))) {
	fprintf(stderr, "failed to duplicate `%s'\n",
		mv2_user_gather_intra);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n",
		mv2_user_gather_intra);
	MPIU_Free(dup);
	return 2;
      }

      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].size_intra_table = nb_intra_element;
      i = 0;
      for (p = strtok_r(dup, ",", &save_p); p;
	   p = strtok_r(NULL, ",", &save_p)) {
	if (regexec(&preg, p, NMATCH, match, 0)) {
	  fprintf(stderr, "failed to match on `%s'\n", p);
	  regfree(&preg);
	  MPIU_Free(dup);
	  return 2;
	}
	/* given () start at 1 */
	switch (atoi(p + match[1].rm_so)) {
	case 2:
	  mv2_tmp_gather_thresholds_table[0].intra_node[i].MV2_pt_Gather_function =
	    &MPIR_Intra_node_LIMIC_Gather_MV2;
	  break;
	case 0:                                  
	case 1:
	  /*0- Direct algo*/
	  /*1- Binomial algo*/

	  /*For limic gather schemes, we only use
	   * MPIR_Intra_node_LIMIC_Gather_MV2 for
	   * intra node communication. So all the other 
	   * intra node algo are just place holders*/
	default:
	  mv2_tmp_gather_thresholds_table[0].intra_node[i].MV2_pt_Gather_function =
	    &MPIR_Intra_node_LIMIC_Gather_MV2;
	}
	mv2_tmp_gather_thresholds_table[0].intra_node[i].min = atoi(p + match[2].rm_so);

	if (p[match[3].rm_so] == '+') {
	  mv2_tmp_gather_thresholds_table[0].intra_node[i].max = -1;
	} else {
	  mv2_tmp_gather_thresholds_table[0].intra_node[i].max =
	    atoi(p + match[3].rm_so);
	}

	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
  }

  /*Tuning for intra-node multi-leader limic schemes*/
  mv2_tmp_gather_thresholds_table[0].nb_limic_scheme = 1;
  if(mv2_user_gather_intra_multi_lvl == NULL) {
    mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = 0; 
  } else {
    i=0;
    int nb_intra_multi_lvl_element = count_sep(mv2_user_gather_intra_multi_lvl) + 1;

    // If one gather tuning table is already defined 
    if(mv2_gather_thresholds_table != NULL) {
      MPIU_Free(mv2_gather_thresholds_table);
    }


    //mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[1];
    mv2_size_gather_tuning_table = 1;

    // We realloc the space for the new gather tuning table
    mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table *
					      sizeof (mv2_gather_tuning_table));

    if (nb_intra_multi_lvl_element == 1) {
      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].min = 0;
      mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].max = -1;
      switch (atoi(mv2_user_gather_intra_multi_lvl)) {
      case 1:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_PT_PT_BINOMIAL;
	break;
      case 2:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_PT_PT_DIRECT;
	break;
      case 3:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_PT_LINEAR_BINOMIAL;
	break;
      case 4:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_PT_LINEAR_DIRECT;
	break;
      case 5:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_LINEAR_PT_BINOMIAL;
	break;
      case 6:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_LINEAR_PT_DIRECT;
	break;
      case 7:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_LINEAR_LINEAR;
	break;
      case 8:
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = USE_GATHER_SINGLE_LEADER;
	break;
      default:
	/*None of the limic schemes are selected. Fallback to default mode*/
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[0].scheme = 0; 
      }
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_gather_intra_multi_lvl))) {
	fprintf(stderr, "failed to duplicate `%s'\n",
		mv2_user_gather_intra_multi_lvl);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n",
		mv2_user_gather_intra_multi_lvl);
	MPIU_Free(dup);
	return 2;
      }

      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].nb_limic_scheme= nb_intra_multi_lvl_element;
      i = 0;
      for (p = strtok_r(dup, ",", &save_p); p;
	   p = strtok_r(NULL, ",", &save_p)) {
	if (regexec(&preg, p, NMATCH, match, 0)) {
	  fprintf(stderr, "failed to match on `%s'\n", p);
	  regfree(&preg);
	  MPIU_Free(dup);
	  return 2;
	}
	/* given () start at 1 */
	switch (atoi(p + match[1].rm_so)) {
	case 1:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_PT_PT_BINOMIAL;
	  break;
	case 2:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_PT_PT_DIRECT;
	  break;
	case 3:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_PT_LINEAR_BINOMIAL;
	  break;
	case 4:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_PT_LINEAR_DIRECT;
	  break;
	case 5:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_LINEAR_PT_BINOMIAL;
	  break;
	case 6:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_LINEAR_PT_DIRECT;
	  break;
	case 7:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_LINEAR_LINEAR;
	  break;
	case 8:
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = USE_GATHER_SINGLE_LEADER;
	  break;
	default:
	  /*None of the limic schemes are selected. Fallback to default mode*/
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].scheme = 0; 
	}
	mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].min = atoi(p + match[2].rm_so);

	if (p[match[3].rm_so] == '+') {
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].max = -1;
	} else {
	  mv2_tmp_gather_thresholds_table[0].limic_gather_scheme[i].max =
	    atoi(p + match[3].rm_so);
	}

	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
  }
  MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table, sizeof
	      (mv2_gather_tuning_table));
  return 0;
}
#endif /*#if defined(_SMP_LIMIC_) */

int MV2_internode_Gather_is_define(char *mv2_user_gather_inter,
                                   char *mv2_user_gather_intra)
{

  int i;
  int nb_element = count_sep(mv2_user_gather_inter) + 1;

  if (mv2_use_indexed_tuning || mv2_use_indexed_gather_tuning) {

    /* If one gather tuning table is already defined */
    if (mv2_gather_indexed_thresholds_table != NULL) {
      if (mv2_gather_indexed_thresholds_table[0] != NULL) {
          MPIU_Free(mv2_gather_indexed_thresholds_table[0]);
      }
      MPIU_Free(mv2_gather_indexed_thresholds_table);
    }

    mv2_gather_indexed_tuning_table mv2_tmp_gather_indexed_thresholds_table[1];
    mv2_gather_indexed_num_ppn_conf = 1;
    if (mv2_size_gather_indexed_tuning_table == NULL) {
      mv2_size_gather_indexed_tuning_table =
	MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
    }
    mv2_size_gather_indexed_tuning_table[0] = 1;

    if (mv2_gather_indexed_table_ppn_conf == NULL) {
      mv2_gather_indexed_table_ppn_conf =
	MPIU_Malloc(mv2_gather_indexed_num_ppn_conf * sizeof(int));
    }
    /* -1 indicates user defined algorithm */
    mv2_gather_indexed_table_ppn_conf[0] = -1;

    /* We realloc the space for the new gather tuning table */
    mv2_gather_indexed_thresholds_table =
      MPIU_Malloc(mv2_gather_indexed_num_ppn_conf *
		  sizeof(mv2_gather_indexed_tuning_table *));
    mv2_gather_indexed_thresholds_table[0] =
      MPIU_Malloc(mv2_size_gather_indexed_tuning_table[0] *
		  sizeof(mv2_gather_indexed_tuning_table));

    if (nb_element == 1) {
      mv2_tmp_gather_indexed_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_indexed_thresholds_table[0].size_inter_table = 1;
      mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].msg_sz = 1;
      mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].msg_sz = 1;
      switch (atoi(mv2_user_gather_inter)) {
      case 1:
	mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_intra;
	break;
      case 2:
	mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
	break;
      case 3:
	mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_two_level_Direct;
	break;
	/* added by abu naser */
	case 4:
      mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CHS;
      break;
	case 5:
      mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CNoSM_RR;
      break;  
	/* end of add */  
      default:
	mv2_tmp_gather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      }
      if (mv2_user_gather_intra == NULL) {
	mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      } else {
	if (atoi(mv2_user_gather_intra) == 1) {
	  mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	    &MPIR_Gather_intra;
	} else {
	  mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_Direct;
	}
      }
    }
    mv2_tmp_gather_indexed_thresholds_table[0].size_intra_table = 1;
    if (mv2_user_gather_intra == NULL) {
      mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct;
    } else {
      if (atoi(mv2_user_gather_intra) == 1) {
	mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_intra;
      } else {
	mv2_tmp_gather_indexed_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      }
    }
    MPIU_Memcpy(mv2_gather_indexed_thresholds_table[0], mv2_tmp_gather_indexed_thresholds_table, sizeof
		(mv2_gather_indexed_tuning_table));
  }
  else {
    // If one gather tuning table is already defined 
    if(mv2_gather_thresholds_table != NULL) {
      MPIU_Free(mv2_gather_thresholds_table);
    }


    mv2_gather_tuning_table mv2_tmp_gather_thresholds_table[1];
    mv2_size_gather_tuning_table = 1;
    
    // We realloc the space for the new gather tuning table
    mv2_gather_thresholds_table = MPIU_Malloc(mv2_size_gather_tuning_table *
					      sizeof (mv2_gather_tuning_table));

    if (nb_element == 1) {
      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].size_inter_table = 1;
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].min = 0;
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].max = -1;
      switch (atoi(mv2_user_gather_inter)) {
      case 1:
	mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_intra;
	break;
      case 2:
	mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
	break;
      case 3:
	mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_two_level_Direct;
	break;
	/* added by abu naser */
	case 4:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CHS;
      break;
	case 5:
      mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct_CNoSM_RR;
      break;  
	/* end of add */  
      default:
	mv2_tmp_gather_thresholds_table[0].inter_leader[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      }
      if (mv2_user_gather_intra == NULL) {
	mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      } else {
	if (atoi(mv2_user_gather_intra) == 1) {
	  mv2_gather_thresholds_table[0].
	    intra_node[0].MV2_pt_Gather_function = &MPIR_Gather_intra;
	} else {
	  mv2_gather_thresholds_table[0].
	    intra_node[0].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_Direct;
	}

      }
    } else {
      char *dup, *p, *save_p;
      regmatch_t match[NMATCH];
      regex_t preg;
      const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

      if (!(dup = MPIU_Strdup(mv2_user_gather_inter))) {
	fprintf(stderr, "failed to duplicate `%s'\n",
		mv2_user_gather_inter);
	return 1;
      }

      if (regcomp(&preg, regexp, REG_EXTENDED)) {
	fprintf(stderr, "failed to compile regexp `%s'\n",
		mv2_user_gather_inter);
	MPIU_Free(dup);
	return 2;
      }

      mv2_tmp_gather_thresholds_table[0].numproc = 1;
      mv2_tmp_gather_thresholds_table[0].size_inter_table = nb_element;
      i = 0;
      for (p = strtok_r(dup, ",", &save_p); p;
	   p = strtok_r(NULL, ",", &save_p)) {
	if (regexec(&preg, p, NMATCH, match, 0)) {
	  fprintf(stderr, "failed to match on `%s'\n", p);
	  regfree(&preg);
	  MPIU_Free(dup);
	  return 2;
	}
	/* given () start at 1 */
	switch (atoi(p + match[1].rm_so)) {
	case 1:
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	    &MPIR_Gather_intra;
	  break;
	case 2:
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_Direct;
	  break;
	case 3:
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_two_level_Direct;
	  break;
	default:
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_Direct;
	}
	mv2_tmp_gather_thresholds_table[0].inter_leader[i].min = atoi(p + match[2].rm_so);

	if (p[match[3].rm_so] == '+') {
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].max = -1;
	} else {
	  mv2_tmp_gather_thresholds_table[0].inter_leader[i].max =
	    atoi(p + match[3].rm_so);
	}

	i++;
      }
      MPIU_Free(dup);
      regfree(&preg);
    }
    mv2_tmp_gather_thresholds_table[0].size_intra_table = 1;
    if (mv2_user_gather_intra == NULL) {
      mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	&MPIR_Gather_MV2_Direct;
    } else {
      if (atoi(mv2_user_gather_intra) == 1) {
	mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_intra;
      } else {
	mv2_tmp_gather_thresholds_table[0].intra_node[0].MV2_pt_Gather_function =
	  &MPIR_Gather_MV2_Direct;
      }
    }
    MPIU_Memcpy(mv2_gather_thresholds_table, mv2_tmp_gather_thresholds_table, sizeof
		(mv2_gather_tuning_table));
  }
  return 0;
}

int MV2_intranode_Gather_is_define(char *mv2_user_gather_intra)
{

  if (mv2_use_indexed_tuning || mv2_use_indexed_gather_tuning) {
    //nop
  }
  else {
    int i, j;
    for (i = 0; i < mv2_size_gather_tuning_table; i++) {
      for (j = 0; j < mv2_gather_thresholds_table[i].size_intra_table; j++) {
	if (atoi(mv2_user_gather_intra) == 1) {
	  mv2_gather_thresholds_table[i].
	    intra_node[j].MV2_pt_Gather_function = &MPIR_Gather_intra;
	} else {
	  mv2_gather_thresholds_table[i].
	    intra_node[j].MV2_pt_Gather_function =
	    &MPIR_Gather_MV2_Direct;
	}
      }
    }
  }
  return 0;
}

void MV2_user_gather_switch_point_is_define(int mv2_user_gather_switch_point)
{
  int i;
  for (i = 0; i < mv2_size_gather_tuning_table; i++) {
    mv2_gather_thresholds_table[0].inter_leader[1].min =
      mv2_user_gather_switch_point;
    mv2_gather_thresholds_table[0].inter_leader[0].max =
      mv2_user_gather_switch_point;
  }
}
