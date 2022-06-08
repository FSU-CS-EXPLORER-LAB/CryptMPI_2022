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
#include "allgather_tuning.h"
#include "tuning/allgather_arch_tuning.h"
#include "mv2_arch_hca_detect.h"

int *mv2_allgather_table_ppn_conf = NULL;
int mv2_allgather_num_ppn_conf = 1;
int *mv2_size_allgather_tuning_table = NULL;
mv2_allgather_tuning_table **mv2_allgather_thresholds_table = NULL;

int *mv2_allgather_indexed_table_ppn_conf = NULL;
int mv2_allgather_indexed_num_ppn_conf = 1;
int *mv2_size_allgather_indexed_tuning_table = NULL;
mv2_allgather_indexed_tuning_table **mv2_allgather_indexed_thresholds_table = NULL;

int MV2_set_allgather_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
  int agg_table_sum = 0;
  int i;
  
  if (mv2_use_indexed_tuning || mv2_use_indexed_allgather_tuning) {
    mv2_allgather_indexed_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Lonestar Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 12;
      mv2_size_allgather_indexed_tuning_table[2] = 6;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_12ppn[] =
	GEN2__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    if ((MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20, MV2_HCA_MLX_CX_CONNIB) ||
        MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20, MV2_HCA_MLX_CX_CONNIB)) && !heterogeneity) {
      /*PSG Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 3;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 20;
      mv2_size_allgather_indexed_tuning_table[2] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_20ppn[] =
	GEN2__INTEL_XEON_E5_2690_V2_2S_20__MLX_CX_CONNIB__20PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_20ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 8;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_QDR__16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[0] = 3;
	table_ptrs[0] = mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[0] = 2;
	table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[1] = 3;
	table_ptrs[1] = mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[1] = 2;
	table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
#endif
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn[] =
        GEN2_CMA__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
        GEN2__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[2] = 4;
	table_ptrs[2] = mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[2] = 5;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_allgather_indexed_num_ppn_conf = 4;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[0] = 4;
	table_ptrs[0] = mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[1] = 4;
	table_ptrs[1] = mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
#endif

      mv2_allgather_indexed_table_ppn_conf[2] = 4;
      mv2_size_allgather_indexed_tuning_table[2] = 1;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_4ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_4ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__4PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_cma_allgather_indexed_thresholds_table_4ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[2] = 1;
	table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_4ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[2] = 1;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_4ppn;
#endif     

      mv2_allgather_indexed_table_ppn_conf[3] = 16;
      mv2_size_allgather_indexed_tuning_table[3] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[3] = 5;
	table_ptrs[3] = mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[3] = 7;
	table_ptrs[3] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[3] = 7;
      table_ptrs[3] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 3;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 32;
      mv2_size_allgather_indexed_tuning_table[2] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_32ppn[] =
	GEN2__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_32ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* RI2 table */
force_default_tables:
      MV2_COLL_TUNING_START_TABLE  (allgather, 6)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, GEN2__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  4, GEN2_CMA__RI2__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, GEN2__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  5, GEN2_CMA__RI2__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, GEN2__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  5, GEN2_CMA__RI2__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, GEN2__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  5, GEN2_CMA__RI2__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, GEN2__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16,  5, GEN2_CMA__RI2__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 28, 5, GEN2__RI2__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 28, 5, GEN2_CMA__RI2__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7551_64, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* AMD EPYC table */
      MV2_COLL_TUNING_START_TABLE  (allgather, 7)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  3, GEN2__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  3, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  4, GEN2__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  4, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  4, GEN2__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  4, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  4, GEN2__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  4, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 4, GEN2__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 4, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 4, GEN2__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 32, 4, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 64, 4, GEN2__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 64, 4, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_AMD_EPYC_7742_128, MV2_HCA_ANY) && !heterogeneity) {
      /* AMD EPYC rome table */
      MV2_COLL_TUNING_START_TABLE  (allgather, 7)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  1, GEN2_CMA__AMD_EPYC__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  2, GEN2_CMA__AMD_EPYC__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  2, GEN2_CMA__AMD_EPYC__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  2, GEN2_CMA__AMD_EPYC__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 2, GEN2_CMA__AMD_EPYC__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 2, GEN2_CMA__AMD_EPYC__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 64, 2, GEN2_CMA__AMD_EPYC__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* Haswell HDR nodes on NOWLAB */
      MV2_COLL_TUNING_START_TABLE  (allgather, 6)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  2, GEN2_CMA__NOWHASWELL__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  2, GEN2_CMA__NOWHASWELL__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  3, GEN2_CMA__NOWHASWELL__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  3, GEN2_CMA__NOWHASWELL__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 3, GEN2_CMA__NOWHASWELL__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 20, 3, GEN2_CMA__NOWHASWELL__20PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather) 
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_PLATINUM_8280_2S_56, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Frontera */
      MV2_COLL_TUNING_START_TABLE  (allgather, 8)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  4, GEN2_CMA__FRONTERA__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  5, GEN2_CMA__FRONTERA__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  5, GEN2_CMA__FRONTERA__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  5, GEN2_CMA__FRONTERA__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 5, GEN2_CMA__FRONTERA__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 28, 3, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 28, 3, GEN2_CMA__FRONTERA__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 4, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 32, 4, GEN2_CMA__FRONTERA__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 56, 5, GEN2_CMA__FRONTERA__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)  
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Mayer */
      MV2_COLL_TUNING_START_TABLE  (allgather, 8)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  3, GEN2_CMA__MAYER__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  4, GEN2_CMA__MAYER__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  4, GEN2_CMA__MAYER__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  4, GEN2_CMA__MAYER__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 4, GEN2_CMA__MAYER__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 28, 4, GEN2_CMA__MAYER__28PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 32, 4, GEN2_CMA__MAYER__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 56, 4, GEN2_CMA__MAYER__56PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if(MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_ARM_CAVIUM_V8_2S_32, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Catalyst */
      MV2_COLL_TUNING_START_TABLE  (allgather, 7)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 32, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 64, 4, GEN2_CMA__ARM_CAVIUM_V8_2S_32_MLX_CX_EDR__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_ARM_CAVIUM_V8_2S_28, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /* ARM system at Hartree Center */
      MV2_COLL_TUNING_START_TABLE  (allgather, 5)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  2, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 24,  3, GEN2_CMA__ARM_CAVIUM_V8_2S_28_MLX_CX_FDR__24PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER8, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Ray Table */
      int pg_size = MPIDI_PG_Get_size(MPIDI_Process.my_pg);
      if (pg_size > 64) goto force_default_tables;

      MV2_COLL_TUNING_START_TABLE  (allgather, 3)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  4, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  2, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  2, GEN2_CMA__IBM_POWER8_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				    MV2_ARCH_IBM_POWER9, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* Sierra Table: Use table for Ray temporarily */
      int pg_size = MPIDI_PG_Get_size(MPIDI_Process.my_pg);
      if (pg_size > 64) goto force_default_tables;

      MV2_COLL_TUNING_START_TABLE  (allgather, 9)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, GEN2__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  4, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, GEN2__IBM_POWER9_MLX_CX_EDR__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, GEN2__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 6,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__6PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, GEN2__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 5, GEN2__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 22, 5, GEN2__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 22, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__22PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 5, GEN2__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 32, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 44, 5, GEN2__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 44, 5, GEN2_CMA__IBM_POWER9_MLX_CX_EDR__44PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*RI Table*/
      MV2_COLL_TUNING_START_TABLE  (allgather, 4)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  2, GEN2__RI__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  2, GEN2__RI__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  1, GEN2__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  8, GEN2__RI__8PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  1, GEN2_CMA__RI__4PPN)
      MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  6, GEN2_CMA__RI__8PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Comet Table*/
      mv2_allgather_indexed_num_ppn_conf = 1;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 24;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_24ppn[] =
	  GEN2__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
      /*
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_24ppn[] =
	  GEN2_CMA__INTEL_XEON_E5_2680_24__MLX_CX_FDR__24PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_cma_allgather_indexed_thresholds_table_24ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[0] = 6;
	table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_24ppn;
      }
#else
      */
      mv2_size_allgather_indexed_tuning_table[0] = 6;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_24ppn;
      /*
#endif
      */
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_ANY, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
      /* RI2 table */
       MV2_COLL_TUNING_START_TABLE  (allgather, 6)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, GEN2__RI2__1PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  4, GEN2_CMA__RI2__1PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, GEN2__RI2__2PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  5, GEN2_CMA__RI2__2PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, GEN2__RI2__4PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  5, GEN2_CMA__RI2__4PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, GEN2__RI2__8PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  5, GEN2_CMA__RI2__8PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, GEN2__RI2__16PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16,  5, GEN2_CMA__RI2__16PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 28, 5, GEN2__RI2__28PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 28, 5, GEN2_CMA__RI2__28PPN)
       MV2_COLL_TUNING_FINISH_TABLE (allgather)
       
      
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                MV2_ARCH_ANY, MV2_HCA_MLX_CX_HDR) && !heterogeneity) {
      /* RI2 table */
       MV2_COLL_TUNING_START_TABLE  (allgather, 6)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, GEN2__RI2__1PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 1,  4, GEN2_CMA__RI2__1PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, GEN2__RI2__2PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 2,  5, GEN2_CMA__RI2__2PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, GEN2__RI2__4PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 4,  5, GEN2_CMA__RI2__4PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, GEN2__RI2__8PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 8,  5, GEN2_CMA__RI2__8PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, GEN2__RI2__16PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 16,  5, GEN2_CMA__RI2__16PPN)
       MV2_COLL_TUNING_ADD_CONF     (allgather, 28, 5, GEN2__RI2__28PPN)
       MV2_COLL_TUNING_ADD_CONF_CMA (allgather, 28, 5, GEN2_CMA__RI2__28PPN)
       MV2_COLL_TUNING_FINISH_TABLE (allgather)
       
      
    }
    else {
      /*Stampede Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[0] = 4;
	table_ptrs[0] = mv2_tmp_cma_allgather_indexed_thresholds_table_1ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[0] = 5;
	table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
#endif
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[1] = 4;
	table_ptrs[1] = mv2_tmp_cma_allgather_indexed_thresholds_table_2ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[1] = 6;
	table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[1] = 6;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
#endif
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	GEN2__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      mv2_allgather_indexed_tuning_table mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn[] =
	GEN2_CMA__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
#if defined(_SMP_CMA_)
      if (g_smp_use_cma) {
	mv2_size_allgather_indexed_tuning_table[2] = 5;
	table_ptrs[2] = mv2_tmp_cma_allgather_indexed_thresholds_table_16ppn;
      }
      else {
	mv2_size_allgather_indexed_tuning_table[2] = 7;
	table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      }
#else
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
#endif
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#elif defined (CHANNEL_NEMESIS_IB)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Lonestar Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 12;
      mv2_size_allgather_indexed_tuning_table[2] = 3;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_12ppn[] =
	NEMESIS__INTEL_XEON_X5650_12__MLX_CX_QDR__12PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Gordon Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_QDR_16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5_2670_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Yellowstone Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2670_16__MLX_CX_FDR__16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
      /*Stampede Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*Trestles Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 4;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 3;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 32;
      mv2_size_allgather_indexed_tuning_table[2] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_32ppn[] =
	NEMESIS__AMD_OPTERON_6136_32__MLX_CX_QDR__32PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_32ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
				  MV2_ARCH_INTEL_XEON_E5630_8, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
      /*RI Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
        = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
                      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
                               * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							    mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));

      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
        NEMESIS__RI__1PPN;
	table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;

      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 2;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
        NEMESIS__RI__2PPN;
	table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;

      mv2_allgather_indexed_table_ppn_conf[2] = 8;
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_8ppn[] =
        NEMESIS__RI__8PPN;
	table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_8ppn;

      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
        agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
        MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
                  (sizeof(mv2_allgather_indexed_tuning_table)
                   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
        mv2_allgather_indexed_thresholds_table[i] =
          mv2_allgather_indexed_thresholds_table[i - 1]
          + mv2_size_allgather_indexed_tuning_table[i - 1];
        MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
                    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else {
      /*Stampede Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
#endif
#else /* !CHANNEL_PSM */
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_QLGIC_QIB) && !heterogeneity) {
      /*Sierra Table*/
      mv2_allgather_indexed_num_ppn_conf = 2;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							    mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 12;
      mv2_size_allgather_indexed_tuning_table[1] = 6;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_12ppn[] =
	PSM__INTEL_XEON_X5650_12__MV2_HCA_QLGIC_QIB__12PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_12ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /*Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (allgather, 6)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* Bebop/Jade/Opal Table */
      MV2_COLL_TUNING_START_TABLE  (allgather, 5)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 36, 5, PSM__INTEL_XEON_E5_2695_V4_2S_36_INTEL_HFI_100__36PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_XEON_PHI_7250, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-KNL Table */
      MV2_COLL_TUNING_START_TABLE  (allgather, 6)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 6, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 32, 5, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__32PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 64, 1, PSM__INTEL_XEON_PHI_7250_68_INTEL_HFI_100__64PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
			     MV2_ARCH_INTEL_PLATINUM_8170_2S_52, MV2_HCA_INTEL_HFI1) && !heterogeneity) {
      /* TACC-Skylake Table */
      MV2_COLL_TUNING_START_TABLE  (allgather, 9)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 24, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__24PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 26, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__26PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 48, 5, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__48PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 52, 4, PSM__INTEL_PLATINUM_8170_2S_52_INTEL_HFI_100__52PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
    else {
      /*default psm table: Bridges Table*/
      MV2_COLL_TUNING_START_TABLE  (allgather, 6)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 1,  4, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__1PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 2,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__2PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 4,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__4PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 8,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__8PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 16,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__16PPN)
      MV2_COLL_TUNING_ADD_CONF     (allgather, 28,  5, PSM__INTEL_XEON_E5_2695_V3_2S_28_INTEL_HFI_100__28PPN)
      MV2_COLL_TUNING_FINISH_TABLE (allgather)
    }
#endif /* !CHANNEL_PSM */
    {
      /*Stampede Table*/
      mv2_allgather_indexed_num_ppn_conf = 3;
      mv2_allgather_indexed_thresholds_table
	= MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
		      * mv2_allgather_indexed_num_ppn_conf);
      table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_indexed_tuning_table *)
			       * mv2_allgather_indexed_num_ppn_conf);
      mv2_size_allgather_indexed_tuning_table = MPIU_Malloc(sizeof(int) *
							mv2_allgather_indexed_num_ppn_conf);
      mv2_allgather_indexed_table_ppn_conf = MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
      
      mv2_allgather_indexed_table_ppn_conf[0] = 1;
      mv2_size_allgather_indexed_tuning_table[0] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_1ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__1PPN;
      table_ptrs[0] = mv2_tmp_allgather_indexed_thresholds_table_1ppn;
      
      mv2_allgather_indexed_table_ppn_conf[1] = 2;
      mv2_size_allgather_indexed_tuning_table[1] = 5;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_2ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__2PPN;
      table_ptrs[1] = mv2_tmp_allgather_indexed_thresholds_table_2ppn;
      
      mv2_allgather_indexed_table_ppn_conf[2] = 16;
      mv2_size_allgather_indexed_tuning_table[2] = 7;
      mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table_16ppn[] =
	NEMESIS__INTEL_XEON_E5_2680_16__MLX_CX_FDR__16PPN;
      table_ptrs[2] = mv2_tmp_allgather_indexed_thresholds_table_16ppn;
      
      agg_table_sum = 0;
      for (i = 0; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	agg_table_sum += mv2_size_allgather_indexed_tuning_table[i];
      }
      mv2_allgather_indexed_thresholds_table[0] =
	MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_indexed_tuning_table));
      MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], table_ptrs[0],
		  (sizeof(mv2_allgather_indexed_tuning_table)
		   * mv2_size_allgather_indexed_tuning_table[0]));
      for (i = 1; i < mv2_allgather_indexed_num_ppn_conf; i++) {
	mv2_allgather_indexed_thresholds_table[i] =
	  mv2_allgather_indexed_thresholds_table[i - 1]
	  + mv2_size_allgather_indexed_tuning_table[i - 1];
	MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[i], table_ptrs[i],
		    (sizeof(mv2_allgather_indexed_tuning_table)
		     * mv2_size_allgather_indexed_tuning_table[i]));
      }
      MPIU_Free(table_ptrs);
      return 0;
    }
  } 
  else {
  mv2_allgather_tuning_table **table_ptrs = NULL;
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 1;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
              * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                     * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 12;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_12ppn[] = {
            {
                12,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                24,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                48,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                96,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                192,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                384,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_12ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 3;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                  * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf 
            = MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 1;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_1ppn[] = {
            {
                2,
                {0},
                1,
                {
                    {0, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                4,
                {0,0},
                2,
                {
                    {0, 262144, &MPIR_Allgather_RD_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                8,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                16,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_1ppn;
        mv2_allgather_table_ppn_conf[1] = 2;
        mv2_size_allgather_tuning_table[1] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_2ppn[] = {
            {
                4,
                {0,0},
                2,
                {
                    {0, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                8,
                {0,1,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                16,
                {0,1,0},
                2,
                {
                    {0, 16384, &MPIR_Allgather_RD_MV2},
                    {16384, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {1,1,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1,1,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {1,1,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[1] = mv2_tmp_allgather_thresholds_table_2ppn;
        mv2_allgather_table_ppn_conf[2] = 16;
        mv2_size_allgather_tuning_table[2] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_16ppn[] = {
            {
                16,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[2] = mv2_tmp_allgather_thresholds_table_16ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
            (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 1;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = 
             MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 32;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_32ppn[] = {
            {
                32,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1, 0, 0},
                3,
                {
                    {0, 8, &MPIR_Allgather_RD_MV2},
                    {8, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {1, 0, 0},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {1, 0, 0},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {1, 0, 1},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 2048, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {2048, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                1024,
                {1, 0, 1},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 2048, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {2048, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_32ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    } else {
        mv2_allgather_num_ppn_conf = 3;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = 
            MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 1;
        mv2_size_allgather_tuning_table[0] = 3;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_1ppn[] = {
            { 
                16,
                {0,0},
                2,
                {
                    {0, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_1ppn;
        mv2_allgather_table_ppn_conf[1] = 2;
        mv2_size_allgather_tuning_table[1] = 4;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_2ppn[] = {
            {
                16,
                {0, 0, 0, 1, 0},
                5,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, 65536, &MPIR_Allgather_Ring_MV2},
                    {65536, 262144, &MPIR_Allgather_Ring_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {1, 0, 0, 1, 0},
                5,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 262144, &MPIR_Allgather_RD_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1, 0, 0, 0},
                4,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0, 0, 1, 0},
                4,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[1] = mv2_tmp_allgather_thresholds_table_2ppn;
        mv2_allgather_table_ppn_conf[2] = 8;
        mv2_size_allgather_tuning_table[2] = 7;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_8ppn[] = {
            {
                8,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            { 
                16,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[2] = mv2_tmp_allgather_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    }
#elif defined (CHANNEL_NEMESIS_IB)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 1;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
              * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                     * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 12;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_12ppn[] = {
            {
                12,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                24,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                48,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                96,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                192,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                384,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_12ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 3;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                  * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf 
            = MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 1;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_1ppn[] = {
            {
                2,
                {0},
                1,
                {
                    {0, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                4,
                {0,0},
                2,
                {
                    {0, 262144, &MPIR_Allgather_RD_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                8,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                16,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_1ppn;
        mv2_allgather_table_ppn_conf[1] = 2;
        mv2_size_allgather_tuning_table[1] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_2ppn[] = {
            {
                4,
                {0,0},
                2,
                {
                    {0, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                8,
                {0,1,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                16,
                {0,1,0},
                2,
                {
                    {0, 16384, &MPIR_Allgather_RD_MV2},
                    {16384, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {1,1,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1,1,0},
                2,
                {
                    {0, 32768, &MPIR_Allgather_RD_MV2},
                    {32768, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {1,1,0},
                2,
                {
                    {0, 65536, &MPIR_Allgather_RD_MV2},
                    {65536, 524288, &MPIR_Allgather_Ring_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[1] = mv2_tmp_allgather_thresholds_table_2ppn;
        mv2_allgather_table_ppn_conf[2] = 16;
        mv2_size_allgather_tuning_table[2] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_16ppn[] = {
            {
                16,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[2] = mv2_tmp_allgather_thresholds_table_16ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
            (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity) {
        mv2_allgather_num_ppn_conf = 1;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = 
             MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 32;
        mv2_size_allgather_tuning_table[0] = 6;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_32ppn[] = {
            {
                32,
                {0,0},
                2,
                {
                    {0, 512, &MPIR_Allgather_RD_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1, 0, 0},
                3,
                {
                    {0, 8, &MPIR_Allgather_RD_MV2},
                    {8, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {1, 0, 0},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 512, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {512, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {1, 0, 0},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {1, 0, 1},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 2048, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {2048, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                1024,
                {1, 0, 1},
                3,
                {
                    {0, 16, &MPIR_Allgather_RD_MV2},
                    {16, 2048, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {2048, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_32ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
    return 0;
    } else {
        mv2_allgather_num_ppn_conf = 3;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = 
            MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 1;
        mv2_size_allgather_tuning_table[0] = 3;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_1ppn[] = {
            { 
                16,
                {0,0},
                2,
                {
                    {0, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_1ppn;
        mv2_allgather_table_ppn_conf[1] = 2;
        mv2_size_allgather_tuning_table[1] = 4;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_2ppn[] = {
            {
                16,
                {0, 0, 0, 1, 0},
                5,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, 65536, &MPIR_Allgather_Ring_MV2},
                    {65536, 262144, &MPIR_Allgather_Ring_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {1, 0, 0, 1, 0},
                5,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 262144, &MPIR_Allgather_RD_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1, 0, 0, 0},
                4,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0, 0, 1, 0},
                4,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[1] = mv2_tmp_allgather_thresholds_table_2ppn;
        mv2_allgather_table_ppn_conf[2] = 8;
        mv2_size_allgather_tuning_table[2] = 7;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_8ppn[] = {
            {
                8,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            { 
                16,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[2] = mv2_tmp_allgather_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    }
#endif
#endif /* !CHANNEL_PSM */
    {
        mv2_allgather_num_ppn_conf = 3;
        mv2_allgather_thresholds_table
            = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        table_ptrs = MPIU_Malloc(sizeof(mv2_allgather_tuning_table *)
                                 * mv2_allgather_num_ppn_conf);
        mv2_size_allgather_tuning_table = MPIU_Malloc(sizeof(int) *
                                                      mv2_allgather_num_ppn_conf);
        mv2_allgather_table_ppn_conf = 
            MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        mv2_allgather_table_ppn_conf[0] = 1;
        mv2_size_allgather_tuning_table[0] = 3;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_1ppn[] = {
            { 
                16,
                {0,0},
                2,
                {
                    {0, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 131072, &MPIR_Allgather_RD_MV2},
                    {131072, -1, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                },
            },
        };
        table_ptrs[0] = mv2_tmp_allgather_thresholds_table_1ppn;
        mv2_allgather_table_ppn_conf[1] = 2;
        mv2_size_allgather_tuning_table[1] = 4;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_2ppn[] = {
            {
                16,
                {0, 0, 0, 1, 0},
                5,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, 65536, &MPIR_Allgather_Ring_MV2},
                    {65536, 262144, &MPIR_Allgather_Ring_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {1, 0, 0, 1, 0},
                5,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 262144, &MPIR_Allgather_RD_MV2},
                    {262144, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {1, 0, 0, 0},
                4,
                {
                    {0, 4, &MPIR_Allgather_RD_MV2},
                    {4, 512, &MPIR_Allgather_RD_MV2},
                    {512, 8192, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {8192, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0, 0, 1, 0},
                4,
                {
                    {0, 128, &MPIR_Allgather_RD_MV2},
                    {128, 4096, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {4096, 524288, &MPIR_Allgather_RD_MV2},
                    {524288, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
        };
        table_ptrs[1] = mv2_tmp_allgather_thresholds_table_2ppn;
        mv2_allgather_table_ppn_conf[2] = 8;
        mv2_size_allgather_tuning_table[2] = 7;
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table_8ppn[] = {
            {
                8,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            { 
                16,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                32,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                64,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                128,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                256,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },
            {
                512,
                {0,0},
                2,
                {
                    {0, 1024, &MPIR_Allgather_RD_Allgather_Comm_MV2},
                    {1024, -1, &MPIR_Allgather_Ring_MV2},
                },
            },

        };
        table_ptrs[2] = mv2_tmp_allgather_thresholds_table_8ppn;
        agg_table_sum = 0;
        for (i = 0; i < mv2_allgather_num_ppn_conf; i++) {
            agg_table_sum += mv2_size_allgather_tuning_table[i];
        }
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(agg_table_sum * sizeof (mv2_allgather_tuning_table));
        MPIU_Memcpy(mv2_allgather_thresholds_table[0], table_ptrs[0],
                    (sizeof(mv2_allgather_tuning_table)
                     * mv2_size_allgather_tuning_table[0]));
        for (i = 1; i < mv2_allgather_num_ppn_conf; i++) {
            mv2_allgather_thresholds_table[i] =
            mv2_allgather_thresholds_table[i - 1]
            + mv2_size_allgather_tuning_table[i - 1];
            MPIU_Memcpy(mv2_allgather_thresholds_table[i], table_ptrs[i],
                      (sizeof(mv2_allgather_tuning_table)
                       * mv2_size_allgather_tuning_table[i]));
        }
        MPIU_Free(table_ptrs);
	return 0;
    }
  }
    return 0;
}

void MV2_cleanup_allgather_tuning_table()
{
  if (mv2_use_indexed_tuning || mv2_use_indexed_allgather_tuning) {
    MPIU_Free(mv2_allgather_indexed_thresholds_table[0]);
    MPIU_Free(mv2_allgather_indexed_table_ppn_conf);
    MPIU_Free(mv2_size_allgather_indexed_tuning_table);
    if (mv2_allgather_indexed_thresholds_table != NULL) {
      MPIU_Free(mv2_allgather_indexed_thresholds_table);
    }
  }
  else {
    MPIU_Free(mv2_allgather_thresholds_table[0]);
    MPIU_Free(mv2_allgather_table_ppn_conf);
    MPIU_Free(mv2_size_allgather_tuning_table);
    if (mv2_allgather_thresholds_table != NULL) {
      MPIU_Free(mv2_allgather_thresholds_table);
    }
  }
}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}


int MV2_internode_Allgather_is_define(char *mv2_user_allgather_inter)
{
    int i = 0;
    int nb_element = count_sep(mv2_user_allgather_inter) + 1;

    if (mv2_use_indexed_tuning || mv2_use_indexed_allgather_tuning) {
        mv2_allgather_indexed_tuning_table mv2_tmp_allgather_indexed_thresholds_table[1];
        mv2_allgather_indexed_num_ppn_conf = 1;
        if (mv2_size_allgather_indexed_tuning_table == NULL) {
            mv2_size_allgather_indexed_tuning_table =
                MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
        }
        mv2_size_allgather_indexed_tuning_table[0] = 1;

        if (mv2_allgather_indexed_table_ppn_conf == NULL) {
            mv2_allgather_indexed_table_ppn_conf =
                MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf * sizeof(int));
        }
        /* -1 indicates user defined algorithm */
        mv2_allgather_indexed_table_ppn_conf[0] = -1;

        /* If one allgather tuning table is already defined */
        if (mv2_allgather_indexed_thresholds_table != NULL) {
            if (mv2_allgather_indexed_thresholds_table[0] != NULL) {
                MPIU_Free(mv2_allgather_indexed_thresholds_table[0]);
            }
            MPIU_Free(mv2_allgather_indexed_thresholds_table);
        }

        /* We realloc the space for the new allgather_indexed tuning table */
        mv2_allgather_indexed_thresholds_table =
            MPIU_Malloc(mv2_allgather_indexed_num_ppn_conf *
                    sizeof(mv2_allgather_indexed_tuning_table *));
        mv2_allgather_indexed_thresholds_table[0] =
            MPIU_Malloc(mv2_size_allgather_indexed_tuning_table[0] *
                    sizeof(mv2_allgather_indexed_tuning_table));

        if (nb_element == 1) {
            mv2_tmp_allgather_indexed_thresholds_table[0].numproc = 1;
            mv2_tmp_allgather_indexed_thresholds_table[0].size_inter_table = 1;
            mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].msg_sz = 1;

            switch (atoi(mv2_user_allgather_inter)) {
                case ALLGATHER_RD_ALLGATHER_COMM:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_Allgather_Comm_MV2;
                    break;
                case ALLGATHER_RD:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_MV2;
                    break;
                case ALLGATHER_BRUCK:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Bruck_MV2;
                    break;
                case ALLGATHER_RING:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Ring_MV2;
                    break;
                case ALLGATHER_DIRECT:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Direct_MV2;
                    break;
                case ALLGATHER_DIRECTSPREAD:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_DirectSpread_MV2;
                    break;
                case ALLGATHER_GATHER_BCAST:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_gather_bcast_MV2;
                    break;
                case ALLGATHER_2LVL_NONBLOCKED:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_nonblocked_MV2;
                    break;
                case ALLGATHER_2LVL_RING_NONBLOCKED:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Ring_nonblocked_MV2;
                    break;
                case ALLGATHER_2LVL_DIRECT:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Direct_MV2;
                    break;
                case ALLGATHER_2LVL_RING:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Ring_MV2;
                    break;
                case ALLGATHER_2LVL_MULTILEADER_RING:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Multileader_Ring_MV2;
                    break;
                
                case ALLGATHER_2LVL_MULTILEADER_RD:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Multileader_RD_MV2;
                    break;
                case ALLGATHER_2LVL_SHMEM:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_SharedMem_Allgather_MV2;
                    break;
                case ALLGATHER_ENC_RDB:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Encrypted_RDB_MV2;
                    break; 
                case ALLGATHER_NP_RDB:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_NaivePlus_RDB_MV2;
                    break; 
                case ALLGATHER_2LVL_ENC_RDB:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Encrypted_RDB_MV2;
                    break; 
                case ALLGATHER_2LVL_SHMEM_CONCURRENT_ENCRYPTION:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather_MV2;
                    break;
                case CONCURRENT_ALLGATHER:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Concurrent_Allgather_MV2;
                    break;
                case ALLGATHER_CONCURRENT_MULTILEADER_SHMEM:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Concurrent_Multileader_SharedMem_Allgather_MV2;
                    break;
                    
                    
                default:
                    mv2_tmp_allgather_indexed_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_MV2;
            }
        }
        MPIU_Memcpy(mv2_allgather_indexed_thresholds_table[0], mv2_tmp_allgather_indexed_thresholds_table, sizeof
                (mv2_allgather_indexed_tuning_table));
    } else {
        mv2_allgather_tuning_table mv2_tmp_allgather_thresholds_table[1];
        mv2_allgather_num_ppn_conf = 1;
        if (mv2_size_allgather_tuning_table == NULL) {
            mv2_size_allgather_tuning_table =
                MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        }
        mv2_size_allgather_tuning_table[0] = 1;

        if (mv2_allgather_table_ppn_conf == NULL) {
            mv2_allgather_table_ppn_conf =
                MPIU_Malloc(mv2_allgather_num_ppn_conf * sizeof(int));
        }
        /* -1 indicates user defined algorithm */
        mv2_allgather_table_ppn_conf[0] = -1;

        /* If one allgather tuning table is already defined */
        if (mv2_allgather_thresholds_table != NULL) {
            MPIU_Free(mv2_allgather_thresholds_table);
        }

        /* We realloc the space for the new allgather tuning table */
        mv2_allgather_thresholds_table =
            MPIU_Malloc(mv2_allgather_num_ppn_conf *
                    sizeof(mv2_allgather_tuning_table *));
        mv2_allgather_thresholds_table[0] =
            MPIU_Malloc(mv2_size_allgather_tuning_table[0] *
                    sizeof(mv2_allgather_tuning_table));

        if (nb_element == 1) {

            mv2_tmp_allgather_thresholds_table[0].numproc = 1;
            mv2_tmp_allgather_thresholds_table[0].size_inter_table = 1;
            mv2_tmp_allgather_thresholds_table[0].inter_leader[0].min = 0;
            mv2_tmp_allgather_thresholds_table[0].inter_leader[0].max = -1;
            mv2_tmp_allgather_thresholds_table[0].two_level[0] = mv2_user_allgather_two_level;

            switch (atoi(mv2_user_allgather_inter)) {
                case ALLGATHER_RD_ALLGATHER_COMM:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_Allgather_Comm_MV2;
                    break;
                case ALLGATHER_RD:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_MV2;
                    break;
                case ALLGATHER_BRUCK:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Bruck_MV2;
                    break;
                case ALLGATHER_RING:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Ring_MV2;
                    break;
                case ALLGATHER_DIRECT:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Direct_MV2;
                    break;
                case ALLGATHER_DIRECTSPREAD:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_DirectSpread_MV2;
                    break;
                case ALLGATHER_GATHER_BCAST:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_gather_bcast_MV2;
                    break;
                case ALLGATHER_2LVL_NONBLOCKED:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_nonblocked_MV2;
                    break;
                case ALLGATHER_2LVL_RING_NONBLOCKED:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Ring_nonblocked_MV2;
                    break;
                case ALLGATHER_2LVL_DIRECT:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Direct_MV2;
                    break;
                case ALLGATHER_2LVL_RING:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Ring_MV2;
                    break;
                case ALLGATHER_2LVL_MULTILEADER_RING:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Multileader_Ring_MV2;
                    break;
                case ALLGATHER_2LVL_MULTILEADER_RD:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Multileader_RD_MV2;
                    break;
                case ALLGATHER_2LVL_SHMEM:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_SharedMem_Allgather_MV2;
                    break;
                case ALLGATHER_ENC_RDB:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_Encrypted_RDB_MV2;
                    break;
                case ALLGATHER_NP_RDB:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_NaivePlus_RDB_MV2;
                    break;
                case ALLGATHER_2LVL_ENC_RDB:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Allgather_Encrypted_RDB_MV2;
                    break;
                case ALLGATHER_2LVL_SHMEM_CONCURRENT_ENCRYPTION:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather_MV2;
                    break;

                case CONCURRENT_ALLGATHER:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Concurrent_Allgather_MV2;
                    break;
                case ALLGATHER_CONCURRENT_MULTILEADER_SHMEM:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_2lvl_Concurrent_Multileader_SharedMem_Allgather_MV2;
                    break;

                default:
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                        &MPIR_Allgather_RD_MV2;
            }

        } else {
            char *dup, *p, *save_p;
            regmatch_t match[NMATCH];
            regex_t preg;
            const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

            if (!(dup = MPIU_Strdup(mv2_user_allgather_inter))) {
                fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_allgather_inter);
                return -1;
            }

            if (regcomp(&preg, regexp, REG_EXTENDED)) {
                fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_allgather_inter);
                MPIU_Free(dup);
                return -1;
            }

            mv2_tmp_allgather_thresholds_table[0].numproc = 1;
            mv2_tmp_allgather_thresholds_table[0].size_inter_table = nb_element;

            i = 0;
            for (p = strtok_r(dup, ",", &save_p); p; p = strtok_r(NULL, ",", &save_p)) {
                if (regexec(&preg, p, NMATCH, match, 0)) {
                    fprintf(stderr, "failed to match on `%s'\n", p);
                    regfree(&preg);
                    MPIU_Free(dup);
                    return -1;
                }
                /* given () start at 1 */
                switch (atoi(p + match[1].rm_so)) {
                    case ALLGATHER_RD_ALLGATHER_COMM:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[i].MV2_pt_Allgather_function =
                            &MPIR_Allgather_RD_Allgather_Comm_MV2;
                        break;
                    case ALLGATHER_RD:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[i].MV2_pt_Allgather_function =
                            &MPIR_Allgather_RD_MV2;
                        break;
                    case ALLGATHER_BRUCK:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[i].MV2_pt_Allgather_function =
                            &MPIR_Allgather_Bruck_MV2;
                        break;
                    case ALLGATHER_RING:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[i].MV2_pt_Allgather_function =
                            &MPIR_Allgather_Ring_MV2;
                        break;
                    case ALLGATHER_DIRECT:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Allgather_Direct_MV2;
                        break;
                    case ALLGATHER_DIRECTSPREAD:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Allgather_DirectSpread_MV2;
                        break;
                    case ALLGATHER_GATHER_BCAST:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Allgather_gather_bcast_MV2;
                        break;
                    case ALLGATHER_2LVL_NONBLOCKED:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_nonblocked_MV2;
                        break;
                    case ALLGATHER_2LVL_RING_NONBLOCKED:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Ring_nonblocked_MV2;
                        break;
                    case ALLGATHER_2LVL_DIRECT:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Direct_MV2;
                        break;
                    case ALLGATHER_2LVL_RING:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Ring_MV2;
                        break;
                    case ALLGATHER_2LVL_MULTILEADER_RING:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Multileader_Ring_MV2;
                        break;
                    case ALLGATHER_2LVL_MULTILEADER_RD:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Multileader_RD_MV2;
                        break;
                    case ALLGATHER_2LVL_SHMEM:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_SharedMem_Allgather_MV2;
                        break;
                    case ALLGATHER_ENC_RDB:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Allgather_Encrypted_RDB_MV2;
                        break;
                    
                    case ALLGATHER_NP_RDB:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Allgather_NaivePlus_RDB_MV2;
                        break;

                    case ALLGATHER_2LVL_ENC_RDB:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Allgather_Encrypted_RDB_MV2;
                        break;
                    case ALLGATHER_2LVL_SHMEM_CONCURRENT_ENCRYPTION:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_SharedMem_Concurrent_Encryption_Allgather_MV2;
                        break;
                    case CONCURRENT_ALLGATHER:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_Concurrent_Allgather_MV2;
                        break;
                    case ALLGATHER_CONCURRENT_MULTILEADER_SHMEM:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[0].MV2_pt_Allgather_function =
                            &MPIR_2lvl_Concurrent_Multileader_SharedMem_Allgather_MV2;
                        break;

                    default:
                        mv2_tmp_allgather_thresholds_table[0].inter_leader[i].MV2_pt_Allgather_function =
                            &MPIR_Allgather_RD_MV2;
                }

                mv2_tmp_allgather_thresholds_table[0].inter_leader[i].min = atoi(p + match[2].rm_so);
                if (p[match[3].rm_so] == '+') {
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[i].max = -1;
                } else {
                    mv2_tmp_allgather_thresholds_table[0].inter_leader[i].max =
                        atoi(p + match[3].rm_so);
                }
                i++;
            }
            MPIU_Free(dup);
            regfree(&preg);
        }

        MPIU_Memcpy(mv2_allgather_thresholds_table[0], mv2_tmp_allgather_thresholds_table, sizeof
                (mv2_allgather_tuning_table));
    }
    return 0;
}
