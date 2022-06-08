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
#include "alltoallv_tuning.h"
#include "tuning/alltoallv_arch_tuning.h"
#include "mv2_arch_hca_detect.h"

/* array used to tune alltoallv */

int *mv2_alltoallv_table_ppn_conf = NULL;
int mv2_alltoallv_num_ppn_conf = 1;
int *mv2_size_alltoallv_tuning_table = NULL;
mv2_alltoallv_tuning_table **mv2_alltoallv_thresholds_table = NULL;

int *mv2_alltoallv_indexed_table_ppn_conf = NULL;
int mv2_alltoallv_indexed_num_ppn_conf = 1;
int *mv2_size_alltoallv_indexed_tuning_table = NULL;
mv2_alltoallv_indexed_tuning_table **mv2_alltoallv_indexed_thresholds_table = NULL;

int MV2_set_alltoallv_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{

    int agg_table_sum = 0;
    int i;

    /* Sample table */
    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoallv_tuning) {
        mv2_alltoallv_indexed_tuning_table **table_ptrs = NULL;
        if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
                    MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28, MV2_HCA_MLX_CX_EDR) && !heterogeneity) {
            MV2_COLL_TUNING_START_TABLE  (alltoallv, 1)
                MV2_COLL_TUNING_ADD_CONF(alltoallv, 4, 2, test_table) // 2 node 4 and 8 ppn test table for RI2
                MV2_COLL_TUNING_FINISH_TABLE (alltoallv)               
        } else {
            MV2_COLL_TUNING_START_TABLE  (alltoallv, 1)
                MV2_COLL_TUNING_ADD_CONF(alltoallv, 4, 2, test_table) // 2 node 4 and 8 ppn test table for RI2
                MV2_COLL_TUNING_FINISH_TABLE (alltoallv)
        }     
    }

    return 0;
}

void MV2_cleanup_alltoallv_tuning_table() 
{
    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoallv_tuning) {
        MPIU_Free(mv2_alltoallv_indexed_thresholds_table[0]);
        MPIU_Free(mv2_alltoallv_indexed_table_ppn_conf);
        MPIU_Free(mv2_size_alltoallv_indexed_tuning_table);
        if (mv2_alltoallv_indexed_thresholds_table != NULL) {
            MPIU_Free(mv2_alltoallv_indexed_thresholds_table);
        }
    } else {
        MPIU_Free(mv2_alltoallv_thresholds_table[0]);
        MPIU_Free(mv2_alltoallv_table_ppn_conf);
        MPIU_Free(mv2_size_alltoallv_tuning_table);
        if (mv2_alltoallv_thresholds_table != NULL) {
            MPIU_Free(mv2_alltoallv_thresholds_table);
        }
    }
}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}

int MV2_Alltoallv_is_define(char *mv2_user_alltoallv)
{
    int i = 0;
    int nb_element = count_sep(mv2_user_alltoallv) + 1;
    if (mv2_use_indexed_tuning || mv2_use_indexed_alltoallv_tuning) {
        mv2_alltoallv_indexed_num_ppn_conf = 1;

        if (mv2_size_alltoallv_indexed_tuning_table == NULL) {
            mv2_size_alltoallv_indexed_tuning_table =
                MPIU_Malloc(mv2_alltoallv_indexed_num_ppn_conf * sizeof(int));
        }
        mv2_size_alltoallv_indexed_tuning_table[0] = 1;

        if (mv2_alltoallv_indexed_table_ppn_conf == NULL) {
            mv2_alltoallv_indexed_table_ppn_conf =
                MPIU_Malloc(mv2_alltoallv_indexed_num_ppn_conf * sizeof(int));
        }
        mv2_alltoallv_indexed_table_ppn_conf[0] = -1;

        mv2_alltoallv_indexed_tuning_table mv2_tmp_alltoallv_indexed_thresholds_table[1];

        /* If one alltoall_indexed tuning table is already defined */
        if (mv2_alltoallv_indexed_thresholds_table != NULL) {
            if (mv2_alltoallv_indexed_thresholds_table[0] != NULL) {
                MPIU_Free(mv2_alltoallv_indexed_thresholds_table[0]);
            }
            MPIU_Free(mv2_alltoallv_indexed_thresholds_table);
        }

        /* We realloc the space for the new alltoallv_indexed tuning table */
        mv2_alltoallv_indexed_thresholds_table =
            MPIU_Malloc(mv2_alltoallv_indexed_num_ppn_conf *
                    sizeof(mv2_alltoallv_indexed_tuning_table *));
        mv2_alltoallv_indexed_thresholds_table[0] =
            MPIU_Malloc(mv2_size_alltoallv_indexed_tuning_table[0] *
                    sizeof(mv2_alltoallv_indexed_tuning_table));

        if (nb_element == 1) {
            mv2_tmp_alltoallv_indexed_thresholds_table[0].numproc = 1;
            mv2_tmp_alltoallv_indexed_thresholds_table[0].size_table = 1;
            mv2_tmp_alltoallv_indexed_thresholds_table[0].algo_table[0].msg_sz = 1;
            mv2_tmp_alltoallv_indexed_thresholds_table[0].in_place_algo_table[0] = 0;
            switch (atoi(mv2_user_alltoallv)) {
                case ALLTOALLV_INTRA_SCATTER_MV2:
                    mv2_tmp_alltoallv_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_scatter_MV2;
                    break;
                case ALLTOALLV_INTRA_MV2:
                    mv2_tmp_alltoallv_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_MV2;
                    break;
                default:
                    mv2_tmp_alltoallv_indexed_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_scatter_MV2;
            }
        }
        MPIU_Memcpy(mv2_alltoallv_indexed_thresholds_table[0], mv2_tmp_alltoallv_indexed_thresholds_table, sizeof
                (mv2_alltoallv_indexed_tuning_table));
    } else {
        mv2_alltoallv_num_ppn_conf = 1;

        if (mv2_size_alltoallv_tuning_table == NULL) {
            mv2_size_alltoallv_tuning_table =
                MPIU_Malloc(mv2_alltoallv_num_ppn_conf * sizeof(int));
        }
        mv2_size_alltoallv_tuning_table[0] = 1;

        if (mv2_alltoallv_table_ppn_conf == NULL) {
            mv2_alltoallv_table_ppn_conf =
                MPIU_Malloc(mv2_alltoallv_num_ppn_conf * sizeof(int));
        }
        mv2_alltoallv_table_ppn_conf[0] = -1;

        mv2_alltoallv_tuning_table mv2_tmp_alltoallv_thresholds_table[1];

        /* If one alltoallv tuning table is already defined */
        if (mv2_alltoallv_thresholds_table != NULL) {
            MPIU_Free(mv2_alltoallv_thresholds_table);
        }

        /* We realloc the space for the new alltoallv tuning table */
        mv2_alltoallv_thresholds_table =
            MPIU_Malloc(mv2_alltoallv_num_ppn_conf *
                    sizeof(mv2_alltoallv_tuning_table *));
        mv2_alltoallv_thresholds_table[0] =
            MPIU_Malloc(mv2_size_alltoallv_tuning_table[0] *
                    sizeof(mv2_alltoallv_tuning_table));

        if (nb_element == 1) {
            mv2_tmp_alltoallv_thresholds_table[0].numproc = 1;
            mv2_tmp_alltoallv_thresholds_table[0].size_table = 1;
            mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].min = 0;
            mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].max = -1;
            switch (atoi(mv2_user_alltoallv)) {
                case ALLTOALLV_INTRA_SCATTER_MV2:
                    mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_scatter_MV2;
                    break;
                case ALLTOALLV_INTRA_MV2:
                    mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_MV2;
                    break;
                default:
                    mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                        &MPIR_Alltoallv_intra_scatter_MV2;
            }
        } else {
            char *dup, *p, *save_p;
            regmatch_t match[NMATCH];
            regex_t preg;
            const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

            if (!(dup = MPIU_Strdup(mv2_user_alltoallv))) {
                fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_alltoallv);
                return 1;
            }

            if (regcomp(&preg, regexp, REG_EXTENDED)) {
                fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_alltoallv);
                MPIU_Free(dup);
                return 2;
            }

            mv2_tmp_alltoallv_thresholds_table[0].numproc = 1;
            mv2_tmp_alltoallv_thresholds_table[0].size_table = nb_element;
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
                    case ALLTOALLV_INTRA_SCATTER_MV2:
                        mv2_tmp_alltoallv_thresholds_table[0].algo_table[0].MV2_pt_Alltoallv_function =
                            &MPIR_Alltoallv_intra_scatter_MV2;
                        break;
                    case ALLTOALLV_INTRA_MV2:
                        mv2_tmp_alltoallv_thresholds_table[0].algo_table[i].MV2_pt_Alltoallv_function =
                            &MPIR_Alltoallv_intra_MV2;
                        break;
                    default:
                        mv2_tmp_alltoallv_thresholds_table[0].algo_table[i].MV2_pt_Alltoallv_function =
                            &MPIR_Alltoallv_intra_scatter_MV2;
                }
                i++;
            }
            MPIU_Free(dup);
            regfree(&preg);
        }
        MPIU_Memcpy(mv2_alltoallv_thresholds_table[0], mv2_tmp_alltoallv_thresholds_table, sizeof
                (mv2_alltoallv_tuning_table));
    }
    return 0;
}
