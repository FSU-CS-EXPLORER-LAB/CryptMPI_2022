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
#include "red_scat_tuning.h"
#include "mv2_arch_hca_detect.h"

enum {
    RED_SCAT_BASIC = 1,
    RED_SCAT_REC_HALF,
    RED_SCAT_PAIRWISE,
    RED_SCAT_RING,
    RED_SCAT_RING_2LVL,
};

int mv2_size_red_scat_tuning_table = 0;
mv2_red_scat_tuning_table *mv2_red_scat_thresholds_table = NULL;

int MV2_set_red_scat_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
#ifndef CHANNEL_PSM
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_red_scat_tuning_table = 6;
        mv2_red_scat_thresholds_table = MPIU_Malloc(mv2_size_red_scat_tuning_table *
                                                  sizeof (mv2_red_scat_tuning_table));
        mv2_red_scat_tuning_table mv2_tmp_red_scat_thresholds_table[] = {
            {
                12,
                4,
                {
                    {0, 256, &MPIR_Reduce_Scatter_Basic_MV2},
                    {256, 8192, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {8192, 65536, &MPIR_Reduce_scatter_Pair_Wise_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                24,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                48,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 131072, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {131072, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                96,
                3,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 262144, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {262144, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                192,
                3,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                384,
                3,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
        }; 
        MPIU_Memcpy(mv2_red_scat_thresholds_table, mv2_tmp_red_scat_thresholds_table,
                  mv2_size_red_scat_tuning_table * sizeof (mv2_red_scat_tuning_table));
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
        mv2_size_red_scat_tuning_table = 6;
        mv2_red_scat_thresholds_table = MPIU_Malloc(mv2_size_red_scat_tuning_table *
                                                  sizeof (mv2_red_scat_tuning_table));
        mv2_red_scat_tuning_table mv2_tmp_red_scat_thresholds_table[] = {
            {
                16,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                32,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 131072, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {131072, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                64,
                3,
                {
                    {0, 1024, &MPIR_Reduce_Scatter_Basic_MV2},
                    {1024, 262144, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {262144, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                128,
                3,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                256,
                3,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                512,
                3,
                {
                    {0, 256, &MPIR_Reduce_Scatter_Basic_MV2},
                    {256, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },

        }; 
        MPIU_Memcpy(mv2_red_scat_thresholds_table, mv2_tmp_red_scat_thresholds_table,
                  mv2_size_red_scat_tuning_table * sizeof (mv2_red_scat_tuning_table));
    }  else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_red_scat_tuning_table = 6;
        mv2_red_scat_thresholds_table = MPIU_Malloc(mv2_size_red_scat_tuning_table *
                                                  sizeof (mv2_red_scat_tuning_table));
        mv2_red_scat_tuning_table mv2_tmp_red_scat_thresholds_table[] = {
            {
                32,
                3,
                {
                    {0, 8, &MPIR_Reduce_Scatter_Basic_MV2},
                    {8, 32768, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {32768, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                64,
                2,
                {
                    {0, 65536, &MPIR_Reduce_Scatter_Basic_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                128,
                2,
                {
                    {0, 65536, &MPIR_Reduce_Scatter_Basic_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                256,
                2,
                {
                    {0, 65536, &MPIR_Reduce_Scatter_Basic_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                512,
                2,
                {
                    {0, 65536, &MPIR_Reduce_Scatter_Basic_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                1024,
                2,
                {
                    {0, 65536, &MPIR_Reduce_Scatter_Basic_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
        }; 
        MPIU_Memcpy(mv2_red_scat_thresholds_table, mv2_tmp_red_scat_thresholds_table,
                  mv2_size_red_scat_tuning_table * sizeof (mv2_red_scat_tuning_table));
    } else


#endif /* !CHANNEL_PSM */
    {
        mv2_size_red_scat_tuning_table = 7;
        mv2_red_scat_thresholds_table = MPIU_Malloc(mv2_size_red_scat_tuning_table *
                                                  sizeof (mv2_red_scat_tuning_table));
        mv2_red_scat_tuning_table mv2_tmp_red_scat_thresholds_table[] = {
            {
                8,
                4,
                {
                    {0, 256, &MPIR_Reduce_Scatter_Basic_MV2},
                    {256, 16384, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {16384, 65536, &MPIR_Reduce_scatter_Pair_Wise_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                16,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                32,
                3,
                {
                    {0, 64, &MPIR_Reduce_Scatter_Basic_MV2},
                    {64, 131072, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {131072, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                64,
                3,
                {
                    {0, 1024, &MPIR_Reduce_Scatter_Basic_MV2},
                    {1024, 262144, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {262144, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                128,
                2,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                256,
                2,
                {
                    {0, 128, &MPIR_Reduce_Scatter_Basic_MV2},
                    {128, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
            {
                512,
                2,
                {
                    {0, 256, &MPIR_Reduce_Scatter_Basic_MV2},
                    {256, 65536, &MPIR_Reduce_scatter_Rec_Halving_MV2},
                    {65536, -1, &MPIR_Reduce_scatter_ring_2lvl},
                },
            },
        };
        MPIU_Memcpy(mv2_red_scat_thresholds_table, mv2_tmp_red_scat_thresholds_table,
                  mv2_size_red_scat_tuning_table * sizeof (mv2_red_scat_tuning_table));

    }
    return 0;
}

void MV2_cleanup_red_scat_tuning_table()
{
    if (mv2_red_scat_thresholds_table != NULL) {
        MPIU_Free(mv2_red_scat_thresholds_table);
    }

}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}


int MV2_internode_Red_scat_is_define(char *mv2_user_red_scat_inter)
{
    int i = 0;
    int nb_element = count_sep(mv2_user_red_scat_inter) + 1;

    /* If one red_scat tuning table is already defined */
    if (mv2_red_scat_thresholds_table != NULL) {
        MPIU_Free(mv2_red_scat_thresholds_table);
    }

    mv2_red_scat_tuning_table mv2_tmp_red_scat_thresholds_table[1];
    mv2_size_red_scat_tuning_table = 1;

    /* We realloc the space for the new red_scat tuning table */
    mv2_red_scat_thresholds_table = MPIU_Malloc(mv2_size_red_scat_tuning_table *
                                             sizeof (mv2_red_scat_tuning_table));

    if (nb_element == 1) {

        mv2_tmp_red_scat_thresholds_table[0].numproc = 1;
        mv2_tmp_red_scat_thresholds_table[0].size_inter_table = 1;
        mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].min = 0;
        mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].max = -1;
    
        switch (atoi(mv2_user_red_scat_inter)) {
        case RED_SCAT_BASIC:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_Scatter_Basic_MV2;
            break;
        case RED_SCAT_REC_HALF:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_scatter_Rec_Halving_MV2;
            break;
        case RED_SCAT_PAIRWISE:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_scatter_Pair_Wise_MV2;
            break;
        case RED_SCAT_RING:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_scatter_ring;
            break;
        case RED_SCAT_RING_2LVL:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_scatter_ring_2lvl;
            break;
        default:
            mv2_tmp_red_scat_thresholds_table[0].inter_leader[0].MV2_pt_Red_scat_function =
                &MPIR_Reduce_Scatter_Basic_MV2;
        }
        
    } else {
        char *dup, *p, *save_p;
        regmatch_t match[NMATCH];
        regex_t preg;
        const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

        if (!(dup = MPIU_Strdup(mv2_user_red_scat_inter))) {
            fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_red_scat_inter);
            return -1;
        }

        if (regcomp(&preg, regexp, REG_EXTENDED)) {
            fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_red_scat_inter);
            MPIU_Free(dup);
            return -1;
        }

        mv2_tmp_red_scat_thresholds_table[0].numproc = 1;
        mv2_tmp_red_scat_thresholds_table[0].size_inter_table = nb_element;

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
            case RED_SCAT_BASIC:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_Scatter_Basic_MV2;
                break;
            case RED_SCAT_REC_HALF:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_scatter_Rec_Halving_MV2;
                break;
            case RED_SCAT_PAIRWISE:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_scatter_Pair_Wise_MV2;
                break;
            case RED_SCAT_RING:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_scatter_ring;
                break;
            case RED_SCAT_RING_2LVL:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_scatter_ring_2lvl;
                break;
            default:
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].MV2_pt_Red_scat_function =
                    &MPIR_Reduce_Scatter_Basic_MV2;
            }

            mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].min = atoi(p +
                                                                         match[2].rm_so);
            if (p[match[3].rm_so] == '+') {
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].max = -1;
            } else {
                mv2_tmp_red_scat_thresholds_table[0].inter_leader[i].max =
                    atoi(p + match[3].rm_so);
                }
            i++;
        }
        MPIU_Free(dup);
        regfree(&preg);
    }
    MPIU_Memcpy(mv2_red_scat_thresholds_table, mv2_tmp_red_scat_thresholds_table, sizeof
                (mv2_red_scat_tuning_table));
    return 0;
}
