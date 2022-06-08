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
#include "allgatherv_tuning.h"
#include "mv2_arch_hca_detect.h"

enum {
    ALLGATHERV_REC = 1,
    ALLGATHERV_BRUCK,
    ALLGATHERV_RING,
    ALLGATHERV_CYCLIC,
};

int mv2_size_allgatherv_tuning_table = 0;
mv2_allgatherv_tuning_table *mv2_allgatherv_thresholds_table = NULL;

int MV2_set_allgatherv_tuning_table(int heterogeneity, struct coll_info *colls_arch_hca)
{
#ifndef CHANNEL_PSM
#ifdef CHANNEL_MRAIL_GEN2
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
            {
                12,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                24,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                48,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                96,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                192,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                384,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
            {
                16,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                32,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                64,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                128,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                256,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                512,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },

        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
        /*Trestles*/
            {
                32,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                64,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                128,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                256,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                512,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                1024,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else
#elif defined (CHANNEL_NEMESIS_IB)
    if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_X5650_12, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
            {
                12,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                24,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                48,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                96,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                192,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                384,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_INTEL_XEON_E5_2680_16, MV2_HCA_MLX_CX_FDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
            {
                16,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                32,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                64,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                128,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                256,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                512,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },

        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else if (MV2_IS_ARCH_HCA_TYPE(MV2_get_arch_hca_type(),
        MV2_ARCH_AMD_OPTERON_6136_32, MV2_HCA_MLX_CX_QDR) && !heterogeneity){
        mv2_size_allgatherv_tuning_table = 6;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
        /*Trestles*/
            {
                32,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                64,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                128,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                256,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                512,
                2,
                {
                    {0, 128, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {128, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                1024,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
        }; 
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));
    } else
#endif
#endif /* !CHANNEL_PSM */
    {
        mv2_size_allgatherv_tuning_table = 7;
        mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                                  sizeof (mv2_allgatherv_tuning_table));
        mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[] = {
            {
                8,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                16,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                32,
                2,
                {
                    {0, 512, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {512, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                64,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                128,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                256,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },
            {
                512,
                2,
                {
                    {0, 256, &MPIR_Allgatherv_Rec_Doubling_MV2},
                    {256, -1, &MPIR_Allgatherv_Ring_MV2},
                },
            },

        };
        MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table,
                  mv2_size_allgatherv_tuning_table * sizeof (mv2_allgatherv_tuning_table));

    }
    return 0;
}

void MV2_cleanup_allgatherv_tuning_table()
{
    if (mv2_allgatherv_thresholds_table != NULL) {
        MPIU_Free(mv2_allgatherv_thresholds_table);
    }

}

/* Return the number of separator inside a string */
static int count_sep(char *string)
{
    return *string == '\0' ? 0 : (count_sep(string + 1) + (*string == ','));
}


int MV2_internode_Allgatherv_is_define(char *mv2_user_allgatherv_inter)
{
    int i = 0;
    int nb_element = count_sep(mv2_user_allgatherv_inter) + 1;

    /* If one allgatherv tuning table is already defined */
    if (mv2_allgatherv_thresholds_table != NULL) {
        MPIU_Free(mv2_allgatherv_thresholds_table);
    }

    mv2_allgatherv_tuning_table mv2_tmp_allgatherv_thresholds_table[1];
    mv2_size_allgatherv_tuning_table = 1;

    /* We realloc the space for the new allgatherv tuning table */
    mv2_allgatherv_thresholds_table = MPIU_Malloc(mv2_size_allgatherv_tuning_table *
                                             sizeof (mv2_allgatherv_tuning_table));

    if (nb_element == 1) {

        mv2_tmp_allgatherv_thresholds_table[0].numproc = 1;
        mv2_tmp_allgatherv_thresholds_table[0].size_inter_table = 1;
        mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].min = 0;
        mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].max = -1;
    
        switch (atoi(mv2_user_allgatherv_inter)) {
        case ALLGATHERV_REC:
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].MV2_pt_Allgatherv_function =
                &MPIR_Allgatherv_Rec_Doubling_MV2;
            break;
        case ALLGATHERV_BRUCK:
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].MV2_pt_Allgatherv_function =
                &MPIR_Allgatherv_Bruck_MV2;
            break;
        case ALLGATHERV_RING:
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].MV2_pt_Allgatherv_function =
                &MPIR_Allgatherv_Ring_MV2;
            break;
        case ALLGATHERV_CYCLIC:
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].MV2_pt_Allgatherv_function =
                &MPIR_Allgatherv_Ring_Cyclic_MV2;
            break;
        default:
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[0].MV2_pt_Allgatherv_function =
                &MPIR_Allgatherv_Bruck_MV2;
        }
        
    } else {
        char *dup, *p, *save_p;
        regmatch_t match[NMATCH];
        regex_t preg;
        const char *regexp = "([0-9]+):([0-9]+)-([0-9]+|\\+)";

        if (!(dup = MPIU_Strdup(mv2_user_allgatherv_inter))) {
            fprintf(stderr, "failed to duplicate `%s'\n", mv2_user_allgatherv_inter);
            return -1;
        }

        if (regcomp(&preg, regexp, REG_EXTENDED)) {
            fprintf(stderr, "failed to compile regexp `%s'\n", mv2_user_allgatherv_inter);
            MPIU_Free(dup);
            return -1;
        }

        mv2_tmp_allgatherv_thresholds_table[0].numproc = 1;
        mv2_tmp_allgatherv_thresholds_table[0].size_inter_table = nb_element;

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
                case ALLGATHERV_REC:
                    mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].MV2_pt_Allgatherv_function =
                        &MPIR_Allgatherv_Rec_Doubling_MV2;
                    break;
                case ALLGATHERV_BRUCK:
                    mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].MV2_pt_Allgatherv_function =
                        &MPIR_Allgatherv_Bruck_MV2;
                    break;
                case ALLGATHERV_RING:
                    mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].MV2_pt_Allgatherv_function =
                        &MPIR_Allgatherv_Ring_MV2;
                    break;
                case ALLGATHERV_CYCLIC:
                    mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].MV2_pt_Allgatherv_function =
                        &MPIR_Allgatherv_Ring_Cyclic_MV2;
                    break;
                default:
                    mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].MV2_pt_Allgatherv_function =
                        &MPIR_Allgatherv_Bruck_MV2;
            } 
            mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].min = atoi(p +
                                                                         match[2].rm_so);
            if (p[match[3].rm_so] == '+') {
                mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].max = -1;
            } else {
                mv2_tmp_allgatherv_thresholds_table[0].inter_leader[i].max =
                    atoi(p + match[3].rm_so);
            }
            i++;
        }
        MPIU_Free(dup);
        regfree(&preg);
    }
    MPIU_Memcpy(mv2_allgatherv_thresholds_table, mv2_tmp_allgatherv_thresholds_table, sizeof
                (mv2_allgatherv_tuning_table));
    return 0;
}
