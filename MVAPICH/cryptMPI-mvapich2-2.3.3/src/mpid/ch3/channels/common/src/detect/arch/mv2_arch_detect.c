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

#include <stdio.h>
#include <string.h>
#if defined(HAVE_LIBIBVERBS)
#include <infiniband/verbs.h>
#endif

#include <mpichconf.h>

#include <hwloc.h>
#include <dirent.h>

#include "hwloc_bind.h"
#include "mv2_arch_hca_detect.h"
#include "debug_utils.h"
#include "upmi.h"
#include "mpi.h"

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
#include "rdma_impl.h"
#include "mv2_mpit_cvars.h"
#endif

#if defined(_MCST_SUPPORT_)
#include "ibv_param.h"
#endif
/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : FORCE_ARCH_TYPE
      category    : CH3
      type        : int
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter forces the architecture type.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
MPI_T_cvar_handle mv2_force_arch_type_handle = NULL;
#endif

#if defined(_SMP_LIMIC_)
#define SOCKETS 32
#define CORES 32
#define HEX_FORMAT 16
#define CORES_REP_AS_BITS 32


/*global variables*/
static int node[SOCKETS][CORES] = {{0}};
static int no_sockets=0;
static int socket_bound=-1; 
static int numcores_persocket[SOCKETS]={0};
#endif /*#if defined(_SMP_LIMIC_)*/

static mv2_arch_type g_mv2_arch_type = MV2_ARCH_UNKWN;
int g_mv2_num_cpus = -1;
static int g_mv2_cpu_model = -1;
static mv2_cpu_family_type g_mv2_cpu_family_type = MV2_CPU_FAMILY_NONE;

extern int mv2_enable_zcpy_bcast;
extern int mv2_use_slot_shmem_coll;

mv2_arch_type table_arch_tmp;
mv2_hca_type  table_hca_tmp;

#define CONFIG_FILE         "/proc/cpuinfo"
#define MAX_LINE_LENGTH     512
#define MAX_NAME_LENGTH     512

#define CLOVERTOWN_MODEL    15
#define HARPERTOWN_MODEL    23
#define NEHALEM_MODEL       26
#define INTEL_E5630_MODEL   44
#define INTEL_X5650_MODEL   44
#define INTEL_E5_2670_MODEL 45
#define INTEL_XEON_E5_2670_V2_MODEL 62
#define INTEL_XEON_E5_2698_V3_MODEL 63
#define INTEL_XEON_E5_2660_V3_MODEL 63
#define INTEL_XEON_E5_2680_V3_MODEL 63
#define INTEL_XEON_E5_2695_V3_MODEL 63
#define INTEL_XEON_E5_2670_V3_MODEL 64
#define INTEL_XEON_E5_2680_V4_MODEL 79
/* Skylake */
#define INTEL_PLATINUM_8160_MODEL   85
#define INTEL_PLATINUM_8170_MODEL   85
/* Cascade Lake */
#define INTEL_PLATINUM_8260_MODEL   85

#define MV2_STR_VENDOR_ID    "vendor_id"
#define MV2_STR_AUTH_AMD     "AuthenticAMD"
#define MV2_STR_MODEL        "model"
#define MV2_STR_WS            " "
#define MV2_STR_PHYSICAL     "physical"
#define MV2_STR_MODEL_NAME   "model name"
#define MV2_STR_POWER8_ID    "POWER8"
#define MV2_STR_POWER9_ID    "POWER9"
#define MV2_STR_CAVIUM_ID    "0x43"
#define MV2_ARM_CAVIUM_V8_MODEL 8

#define INTEL_E5_2670_MODEL_NAME    "Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz"
#define INTEL_E5_2680_MODEL_NAME    "Intel(R) Xeon(R) CPU E5-2680 0 @ 2.70GHz"
#define INTEL_E5_2670_V2_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz"
#define INTEL_E5_2630_V2_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz"
#define INTEL_E5_2680_V2_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz"
#define INTEL_E5_2690_V2_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2690 v2 @ 3.00GHz"
#define INTEL_E5_2690_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz"
#define INTEL_E5_2698_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz"
#define INTEL_E5_2660_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz"
#define INTEL_E5_2680_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz"
#define INTEL_E5_2680_V4_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"
#define INTEL_E5_2687W_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2687W v3 @ 3.10GHz"
#define INTEL_E5_2670_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz"
#define INTEL_E5_2695_V3_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz"
#define INTEL_E5_2695_V4_MODEL_NAME "Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz"

/* For both Skylake and Cascade Lake, generic models are tthe same */
#define INTEL_PLATINUM_GENERIC_MODEL_NAME  "Intel(R) Xeon(R) Platinum"
#define INTEL_PLATINUM_8160_MODEL_NAME "Intel(R) Xeon(R) Platinum 8160 CPU @ 2.10GHz"
#define INTEL_PLATINUM_8170_MODEL_NAME "Intel(R) Xeon(R) Platinum 8170 CPU @ 2.10GHz"
#define INTEL_PLATINUM_8260_MODEL_NAME "Intel(R) Xeon(R) Platinum 8260Y CPU @ 2.40GHz"
#define INTEL_PLATINUM_8280_MODEL_NAME "Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz"

#define INTEL_GOLD_GENERIC_MODEL_NAME  "Intel(R) Xeon(R) Gold"
#define INTEL_GOLD_6132_MODEL_NAME "Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz"


#define INTEL_XEON_PHI_GENERIC_MODEL_NAME "Intel(R) Xeon Phi(TM) CPU"
#define INTEL_XEON_PHI_7210_MODEL_NAME    "Intel(R) Xeon Phi(TM) CPU 7210 @ 1.30GHz"
#define INTEL_XEON_PHI_7230_MODEL_NAME    "Intel(R) Xeon Phi(TM) CPU 7230 @ 1.30GHz"
#define INTEL_XEON_PHI_7250_MODEL_NAME    "Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz"
#define INTEL_XEON_PHI_7290_MODEL_NAME    "Intel(R) Xeon Phi(TM) CPU 7290 @ 1.50GHz"

typedef struct _mv2_arch_types_log_t{
    uint64_t arch_type;
    char *arch_name;
}mv2_arch_types_log_t;

#define MV2_ARCH_LAST_ENTRY -1
static mv2_arch_types_log_t mv2_arch_types_log[] = 
{
    /* Intel Architectures */
    {MV2_ARCH_INTEL_GENERIC,        "MV2_ARCH_INTEL_GENERIC"},
    {MV2_ARCH_INTEL_CLOVERTOWN_8,   "MV2_ARCH_INTEL_CLOVERTOWN_8"},
    {MV2_ARCH_INTEL_NEHALEM_8,      "MV2_ARCH_INTEL_NEHALEM_8"},
    {MV2_ARCH_INTEL_NEHALEM_16,     "MV2_ARCH_INTEL_NEHALEM_16"},
    {MV2_ARCH_INTEL_HARPERTOWN_8,   "MV2_ARCH_INTEL_HARPERTOWN_8"},
    {MV2_ARCH_INTEL_XEON_DUAL_4,    "MV2_ARCH_INTEL_XEON_DUAL_4"},
    {MV2_ARCH_INTEL_XEON_E5630_8,   "MV2_ARCH_INTEL_XEON_E5630_8"},
    {MV2_ARCH_INTEL_XEON_X5650_12,  "MV2_ARCH_INTEL_XEON_X5650_12"},
    {MV2_ARCH_INTEL_XEON_E5_2670_16,"MV2_ARCH_INTEL_XEON_E5_2670_16"},
    {MV2_ARCH_INTEL_XEON_E5_2680_16,"MV2_ARCH_INTEL_XEON_E5_2680_16"},
    {MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,"MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20"},
    {MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,"MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12"},
    {MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,"MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20"},
    {MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,"MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20"},
    {MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,"MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24"},
    {MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,"MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32"},
    {MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,"MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20"},
    {MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,"MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24"},
    {MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,"MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20"},
    {MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,"MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24"},
    {MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28,"MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28"},
    {MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36,"MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36"},
    {MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,"MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28"},

    /* Skylake and Cascade Lake Architectures */
    {MV2_ARCH_INTEL_PLATINUM_GENERIC,      "MV2_ARCH_INTEL_PLATINUM_GENERIC"},
    {MV2_ARCH_INTEL_PLATINUM_8160_2S_48,   "MV2_ARCH_INTEL_PLATINUM_8160_2S_48"},
    {MV2_ARCH_INTEL_PLATINUM_8260_2S_48,   "MV2_ARCH_INTEL_PLATINUM_8260_2S_48"},
    {MV2_ARCH_INTEL_PLATINUM_8280_2S_56,   "MV2_ARCH_INTEL_PLATINUM_8280_2S_56"},
    {MV2_ARCH_INTEL_PLATINUM_8170_2S_52,   "MV2_ARCH_INTEL_PLATINUM_8170_2S_52"},
    {MV2_ARCH_INTEL_GOLD_GENERIC,          "MV2_ARCH_INTEL_GOLD_GENERIC"},
    {MV2_ARCH_INTEL_GOLD_6132_2S_28,       "MV2_ARCH_INTEL_GOLD_6132_2S_28"},


    /* KNL Architectures */
    {MV2_ARCH_INTEL_KNL_GENERIC,    "MV2_ARCH_INTEL_KNL_GENERIC"},
    {MV2_ARCH_INTEL_XEON_PHI_7210,  "MV2_ARCH_INTEL_XEON_PHI_7210"},
    {MV2_ARCH_INTEL_XEON_PHI_7230,  "MV2_ARCH_INTEL_XEON_PHI_7230"},
    {MV2_ARCH_INTEL_XEON_PHI_7250,  "MV2_ARCH_INTEL_XEON_PHI_7250"},
    {MV2_ARCH_INTEL_XEON_PHI_7290,  "MV2_ARCH_INTEL_XEON_PHI_7290"},

    /* AMD Architectures */
    {MV2_ARCH_AMD_GENERIC,          "MV2_ARCH_AMD_GENERIC"},
    {MV2_ARCH_AMD_BARCELONA_16,     "MV2_ARCH_AMD_BARCELONA_16"},
    {MV2_ARCH_AMD_MAGNY_COURS_24,   "MV2_ARCH_AMD_MAGNY_COURS_24"},
    {MV2_ARCH_AMD_OPTERON_DUAL_4,   "MV2_ARCH_AMD_OPTERON_DUAL_4"},
    {MV2_ARCH_AMD_OPTERON_6136_32,  "MV2_ARCH_AMD_OPTERON_6136_32"},
    {MV2_ARCH_AMD_OPTERON_6276_64,  "MV2_ARCH_AMD_OPTERON_6276_64"},
    {MV2_ARCH_AMD_BULLDOZER_4274HE_16,"MV2_ARCH_AMD_BULLDOZER_4274HE_16"},
    {MV2_ARCH_AMD_EPYC_7551_64, "MV2_ARCH_AMD_EPYC_7551_64"},
    {MV2_ARCH_AMD_EPYC_7742_128, "MV2_ARCH_AMD_EPYC_7742_128"},

    /* IBM Architectures */
    {MV2_ARCH_IBM_PPC,              "MV2_ARCH_IBM_PPC"},
    {MV2_ARCH_IBM_POWER8,           "MV2_ARCH_IBM_POWER8"},
    {MV2_ARCH_IBM_POWER9,           "MV2_ARCH_IBM_POWER9"},

    /* ARM Architectures */
    {MV2_ARCH_ARM_CAVIUM_V8_2S_28,  "MV2_ARCH_ARM_CAVIUM_V8_2S_28"},
    {MV2_ARCH_ARM_CAVIUM_V8_2S_32,  "MV2_ARCH_ARM_CAVIUM_V8_2S_32"},

    /* Unknown */
    {MV2_ARCH_UNKWN,                "MV2_ARCH_UNKWN"},
    {MV2_ARCH_LAST_ENTRY,           "MV2_ARCH_LAST_ENTRY"},
};

typedef struct _mv2_cpu_family_types_log_t{
    mv2_cpu_family_type family_type;
    char *cpu_family_name;
}mv2_cpu_family_types_log_t;

static mv2_cpu_family_types_log_t mv2_cpu_family_types_log[] = 
{
    {MV2_CPU_FAMILY_NONE,  "MV2_CPU_FAMILY_NONE"},
    {MV2_CPU_FAMILY_INTEL, "MV2_CPU_FAMILY_INTEL"},
    {MV2_CPU_FAMILY_AMD,   "MV2_CPU_FAMILY_AMD"},
    {MV2_CPU_FAMILY_POWER, "MV2_CPU_FAMILY_POWER"},
    {MV2_CPU_FAMILY_ARM,   "MV2_CPU_FAMILY_ARM"},
};

char *mv2_get_cpu_family_name(mv2_cpu_family_type cpu_family_type)
{
    return mv2_cpu_family_types_log[cpu_family_type].cpu_family_name;
}

char*  mv2_get_arch_name(mv2_arch_type arch_type)
{
    int i=0;
    while(mv2_arch_types_log[i].arch_type != MV2_ARCH_LAST_ENTRY){

        if(mv2_arch_types_log[i].arch_type == arch_type){
            return(mv2_arch_types_log[i].arch_name);
        }
        i++;
    }
    return("MV2_ARCH_UNKWN");
}

int mv2_check_proc_arch(mv2_arch_type type, int rank)
{
    if (type <= MV2_ARCH_LIST_START  || type >= MV2_ARCH_LIST_END  ||
        type == MV2_ARCH_INTEL_START || type == MV2_ARCH_INTEL_END ||
        type == MV2_ARCH_AMD_START   || type == MV2_ARCH_AMD_END   ||
        type == MV2_ARCH_IBM_START   || type == MV2_ARCH_IBM_END   ||
        type == MV2_ARCH_ARM_START   || type == MV2_ARCH_ARM_END) {

        PRINT_INFO((rank==0), "Wrong value specified for MV2_FORCE_ARCH_TYPE\n");
        PRINT_INFO((rank==0), "Value must be greater than %d and less than %d \n",
                    MV2_ARCH_LIST_START, MV2_ARCH_LIST_END);
        PRINT_INFO((rank==0), "For Intel Architectures: Please enter value greater than %d and less than %d\n",
                    MV2_ARCH_INTEL_START, MV2_ARCH_INTEL_END);
        PRINT_INFO((rank==0), "For AMD Architectures: Please enter value greater than %d and less than %d\n",
                    MV2_ARCH_AMD_START, MV2_ARCH_AMD_END);
        PRINT_INFO((rank==0), "For IBM Architectures: Please enter value greater than %d and less than %d\n",
                    MV2_ARCH_IBM_START, MV2_ARCH_IBM_END);
        PRINT_INFO((rank==0), "For ARM Architectures: Please enter value greater than %d and less than %d\n",
                    MV2_ARCH_ARM_START, MV2_ARCH_ARM_END);
        return 1;
    }
    return 0;
}

mv2_arch_type mv2_get_intel_arch_type(char *model_name, int num_sockets, int num_cpus)
{
    mv2_arch_type arch_type = MV2_ARCH_UNKWN;
    arch_type = MV2_ARCH_INTEL_GENERIC;

    if (1 == num_sockets) {
        if (64 == num_cpus ||
            68 == num_cpus ||
            72 == num_cpus )
        {
            /* Map all KNL CPUs to 7250 */
            if (NULL != strstr(model_name, INTEL_XEON_PHI_GENERIC_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_XEON_PHI_7250;
            }
        }
    } else if(2 == num_sockets) {
        if(4 == num_cpus) {
            arch_type = MV2_ARCH_INTEL_XEON_DUAL_4;

        } else if(8 == num_cpus) {

            if(CLOVERTOWN_MODEL == g_mv2_cpu_model) {
                arch_type = MV2_ARCH_INTEL_CLOVERTOWN_8;

            } else if(HARPERTOWN_MODEL == g_mv2_cpu_model) {
                arch_type = MV2_ARCH_INTEL_HARPERTOWN_8;

            } else if(NEHALEM_MODEL == g_mv2_cpu_model) {
                arch_type = MV2_ARCH_INTEL_NEHALEM_8;

            } else if(INTEL_E5630_MODEL == g_mv2_cpu_model){
                arch_type = MV2_ARCH_INTEL_XEON_E5630_8;
            } 

        } else if(12 == num_cpus) {
            if(INTEL_X5650_MODEL == g_mv2_cpu_model) {  
                /* Westmere EP model, Lonestar */
                arch_type = MV2_ARCH_INTEL_XEON_X5650_12;
            } else if (INTEL_XEON_E5_2670_V2_MODEL == g_mv2_cpu_model) {
                if(NULL != strstr(model_name, INTEL_E5_2630_V2_MODEL_NAME)){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12;
                }
            }
        } else if(16 == num_cpus) {

            if(NEHALEM_MODEL == g_mv2_cpu_model) {  /* nehalem with smt on */
                arch_type = MV2_ARCH_INTEL_NEHALEM_16;

            }else if(INTEL_E5_2670_MODEL == g_mv2_cpu_model) {
                if(strncmp(model_name, INTEL_E5_2670_MODEL_NAME, 
                            strlen(INTEL_E5_2670_MODEL_NAME)) == 0){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2670_16;

                } else if(strncmp(model_name, INTEL_E5_2680_MODEL_NAME, 
                            strlen(INTEL_E5_2680_MODEL_NAME)) == 0){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2680_16;

                } else {
                    arch_type = MV2_ARCH_INTEL_GENERIC;
                }
            }
        } else if(20 == num_cpus){
            if(INTEL_XEON_E5_2670_V2_MODEL == g_mv2_cpu_model) {
                if(NULL != strstr(model_name, INTEL_E5_2670_V2_MODEL_NAME)){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20;
                }else if(NULL != strstr(model_name, INTEL_E5_2680_V2_MODEL_NAME)){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20;
                }else if(NULL != strstr(model_name, INTEL_E5_2690_V2_MODEL_NAME)){
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20;
                }
            } else if(NULL != strstr(model_name, INTEL_E5_2687W_V3_MODEL_NAME)){
                arch_type = MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20;
            } else if (INTEL_XEON_E5_2660_V3_MODEL == g_mv2_cpu_model) {
                if(NULL != strstr(model_name, INTEL_E5_2660_V3_MODEL_NAME)) {
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20;
                }
            }
        } else if(24 == num_cpus){
            if(NULL != strstr(model_name, INTEL_E5_2680_V3_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24;
            } else if(NULL != strstr(model_name, INTEL_E5_2690_V3_MODEL_NAME)){
                arch_type = MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24;
            } else if(NULL != strstr(model_name, INTEL_E5_2670_V3_MODEL_NAME)){
                arch_type = MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24;
            }
        } else if(28 == num_cpus){
            if(NULL != strstr(model_name, INTEL_E5_2695_V3_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28;
            } else if(NULL != strstr(model_name, INTEL_E5_2680_V4_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28;
            } else if(NULL != strstr(model_name, INTEL_GOLD_GENERIC_MODEL_NAME)) { /* SkL Gold */
                arch_type = MV2_ARCH_INTEL_PLATINUM_8170_2S_52; /* Use generic SKL tables */
            }
        } else if(32 == num_cpus){
            if(INTEL_XEON_E5_2698_V3_MODEL == g_mv2_cpu_model) {
                if(NULL != strstr(model_name, INTEL_E5_2698_V3_MODEL_NAME)) {
                    arch_type = MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32;
                }
            }
        /* Support Pitzer cluster */
        } else if(40 == num_cpus){
            if(NULL != strstr(model_name, INTEL_GOLD_GENERIC_MODEL_NAME)) { /* SkL Gold */
                arch_type = MV2_ARCH_INTEL_PLATINUM_8170_2S_52; /* Use generic SKL tables */
            }
	/* detect skylake or cascade lake CPUs */
        } else if(48 == num_cpus || 52 == num_cpus || 56 == num_cpus || 44 == num_cpus /* azure skx */){
            if (NULL != strstr(model_name, INTEL_PLATINUM_GENERIC_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_PLATINUM_8170_2S_52;
            }

            /* Check if the model is Cascade lake, if yes then change from generic */
            if (NULL != strstr(model_name, INTEL_PLATINUM_8260_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_PLATINUM_8260_2S_48;
            }

            /* Frontera */
            if(NULL != strstr(model_name, INTEL_PLATINUM_8280_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_PLATINUM_8280_2S_56;
            }
        }  else if(36 == num_cpus || 72 == num_cpus){
            if(NULL != strstr(model_name, INTEL_E5_2695_V4_MODEL_NAME)) {
                arch_type = MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36;
            }
        }

    }

    return arch_type;
}

/* Identify architecture type */
mv2_arch_type mv2_get_arch_type()
{
    int my_rank = -1;
    char *value = NULL;

    UPMI_GET_RANK(&my_rank);

    if ((value = getenv("MV2_FORCE_ARCH_TYPE")) != NULL) {
        mv2_arch_type val = atoi(value);
        int retval = mv2_check_proc_arch(val, my_rank);
        if (retval) {
            PRINT_INFO((my_rank==0), "Falling back to automatic architecture detection\n");
        } else {
            g_mv2_arch_type = val;
        }
    }

    if ( MV2_ARCH_UNKWN == g_mv2_arch_type ) {
        FILE *fp;
        int num_sockets = 0, num_cpus = 0;
        int model_name_set=0;
        unsigned topodepth = -1, depth = -1;
        char line[MAX_LINE_LENGTH], *tmp, *key;
        char model_name[MAX_NAME_LENGTH]={0};

        mv2_arch_type arch_type = MV2_ARCH_UNKWN;
        smpi_load_hwloc_topology();

        /* Determine topology depth */
        topodepth = hwloc_topology_get_depth(topology);
        if( HWLOC_TYPE_DEPTH_UNKNOWN == topodepth ) {
            fprintf(stderr, "Warning: %s: Failed to determine topology depth.\n", __func__ );
            return arch_type;
        }

        /* Count number of (logical) processors */
        depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);

        if( HWLOC_TYPE_DEPTH_UNKNOWN == depth ) {
            fprintf(stderr, "Warning: %s: Failed to determine number of processors.\n", __func__ );
            return arch_type;
        }
        if(! (num_cpus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE))){
            fprintf(stderr, "Warning: %s: Failed to determine number of processors.\n", __func__);
            return arch_type;
        }
        g_mv2_num_cpus = num_cpus;

        /* Count number of sockets */
        depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
        if( HWLOC_TYPE_DEPTH_UNKNOWN == depth ) {
            fprintf(stderr, "Warning: %s: Failed to determine number of sockets.\n", __func__);
            return arch_type;
        } else {
            num_sockets = hwloc_get_nbobjs_by_depth(topology, depth);
        }

        /* Parse /proc/cpuinfo for additional useful things */
        if((fp = fopen(CONFIG_FILE, "r"))) { 

            while(! feof(fp)) {
                memset(line, 0, MAX_LINE_LENGTH);
                fgets(line, MAX_LINE_LENGTH - 1, fp); 

                if(! (key = strtok(line, "\t:"))) {
                    continue;
                }

                /* Identify the CPU Family */
                if(! strcmp(key, MV2_STR_VENDOR_ID)) {
                    strtok(NULL, MV2_STR_WS);
                    tmp = strtok(NULL, MV2_STR_WS);

                    if (! strncmp(tmp, MV2_STR_AUTH_AMD, strlen( MV2_STR_AUTH_AMD
                                    ))) {
                        g_mv2_cpu_family_type = MV2_CPU_FAMILY_AMD;

                    } else {
                        g_mv2_cpu_family_type = MV2_CPU_FAMILY_INTEL;
                    }
                    continue;
                }

                /* Identify the CPU Family for POWER */
                if(! strcmp(key, "cpu")) {
                    strtok(NULL, MV2_STR_WS);
                    tmp = strtok(NULL, MV2_STR_WS);
                    if (! strncmp(tmp, MV2_STR_POWER8_ID, strlen(MV2_STR_POWER8_ID))) {
                        g_mv2_cpu_family_type = MV2_CPU_FAMILY_POWER;
                    	arch_type = MV2_ARCH_IBM_POWER8;
                        continue;
                    } else if (! strncmp(tmp, MV2_STR_POWER9_ID, strlen(MV2_STR_POWER9_ID))) {
                        g_mv2_cpu_family_type = MV2_CPU_FAMILY_POWER;
                    	arch_type = MV2_ARCH_IBM_POWER9;
                        continue;
                    }
                }

                /* Identify the CPU Family for ARM */
                if(! strcmp(key, "CPU implementer")) {
                    /* Skip ':' */
                    strtok(NULL, MV2_STR_WS);
                    tmp = strtok(NULL, MV2_STR_WS);
                    if (! strncmp(tmp, MV2_STR_CAVIUM_ID, strlen(MV2_STR_CAVIUM_ID))) {
                        g_mv2_cpu_family_type = MV2_CPU_FAMILY_ARM;
                        g_mv2_cpu_model = MV2_ARM_CAVIUM_V8_MODEL;
			if (num_cpus == 56) {
			    arch_type = MV2_ARCH_ARM_CAVIUM_V8_2S_28;
			} else if (num_cpus == 64) {
			    arch_type = MV2_ARCH_ARM_CAVIUM_V8_2S_32;
			}
                        continue;
                    }
                }

                if( -1 == g_mv2_cpu_model ) {

                    if(! strcmp(key, MV2_STR_MODEL)) {
                        strtok(NULL, MV2_STR_WS);
                        tmp = strtok(NULL, MV2_STR_WS);
                        sscanf(tmp, "%d", &g_mv2_cpu_model);
                        continue;
                    }
                }

                if (!model_name_set){
                    if (strncmp(key, MV2_STR_MODEL_NAME, strlen(MV2_STR_MODEL_NAME)) == 0) {
                        strtok(NULL, MV2_STR_WS);
                        tmp = strtok(NULL, "\n");
                        sscanf(tmp, "%[^\n]\n", model_name);
                        model_name_set = 1;
                    }
                }
            }
            fclose(fp);

            if( MV2_CPU_FAMILY_INTEL == g_mv2_cpu_family_type ) {
                arch_type = mv2_get_intel_arch_type(model_name, num_sockets, num_cpus);
            } else if(MV2_CPU_FAMILY_AMD == g_mv2_cpu_family_type) {
                arch_type = MV2_ARCH_AMD_GENERIC;
                if(2 == num_sockets) {
                    if(4 == num_cpus) {
                        arch_type = MV2_ARCH_AMD_OPTERON_DUAL_4;
                    } else if(16 == num_cpus) {
                        arch_type =  MV2_ARCH_AMD_BULLDOZER_4274HE_16;

                    } else if(24 == num_cpus) {
                        arch_type =  MV2_ARCH_AMD_MAGNY_COURS_24;
                    } else if(64 == num_cpus || 60 == num_cpus /* azure vm */) {
                        arch_type =  MV2_ARCH_AMD_EPYC_7551_64;
                    } else if(128 == num_cpus) { /* rome */
                        arch_type = MV2_ARCH_AMD_EPYC_7742_128;
                    }
                } else if(4 == num_sockets) {
                    if(16 == num_cpus) {
                        arch_type =  MV2_ARCH_AMD_BARCELONA_16;
                    } else if(32 == num_cpus) {
                        arch_type =  MV2_ARCH_AMD_OPTERON_6136_32;
                    } else if(64 == num_cpus) {
                        arch_type =  MV2_ARCH_AMD_OPTERON_6276_64;
                    }
                }
            }
        } else {
            fprintf(stderr, "Warning: %s: Failed to open \"%s\".\n", __func__,
                    CONFIG_FILE);
        }
        g_mv2_arch_type = arch_type;
        if (MV2_ARCH_UNKWN == g_mv2_arch_type) {
            PRINT_INFO((my_rank==0), "**********************WARNING***********************\n");
            PRINT_INFO((my_rank==0), "Failed to automatically detect the CPU architecture.\n");
            PRINT_INFO((my_rank==0), "This may lead to subpar communication performance.\n");
            PRINT_INFO((my_rank==0), "****************************************************\n");
        }
        return arch_type;
    } else {
        return g_mv2_arch_type;
    }

}

/* API for getting the number of cpus */
int mv2_get_num_cpus()
{
    /* Check if num_cores is already identified */
    if ( -1 == g_mv2_num_cpus ){
        g_mv2_arch_type = mv2_get_arch_type();
    }
    return g_mv2_num_cpus;
}

/* API for getting the CPU model */
int mv2_get_cpu_model()
{
    /* Check if cpu model is already identified */
    if (-1 == g_mv2_cpu_model){
        g_mv2_arch_type = mv2_get_arch_type();
    }
    return g_mv2_cpu_model;
}

/* Get CPU family */
mv2_cpu_family_type mv2_get_cpu_family()
{
    /* Check if cpu family is already identified */
    if (MV2_CPU_FAMILY_NONE == g_mv2_cpu_family_type){
        g_mv2_arch_type = mv2_get_arch_type();
    }
    return g_mv2_cpu_family_type;
}

/* Check arch-hca type */
int mv2_is_arch_hca_type(mv2_arch_hca_type arch_hca_type, 
        mv2_arch_type arch_type, mv2_hca_type hca_type)
{
    int ret;
    uint16_t my_arch_type, my_hca_type;
    uint64_t mask = UINT16_MAX;
    arch_hca_type >>= 16;
    my_hca_type = arch_hca_type & mask;
    arch_hca_type >>= 16;
    my_arch_type = arch_hca_type & mask;

    table_arch_tmp = my_arch_type;
    table_hca_tmp = my_hca_type;

    if (((MV2_ARCH_ANY == arch_type) || (MV2_ARCH_ANY == my_arch_type)) &&
        ((MV2_HCA_ANY == hca_type) || (MV2_HCA_ANY == my_hca_type))) {
        ret = 1;
    } else if (MV2_ARCH_ANY == arch_type){  // cores
        ret = (my_hca_type==hca_type) ? 1: 0;
    } else if ((MV2_HCA_ANY == hca_type) || (MV2_HCA_ANY == my_hca_type)) {
        ret = (my_arch_type==arch_type) ? 1: 0;
    } else{
        ret = (my_arch_type==arch_type && my_hca_type==hca_type) ? 1:0;
    }
    return ret;
}
#if defined(_SMP_LIMIC_)
void hwlocSocketDetection(int print_details)
{
    int depth;
    unsigned i, j;
    char *str;
    char * pEnd;
    char * pch;
    int more32bit=0,offset=0;
    long int core_cnt[2];
    hwloc_cpuset_t cpuset;
    hwloc_obj_t sockets;
    
    /* Perform the topology detection. */
    smpi_load_hwloc_topology();
    /*clear all the socket information and reset to -1*/
    for(i=0;i<SOCKETS;i++)
        for(j=0;j<CORES;j++)
            node[i][j]=-1;
    
    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
    no_sockets=hwloc_get_nbobjs_by_depth(topology, depth);
    
    PRINT_DEBUG(DEBUG_SHM_verbose>0, "Total number of sockets=%d\n", no_sockets);

    for(i=0;i<no_sockets;i++)
    {   
        sockets = hwloc_get_obj_by_type(topology, HWLOC_OBJ_SOCKET, i); 
        cpuset = sockets->cpuset;
        hwloc_bitmap_asprintf(&str, cpuset);

        /*tokenize the str*/
        pch = strtok (str,",");
        while (pch != NULL)
        {   
            pch = strtok (NULL, ",");
            if(pch != NULL)
            {   
                more32bit=1;
                break;
            }   
        }   
        
        core_cnt[0]= strtol (str,&pEnd,HEX_FORMAT);
        /*if more than bits, then explore the values*/
        if(more32bit)
        {   
            /*tells multiple of 32 bits(eg if 0, then 64 bits)*/
            core_cnt[1] = strtol (pEnd,NULL,0);
            offset = (core_cnt[1] + 1)*CORES_REP_AS_BITS;
        }   

        for(j=0;j<CORES_REP_AS_BITS;j++)
        {   
            if(core_cnt[0] & 1)
            {   
                node[i][j]=j+offset;
                (numcores_persocket[i])++;
            }   
            core_cnt[0] = (core_cnt[0] >> 1); 
        } 
        
        if (DEBUG_SHM_verbose>0) {
            printf("Socket %d, num of cores / socket=%d\n", i, (numcores_persocket[i]));
            printf("core id\n");

            for (j=0;j<CORES_REP_AS_BITS;j++) {
                printf("%d\t", node[i][j]);
            }
            printf("\n");
        }
    }   
    free(str);

}

//Check the core, where the process is bound to
int getProcessBinding(pid_t pid)
{
    int res,i=0,j=0;
    char *str=NULL;
    char *pEnd=NULL;
    char *pch=NULL;
    int more32bit=0,offset=0;
    unsigned int core_bind[2];
    hwloc_bitmap_t cpubind_set;

    /* Perform the topology detection. */
    smpi_load_hwloc_topology();
    cpubind_set = hwloc_bitmap_alloc();
    res = hwloc_get_proc_cpubind(topology, pid, cpubind_set, 0);
    if(-1 == res)
        fprintf(stderr, "getProcessBinding(): Error in getting cpubinding of process");

    hwloc_bitmap_asprintf(&str, cpubind_set);
    
    /*tokenize the str*/
    pch = strtok (str,",");
    while (pch != NULL)
    {   
        pch = strtok (NULL, ",");
        if(pch != NULL)
        {   
            more32bit=1;
            break;
        }   
    }   

    core_bind[0]= strtol (str,&pEnd,HEX_FORMAT);
    
    /*if more than bits, then explore the values*/
    if(more32bit)
    {   
        /*tells multiple of 32 bits(eg if 0, then 64 bits)*/
        PRINT_DEBUG(DEBUG_SHM_verbose>0, "more bits set\n");
        core_bind[1] = strtol (pEnd,NULL,0);
        PRINT_DEBUG(DEBUG_SHM_verbose>0, "core_bind[1]=%x\n", core_bind[1]);
        offset = (core_bind[1] + 1)*CORES_REP_AS_BITS;
        PRINT_DEBUG(DEBUG_SHM_verbose>0, "Offset=%d\n", offset);
    }   

    for(j=0;j<CORES_REP_AS_BITS;j++)
    {   
        if(core_bind[0] & 1)
        {   
            core_bind[0]=j+offset;
            break;
        }   
        core_bind[0]= (core_bind[0] >> 1); 
    }   

    /*find the socket, where the core is present*/
    for(i=0;i<no_sockets;i++)
    {   
        j=core_bind[0]-offset;
        if(node[i][j]== j+offset)
        {
	        free(str);
            hwloc_bitmap_free(cpubind_set);
            return i; /*index of socket where the process is bound*/
        }
    }   
    fprintf(stderr, "Error: Process not bound on any core ??\n");
    free(str);
    hwloc_bitmap_free(cpubind_set);
    return -1;
}

int numOfCoresPerSocket(int socket)
{
    return numcores_persocket[socket];
}

int numofSocketsPerNode (void)
{
    return no_sockets;
}

int get_socket_bound(void)
{ 
   if(socket_bound == -1) { 
       socket_bound = getProcessBinding(getpid()); 
   } 
   return socket_bound; 
}
#else
void hwlocSocketDetection(int print_details) { }
int numOfCoresPerSocket(int socket) { return 0; }
int numofSocketsPerNode (void) { return 0; }
int get_socket_bound(void) { return -1; }
#endif /*#if defined(_SMP_LIMIC_)*/

/* return a number with the value'th bit set */
int find_bit_pos(int value)
{
  int pos = 1, tmp = 1;
  if(value == 0) return 0;
  while(tmp < value)
  {
    pos++;
    tmp = tmp << 1;
  }
  return pos;
}

/* given a socket object, find the number of cores per socket.
 *  * This could be useful on systems where cores-per-socket
 *   * are not uniform */
int get_core_count_per_socket (hwloc_topology_t topology, hwloc_obj_t obj, int depth) {
    int i, count = 0;
    if (obj->type == HWLOC_OBJ_CORE)
        return 1;

    for (i = 0; i < obj->arity; i++) {
        count += get_core_count_per_socket (topology, obj->children[i], depth+1);
    }
    return count;
}

int get_socket_bound_info(int *socket_bound, int *num_sockets, int *num_cores_socket, int *is_uniform)
{
    hwloc_cpuset_t cpuset;
    hwloc_obj_t socket;
    int i,num_cores;
    int err = -1;
    smpi_load_hwloc_topology_whole();
    *num_sockets = hwloc_get_nbobjs_by_type(topology_whole, HWLOC_OBJ_SOCKET);
    num_cores = hwloc_get_nbobjs_by_type(topology_whole, HWLOC_OBJ_CORE);
    pid_t pid = getpid();

    hwloc_bitmap_t cpubind_set = hwloc_bitmap_alloc();

    int result = hwloc_get_proc_cpubind(topology_whole, pid, cpubind_set, 0);
    if(result == -1)
    {
        PRINT_DEBUG(DEBUG_SHM_verbose > 0, "Error in getting cpubinding of process\n");
        return -1;
    }

    int topodepth = hwloc_get_type_depth (topology_whole, HWLOC_OBJ_SOCKET);
    *is_uniform = 1;

    for(i = 0; i < *num_sockets; i++)
    {
        socket = hwloc_get_obj_by_type(topology_whole, HWLOC_OBJ_SOCKET, i);
        cpuset = socket->cpuset;
        hwloc_obj_t obj = hwloc_get_obj_by_depth (topology_whole, topodepth, i); 
        int num_cores_in_socket = get_core_count_per_socket (topology_whole, obj, topodepth);

        if(num_cores_in_socket != (num_cores / (*num_sockets)))
        {
            *is_uniform = 0;
        }
       
        hwloc_bitmap_t result_set = hwloc_bitmap_alloc();                                                  
        hwloc_bitmap_and(result_set, cpuset, cpubind_set);                                                   
        if(hwloc_bitmap_last(result_set) != -1)                                                              
        {                                                                                           
            *num_cores_socket = num_cores_in_socket;                                                
            *socket_bound = i;                                                                      
            PRINT_DEBUG(DEBUG_SHM_verbose > 0, "Socket : %d Num cores :%d" 
                        " Num cores in socket: %d Num sockets : %d Uniform :%d"
                        "\n",i,num_cores, *num_cores_socket, *num_sockets, 
                        *is_uniform);
            err = 0;                                                                                
        }
        hwloc_bitmap_free(result_set); 
    }

    //Socket aware collectives don't support non-uniform architectures yet
    if(!*is_uniform)
    {
        err = 1;
    }
    return err;
}

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
int mv2_set_force_arch_type()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_FORCE_ARCH_TYPE, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_FORCE_ARCH_TYPE";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_force_arch_type_handle;
    wrapper.default_cvar_value = MV2_ARCH_UNKWN;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_FORCE_ARCH_TYPE";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = MV2_ARCH_LIST_END-1;
    wrapper.check_min = 1;
    wrapper.min_value = MV2_ARCH_LIST_START+1;
    wrapper.boundary_error_msg = "Wrong value specified for MPIR_CVAR_FORCE_ARCH_TYPE";
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        mv2_arch_type val = read_value;
        int retval = mv2_check_proc_arch(val, MPIDI_Process.my_pg_rank);
        if (retval) {
            PRINT_INFO(MPIDI_Process.my_pg_rank == 0, "Falling back to automatic"
                " architecture detection\n");
        } else {
            g_mv2_arch_type = val;
        }
    }
    fn_fail:
    fn_exit:
        return mpi_errno;
}
void mv2_free_arch_handle () {
    if (mv2_force_arch_type_handle) {
        MPIU_Free(mv2_force_arch_type_handle);
        mv2_force_arch_type_handle = NULL;
    }
}
#endif
