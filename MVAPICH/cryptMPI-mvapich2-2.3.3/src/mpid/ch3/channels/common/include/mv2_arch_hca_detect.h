/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 * Copyright (c) 2016, Intel, Inc. All rights reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#ifndef MV2_ARCH_HCA_DETECT_H
#define MV2_ARCH_HCA_DETECT_H

#include <stdint.h>

#if defined(HAVE_LIBIBVERBS)
#include <infiniband/verbs.h>
#endif

/* HCA Types */
#define MV2_HCA_UNKWN   0
#define MV2_HCA_ANY     (UINT16_MAX)

#define MV2_HCA_TYPE_IB

/* 
 * Layout:
 *
 * 1-4095 - IB Cards
 *         1 - 1000 - Mellanox Cards
 *      1001 - 2000 - Qlogic Cards
 *      2001 - 3000 - IBM Cards
 *      3001 - 4000 - Intel HFI Cards
 *
 * 4096-8191 - iWarp Cards 
 *      5001 - 6000 - Chelsio Cards
 *      6001 - 7000 - Intel iWarp Cards
 */

/* Mellanox Cards */
typedef enum {
        MV2_HCA_LIST_START=1,
/* Chelsio Cards */
        MV2_HCA_IWARP_TYPE_START,
        MV2_HCA_CHLSIO_START,
        MV2_HCA_CHELSIO_T3,
        MV2_HCA_CHELSIO_T4,
        MV2_HCA_CHLSIO_END,

/* Intel iWarp Cards */
        MV2_HCA_INTEL_IWARP_START,
        MV2_HCA_INTEL_NE020,
        MV2_HCA_INTEL_IWARP_END,
        MV2_HCA_IWARP_TYPE_END,

/* Mellanox IB HCAs */
        MV2_HCA_IB_TYPE_START,
        MV2_HCA_MLX_START,
        MV2_HCA_MLX_PCI_X,
        MV2_HCA_MLX_PCI_EX_SDR,
        MV2_HCA_MLX_PCI_EX_DDR,
        MV2_HCA_MLX_CX_SDR,
        MV2_HCA_MLX_CX_DDR,
        MV2_HCA_MLX_CX_QDR,
        MV2_HCA_MLX_CX_FDR,
        MV2_HCA_MLX_CX_CONNIB,
        MV2_HCA_MLX_CX_EDR,
        MV2_HCA_MLX_CX_HDR,
        MV2_HCA_MLX_END,
        MV2_HCA_IB_TYPE_END,

/* Qlogic Cards */
        MV2_HCA_QLGIC_START,
        MV2_HCA_QLGIC_PATH_HT,
        MV2_HCA_QLGIC_QIB,
        MV2_HCA_QLGIC_END,

/* IBM Cards */
        MV2_HCA_IBM_START,
        MV2_HCA_IBM_EHCA,
        MV2_HCA_IBM_END,

/* Intel Cards */
        MV2_HCA_INTEL_START,
        MV2_HCA_INTEL_HFI1,
        MV2_HCA_INTEL_END,

/* Marvel Cards */
        MV2_HCA_MARVEL_START,
        MV2_HCA_MARVEL_QEDR,
        MV2_HCA_MARVEL_END,

        MV2_HCA_LIST_END,
} mv2_hca_types_list;


/* Check if given card is IB card or not */
#define MV2_IS_IB_CARD(_x) \
    ((_x) > MV2_HCA_IB_TYPE_START && (_x) < MV2_HCA_IB_TYPE_END)

/* Check if given card is iWarp card or not */
#define MV2_IS_IWARP_CARD(_x) \
    ((_x) > MV2_HCA_IWARP_TYPE_START && (_x) < MV2_HCA_IWARP_TYPE_END)

/* Check if given card is Chelsio iWarp card or not */
#define MV2_IS_CHELSIO_IWARP_CARD(_x) \
    ((_x) > MV2_HCA_CHLSIO_START && (_x) < MV2_HCA_CHLSIO_END)

/* Check if given card is QLogic card or not */
#define MV2_IS_QLE_CARD(_x) \
    ((_x) > MV2_HCA_QLGIC_START && (_x) < MV2_HCA_QLGIC_END)

/* Check if given card is Intel card or not */
#define MV2_IS_INTEL_CARD(_x) \
    ((_x) > MV2_HCA_INTEL_START && (_x) < MV2_HCA_INTEL_END)

/* Check if given card is Marvel card or not */
#define MV2_IS_MARVEL_CARD(_x) \
    ((_x) > MV2_HCA_MARVEL_START && (_x) < MV2_HCA_MARVEL_END)

/* Architecture Type 
 * Layout:
 *    1 - 1000 - Intel architectures
 * 1001 - 2000 - AMD architectures
 * 2001 - 3000 - IBM architectures
 */
#define MV2_ARCH_UNKWN  0
#define MV2_ARCH_ANY    (UINT16_MAX)

/* Intel Architectures */
typedef enum {
        MV2_ARCH_LIST_START=1,
        MV2_ARCH_INTEL_START,
        MV2_ARCH_INTEL_GENERIC,
        MV2_ARCH_INTEL_CLOVERTOWN_8,
        MV2_ARCH_INTEL_NEHALEM_8,
        MV2_ARCH_INTEL_NEHALEM_16,
        MV2_ARCH_INTEL_HARPERTOWN_8,
        MV2_ARCH_INTEL_XEON_DUAL_4,
        MV2_ARCH_INTEL_XEON_E5630_8,
        MV2_ARCH_INTEL_XEON_X5650_12,
        MV2_ARCH_INTEL_XEON_E5_2670_16,
        MV2_ARCH_INTEL_XEON_E5_2680_16,
        MV2_ARCH_INTEL_XEON_E5_2670_V2_2S_20,
        MV2_ARCH_INTEL_XEON_E5_2630_V2_2S_12,
        MV2_ARCH_INTEL_XEON_E5_2680_V2_2S_20,
        MV2_ARCH_INTEL_XEON_E5_2690_V2_2S_20,
        MV2_ARCH_INTEL_XEON_E5_2698_V3_2S_32,
        MV2_ARCH_INTEL_XEON_E5_2660_V3_2S_20,
        MV2_ARCH_INTEL_XEON_E5_2680_V3_2S_24,
        MV2_ARCH_INTEL_XEON_E5_2690_V3_2S_24,
        MV2_ARCH_INTEL_XEON_E5_2687W_V3_2S_20,
        MV2_ARCH_INTEL_XEON_E5_2670_V3_2S_24,
        MV2_ARCH_INTEL_XEON_E5_2695_V3_2S_28,
        MV2_ARCH_INTEL_XEON_E5_2680_V4_2S_28,
        MV2_ARCH_INTEL_XEON_E5_2695_V4_2S_36,
        MV2_ARCH_INTEL_PLATINUM_8160_2S_48,
        MV2_ARCH_INTEL_PLATINUM_8260_2S_48,
        MV2_ARCH_INTEL_PLATINUM_8280_2S_56,
        MV2_ARCH_INTEL_PLATINUM_8170_2S_52,
        MV2_ARCH_INTEL_PLATINUM_GENERIC,
        MV2_ARCH_INTEL_GOLD_6132_2S_28,
        MV2_ARCH_INTEL_GOLD_GENERIC,
        MV2_ARCH_INTEL_KNL_GENERIC,
        MV2_ARCH_INTEL_XEON_PHI_7210,
        MV2_ARCH_INTEL_XEON_PHI_7230,
        MV2_ARCH_INTEL_XEON_PHI_7250,
        MV2_ARCH_INTEL_XEON_PHI_7290,
        MV2_ARCH_INTEL_END,
/* AMD Architectures */
        MV2_ARCH_AMD_START,
        MV2_ARCH_AMD_GENERIC,
        MV2_ARCH_AMD_BARCELONA_16,
        MV2_ARCH_AMD_MAGNY_COURS_24,
        MV2_ARCH_AMD_OPTERON_DUAL_4,
        MV2_ARCH_AMD_OPTERON_6136_32,
        MV2_ARCH_AMD_OPTERON_6276_64,
        MV2_ARCH_AMD_BULLDOZER_4274HE_16,
    	MV2_ARCH_AMD_EPYC_7551_64,
        MV2_ARCH_AMD_EPYC_7742_128,
        MV2_ARCH_AMD_END,
/* IBM Architectures */
        MV2_ARCH_IBM_START,
        MV2_ARCH_IBM_PPC,
        MV2_ARCH_IBM_POWER8,
        MV2_ARCH_IBM_POWER9,
        MV2_ARCH_IBM_END,
/* ARM Architectures */
        MV2_ARCH_ARM_START,
        MV2_ARCH_ARM_CAVIUM_V8_2S_28,
        MV2_ARCH_ARM_CAVIUM_V8_2S_32,
        MV2_ARCH_ARM_END,
        MV2_ARCH_LIST_END, 
} mv2_proc_arch_list;

typedef uint64_t mv2_arch_hca_type;
typedef uint16_t mv2_arch_type;
typedef uint16_t mv2_hca_type;
typedef uint16_t mv2_arch_num_cores;
typedef uint16_t mv2_arch_reserved;  /* reserved 16-bits for future use */

#define NUM_HCA_BITS (16)
#define NUM_ARCH_BITS (16)

#define MV2_GET_ARCH(_arch_hca) ((_arch_hca) >> 32)
#define MV2_GET_HCA(_arch_hca) (((_arch_hca) << 32) >> 48)

/* CPU Family */
typedef enum{
    MV2_CPU_FAMILY_NONE=0,
    MV2_CPU_FAMILY_INTEL,
    MV2_CPU_FAMILY_AMD,
    MV2_CPU_FAMILY_POWER,
    MV2_CPU_FAMILY_ARM,
}mv2_cpu_family_type;

/* Multi-rail info */
typedef enum{
    mv2_num_rail_unknown = 0,
    mv2_num_rail_1,
    mv2_num_rail_2,
    mv2_num_rail_3,
    mv2_num_rail_4,
} mv2_multirail_info_type;


#define MV2_IS_ARCH_HCA_TYPE(_arch_hca, _arch, _hca) \
    mv2_is_arch_hca_type(_arch_hca, _arch, _hca)

enum collectives {
    allgather = 0,
    allreduce,
    alltoall,
    alltoallv,
    bcast,
    gather,
    reduce,
    scatter,
    colls_max
};

static const char collective_names[colls_max][12] = {
    "Allgather",
    "Allreduce",
    "Alltoall",
    "Alltoallv",
    "Broadcast",
    "Gather",
    "Reduce",
    "Scatter"
};

struct coll_info {
    mv2_hca_type                 hca_type;
    mv2_arch_type                arch_type;
};

extern mv2_arch_type table_arch_tmp;
extern mv2_hca_type  table_hca_tmp;
extern int mv2_suppress_hca_warnings;

/* ************************ FUNCTION DECLARATIONS ************************** */

/* Check arch-hca type */
int mv2_is_arch_hca_type(mv2_arch_hca_type arch_hca_type, 
        mv2_arch_type arch_type, mv2_hca_type hca_type);

/* Get architecture-hca type */
#if defined(HAVE_LIBIBVERBS)
mv2_arch_hca_type mv2_get_arch_hca_type (struct ibv_device *dev);
mv2_arch_hca_type mv2_new_get_arch_hca_type (mv2_hca_type hca_type);
#else
mv2_arch_hca_type mv2_get_arch_hca_type (void *dev);
#endif

/* Check if the host has multiple rails or not */
mv2_multirail_info_type mv2_get_multirail_info(void);

/* Get architecture type */
mv2_arch_type mv2_get_arch_type(void);

/* Get card type */
#if defined(HAVE_LIBIBVERBS)
mv2_hca_type mv2_get_hca_type(struct ibv_device *dev);
mv2_hca_type mv2_new_get_hca_type(struct ibv_context *ctx, struct ibv_device *dev, uint64_t *guid);
#else
mv2_hca_type mv2_get_hca_type(void *dev);
#endif

/* Get combined architecture + hca type */
mv2_arch_hca_type MV2_get_arch_hca_type(void);

/* Get number of cpus */
int mv2_get_num_cpus(void);

/* Get the CPU model */
int mv2_get_cpu_model(void);

/* Get CPU family */
mv2_cpu_family_type mv2_get_cpu_family(void);

/* Log arch-hca type */
void mv2_log_arch_hca_type(mv2_arch_hca_type arch_hca);

char* mv2_get_hca_name(mv2_hca_type hca_type);
char* mv2_get_arch_name(mv2_arch_type arch_type);
char *mv2_get_cpu_family_name(mv2_cpu_family_type cpu_family_type);

#if defined(_SMP_LIMIC_)
/*Detecting number of cores in a socket, and number of sockets*/
void hwlocSocketDetection(int print_details);

/*Returns the socket where the process is bound*/
int getProcessBinding(pid_t pid);

/*Returns the number of cores in the socket*/
int numOfCoresPerSocket(int socket);

/*Returns the total number of sockets within the node*/
int numofSocketsPerNode(void);

/*Return socket bind to */
int get_socket_bound(void);
#endif /* defined(_SMP_LIMIC_) */

#endif /*  #ifndef MV2_ARCH_HCA_DETECT_H */

