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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "debug_utils.h"

// Prefix to distinguish output from different processes
#define OUTPUT_PREFIX_LENGTH 256
char output_prefix[OUTPUT_PREFIX_LENGTH] = "";

void set_output_prefix( char* prefix ) {
    strncpy( output_prefix, prefix, OUTPUT_PREFIX_LENGTH );
    output_prefix[OUTPUT_PREFIX_LENGTH-1]= '\0';
}

const char *get_output_prefix() {
    return output_prefix;
}


// Verbosity level for sharp  operations in collectives
int DEBUG_Sharp_verbose = 0;

// Verbosity level for fork/kill/waitpid operations in mpirun_rsh and mpispawn
int DEBUG_Fork_verbose = 0;

// Verbosity level for Fault Tolerance operations
int DEBUG_FT_verbose = 0;

// Verbosity level for Checkpoint/Restart operations
int DEBUG_CR_verbose = 0;

// Verbosity level for Migration operations
int DEBUG_MIG_verbose = 0;

// Verbosity level for UD flow control
int DEBUG_UD_verbose = 0;

// Verbosity level for UD ZCOPY Rndv
int DEBUG_ZCY_verbose = 0;

// Verbosity level for On-Demand Connection Management
int DEBUG_CM_verbose = 0;

// Verbosity level for XRC.
int DEBUG_XRC_verbose = 0;

// Verbosity level for UD stats
int DEBUG_UDSTAT_verbose = 0;

// Verbosity level for memory stats
int DEBUG_MEM_verbose = 0;

// Verbosity level for GPU CUDA
int DEBUG_CUDA_verbose = 0;

// Verbosity level for IB MULTICAST
int DEBUG_MCST_verbose = 0;

// Verbosity level for SHMEM Collectives
int DEBUG_SHM_verbose;

// Verbosity level for Channel manager
int DEBUG_CHM_verbose;

// Verbosity level for RNDV transfers
int DEBUG_RNDV_verbose;

// Verbosity level for Init phase
int DEBUG_INIT_verbose;

// Verbosity level for RDMA_CM
int DEBUG_RDMACM_verbose;

// Verbosity level for One-sided
int DEBUG_1SC_verbose;

// Verbosity level for dreg cache
int DEBUG_DREG_verbose;

static inline int env2int (char *name)
{
    char* env_str = getenv( name );
    if ( env_str == NULL ) {
        return 0;
    } else {
        return atoi( env_str );
    }
}


// Initialize the verbosity level of the above variables
int initialize_debug_variables() {
    DEBUG_Sharp_verbose = env2int( "MV2_DEBUG_SHARP_VERBOSE" );
    DEBUG_Fork_verbose = env2int( "MV2_DEBUG_FORK_VERBOSE" );
    DEBUG_FT_verbose = env2int( "MV2_DEBUG_FT_VERBOSE" );
    DEBUG_CR_verbose = env2int( "MV2_DEBUG_CR_VERBOSE" );
    DEBUG_MIG_verbose = env2int( "MV2_DEBUG_MIG_VERBOSE" );
    DEBUG_UD_verbose = env2int( "MV2_DEBUG_UD_VERBOSE" );
    DEBUG_ZCY_verbose = env2int( "MV2_DEBUG_ZCOPY_VERBOSE" );
    DEBUG_CM_verbose = env2int( "MV2_DEBUG_CM_VERBOSE" );
    DEBUG_XRC_verbose = env2int( "MV2_DEBUG_XRC_VERBOSE" );
    DEBUG_UDSTAT_verbose = env2int( "MV2_DEBUG_UDSTAT_VERBOSE" );
    DEBUG_MEM_verbose = env2int( "MV2_DEBUG_MEM_USAGE_VERBOSE" );
    DEBUG_CUDA_verbose = env2int( "MV2_DEBUG_CUDA_VERBOSE" );
    DEBUG_MCST_verbose = env2int( "MV2_DEBUG_MCST_VERBOSE" );
    DEBUG_SHM_verbose = env2int( "MV2_DEBUG_SHM_VERBOSE" );
    DEBUG_CHM_verbose = env2int( "MV2_DEBUG_CHM_VERBOSE" );
    DEBUG_RNDV_verbose = env2int( "MV2_DEBUG_RNDV_VERBOSE" );
    DEBUG_INIT_verbose = env2int( "MV2_DEBUG_INIT_VERBOSE" );
    DEBUG_RDMACM_verbose = env2int( "MV2_DEBUG_RDMACM_VERBOSE" );
    DEBUG_1SC_verbose = env2int( "MV2_DEBUG_1SC_VERBOSE" );
    DEBUG_DREG_verbose = env2int( "MV2_DEBUG_DREG_VERBOSE" );
    return 0;
}

void mv2_print_mem_usage()
{
    FILE *file = fopen ("/proc/self/status", "r");
    char vmpeak[100], vmhwm[100];

    if ( file != NULL ) {
        char line[100];
        while (fgets(line, 100, file) != NULL) {
            if (strstr(line, "VmPeak") != NULL) {
                strcpy(vmpeak, line);
                vmpeak[strcspn(vmpeak, "\n")] = '\0';
            }
            if (strstr(line, "VmHWM") != NULL) {
                strcpy(vmhwm, line);
                vmhwm[strcspn(vmhwm, "\n")] = '\0';
            }
        }
        PRINT_INFO(DEBUG_MEM_verbose, "%s %s\n", vmpeak, vmhwm);
        fclose(file);
    } else {
        PRINT_INFO(DEBUG_MEM_verbose, "Status file could not be opened \n");
    }
}

#ifdef _OSU_MVAPICH_
inline void dump_device_cap(struct ibv_device_attr dev_attr)
{
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported QPs                                                               : %6d\n", dev_attr.max_qp);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of outstanding WR on any work queue                                            : %6d\n", dev_attr.max_qp_wr);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of s/g per WR for non-RD QPs                                                   : %6d\n", dev_attr.max_sge);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of s/g per WR for RD QPs                                                       : %6d\n", dev_attr.max_sge_rd);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported CQs                                                               : %6d\n", dev_attr.max_cq);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of CQE capacity per CQ                                                         : %6d\n", dev_attr.max_cqe);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported MRs                                                               : %6d\n", dev_attr.max_mr);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported PDs                                                               : %6d\n", dev_attr.max_pd);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of RDMA Read & Atomic operations that can be outstanding per QP                : %6d\n", dev_attr.max_qp_rd_atom);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of RDMA Read & Atomic operations that can be outstanding per EEC               : %6d\n", dev_attr.max_ee_rd_atom);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of resources used for RDMA Read & Atomic operations by this HCA as the Target  : %6d\n", dev_attr.max_res_rd_atom);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum depth per QP for initiation of RDMA Read & Atomic operations                          : %6d\n", dev_attr.max_qp_init_rd_atom);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum depth per EEC for initiation of RDMA Read & Atomic operations                         : %6d\n", dev_attr.max_ee_init_rd_atom);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Atomic operations support level                                                               : %s\n",
                (dev_attr.atomic_cap == IBV_ATOMIC_NONE)?"No Support":
                (dev_attr.atomic_cap == IBV_ATOMIC_HCA)?"HCA Level":
                (dev_attr.atomic_cap == IBV_ATOMIC_GLOB)?"Node Level":"Un-known");
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported EE contexts                                                       : %6d\n", dev_attr.max_ee);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported RD domains                                                        : %6d\n", dev_attr.max_rdd);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported MWs                                                               : %6d\n", dev_attr.max_mw);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported raw IPv6 datagram QPs                                             : %6d\n", dev_attr.max_raw_ipv6_qp);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported Ethertype datagram QPs                                            : %6d\n", dev_attr.max_raw_ethy_qp);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported multicast groups                                                  : %6d\n", dev_attr.max_mcast_grp);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of QPs per multicast group which can be attached                               : %6d\n", dev_attr.max_mcast_qp_attach);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported address handles                                                   : %6d\n", dev_attr.max_ah);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of supported FMRs                                                              : %6d\n", dev_attr.max_fmr);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of (re)maps per FMR before an unmap operation in required                      : %6d\n", dev_attr.max_map_per_fmr);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of supported SRQs                                                              : %6d\n", dev_attr.max_srq);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of WRs per SRQ                                                                 : %6d\n", dev_attr.max_srq_wr);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of s/g per SRQ                                                                 : %6d\n", dev_attr.max_srq_sge);
    PRINT_DEBUG(DEBUG_INIT_verbose>1, "Maximum number of partitions                                                                  : %6d\n", dev_attr.max_pkeys);
    PRINT_DEBUG(DEBUG_INIT_verbose>0, "Maximum number of QPs which can be attached to multicast groups                               : %6d\n", dev_attr.max_total_mcast_qp_attach);
}
#endif
