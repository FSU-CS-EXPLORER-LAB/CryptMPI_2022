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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mpichconf.h"

#if defined(HAVE_LIBIBUMAD)
#include <infiniband/umad.h>
#endif

#include "mv2_arch_hca_detect.h"
#include "upmi.h"
#include "debug_utils.h"

#include "upmi.h"
#include "mpi.h"
#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
#include "rdma_impl.h"
#include "mv2_mpit_cvars.h"
#endif

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : FORCE_HCA_TYPE
      category    : CH3
      type        : int
      default     : 0
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This parameter forces the HCA type.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

int mv2_suppress_hca_warnings = 0;
extern int g_mv2_num_cpus;
static mv2_multirail_info_type g_mv2_multirail_info = mv2_num_rail_unknown;

#define MV2_STR_MLX          "mlx"
#define MV2_STR_MLX4         "mlx4"
#define MV2_STR_MLX5         "mlx5"
#define MV2_STR_MTHCA        "mthca"
#define MV2_STR_IPATH        "ipath"
#define MV2_STR_QIB          "qib"
#define MV2_STR_HFI1         "hfi1"
#define MV2_STR_EHCA         "ehca"
#define MV2_STR_CXGB3        "cxgb3"
#define MV2_STR_CXGB4        "cxgb4"
#define MV2_STR_NES0         "nes0"
#define MV2_STR_QEDR         "qedr"

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
MPI_T_cvar_handle mv2_force_hca_type_handle = NULL;
extern int mv2_set_force_hca_type();
extern void mv2_free_hca_handle ();
void mv2_free_hca_handle () {
    if (mv2_force_hca_type_handle) {
        MPIU_Free(mv2_force_hca_type_handle);
        mv2_force_hca_type_handle = NULL;
    }
}
#endif

typedef struct _mv2_hca_types_log_t{
    mv2_hca_type hca_type;
    char *hca_name;
}mv2_hca_types_log_t;

#define MV2_HCA_LAST_ENTRY MV2_HCA_LIST_END
static mv2_hca_types_log_t mv2_hca_types_log[] = 
{
    /*Unknown */
    {MV2_HCA_UNKWN,         "MV2_HCA_UNKWN"},

    /* Mellanox Cards */
    {MV2_HCA_MLX_PCI_EX_SDR,"MV2_HCA_MLX_PCI_EX_SDR"},
    {MV2_HCA_MLX_PCI_EX_DDR,"MV2_HCA_MLX_PCI_EX_DDR"},
    {MV2_HCA_MLX_CX_SDR,    "MV2_HCA_MLX_CX_SDR"},
    {MV2_HCA_MLX_CX_DDR,    "MV2_HCA_MLX_CX_DDR"},
    {MV2_HCA_MLX_CX_QDR,    "MV2_HCA_MLX_CX_QDR"},
    {MV2_HCA_MLX_CX_FDR,    "MV2_HCA_MLX_CX_FDR"},
    {MV2_HCA_MLX_CX_EDR,    "MV2_HCA_MLX_CX_EDR"},
    {MV2_HCA_MLX_CX_HDR,    "MV2_HCA_MLX_CX_HDR"},
    {MV2_HCA_MLX_CX_CONNIB, "MV2_HCA_MLX_CX_CONNIB"},
    {MV2_HCA_MLX_PCI_X,     "MV2_HCA_MLX_PCI_X"},

    /* Qlogic Cards */
    {MV2_HCA_QLGIC_PATH_HT, "MV2_HCA_QLGIC_PATH_HT"},
    {MV2_HCA_QLGIC_QIB,     "MV2_HCA_QLGIC_QIB"},

    /* IBM Cards */
    {MV2_HCA_IBM_EHCA,      "MV2_HCA_IBM_EHCA"},

    /* Intel Cards */
    {MV2_HCA_INTEL_HFI1,    "MV2_HCA_INTEL_HFI1"},
    
    /* Chelsio Cards */
    {MV2_HCA_CHELSIO_T3,    "MV2_HCA_CHELSIO_T3"},
    {MV2_HCA_CHELSIO_T4,    "MV2_HCA_CHELSIO_T4"},

    /* Intel iWarp Cards */
    {MV2_HCA_INTEL_NE020,   "MV2_HCA_INTEL_NE020"},

    /* Marvel RoCE Cards */
    {MV2_HCA_MARVEL_QEDR,   "MV2_HCA_MARVEL_QEDR"},

    /* Last Entry */
    {MV2_HCA_LAST_ENTRY,    "MV2_HCA_LAST_ENTRY"},
};

char* mv2_get_hca_name(mv2_hca_type hca_type)
{
    int i=0;
    if (hca_type == MV2_HCA_ANY) {
        return("MV2_HCA_ANY");
    }
    while(mv2_hca_types_log[i].hca_type != MV2_HCA_LAST_ENTRY){

        if(mv2_hca_types_log[i].hca_type == hca_type){
            return(mv2_hca_types_log[i].hca_name);
        }
        i++;
    }
    return("MV2_HCA_UNKWN");
}

#if defined(HAVE_LIBIBUMAD)
static int get_rate(umad_ca_t *umad_ca)
{
    int i;
    char *value;

    if ((value = getenv("MV2_DEFAULT_PORT")) != NULL) {
        int default_port = atoi(value);
        
        if(default_port <= umad_ca->numports){
            if (IBV_PORT_ACTIVE == umad_ca->ports[default_port]->state) {
                return umad_ca->ports[default_port]->rate;
            }
        }
    }

    for (i = 1; i <= umad_ca->numports; i++) {
        if (IBV_PORT_ACTIVE == umad_ca->ports[i]->state) {
            return umad_ca->ports[i]->rate;
        }
    }    
    return 0;
}
#endif

static const int get_link_width(uint8_t width)
{
    switch (width) {
    case 1:  return 1;
    case 2:  return 4;
    case 4:  return 8;
    case 8:  return 12;
    /* Links on Frontera are returning 16 as link width for now.
     * This is a temporary work around for that. */
    case 16:  return 2;
    default:
        PRINT_ERROR("Invalid link width %u\n", width);
        return 0;
    }
}

static const float get_link_speed(uint8_t speed)
{
    switch (speed) {
    case 1:  return 2.5;  /* SDR */
    case 2:  return 5.0;  /* DDR */

    case 4:  /* fall through */
    case 8:  return 10.0; /* QDR */

    case 16: return 14.0; /* FDR */
    case 32: return 25.0; /* EDR */
    case 64: return 50.0; /* EDR */
    default:
        PRINT_ERROR("Invalid link speed %u\n", speed);
        return 0;    /* Invalid speed */
    }
}

int mv2_check_hca_type(mv2_hca_type type, int rank)
{
    if (type <= MV2_HCA_LIST_START        || type >= MV2_HCA_LIST_END        ||
        type == MV2_HCA_IB_TYPE_START     || type == MV2_HCA_IB_TYPE_END     ||
        type == MV2_HCA_MLX_START         || type == MV2_HCA_MLX_END         ||
        type == MV2_HCA_IWARP_TYPE_START  || type == MV2_HCA_IWARP_TYPE_END  ||
        type == MV2_HCA_CHLSIO_START      || type == MV2_HCA_CHLSIO_END      ||
        type == MV2_HCA_INTEL_IWARP_START || type == MV2_HCA_INTEL_IWARP_END ||
        type == MV2_HCA_QLGIC_START       || type == MV2_HCA_QLGIC_END       ||
        type == MV2_HCA_MARVEL_START       || type == MV2_HCA_MARVEL_END       ||
        type == MV2_HCA_INTEL_START       || type == MV2_HCA_INTEL_END) {

        PRINT_INFO((rank==0), "Wrong value specified for MV2_FORCE_HCA_TYPE\n");
        PRINT_INFO((rank==0), "Value must be greater than %d and less than %d \n",
                    MV2_HCA_LIST_START, MV2_HCA_LIST_END);
        PRINT_INFO((rank==0), "For IB Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_MLX_START, MV2_HCA_MLX_END);
        PRINT_INFO((rank==0), "For IBM Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_IBM_START, MV2_HCA_IBM_END);
        PRINT_INFO((rank==0), "For Intel IWARP Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_INTEL_IWARP_START, MV2_HCA_INTEL_IWARP_END);
        PRINT_INFO((rank==0), "For Chelsio IWARP Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_CHLSIO_START, MV2_HCA_CHLSIO_END);
        PRINT_INFO((rank==0), "For QLogic Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_QLGIC_START, MV2_HCA_QLGIC_END);
        PRINT_INFO((rank==0), "For Marvel Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_MARVEL_START, MV2_HCA_MARVEL_END);
        PRINT_INFO((rank==0), "For Intel Cards: Please enter value greater than %d and less than %d\n",
                    MV2_HCA_INTEL_START, MV2_HCA_INTEL_END);
        return 1;
    }
    return 0;
}

#if defined(HAVE_LIBIBVERBS)
mv2_hca_type mv2_new_get_hca_type(struct ibv_context *ctx,
                                    struct ibv_device *ib_dev,
                                    uint64_t *guid)
{
    int rate=0;
    int my_rank = -1;
    char *value = NULL;
    char *dev_name = NULL;
    struct ibv_device_attr device_attr;
    int max_ports = 0;
    mv2_hca_type hca_type = MV2_HCA_UNKWN;

    UPMI_GET_RANK(&my_rank);

    if ((value = getenv("MV2_SUPPRESS_HCA_WARNINGS")) != NULL) {
        mv2_suppress_hca_warnings = !!atoi(value);
    }

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
    int cvar_forced = mv2_set_force_hca_type();
    if (cvar_forced) {
        return mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
    }
#endif /*ENABLE_PVAR_MV2 && CHANNEL_MRAIL*/

    if ((value = getenv("MV2_FORCE_HCA_TYPE")) != NULL) {
        hca_type = atoi(value);
        int retval = mv2_check_hca_type(hca_type, my_rank);
        if (retval) {
            PRINT_INFO((my_rank==0), "Falling back to Automatic HCA detection\n");
            hca_type = MV2_HCA_UNKWN;
        } else {
            return hca_type;
        }
    }

    dev_name = (char*) ibv_get_device_name( ib_dev );

    if ((!dev_name) && !mv2_suppress_hca_warnings) {
        PRINT_INFO((my_rank==0), "**********************WARNING***********************\n");
        PRINT_INFO((my_rank==0), "Failed to automatically detect the HCA architecture.\n");
        PRINT_INFO((my_rank==0), "This may lead to subpar communication performance.\n");
        PRINT_INFO((my_rank==0), "****************************************************\n");
        return MV2_HCA_UNKWN;
    }

    memset(&device_attr, 0, sizeof(struct ibv_device_attr));
    if(!ibv_query_device(ctx, &device_attr)){
        max_ports = device_attr.phys_port_cnt;
        *guid = device_attr.node_guid;
    }

    if (!strncmp(dev_name, MV2_STR_MLX, 3)
        || !strncmp(dev_name, MV2_STR_MTHCA, 5)) {

        hca_type = MV2_HCA_MLX_PCI_X;

        int query_port = 1;
        struct ibv_port_attr port_attr;

        /* honor MV2_DEFAULT_PORT, if set */
        if ((value = getenv("MV2_DEFAULT_PORT")) != NULL) {

            int default_port = atoi(value);
            query_port = (default_port <= max_ports) ? default_port : 1;
        }

        if (!ibv_query_port(ctx, query_port, &port_attr)) {
            rate = (int) (get_link_width(port_attr.active_width)
                    * get_link_speed(port_attr.active_speed));
            PRINT_DEBUG(DEBUG_INIT_verbose, "rate : %d\n", rate);
        }
        /* mlx4, mlx5 */ 
        switch(rate) {
            case 200:
                hca_type = MV2_HCA_MLX_CX_HDR;
                break;

            case 100:
                hca_type = MV2_HCA_MLX_CX_EDR;
                break;

            case 56:
                hca_type = MV2_HCA_MLX_CX_FDR;
                break;

            case 40:
                hca_type = MV2_HCA_MLX_CX_QDR;
                break;

            case 20:
                hca_type = MV2_HCA_MLX_CX_DDR;
                break;

            case 10:
                hca_type = MV2_HCA_MLX_CX_SDR;
                break;

            default:
                hca_type = MV2_HCA_MLX_CX_FDR;
                break;
        }
        if (!strncmp(dev_name, MV2_STR_MLX5, 4) && rate == 56)
                hca_type = MV2_HCA_MLX_CX_CONNIB; 
    } else if(!strncmp(dev_name, MV2_STR_IPATH, 5)) {
        hca_type = MV2_HCA_QLGIC_PATH_HT;

    } else if(!strncmp(dev_name, MV2_STR_QIB, 3)) {
        hca_type = MV2_HCA_QLGIC_QIB;

    } else if(!strncmp(dev_name, MV2_STR_HFI1, 4)) {
        hca_type = MV2_HCA_INTEL_HFI1;

    } else if(!strncmp(dev_name, MV2_STR_EHCA, 4)) {
        hca_type = MV2_HCA_IBM_EHCA;

    } else if (!strncmp(dev_name, MV2_STR_CXGB3, 5)) {
        hca_type = MV2_HCA_CHELSIO_T3;

    } else if (!strncmp(dev_name, MV2_STR_CXGB4, 5)) {
        hca_type = MV2_HCA_CHELSIO_T4;

    } else if (!strncmp(dev_name, MV2_STR_NES0, 4)) {
        hca_type = MV2_HCA_INTEL_NE020;

    } else if (!strncmp(dev_name, MV2_STR_QEDR, 4)) {
        hca_type = MV2_HCA_MARVEL_QEDR;

    } else {
        hca_type = MV2_HCA_UNKWN;
    }    

    if ((hca_type == MV2_HCA_UNKWN) && !mv2_suppress_hca_warnings) {
        PRINT_INFO((my_rank==0), "**********************WARNING***********************\n");
        PRINT_INFO((my_rank==0), "Failed to automatically detect the HCA architecture.\n");
        PRINT_INFO((my_rank==0), "This may lead to subpar communication performance.\n");
        PRINT_INFO((my_rank==0), "****************************************************\n");
    }

    return hca_type;
}

mv2_hca_type mv2_get_hca_type( struct ibv_device *dev )
{
    int rate=0;
    char *value = NULL;
    char *dev_name;
    int my_rank = -1;
    mv2_hca_type hca_type = MV2_HCA_UNKWN;

    UPMI_GET_RANK(&my_rank);
    
    if ((value = getenv("MV2_SUPPRESS_HCA_WARNINGS")) != NULL) {
        mv2_suppress_hca_warnings = !!atoi(value);
    }
#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
    int cvar_forced = mv2_set_force_hca_type();
    if (cvar_forced) {
        return mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
    }
#endif /*ENABLE_PVAR_MV2 && CHANNEL_MRAIL*/

    if ((value = getenv("MV2_FORCE_HCA_TYPE")) != NULL) {
        hca_type = atoi(value);
        int retval = mv2_check_hca_type(hca_type, my_rank);
        if (retval) {
            PRINT_INFO((my_rank==0), "Falling back to Automatic HCA detection\n");
            hca_type = MV2_HCA_UNKWN;
        } else {
            return hca_type;
        }
    }

    dev_name = (char*) ibv_get_device_name( dev );

    if ((!dev_name) && !mv2_suppress_hca_warnings) {
        PRINT_INFO((my_rank==0), "**********************WARNING***********************\n");
        PRINT_INFO((my_rank==0), "Failed to automatically detect the HCA architecture.\n");
        PRINT_INFO((my_rank==0), "This may lead to subpar communication performance.\n");
        PRINT_INFO((my_rank==0), "****************************************************\n");
        return MV2_HCA_UNKWN;
    }

#ifdef HAVE_LIBIBUMAD
    static char last_name[UMAD_CA_NAME_LEN] = { '\0' };
    static mv2_hca_type last_type = MV2_HCA_UNKWN;
    if (!strncmp(dev_name, last_name, UMAD_CA_NAME_LEN)) {
        return last_type;
    } else {
        strncpy(last_name, dev_name, UMAD_CA_NAME_LEN);
    }
#endif /* #ifdef HAVE_LIBIBUMAD */

    if (!strncmp(dev_name, MV2_STR_MLX4, 4)
        || !strncmp(dev_name, MV2_STR_MLX5, 4) 
        || !strncmp(dev_name, MV2_STR_MTHCA, 5)) {

        hca_type = MV2_HCA_UNKWN;
#if !defined(HAVE_LIBIBUMAD)
        int query_port = 1;
        struct ibv_context *ctx= NULL;
        struct ibv_port_attr port_attr;


        ctx = ibv_open_device(dev);
        if (!ctx) {
            return MV2_HCA_UNKWN;
        }

        /* honor MV2_DEFAULT_PORT, if set */
        if ((value = getenv("MV2_DEFAULT_PORT")) != NULL) {

            int max_ports = 1;
            struct ibv_device_attr device_attr;
            int default_port = atoi(value);
            
            memset(&device_attr, 0, sizeof(struct ibv_device_attr));
            if(!ibv_query_device(ctx, &device_attr)){
                max_ports = device_attr.phys_port_cnt;
            }
            query_port = (default_port <= max_ports) ? default_port : 1;
        }
        
        if (!ibv_query_port(ctx, query_port, &port_attr) &&
            (port_attr.state == IBV_PORT_ACTIVE)) {
            rate = (int) (get_link_width(port_attr.active_width)
                    * get_link_speed(port_attr.active_speed));
            PRINT_DEBUG(DEBUG_INIT_verbose, "rate : %d\n", rate);
        }
#else
        umad_ca_t umad_ca;
        if (umad_init() < 0) {
            last_type = hca_type;
            return hca_type;
        }

        memset(&umad_ca, 0, sizeof(umad_ca_t));

        if (umad_get_ca(dev_name, &umad_ca) < 0) {
            last_type = hca_type;
            return hca_type;
        }

        if (!getenv("MV2_USE_RoCE")) {
            rate = get_rate(&umad_ca);
            if (!rate) {
                umad_release_ca(&umad_ca);
                umad_done();
                last_type = hca_type;
                return hca_type;
            }
        }

        umad_release_ca(&umad_ca);
        umad_done();

        if (!strncmp(dev_name, MV2_STR_MTHCA, 5)) {
            hca_type = MV2_HCA_MLX_PCI_X;


            if (!strncmp(umad_ca.ca_type, "MT25", 4)) {
                switch (rate) {
                    case 20:
                        hca_type = MV2_HCA_MLX_PCI_EX_DDR;
                        break;

                    case 10:
                        hca_type = MV2_HCA_MLX_PCI_EX_SDR;
                        break;

                    default:
                        hca_type = MV2_HCA_MLX_PCI_EX_SDR;
                        break;
                }

            } else if (!strncmp(umad_ca.ca_type, "MT23", 4)) {
                hca_type = MV2_HCA_MLX_PCI_X;

            } else {
                hca_type = MV2_HCA_MLX_PCI_EX_SDR; 
            }
        } else 
#endif
        { /* mlx4, mlx5 */ 
            switch(rate) {
                case 200:
                    hca_type = MV2_HCA_MLX_CX_HDR;
                    break;

                case 100:
                    hca_type = MV2_HCA_MLX_CX_EDR;
                    break;

                case 56:
                    hca_type = MV2_HCA_MLX_CX_FDR;
                    break;

                case 40:
                    hca_type = MV2_HCA_MLX_CX_QDR;
                    break;

                case 20:
                    hca_type = MV2_HCA_MLX_CX_DDR;
                    break;

                case 10:
                    hca_type = MV2_HCA_MLX_CX_SDR;
                    break;

                default:
                    hca_type = MV2_HCA_MLX_CX_SDR;
                    break;
            }
            if (!strncmp(dev_name, MV2_STR_MLX5, 4) && rate == 56)
                    hca_type = MV2_HCA_MLX_CX_CONNIB; 
        }

    } else if(!strncmp(dev_name, MV2_STR_IPATH, 5)) {
        hca_type = MV2_HCA_QLGIC_PATH_HT;

    } else if(!strncmp(dev_name, MV2_STR_QIB, 3)) {
        hca_type = MV2_HCA_QLGIC_QIB;

    } else if (!strncmp(dev_name, MV2_STR_HFI1, 4)) {
        hca_type = MV2_HCA_INTEL_HFI1;

    } else if(!strncmp(dev_name, MV2_STR_EHCA, 4)) {
        hca_type = MV2_HCA_IBM_EHCA;

    } else if (!strncmp(dev_name, MV2_STR_CXGB3, 5)) {
        hca_type = MV2_HCA_CHELSIO_T3;

    } else if (!strncmp(dev_name, MV2_STR_CXGB4, 5)) {
        hca_type = MV2_HCA_CHELSIO_T4;

    } else if (!strncmp(dev_name, MV2_STR_NES0, 4)) {
        hca_type = MV2_HCA_INTEL_NE020;

    } else if (!strncmp(dev_name, MV2_STR_QEDR, 4)) {
        hca_type = MV2_HCA_MARVEL_QEDR;

    } else {
        hca_type = MV2_HCA_UNKWN;
    }    
#ifdef HAVE_LIBIBUMAD
    last_type = hca_type;
#endif /* #ifdef HAVE_LIBIBUMAD */
    if ((hca_type == MV2_HCA_UNKWN) && !mv2_suppress_hca_warnings) {
        PRINT_INFO((my_rank==0), "**********************WARNING***********************\n");
        PRINT_INFO((my_rank==0), "Failed to automatically detect the HCA architecture.\n");
        PRINT_INFO((my_rank==0), "This may lead to subpar communication performance.\n");
        PRINT_INFO((my_rank==0), "****************************************************\n");
    }
    return hca_type;
}
#else
mv2_hca_type mv2_get_hca_type(void *dev)
{
    int my_rank = -1;
    char *value = NULL;
    mv2_hca_type hca_type = MV2_HCA_UNKWN;

    UPMI_GET_RANK(&my_rank);

    if ((value = getenv("MV2_SUPPRESS_HCA_WARNINGS")) != NULL) {
        mv2_suppress_hca_warnings = !!atoi(value);
    }
#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
    int cvar_forced = mv2_set_force_hca_type();
    if (cvar_forced) {
        return mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
    }
#endif /*ENABLE_PVAR_MV2 && CHANNEL_MRAIL*/

    if ((value = getenv("MV2_FORCE_HCA_TYPE")) != NULL) {
        hca_type = atoi(value);
        int retval = mv2_check_hca_type(hca_type, my_rank);
        if (retval) {
            PRINT_INFO((my_rank==0), "Falling back to Automatic HCA detection\n");
            hca_type = MV2_HCA_UNKWN;
        } else {
            return hca_type;
        }
    }

#ifdef HAVE_LIBPSM2
    hca_type = MV2_HCA_INTEL_HFI1;
#elif HAVE_LIBPSM_INFINIPATH
    hca_type = MV2_HCA_QLGIC_QIB;
#else
    hca_type = MV2_HCA_UNKWN;
#endif

    return hca_type;
}
#endif

#if defined(HAVE_LIBIBVERBS)
mv2_arch_hca_type mv2_new_get_arch_hca_type (mv2_hca_type hca_type)
{
    mv2_arch_hca_type arch_hca = mv2_get_arch_type();
    arch_hca = arch_hca << 16 | hca_type;
    arch_hca = arch_hca << 16 | (mv2_arch_num_cores) g_mv2_num_cpus;
    return arch_hca;
}

mv2_arch_hca_type mv2_get_arch_hca_type (struct ibv_device *dev)
{
    mv2_arch_hca_type arch_hca = mv2_get_arch_type();
    arch_hca = arch_hca << 16 | mv2_get_hca_type(dev);
    arch_hca = arch_hca << 16 | (mv2_arch_num_cores) g_mv2_num_cpus;
    return arch_hca;
}
#else 
mv2_arch_hca_type mv2_get_arch_hca_type (void *dev)
{
    mv2_arch_hca_type arch_hca = mv2_get_arch_type();
    arch_hca = arch_hca << 16 | mv2_get_hca_type(dev);
    arch_hca = arch_hca << 16 | (mv2_arch_num_cores) g_mv2_num_cpus;
    return arch_hca;
}
#endif

#if defined(HAVE_LIBIBVERBS)
mv2_multirail_info_type mv2_get_multirail_info()
{
    if ( mv2_num_rail_unknown == g_mv2_multirail_info ) {
        int num_devices;
        struct ibv_device **dev_list = NULL;

        /* Get the number of rails */
        dev_list = ibv_get_device_list(&num_devices);

        switch (num_devices){
            case 1:
                g_mv2_multirail_info = mv2_num_rail_1;
                break;
            case 2:
                g_mv2_multirail_info = mv2_num_rail_2;
                break;
            case 3:
                g_mv2_multirail_info = mv2_num_rail_3;
                break;
            case 4:
                g_mv2_multirail_info = mv2_num_rail_4;
                break;
            default:
                g_mv2_multirail_info = mv2_num_rail_unknown;
                break;
        }
        if (dev_list) {
            ibv_free_device_list(dev_list);
        }
    }
    return g_mv2_multirail_info;
}
#else
mv2_multirail_info_type mv2_get_multirail_info()
{
    return mv2_num_rail_unknown;
}

#endif

#if ENABLE_PVAR_MV2 && CHANNEL_MRAIL
int mv2_set_force_hca_type()
{
    int mpi_errno = MPI_SUCCESS;
    int cvar_index = 0;
    int skip_setting = 0;
    int read_value = 0;

    /* Get CVAR index by name */
    MPIR_CVAR_GET_INDEX_impl(MPIR_CVAR_FORCE_HCA_TYPE, cvar_index);
    if (cvar_index < 0) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
    mv2_mpit_cvar_access_t wrapper;
    wrapper.cvar_name = "MPIR_CVAR_FORCE_HCA_TYPE";
    wrapper.cvar_index = cvar_index;
    wrapper.cvar_handle = mv2_force_hca_type_handle;
    wrapper.default_cvar_value = MV2_HCA_UNKWN;
    wrapper.skip_if_default_has_set = 1;
    wrapper.error_type = MV2_CVAR_FATAL_ERR;
    wrapper.check4_associate_env_conflict = 1;
    wrapper.env_name = "MV2_FORCE_HCA_TYPE";
    wrapper.env_conflict_error_msg = "the CVAR will set up to default";
    wrapper.check_max = 1;
    wrapper.max_value = MV2_HCA_LIST_END-1;
    wrapper.check_min = 1;
    wrapper.min_value = MV2_HCA_LIST_START+1;
    wrapper.boundary_error_msg = "Wrong value specified for MPIR_CVAR_FORCE_HCA_TYPE";
    wrapper.skip = &skip_setting;
    wrapper.value = &read_value;
    mpi_errno = mv2_read_and_check_cvar(wrapper);
    if (mpi_errno != MPI_SUCCESS){
        goto fn_fail;
    }
    /* Choose algorithm based on CVAR */
    if (!skip_setting) {
        mv2_hca_type hca_type = read_value;
        int retval = mv2_check_hca_type(hca_type,  MPIDI_Process.my_pg_rank);
        if (retval) {
            PRINT_INFO( (MPIDI_Process.my_pg_rank==0), 
                "### cvar func### Falling back to Automatic HCA detection\n");
            hca_type = MV2_HCA_UNKWN;
        } 
        else {
            mv2_MPIDI_CH3I_RDMA_Process.hca_type = hca_type;
            mv2_arch_hca_type arch_hca = mv2_get_arch_type();
            mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type = 
                (((arch_hca << 16 | hca_type) << 16) | g_mv2_num_cpus);
            goto fn_change;
        }
    }

    fn_fail:
    fn_exit:
        return mpi_errno;
    fn_change:
        return 1;
}
#endif
