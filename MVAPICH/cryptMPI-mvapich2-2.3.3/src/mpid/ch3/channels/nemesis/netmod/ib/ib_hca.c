/*! \file */
/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

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

#include <infiniband/verbs.h>

#include "mpidimpl.h"
#include "mpidbg.h"
#include "upmi.h"

#include "ib_errors.h"
#include "ib_process.h"

#include "ib_hca.h"


/**
 * Define ENABLE_HCA_REPORT in order to print a report on the local HCAs status.
 * The function <code>MPID_nem_ib_hca_check</code> prints the report.
 *
 * #define ENABLE_HCA_REPORT
*/
#undef ENABLE_HCA_REPORT

/**
 *  Number of HCSs.
 *  (Was rdma_num_hcas).
 */
int ib_hca_num_hcas = 1;

/**
 *  Number of ports.
 *  (Was rdma_num_ports).
 */
int ib_hca_num_ports = 1;

/**
 * The list of the HCAs found in the system.
 */
MPID_nem_ib_nem_hca hca_list[MAX_NUM_HCAS];

/**
 * Check the ibv_port_attr and ibv_device_attr.
 */
static int check_attrs( struct ibv_port_attr *port_attr, struct ibv_device_attr *dev_attr)
{
    int ret = 0;
#ifdef _ENABLE_XRC_
    if (USE_XRC && !(dev_attr->device_cap_flags & IBV_DEVICE_XRC)) {
        fprintf (stderr, "HCA does not support XRC. Disable MV2_USE_XRC.\n");
        ret = 1;
    }
#endif /* _ENABLE_XRC_ */
    if(port_attr->active_mtu < rdma_default_mtu) {
    	MPL_error_printf( "Active MTU is %d, MV2_DEFAULT_MTU set to %d. See User Guide\n",
                port_attr->active_mtu, rdma_default_mtu);
        ret = 1;
    }

    if(dev_attr->max_qp_rd_atom < rdma_default_qp_ous_rd_atom) {
    	MPL_error_printf( "Max MV2_DEFAULT_QP_OUS_RD_ATOM is %d, set to %d\n",
                dev_attr->max_qp_rd_atom, rdma_default_qp_ous_rd_atom);
        ret = 1;
    }

    if(process_info.has_srq) {
        if(dev_attr->max_srq_sge < rdma_default_max_sg_list) {
        	MPL_error_printf( "Max MV2_DEFAULT_MAX_SG_LIST is %d, set to %d\n",
                    dev_attr->max_srq_sge, rdma_default_max_sg_list);
            ret = 1;
        }

        if(dev_attr->max_srq_wr < mv2_srq_alloc_size) {
        	MPL_error_printf( "Max MV2_SRQ_SIZE is %d, set to %d\n",
                    dev_attr->max_srq_wr, (int) mv2_srq_alloc_size);
            ret = 1;
        }
    } else {
        if(dev_attr->max_sge < rdma_default_max_sg_list) {
        	MPL_error_printf( "Max MV2_DEFAULT_MAX_SG_LIST is %d, set to %d\n",
                    dev_attr->max_sge, rdma_default_max_sg_list);
            ret = 1;
        }

        if(dev_attr->max_qp_wr < rdma_default_max_send_wqe) {
        	MPL_error_printf( "Max MV2_DEFAULT_MAX_SEND_WQE is %d, set to %d\n",
                    dev_attr->max_qp_wr, (int) rdma_default_max_send_wqe);
            ret = 1;
        }
    }
    if(dev_attr->max_cqe < rdma_default_max_cq_size) {
    	MPL_error_printf( "Max MV2_DEFAULT_MAX_CQ_SIZE is %d, set to %d\n",
                dev_attr->max_cqe, (int) rdma_default_max_cq_size);
        ret = 1;
    }

    return ret;
}

/*
 * Function: rdma_find_active_port
 *
 * Description:
 *      Finds if the given device has any active ports.
 *
 * Input:
 *      context -   Pointer to the device context obtained by opening device.
 *      ib_dev  -   Pointer to the device from ibv_get_device_list.
 *
 * Return:
 *      Success:    Port number of the active port.
 *      Failure:    ERROR (-1).
 */
static int rdma_find_active_port(struct ibv_context *context,struct ibv_device *ib_dev)
{
    int j = 0;
    const char *dev_name = NULL;
    struct ibv_port_attr port_attr;

    if (NULL == ib_dev) {
        return -1;
    } else {
        dev_name = ibv_get_device_name(ib_dev);
    }

    for (j = 1; j <= RDMA_DEFAULT_MAX_PORTS; ++ j) {
        if ((! ibv_query_port(context, j, &port_attr)) &&
             port_attr.state == IBV_PORT_ACTIVE) {
            if (!strncmp(dev_name, "cxgb3", 5) || !strncmp(dev_name, "cxgb4", 5)
                || port_attr.lid) {
                /* Chelsio RNIC's don't get LID's as they're not IB devices.
                 * So dont do this check for them.
                 */
                DEBUG_PRINT("Active port number = %d, state = %s, lid = %d\r\n",
                    j, (port_attr.state==IBV_PORT_ACTIVE)?"Active":"Not Active",
                    port_attr.lid);
                return j;
            }
        }
    }

    return -1;
}


#ifdef ENABLE_HCA_REPORT
static char *port_state_str[] = {
        "???",
        "Down",
        "Initializing",
        "Armed",
        "Active"
};

static char *port_phy_state_str[] = {
        "No state change",
        "Sleep",
        "Polling",
        "Disabled",
        "PortConfigurationTraining",
        "LinkUp",
        "LinkErrorRecovery",
        "PhyTest"
};
#endif



#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_init_hca
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * Initialize the HCAs
 * Look at rdma_open_hca() & rdma_iba_hca_init_noqp() in
 * mvapich2/trunk/src/mpid/ch3/channels/mrail/src/gen2/rdma_iba_priv.c
 *
 * Store all the HCA info in mv2_nem_dev_info_t->hca[hca_num]
 *
 * Output:
 *         hca_list: fill it with the HCAs information
 *
 * \see hca_list
 */
int MPID_nem_ib_init_hca()
{
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_INIT_HCA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_INIT_HCA);


    struct ibv_device *ib_dev    = NULL;
    struct ibv_device **dev_list = NULL;
    int nHca;
    int num_devices = 0;

#ifdef CRC_CHECK
    gen_crc_table();
#endif
    memset( hca_list, 0, sizeof(hca_list) );

    /* Get the list of devices */
    dev_list = ibv_get_device_list(&num_devices);
    if (dev_list==NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
	            "**fail %s", "No IB device found");
    }

    /* Runtime checks */
    MPIU_Assert( num_devices<=MAX_NUM_HCAS );
    if ( num_devices> MAX_NUM_HCAS) {
        MPL_error_printf( "WARNING: found %d IB devices, the maximum is %d (MAX_NUM_HCAS). ",
        		num_devices, MAX_NUM_HCAS);
        num_devices = MAX_NUM_HCAS;
    }

    if ( ib_hca_num_hcas > num_devices) {
    	MPL_error_printf( "WARNING: user requested %d IB devices, the available number is %d. ",
        		ib_hca_num_hcas, num_devices);
        ib_hca_num_hcas = num_devices;
    }

    MPIU_DBG_MSG_P( CH3_CHANNEL, VERBOSE, "[HCA] Found %d HCAs\n", num_devices);
    MPIU_DBG_MSG_P( CH3_CHANNEL, VERBOSE, "[HCA] User requested %d\n", ib_hca_num_hcas);


    /* Retrieve information for each found device */
    for (nHca = 0; nHca < ib_hca_num_hcas; nHca++) {

    	/* Check for user choice */
        if( (rdma_iba_hca[0]==0) || (strncmp(rdma_iba_hca, RDMA_IBA_NULL_HCA, 32)==0) || (ib_hca_num_hcas > 1)) {
            /* User hasn't specified any HCA name, or the number of HCAs is greater then 1 */
            ib_dev = dev_list[nHca];

        } else {
            /* User specified a HCA, try to look for it */
            int dev_count;

            dev_count = 0;
            while(dev_list[dev_count]) {
                if(!strncmp(ibv_get_device_name(dev_list[dev_count]), rdma_iba_hca, 32)) {
                    ib_dev = dev_list[dev_count];
                    break;
                }
                dev_count++;
            }
        }

        /* Check if device has been identified */
        hca_list[nHca].ib_dev = ib_dev;
        if (!ib_dev) {
	        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
		            "**fail %s", "No IB device found");
        }

        MPIU_DBG_MSG_P( CH3_CHANNEL, VERBOSE, "[HCA] HCA device %d found\n", nHca);



        hca_list[nHca].nic_context = ibv_open_device(ib_dev);
        if (hca_list[nHca].nic_context==NULL) {
	        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER, "**fail",
		            "%s %d", "Failed to open HCA number", nHca);
        }

        hca_list[nHca].ptag = ibv_alloc_pd(hca_list[nHca].nic_context);
        if (!hca_list[nHca].ptag) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER,
                    "**fail", "%s%d", "Failed to alloc pd number ", nHca);
        }


        /* Set the hca type */
    #if defined(RDMA_CM)
        if (process_info.use_iwarp_mode) {
    	    if ((mpi_errno = rdma_cm_get_hca_type(process_info.use_iwarp_mode, &process_info.hca_type)) != MPI_SUCCESS)
    	    {
    		    MPIR_ERR_POP(mpi_errno);
    	    }

    	    if (process_info.hca_type == CHELSIO_T3)
    	    {
    		    process_info.use_iwarp_mode = 1;
    	    }
        }
        else
    #endif /* defined(RDMA_CM) */
        {
            process_info.hca_type = hca_list[nHca].hca_type = mv2_get_hca_type(hca_list[nHca].ib_dev);
            process_info.arch_hca_type = mv2_get_arch_hca_type(hca_list[nHca].ib_dev);
        }
    }



    if (!strncmp(rdma_iba_hca, RDMA_IBA_NULL_HCA, 32) &&
        (ib_hca_num_hcas==1) && (num_devices > nHca) &&
        (rdma_find_active_port(hca_list[0].nic_context, hca_list[nHca].ib_dev)==-1)) {
        /* Trac #376 - There are multiple rdma capable devices (num_devices) in
         * the system. The user has asked us to use ANY (!strncmp) ONE device
         * (rdma_num_hcas), and the first device does not have an active port. So
         * try to find some other device with an active port.
         */
    	int j;
        for (j = 0; dev_list[j]; j++) {
            ib_dev = dev_list[j];
            if (ib_dev) {
            	hca_list[0].nic_context = ibv_open_device(ib_dev);
                if (!hca_list[0].nic_context) {
                    /* Go to next device */
                    continue;
                }
                if (rdma_find_active_port(hca_list[0].nic_context, ib_dev)!=-1) {
                	hca_list[0].ib_dev = ib_dev;
                	hca_list[0].ptag = ibv_alloc_pd(hca_list[0].nic_context);
                    if (!hca_list[0].ptag) {
                        MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER,
                             "**fail", "%s%d", "Failed to alloc pd number ", nHca);
                    }
                }
            }
        }
    }

fn_exit:
    /* Clean up before exit */
	if (dev_list!=NULL)
	  ibv_free_device_list(dev_list);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_INIT_HCA);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


/**
 * Create a srq using process info data.
 */
struct ibv_srq *create_srq(int hca_num)
{
    struct ibv_srq_init_attr srq_init_attr;
    struct ibv_srq *srq_ptr = NULL;

    memset(&srq_init_attr, 0, sizeof(srq_init_attr));

    srq_init_attr.srq_context    = hca_list[hca_num].nic_context;
    srq_init_attr.attr.max_wr    = mv2_srq_alloc_size;
    srq_init_attr.attr.max_sge   = 1;
    /* The limit value should be ignored during SRQ create */
    srq_init_attr.attr.srq_limit = mv2_srq_limit;

    srq_ptr = ibv_create_srq(hca_list[hca_num].ptag, &srq_init_attr);

    if (!srq_ptr) {
        ibv_error_abort(-1, "Error creating SRQ\n");
    }

    return srq_ptr;
}


#undef FUNCNAME
#define FUNCNAME MPID_nem_ib_open_ports
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/**
 * the first step in original MPID_nem_ib_setup_conn() function
 * open hca, create ptags  and create cqs
 */
int MPID_nem_ib_open_ports()
{
    int mpi_errno = MPI_SUCCESS;

    /* Infiniband Verb Structures */
    struct ibv_port_attr    port_attr;
    struct ibv_device_attr  dev_attr;

    int nHca; /* , curRank, rail_index ; */

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_OPEN_HCA);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_OPEN_HCA);

    for (nHca = 0; nHca < ib_hca_num_hcas; nHca++) {
        if (ibv_query_device(hca_list[nHca].nic_context, &dev_attr)) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "**fail %s", "Error getting HCA attributes");
        }

        /* detecting active ports */
        if (rdma_default_port < 0 || ib_hca_num_ports > 1) {
            int nPort;
            int k = 0;
            for (nPort = 1; nPort <= RDMA_DEFAULT_MAX_PORTS; nPort ++) {
                if ((! ibv_query_port(hca_list[nHca].nic_context, nPort, &port_attr)) &&
                            port_attr.state == IBV_PORT_ACTIVE &&
                            (port_attr.lid || (!port_attr.lid && use_iboeth))) {
                    if (use_iboeth) {
                        if (ibv_query_gid(hca_list[nHca].nic_context,
                                        nPort, 0, &hca_list[nHca].gids[k])) {
                            /* new error information function needed */
                            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                    "**fail", "Failed to retrieve gid on rank %d", process_info.rank);
                        }
                        DEBUG_PRINT("[%d] %s(%d): Getting gid[%d][%d] for"
                                " port %d subnet_prefix = %llx,"
                                " intf_id = %llx\r\n",
                                process_info.rank, __FUNCTION__, __LINE__, nHca, k, k,
                                hca_list[nHca].gids[k].global.subnet_prefix,
                                hca_list[nHca].gids[k].global.interface_id);
                    } else {
                        hca_list[nHca].lids[k]    = port_attr.lid;
                    }
                    hca_list[nHca].ports[k++] = nPort;

                    if (check_attrs(&port_attr, &dev_attr)) {
                        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                "**fail", "**fail %s",
                                "Attributes failed sanity check");
                    }
                }
            }
            if (k < ib_hca_num_ports) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**activeports", "**activeports %d", ib_hca_num_ports);
            }
        } else {
            if(ibv_query_port(hca_list[nHca].nic_context,
                        rdma_default_port, &port_attr)
                || (!port_attr.lid && !use_iboeth)
                || (port_attr.state != IBV_PORT_ACTIVE)) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                        "**portquery", "**portquery %d", rdma_default_port);
            }

            hca_list[nHca].ports[0] = rdma_default_port;

            if (use_iboeth) {
                if (ibv_query_gid(hca_list[nHca].nic_context, 0, 0, &hca_list[nHca].gids[0])) {
                    /* new error function needed */
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                            "**fail", "Failed to retrieve gid on rank %d", process_info.rank);
                }

                if (check_attrs(&port_attr, &dev_attr)) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                            "**fail", "**fail %s", "Attributes failed sanity check");
                }
            } else {
                hca_list[nHca].lids[0]  = port_attr.lid;
            }
        }

        if (rdma_use_blocking) {
            hca_list[nHca].comp_channel = ibv_create_comp_channel(hca_list[nHca].nic_context);

            if (!hca_list[nHca].comp_channel) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                        "**fail", "**fail %s", "cannot create completion channel");
            }

            hca_list[nHca].send_cq_hndl = NULL;
            hca_list[nHca].recv_cq_hndl = NULL;
            hca_list[nHca].cq_hndl = ibv_create_cq(hca_list[nHca].nic_context,
                    rdma_default_max_cq_size, NULL, hca_list[nHca].comp_channel, 0);
            if (!hca_list[nHca].cq_hndl) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                        "**fail", "**fail %s", "cannot create cq");
            }

            if (ibv_req_notify_cq(hca_list[nHca].cq_hndl, 0)) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                        "**fail", "**fail %s", "cannot request cq notification");
            }
        } else {
            /* Allocate the completion queue handle for the HCA */
            hca_list[nHca].send_cq_hndl = NULL;
            hca_list[nHca].recv_cq_hndl = NULL;

            hca_list[nHca].cq_hndl = ibv_create_cq(hca_list[nHca].nic_context,
                    rdma_default_max_cq_size, NULL, NULL, 0);
            if (!hca_list[nHca].cq_hndl) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                        "**fail", "**fail %s", "cannot create cq");
            }
        }

        /* to decouple process_info, may need to store has_srq to hca structure??? */
        if (process_info.has_srq) {
            hca_list[nHca].srq_hndl = create_srq(nHca);
        }
    }

    rdma_default_port       = hca_list[0].ports[0];

    fn_exit:

        MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_OPEN_HCA);
        return mpi_errno;

    fn_fail:
        goto fn_exit;
}

