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
#include <unistd.h>
#include <infiniband/verbs.h>
/*#include <infiniband/umad.h>*/

#include "sysreport.h"

#include "rdma_impl.h"
#include "upmi.h"
#include "ibv_param.h"

/**
 * Flag to enable the system reporting. Must be checked before the call
 * to mv2_system_report.
 */
int enable_sysreport = 0;

/**
 * Proc file to read memory info.
 */
#define PROCMEM_FILENAME        "/proc/meminfo"

/**
 * Print information about the system memory.
 */
static int mem_info( int rank ) {
  int mpi_errno = MPI_SUCCESS;
  char key[256];
  char value[256];
  unsigned int uivalue;

  FILE *fin=fopen( PROCMEM_FILENAME, "r" );
  while (!feof(fin))
  {
    fscanf(fin, "%255s %u%255[^\n]", key, &uivalue, value);
    if (strcmp(key, "MemTotal:") ==0) {
        printf("<memtotal rank='%d'>%u</memtotal>\n", rank, uivalue);
    } else if (strcmp(key, "MemFree:") ==0) {
        printf("<memfree rank='%d'>%u</memtotal>\n", rank, uivalue);
    }
  }
  fclose(fin);
  return mpi_errno;
}
/*
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
*/

/**
 * Check the status of Infiniband devices, and print a report.
 */
static int hca_check(int rank) {
    int mpi_errno = MPI_SUCCESS;
    int hcan;
    /* char names[UMAD_MAX_DEVICES][UMAD_CA_NAME_LEN]; */
    struct ibv_device **dev_list = NULL;
    int num_devices;
    dev_list = ibv_get_device_list(&num_devices);
  /*    umad_init(); */


    for (hcan=0; hcan<rdma_num_hcas; hcan++) {
        /* umad_ca_t ca; */
        char *ca_name;
        ca_name = (char *)ibv_get_device_name(dev_list[hcan]);
        printf ("device_name: %s\n",ca_name);
        /*
        ret = umad_get_ca(ca_name, &ca);
        if (ret<0) {
            mpi_errno = 1;
        }

        printf( "<hca rank='%d' num='%d' name='%s' fw='%s' hw='%s' ports='%d' />", \
                rank, hcan, ca_name, ca.fw_ver, ca.hw_ver, ca.numports );

        for (p=0; p<ca.numports; p++) {
            umad_port_t *port = ca.ports[p+1];
            if (port!=0) {
                printf( "<hcaport rank='%d' hca='%d' port='%d' state='%s (%d)'/>\n", rank, hcan, port->portnum,
                        (uint)port->state <= 4 ? port_state_str[port->state] : "???", (uint)port->state);
            }
        }
        umad_release_ca( &ca );
       */
    }

    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
    return mpi_errno;
}


/**
 * Check the system and produce a report.
 */
int mv2_system_report(void) {
    int mpi_errno = MPI_SUCCESS;
    char hostname[HOST_NAME_MAX];
    int rank;

    /* Rank and hostname */
    UPMI_GET_RANK(&rank);
    gethostname(hostname, HOST_NAME_MAX);
    printf( "<proc rank='%d' hostname='%s' />\n", rank, hostname );

    errno &= mem_info( rank );
    errno &= hca_check( rank );
    PMPI_Barrier( MPI_COMM_WORLD );

    return mpi_errno;
}
