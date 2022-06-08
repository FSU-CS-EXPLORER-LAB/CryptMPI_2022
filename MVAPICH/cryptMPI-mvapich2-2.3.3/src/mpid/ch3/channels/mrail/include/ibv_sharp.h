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

#ifndef _IBV_SHARP_H_
#define _IBV_SHARP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include "debug_utils.h"
#include "api/sharp_coll.h"
#include <mpiimpl.h>

struct coll_sharp_module_t {
    struct sharp_coll_comm *sharp_coll_comm;
    int         is_leader;
    int         ppn;
    MPI_Comm    comm;
};
typedef struct coll_sharp_module_t coll_sharp_module_t;

struct sharp_conf {
    int         rank;
    int         size;
    int         ib_port;
    char        *hostlist; 
    char        *jobid;
    char        *ib_dev_list;
    char        *ib_devname;
};
typedef struct sharp_conf sharp_conf_t;

struct sharp_reduce_datatyepe_size {
    enum sharp_datatype sharp_data_type;
    int size;
};

struct coll_sharp_component_t {
        struct sharp_coll_context *sharp_coll_context;
};
typedef struct coll_sharp_component_t coll_sharp_component_t;

/* contains sharp_coll_context */
coll_sharp_component_t coll_sharp_component;

struct sharp_info {
    coll_sharp_module_t     *sharp_comm_module;
    sharp_conf_t            *sharp_conf;
};
typedef struct sharp_info sharp_info_t;

void  mv2_get_sharp_datatype(MPI_Datatype  mpi_datatype, struct
        sharp_reduce_datatyepe_size ** dt_size_out); 
enum sharp_reduce_op mv2_get_sharp_reduce_op(MPI_Op mpi_op); 
int     mv2_sharp_coll_init(sharp_conf_t *sharp_conf, int rank);
int     mv2_setup_sharp_env(sharp_conf_t *sharp_conf, MPI_Comm comm);
int     mv2_sharp_coll_comm_init(coll_sharp_module_t *sharp_module);
char *  sharp_create_hostlist(MPI_Comm comm);
int     mv2_free_sharp_handlers (sharp_info_t * sharp_info);
#endif /* _IBV_SHARP_H_ */
