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

#include "mpidi_ch3_impl.h"
#include "mpichconf.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "rdma_impl.h"
#if defined (_SHARP_SUPPORT_)
#include "ibv_sharp.h"

#define MAX_SHARP_PORT 5

extern mv2_MPIDI_CH3I_RDMA_Process_t mv2_MPIDI_CH3I_RDMA_Process;

MPI_Comm mv2_sharp_comm;
struct sharp_data_types_t {
    char * name;
    enum sharp_datatype sharp_data_type;
    MPI_Datatype mpi_data_type;
    MPI_Datatype mpi_data_type_pair;
    int size;
};

struct sharp_op_type_t {
    char * name;
    enum sharp_reduce_op sharp_op_type;
    MPI_Datatype mpi_op_type;
};

const struct  sharp_coll_config sharp_coll_default_config = {
    .ib_dev_list             = "mlx5_0",  
    .user_progress_num_polls = 128,
    .coll_timeout            = 100
};


int g_sharp_port = 0;
char * g_sharp_dev_name = "";

/* sharp supported datatype info*/
struct sharp_data_types_t sharp_data_types [] =
{
    {"UINT_32_BIT", SHARP_DTYPE_UNSIGNED,   MPI_UNSIGNED,   MPI_2INT, 4},
    {"INT_32_BIT",  SHARP_DTYPE_INT,    MPI_INT,    MPI_2INT, 4},
    {"UINT_64_BIT", SHARP_DTYPE_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_LONG_INT, 8},
    {"INT_64_BIT",  SHARP_DTYPE_LONG,   MPI_LONG,   MPI_LONG_INT, 8},
    {"FLOAT_32_BIT",SHARP_DTYPE_FLOAT,  MPI_FLOAT,  MPI_FLOAT_INT, 4},
    {"FLOAT_64_BIT",SHARP_DTYPE_DOUBLE, MPI_DOUBLE, MPI_DOUBLE_INT, 8},
    {"NULL", SHARP_DTYPE_NULL, 0, 0, 0}
};

/*sharp supported reduce op info */
struct sharp_op_type_t sharp_ops [] =
{
    {"MAX", SHARP_OP_MAX, MPI_MAX},
    {"MIN", SHARP_OP_MIN, MPI_MIN},
    {"SUM", SHARP_OP_SUM, MPI_SUM},
    {"LAND", SHARP_OP_LAND, MPI_LAND},
    {"BAND", SHARP_OP_BAND, MPI_BAND},
    {"LOR", SHARP_OP_LOR, MPI_LOR},
    {"BOR",SHARP_OP_BOR, MPI_BOR},
    {"LXOR", SHARP_OP_LXOR, MPI_LXOR},
    {"BXOR", SHARP_OP_BXOR, MPI_BXOR},
    {"MAXLOC", SHARP_OP_MAXLOC, MPI_MAXLOC},
    {"MINLOC", SHARP_OP_MINLOC, MPI_MINLOC},
    {"NOOP", SHARP_OP_NULL, 0}
};


void  mv2_get_sharp_datatype(MPI_Datatype  mpi_datatype, struct
        sharp_reduce_datatyepe_size ** dt_size_out) 
{  
    int i = 0;

    struct sharp_reduce_datatyepe_size * dt_size = 
        MPIU_Malloc(sizeof(struct sharp_reduce_datatyepe_size));
    dt_size->sharp_data_type = SHARP_DTYPE_NULL;

    for (i = 0; sharp_data_types[i].sharp_data_type != SHARP_DTYPE_NULL; i++) {
        if (mpi_datatype == sharp_data_types[i].mpi_data_type) {
            dt_size->sharp_data_type = sharp_data_types[i].sharp_data_type;
            dt_size->size           = sharp_data_types[i].size;
            break;
        }
    }

    *dt_size_out = dt_size;
}

enum sharp_reduce_op mv2_get_sharp_reduce_op(MPI_Op mpi_op) 
{
    int i = 0;
    for (i = 0; sharp_ops[i].sharp_op_type != SHARP_OP_NULL; i++) {
        if (mpi_op == sharp_ops[i].mpi_op_type) {
            return sharp_ops[i].sharp_op_type;
        }
    }
    /* undefined sharp reduce op type */
    return SHARP_OP_NULL;
}

int mv2_oob_bcast(void *comm_context, void *buf, int size, int root)
{
    MPI_Comm comm;
    MPID_Comm *comm_ptr = NULL;
    int rank;
    MPIR_Errflag_t errflag = FALSE;
    int mpi_errno = MPI_SUCCESS;
    
    if (comm_context == NULL) {
        comm = mv2_sharp_comm;
    } else {
        comm = ((coll_sharp_module_t *) comm_context)->comm;
    }

    MPI_Comm_rank (comm, &rank);
    MPID_Comm_get_ptr(comm, comm_ptr); 
    mpi_errno = MPIR_Bcast_impl(buf, size, MPI_BYTE, root, comm_ptr, &errflag);
    if (mpi_errno) {
        PRINT_ERROR("sharp mv2_oob_bcast failed\n");
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    return (mpi_errno);

fn_fail:
    goto fn_exit;
}

int mv2_oob_barrier(void *comm_context)
{
    MPI_Comm comm;
    MPID_Comm *comm_ptr = NULL;
    int rank = 0;
    MPIR_Errflag_t errflag = FALSE;
    int mpi_errno = MPI_SUCCESS;
    
    if (comm_context == NULL) {
        comm = mv2_sharp_comm;
    } else {
        comm = ((coll_sharp_module_t*) comm_context)->comm;
    }

    MPI_Comm_rank (comm, &rank); 
    MPID_Comm_get_ptr(comm, comm_ptr);
    mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
    if (mpi_errno) {
        PRINT_ERROR("sharp mv2_oob_barrier failed\n");
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    return (mpi_errno);

fn_fail:
    goto fn_exit;
}

int mv2_oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len)
{
    MPI_Comm comm;
    MPID_Comm *comm_ptr = NULL;
    int rank = 0;
    MPIR_Errflag_t errflag = FALSE;
    int mpi_errno = MPI_SUCCESS;
    
    if (comm_context == NULL) {
        comm = mv2_sharp_comm;
    } else {
        comm = ((coll_sharp_module_t*) comm_context)->comm;
    }

    MPI_Comm_rank (comm, &rank); 
    MPID_Comm_get_ptr(comm, comm_ptr);
    mpi_errno = MPIR_Gather_impl(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root, comm_ptr, &errflag);
    if (mpi_errno) {
        PRINT_ERROR("sharp mv2_oob_gather failed\n");
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    return (mpi_errno);

fn_fail:
    goto fn_exit;
}

/* initilize sharp */
int mv2_sharp_coll_init(sharp_conf_t *sharp_conf, int rank)
{
    int mpi_errno = SHARP_COLL_SUCCESS;
    struct sharp_coll_init_spec * init_spec = 
        MPIU_Malloc(sizeof(struct sharp_coll_init_spec));

    init_spec->progress_func             = NULL;
    init_spec->job_id                    = atol(sharp_conf->jobid);
    init_spec->hostlist                  = sharp_conf->hostlist; 
    init_spec->oob_colls.barrier         = mv2_oob_barrier;
    init_spec->oob_colls.bcast           = mv2_oob_bcast;
    init_spec->oob_colls.gather          = mv2_oob_gather;
    init_spec->config                    = sharp_coll_default_config;
    init_spec->group_channel_idx         = rank;
    init_spec->config.ib_dev_list        = sharp_conf->ib_dev_list; 
    MPI_Comm_rank(MPI_COMM_WORLD, &(init_spec->world_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(init_spec->world_size));

    mpi_errno = sharp_coll_init(init_spec, &(coll_sharp_component.sharp_coll_context));   
    if (mpi_errno != SHARP_COLL_SUCCESS) {
        if(rank == 0)
        {
            PRINT_ERROR("SHARP initialization failed. Continuing without SHARP support. Errno: %d (%s) \n", mpi_errno, sharp_coll_strerror(mpi_errno));
        }
        mpi_errno = MPI_ERR_INTERN;
        goto fn_exit;
    } 

fn_exit:
    MPIU_Free(init_spec);
    return (mpi_errno);

fn_fail:
    goto fn_exit;
}


int sharp_get_hca_name_and_port () 
{
    int mpi_errno = MPI_SUCCESS;
#if defined(HAVE_LIBIBVERBS)
    int num_devices = 0, i = 0;
    int sharp_port = 0;
    struct ibv_device **dev_list = NULL;
    struct ibv_context *ctx;
    struct ibv_device_attr dev_attr;
    mv2_hca_type hca_type = 0;

    dev_list = ibv_get_device_list(&num_devices);
    for(i=0; i<num_devices; i++){
        hca_type = mv2_get_hca_type(dev_list[i]);
        if(!MV2_IS_IB_CARD(hca_type)) {
            PRINT_ERROR( "Unable to get the device name, please set MV2_SHARP_HCA_NAME and MV2_SHARP_PORT and rerun the program\n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
        
        ctx = mv2_MPIDI_CH3I_RDMA_Process.nic_context[i];
        if (ctx == NULL) {
            ctx = ibv_open_device(dev_list[i]);
        }
        if ((ibv_query_device(ctx, &dev_attr))) {
            PRINT_ERROR("ibv_query_device failed \n");
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    
        sharp_port = rdma_find_active_port(ctx, ctx->device);
        if (sharp_port != -1) {
            /* found a open port to be used for SHArP */
            goto finished_query;
        }
    }

finished_query: 
    g_sharp_port = sharp_port;
    g_sharp_dev_name = (char*) ibv_get_device_name( dev_list[i] ); 
    if (!g_sharp_dev_name) { 
        PRINT_ERROR( "Unable to get the device name, please set MV2_SHARP_HCA_NAME and MV2_SHARP_PORT, then rerun the program\n");
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }  

    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
    return 0;
#else
    PRINT_ERROR( "Unable to get the device name, please set MV2_SHARP_HCA_NAME and MV2_SHARP_PORT, then rerun the program. ");
    mpi_errno = MPI_ERR_INTERN;
    MPIR_ERR_POP(mpi_errno);
#endif
    
fn_fail:
    return mpi_errno;
}

static char * sharp_get_kvs_id () 
{
    int  i = 0;
    char * id_str = MPIU_Malloc(100);
    MPIU_Memset(id_str, 0, 100);
    char KVSname[100] = {0};
    UPMI_KVS_GET_MY_NAME(KVSname, 100); /* set kvsname as job id */
    int kvs_offset = 4; //offset to skip kvs_
    for (i = kvs_offset; i < 100; i++) {
        if (KVSname[i] == '_') break; /* format is like kvs_906_storage03.cluster_26667_0 */
        if (isdigit(KVSname[i])) {
            id_str[i-kvs_offset] = KVSname[i];
        }
    }
    return id_str;
}

int mv2_setup_sharp_env(sharp_conf_t *sharp_conf, MPI_Comm comm)
{
    char * dev_list = NULL;
    int mpi_errno   = MPI_SUCCESS;
    mv2_sharp_comm = comm;
    
    /* detect architecture and hca type */
    if (mv2_sharp_hca_name != 0) {
        /* user set HCA name and port for SHArP */
        g_sharp_dev_name = mv2_sharp_hca_name;
        if (mv2_sharp_port != -1) {
            g_sharp_port = mv2_sharp_port;
        } else {
            PRINT_DEBUG(DEBUG_Sharp_verbose, "sharp port number is not set. here we choose port 1 \n");
            g_sharp_port = 1;
        }
    } else {
        /* user did not specify the HCA name and port for SHArP, so here we
         * get these information */
        mpi_errno = sharp_get_hca_name_and_port(); 
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    sharp_conf->ib_devname = g_sharp_dev_name; 
    
    /* set device name list and port number which is used for SHArP in
     * HCA_NAME:PORT_NUM format */ 
    dev_list = MPIU_Malloc(sizeof(sharp_conf->ib_devname) + 3); 
    sprintf(dev_list, "%s:%d", sharp_conf->ib_devname, g_sharp_port);
    sharp_conf->ib_dev_list = dev_list;
    
    /* set comam separated hostlist */
    sharp_conf->hostlist = sharp_create_hostlist(comm);
    
    /* set kvsname as job id */  
    sharp_conf->jobid = sharp_get_kvs_id();
    PRINT_DEBUG(DEBUG_Sharp_verbose>2, "sharp_conf->jobid = %s and dev_list =  %s\n", sharp_conf->jobid, dev_list); 
    
    MPI_Comm_rank(comm, &(sharp_conf->rank));
    MPI_Comm_size(comm, &(sharp_conf->size));

fn_exit:
    return mpi_errno;

fn_fail:
    goto fn_exit;
}

int mv2_sharp_coll_comm_init(coll_sharp_module_t *sharp_module)
{
    int size = 0, rank = 0;
    int mpi_errno = SHARP_COLL_SUCCESS;
    MPI_Comm comm;
    struct sharp_coll_comm_init_spec * comm_spec =
        MPIU_Malloc(sizeof (struct sharp_coll_comm_init_spec));
    
    comm = sharp_module->comm; 
    sharp_module->is_leader = 1;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    comm_spec->rank              = rank;
    comm_spec->size              = size;
    comm_spec->is_comm_world     = (comm == MPI_COMM_WORLD);
    comm_spec->oob_ctx           = sharp_module;

    mpi_errno = sharp_coll_comm_init(coll_sharp_component.sharp_coll_context,
                 comm_spec, &(sharp_module->sharp_coll_comm));
    if (mpi_errno != SHARP_COLL_SUCCESS) {
        PRINT_ERROR("sharp communicator init failed with error code %d = %s \n", mpi_errno, sharp_coll_strerror(mpi_errno));
        mpi_errno = MPI_ERR_INTERN;
        MPIR_ERR_POP(mpi_errno);
    }
    
    PRINT_DEBUG(DEBUG_Sharp_verbose, "sharp communicator was initialized successfully for rank %d \n", rank);

fn_exit:
    MPIU_Free(comm_spec);
    return (mpi_errno);

fn_fail:
    goto fn_exit;
}
/* comma separated hostlist */
char * sharp_create_hostlist(MPI_Comm comm)
{
    int  name_length, size, i, rank;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int name_len[size];
    int offsets[size];
    MPIR_Errflag_t errflag = FALSE;
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm *comm_ptr = NULL;

    MPI_Get_processor_name(name, &name_length);
    if (rank < size-1) {
        name[name_length++] = ',';
    }
    
    MPID_Comm_get_ptr(comm, comm_ptr);
    mpi_errno = MPIR_Allgather_impl(&name_length, 1, MPI_INT, &name_len[0], 1, MPI_INT, comm_ptr, &errflag);
    if (mpi_errno) { 
        PRINT_ERROR ("collect hostname len from all ranks failed\n");
        MPIR_ERR_POP(mpi_errno);
    }
    int bytes = 0;
    for (i = 0; i < size; ++i) {
        offsets[i] = bytes;
        bytes += name_len[i];
    }
    bytes++; 
    char * receive_buffer = MPIU_Malloc(bytes);
    receive_buffer[bytes-1] = 0;
    mpi_errno = MPIR_Allgatherv_impl(&name[0], name_length, MPI_CHAR, &receive_buffer[0], &name_len[0], &offsets[0], MPI_CHAR, comm_ptr, &errflag);

    if (mpi_errno != MPI_SUCCESS) {
        PRINT_ERROR("sharp hostname collect failed\n");
        MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    return (receive_buffer);

fn_fail:
    goto fn_exit;

}

int mv2_free_sharp_handlers (sharp_info_t * sharp_info)
{
    int mpi_errno = SHARP_COLL_SUCCESS;
    
    if (sharp_info != NULL && sharp_info->sharp_comm_module != NULL
            && sharp_info->sharp_comm_module->sharp_coll_comm != NULL) {
        /* destroy the sharp comm */
        mpi_errno = sharp_coll_comm_destroy(sharp_info->sharp_comm_module->sharp_coll_comm);
        if (mpi_errno != SHARP_COLL_SUCCESS) {
            PRINT_ERROR("sharp comm destroy failed with error code %d = %s \n", mpi_errno, sharp_coll_strerror(mpi_errno));
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
        
        /* finalize SHArP */
        mpi_errno = sharp_coll_finalize(coll_sharp_component.sharp_coll_context);
        if (mpi_errno != SHARP_COLL_SUCCESS) {
            PRINT_ERROR("sharp coll finalize failed with error code %d = %s \n", mpi_errno, sharp_coll_strerror(mpi_errno));
            mpi_errno = MPI_ERR_INTERN;
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* clean up */
    if (sharp_info != NULL) {    
        if (sharp_info->sharp_conf != NULL) {
            MPIU_Free(sharp_info->sharp_conf->hostlist);
            MPIU_Free(sharp_info->sharp_conf->jobid);
            MPIU_Free(sharp_info->sharp_conf->ib_dev_list); 
            sharp_info->sharp_conf->ib_dev_list = NULL;
            sharp_info->sharp_conf->hostlist    = NULL;
            sharp_info->sharp_conf->jobid       = NULL;
            MPIU_Free(sharp_info->sharp_conf);
            sharp_info->sharp_conf = NULL;
        }
        if (sharp_info->sharp_comm_module != NULL) {
            MPIU_Free(sharp_info->sharp_comm_module);
            sharp_info->sharp_comm_module = NULL;
        }

        MPIU_Free(sharp_info);
    }
    if (mv2_sharp_hca_name != NULL) {
        MPIU_Free(mv2_sharp_hca_name);
    }

    mpi_errno = MPI_SUCCESS;

fn_exit: 
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
#endif /* if defined (_SHARP_SUPPORT_) */
