/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "rdma_impl.h"
#include "mpichconf.h"
#include "upmi.h"
#include "vbuf.h"
#include "ibv_param.h"
#include "rdma_cm.h"
#include "mpiutil.h"
#include "cm.h"
#include "dreg.h"
#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#endif

/* For mv2_system_report */
#include "sysreport.h"

extern int MPIDI_CH3I_CM_Create_region(MPIDI_PG_t *pg);
extern int MPIDI_CH3I_CM_Destroy_region(MPIDI_PG_t *pg);

/* global rdma structure for the local process */
mv2_MPIDI_CH3I_RDMA_Process_t mv2_MPIDI_CH3I_RDMA_Process;
char ufile[500];
char xrc_file[512];
int ring_setup_done = 0;
#ifdef _ENABLE_UD_
static int mv2_ud_start_offset   = 0;
#endif /*_ENABLE_UD_*/

/*keep track of registered resources for RMA windows*/
win_elem_t *mv2_win_list = NULL;

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAIL_CM_Alloc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAIL_CM_Alloc(MPIDI_PG_t * pg)
{
    int mpi_errno   = MPI_SUCCESS;

#ifdef _ENABLE_UD_
    pg->ch.mrail->cm_shmem.remote_ud_info = NULL;
    if (rdma_enable_hybrid) {
        pg->ch.mrail->cm_shmem.remote_ud_info = (mv2_ud_exch_info_t **)
                        MPIU_Malloc(pg->size * sizeof(mv2_ud_exch_info_t*));
        if (pg->ch.mrail->cm_shmem.remote_ud_info == NULL) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**nomem",
                    "**nomem %s", "remote_ud_info");
        }
    }
#endif /* _ENABLE_UD_ */

    if (mv2_shmem_backed_ud_cm) {
        mpi_errno = MPIDI_CH3I_CM_Create_region(pg);
    } else {
#ifdef _ENABLE_UD_
        if (rdma_enable_hybrid) {
            int i = 0;
            for (i = 0; i < pg->size; i++) {
                pg->ch.mrail->cm_shmem.remote_ud_info[i] = (mv2_ud_exch_info_t *)
                            MPIU_Malloc(rdma_num_hcas * sizeof(mv2_ud_exch_info_t));
            }
        }
#endif /* _ENABLE_UD_ */
    }

#ifdef _ENABLE_UD_
  fn_fail:
#endif
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAIL_CM_Dealloc
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAIL_CM_Dealloc(MPIDI_PG_t * pg)
{
    int mpi_errno   = MPI_SUCCESS;

    if (pg->ch.mrail == NULL) {
        /* Nothing to do */
        goto fn_exit;
    }

    if (mv2_shmem_backed_ud_cm) {
        mpi_errno = MPIDI_CH3I_CM_Destroy_region(pg);
    } else {
        if (pg->ch.mrail->cm_shmem.ud_cm) {
            if (!MPIDI_CH3I_Process.has_dpm) {
                /* FIXME: This is a temporary work around to fix an error during
                 * free. This needs to be fixed */
                MPIU_Free(pg->ch.mrail->cm_shmem.ud_cm);
            }
            pg->ch.mrail->cm_shmem.ud_cm = NULL;
        }
#ifdef _ENABLE_UD_
        if (pg->ch.mrail->cm_shmem.remote_ud_info) {
            int i = 0;
            for (i = 0; i < pg->size; ++i) {
                MPIU_Free(pg->ch.mrail->cm_shmem.remote_ud_info[i]);
            }
        }
#endif /* _ENABLE_UD_ */
    }

#ifdef _ENABLE_UD_
    if (pg->ch.mrail->cm_shmem.remote_ud_info) {
        MPIU_Free(pg->ch.mrail->cm_shmem.remote_ud_info);
        pg->ch.mrail->cm_shmem.remote_ud_info = NULL;
    }
#endif /* _ENABLE_UD_ */

    if (pg->ch.mrail->cm_ah) {
        MPIU_Free(pg->ch.mrail->cm_ah);
        pg->ch.mrail->cm_ah = NULL;
    }

    if (pg->ch.mrail) {
        MPIU_Free(pg->ch.mrail);
        pg->ch.mrail = NULL;
    }

fn_exit:
    return mpi_errno;
}

static inline uint16_t get_local_lid(struct ibv_context *ctx, int port)
{
    struct ibv_port_attr attr;

    if (ibv_query_port(ctx, port, &attr)) {
        return -1;
    }

    mv2_MPIDI_CH3I_RDMA_Process.lmc = attr.lmc;

    return attr.lid;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_init(MPIDI_PG_t * pg, int pg_rank)
{

    /* Initialize the rdma implemenation. */
    /* This function is called after the RDMA channel has initialized its
     * structures - like the vc_table. */

    MPIDI_VC_t *vc = NULL;
    int i = 0;
    int error;
    int rail_index;

    char rdmakey[512];
    char rdmavalue[512];
    char *buf = NULL;
    int mpi_errno = MPI_SUCCESS;
    struct process_init_info *init_info = NULL;
    mv2_arch_hca_type my_arch_hca_type;
    int pg_size;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_INIT);

    pg_size = MPIDI_PG_Get_size(pg);

    if (!mv2_MPIDI_CH3I_RDMA_Process.has_lazy_mem_unregister) {
        rdma_r3_threshold = rdma_r3_threshold_nocache;
    }

    rdma_num_rails = rdma_num_hcas * rdma_num_ports * rdma_num_qp_per_port;
    rdma_num_rails_per_hca = rdma_num_ports * rdma_num_qp_per_port;

    if (rdma_multirail_usage_policy == MV2_MRAIL_SHARING) {
        if (mrail_use_default_mapping) {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                (rdma_local_id % rdma_num_hcas);
        } else {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                mrail_user_defined_p2r_mapping;
        }
    }

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "num_qp_per_port %d, num_rails = %d, "
                "rdma_num_rails_per_hca = %d, "
                "rdma_process_binding_rail_offset = %d\n", rdma_num_qp_per_port,
                rdma_num_rails, rdma_num_rails_per_hca,
                rdma_process_binding_rail_offset);

    init_info = alloc_process_init_info(pg_size, rdma_num_rails);
    if (!init_info) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s", "init_info");
    }

    if (pg_size > 1) {
        my_arch_hca_type = mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
        if (mv2_MPIDI_CH3I_RDMA_Process.has_ring_startup) {
            mpi_errno =
                rdma_setup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process, pg_rank,
                                        pg_size);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
            ring_setup_done = 1;

            mpi_errno =
                rdma_ring_based_allgather(&my_arch_hca_type,
                                          sizeof(my_arch_hca_type), pg_rank,
                                          init_info->arch_hca_type, pg_size,
                                          &mv2_MPIDI_CH3I_RDMA_Process);

            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        } else {
            /* Generate the key and value pair */
            MPL_snprintf(rdmakey, 512, "ARCHHCA-%d", pg_rank);
            buf = rdmavalue;
            sprintf(buf, "%016lx", my_arch_hca_type);
            buf += 16;

            /* put the kvs into PMI */
            MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
            MPIU_Strncpy(mv2_pmi_val, rdmavalue, mv2_pmi_max_vallen);
            PRINT_DEBUG(DEBUG_CM_verbose > 0, "put: rdmavalue %s len:%lu arch_hca:%lu\n",
                        mv2_pmi_val, strlen(mv2_pmi_val), my_arch_hca_type);

            error = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
            if (error != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                          "**pmi_kvs_put", "**pmi_kvs_put %d",
                                          error);
            }


            error = UPMI_KVS_COMMIT(pg->ch.kvs_name);
            if (error != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                          "**pmi_kvs_commit",
                                          "**pmi_kvs_commit %d", error);
            }

            error = UPMI_BARRIER();
            if (error != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                          "**pmi_barrier", "**pmi_barrier %d",
                                          error);
            }

            for (i = 0; i < pg_size; i++) {
                if (pg_rank == i) {
                    init_info->arch_hca_type[i] = my_arch_hca_type;
                    continue;
                }

                MPL_snprintf(rdmakey, 512, "ARCHHCA-%d", i);
                MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);

                error = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_get",
                                              "**pmi_kvs_get %d", error);
                }

                MPIU_Strncpy(rdmavalue, mv2_pmi_val, mv2_pmi_max_vallen);
                buf = rdmavalue;

                sscanf(buf, "%016lx", &init_info->arch_hca_type[i]);
                buf += 16;
                PRINT_DEBUG(DEBUG_CM_verbose > 0, "get: rdmavalue %s len:%lu arch_hca[%d]:%lu\n",
                            mv2_pmi_val, strlen(mv2_pmi_val), i, init_info->arch_hca_type[i]);
            }
        }

        /* Check heterogeneity */
        rdma_param_handle_heterogeneity(init_info->arch_hca_type, pg_size);
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.has_apm) {
        init_apm_lock();
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
        mpi_errno = init_vbuf_lock();
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* the vc structure has to be initialized */

    for (i = 0; i < pg_size; ++i) {

        MPIDI_PG_Get_vc(pg, i, &vc);
        MPIU_Memset(&(vc->mrail), 0, sizeof(vc->mrail));

        /* This assmuption will soon be done with */
        vc->mrail.num_rails = rdma_num_rails;
    }

    /* Open the device and create cq and qp's */
    if ((mpi_errno = rdma_iba_hca_init(&mv2_MPIDI_CH3I_RDMA_Process,
                                       pg_rank,
                                       pg, init_info)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize = RDMA_MAX_RDMA_SIZE;

    if (pg_size > 1) {

        /* Exchange the information about HCA_lid / HCA_gid and qp_num */
        if (!mv2_MPIDI_CH3I_RDMA_Process.has_ring_startup) {

            /* For now, here exchange the information of each LID separately */
            for (i = 0; i < pg_size; i++) {
                if (pg_rank == i) {
                    continue;
                }
                /* Generate the key and value pair */
                MPL_snprintf(rdmakey, 512, "2-%d-%08x", pg_rank, i);
                buf = rdmavalue;

                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
                    if (use_iboeth) {
                        sprintf(buf, "%016" SCNx64,
                                init_info->gid[i][rail_index].global.
                                subnet_prefix);
                        buf += 16;
                        sprintf(buf, "%016" SCNx64,
                                init_info->gid[i][rail_index].global.
                                interface_id);
                        buf += 16;
                        PRINT_DEBUG(DEBUG_CM_verbose > 0,"[%d] put subnet prefix = %" PRIx64
                                    " interface id = %" PRIx64 "\r\n", pg_rank,
                                    init_info->gid[i][rail_index].global.
                                    subnet_prefix,
                                    init_info->gid[i][rail_index].global.
                                    interface_id);
                    } else {
                        sprintf(buf, "%08x", init_info->lid[i][rail_index]);
                        PRINT_DEBUG(DEBUG_CM_verbose > 0,"put my hca %d lid %d\n", rail_index,
                                    init_info->lid[i][rail_index]);
                        buf += 8;
                    }
                }

                sprintf(buf, "%016lx", init_info->arch_hca_type[i]);
                buf += 16;
                sprintf(buf, "%016" SCNx64, init_info->vc_addr[i]);
                buf += 16;
                PRINT_DEBUG(DEBUG_CM_verbose > 0,"Put hca type %ld, vc addr %" PRIx64 ", max val %d\n",
                            init_info->arch_hca_type[i], init_info->vc_addr[i],
                            mv2_pmi_max_vallen);

                /* put the kvs into PMI */
                MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
                MPIU_Strncpy(mv2_pmi_val, rdmavalue, mv2_pmi_max_vallen);
                PRINT_DEBUG(DEBUG_CM_verbose > 0,"rdmavalue %s len:%lu \n",
                            mv2_pmi_val, strlen(mv2_pmi_val));

                error = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_put",
                                              "**pmi_kvs_put %d", error);
                }

                PRINT_DEBUG(DEBUG_CM_verbose > 0,"after put, before barrier\n");

                error = UPMI_KVS_COMMIT(pg->ch.kvs_name);

                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_commit",
                                              "**pmi_kvs_commit %d", error);
                }

                /*
                 ** This barrer is placed here because PMI is not allowing to put
                 ** multiple key-value pairs. otherwise this berrier is not required.
                 ** This should be moved out of this loop if PMI allows multiple pairs.
                 */
                error = UPMI_BARRIER();
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_barrier",
                                              "**pmi_barrier %d", error);
                }
            }


            /* Here, all the key and value pairs are put, now we can get them */
            for (i = 0; i < pg_size; i++) {
                rail_index = 0;
                if (pg_rank == i) {
                    if (!use_iboeth) {
                        init_info->lid[i][0] =
                            get_local_lid(mv2_MPIDI_CH3I_RDMA_Process.
                                          nic_context[0], rdma_default_port);
                        init_info->arch_hca_type[i] =
                            mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
                    }
                    PRINT_DEBUG(DEBUG_CM_verbose > 0,"[%d] get subnet prefix = %" PRIx64
                                ", interface id = %" PRIx64 " from proc %d \n",
                                pg_rank,
                                init_info->gid[i][rail_index].global.
                                subnet_prefix,
                                init_info->gid[i][rail_index].global.
                                interface_id, i);
                    continue;
                }

                /* Generate the key */
                MPL_snprintf(rdmakey, 512, "2-%d-%08x", i, pg_rank);
                MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);

                error = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_get",
                                              "**pmi_kvs_get %d", error);
                }

                MPIU_Strncpy(rdmavalue, mv2_pmi_val, mv2_pmi_max_vallen);
                buf = rdmavalue;

                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
                    if (use_iboeth) {
                        sscanf(buf, "%016" SCNx64, (uint64_t *)
                               & init_info->gid[i][rail_index].global.
                               subnet_prefix);
                        buf += 16;
                        sscanf(buf, "%016" SCNx64, (uint64_t *)
                               & init_info->gid[i][rail_index].global.
                               interface_id);
                        buf += 16;
                        PRINT_DEBUG(DEBUG_CM_verbose > 0,"[%d] get subnet prefix = %" PRIx64
                                    "interface id = %" PRIx64 " from proc %d\n",
                                    pg_rank,
                                    init_info->gid[i][rail_index].global.
                                    subnet_prefix,
                                    init_info->gid[i][rail_index].global.
                                    interface_id, i);
                    } else {
                        sscanf(buf, "%08hx",
                               (uint16_t *) &init_info->lid[i][rail_index]);
                        buf += 8;
                        PRINT_DEBUG(DEBUG_CM_verbose > 0,"get rail %d, lid %08d\n", rail_index,
                                    (int) init_info->lid[i][rail_index]);
                    }
                }

                sscanf(buf, "%016lx", &init_info->arch_hca_type[i]);
                buf += 16;
                sscanf(buf, "%016" SCNx64, &init_info->vc_addr[i]);
                buf += 16;
                PRINT_DEBUG(DEBUG_CM_verbose > 0,"Get vc addr %" PRIx64 "\n", init_info->vc_addr[i]);
            }

            /* This barrier is to prevent some process from
             * overwriting values that has not been get yet
             */
            error = UPMI_BARRIER();
            if (error != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                          "**pmi_barrier", "**pmi_barrier %d",
                                          error);
            }

            /* STEP 2: Exchange qp_num and vc addr */
            for (i = 0; i < pg_size; i++) {
                if (pg_rank == i) {
                    continue;
                }
                /* Generate the key and value pair */
                MPL_snprintf(rdmakey, 512, "1-%d-%08x", pg_rank, i);
                buf = rdmavalue;

                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
                    sprintf(buf, "%08X", init_info->qp_num_rdma[i][rail_index]);
                    buf += 8;
                    PRINT_DEBUG(DEBUG_CM_verbose > 0,"target %d, put qp %d, num %08X \n", i,
                                rail_index,
                                init_info->qp_num_rdma[i][rail_index]);
                    PRINT_DEBUG(DEBUG_CM_verbose > 0,"[%d] %s(%d) put qp %08X \n", pg_rank,
                                __FUNCTION__, __LINE__,
                                init_info->qp_num_rdma[i][rail_index]);
                }

                PRINT_DEBUG(DEBUG_CM_verbose > 0,"put rdma value %s\n", rdmavalue);
                /* Put the kvs into PMI */
                MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
                MPIU_Strncpy(mv2_pmi_val, rdmavalue, mv2_pmi_max_vallen);

                error = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_put",
                                              "**pmi_kvs_put %d", error);
                }

                error = UPMI_KVS_COMMIT(pg->ch.kvs_name);
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_commit",
                                              "**pmi_kvs_commit %d", error);
                }

                /*
                 ** This barrer is placed here because PMI is not allowing to put
                 ** multiple key-value pairs. otherwise this berrier is not required.
                 ** This should be moved out of this loop if PMI allows multiple pairs.
                 */

                error = UPMI_BARRIER();
                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_barrier",
                                              "**pmi_barrier %d", error);
                }
            }


            /* Here, all the key and value pairs are put, now we can get them */
            for (i = 0; i < pg_size; i++) {
                if (pg_rank == i) {
                    continue;
                }

                /* Generate the key */
                MPL_snprintf(rdmakey, 512, "1-%d-%08x", i, pg_rank);
                MPIU_Strncpy(mv2_pmi_key, rdmakey, mv2_pmi_max_keylen);
                error = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);

                if (error != UPMI_SUCCESS) {
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                              "**pmi_kvs_get",
                                              "**pmi_kvs_get %d", error);
                }
                MPIU_Strncpy(rdmavalue, mv2_pmi_val, mv2_pmi_max_vallen);

                buf = rdmavalue;
                PRINT_DEBUG(DEBUG_CM_verbose > 0,"get rdmavalue %s\n", rdmavalue);
                for (rail_index = 0; rail_index < rdma_num_rails; rail_index++) {
                    sscanf(buf, "%08X", &init_info->qp_num_rdma[i][rail_index]);
                    buf += 8;
                    PRINT_DEBUG(DEBUG_CM_verbose > 0,"[%d] %s(%d) get qp %08X from %d\n", pg_rank,
                                __FUNCTION__, __LINE__,
                                init_info->qp_num_rdma[i][rail_index], i);
                }
            }

            error = UPMI_BARRIER();
            if (error != UPMI_SUCCESS) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                          "**pmi_barrier", "**pmi_barrier %d",
                                          error);
            }

            PRINT_DEBUG(DEBUG_CM_verbose > 0,"After barrier\n");
        } else {
            /* Exchange the information about HCA_lid, qp_num, and memory,
             * With the ring-based queue pair */
            mpi_errno =
                rdma_ring_boot_exchange(&mv2_MPIDI_CH3I_RDMA_Process, pg,
                                        pg_rank, init_info);
            if (mpi_errno) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }

    /* Initialize the registration cache */
    mpi_errno = dreg_init();

    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Allocate RDMA Buffers */
    mpi_errno = rdma_iba_allocate_memory(&mv2_MPIDI_CH3I_RDMA_Process,
                                         pg_rank, pg_size);

    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Enable all the queue pair connections */
    PRINT_DEBUG(DEBUG_CM_verbose > 0,"Address exchange finished, proceed to enabling connection\n");
    rdma_iba_enable_connections(&mv2_MPIDI_CH3I_RDMA_Process, pg_rank,
                                pg, init_info);
    PRINT_DEBUG(DEBUG_CM_verbose > 0,"Finishing enabling connection\n");

    error = UPMI_BARRIER();

    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", error);
    }

    /* Prefill post descriptors */
    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

        if (!qp_required(vc, pg_rank, i)) {
            vc->state = MPIDI_VC_STATE_ACTIVE;
            MPIDI_CH3I_SMP_Init_VC(vc);
            continue;
        }

        vc->state = MPIDI_VC_STATE_ACTIVE;
        MRAILI_Init_vc(vc);
        if (mv2_use_eager_fast_send &&
            !(SMP_INIT && (vc->smp.local_nodes >= 0))) {
            if (likely(rdma_use_coalesce)) {
                vc->eager_fast_fn = mv2_eager_fast_coalesce_send;
            } else {
                vc->eager_fast_fn = mv2_eager_fast_send;
            }
        }
    }

    error = UPMI_BARRIER();

    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", error);
    }

    if (ring_setup_done) {
        /* clean up the bootstrap qps and free memory */
        if ((mpi_errno =
             rdma_cleanup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process))
            != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* If requested produces a report on the system.  See sysreport.h and sysreport.c */
    if (enable_sysreport) {
        mv2_system_report();
    }


    PRINT_DEBUG(DEBUG_CM_verbose > 0,"Finished MPIDI_CH3I_RDMA_init()\n");

  fn_exit:
    if (init_info) {
        free_process_init_info(init_info, pg_size);
    }

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_INIT);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_MRAILI_Flush
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_MRAILI_Flush(void)
{
    MPIDI_PG_t *pg;
    MPIDI_VC_t *vc;
    int i, pg_rank, pg_size, rail;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_MRAILI_FLUSH);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_MRAILI_FLUSH);

    pg = MPIDI_Process.my_pg;
    pg_rank = MPIDI_Process.my_pg_rank;
    pg_size = MPIDI_PG_Get_size(pg);

    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

        if (!qp_required(vc, pg_rank, i)) {
            continue;
        }

        /* Skip SMP VCs */
        if (SMP_INIT && (vc->smp.local_nodes >= 0)) {
            continue;
        }

#ifdef _ENABLE_UD_
        if (rdma_enable_hybrid) {
            if (vc->mrail.ack_need_tosend == 1) {
                PRINT_DEBUG(DEBUG_UD_verbose > 0, "Sending explicit ack to %d\n", vc->pg_rank);
                mv2_send_explicit_ack(vc);
            }
        }
#endif

        if (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE) {
            continue;
        }

        for (rail = 0; rail < vc->mrail.num_rails; rail++) {
            while (0 != vc->mrail.srp.credits[rail].backlog.len) {
                if ((mpi_errno = MPIDI_CH3I_Progress_test()) != MPI_SUCCESS) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }

            while (NULL != vc->mrail.rails[rail].ext_sendq_head) {
                if ((mpi_errno = MPIDI_CH3I_Progress_test()) != MPI_SUCCESS) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
        }
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_MRAILI_FLUSH);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_finalize(void)
{
    /* Finalize the rdma implementation */
    /* No rdma functions will be called after this function */
    int error;
    int pg_rank;
    int pg_size;
    int i;
    int rail_index;
    int hca_index;

    MPIDI_PG_t *pg;
    MPIDI_VC_t *vc;
    int err;
    int mpi_errno = MPI_SUCCESS;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_FINALIZE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_FINALIZE);

    /* Insert implementation here */
    pg = MPIDI_Process.my_pg;
    pg_rank = MPIDI_Process.my_pg_rank;
    pg_size = MPIDI_PG_Get_size(pg);

    /* Show memory usage statistics */
    if (DEBUG_MEM_verbose) {
        if (pg_rank == 0 || DEBUG_MEM_verbose > 1) {
            mv2_print_mem_usage();
            mv2_print_vbuf_usage_usage();
        }
    }

    /* make sure everything has been sent */
    if (!use_iboeth && (rdma_3dtorus_support || rdma_path_sl_query)) {
        mv2_release_3d_torus_resources();
    }

    MPIDI_CH3I_MRAILI_Flush();
    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

        if (!rdma_use_blocking && !qp_required(vc, pg_rank, i)) {
            continue;
        }

        for (rail_index = 0; rail_index < vc->mrail.num_rails; rail_index++) {
            while ((rdma_default_max_send_wqe) !=
                   vc->mrail.rails[rail_index].send_wqes_avail) {
                MPIDI_CH3I_Progress_test();
            }
        }
    }

    mv2_finalize_upmi_barrier_complete = 0;
    MPICM_Create_finalize_thread();

    /*barrier to make sure queues are initialized before continuing */
    error = UPMI_BARRIER();

    mv2_finalize_upmi_barrier_complete = 1;
    pthread_join(cm_finalize_progress_thread, NULL);

#if defined(_MCST_SUPPORT_)
    if (rdma_enable_mcast) {
        mv2_ud_destroy_ctx(mcast_ctx->ud_ctx);
    }
    if (mcast_ctx) {
        MPIU_Free(mcast_ctx);
        mcast_ctx = NULL;
    }
#endif

    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", error);
    }

    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

        if (!rdma_use_blocking && !qp_required(vc, pg_rank, i)) {
            continue;
        }

        if (!mv2_rdma_fast_path_preallocate_buffers) {
            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                if (vc->mrail.rfp.RDMA_send_buf_mr[hca_index]) {
                    err = ibv_dereg_mr(vc->mrail.rfp.RDMA_send_buf_mr[hca_index]);
                    if (err)
                        MPL_error_printf("Failed to deregister mr (%d)\n", err);
                }
                if (vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]) {
                    err = ibv_dereg_mr(vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]);
                    if (err)
                        MPL_error_printf("Failed to deregister mr (%d)\n", err);
                }
            }
#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda && rdma_eager_cudahost_reg) {
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_send_buf_DMA);
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }
#endif

            if (vc->mrail.rfp.RDMA_send_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf_DMA);
            }
            if (vc->mrail.rfp.RDMA_recv_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }
            if (vc->mrail.rfp.RDMA_send_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf);
            }
            if (vc->mrail.rfp.RDMA_recv_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf);
            }
        }
    }

    /* STEP 2: destroy all the qps, tears down all connections */
    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

#ifdef _ENABLE_UD_
        if (vc->mrail.ud) {
            MPIU_Free(vc->mrail.ud);
        }
#endif /* _ENABLE_UD_ */

        if (!rdma_use_blocking && !qp_required(vc, pg_rank, i)) {
            continue;
        }
#ifndef MV2_DISABLE_HEADER_CACHING
        MPIU_Free(vc->mrail.rfp.cached_incoming);
        MPIU_Free(vc->mrail.rfp.cached_outgoing);
#endif

        for (rail_index = 0; rail_index < vc->mrail.num_rails; rail_index++) {
            err = ibv_destroy_qp(vc->mrail.rails[rail_index].qp_hndl);
            if (err) {
                MPL_error_printf("Failed to destroy QP (%d)\n", err);
            }
        }
    }


    /* STEP 3: release all the cq resource,
     * release all the unpinned buffers,
     * release the ptag and finally,
     * release the hca */

    for (i = 0; i < rdma_num_hcas; i++) {
        if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {

            /* Signal thread if waiting */
            pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                               srq_post_mutex_lock[i]);
            mv2_MPIDI_CH3I_RDMA_Process.is_finalizing = 1;
            pthread_cond_signal(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
            pthread_mutex_unlock(&mv2_MPIDI_CH3I_RDMA_Process.
                                 srq_post_mutex_lock[i]);

            /* wait for async thread to finish processing */
            pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                               async_mutex_lock[i]);

            /* destroy mutex and cond and cancel thread */
            pthread_cond_destroy(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
            pthread_mutex_destroy(&mv2_MPIDI_CH3I_RDMA_Process.
                                  srq_post_mutex_lock[i]);
            pthread_cancel(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i]);

            pthread_join(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i], NULL);

            err = ibv_destroy_srq(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i]);
            pthread_mutex_unlock(&mv2_MPIDI_CH3I_RDMA_Process.
                                 async_mutex_lock[i]);
            pthread_mutex_destroy(&mv2_MPIDI_CH3I_RDMA_Process.
                                  async_mutex_lock[i]);
            if (err)
                MPL_error_printf("Failed to destroy SRQ (%d)\n", err);
        }

        err = ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i]);
        if (err)
            MPL_error_printf("Failed to destroy CQ (%d)\n", err);

        if (mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i]) {
            err = ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.send_cq_hndl[i]);
            if (err) {
                MPL_error_printf("Failed to destroy CQ (%d)\n", err);
            }
        }

        if (mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i]) {
            err = ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.recv_cq_hndl[i]);
            if (err) {
                MPL_error_printf("Failed to destroy CQ (%d)\n", err);
            }
        }

        if (rdma_use_blocking) {
            err =
                ibv_destroy_comp_channel(mv2_MPIDI_CH3I_RDMA_Process.
                                         comp_channel[i]);
            if (err)
                MPL_error_printf("Failed to destroy CQ channel (%d)\n", err);
        }

        deallocate_vbufs(i);
    }

    mv2_free_prealloc_rdma_fp_bufs();
    deallocate_vbuf_region();

    win_elem_t * curr_ptr, *tmp;
    curr_ptr = mv2_win_list;
    while(curr_ptr != NULL) {
        if (curr_ptr->win_base != NULL) {
            dreg_unregister((dreg_entry *)curr_ptr->win_base);
        }
        if (curr_ptr->complete_counter) {
            dreg_unregister((dreg_entry *)curr_ptr->complete_counter);
        }
        if (curr_ptr->post_flag != NULL) {
            dreg_unregister((dreg_entry *)curr_ptr->post_flag);
        }
        tmp = curr_ptr;
        curr_ptr = curr_ptr->next;
        MPIU_Free(tmp);
    }
    mv2_win_list = NULL;

    dreg_finalize();

    for (i = 0; i < rdma_num_hcas; i++) {
        err = ibv_dealloc_pd(mv2_MPIDI_CH3I_RDMA_Process.ptag[i]);
        if (err) {
            MPL_error_printf("[%d] Failed to dealloc pd (%s)\n",
                              pg_rank, strerror(errno));
        }
        err = ibv_close_device(mv2_MPIDI_CH3I_RDMA_Process.nic_context[i]);
        if (err) {
            MPL_error_printf("[%d] Failed to close ib device (%s)\n",
                              pg_rank, strerror(errno));
        }
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.polling_set != NULL) {
        MPIU_Free(mv2_MPIDI_CH3I_RDMA_Process.polling_set);
        mv2_MPIDI_CH3I_RDMA_Process.polling_group_size = 0;
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_FINALIZE);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#ifdef _ENABLE_XRC_
#undef FUNCNAME
#define FUNCNAME mv2_xrc_cleanup
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int mv2_xrc_cleanup(int start)
{
    int i = 0;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    for (i = start; i >= 0; --i) {
        MPL_snprintf(xrc_file, 512, "/dev/shm/%s-%d", ufile, i);
        close(proc->xrc_fd[i]);
        unlink(xrc_file);
    }

    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME mv2_xrc_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int mv2_xrc_init(MPIDI_PG_t * pg)
{
    int i, mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_CH3I_MV2_XRC_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_CH3I_MV2_XRC_INIT);

    PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Init XRC Start\n");

    xrc_rdmafp_init = 1;
    mv2_MPIDI_CH3I_RDMA_Process_t *proc = &mv2_MPIDI_CH3I_RDMA_Process;

    if (!MPIDI_CH3I_Process.has_dpm) {
        memset(ufile, 0, sizeof(ufile));
        sprintf(ufile, "mv2_xrc_%s_%d", pg->ch.kvs_name, getuid());
    }

    for (i = 0; i < rdma_num_hcas; i++) {
        MPL_snprintf(xrc_file, 512, "/dev/shm/%s-%d", ufile, i);
        PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Opening xrc file: %s\n", xrc_file);
        proc->xrc_fd[i] = open(xrc_file, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (proc->xrc_fd[i] < 0) {
            /* Cleanup all the XRC files and FD's open till this point */
            mv2_xrc_cleanup(i-1);
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno,
                                      MPI_ERR_INTERN,
                                      "**fail",
                                      "%s: %s", "open", strerror(errno));
        }

        proc->xrc_domain[i] = ibv_open_xrc_domain(proc->nic_context[i],
                                                  proc->xrc_fd[i], O_CREAT);

        if (NULL == proc->xrc_domain[i]) {
            /* Cleanup all the XRC files and FD's open till this point */
            mv2_xrc_cleanup(i);
            perror("xrc_domain");
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**fail",
                                      "**fail %s", "Can't open XRC domain");
        }
    }

    PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Init XRC DONE\n");
  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_CH3I_MV2_XRC_INIT);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME mv2_xrc_unlink_file
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int mv2_xrc_unlink_file(MPIDI_PG_t * pg)
{
    int i           = 0;
    int mpi_errno   = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_CH3I_MV2_XRC_UNLINK_FILE);
    MPIDI_FUNC_ENTER(MPID_STATE_CH3I_MV2_XRC_UNLINK_FILE);

    PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Unlink XRC Start\n");

    if (!MPIDI_CH3I_Process.has_dpm) {
        memset(ufile, 0, sizeof(ufile));
        MPL_snprintf(ufile, 500, "mv2_xrc_%s_%d", pg->ch.kvs_name, getuid());
    }

    for (i = 0; i < rdma_num_hcas; i++) {
        MPL_snprintf(xrc_file, 512, "/dev/shm/%s-%d", ufile, i);
        PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Unlink XRC file: %s\n", xrc_file);

        if (!MPIDI_CH3I_Process.has_dpm) {
            unlink(xrc_file);
        }
    }

    PRINT_DEBUG(DEBUG_XRC_verbose > 0, "Unlink XRC DONE\n");

    MPIDI_FUNC_EXIT(MPID_STATE_CH3I_MV2_XRC_UNLINK_FILE);
    return mpi_errno;
}
#endif /* _ENABLE_XRC_ */

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Ring_Exchange_Init_Info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Ring_Exchange_Init_Info(MPIDI_PG_t * pg, int pg_rank,
                                        mv2_process_init_info_t *my_info,
                                        mv2_arch_hca_type *arch_hca_type_all)
{
    int i         = 0;
    int pg_size   = 0;
    int mpi_errno = MPI_SUCCESS;
    mv2_process_init_info_t *all_info = NULL;
#ifdef _ENABLE_UD_
    int hca_index = 0;
#endif /* _ENABLE_UD_ */

    MPIDI_STATE_DECL(MPIDI_CH3I_RING_EXCHANGE_INIT_INFO);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_RING_EXCHANGE_INIT_INFO);

    pg_size = MPIDI_PG_Get_size(pg);

    /* Setup IB ring */
    mpi_errno = rdma_setup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process,
                                        pg_rank, pg_size);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }
    /* This is used for cleanup later */
    ring_setup_done = 1;

    all_info = (mv2_process_init_info_t*)
                    MPIU_Malloc(sizeof(mv2_process_init_info_t) * pg_size);
    if (NULL == all_info) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Exchange data over IB ring */
    mpi_errno = rdma_ring_based_allgather(my_info,
                                          sizeof(mv2_process_init_info_t),
                                          pg_rank, all_info, pg_size,
                                          &mv2_MPIDI_CH3I_RDMA_Process);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Cleanup IB ring */
    mpi_errno = rdma_cleanup_startup_ring(&mv2_MPIDI_CH3I_RDMA_Process);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    for (i = 0; i < pg_size; i++) {
        pg->ch.mrail->cm_shmem.ud_cm[i].cm_lid = all_info[i].lid[0][0];
        memcpy(&pg->ch.mrail->cm_shmem.ud_cm[i].cm_gid, &all_info[i].gid[0][0],
                sizeof(union ibv_gid));
        arch_hca_type_all[i] = all_info[i].my_arch_hca_type;
        pg->ch.mrail->cm_shmem.ud_cm[i].cm_ud_qpn = all_info[i].ud_cm_qpn;
#ifdef _ENABLE_XRC_
        if (USE_XRC) {
            pg->ch.mrail->cm_shmem.ud_cm[i].xrc_hostid = all_info[i].hostid;
        }
#endif
#ifdef _ENABLE_UD_
        if (rdma_enable_hybrid) {
            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                pg->ch.mrail->cm_shmem.remote_ud_info[i][hca_index].lid =
                                            all_info[i].lid[hca_index][0];
                pg->ch.mrail->cm_shmem.remote_ud_info[i][hca_index].qpn =
                                            all_info[i].ud_data_qpn[hca_index];
                PRINT_DEBUG(DEBUG_UD_verbose > 0, "rank:%d, hca:%d Get lid:%d"
                            " ud_qpn:%d\n", i, hca_index,
                            pg->ch.mrail->cm_shmem.remote_ud_info[i][hca_index].lid,
                            pg->ch.mrail->cm_shmem.remote_ud_info[i][hca_index].qpn);
            }
        }
#endif /* _ENABLE_UD_ */
    }

  fn_exit:
    MPIU_Free(all_info);
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_Ring_Exchange_Init_Info() \n");

    MPIDI_FUNC_EXIT(MPIDI_CH3I_RING_EXCHANGE_INIT_INFO);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Iallgather_Init_Info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Iallgather_Init_Info(MPIDI_PG_t * pg, int pg_rank,
                                        mv2_process_init_info_t *my_info,
                                        mv2_arch_hca_type *arch_hca_type_all)
{
    int mpi_errno = MPI_SUCCESS;
#ifdef _ENABLE_UD_
    char temp2[32];
    char temp1[512];
    int hca_index = 0;
#endif

    MPIDI_STATE_DECL(MPIDI_CH3I_IALLGATHER_INIT_INFO);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_IALLGATHER_INIT_INFO);

    /* Generate the value */
    if (use_iboeth) {
        MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                      "%08hx:%08x:%016lx:%08x:%016" SCNx64 ":%016" SCNx64,
                      my_info->lid[0][0], my_info->ud_cm_qpn,
                      my_info->my_arch_hca_type, my_info->hostid,
                      my_info->gid[0][0].global.subnet_prefix,
                      my_info->gid[0][0].global.interface_id);
    } else {
#ifdef _ENABLE_XRC_
        if (USE_XRC) {
            if (mv2_homogeneous_cluster) {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x::%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->hostid);
            } else {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x:%016lx:%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->my_arch_hca_type, my_info->hostid);
            }
        } else
#endif /* _ENABLE_XRC_ */
        {
            if (mv2_homogeneous_cluster) {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn);
            } else {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x:%016lx",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->my_arch_hca_type);
            }
        }
    }

    arch_hca_type_all[pg_rank] = my_info->my_arch_hca_type;

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        mv2_ud_start_offset = strlen(mv2_pmi_val);

        memset(temp1, 0, sizeof(temp1));
        memset(temp2, 0, sizeof(temp2));

        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
            sprintf(temp2, ":%08hx:%08x", my_info->lid[hca_index][0],
                    my_info->ud_data_qpn[hca_index]);
            strcat(temp1, temp2);
    
            PRINT_DEBUG(DEBUG_CM_verbose > 0, "rank:%d Put lids: %d ud_qp: %d\n",
                        pg_rank, my_info->lid[hca_index][0],
                        my_info->ud_data_qpn[hca_index]);
            pg->ch.mrail->cm_shmem.remote_ud_info[pg_rank][hca_index].lid = my_info->lid[hca_index][0];
            pg->ch.mrail->cm_shmem.remote_ud_info[pg_rank][hca_index].qpn = my_info->ud_data_qpn[hca_index];
        }
        MPIU_Strnapp(mv2_pmi_val, temp1, mv2_pmi_max_vallen);
    }
#endif /* _ENABLE_UD_ */

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Iallgather: Init info: %s\n Len: %lu\n",
                mv2_pmi_val, strlen(mv2_pmi_val));

    mpi_errno = UPMI_IALLGATHER(mv2_pmi_val);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_barrier",
                "**pmi_barrier %d", mpi_errno);
    }

  fn_exit:
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_Iallgather_Init_Info() \n");

    MPIDI_FUNC_EXIT(MPIDI_CH3I_IALLGATHER_INIT_INFO);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_PMI_Get_Init_Info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_PMI_Get_Init_Info(MPIDI_PG_t * pg, int tgt_rank,
                                    mv2_arch_hca_type *arch_hca_type_all)
{
#ifdef _ENABLE_UD_
    char *ptr       = NULL;
    int ud_width    = 18;
    int hca_index   = 0;
#endif /*_ENABLE_UD_*/
    int hostid      = 0;
    int offset      = 0;
    int mpi_errno   = MPI_SUCCESS;

    MPIU_Assert(mv2_MPIDI_CH3I_RDMA_Process.has_ring_startup == 0);

    MPIDI_STATE_DECL(MPIDI_CH3I_PMI_GET_INIT_INFO);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_PMI_GET_INIT_INFO);
        
    MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen,
                    "MV2INITINFO-%d", tgt_rank);

    if (mv2_use_pmi_ibarrier) {
        mpi_errno = UPMI_WAIT();
        if (mpi_errno != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_wait", "**pmi_wait %d", mpi_errno);
        }
    } else if (mv2_use_pmi_iallgather) {
        mpi_errno = UPMI_IALLGATHER_WAIT((void **)&mv2_pmi_iallgather_buf);
        if (mpi_errno != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_iallgather_wait", "**pmi_iallgather_wait %d", mpi_errno);
        }
    }

    if (mv2_shmem_backed_ud_cm) {
#ifdef _ENABLE_XRC_
        if ((pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid != 0) &&
            (pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn != 0) &&
            (pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type != 0L) &&
            (use_iboeth && pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.interface_id != 0) &&
            (use_iboeth && pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.subnet_prefix != 0) &&
            (USE_XRC && pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].xrc_hostid != 0))
#else
        if ((pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid != 0) &&
            (pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn != 0) &&
            (pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type != 0L) &&
            (use_iboeth && pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.interface_id != 0) &&
            (use_iboeth && pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.subnet_prefix != 0))
#endif
        {
            /* Some local process already got UD CM info */
            PRINT_DEBUG(DEBUG_CM_verbose > 0, "Already have UD CM info for %d (non-lock)\n", tgt_rank);
            PRINT_DEBUG(DEBUG_CM_verbose > 0, "Saved rank:%d, lid:%d, cm_ud_qpn: %d, arch_type: %ld\n",
                    tgt_rank, pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid,
                    pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn, 
                    pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type);
            if (arch_hca_type_all != NULL) {
                arch_hca_type_all[tgt_rank] =
                    pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type;
            }
            return UPMI_SUCCESS;
        }
    }

    if (mv2_use_pmi_iallgather) {
        MPIU_Assert(mv2_pmi_iallgather_buf);
        offset = tgt_rank * mv2_pmi_max_vallen;
        MPIU_Strncpy(mv2_pmi_val, (mv2_pmi_iallgather_buf + offset), mv2_pmi_max_vallen);
    } else {
        mpi_errno = UPMI_KVS_GET(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val,
                mv2_pmi_max_vallen);
        if (mpi_errno != UPMI_SUCCESS) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_get",
                    "**pmi_kvs_get %d", mpi_errno);
        }
    }

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Get from %d - %s\n", tgt_rank, mv2_pmi_val);
    if (use_iboeth) {
        sscanf(mv2_pmi_val,
            "%08hx:%08x:%016lx:%08x:%016" SCNx64 ":%016" SCNx64,
            (uint16_t *) &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid),
            &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn), 
            &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type),
            &hostid, &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.subnet_prefix),
            &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_gid.global.interface_id));
#ifdef _ENABLE_XRC_
        if (USE_XRC) {
            pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].xrc_hostid = hostid;
	}
#endif /* _ENABLE_XRC_ */
    } else {
#ifdef _ENABLE_XRC_
        if (USE_XRC) {
            if (mv2_homogeneous_cluster) {
                sscanf(mv2_pmi_val, "%08hx:%08x:%08x",
                    (uint16_t *) &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid),
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn), &hostid);
            } else {
                sscanf(mv2_pmi_val, "%08hx:%08x:%016lx:%08x",
                    (uint16_t *) &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid),
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn), 
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type),
                    &hostid);
            }
            pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].xrc_hostid = hostid;
        } else
#endif /* _ENABLE_XRC_ */
        {
            if (mv2_homogeneous_cluster) {
                sscanf(mv2_pmi_val, "%08hx:%08x",
                    (uint16_t *) &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid),
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn));
            } else {
                sscanf(mv2_pmi_val, "%08hx:%08x:%016lx",
                    (uint16_t *) &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid),
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn), 
                    &(pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type));
            }
        }
        PRINT_DEBUG(DEBUG_CM_verbose > 0, "rank:%d, lid:%d, cm_ud_qpn: %d, arch_type: %ld\n",
                    tgt_rank, pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_lid,
                    pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].cm_ud_qpn, 
                    pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type);
    }
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
            ptr = (char*) (mv2_pmi_val + mv2_ud_start_offset) + (hca_index * ud_width);
            sscanf(ptr, ":%08hx:%08x",
                    &(pg->ch.mrail->cm_shmem.remote_ud_info[tgt_rank][hca_index].lid),
                    &(pg->ch.mrail->cm_shmem.remote_ud_info[tgt_rank][hca_index].qpn));
            PRINT_DEBUG(DEBUG_UD_verbose > 0, "rank:%d, hca:%d Get lid:%d"
                        " ud_qpn:%d\n", tgt_rank, hca_index,
                        pg->ch.mrail->cm_shmem.remote_ud_info[tgt_rank][hca_index].lid,
                        pg->ch.mrail->cm_shmem.remote_ud_info[tgt_rank][hca_index].qpn);
        }
    }
#endif /* _ENABLE_UD_ */

    if (arch_hca_type_all != NULL) {
        arch_hca_type_all[tgt_rank] =
            pg->ch.mrail->cm_shmem.ud_cm[tgt_rank].arch_hca_type;
    }
  fn_exit:
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_PMI_Get_Init_Info() \n");

    MPIDI_FUNC_EXIT(MPIDI_CH3I_PMI_GET_INIT_INFO);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_PMI_Exchange_Init_Info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_PMI_Exchange_Init_Info(MPIDI_PG_t * pg, int pg_rank,
                                        mv2_process_init_info_t *my_info,
                                        mv2_arch_hca_type *arch_hca_type_all)
{
    int i           = 0;
    int pg_size     = 0;
    int mpi_errno   = MPI_SUCCESS;
#ifdef _ENABLE_UD_
    char temp2[32];
    char temp1[512];
    int hca_index   = 0;
#endif

    MPIDI_STATE_DECL(MPIDI_CH3I_PMI_EXCHANGE_INIT_INFO);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_PMI_EXCHANGE_INIT_INFO);

    pg_size = MPIDI_PG_Get_size(pg);

    /* Generate the key and value pair */
    MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "MV2INITINFO-%d", pg_rank);
    if (use_iboeth) {
        MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                      "%08hx:%08x:%016lx:%08x:%016" SCNx64 ":%016" SCNx64,
                      my_info->lid[0][0], my_info->ud_cm_qpn,
                      my_info->my_arch_hca_type, my_info->hostid,
                      my_info->gid[0][0].global.subnet_prefix,
                      my_info->gid[0][0].global.interface_id);
    } else {
#ifdef _ENABLE_XRC_
        if (USE_XRC) {
            if (mv2_homogeneous_cluster) {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x::%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->hostid);
            } else {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x:%016lx:%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->my_arch_hca_type, my_info->hostid);
            }
        } else
#endif /* _ENABLE_XRC_ */
        {
            if (mv2_homogeneous_cluster) {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x",
                          my_info->lid[0][0], my_info->ud_cm_qpn);
            } else {
                MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen,
                          "%08hx:%08x:%016lx",
                          my_info->lid[0][0], my_info->ud_cm_qpn,
                          my_info->my_arch_hca_type);
            }
        }
    }

    arch_hca_type_all[pg_rank] = my_info->my_arch_hca_type;

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        mv2_ud_start_offset = strlen(mv2_pmi_val);

        memset(temp1, 0, sizeof(temp1));
        memset(temp2, 0, sizeof(temp2));

        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
            sprintf(temp2, ":%08hx:%08x", my_info->lid[hca_index][0],
                    my_info->ud_data_qpn[hca_index]);
            strcat(temp1, temp2);
    
            PRINT_DEBUG(DEBUG_CM_verbose > 0, "rank:%d Put lids: %d ud_qp: %d\n",
                        pg_rank, my_info->lid[hca_index][0],
                        my_info->ud_data_qpn[hca_index]);
            pg->ch.mrail->cm_shmem.remote_ud_info[pg_rank][hca_index].lid = my_info->lid[hca_index][0];
            pg->ch.mrail->cm_shmem.remote_ud_info[pg_rank][hca_index].qpn = my_info->ud_data_qpn[hca_index];
        }
        MPIU_Strnapp(mv2_pmi_val, temp1, mv2_pmi_max_vallen);
    }
#endif /* _ENABLE_UD_ */

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Put: Init info: %s. Len: %lu\n",
                mv2_pmi_val, strlen(mv2_pmi_val));

    mpi_errno = UPMI_KVS_PUT(pg->ch.kvs_name, mv2_pmi_key, mv2_pmi_val);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_put",
                                    "**pmi_kvs_put %d", mpi_errno);
    }

    mpi_errno = UPMI_KVS_COMMIT(pg->ch.kvs_name);
    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_kvs_commit",
                                    "**pmi_kvs_commit %d", mpi_errno);
    }

    if (mv2_use_pmi_ibarrier) {
        mpi_errno = UPMI_IBARRIER();
    } else {
        mpi_errno = UPMI_BARRIER();
    }

    if (mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmi_barrier",
                "**pmi_barrier %d", mpi_errno);
    }

    if (!mv2_on_demand_ud_info_exchange || !mv2_homogeneous_cluster) {
        for (i = 0; i < pg_size; i++) {
            if (i != pg_rank) {
                mpi_errno = MPIDI_CH3I_PMI_Get_Init_Info(pg, i, arch_hca_type_all);
                if (mpi_errno != MPI_SUCCESS) {
                    MPIR_ERR_POP(mpi_errno);
                }
            }
        }
    }

  fn_exit:
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_PMI_Exchange_Init_Info() \n");

    MPIDI_FUNC_EXIT(MPIDI_CH3I_PMI_EXCHANGE_INIT_INFO);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

extern struct ibv_qp *cm_ud_qp;

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Exchange_Init_Info
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_Exchange_Init_Info(MPIDI_PG_t * pg, int pg_rank)
{
    int pg_size   = 0;
#ifdef _ENABLE_UD_
    int hca_index = 0;
#endif /*_ENABLE_UD_*/
    int mpi_errno = MPI_SUCCESS;
    struct hostent *hostent = NULL;
    char hostname[HOSTNAME_LEN + 1];
    mv2_process_init_info_t my_proc_info;
    mv2_arch_hca_type *arch_hca_type_all = NULL;

    MPIDI_STATE_DECL(MPIDI_CH3I_EXCHANGE_INIT_INFO);
    MPIDI_FUNC_ENTER(MPIDI_CH3I_EXCHANGE_INIT_INFO);

    pg_size = MPIDI_PG_Get_size(pg);

    MPIU_Memset(&my_proc_info, 0, sizeof(my_proc_info));
    arch_hca_type_all = (mv2_arch_hca_type *)
                            MPIU_Malloc(pg_size * sizeof(mv2_arch_hca_type));
    if (!arch_hca_type_all) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**nomem",
                                  "**nomem %s",
                                  "archtype struct to exchange information");
    }

#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        /* Get the hostid */
        mpi_errno = gethostname(hostname, HOSTNAME_LEN);
        if (mpi_errno != 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                      "**fail %s", "Could not get hostname");
        }
        hostent = gethostbyname(hostname);
        if (hostent == NULL) {
            MPIR_ERR_SETFATALANDJUMP2(mpi_errno, MPI_ERR_OTHER,
                        "**gethostbyname", "**gethostbyname %s %d",
                        hstrerror(h_errno), h_errno );
        }
        my_proc_info.hostid             = (int) ((struct in_addr *)
                                            hostent->h_addr_list[0])->s_addr;
    }
#endif /*_ENABLE_XRC_*/

    my_proc_info.ud_cm_qpn          = cm_ud_qp->qp_num;
    my_proc_info.my_arch_hca_type   = mv2_MPIDI_CH3I_RDMA_Process.arch_hca_type;
    MPIU_Memcpy(&my_proc_info.lid, &mv2_MPIDI_CH3I_RDMA_Process.lids,
           sizeof(uint16_t) * MAX_NUM_HCAS * MAX_NUM_PORTS);
    if (use_iboeth) {
        MPIU_Memcpy(&my_proc_info.gid, &mv2_MPIDI_CH3I_RDMA_Process.gids,
            sizeof(union ibv_gid) * MAX_NUM_HCAS * MAX_NUM_PORTS);
    }

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
            my_proc_info.ud_data_qpn[hca_index] =
                mv2_MPIDI_CH3I_RDMA_Process.ud_rails[hca_index]->qp->qp_num;
        }
    }
#endif /*_ENABLE_UD_*/

    /* Initialize local PG structure */
    pg->ch.mrail->cm_shmem.ud_cm[pg_rank].arch_hca_type = my_proc_info.my_arch_hca_type;
    pg->ch.mrail->cm_shmem.ud_cm[pg_rank].cm_ud_qpn = my_proc_info.ud_cm_qpn;
    pg->ch.mrail->cm_shmem.ud_cm[pg_rank].cm_lid    = my_proc_info.lid[0][0];
    if (use_iboeth) {
        MPIU_Memcpy(&pg->ch.mrail->cm_shmem.ud_cm[pg_rank].cm_gid,
                    &my_proc_info.gid, sizeof(union ibv_gid));
    }
#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        pg->ch.mrail->cm_shmem.ud_cm[pg_rank].xrc_hostid    = my_proc_info.hostid;
    }
#endif

    if (pg_size == 1) {
        /* With only one process, we don't need to do anything else */
        goto fn_exit;
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.has_ring_startup) {
        mpi_errno = MPIDI_CH3I_Ring_Exchange_Init_Info(pg, pg_rank,
                                            &my_proc_info, arch_hca_type_all);
    } else if (mv2_use_pmi_iallgather) {
        mpi_errno = MPIDI_CH3I_Iallgather_Init_Info(pg, pg_rank,
                                            &my_proc_info, arch_hca_type_all);
    } else {
        mpi_errno = MPIDI_CH3I_PMI_Exchange_Init_Info(pg, pg_rank,
                                            &my_proc_info, arch_hca_type_all);
    }
    if (mpi_errno != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s", "Could exchange init info");
    }

    if (!mv2_homogeneous_cluster) {
        /* Check heterogeneity */
        rdma_param_handle_heterogeneity(arch_hca_type_all, pg_size);
    }

  fn_exit:
    MPIU_Free(arch_hca_type_all);
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_Exchange_Init_Info() \n");

    MPIDI_FUNC_EXIT(MPIDI_CH3I_EXCHANGE_INIT_INFO);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_Init(MPIDI_PG_t * pg, int pg_rank, char **conn_info_ptr)
{
    int i                   = 0;
    int pg_size             = 0;
    int mpi_errno           = MPI_SUCCESS;
    MPIDI_VC_t *vc          = NULL;
    uint32_t ud_qpn_self    = 0;

    MPIDI_STATE_DECL(MPID_STATE_CH3I_CM_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_CH3I_CM_INIT);

    pg_size = MPIDI_PG_Get_size(pg);

    /* Initialize parameters for XRC */
#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        mpi_errno = mv2_xrc_init(pg);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif /* _ENABLE_XRC_ */

    if ((mpi_errno = rdma_iba_hca_init_noqp(&mv2_MPIDI_CH3I_RDMA_Process,
                                            pg_rank, pg_size)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    if ((mpi_errno = MPICM_Init_UD_CM(&ud_qpn_self)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        if ((mpi_errno =
             rdma_init_ud(&mv2_MPIDI_CH3I_RDMA_Process)) != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }

        if (rdma_use_ud_zcopy) {
            if ((mpi_errno =
                 mv2_ud_setup_zcopy_rndv(&mv2_MPIDI_CH3I_RDMA_Process)) !=
                MPI_SUCCESS) {
                MPIR_ERR_POP(mpi_errno);
            }
        }
    }
#endif /* _ENABLE_UD_ */

    mpi_errno = MPIDI_CH3I_Exchange_Init_Info(pg, pg_rank);
    if (mpi_errno) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s", "MPIDI_CH3I_Exchange_Init_Info");
    }

    /* Unlink XRC file */
#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        mpi_errno = mv2_xrc_unlink_file(pg);
        if (mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }
#endif /* _ENABLE_XRC_ */

    for (i = 0; i < rdma_num_hcas; i++) {
        pthread_mutex_init(&mv2_MPIDI_CH3I_RDMA_Process.async_mutex_lock[i], 0);
    }

    rdma_num_rails = rdma_num_hcas * rdma_num_ports * rdma_num_qp_per_port;
    rdma_num_rails_per_hca = rdma_num_ports * rdma_num_qp_per_port;

    if (rdma_multirail_usage_policy == MV2_MRAIL_SHARING) {
        if (mrail_use_default_mapping) {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                (rdma_local_id % rdma_num_hcas);
        } else {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                mrail_user_defined_p2r_mapping;
        }
    }

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "num_qp_per_port %d, num_rails = %d, "
                "rdma_num_rails_per_hca = %d, "
                "rdma_process_binding_rail_offset = %d\n", rdma_num_qp_per_port,
                rdma_num_rails, rdma_num_rails_per_hca,
                rdma_process_binding_rail_offset);

    if (mv2_MPIDI_CH3I_RDMA_Process.has_apm) {
        init_apm_lock();
    }

    mpi_errno = init_vbuf_lock();
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* the vc structure has to be initialized */
    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);
        MPIU_Memset(&(vc->mrail), 0, sizeof(vc->mrail));
    }

    mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize = RDMA_MAX_RDMA_SIZE;

    /* Initialize the registration cache */
    mpi_errno = dreg_init();
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Allocate RDMA Buffers */
    mpi_errno = rdma_iba_allocate_memory(&mv2_MPIDI_CH3I_RDMA_Process,
                                         pg_rank, pg_size);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Create address handles for UD CM */
    if (mv2_on_demand_ud_info_exchange) {
        mpi_errno = MPICM_Init_Local_UD_struct(pg);
    } else {
        mpi_errno = MPICM_Init_UD_struct(pg);
    }
    if (mpi_errno) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s", "init_ud_struct");
    }

    /* Create threads for UD CM */
    mpi_errno = MPICM_Create_UD_threads();
    if (mpi_errno) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s", "create_ud_threads");
    }

    /* Create conn info */
    *conn_info_ptr = MPIU_Malloc(128);
    if (!*conn_info_ptr) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**nomem %s", "conn_info_str");
    }
    MPIU_Memset(*conn_info_ptr, 0, 128);

    MPIDI_CH3I_CM_Get_port_info(*conn_info_ptr, 128);

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        if ((mpi_errno =
            rdma_ud_post_buffers(&mv2_MPIDI_CH3I_RDMA_Process)) != 
                MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }

        if ((mpi_errno =
             MPIDI_CH3I_UD_Generate_addr_handles(pg, pg_rank,
                                                 pg_size)) != MPI_SUCCESS) {
             MPIR_ERR_POP(mpi_errno);
        }
    }
#endif /* _ENABLE_UD_ */

  fn_exit:
    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_CM_Init() \n");

    MPIDI_FUNC_EXIT(MPID_STATE_CH3I_CM_INIT);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_CM_Finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_CM_Finalize(void)
{
    /* Finalize the rdma implementation */
    /* No rdma functions will be called after this function */
    int retval = 0;
    int i = 0;
    int rail_index;
    int hca_index;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_VC_t *vc = NULL;

    /* Insert implementation here */
    MPIDI_PG_t *pg = MPIDI_Process.my_pg;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_PG_Get_size(pg);

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_CM_FINALIZE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_CM_FINALIZE);

#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid && DEBUG_UDSTAT_verbose) {
        MPIDI_CH3I_UD_Stats(pg);
    }
#endif

    /* Show memory usage statistics */
    if (DEBUG_MEM_verbose) {
        if (pg_rank == 0 || DEBUG_MEM_verbose > 1) {
            mv2_print_mem_usage();
            mv2_print_vbuf_usage_usage();
        }
    }

    if (mv2_use_pmi_ibarrier) {
        retval = UPMI_WAIT();
        if (retval != 0) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                    "**pmi_wait", "**pmi_wait %d", retval);
        }
    } else if (mv2_use_pmi_iallgather) {
        retval = UPMI_IALLGATHER_WAIT((void **)&mv2_pmi_iallgather_buf);
        retval = UPMI_IALLGATHER_FREE();
    }

    mv2_finalize_upmi_barrier_complete = 0;
    MPICM_Create_finalize_thread();

    /*barrier to make sure queues are initialized before continuing */
    if ((retval = UPMI_BARRIER()) != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", retval);
    }

    mv2_finalize_upmi_barrier_complete = 1;
    pthread_join(cm_finalize_progress_thread, NULL);

#if defined(_MCST_SUPPORT_)
    if (rdma_enable_mcast) {
        mv2_ud_destroy_ctx(mcast_ctx->ud_ctx);
    }
    if (mcast_ctx) {
        MPIU_Free(mcast_ctx);
    }
#endif

    if ((retval = MPICM_Finalize_UD()) != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**fail", "**fail %d", retval);
    }

    if (!use_iboeth && (rdma_3dtorus_support || rdma_path_sl_query)) {
        mv2_release_3d_torus_resources();
    }

    for (; i < pg_size; ++i) {

        MPIDI_PG_Get_vc(pg, i, &vc);

#ifdef _ENABLE_UD_
        if (vc->mrail.ud) {
            MPIU_Free(vc->mrail.ud);
        }
#endif /* _ENABLE_UD_ */
#ifndef MV2_DISABLE_HEADER_CACHING
        MPIU_Free(vc->mrail.rfp.cached_incoming);
        MPIU_Free(vc->mrail.rfp.cached_outgoing);
#endif /* !MV2_DISABLE_HEADER_CACHING */

        if (!qp_required(vc, pg_rank, i)) {
            continue;
        }

        if (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE
#ifdef _ENABLE_XRC_
            && VC_XSTS_ISUNSET(vc, (XF_SEND_IDLE | XF_SEND_CONNECTING |
                                    XF_RECV_IDLE))
#endif
        ) {
            continue;
        }

        if (!mv2_rdma_fast_path_preallocate_buffers) {
            for (hca_index = 0; hca_index < rdma_num_hcas; ++hca_index) {
                if (vc->mrail.rfp.RDMA_send_buf_mr[hca_index]) {
                    ibv_dereg_mr(vc->mrail.rfp.RDMA_send_buf_mr[hca_index]);
                }

                if (vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]) {
                    ibv_dereg_mr(vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]);
                }
            }

#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda && rdma_eager_cudahost_reg) {
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_send_buf_DMA);
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }
#endif
            if (vc->mrail.rfp.RDMA_send_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf_DMA);
            }

            if (vc->mrail.rfp.RDMA_recv_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }

            if (vc->mrail.rfp.RDMA_send_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf);
            }

            if (vc->mrail.rfp.RDMA_recv_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf);
            }
        }

        for (rail_index = 0; rail_index < vc->mrail.num_rails; ++rail_index) {
#ifdef _ENABLE_XRC_
            hca_index = rail_index / (rdma_num_ports *
                                      rdma_num_qp_per_port);
            if (USE_XRC && vc->ch.xrc_my_rqpn[rail_index] != 0) {
                /*  Unregister recv QP */
                PRINT_DEBUG(DEBUG_XRC_verbose > 0, "unreg %d",
                            vc->ch.xrc_my_rqpn[rail_index]);
                if ((retval =
                     ibv_unreg_xrc_rcv_qp(mv2_MPIDI_CH3I_RDMA_Process.
                                          xrc_domain[hca_index],
                                          vc->ch.
                                          xrc_my_rqpn[rail_index])) != 0) {
                    PRINT_DEBUG(DEBUG_XRC_verbose > 0, "unreg failed %d %d",
                                vc->ch.xrc_rqpn[rail_index], retval);
                    MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN,
                                              "**fail", "**fail %s",
                                              "Can't unreg RCV QP");
                }
            }
            if (!USE_XRC || (VC_XST_ISUNSET(vc, XF_INDIRECT_CONN) &&
                             VC_XST_ISSET(vc, XF_SEND_IDLE)))
                /* Destroy SEND QP */
#endif
            {
                if (vc->mrail.rails && vc->mrail.rails[rail_index].qp_hndl) {
                    ibv_destroy_qp(vc->mrail.rails[rail_index].qp_hndl);
                }
            }
        }
    }

#ifdef _ENABLE_UD_
    /* destroy ud context */
    if (rdma_enable_hybrid) {
        for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
            if (mv2_MPIDI_CH3I_RDMA_Process.ud_rails[hca_index]) {
                mv2_ud_destroy_ctx(mv2_MPIDI_CH3I_RDMA_Process.ud_rails[hca_index]);
            }
        }

        if (rdma_use_ud_zcopy && mv2_MPIDI_CH3I_RDMA_Process.zcopy_info.rndv_qp_pool) {
            mv2_ud_zcopy_info_t *zcopy_info =
                &mv2_MPIDI_CH3I_RDMA_Process.zcopy_info;
            dreg_unregister(zcopy_info->grh_mr);

            for (hca_index = 0; hca_index < rdma_num_hcas; hca_index++) {
                mv2_ud_destroy_ctx(zcopy_info->rndv_ud_qps[hca_index]);
                ibv_destroy_cq(zcopy_info->rndv_ud_cqs[hca_index]);
            }

            for (i = 0; i < rdma_ud_num_rndv_qps; i++) {
                ibv_destroy_qp(zcopy_info->rndv_qp_pool[i].ud_qp);
                ibv_destroy_cq(zcopy_info->rndv_qp_pool[i].ud_cq);
            }

            MPIU_Free(zcopy_info->rndv_ud_qps);
            MPIU_Free(zcopy_info->rndv_ud_cqs);
            MPIU_Free(zcopy_info->rndv_qp_pool);
            MPIU_Free(zcopy_info->grh_buf);
        }
    }
#endif

    /* STEP 3: release all the cq resource, relaes all the unpinned buffers, release the
     * ptag
     *   and finally, release the hca */
    for (i = 0; i < rdma_num_hcas; ++i) {
        ibv_destroy_cq(mv2_MPIDI_CH3I_RDMA_Process.cq_hndl[i]);

        if (mv2_MPIDI_CH3I_RDMA_Process.has_srq) {
            /* Signal thread if waiting */
            pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                    srq_post_mutex_lock[i]);
            mv2_MPIDI_CH3I_RDMA_Process.is_finalizing = 1;
            pthread_cond_signal(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
            pthread_mutex_unlock(&mv2_MPIDI_CH3I_RDMA_Process.
                    srq_post_mutex_lock[i]);

            /* wait for async thread to finish processing */
            pthread_mutex_lock(&mv2_MPIDI_CH3I_RDMA_Process.
                    async_mutex_lock[i]);

            /* destroy mutex and cond and cancel thread */
            pthread_cond_destroy(&mv2_MPIDI_CH3I_RDMA_Process.srq_post_cond[i]);
            pthread_mutex_destroy(&mv2_MPIDI_CH3I_RDMA_Process.
                    srq_post_mutex_lock[i]);

            pthread_cancel(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i]);
            pthread_join(mv2_MPIDI_CH3I_RDMA_Process.async_thread[i], NULL);
            PRINT_DEBUG(DEBUG_XRC_verbose > 0, "destroyed SRQ: %d\n", i);
            ibv_destroy_srq(mv2_MPIDI_CH3I_RDMA_Process.srq_hndl[i]);
#ifdef _ENABLE_XRC_
            if (USE_XRC) {
                int err;
                if (MPIDI_CH3I_Process.has_dpm) {
                    hca_index = i / (rdma_num_ports * rdma_num_qp_per_port);
                    MPL_snprintf(xrc_file, 512, "/dev/shm/%s-%d", ufile, hca_index);
                    unlink(xrc_file);
                }
                ibv_close_xrc_domain(mv2_MPIDI_CH3I_RDMA_Process.
                                     xrc_domain[i]);
                if ((err = close(mv2_MPIDI_CH3I_RDMA_Process.xrc_fd[i]))) {
                    MPIR_ERR_SETFATALANDJUMP2(mpi_errno,
                                              MPI_ERR_INTERN,
                                              "**fail",
                                              "%s: %s",
                                              "close", strerror(err));
                }
            }
#endif
        }

        deallocate_vbufs(i);
    }

    mv2_free_prealloc_rdma_fp_bufs();
    deallocate_vbuf_region();
    dreg_finalize();

    for (i = 0; i < rdma_num_hcas; ++i) {
        ibv_dealloc_pd(mv2_MPIDI_CH3I_RDMA_Process.ptag[i]);
        ibv_close_device(mv2_MPIDI_CH3I_RDMA_Process.nic_context[i]);
    }
#ifdef _ENABLE_XRC_
    if (USE_XRC) {
        clear_xrc_hash();
    }
#endif

    if (mv2_MPIDI_CH3I_RDMA_Process.polling_set != NULL) {
        MPIU_Free(mv2_MPIDI_CH3I_RDMA_Process.polling_set);
        mv2_MPIDI_CH3I_RDMA_Process.polling_group_size = 0;
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_CM_FINALIZE);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#if defined(RDMA_CM)
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_CM_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_CM_Init(MPIDI_PG_t * pg, int pg_rank, char **conn_info_ptr)
{
    /* Initialize the rdma implemenation. */
    /* This function is called after the RDMA channel has initialized its
     * structures - like the vc_table. */
    MPIDI_VC_t *vc;
    int pg_size;
    int i, error;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_STATE_DECL(MPID_STATE_CH3I_RDMA_CM_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_CH3I_RDMA_CM_INIT);

    pg_size = MPIDI_PG_Get_size(pg);

    /* We can't setup UD Ring for iWARP devices. Use PMI here for hostid exchange
     ** in the case of hydra. Consider using hydra process mapping in the next
     ** release as it is not correct in the current release.
     */
    if (!using_mpirun_rsh) {
        rdma_cm_exchange_hostid(pg, pg_rank, pg_size);
    }

    for (i = 0; i < rdma_num_hcas; i++) {
        pthread_mutex_init(&mv2_MPIDI_CH3I_RDMA_Process.async_mutex_lock[i], 0);
    }

    rdma_num_rails = rdma_num_hcas * rdma_num_ports * rdma_num_qp_per_port;
    rdma_num_rails_per_hca = rdma_num_ports * rdma_num_qp_per_port;

    if (rdma_multirail_usage_policy == MV2_MRAIL_SHARING) {
        if (mrail_use_default_mapping) {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                (rdma_local_id % rdma_num_hcas);
        } else {
            rdma_process_binding_rail_offset = rdma_num_rails_per_hca *
                mrail_user_defined_p2r_mapping;
        }
    }

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "num_qp_per_port %d, num_rails = %d, "
                "rdma_num_rails_per_hca = %d, "
                "rdma_process_binding_rail_offset = %d\n", rdma_num_qp_per_port,
                rdma_num_rails, rdma_num_rails_per_hca,
                rdma_process_binding_rail_offset);

    if (mv2_MPIDI_CH3I_RDMA_Process.has_apm) {
        init_apm_lock();
    }

    mpi_errno = init_vbuf_lock();
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* the vc structure has to be initialized */
    for (i = 0; i < pg_size; i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);
        MPIU_Memset(&(vc->mrail), 0, sizeof(vc->mrail));
        /* This assmuption will soon be done with */
        vc->mrail.num_rails = rdma_num_rails;
    }

    mpi_errno = rdma_iba_hca_init(&mv2_MPIDI_CH3I_RDMA_Process,
                                  pg_rank, pg, NULL);

    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = rdma_cm_get_hostnames(pg_rank, pg);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (MV2_IS_CHELSIO_IWARP_CARD(mv2_MPIDI_CH3I_RDMA_Process.hca_type)) {
        /* TRAC Ticket #455 */
        if (g_num_smp_peers + 1 < pg_size) {
            int avail_cq_entries = 0;
            avail_cq_entries = rdma_default_max_cq_size /
                ((pg_size - g_num_smp_peers - 1) * rdma_num_rails);
            avail_cq_entries = avail_cq_entries -
                rdma_initial_prepost_depth - 1;
            if (avail_cq_entries < rdma_prepost_depth) {
                rdma_prepost_depth = avail_cq_entries;
            }
            mv2_MPIDI_CH3I_RDMA_Process.global_used_recv_cq =
                (rdma_prepost_depth + rdma_initial_prepost_depth + 1)
                * (pg_size - g_num_smp_peers - 1);
        }
    }

    /* Initialize the registration cache. */
    mpi_errno = dreg_init();

    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = rdma_iba_allocate_memory(&mv2_MPIDI_CH3I_RDMA_Process,
            pg_rank, pg_size);

    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize = RDMA_MAX_RDMA_SIZE;

    error = UPMI_BARRIER();
    if (error != UPMI_SUCCESS) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**fail", "**fail %s",
                                  "PMI Barrier failed");
    }

    if ((mpi_errno =
         rdma_cm_connect_all(pg_rank, pg)) != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (g_num_smp_peers + 1 == pg_size) {
        mv2_MPIDI_CH3I_RDMA_Process.has_one_sided = 0;
    }

  fn_exit:

    PRINT_DEBUG(DEBUG_CM_verbose > 0, "Done MPIDI_CH3I_RDMA_CM_Init() \n");

    MPIDI_FUNC_EXIT(MPID_STATE_CH3I_RDMA_CM_INIT);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_CM_Finalize
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3I_RDMA_CM_Finalize(void)
{
    /* Finalize the rdma implementation */
    /* No rdma functions will be called after this function */
    int retval;
    int i = 0;
    int hca_index;
    int mpi_errno = MPI_SUCCESS;

    MPIDI_VC_t *vc = NULL;

    /* Insert implementation here */
    MPIDI_PG_t *pg = MPIDI_Process.my_pg;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_PG_Get_size(pg);

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_CM_FINALIZE);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_CM_FINALIZE);

    /* Show memory usage statistics */
    if (DEBUG_MEM_verbose) {
        if (pg_rank == 0 || DEBUG_MEM_verbose > 1) {
            mv2_print_mem_usage();
            mv2_print_vbuf_usage_usage();
        }
    }

    mv2_finalize_upmi_barrier_complete = 0;
    MPICM_Create_finalize_thread();

    /*barrier to make sure queues are initialized before continuing */
    if ((retval = UPMI_BARRIER()) != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", retval);
    }

    mv2_finalize_upmi_barrier_complete = 1;
    pthread_join(cm_finalize_progress_thread, NULL);

    /* Barrier to make sure progress thread has exited before continuing */
    if ((retval = UPMI_BARRIER()) != 0) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                                  "**pmi_barrier", "**pmi_barrier %d", retval);
    }

#if defined(_MCST_SUPPORT_)
    if (rdma_enable_mcast) {
        mv2_ud_destroy_ctx(mcast_ctx->ud_ctx);
    }
    if (mcast_ctx) {
        MPIU_Free(mcast_ctx);
        mcast_ctx = NULL;
    }
#endif

    if (!use_iboeth && (rdma_3dtorus_support || rdma_path_sl_query)) {
        mv2_release_3d_torus_resources();
    }

    for (; i < pg_size; ++i) {

        MPIDI_PG_Get_vc(pg, i, &vc);

#ifndef MV2_DISABLE_HEADER_CACHING
        MPIU_Free(vc->mrail.rfp.cached_incoming);
        MPIU_Free(vc->mrail.rfp.cached_outgoing);
#endif /* !MV2_DISABLE_HEADER_CACHING */

        if (!qp_required(vc, pg_rank, i)) {
            continue;
        }

        if (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE) {
            continue;
        }

        if (!mv2_rdma_fast_path_preallocate_buffers) {
            for (hca_index = 0; hca_index < rdma_num_hcas; ++hca_index) {
                if (vc->mrail.rfp.RDMA_send_buf_mr[hca_index]) {
                    ibv_dereg_mr(vc->mrail.rfp.RDMA_send_buf_mr[hca_index]);
                }

                if (vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]) {
                    ibv_dereg_mr(vc->mrail.rfp.RDMA_recv_buf_mr[hca_index]);
                }
            }

#if defined(_ENABLE_CUDA_)
            if (rdma_enable_cuda && rdma_eager_cudahost_reg) {
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_send_buf_DMA);
                ibv_cuda_unregister(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }
#endif

            if (vc->mrail.rfp.RDMA_send_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf_DMA);
            }

            if (vc->mrail.rfp.RDMA_recv_buf_DMA) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf_DMA);
            }

            if (vc->mrail.rfp.RDMA_send_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_send_buf);
            }

            if (vc->mrail.rfp.RDMA_recv_buf) {
                MPIU_Memalign_Free(vc->mrail.rfp.RDMA_recv_buf);
            }
        }
    }

    ib_finalize_rdma_cm(pg_rank, pg);
#ifdef _MULTI_SUBNET_SUPPORT_
    if (mv2_rdma_cm_multi_subnet_support) {
        MPIU_Free(rdma_cm_host_gid_list);
    } else
#endif /* _MULTI_SUBNET_SUPPORT_ */
    {
        MPIU_Free(rdma_cm_host_list);
    }

    if (mv2_MPIDI_CH3I_RDMA_Process.polling_set != NULL) {
        MPIU_Free(mv2_MPIDI_CH3I_RDMA_Process.polling_set);
        mv2_MPIDI_CH3I_RDMA_Process.polling_group_size = 0;
    }

    mv2_free_prealloc_rdma_fp_bufs();
    deallocate_vbuf_region();
    if (g_is_dreg_initialized) {
        dreg_finalize();
        g_is_dreg_initialized = 0;
    }

    for (i = 0; i < rdma_num_hcas; ++i) {
        if (mv2_MPIDI_CH3I_RDMA_Process.ptag[i]) {
            ibv_dealloc_pd(mv2_MPIDI_CH3I_RDMA_Process.ptag[i]);
            mv2_MPIDI_CH3I_RDMA_Process.ptag[i] = NULL;
        }
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_CM_FINALIZE);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}
#endif

/* vi:set sw=4 tw=80: */
