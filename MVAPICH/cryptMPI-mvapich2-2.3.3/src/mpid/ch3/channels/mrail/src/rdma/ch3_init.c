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
#include "mpid_mrail_rndv.h"
#include "rdma_impl.h"
#include "mem_hooks.h"
#include "coll_shmem.h"
#include "hwloc_bind.h"
#include "cm.h"
#if defined(_MCST_SUPPORT_)
#include "ibv_mcast.h"
#endif
#if defined (_SHARP_SUPPORT_)
#include "ibv_sharp.h"
#endif
#ifdef HAVE_ROMIO
#include "romioconf.h"
#endif

#define MPIDI_CH3I_HOST_DESCRIPTION_KEY "description"

MPIDI_CH3I_Process_t MPIDI_CH3I_Process;
int (*check_cq_overflow) (MPIDI_VC_t *c, int rail);
int (*perform_blocking_progress) (int hca_num, int num_cqs);
void (*handle_multiple_cqs) (int num_cqs, int cq_choice, int is_send_completion);
extern int MPIDI_Get_local_host(MPIDI_PG_t *pg, int our_pg_rank);
extern void ib_finalize_rdma_cm(int pg_rank, MPIDI_PG_t *pg);

#undef FUNCNAME
#define FUNCNAME split_type
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int split_type(MPID_Comm * comm_ptr, int stype, int key,
                      MPID_Info *info_ptr, MPID_Comm ** newcomm_ptr)
{
    MPID_Node_id_t id;
    MPIDI_Rank_t nid;
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPID_Get_node_id(comm_ptr, comm_ptr->rank, &id);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    nid = (stype == MPI_COMM_TYPE_SHARED) ? id : MPI_UNDEFINED;
    mpi_errno = MPIR_Comm_split_impl(comm_ptr, nid, key, newcomm_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;

    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

static MPID_CommOps comm_fns = {
    split_type
};

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Init(int has_parent, MPIDI_PG_t * pg, int pg_rank)
{
    int mpi_errno = MPI_SUCCESS;
    int pg_size, threshold, dpm = 0, p;
    char *value, *conn_info = NULL;
    int mv2_rdma_init_timers = 0;
    int user_selected_rdma_cm = 0;
    MPIDI_VC_t *vc;

    /* Override split_type */
    MPID_Comm_fns = &comm_fns;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_INIT);

    /* Explicitly initializing RDMA_FP to 0 */
    mv2_MPIDI_CH3I_RDMA_Process.has_adaptive_fast_path = 0;

    if (MPIDI_CH3_Pkt_size_index[MPIDI_CH3_PKT_CLOSE] !=
        sizeof(MPIDI_CH3_Pkt_close_t)) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                                  "**fail %s",
                                  "Failed sanity check! Packet size table mismatch");
    }

    pg_size = MPIDI_PG_Get_size(pg);

    /* Allocate PMI Key Value Pair */
    mv2_allocate_pmi_keyval();

    mpi_errno = MPIDI_CH3U_Comm_register_create_hook(MPIDI_CH3I_comm_create, NULL);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    
    /* Choose default startup method and set default on-demand threshold */
#if defined(RDMA_CM) && !defined(CKPT) && !(ROMIO_IME)
    /* If user has not forcefully disabled RDMA_CM, and if user has not
     * specified the use of MCAST use it by default */
    if (
        (((value = getenv("MV2_USE_RDMA_CM")) == NULL) || !!atoi(value))
        && (((value = getenv("MV2_USE_RoCE")) == NULL) || !!!atoi(value))
#if defined(_MCST_SUPPORT_)
        && (((value = getenv("MV2_USE_MCAST")) == NULL) || !!!atoi(value))
        && rdma_enable_mcast != 1
#endif /*defined(_MCST_SUPPORT_)*/
       )
    {
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_RDMA_CM;
        threshold = MPIDI_CH3I_RDMA_CM_DEFAULT_ON_DEMAND_THRESHOLD;
    } else
#endif /*defined(RDMA_CM) && !defined(CKPT)*/
    {
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
        threshold = MPIDI_CH3I_CM_DEFAULT_ON_DEMAND_THRESHOLD;
    }

    /*check ON_DEMAND_THRESHOLD */
    value = getenv("MV2_ON_DEMAND_THRESHOLD");
    if (value) {
        threshold = atoi(value);
    }

    if ((value = getenv("MV2_SUPPORT_DPM")) != NULL) {
        dpm = !!atoi(value);
        MPIDI_CH3I_Process.has_dpm = dpm;
    }
    if (MPIDI_CH3I_Process.has_dpm) {
        setenv("MV2_ENABLE_AFFINITY", "0", 1);
#if defined(RDMA_CM) && !defined(CKPT)
        /* DPM is not supported with RDMA_CM. Fall back to basic alltoall CM */
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
        /* Reset value of threshold if user has not set it already */
        if ((value = getenv("MV2_ON_DEMAND_THRESHOLD")) == NULL) {
            threshold = MPIDI_CH3I_CM_DEFAULT_ON_DEMAND_THRESHOLD;
        }
#endif /*defined(RDMA_CM) && !defined(CKPT)*/
#ifdef _ENABLE_UD_
        /* DPM and Hybrid cannot be enabled at the same time */
        rdma_enable_hybrid = 0;
#endif /*_ENABLE_UD_*/
    }

#ifdef _ENABLE_CUDA_
    if ((value = getenv("MV2_USE_CUDA")) != NULL) {
        rdma_enable_cuda = !!atoi(value);
        if (rdma_enable_cuda) {
            cuda_get_user_parameters();
        }
#ifdef ENABLE_LLNL_SITE_SPECIFIC_OPTIONS
    } else {
        rdma_enable_cuda = 1;
        if (!(((value = getenv("MV2_SUPPRESS_CUDA_USAGE_WARNING")) != NULL) && !!atoi(value))) {
            PRINT_INFO((pg_rank == 0), " Automatically enabling CUDA support."
                        " If not using GPU buffers, disabling CUDA support by"
                        " setting MV2_USE_CUDA=0 may improve performance.\n"
                        "To suppress this message, please set"
                        " MV2_SUPPRESS_CUDA_USAGE_WARNING to 1\n");
        }
#endif /*ENABLE_LLNL_SITE_SPECIFIC_OPTIONS*/
    }
    if (!rdma_enable_cuda) {
        if (!(((value = getenv("MV2_SUPPRESS_CUDA_USAGE_WARNING")) != NULL) && !!atoi(value))) {
            PRINT_INFO((pg_rank == 0), "MVAPICH2 has been built with support for CUDA."
                        " But, MV2_USE_CUDA not set to 1. This can lead to errors in"
                        " using GPU buffers. If you are running applications that use"
                        " GPU buffers, please set MV2_USE_CUDA=1 and try again.\n");
            PRINT_INFO((pg_rank == 0), "To suppress this warning, please set"
                        " MV2_SUPPRESS_CUDA_USAGE_WARNING to 1\n");
        }
    }
#endif

#ifdef _ENABLE_UD_
    int i = 0;
    for (i = 0; i < MAX_NUM_HCAS; ++i) {
        mv2_MPIDI_CH3I_RDMA_Process.ud_rails[i] = NULL;
    }
    if ((value = getenv("MV2_HYBRID_ENABLE_THRESHOLD")) != NULL) {
        rdma_hybrid_enable_threshold = atoi(value);
    }
    if ((value = getenv("MV2_USE_UD_HYBRID")) != NULL) {
        rdma_enable_hybrid = atoi(value);
    }
    if (((value = getenv("MV2_USE_ONLY_UD")) != NULL) && !!atoi(value)) {
        rdma_enable_hybrid = atoi(value);
        rdma_enable_only_ud = atoi(value);
        if ((value = getenv("MV2_HYBRID_ENABLE_THRESHOLD")) != NULL) {
            if (atoi(value) > 0) {
                PRINT_INFO((pg_rank == 0),
                           "User requested only UD. Resetting MV2_HYBRID_ENABLE_THRESHOLD to 0.\n");
            }
        }
        rdma_hybrid_enable_threshold = 0;
    }

    if(((value = getenv("MV2_SUPPORT_DPM")) != NULL) && !!atoi(value)) {
            rdma_enable_hybrid = 0;
    }

    if (pg_size < rdma_hybrid_enable_threshold) {
        rdma_enable_hybrid = 0;
    }
    if (rdma_enable_hybrid && MPIDI_CH3I_Process.has_dpm) {
        PRINT_INFO((pg_rank==0), "DPM is not supported with Hybrid builds. Disabling Hybrid\n");
        rdma_enable_only_ud = 0;
        rdma_enable_hybrid = 0;
    }
    if(rdma_enable_hybrid == 1) { 
        /* The zero-copy bcast design is disabled when 
         * hybrid is used */ 
        mv2_enable_zcpy_bcast = 0; 
        mv2_enable_zcpy_reduce = 0; 
        mv2_rdma_init_timers = 1;
#if defined(RDMA_CM) && !defined(CKPT)
        /* UD/Hybrid is not supported with RDMA_CM. Fall back to basic alltoall CM */
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
        /* Reset value of threshold if user has not set it already */
        if ((value = getenv("MV2_ON_DEMAND_THRESHOLD")) == NULL) {
            threshold = MPIDI_CH3I_CM_DEFAULT_ON_DEMAND_THRESHOLD;
        }
#endif /*defined(RDMA_CM) && !defined(CKPT)*/
        if (MPIDI_CH3I_Process.has_dpm) {
            MPL_error_printf("Error: DPM is not supported with Hybrid builds.\n"
                    "Please reconfigure MVAPICH2 library without --enable-hybrid option.\n");
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**fail");
        }
    } 
#endif /* #ifdef _ENABLE_UD_ */

    value = getenv("MV2_USE_XRC");
    if (value) {
#ifdef _ENABLE_XRC_
        USE_XRC = !!atoi(value);
        if (atoi(value)) {
#ifdef _ENABLE_UD_
            if (rdma_enable_only_ud) {
                PRINT_INFO((pg_rank == 0), "XRC and only UD cannot be set at the same time.\n");
                PRINT_INFO((pg_rank == 0), "Proceeding after disabling XRC.\n");
                USE_XRC = 0;
            } else
#endif /*_ENABLE_UD_*/
            {
#if defined(RDMA_CM) && !defined(CKPT)
                /* XRC is not supported with RDMA_CM. Fall back to basic alltoall CM.
                 * This will get reset to on-demand CM later on in this function. */
                MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
#endif /*defined(RDMA_CM) && !defined(CKPT)*/
                /* Enable on-demand */
                threshold = 0;
            }
            /* RGET is not supporpted with XRC. Use RPUT by default */
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        }
#else
        if (atoi(value)) {
            PRINT_INFO((pg_rank == 0), "XRC support is not configured. Please retry with"
                 "MV2_USE_XRC=0 (or) Reconfigure MVAPICH2 library without --disable-xrc.\n");
            PRINT_INFO((pg_rank == 0), "Proceeding after disabling XRC.\n");
        }
#endif
    }

#if defined(RDMA_CM)
    if (((value = getenv("MV2_USE_RDMA_CM")) != NULL
         || (value = getenv("MV2_USE_IWARP_MODE")) != NULL)
        && atoi(value) && !dpm) {
#if defined (ROMIO_IME)
        PRINT_INFO((pg_rank == 0), "Error: IME FS does not work with RDMA CM. "
                                   "Proceeding without RDMA support.\n");
#else
        MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_RDMA_CM;
#endif
#ifdef _ENABLE_XRC_
        USE_XRC = 0;
        value = getenv("MV2_USE_XRC");
        if (value && (pg_rank == 0)) {
            if (atoi(value)) {
                MPL_error_printf("Error: XRC does not work with RDMA CM. "
                                  "Proceeding without XRC support.\n");
            }
        }
#endif
    }
#endif /* defined(RDMA_CM) */

    if (MPIDI_CH3I_Process.cm_type != MPIDI_CH3I_CM_RDMA_CM) {
        if (pg_size > threshold || dpm
#ifdef _ENABLE_XRC_
            || USE_XRC
#endif /* _ENABLE_XRC_ */
#ifdef _ENABLE_UD_
            || rdma_enable_hybrid
#endif
    ) {
            MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_ON_DEMAND;
            MPIDI_CH3I_Process.num_conn = 0;
        } else {
            MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
        }
    }

    MPIDI_PG_GetConnKVSname(&pg->ch.kvs_name);

#if defined(CKPT)
#if defined(RDMA_CM)
    if (MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_RDMA_CM) {
        MPL_error_printf("Error: Checkpointing does not work with RDMA CM.\n"
                          "Please configure and compile MVAPICH2 with checkpointing disabled "
                          "or without support for RDMA CM.\n");
        MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**fail");
    }
#endif /* defined(RDMA_CM) */

    // Always use CM_ON_DEMAND for Checkpoint/Restart and Migration
    MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_ON_DEMAND;

#endif /* defined(CKPT) */
#ifdef _ENABLE_UD_
    if (rdma_enable_hybrid) {
        MPIU_Assert(MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND);
    }
#endif

#if defined(RDMA_CM) && !defined(CKPT)
    if (MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_RDMA_CM) {
        if((value = getenv("MV2_USE_RDMA_CM")) != NULL && !!atoi(value)) {
            user_selected_rdma_cm = 1;
        }
        setenv("MV2_USE_RDMA_CM", "1", 1);
        if (mv2_MPIDI_CH3I_RDMA_Process.use_iwarp_mode ||
                (((value = getenv("MV2_USE_IWARP_MODE")) != NULL) && !!atoi(value))) {
            check_cq_overflow           = check_cq_overflow_for_iwarp;
            handle_multiple_cqs         = handle_multiple_cqs_for_iwarp;
            MPIDI_CH3I_MRAILI_Cq_poll   = MPIDI_CH3I_MRAILI_Cq_poll_iwarp;
            perform_blocking_progress   = perform_blocking_progress_for_iwarp;
        } else {
            check_cq_overflow           = check_cq_overflow_for_ib;
            handle_multiple_cqs         = handle_multiple_cqs_for_ib;
            MPIDI_CH3I_MRAILI_Cq_poll   = MPIDI_CH3I_MRAILI_Cq_poll_ib;
            perform_blocking_progress   = perform_blocking_progress_for_ib;
        }
    } else 
#endif /* defined(RDMA_CM) && !defined(CKPT) */
    {
        check_cq_overflow           = check_cq_overflow_for_ib;
        handle_multiple_cqs         = handle_multiple_cqs_for_ib;
        MPIDI_CH3I_MRAILI_Cq_poll   = MPIDI_CH3I_MRAILI_Cq_poll_ib;
        perform_blocking_progress   = perform_blocking_progress_for_ib;
    }

    /* save my vc_ptr for easy access */
    MPIDI_PG_Get_vc(pg, pg_rank, &MPIDI_CH3I_Process.vc);

    /* Initialize Progress Engine */
    if ((mpi_errno = MPIDI_CH3I_Progress_init())) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Get parameters from the job-launcher */
    rdma_get_pm_parameters(&mv2_MPIDI_CH3I_RDMA_Process);

    /* Check for SMP only */
    MPIDI_CH3I_set_smp_only();

    if ((value = getenv("MV2_ENABLE_EAGER_THRESHOLD_REDUCTION")) != NULL) {
        mv2_enable_eager_threshold_reduction = !!atoi(value);
    }

    if ((value = getenv("MV2_USE_EAGER_FAST_SEND")) != NULL) {
        mv2_use_eager_fast_send = !!atoi(value);
    }

    if (MPIDI_CH3I_Process.has_dpm) {
        mv2_use_eager_fast_send = 0;
    }

    if ((value = getenv("MV2_POLLING_LEVEL")) != NULL) {
        rdma_polling_level = atoi(value);
    }
    if (!SMP_ONLY) {
        /*
         * Identify local rank and number of local processes
         */
        if (pg->ch.local_process_id == -1) {
            mpi_errno = MPIDI_Get_local_host(pg, pg_rank);
            if (mpi_errno) {
                MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                        "**fail %s", "MPIDI_Get_local_host");
            }
        }

        rdma_local_id = MPIDI_Get_local_process_id(pg);
        rdma_num_local_procs = MPIDI_Num_local_processes(pg);

        /* Reading the values from user first and then allocating the memory */
        mpi_errno = rdma_get_control_parameters(&mv2_MPIDI_CH3I_RDMA_Process);
        if (mpi_errno) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail",
                    "**fail %s", "rdma_get_control_parameters");
        }
        /* Set default values for parameters */
        rdma_set_default_parameters(&mv2_MPIDI_CH3I_RDMA_Process);
        /* Read user defined values for parameters */
        rdma_get_user_parameters(pg_size, pg_rank);

        /* Allocate structures to store CM information
         * This MUST come after reading env vars */
        mpi_errno = MPIDI_CH3I_MRAIL_CM_Alloc(pg);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

#if !defined(DISABLE_PTMALLOC)
        if (mvapich2_minit()) {
            if (pg_rank == 0) {
                MPL_error_printf("WARNING: Error in initializing MVAPICH2 ptmalloc library."
                "Continuing without InfiniBand registration cache support.\n");
            }
            mv2_MPIDI_CH3I_RDMA_Process.has_lazy_mem_unregister = 0;
        }
#else /* !defined(DISABLE_PTMALLOC) */
        mallopt(M_TRIM_THRESHOLD, -1);
        mallopt(M_MMAP_MAX, 0);
        mv2_MPIDI_CH3I_RDMA_Process.has_lazy_mem_unregister = 0;
#endif /* !defined(DISABLE_PTMALLOC) */

        /* Read RDMA FAST Path related params */
        rdma_set_rdma_fast_path_params(pg_size);
        switch (MPIDI_CH3I_Process.cm_type) {
#if defined(RDMA_CM)
            case MPIDI_CH3I_CM_RDMA_CM:
                mpi_errno = MPIDI_CH3I_RDMA_CM_Init(pg, pg_rank, &conn_info);
                if (mpi_errno != MPI_SUCCESS) {
                    if (user_selected_rdma_cm) {
                        /* Print backtrace and exit */
                        MPIR_ERR_POP(mpi_errno);
                    } else if (!pg_rank) {
                        MPL_error_printf("Warning: RDMA CM Initialization failed. "
                                "Continuing without RDMA CM support. "
                                "Please set MV2_USE_RDMA_CM=0 to disable RDMA CM.\n");
                    }
                    /* Fall back to On-Demand CM */
                    ib_finalize_rdma_cm(pg_rank, pg);
                    rdma_default_port = RDMA_DEFAULT_PORT;
                    mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm = 0;
                    mv2_MPIDI_CH3I_RDMA_Process.use_rdma_cm_on_demand = 0;
                } else {
                    break;
                }
#endif /* defined(RDMA_CM) */
            case MPIDI_CH3I_CM_ON_DEMAND:
                MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_ON_DEMAND;
                mpi_errno = MPIDI_CH3I_CM_Init(pg, pg_rank, &conn_info);
                if (mpi_errno != MPI_SUCCESS) {
                    MPIR_ERR_POP(mpi_errno);
                }
                break;
            default:
                /*call old init to setup all connections */
                MPIDI_CH3I_Process.cm_type = MPIDI_CH3I_CM_BASIC_ALL2ALL;
                if ((mpi_errno =
                     MPIDI_CH3I_RDMA_init(pg, pg_rank)) != MPI_SUCCESS) {
                    MPIR_ERR_POP(mpi_errno);
                }

                /* All vc should be connected */
                for (p = 0; p < pg_size; ++p) {
                    MPIDI_PG_Get_vc(pg, p, &vc);
                    vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
                }
                break;
        }
#if defined(RDMA_CM)
    } else {
        /* If SMP_ONLY, we need to get the HCA type */
        rdma_cm_get_hca_type(&mv2_MPIDI_CH3I_RDMA_Process);
#endif /*defined(RDMA_CM)*/
    }

    if ((value = getenv("MV2_RNDV_PROTOCOL")) != NULL) {
        if (strncmp(value, "RPUT", 4) == 0) {
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        } else if (strncmp(value, "RGET", 4) == 0) {
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
        } else {
            rdma_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
        }
    }
    if ((value = getenv("MV2_SMP_RNDV_PROTOCOL")) != NULL) {
        if (strncmp(value, "RPUT", 4) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_RPUT;
        } else if (strncmp(value, "RGET", 4) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_RGET;
        } else if (strncmp(value, "R3", 2) == 0) {
            smp_rndv_protocol = MV2_RNDV_PROTOCOL_R3;
        } else {
            MPL_usage_printf("MV2_SMP_RNDV_PROTOCOL "
                    "must be one of: RPUT, RGET, R3\n");
            smp_rndv_protocol = rdma_rndv_protocol;
        }
    }
    if ((value = getenv("MV2_RNDV_IMMEDIATE")) != NULL) {
        rdma_rndv_immediate = !!atoi(value);
    }
#if defined(CKPT)
#if defined(DISABLE_PTMALLOC)
    MPL_error_printf("Error: Checkpointing does not work without registration "
                      "caching enabled.\nPlease configure and compile MVAPICH2 without checkpointing "
                      " or enable registration caching.\n");
    MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**fail");
#endif /* defined(DISABLE_PTMALLOC) */

    if ((mpi_errno = MPIDI_CH3I_CR_Init(pg, pg_rank, pg_size))) {
        MPIR_ERR_POP(mpi_errno);
    }
#endif /* defined(CKPT) */

    if (conn_info) {
        /* set connection info for dynamic process management */
        if (dpm) {
	        mpi_errno = MPIDI_PG_SetConnInfo(pg_rank, (const char *) conn_info);
	        if (mpi_errno != MPI_SUCCESS) {
	            MPIR_ERR_POP(mpi_errno);
	        }
        }
        MPIU_Free(conn_info);
    }


    struct coll_info colls_arch_hca[colls_max];

    mpi_errno = MV2_collectives_arch_init(mv2_MPIDI_CH3I_RDMA_Process.heterogeneity, colls_arch_hca);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* Initialize the smp channel */
    if ((mpi_errno = MPIDI_CH3I_SMP_init(pg))) {
        MPIR_ERR_POP(mpi_errno);
    }

    if (SMP_INIT) {
        for (p = 0; p < pg_size; ++p) {
            MPIDI_PG_Get_vc(pg, p, &vc);
            /* Mark the SMP VC as Idle */
            if (vc->smp.local_nodes >= 0) {
                vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE;
                /* Enable fast send */
                if (mv2_use_eager_fast_send) {
                    vc->eager_fast_fn = mv2_smp_fast_write_contig;
                }
                if (SMP_ONLY) {
                    MPIDI_CH3I_SMP_Init_VC(vc);
                }
#ifdef _ENABLE_XRC_
                VC_XST_SET(vc, XF_SMP_VC);
#endif
            }
        }
    } else {
        extern int mv2_enable_shmem_collectives;
        mv2_enable_shmem_collectives = SMP_INIT;
    }

    /* Allocate and Init Dummy request */
    mpi_errno = mv2_create_dummy_request();

    /* Set the eager max msg size now that we know SMP and RDMA are initialized.
     * The max message size is also set during VC initialization, but the state
     * of SMP is unknown at that time.
     */
    for (p = 0; p < pg_size; ++p) {
        MPIDI_PG_Get_vc(pg, p, &vc);
        vc->eager_max_msg_sz = MPIDI_CH3_EAGER_MAX_MSG_SIZE(vc);
        if (mv2_use_eager_fast_send) {
            vc->eager_fast_max_msg_sz = MPIDI_CH3_EAGER_FAST_MAX_MSG_SIZE(vc);
        } else {
            vc->eager_fast_max_msg_sz = 0;
        }
    }

    if ((value = getenv("MV2_SHOW_ENV_INFO")) != NULL) {
        mv2_show_env_info = atoi(value);
    }
    if (pg_rank == 0 && mv2_show_env_info) {
        mv2_print_env_info(&mv2_MPIDI_CH3I_RDMA_Process, colls_arch_hca);
    }

#if defined(_MCST_SUPPORT_)
    if (!SMP_ONLY && rdma_enable_mcast) {
        mv2_rdma_init_timers = 1;
        /* TODO : Is there a better way to seed? */
        srand(time(NULL) * pg_rank);

        /* initialize comm table */
        for (p = 0; p < MV2_MCAST_MAX_COMMS; p++) {
            comm_table[p] = NULL;
        }
        /* init mcast context */
        if (mcast_ctx == NULL) {
            mcast_ctx = MPIU_Malloc (sizeof(mcast_context_t));
            if (mcast_ctx == NULL) {
                MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_NO_MEM, goto fn_fail,
                        "**fail", "**fail %s",
                        "Failed to allocate resources for multicast");
            }
            mcast_ctx->selected_rail = 0;
            PRINT_DEBUG(DEBUG_MCST_verbose>1,"mcast using default rail:"
                    " %d\n",mcast_ctx->selected_rail);
        }
        mcast_ctx->init_list = NULL;
#if defined(RDMA_CM) 
        if(rdma_use_rdma_cm_mcast == 1){
            int ret = 0;
            mcast_ctx->src_addr = (struct sockaddr *) &(mcast_ctx->src_in);
            PRINT_DEBUG(DEBUG_MCST_verbose>1,"RDMA CM mcast source ip"
                   " address:%s\n",ip_address_enabled_devices[mcast_ctx->ip_index].ip_address);
            
            ret = mv2_rdma_cm_mcst_get_addr_info(ip_address_enabled_devices[mcast_ctx->ip_index].ip_address,
                    (struct sockaddr *) &mcast_ctx->src_in);
            if(ret){
                if(MPIDI_Process.my_pg_rank == 0) { 
                    PRINT_ERROR("[Warning]: get src addr failed: not using rdma cm"
                            " based mcast\n");
                }
                rdma_use_rdma_cm_mcast = 0;
            }
        }
#endif /* #if defined(RDMA_CM) */
        mcast_ctx->ud_ctx = mv2_mcast_prepare_ud_ctx();
        if (mcast_ctx->ud_ctx == NULL) {
            MPIR_ERR_SETFATALANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                    "**fail", "**fail %s",
                    "Error in create multicast UD context for multicast");
        }
        PRINT_DEBUG(DEBUG_MCST_verbose,"Created multicast UD context \n");
    }
#endif

#if defined(_SHARP_SUPPORT_)
    if ((value = getenv("MV2_ENABLE_SHARP")) != NULL) {
        mv2_enable_sharp_coll = atoi(value);
    } else {
        mv2_enable_sharp_coll = MPIR_CVAR_ENABLE_SHARP; 
    }
    if ((value = getenv("MV2_SHARP_PORT")) != NULL) {
        mv2_sharp_port = atoi(value);
    }
    if ((value = getenv("MV2_SHARP_HCA_NAME")) != NULL) {
        mv2_sharp_hca_name = MPIU_Malloc(sizeof(value));
        MPIU_Memcpy(mv2_sharp_hca_name, value, sizeof(value));
    }
#endif

    if (mv2_rdma_init_timers) {
        mv2_init_timers();
    }

    mpi_errno = MPIDI_CH3U_Comm_register_destroy_hook(MPIDI_CH3I_comm_destroy, NULL);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    if (MPIDI_CH3I_Process.cm_type == MPIDI_CH3I_CM_ON_DEMAND) {
        if (g_atomics_support || ((rdma_use_blocking) && (pg_size > threshold))) {
            MPIDI_PG_Get_vc(pg, pg_rank, &vc);
            MPIDI_CH3I_CM_Connect_self(vc);
        }
    }

  fn_exit:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_INIT);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_VC_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_VC_Init(MPIDI_VC_t * vc)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_VC_INIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_VC_INIT);
    int mpi_errno = MPI_SUCCESS;
    vc->smp.local_nodes = -1;
#if !defined (CHANNEL_PSM)
    vc->smp.sendq_head = NULL;
    vc->smp.sendq_tail = NULL;
    vc->smp.recv_active = NULL;
    vc->smp.send_active = NULL;
    vc->ch.req = NULL;
    vc->mrail.rails = NULL;
    vc->mrail.srp.credits = NULL;
    vc->mrail.cmanager.msg_channels = NULL;
#endif /* #if !defined (CHANNEL_PSM) */
    vc->ch.sendq_head = NULL;
    vc->ch.sendq_tail = NULL;
    vc->ch.req = (MPID_Request *) MPIU_Malloc(sizeof(MPID_Request));
    if (!vc->ch.req) {
        MPIU_CHKMEM_SETERR(mpi_errno, sizeof(MPID_Request), "MPID Request");
    }
    /* vc->ch.state = MPIDI_CH3I_VC_STATE_IDLE; */
    vc->ch.state = MPIDI_CH3I_VC_STATE_UNCONNECTED;
    vc->ch.read_state = MPIDI_CH3I_READ_STATE_IDLE;
    vc->ch.recv_active = NULL;
    vc->ch.send_active = NULL;
    vc->ch.cm_sendq_head = NULL;
    vc->ch.cm_sendq_tail = NULL;
    vc->ch.cm_1sc_sendq_head = NULL;
    vc->ch.cm_1sc_sendq_tail = NULL;
    vc->ch.pending_r3_data = 0;
    vc->ch.received_r3_data = 0;
#ifdef _ENABLE_XRC_
    vc->ch.xrc_flags = 0;
    vc->ch.xrc_conn_queue = NULL;
    vc->ch.orig_vc = NULL;
    memset(vc->ch.xrc_srqn, 0, sizeof(uint32_t) * MAX_NUM_HCAS);
    memset(vc->ch.xrc_rqpn, 0, sizeof(uint32_t) * MAX_NUM_SUBRAILS);
    memset(vc->ch.xrc_my_rqpn, 0, sizeof(uint32_t) * MAX_NUM_SUBRAILS);
#endif

    vc->smp.hostid = -1;
    vc->force_rndv = 0;

    vc->rndvSend_fn = MPID_MRAIL_RndvSend;
    vc->rndvRecv_fn = MPID_MRAIL_RndvRecv;

#if defined(CKPT)
    vc->ch.rput_stop = 0;
#endif /* defined(CKPT) */

#ifdef USE_RDMA_UNEX
    vc->ch.unex_finished_next = NULL;
    vc->ch.unex_list = NULL;
#endif
    /* It is needed for temp vc */
    vc->eager_max_msg_sz = rdma_iba_eager_threshold;

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_VC_INIT);
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PortFnsInit
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_PortFnsInit(MPIDI_PortFns * portFns)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_RDMA_PORTFNSINIT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_RDMA_PORTFNSINIT);

    if (!MPIDI_CH3I_Process.has_dpm) {
        portFns->OpenPort = 0;
        portFns->ClosePort = 0;
        portFns->CommAccept = 0;
        portFns->CommConnect = 0;
    } else
        MPIU_UNREFERENCED_ARG(portFns);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_RDMA_PORTFNSINIT);
    return MPI_SUCCESS;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Connect_to_root
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Connect_to_root(const char *port_name, MPIDI_VC_t ** new_vc)
{
    int mpi_errno = MPI_SUCCESS;
    int str_errno;
    char ifname[MAX_HOST_DESCRIPTION_LEN];
    MPIDI_VC_t *vc;
    MPIDI_CH3_Pkt_cm_establish_t pkt;
    MPID_Request *sreq;
    int seqnum;

    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_CONNECT_TO_ROOT);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_CONNECT_TO_ROOT);

    *new_vc = NULL;
    if (!MPIDI_CH3I_Process.has_dpm)
        return MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME,
                                    __LINE__, MPI_ERR_OTHER, "**notimpl", 0);

    str_errno = MPIU_Str_get_string_arg(port_name,
                                        MPIDI_CH3I_HOST_DESCRIPTION_KEY,
                                        ifname, MAX_HOST_DESCRIPTION_LEN);
    if (str_errno != MPIU_STR_SUCCESS) {
        /* --BEGIN ERROR HANDLING */
        if (str_errno == MPIU_STR_FAIL) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER,
                                "**argstr_missinghost");
        } else {
            /* MPIU_STR_TRUNCATED or MPIU_STR_NONEM */
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**argstr_hostd");
        }
        /* --END ERROR HANDLING-- */
    }

    vc = MPIU_Malloc(sizeof(MPIDI_VC_t));
    if (!vc) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
    }
    MPIDI_VC_Init(vc, NULL, 0);

    mpi_errno = MPIDI_CH3I_CM_Connect_raw_vc(vc, ifname);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    while (vc->ch.state != MPIDI_CH3I_VC_STATE_IDLE) {
        mpi_errno = MPID_Progress_test();
        /* --BEGIN ERROR HANDLING-- */
        if (mpi_errno != MPI_SUCCESS) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    /* fprintf(stderr, "[###] vc state to idel, now send cm_establish msg\n") */
    /* Now a connection is created, send a cm_establish message */
    /* FIXME: vc->mrail.remote_vc_addr is used to find remote vc
     * A more elegant way is needed */
    MPIDI_Pkt_init(&pkt, MPIDI_CH3_PKT_CM_ESTABLISH);
    MPIDI_VC_FAI_send_seqnum(vc, seqnum);
    MPIDI_Pkt_set_seqnum(&pkt, seqnum);
    pkt.vc_addr = vc->mrail.remote_vc_addr;
    mpi_errno = MPIDI_GetTagFromPort(port_name, &pkt.port_name_tag);
    if (mpi_errno != MPIU_STR_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**argstr_port_name_tag");
    }

    mpi_errno = MPIDI_CH3_iStartMsg(vc, &pkt, sizeof(pkt), &sreq);
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**fail", "**fail %s",
                             "Failed to send cm establish message");
    }

    if (sreq != NULL) {
        if (sreq->status.MPI_ERROR != MPI_SUCCESS) {
            mpi_errno = MPIR_Err_create_code(sreq->status.MPI_ERROR,
                                             MPIR_ERR_FATAL, FCNAME, __LINE__,
                                             MPI_ERR_OTHER, "**fail", 0);
            MPID_Request_release(sreq);
            goto fn_fail;
        }
        MPID_Request_release(sreq);
    }

    *new_vc = vc;

  fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_CONNECT_TO_ROOT);

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_Get_business_card
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_Get_business_card(int myRank, char *value, int length)
{
    char ifname[MAX_HOST_DESCRIPTION_LEN];
    int mpi_errno;
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_GET_BUSINESS_CARD);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_GET_BUSINESS_CARD);

    mpi_errno = MPIDI_CH3I_CM_Get_port_info(ifname, MAX_HOST_DESCRIPTION_LEN);
    if (mpi_errno) {
        MPIR_ERR_POP(mpi_errno);
    }

    mpi_errno = MPIU_Str_add_string_arg(&value, &length,
                                        MPIDI_CH3I_HOST_DESCRIPTION_KEY,
                                        ifname);
    if (mpi_errno != MPIU_STR_SUCCESS) {
        if (mpi_errno == MPIU_STR_NOMEM) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**buscard_len");
        } else {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**buscard");
        }
    }

  fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_GET_BUSINESS_CARD);
    return mpi_errno;
}

/* This routine is a hook for initializing information for a process
   group before the MPIDI_CH3_VC_Init routine is called */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PG_Init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_PG_Init(MPIDI_PG_t * pg)
{
    char *value     = NULL;
    int mpi_errno   = MPI_SUCCESS;

    if ((value = getenv("MV2_SHMEM_BACKED_UD_CM")) != NULL) {
        mv2_shmem_backed_ud_cm = !!atoi(value);
    }
    if ((value = getenv("MV2_SUPPORT_DPM")) != NULL) {
        if (!!atoi(value)) {
            mv2_shmem_backed_ud_cm = 0;
        }
    }

    pg->ch.mrail = MPIU_Malloc(sizeof(MPIDI_CH3I_MRAIL_CM_t));
    if (pg->ch.mrail == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**nomem",
                "**nomem %s", "ud_cm mrail");
    }
    MPIU_Memset(pg->ch.mrail, 0, sizeof(MPIDI_CH3I_MRAIL_CM_t));

    pg->ch.mrail->cm_ah = MPIU_Malloc(pg->size * sizeof(struct ibv_ah *));
    if (pg->ch.mrail->cm_ah == NULL) {
        MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**nomem",
                "**nomem %s", "cm_ah");
    }
    MPIU_Memset(pg->ch.mrail->cm_ah, 0, pg->size * sizeof(struct ibv_ah *));

    if (!mv2_shmem_backed_ud_cm) {
        pg->ch.mrail->cm_shmem.ud_cm =
                MPIU_Malloc(pg->size * sizeof(MPIDI_CH3I_MRAIL_UD_CM_t));
        if (pg->ch.mrail->cm_shmem.ud_cm == NULL) {
            MPIR_ERR_SETFATALANDJUMP1(mpi_errno, MPI_ERR_INTERN, "**nomem",
                    "**nomem %s", "ud_cm");
        }
        MPIU_Memset(pg->ch.mrail->cm_shmem.ud_cm, 0,
                    pg->size * sizeof(MPIDI_CH3I_MRAIL_UD_CM_t));
    }

fn_fail:
    return MPI_SUCCESS;
}

/* This routine is a hook for any operations that need to be performed before
   freeing a process group */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_PG_Destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_PG_Destroy(struct MPIDI_PG *pg)
{
    return MPIDI_CH3I_MRAIL_CM_Dealloc(pg);
}

/* This routine is a hook for any operations that need to be performed before
   freeing a virtual connection */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_VC_Destroy
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_VC_Destroy(struct MPIDI_VC *vc)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_VC_DESTROY);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_VC_DESTROY);

#if !defined (CHANNEL_PSM)
    if (vc->smp.sendq_head != NULL) {
        MPIU_Free(vc->smp.sendq_head);
    }
    if (vc->smp.sendq_tail != NULL) {
        MPIU_Free(vc->smp.sendq_tail);
    }
    if (vc->smp.recv_active != NULL) {
        MPIU_Free(vc->smp.recv_active);
    }
    if (vc->smp.send_active != NULL) {
        MPIU_Free(vc->smp.send_active);
    }
    if (vc->ch.req != NULL) {
        MPIU_Free(vc->ch.req);
    }
    if (vc->mrail.cmanager.msg_channels != NULL) {
        MPIU_Free(vc->mrail.cmanager.msg_channels);
    }
    if (vc->mrail.srp.credits != NULL) {
        MPIU_Free(vc->mrail.srp.credits);
    }
    if (vc->mrail.rails != NULL) {
        MPIU_Free(vc->mrail.rails);
    }
#endif /* #if !defined (CHANNEL_PSM) */


    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_VC_DESTROY);
    return MPI_SUCCESS;
}

/* A dummy function so that all channels provide the same set of functions,
   enabling dll channels */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3_InitCompleted
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIDI_CH3_InitCompleted(void)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3_INITCOMPLETED);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3_INITCOMPLETED);

    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3_INITCOMPLETED);
    return MPI_SUCCESS;
}


void rdma_process_hostid(MPIDI_PG_t * pg, int *host_ids, int my_rank,
                         int pg_size)
{
    int i;
    int my_host_id;;
    MPIDI_VC_t *vc = NULL;

    pg->ch.local_process_id = 0;
    pg->ch.num_local_processes = 0;

    my_host_id = host_ids[my_rank];
    for (i = 0; i < pg_size; ++i) {
        MPIDI_PG_Get_vc(pg, i, &vc);
        if (host_ids[i] == my_host_id) {
            vc->smp.local_rank = pg->ch.num_local_processes++;
            if (i == my_rank) {
                pg->ch.local_process_id = vc->smp.local_rank;
            }
        } else {
            vc->smp.local_rank = -1;
        }
    }
}

/* vi: set sw=4 */
