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
#include <unistd.h>
#include "psmpriv.h"
#include "psm_vbuf.h"
#include <dirent.h>
#include "coll_shmem.h"
#include "debug_utils.h"
#include "mv2_utils.h"
#include <mv2_arch_hca_detect.h>
#include <upmi.h>
#include "coll_shmem.h"
extern int g_mv2_num_cpus;

volatile unsigned int MPIDI_CH3I_progress_completion_count = 0; //ODOT: what is this ?
volatile int MPIDI_CH3I_progress_blocked = FALSE;
volatile int MPIDI_CH3I_progress_wakeup_signalled = FALSE;

/* Globals */
/* psm device instance */
struct psmdev_info_t    psmdev_cw;
uint32_t                ipath_rndv_thresh = DEFAULT_IPATH_RNDV_THRESH;
uint32_t                mv2_hfi_rndv_thresh = DEFAULT_PSM_HFI_RNDV_THRESH;
uint32_t                mv2_shm_rndv_thresh = DEFAULT_PSM_SHM_RNDV_THRESH;
uint8_t                 ipath_debug_enable;
uint32_t                ipath_dump_frequency;
uint8_t                 ipath_enable_func_lock;
uint32_t                ipath_progress_yield_count;
size_t                  ipath_max_transfer_size = DEFAULT_IPATH_MAX_TRANSFER_SIZE;
int g_mv2_show_env_info = 0;
int mv2_psm_bcast_uuid  = 0;
int mv2_use_pmi_ibarrier = 0;
int mv2_use_on_demand_cm = 0;
int mv2_homogeneous_cluster = 0;
int mv2_on_demand_threshold = MPIDI_PSM_DEFAULT_ON_DEMAND_THRESHOLD;
mv2_arch_hca_type g_mv2_arch_hca_type = 0;

/* Number of retry attempts if psm_ep_open fails */
static int mv2_psm_ep_open_retry_count = 10;
/* Number of seconds to sleep between psm_ep_open retries */
static int mv2_psm_ep_open_retry_secs  = 10;

int mv2_pmi_max_keylen;
int mv2_pmi_max_vallen;
char *mv2_pmi_key;
char *mv2_pmi_val;

static char    scratch[WRBUFSZ];
static char             *kvsid;
static PSM_UUID_T       psm_uuid;

static void psm_read_user_params(void);
static int  psm_bcast_uuid(int pg_size, int pg_rank);
static int  psm_create_uuid(void);
static int  psm_start_epid_exchange(PSM_EPID_T myid, int pg_size, int pg_rank);
static void psm_other_init(MPIDI_PG_t *pg);
static void psm_preinit(MPIDI_PG_t *pg);
static int  decode(unsigned s_len, char *src, unsigned d_len, char *dst);
static int  encode(unsigned s_len, char *src, unsigned d_len, char *dst);
static int psm_connect_alltoall(PSM_EPADDR_T *addrs, int pg_size, int pg_rank);
static int psm_detect_heterogeneity(mv2_arch_hca_type myarch, int pg_size, int pg_rank);
static unsigned int psm_hash_str(char *str);

extern void MPIDI_CH3I_SHMEM_COLL_Cleanup();

/* ensure that all procs have completed their call to psm_mq_init */
static int psm_mq_init_barrier(PSM_MQ_T mq, int rank, int ranks, PSM_EPADDR_T* addrs)
{
    int tmp_rc;
    int rc = PSM_OK;

    /* implement barrier dissemination algorithm */
    int dist = 1;
    while (dist < ranks) {
        /* compute rank of source for this phase */
        int src = rank - dist;
        if (src < 0) {
            src += ranks;
        }

        /* compute rank of destination for this phase */
        int dst = rank + dist;
        if (dst >= ranks) {
            dst -= ranks;
        }

        /* post non-blocking receive for message with tag equal to source rank plus one */
        #if PSM_VERNO >= PSM_2_1_VERSION
            psm2_mq_tag_t rtag, rtagsel;
            rtagsel.tag0 = MQ_TAGSEL_ALL;
            rtagsel.tag1 = MQ_TAGSEL_ALL;
            rtagsel.tag2 = MQ_TAGSEL_ALL;

        #else
            uint64_t rtag;
            uint64_t rtagsel = MQ_TAGSEL_ALL;
        #endif

        MAKE_PSM_SELECTOR(rtag, 0, 0, (src+1));

        PSM_MQ_REQ_T request;

        tmp_rc = PSM_IRECV(mq, rtag, rtagsel, MQ_FLAGS_NONE, NULL, 0, NULL, &request);
        if (tmp_rc != PSM_OK) {
            rc = tmp_rc;
        }

        /* post blocking send to destination, set tag to be our rank plus one */
        #if PSM_VERNO >= PSM_2_1_VERSION
            psm2_mq_tag_t stag;
        #else
            uint64_t stag;
        #endif

        MAKE_PSM_SELECTOR(stag, 0, 0, (rank+1));

        tmp_rc = PSM_SEND(mq, addrs[dst], MQ_FLAGS_NONE, stag, NULL, 0);
        if (tmp_rc != PSM_OK) {
            rc = tmp_rc;
        }

        /* wait on non-blocking receive to complete */
        tmp_rc = PSM_WAIT(&request, NULL);

        if (tmp_rc != PSM_OK) {
            rc = tmp_rc;
        }

        /* increase our distance by a factor of two */
        dist <<= 1;
    }

    return rc;
}

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

void mv2_print_env_info(struct coll_info *colls_arch_hca)
{
    int i = 0;

    mv2_arch_type arch_type = MV2_GET_ARCH(g_mv2_arch_hca_type);
    mv2_hca_type hca_type = MV2_GET_HCA(g_mv2_arch_hca_type);
    mv2_cpu_family_type family_type = mv2_get_cpu_family();

    fprintf(stderr, "\n MVAPICH2-%s Parameters\n", MPIR_Version_string);
    fprintf(stderr,
            "---------------------------------------------------------------------\n");
    fprintf(stderr, "\tPROCESSOR ARCH NAME            : %s\n",
            mv2_get_arch_name(arch_type));
    fprintf(stderr, "\tPROCESSOR FAMILY NAME          : %s\n",
            mv2_get_cpu_family_name(family_type));
    fprintf(stderr, "\tPROCESSOR MODEL NUMBER         : %d\n",
            mv2_get_cpu_model());
    fprintf(stderr, "\tHCA NAME                       : %s\n",
            mv2_get_hca_name(hca_type));
    fprintf(stderr, "\tHeterogeneity                  : %s\n",
            (!mv2_homogeneous_cluster) ? "YES" : "NO");
    fprintf(stderr, "\tSMP Eagersize                  : %d\n",
            mv2_shm_rndv_thresh);
    fprintf(stderr, "\tHFI Eagersize                  : %d\n",
            mv2_hfi_rndv_thresh);
    fprintf(stderr, "\t%s                  : %s %s\n", "Tuning Table:", mv2_get_arch_name(colls_arch_hca[0].arch_type), mv2_get_hca_name(colls_arch_hca[0].hca_type));
    fprintf(stderr, "---------------------------------------------------------------------\n");
    
    if (g_mv2_show_env_info >= 2) {
        fprintf(stderr, "\tMV2_ENABLE_SOCKET_AWARE_COLLECTIVES  : %d\n", mv2_enable_socket_aware_collectives);
        fprintf(stderr, "\tMV2_USE_SOCKET_AWARE_ALLREDUCE       : %d\n", mv2_use_socket_aware_allreduce);
        fprintf(stderr, "\tMV2_USE_SOCKET_AWARE_BARRIER         : %d\n", mv2_use_socket_aware_barrier);
        fprintf(stderr, "\tMV2_USE_SOCKET_AWARE_SHARP_ALLREDUCE : %d\n", mv2_use_socket_aware_sharp_allreduce);
        fprintf(stderr, "\tMV2_SOCKET_AWARE_ALLREDUCE_MIN_MSG   : %d\n", mv2_socket_aware_allreduce_min_msg);
        fprintf(stderr, "\tMV2_SOCKET_AWARE_ALLREDUCE_MAX_MSG   : %d\n", mv2_socket_aware_allreduce_max_msg);
        fprintf(stderr, "---------------------------------------------------------------------\n");
    }

    if (g_mv2_show_env_info >= 3) {
        fprintf(stderr, "\nCollective Tuning Tables\n");
        fprintf(stderr, "\t%-20s %-40s %-40s\n", "Collective", "Architecture", "Interconnect");
        for(i = 0; i < colls_max; i++) {
            fprintf(stderr, "\t%-20s %-40s %-40s\n", collective_names[i], mv2_get_arch_name(colls_arch_hca[i].arch_type), mv2_get_hca_name(colls_arch_hca[i].hca_type));
        }
        fprintf(stderr, "\n---------------------------------------------------------------------\n");
    }
}

#undef FUNCNAME
#define FUNCNAME MV2_get_arch_hca_type
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
mv2_arch_hca_type MV2_get_arch_hca_type(void)
{
    if(g_mv2_arch_hca_type)
        return g_mv2_arch_hca_type;

#if defined(HAVE_LIBIBVERBS)
    int num_devices = 0, i;
    struct ibv_device **dev_list = NULL;
    mv2_hca_type hca_type = 0;
    mv2_arch_type arch_type = 0;
    dev_list = ibv_get_device_list(&num_devices);

    for(i=0; i<num_devices; i++){
        hca_type = mv2_get_hca_type(dev_list[i]);
        if(MV2_IS_INTEL_CARD(hca_type) || MV2_IS_QLE_CARD(hca_type))
            break;
    }

    if(i == num_devices)
        hca_type = MV2_HCA_ANY;

    arch_type = mv2_get_arch_type();
    g_mv2_arch_hca_type = arch_type;
    g_mv2_arch_hca_type <<= 16;
    g_mv2_arch_hca_type |= hca_type;
    g_mv2_arch_hca_type <<= 16;
    g_mv2_arch_hca_type |= g_mv2_num_cpus;

    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
#else
    g_mv2_arch_hca_type = mv2_get_arch_hca_type(NULL);
#endif
    return g_mv2_arch_hca_type;
}

/* print error string to stderr, flush stderr, and return error */
static PSM_ERROR_T mv2_psm_err_handler(PSM_EP_T ep, const PSM_ERROR_T error,
        const char* error_string, PSM_ERROR_TOKEN_T token)
{
    /* print error and flush stderr */
    PRINT_ERROR("PSM error handler: %s : %s\n",
                PSM_ERROR_GET_STRING(error), error_string);
    return error;
}

#undef FUNCNAME
#define FUNCNAME psm_doinit
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_doinit(int has_parent, MPIDI_PG_t *pg, int pg_rank)
{
    char *flag = NULL;
    int verno_major, verno_minor;
    int pg_size, mpi_errno;
    int heterogeneity, i; 
    PSM_ERROR_T err;
    struct PSM_EP_OPEN_OPTS psm_opts;

    /* Override split_type */
    MPID_Comm_fns = &comm_fns;

    pg_size = MPIDI_PG_Get_size(pg);
    MPIU_Assert(pg_rank < pg_size);
    MPIDI_PG_GetConnKVSname(&kvsid);
    psmdev_cw.pg_rank = pg_rank;
    psmdev_cw.pg_size = pg_size;
    verno_major = PSM_VERNO_MAJOR;
    verno_minor = PSM_VERNO_MINOR;

    mv2_allocate_pmi_keyval();
    psm_read_user_params();
    if(pg_size > mv2_on_demand_threshold) {
        mv2_use_on_demand_cm = 1;
    }

    mpi_errno = MPIDI_CH3U_Comm_register_create_hook(MPIDI_CH3I_comm_create, NULL);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* detect architecture and hca type */
    g_mv2_arch_hca_type = MV2_get_arch_hca_type();

    /* Detect heterogeneity if not overriden by user */
    psm_detect_heterogeneity(g_mv2_arch_hca_type, pg_size, pg_rank);

    /* initialize tuning-table for collectives. 
     * Its ok to pass heterogeneity as 0. We anyway fall-back to the 
     * basic case for PSM */ 
    heterogeneity = !mv2_homogeneous_cluster;
    struct coll_info colls_arch_hca[colls_max];
    mpi_errno = MV2_collectives_arch_init(heterogeneity, colls_arch_hca); 
    if (mpi_errno != MPI_SUCCESS) {
        MPIR_ERR_POP(mpi_errno);
    }

    /* initialize shared memory for collectives */
    if (mv2_enable_shmem_collectives) {
        if ((mpi_errno = MPIDI_CH3I_SHMEM_COLL_init(pg, pg->ch.local_process_id)) != MPI_SUCCESS)
        {
            mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                   FCNAME, __LINE__, MPI_ERR_OTHER, "**fail",
                   "%s", "SHMEM_COLL_init failed");
            goto cleanup_files;
        }

        UPMI_BARRIER();

        /* Memory Mapping shared files for collectives*/
        if ((mpi_errno = MPIDI_CH3I_SHMEM_COLL_Mmap(pg, pg->ch.local_process_id)) != MPI_SUCCESS)
        {
           mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPI_ERR_OTHER,
                 FCNAME, __LINE__, MPI_ERR_OTHER, "**fail", "%s",
                 "SHMEM_COLL_Mmap failed");
           goto cleanup_files;
        }

        MPIDI_CH3I_SHMEM_COLL_Unlink();
    }  

    if (mv2_psm_bcast_uuid) {
        mpi_errno = psm_bcast_uuid(pg_size, pg_rank);
    } else {
        mpi_errno = psm_create_uuid();
    }
    if(mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }

    psm_preinit(pg);

    /* override global error handler so we can print error messages */
    PSM_ERROR_REGISTER_HANDLER(NULL, mv2_psm_err_handler);

    err = PSM_INIT(&verno_major, &verno_minor);
    if(err != PSM_OK) {
        #if PSM_VERNO >= PSM_2_1_VERSION
            fprintf(stderr, "psm2_init failed with error: %s\n", PSM_ERROR_GET_STRING(err));
        #else
            fprintf(stderr, "psm_init failed with error: %s\n", PSM_ERROR_GET_STRING(err));
        #endif

        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**psminit");
    }

    /* By default, PSM sets cpu affinity on a process if it's not
     * already set.  We disable cpu affinity in PSM here.  MVAPICH
     * or the process launcher will set affinity, unless the user
     * disabled it, but in that case, he probably doesn't want
     * PSM to set it either.
     */
    PSM_EP_OPEN_OPTS_GET_DEFAULTS(&psm_opts);
    psm_opts.affinity = PSM_EP_OPEN_AFFINITY_SKIP;

    /* number of times to retry psm_ep_open upon failure */
    if ((flag = getenv("MV2_PSM_EP_OPEN_RETRY_COUNT")) != NULL) {
        int value = atoi(flag);

        if (value >= 0) {
            mv2_psm_ep_open_retry_count = value;
        } else {
            PRINT_ERROR("MV2_WARNING: Attempted to set "
                    "MV2_PSM_EP_OPEN_RETRY_COUNT to invalid value [%s]\n",
                    flag);
            PRINT_ERROR("MV2_WARNING: Using default value of `%d' instead\n",
                    mv2_psm_ep_open_retry_count);
        }
    }

    /* sleep time in seconds between open retries */
    if ((flag = getenv("MV2_PSM_EP_OPEN_RETRY_SECS")) != NULL) {
        int value = atoi(flag);

        if (value > 0) {
            mv2_psm_ep_open_retry_secs = value;
        } else {
            PRINT_ERROR("MV2_WARNING: Attempted to set "
                    "MV2_PSM_EP_OPEN_RETRY_SECS to invalid value [%s]\n",
                    flag);
            PRINT_ERROR("MV2_WARNING: Using default value of `%d' instead\n",
                    mv2_psm_ep_open_retry_secs);
        }
    }

    int attempts = 0;
    do {
        if (err != PSM_OK) {
            PRINT_ERROR("MV2_WARNING: Failed to open an end-point: %s,"
                        " retry attempt %d of %d in %d seconds\n",
                        PSM_ERROR_GET_STRING(err), attempts,
                        mv2_psm_ep_open_retry_count, mv2_psm_ep_open_retry_secs);
            sleep(mv2_psm_ep_open_retry_secs);
        }
        err = PSM_EP_OPEN(psm_uuid, &psm_opts, &psmdev_cw.ep, &psmdev_cw.epid);
        attempts++;
    } while ((err != PSM_OK) && (attempts <= mv2_psm_ep_open_retry_count));
    if (err != PSM_OK) {
        fprintf(stderr, "psm_ep_open failed with error %s\n",
                PSM_ERROR_GET_STRING(err));
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**psmepopen");
    }

    mpi_errno = psm_start_epid_exchange(psmdev_cw.epid, pg_size, pg_rank);
    if(mpi_errno != MPI_SUCCESS) {
        goto fn_fail;
    }

    psmdev_cw.epaddrs = (PSM_EPADDR_T *) MPIU_Malloc(pg_size * sizeof(PSM_EPADDR_T));
    if(psmdev_cw.epaddrs == NULL) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_NO_MEM, "**psmnomem");
    }
    MPIU_Memset(psmdev_cw.epaddrs, 0, pg_size * sizeof(PSM_EPADDR_T));

    if((err = PSM_MQ_INIT(psmdev_cw.ep, PSM_MQ_ORDERMASK_ALL, NULL, 0,
                &psmdev_cw.mq)) != PSM_OK) {
        PRINT_ERROR("psm_mq_init failed\n");
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_INTERN, "**psm_mqinitfailed");
    }

    if (!mv2_use_on_demand_cm) {
        psm_connect_alltoall(psmdev_cw.epaddrs, pg_size, pg_rank);

        /* execute barrier to ensure all tasks have returned from psm_ep_connect */
        if((err = psm_mq_init_barrier(psmdev_cw.mq, pg_rank, pg_size, psmdev_cw.epaddrs)) != PSM_OK) {
            PRINT_ERROR("psm_mq_init_barrier failed\n");
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_INTERN, "**fail");
        }
    }

    /* initialize VC state, eager size value, queues etc */
    psm_other_init(pg);

    if(0==pg_rank && g_mv2_show_env_info){
        mv2_print_env_info(colls_arch_hca);
    }

    mpi_errno = MPIDI_CH3U_Comm_register_destroy_hook(MPIDI_CH3I_comm_destroy, NULL);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* Initialize progress hook slots */
    for (i = 0; i < MAX_PROGRESS_HOOKS; i++) {
        progress_hooks[i].func_ptr = NULL;
        progress_hooks[i].active = FALSE;
    }

    return MPI_SUCCESS;

cleanup_files:
    MPIDI_CH3I_SHMEM_COLL_Cleanup();
fn_fail:
    return MPI_ERR_INTERN;
}

/*  handle special psm init. PSM_DEVICES init, version test for setting
 *  MPI_LOCALRANKS, MPI_LOCALRANKID 
 *  Updated on Fed 2 2010 based on patch provided by Ben Truscott. Refer to 
 *  TRAC Ticket #457 i
 *  Updated on Jan 3 2018 to remove unnecessary barriers */
static void psm_preinit(MPIDI_PG_t *pg)
{
    int id, n;
    int pg_size, universesize;

    pg_size = MPIDI_PG_Get_size(pg);
    id = pg->ch.local_process_id;
    n = pg->ch.num_local_processes;

    if(pg_size > 0)
        universesize = pg_size;
    else
        universesize = 1; /*May be started without mpiexec.*/

    PRINT_DEBUG(DEBUG_CM_verbose, "localid %d localranks %d\n", id, n);

    /* We should not override user settings for these parameters. 
     * This might cause problems with the new greedy context acquisition 
     * when multiple jobs share the same node. Refer to TRAC Ticket #457
     * putenv("PSM_SHAREDCONTEXTS=1");
     * putenv("PSM_SHAREDCONTEXTS_MAX=16");*/

    /* for psm versions 2.0 or later, hints are needed for context sharing */
    if(PSM_VERNO_MAJOR >= PSM_2_VERSION_MAJOR) {
        snprintf(scratch, sizeof(scratch), "%d", n);
        setenv("MPI_LOCALNRANKS", scratch, 1);
        snprintf(scratch, sizeof(scratch), "%d", id);
        setenv("MPI_LOCALRANKID", scratch, 1);

        /* Should not override user settings. Updating to handle all 
         * possible scenarios. Refer to TRAC Ticket #457 */
        #if PSM_VERNO >= PSM_2_1_VERSION
            /* PSM2 renamed PSM_* env vars to equivalent PSM2_* vars */
            if ( getenv("PSM2_DEVICES") == NULL ) {
        #else
            if ( getenv("PSM_DEVICES") == NULL ) {
        #endif
            if (universesize > n && n > 1) {
                /* There are both local and remote ranks present;
                 * we require both the shm and ipath devices in
                 * this case. */
                #if PSM_VERNO >= PSM_2_1_VERSION
                    putenv("PSM2_DEVICES=self,shm,hfi");
                #else
                    putenv("PSM_DEVICES=self,shm,ipath");
                #endif
            }
            else if (universesize > n && n == 1) {
                /* There are only remote ranks; we do not require
                 * the shm device. */
                #if PSM_VERNO >= PSM_2_1_VERSION
                    putenv("PSM2_DEVICES=self,hfi");
                #else
                    putenv("PSM_DEVICES=self,ipath");
                #endif
            }
            else if (universesize == n && n > 1) {
                /* There are only local ranks; we do not require the
                 * ipath device. */
                #if PSM_VERNO >= PSM_2_1_VERSION
                    putenv("PSM2_DEVICES=self,shm");
                #else
                    putenv("PSM_DEVICES=self,shm");
                #endif
            }
            else if (universesize == 1 && n == 1) {
                /* This is the only rank; we do not need either the
                   shm or the ipath device. */
                #if PSM_VERNO >= PSM_2_1_VERSION
                    putenv("PSM2_DEVICES=self");
                #else
                    putenv("PSM_DEVICES=self");
                #endif
            }
            else {
                /* Impossible situation? Leave PSM_DEVICES as it
                 * previously was. */
            }
        }
    }

}

/* detect if arch and hca type is same for all processes */
static int psm_detect_heterogeneity(mv2_arch_hca_type myarch, int pg_size, int pg_rank)
{
    int i, mpi_errno = MPI_SUCCESS;
    mv2_arch_hca_type arch = 0;
    mv2_homogeneous_cluster = 1;
    char *flag;

    if ((flag = getenv("MV2_HOMOGENEOUS_CLUSTER")) != NULL) {
        mv2_homogeneous_cluster = !!atoi(flag);
        goto fn_exit;
    }

    if (pg_size == 1) {
        mv2_homogeneous_cluster = 1;
        goto fn_exit;
    }

    PRINT_DEBUG(DEBUG_CM_verbose>1, "my arch_hca_type = %016lx\n", myarch);
    MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "pmi_ahkey-%d", pg_rank);
    MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen, "%016lx", myarch);

    if(UPMI_KVS_PUT(kvsid, mv2_pmi_key, mv2_pmi_val) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ahtype_putfailed");
    }
    if(UPMI_KVS_COMMIT(kvsid) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ahtype_putcommit");
    }
    if(UPMI_BARRIER() != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**ahtype_putcommit");
    }

    for (i = 0; i < pg_size; i++) {
        if (i != pg_rank) {
            MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "pmi_ahkey-%d", i);
            mpi_errno = UPMI_KVS_GET(kvsid, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen);
            if(mpi_errno != UPMI_SUCCESS) {
                MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_getfailed");
            }

            sscanf(mv2_pmi_val, "%016lx", &arch);
            PRINT_DEBUG(DEBUG_CM_verbose>1,
                    "peer: %d, val: %s, arch: %016lx\n", i, mv2_pmi_val, arch);
            if (arch != myarch) {
                mv2_homogeneous_cluster = 0;
                break;
            }
        }
    }

fn_exit:
    PRINT_DEBUG(DEBUG_CM_verbose>1, "mv2_homogeneous_cluster = %d\n", mv2_homogeneous_cluster);
    return mpi_errno;

fn_fail:
    PRINT_ERROR("ahtype put/commit/get failed\n");
    goto fn_exit;
}

/* all ranks provide their epid via PMI put/get */
static int psm_start_epid_exchange(PSM_EPID_T myid, int pg_size, int pg_rank)
{
    int mpi_errno = MPI_SUCCESS;

    if (pg_size == 1) {
        goto fn_exit;
    }

    PRINT_DEBUG(DEBUG_CM_verbose>1, "[%d] my epid = %lu\n", pg_rank, myid);
    MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "pmi_epidkey-%d", pg_rank);
    MPL_snprintf(mv2_pmi_val, mv2_pmi_max_vallen, "%lu", myid);
    if(UPMI_KVS_PUT(kvsid, mv2_pmi_key, mv2_pmi_val) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_putfailed");
    }
    if(UPMI_KVS_COMMIT(kvsid) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_putcommit");
    }
    if (mv2_use_pmi_ibarrier) {
        mpi_errno = UPMI_IBARRIER();
    } else {
        mpi_errno = UPMI_BARRIER();
    }
    if(mpi_errno != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_putcommit");
    }

fn_exit:
    return mpi_errno;

fn_fail:
    PRINT_ERROR("epid put/commit/get failed\n");
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME psm_connect_peer
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int psm_connect_peer(int peer)
{
    int err, mpi_errno = MPI_SUCCESS;
    PSM_EPID_T epidlist[1];
    PSM_ERROR_T errs[1];

    /* Should it fail if connection is already established? */
    assert(!PSM_ADDR_RESOLVED(peer));
    if (mv2_use_pmi_ibarrier) {
        UPMI_WAIT();
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Connecting to peer %d\n", peer);

    if (unlikely(psmdev_cw.pg_rank == peer)) {
        epidlist[0] = psmdev_cw.epid;
    } else {
        MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "pmi_epidkey-%d", peer);
        if(UPMI_KVS_GET(kvsid, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen) != UPMI_SUCCESS) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_getfailed");
        }
        PRINT_DEBUG(DEBUG_CM_verbose>1, "peer: %d, got epid: %s\n", peer, mv2_pmi_val);
        sscanf(mv2_pmi_val, "%lu", &epidlist[0]);
    }

    if((err = PSM_EP_CONNECT(psmdev_cw.ep, 1, epidlist, NULL, errs,
                &psmdev_cw.epaddrs[peer], TIMEOUT * SEC_IN_NS)) != PSM_OK) {
        fprintf(stderr, "psm_ep_connect failed with error %s\n", PSM_ERROR_GET_STRING(err));
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_INTERN, "**psmconnectfailed");
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Connection established with peer %d\n", peer);

    return MPI_SUCCESS;

fn_fail:
    PRINT_ERROR("psm_connect_peer failed\n");
    return MPI_ERR_INTERN;
}

#undef FUNCNAME
#define FUNCNAME psm_connect_alltoall
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int psm_connect_alltoall(PSM_EPADDR_T *addrs, int pg_size, int pg_rank)
{
    int i;
    int err, mpi_errno = MPI_SUCCESS;
    PSM_EPID_T *epidlist = NULL;
    PSM_ERROR_T *errlist = NULL;

    if (pg_size == 1) {
        goto fn_exit;
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Establishing alltoall connectivity\n");
    epidlist = (PSM_EPID_T*) MPIU_Malloc (pg_size * sizeof(PSM_EPID_T));
    errlist = (PSM_ERROR_T*) MPIU_Malloc (pg_size * sizeof(PSM_ERROR_T));

    if (epidlist == NULL || errlist == NULL) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_NO_MEM, "**psmnomem");
    }

    if (mv2_use_pmi_ibarrier) {
        UPMI_WAIT();
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Looking up epids\n");
    for (i=0; i<pg_size; i++) {
        MPL_snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "pmi_epidkey-%d", i);
        if(UPMI_KVS_GET(kvsid, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen) != UPMI_SUCCESS) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**epid_getfailed");
        }
        PRINT_DEBUG(DEBUG_CM_verbose>1, "peer: %d, got epid: %s\n", i, mv2_pmi_val);
        sscanf(mv2_pmi_val, "%lu", &epidlist[i]);
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Connecting to peers\n");
    if((err = PSM_EP_CONNECT(psmdev_cw.ep, pg_size, epidlist, NULL, errlist,
                    addrs, TIMEOUT * SEC_IN_NS)) != PSM_OK) {
        fprintf(stderr, "psm_ep_connect failed with error %s\n", PSM_ERROR_GET_STRING(err));
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_INTERN, "**psmconnectfailed");
    }

    PRINT_DEBUG(DEBUG_CM_verbose>0, "Successfully established alltoall connectivity\n");
fn_exit:
    MPIU_Free(epidlist);
    MPIU_Free(errlist);
    return MPI_SUCCESS;

fn_fail:
    PRINT_ERROR("psm_connect_alltoall failed\n");
    return MPI_ERR_INTERN;
}

static int psm_create_uuid(void)
{
    int mpi_errno = MPI_SUCCESS;
    int i, len = sizeof(PSM_UUID_T);
    char *uuid_str = NULL;
    unsigned int kvs_hash = 0;
    
    uuid_str = MPIU_Malloc(sizeof(char) * (len+1));
    MPIU_Memset(uuid_str, 0, len+1);

    kvs_hash = psm_hash_str(kvsid);
    srand(kvs_hash);

    for (i=0; i<len; i++) {
        uuid_str[i] = psm_uuid[i] = rand() % UCHAR_MAX;
    }
    uuid_str[i] = '\0';

    PRINT_DEBUG(DEBUG_CM_verbose, "kvsid: %s, kvs_hash: %u\n", kvsid, kvs_hash);

    MPIU_Free(uuid_str);
    return mpi_errno;
}

/* broadcast the uuid to all ranks via PMI put/get */
static int psm_bcast_uuid(int pg_size, int pg_rank)
{
    int mpi_errno = MPI_SUCCESS;
    int srclen = sizeof(PSM_UUID_T);
    int dstlen = mv2_pmi_max_vallen;

    if(pg_rank == ROOT) {
        PSM_UUID_GENERATE(psm_uuid);
    }

    if(pg_size == 1)
        return MPI_SUCCESS;

    MPIU_Memset(mv2_pmi_key, 0, mv2_pmi_max_keylen);
    MPIU_Memset(mv2_pmi_val, 0, mv2_pmi_max_vallen);

    if(pg_rank == ROOT) {
        snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "%s", MPID_PSM_UUID);
        encode(srclen, (char *)&psm_uuid, dstlen, mv2_pmi_val);
        PRINT_DEBUG(DEBUG_CM_verbose>1, "uuid key: %s, value: %s\n", mv2_pmi_key, mv2_pmi_val);
    } else {
        snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "dummy-key");
        strcpy(mv2_pmi_val, "dummy-value");
    }

    if(UPMI_KVS_PUT(kvsid, mv2_pmi_key, mv2_pmi_val) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**pmiputuuid");
    }
    if(UPMI_KVS_COMMIT(kvsid) != UPMI_SUCCESS) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**pmicommituuid");
    }
    UPMI_BARRIER();

    if(pg_rank != ROOT) {
        snprintf(mv2_pmi_key, mv2_pmi_max_keylen, "%s", MPID_PSM_UUID);
        if(UPMI_KVS_GET(kvsid, mv2_pmi_key, mv2_pmi_val, mv2_pmi_max_vallen) != UPMI_SUCCESS) {
            MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**pmigetuuid");
        }

        PRINT_DEBUG(DEBUG_CM_verbose>1, "uuid key: %s, value: %s\n", mv2_pmi_key, mv2_pmi_val);
        strcat(mv2_pmi_val, "==");
        srclen = strlen(mv2_pmi_val);
        if(decode(srclen, mv2_pmi_val, sizeof(PSM_UUID_T), (char *)&psm_uuid)) {
            fprintf(stderr, "base-64 decode failed of UUID\n");
            goto fn_fail;
        }
    }

    return MPI_SUCCESS;

fn_fail:
    PRINT_ERROR("uuid bcast failed\n");
    return MPI_ERR_INTERN;
}

static void psm_read_user_params(void)
{
    char *flag;
    ipath_debug_enable = 0;
    if((flag = getenv("MV2_PSM_DEBUG")) != NULL) {
        ipath_debug_enable = !!atoi(flag);
    }
    ipath_dump_frequency = 10;
    if((flag = getenv("MV2_PSM_DUMP_FREQUENCY")) != NULL) {
        ipath_dump_frequency = atoi(flag);
    }
    ipath_enable_func_lock = 1;
    if((flag = getenv("MV2_PSM_ENABLE_FUNC_LOCK")) != NULL) {
        ipath_enable_func_lock = atoi(flag);
    }
    ipath_progress_yield_count = 3;
    if((flag = getenv("MV2_PSM_YIELD_COUNT")) != NULL) {
        ipath_progress_yield_count = atoi(flag);
    }
    if ((flag = getenv("MV2_PSM_BCAST_UUID")) != NULL) {
        mv2_psm_bcast_uuid = !!atoi(flag);
    }
    if ((flag = getenv("MV2_SHOW_ENV_INFO")) != NULL) {
        g_mv2_show_env_info = atoi(flag);
    }
    if ((flag = getenv("MV2_ON_DEMAND_THRESHOLD")) != NULL) {
        mv2_on_demand_threshold = atoi(flag);
    }
#if (defined(HAVE_PMI2_KVS_IFENCE) && defined(HAVE_PMI2_KVS_WAIT)) \
    || (defined(HAVE_PMI_IBARRIER) && defined(HAVE_PMI_WAIT))
    mv2_use_pmi_ibarrier = 1; /* enable by default if available */

    if ((flag = getenv("MV2_USE_PMI_IBARRIER")) != NULL) {
        mv2_use_pmi_ibarrier = !!atoi(flag);
    }
#endif
}

/* Ch3 expects channel to initialize VC fields.
   force_eager is used because psm internally manages eager/rndv so
   we can just force one code-path for all message sizes */

static void psm_other_init(MPIDI_PG_t *pg)
{
    MPIDI_VC_t *vc;
    int i;
    char *flag;
    uint32_t value = 0;

    for(i = 0; i < MPIDI_PG_Get_size(pg); i++) {
        MPIDI_PG_Get_vc(pg, i, &vc);

        vc->state = MPIDI_VC_STATE_ACTIVE;
        vc->force_eager = 1;
        vc->eager_max_msg_sz = PSM_VBUFSZ;
        vc->rndvSend_fn = NULL;
        vc->rndvRecv_fn = NULL;
    }

    if ((flag = getenv("MV2_IBA_EAGER_THRESHOLD")) != NULL) {
        mv2_hfi_rndv_thresh = user_val_to_bytes(flag, "MV2_IBA_EAGER_THRESHOLD");
    } else {
        /* Check if default PSM2 threshold is higher and if so, use it */
        PSM_MQ_GETOPT(psmdev_cw.mq, PSM_MQ_RNDV_IPATH_SZ, &value);
        if (value > mv2_hfi_rndv_thresh) {
            mv2_hfi_rndv_thresh = value;
        }
    }
    /* Set the value of HFI rendezvous threshold */
    PSM_MQ_SETOPT(psmdev_cw.mq, PSM_MQ_RNDV_IPATH_SZ, &mv2_hfi_rndv_thresh);
    /* Validate that the desired values were set */
    PSM_MQ_GETOPT(psmdev_cw.mq, PSM_MQ_RNDV_IPATH_SZ, &value);
    if (value != mv2_hfi_rndv_thresh) {
        PRINT_ERROR("Specified HFI rendezvous threshold was not set correctly by PSM.\n");
        PRINT_ERROR("Requested: %d, Set: %d\n", mv2_hfi_rndv_thresh, value);
    }

    if ((flag = getenv("MV2_SMP_EAGERSIZE")) != NULL) {
        mv2_shm_rndv_thresh = user_val_to_bytes(flag, "MV2_SMP_EAGERSIZE");
    } else {
        /* Check if default PSM2 threshold is higher and if so, use it */
        PSM_MQ_GETOPT(psmdev_cw.mq, PSM_MQ_RNDV_SHM_SZ, &value);
        if (value > mv2_shm_rndv_thresh) {
            mv2_shm_rndv_thresh = value;
        }
    }
    /* Set the value of SHM rendezvous threshold */
    PSM_MQ_SETOPT(psmdev_cw.mq, PSM_MQ_RNDV_SHM_SZ, &mv2_shm_rndv_thresh);
    /* Validate that the desired values were set */
    PSM_MQ_GETOPT(psmdev_cw.mq, PSM_MQ_RNDV_SHM_SZ, &value);
    if (value != mv2_shm_rndv_thresh) {
        PRINT_ERROR("Specified SHM rendezvous threshold was not set correctly by PSM\n");
        PRINT_ERROR("Requested: %d, Set: %d\n", mv2_shm_rndv_thresh, value);
    }

    /* Select the smaller threshold */
    if (mv2_shm_rndv_thresh < mv2_hfi_rndv_thresh) {
        ipath_rndv_thresh = mv2_shm_rndv_thresh;
    } else {
        ipath_rndv_thresh = mv2_hfi_rndv_thresh;
    }
    PRINT_DEBUG(DEBUG_CM_verbose>0,
            "hfi threshold: %d, shm threshold: %d, blocking threshold %d\n",
            mv2_hfi_rndv_thresh, mv2_shm_rndv_thresh, ipath_rndv_thresh);

    psm_queue_init();
    psm_init_vbuf_lock();
    psm_allocate_vbufs(PSM_INITIAL_POOL_SZ);
    psm_init_1sided();
}

static char base64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       "abcdefghijklmnopqrstuvwxyz"
                       "0123456789"
                       "+/";
/*
** ENCODE RAW into BASE64
*/

/* Encode source from raw data into Base64 encoded string */
static int encode(unsigned s_len, char *src, unsigned d_len, char *dst)
{
    unsigned triad;

    for (triad = 0; triad < s_len; triad += 3) {
        unsigned long int sr;
        unsigned byte;

        for (byte = 0; (byte<3)&&(triad+byte<s_len); ++byte) {
            sr <<= 8;
            sr |= (*(src+triad+byte) & 0xff);
        }

        sr <<= (6-((8*byte)%6))%6; /*shift left to next 6bit alignment*/

        if (d_len < 4) 
            return 1; /* error - dest too short */

        *(dst+0) = *(dst+1) = *(dst+2) = *(dst+3) = '=';
        switch(byte) {
        case 3:
            *(dst+3) = base64[sr&0x3f];
            sr >>= 6;
        case 2:
            *(dst+2) = base64[sr&0x3f];
            sr >>= 6;
        case 1:
            *(dst+1) = base64[sr&0x3f];
            sr >>= 6;
            *(dst+0) = base64[sr&0x3f];
        }
        dst += 4; d_len -= 4;
    }
    return 0;
}

/*
** DECODE BASE64 into RAW
*/

/* determine which sextet value a Base64 character represents */
static int tlu(int byte)
{
    int index;

    for (index = 0; index < 64; ++index)
        if (base64[index] == byte)
            break;
        if (index > 63) index = -1;
            return index;
}

/*
** Decode source from Base64 encoded string into raw data
**
** Returns: 0 - Success
** 1 - Error - Source underflow - need more base64 data
** 2 - Error - Chunk contains half a byte of data
** 3 - Error - Decoded results will overflow output buffer
*/
static int decode(unsigned s_len, char *src, unsigned d_len, char *dst)
{
    unsigned six, dix;

    dix = 0;

    for (six = 0; six < s_len; six += 4) {
        unsigned long sr;
        unsigned ix;

        sr = 0;
        for (ix = 0; ix < 4; ++ix) {
            int sextet;

            if (six+ix >= s_len)
                return 1;
            if ((sextet = tlu(*(src+six+ix))) < 0)
                break;
            sr <<= 6;
            sr |= (sextet & 0x3f);
        }

        switch (ix) {
        case 0: /* end of data, no padding */
            return 0;

        case 1: /* can't happen */
            return 2;

        case 2: /* 1 result byte */
            sr >>= 4;
            if (dix > d_len) 
                return 3;
            *(dst+dix) = (sr & 0xff);
            ++dix;
            break;

        case 3: /* 2 result bytes */
            sr >>= 2;
            if (dix+1 > d_len) 
                return 3;
            *(dst+dix+1) = (sr & 0xff);
            sr >>= 8;
            *(dst+dix) = (sr & 0xff);
            dix += 2;
            break;

        case 4: /* 3 result bytes */
            if (dix+2 > d_len) 
                return 3;
            *(dst+dix+2) = (sr & 0xff);
            sr >>= 8;
            *(dst+dix+1) = (sr & 0xff);
            sr >>= 8;
            *(dst+dix) = (sr & 0xff);
            dix += 3;
            break;
        }
    }
    return 0;
}

/* djb2 hash function */
static unsigned int psm_hash_str(char *str)
{
    unsigned int hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;

    return hash;
}

int mv2_allocate_pmi_keyval(void)
{
    if (!mv2_pmi_max_keylen) {
        UPMI_KVS_GET_KEY_LENGTH_MAX(&mv2_pmi_max_keylen);
    }
    if (!mv2_pmi_max_vallen) {
        UPMI_KVS_GET_VALUE_LENGTH_MAX(&mv2_pmi_max_vallen);
    }

    mv2_pmi_key = MPIU_Malloc(mv2_pmi_max_keylen+1);
    mv2_pmi_val = MPIU_Malloc(mv2_pmi_max_vallen+1);

    if (mv2_pmi_key==NULL || mv2_pmi_val==NULL) {
        mv2_free_pmi_keyval();
        return -1; 
    }
    return 0;
}

void mv2_free_pmi_keyval(void)
{
    if (mv2_pmi_key!=NULL) {
        MPIU_Free(mv2_pmi_key);
        mv2_pmi_key = NULL;
    }

    if (mv2_pmi_val!=NULL) {
        MPIU_Free(mv2_pmi_val);
        mv2_pmi_val = NULL;
    }
}

