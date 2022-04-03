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

#include "rdma_impl.h"
#include "ibv_param.h"
#include "ibv_cuda_util.h"
#if defined(_ENABLE_CUDA_)
#if defined(HAVE_CUDA_IPC)

#define CUDAIPC_SUCESS 0
#define CUDAIPC_OPENMEM_FAILED 1

#define CUDAIPC_CACHE_INVALID 0x01

cuda_regcache_entry_t **cudaipc_cache_list = NULL;
int *num_cudaipc_cache_entries = NULL;
int cudaipc_num_local_procs = 0;
int cudaipc_my_local_id = -1;
static char *cuda_shmem_file = NULL;
static char *cuda_shmem_ptr = NULL;
static int cuda_shmem_fd = -1;
static int cuda_shmem_size = 0;
cudaipc_shared_info_t *cudaipc_shared_data;
cudaipc_device_id_t *cudaipc_shared_device_id;
cudaipc_local_info_t *cudaipc_local_data;
cudaipc_remote_info_t *cudaipc_remote_data;
int cudaipc_num_stage_buffers = 2;
int cudaipc_stage_buffer_size = (1 << 19);
int cudaipc_stage_buffered = 1;
size_t cudaipc_stage_buffered_limit = (1 << 25);
int cudaipc_sync_limit = 16 * 1024;

static inline int cudaipc_openmemhandle(cuda_regcache_entry_t *reg)
{
    cudaError_t cudaerr = cudaSuccess;
    cudaIpcMemHandle_t memhandle;
    MPIU_Memcpy((void *)&memhandle, reg->cuda_memHandle, sizeof(cudaIpcMemHandle_t));
    cudaerr = cudaIpcOpenMemHandle(&reg->remote_base, memhandle, 
                                        cudaIpcMemLazyEnablePeerAccess);
    if (cudaerr != cudaSuccess) {
        return CUDAIPC_OPENMEM_FAILED;
    } 

    return CUDAIPC_SUCESS;
}
static inline int cudaipc_closememhandle(cuda_regcache_entry_t *reg)
{
    cudaError_t cudaerr = cudaSuccess;
    cudaerr = cudaIpcCloseMemHandle(reg->remote_base);
    if (cudaerr != cudaSuccess) {
        ibv_error_abort(IBV_RETURN_ERR,"cudaIpcCloseMemHandle failed\n");
    }
    return CUDAIPC_SUCESS;
}

static inline int is_cudaipc_cachable(cuda_regcache_entry_t *reg)
{
    return !(reg->flags & CUDAIPC_CACHE_INVALID);
}


static inline void cudaipc_regcache_find (void *addr, int size, 
                            cuda_regcache_entry_t **reg, int rank)
{
    cuda_regcache_entry_t *temp = NULL;
    temp = cudaipc_cache_list[rank];
    *reg = NULL;
    while(temp) {
        if(!(((uint64_t)temp->addr + temp->size <= (uint64_t)addr) 
                || ((uint64_t)temp->addr >= (uint64_t)addr + size))) {
            *reg = temp;
            break;
        }
        temp = temp->next; 
    }
}

static inline void cudaipc_regcache_insert(cuda_regcache_entry_t *new_reg)
{
    int rank = new_reg->rank;
    if (num_cudaipc_cache_entries[rank] >= cudaipc_cache_max_entries) {
        /*flush a entry from the cache */
        cudaipc_flush_regcache(rank, 1);
    }
        
    num_cudaipc_cache_entries[rank]++;
    
    new_reg->next = new_reg->prev = NULL;
    if (cudaipc_cache_list[rank] == NULL) {
        cudaipc_cache_list[rank] = new_reg;
    } else {
        new_reg->next = cudaipc_cache_list[rank];
        cudaipc_cache_list[rank]->prev = new_reg;
        cudaipc_cache_list[rank] = new_reg;
    }
}

void cudaipc_regcache_delete(cuda_regcache_entry_t *reg)
{
    if (reg == NULL) {
        return;
    }
    int rank =  reg->rank;
    if (cudaipc_cache_list[rank] == NULL) {
        return;
    }

    if (cudaipc_cache_list[rank] == reg) {
        cudaipc_cache_list[rank] = reg->next;
    }

    if (reg->next) {
        reg->next->prev = reg->prev;
    }

    if (reg->prev) {
        reg->prev->next = reg->next;
    }
    num_cudaipc_cache_entries[rank]--;
}

void cudaipc_register(void *base_ptr, size_t size, int rank, 
        cudaIpcMemHandle_t memhandle, cuda_regcache_entry_t **cuda_reg)
{
    cuda_regcache_entry_t *reg = NULL, *new_reg = NULL;

start_find:
    if (rdma_cuda_enable_ipc_cache) {
        cudaipc_regcache_find(base_ptr, size, &reg, rank);
    }

    if (reg != NULL) {
        PRINT_DEBUG(CUDAIPC_DEBUG, "cache hit base_addr:%p size:%ld rank:%d \n", 
                base_ptr, size, rank);
        /* validate if the cached memhandle is valid */
        if ((0 == memcmp(&memhandle, reg->cuda_memHandle,sizeof(cudaIpcMemHandle_t))) 
                    && size == reg->size && base_ptr == reg->addr) {
            /* valid cache */
            new_reg = reg;
        } else {
            /* remove cache entry */
            PRINT_DEBUG(CUDAIPC_DEBUG, "Mismatch. Evict the entry: "
                    "old base: %p (old size:%lu, old memhandle:%lu) rank:%d\n ",
                    base_ptr, reg->size, reg->cuda_memHandle[0], rank);
            reg->refcount++;
            reg->flags |= CUDAIPC_CACHE_INVALID;
            cudaipc_deregister(reg);
            reg = NULL;
            goto start_find;
        }
    }

    if (reg == NULL) {
        new_reg = MPIU_Malloc(sizeof (cuda_regcache_entry_t));
        new_reg->addr = base_ptr;
        new_reg->size = size;
        new_reg->refcount = 0;
        new_reg->rank = rank;
        MPIU_Memcpy(new_reg->cuda_memHandle, (const void *) &memhandle, sizeof(cudaIpcMemHandle_t));
        if (cudaipc_openmemhandle(new_reg) != CUDAIPC_SUCESS) {
            PRINT_DEBUG(CUDAIPC_DEBUG,"cudaIpcOpenMemHandle failed\n");
            /* try flush open it after flushing all the entries */
            cudaipc_flush_regcache(rank, num_cudaipc_cache_entries[rank]);
            if (cudaipc_openmemhandle(new_reg) != CUDAIPC_SUCESS) {
                ibv_error_abort(IBV_RETURN_ERR,"cudaIpcOpenMemHandle failed\n");
            }
        }
            
        if (rdma_cuda_enable_ipc_cache) {
            cudaipc_regcache_insert(new_reg);
            PRINT_DEBUG(CUDAIPC_DEBUG, "cahce miss. New entry added: base_ptr: "
                    "%p end:%p size:%lu, memhandle:%lu rank:%d\n", base_ptr, 
                     (char *)base_ptr + size, size, new_reg->cuda_memHandle[0], rank);
        }
    }
    new_reg->refcount++;
    *cuda_reg = new_reg;
}

void cudaipc_deregister(cuda_regcache_entry_t *reg)
{
    MPIU_Assert(reg->refcount > 0);
    reg->refcount--;
    if (reg->refcount > 0) {
        return;
    }
    if (!(rdma_cuda_enable_ipc_cache && is_cudaipc_cachable(reg))) {
        if (rdma_cuda_enable_ipc_cache) {
            cudaipc_regcache_delete(reg);
        }
        PRINT_DEBUG(CUDAIPC_DEBUG, "deleted entry: base_ptr:%p size:%lu, "
                "memhandle:%lu rank:%d\n", reg->addr, reg->size, 
                reg->cuda_memHandle[0], reg->rank);
        cudaipc_closememhandle(reg);
        MPIU_Free(reg);    
    }
}

void cudaipc_flush_regcache(int rank, int count)
{
    cuda_regcache_entry_t *reg = NULL;
    int n = 0;
    while(cudaipc_cache_list[rank] && n < count) {
        reg = cudaipc_cache_list[rank];
        PRINT_DEBUG(CUDAIPC_DEBUG, "Free entry: base_ptr:%p size:%lu, "
                "memhandle:%lu rank:%d\n", reg->addr, reg->size, 
                 reg->cuda_memHandle[0], rank);
        reg->refcount++;
        reg->flags |= CUDAIPC_CACHE_INVALID;
        cudaipc_deregister(reg);
        n++;
    }
}

void cudaipc_initialize_cache()
{
    int i;
    cudaipc_cache_list = (cuda_regcache_entry_t **) 
                MPIU_Malloc(cudaipc_num_local_procs * sizeof(cuda_regcache_entry_t *));
    num_cudaipc_cache_entries = (int *) MPIU_Malloc(cudaipc_num_local_procs * sizeof(int));
    for (i=0; i< cudaipc_num_local_procs; i++) {
        num_cudaipc_cache_entries[i] = 0;
        cudaipc_cache_list[i] = NULL;
    }
}

void cudaipc_shmem_cleanup()
{
    PRINT_DEBUG(CUDAIPC_DEBUG, "cuda ipc finalized shmemfile:%s size:%d\n",
                            cuda_shmem_file, cuda_shmem_size);
    if (cuda_shmem_ptr != NULL) {
        munmap(cuda_shmem_ptr, cuda_shmem_size); 
    }
    if (cuda_shmem_fd != -1) { 
        close(cuda_shmem_fd);
    } 
    if (cuda_shmem_file != NULL) {
        unlink(cuda_shmem_file);
        MPIU_Free(cuda_shmem_file);
        cuda_shmem_file = NULL;
    }
}

void cudaipc_allocate_shared_region (MPIDI_PG_t *pg, int num_processes, int my_rank)
{
    char *shmem_dir;
    char s_hostname[HOSTNAME_LEN];
    int i;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    struct stat file_status;
    MPIDI_VC_t *vc;
    
    if ((shmem_dir = getenv("MV2_SHMEM_DIR")) == NULL) {
        shmem_dir = "/dev/shm";
    }
    
    if (gethostname(s_hostname, sizeof(char) * HOSTNAME_LEN) < 0) {
        ibv_error_abort(GEN_EXIT_ERR,"gethostname filed\n");
    }

    cuda_shmem_file =
        (char *) MPIU_Malloc(sizeof(char) * (strlen(shmem_dir) + 
                      HOSTNAME_LEN + 26 + PID_CHAR_LEN));
    if(!cuda_shmem_file) {
        ibv_error_abort(GEN_EXIT_ERR,"mem alloacation filed\n");
    }

    if (cudaipc_stage_buffered) { 
        cuda_shmem_size = (cudaipc_num_local_procs*sizeof(cudaipc_device_id_t)) + 
                (cudaipc_num_local_procs * cudaipc_num_local_procs *  
                cudaipc_num_stage_buffers * sizeof(cudaipc_shared_info_t));
    } else {
        cuda_shmem_size = (cudaipc_num_local_procs*sizeof(cudaipc_device_id_t));
    }
    
    sprintf(cuda_shmem_file, "%s/cudaipc_shmem-%s-%s-%d.tmp",
        shmem_dir, pg->ch.kvs_name, s_hostname, getuid());
    cuda_shmem_fd = open(cuda_shmem_file, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
    if (cuda_shmem_fd < 0) {
        PRINT_ERROR("open failed for file:%s\n", cuda_shmem_file);
        goto cleanup_shmem_file;
    }
        
    MPIDI_PG_Get_vc(pg, my_rank, &vc);
    if (vc->smp.local_rank == 0) {
        if (ftruncate(cuda_shmem_fd, 0)) {
            PRINT_ERROR("ftruncate failed file:%s\n", cuda_shmem_file);
            goto cleanup_shmem_file;
        }

        if (ftruncate(cuda_shmem_fd, cuda_shmem_size)) {
            PRINT_ERROR("ftruncate failed file:%s\n", cuda_shmem_file);
            goto cleanup_shmem_file;
        }
    }

    do {
        if (fstat(cuda_shmem_fd, &file_status) != 0) {
            PRINT_ERROR("fstat failed. file:%s\n", cuda_shmem_file);
            goto cleanup_shmem_file;
        }
    } while (file_status.st_size != cuda_shmem_size);

    cuda_shmem_ptr = mmap(0, cuda_shmem_size,
        (PROT_READ | PROT_WRITE), (MAP_SHARED), cuda_shmem_fd, 0);
    if (cuda_shmem_ptr == (void *) -1) {
        PRINT_ERROR("mmap failed. file:%s\n", cuda_shmem_file);
        goto cleanup_shmem_file;
    }
    if (vc->smp.local_rank == 0) { 
        memset(cuda_shmem_ptr, 0, cuda_shmem_size);
    }

    cudaipc_shared_device_id = (cudaipc_device_id_t *)cuda_shmem_ptr;
    if (vc->smp.local_rank == 0) {
        for (i=0; i<cudaipc_num_local_procs; i++) { 
            cudaipc_shared_device_id[i] = -1;
        }
    }

    if (cudaipc_stage_buffered) {
        cudaipc_shared_data = (cudaipc_shared_info_t *) ((uintptr_t) cuda_shmem_ptr 
                    + sizeof(cudaipc_device_id_t) * cudaipc_num_local_procs);

        cudaipc_local_data = (cudaipc_local_info_t *) 
                    MPIU_Malloc(cudaipc_num_local_procs * cudaipc_num_stage_buffers * 
                            sizeof (cudaipc_local_info_t));

        cudaipc_remote_data = (cudaipc_remote_info_t *)
                    MPIU_Malloc(cudaipc_num_local_procs * cudaipc_num_stage_buffers
                                        * sizeof(cudaipc_remote_info_t));
    }

    MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);

    /* unlink the shmem file */
    if (cuda_shmem_file != NULL) {
        unlink(cuda_shmem_file);
    }

    return;

cleanup_shmem_file:
    cudaipc_shmem_cleanup();
    ibv_error_abort(GEN_EXIT_ERR, "cudaipc initialization failed\n");
}

void cudaipc_share_device_info ()
{
    CUDA_CHECK(cudaGetDevice((int*)&cudaipc_shared_device_id[cudaipc_my_local_id]));
}

void cudaipc_allocate_ipc_region (MPIDI_PG_t *pg, int num_processes, int my_rank)
{
    int i, j, local_index, shared_index;
    MPIDI_VC_t *vc;

    for (j=0; j < num_processes; j++) {
        if (j == my_rank) continue;
        MPIDI_PG_Get_vc(pg, j, &vc);

        if (vc->smp.local_rank != -1 && vc->smp.can_access_peer != CUDA_IPC_DISABLED) {
            local_index = CUDAIPC_BUF_LOCAL_IDX(vc->smp.local_rank);
            shared_index = 
            CUDAIPC_BUF_SHARED_IDX(cudaipc_my_local_id, vc->smp.local_rank);

            for (i = 0; i < cudaipc_num_stage_buffers; i++) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &cudaipc_local_data[local_index + i].ipcEvent, 
                    cudaEventDisableTiming | cudaEventInterprocess));
                CUDA_CHECK(cudaIpcGetEventHandle(
                    &cudaipc_shared_data[shared_index + i].ipcEventHandle,
                    cudaipc_local_data[local_index + i].ipcEvent));
                MPIU_Malloc_CUDA(cudaipc_local_data[local_index + i].buffer, 
                                    cudaipc_stage_buffer_size);
                CUDA_CHECK(cudaIpcGetMemHandle(
                    &cudaipc_shared_data[shared_index + i].ipcMemHanlde,
                    cudaipc_local_data[local_index + i].buffer));

                PRINT_DEBUG(CUDAIPC_DEBUG, "remote rank:%d shared_index:%d eventhandle:%lu memhandle:%lu\n",
                    vc->smp.local_rank, shared_index+i,
                    *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcEventHandle),
                    *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcMemHanlde));
            }
        }
    }

    CUDA_CHECK(cudaGetDevice((int*)&cudaipc_shared_device_id[cudaipc_my_local_id]));
}

void cudaipc_initialize(MPIDI_PG_t *pg, int num_processes, int my_rank)
{
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int i, j, local_index, shared_index;
    MPIDI_VC_t *vc;
  
    /*allocate shared memory region to exchange info*/ 
    cudaipc_allocate_shared_region (pg, num_processes, my_rank); 
   
    /*allocate local IPC staging buffers and events*/
    cudaipc_allocate_ipc_region (pg, num_processes, my_rank); 

    MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);

    /* Open remote ipc event and mem handles */
    for (j = 0; j < num_processes; j++) {
        if (j == my_rank) continue;

        MPIDI_PG_Get_vc(pg, j, &vc);
        if (vc->smp.local_rank != -1 && vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
            local_index = CUDAIPC_BUF_LOCAL_IDX(vc->smp.local_rank);
            shared_index = 
                CUDAIPC_BUF_SHARED_IDX(vc->smp.local_rank, cudaipc_my_local_id);

            MPIU_Assert (cudaipc_shared_device_id[vc->smp.local_rank] != -1);

            for (i = 0; i < cudaipc_num_stage_buffers; i++) {
                CUDA_CHECK(cudaIpcOpenEventHandle(
                    &cudaipc_remote_data[local_index + i].ipcEvent, 
                    cudaipc_shared_data[shared_index + i].ipcEventHandle));
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    (void **)&cudaipc_remote_data[ local_index + i].buffer,
                    cudaipc_shared_data[shared_index + i].ipcMemHanlde,
                    cudaIpcMemLazyEnablePeerAccess));
                PRINT_DEBUG(CUDAIPC_DEBUG, "open: remote rank:%d shared_index:%d "
                    "eventhandle:%lu memhandle:%lu\n", vc->smp.local_rank, shared_index + i,
                    *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcEventHandle),
                    *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcMemHanlde));
            }
        }
    }
    
    /*shared memory region is cleaned up during ipc finalize*/
}

void cudaipc_init_dynamic (MPIDI_VC_t *vc)
{
    int i, local_index, shared_index;
    int my_device_id, peer_device_id;
  
    /*check if peer device is setup for the IPC buffered channel*/
    if (cudaipc_shared_device_id[vc->smp.local_rank] == -1) { 
        return;
    }

    /*check if peer device is IPC accessible*/
    my_device_id = cudaipc_shared_device_id[cudaipc_my_local_id];
    peer_device_id = cudaipc_shared_device_id[vc->smp.local_rank];   
    if(rdma_enable_ipc_share_gpu){    
        if (my_device_id == peer_device_id) {
            vc->smp.can_access_peer = CUDA_IPC_ENABLED;
        } else {
            CUDA_CHECK(cudaDeviceCanAccessPeer((int*)&vc->smp.can_access_peer, 
                           my_device_id, peer_device_id));
            vc->smp.can_access_peer = (vc->smp.can_access_peer == 0) ? CUDA_IPC_DISABLED : CUDA_IPC_ENABLED;
        }
    }else{
        CUDA_CHECK(cudaDeviceCanAccessPeer((int*)&vc->smp.can_access_peer,
                           my_device_id, peer_device_id));
            vc->smp.can_access_peer = (vc->smp.can_access_peer == 0) ? CUDA_IPC_DISABLED : CUDA_IPC_ENABLED;
    }    
    
    if (vc->smp.can_access_peer == CUDA_IPC_ENABLED && cudaipc_stage_buffered) {
        /*compute the local and peer indices*/
        local_index = CUDAIPC_BUF_LOCAL_IDX(vc->smp.local_rank);
        shared_index = 
             CUDAIPC_BUF_SHARED_IDX(vc->smp.local_rank, cudaipc_my_local_id);

        /* Open remote ipc event and mem handles for the process*/
        for (i = 0; i < cudaipc_num_stage_buffers; i++) {
             CUDA_CHECK(cudaIpcOpenEventHandle(
                 &cudaipc_remote_data[local_index + i].ipcEvent, 
                 cudaipc_shared_data[shared_index + i].ipcEventHandle));
             CUDA_CHECK(cudaIpcOpenMemHandle(
                 (void **)&cudaipc_remote_data[ local_index + i].buffer,
                 cudaipc_shared_data[shared_index + i].ipcMemHanlde,
                 cudaIpcMemLazyEnablePeerAccess));
             PRINT_DEBUG(CUDAIPC_DEBUG, "open: remote rank:%d shared_index:%d "
                 "eventhandle:%lu memhandle:%lu\n", vc->smp.local_rank, shared_index + i,
                 *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcEventHandle),
                 *((unsigned long *)&cudaipc_shared_data[shared_index + i].ipcMemHanlde));
        }
    }
}


void cudaipc_finalize()
{
    MPIDI_PG_t *pg;
    MPIDI_VC_t *vc;
    int i, j, pg_size, my_rank, local_index;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
 
    if (!cudaipc_stage_buffered) {
        goto cleanup_shmem_file;
    }
  
    if (cudaipc_init) {   
        pg = MPIDI_Process.my_pg;
        my_rank = MPIDI_Process.my_pg_rank;
        pg_size = MPIDI_PG_Get_size(pg);
        for (j=0; j < pg_size; j++) {
            if (j == my_rank) continue;
            MPIDI_PG_Get_vc(pg, j, &vc);
 
            if (vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
                local_index = CUDAIPC_BUF_LOCAL_IDX(vc->smp.local_rank);
                for (i = 0; i < cudaipc_num_stage_buffers; i++) {
                    CUDA_CHECK(cudaEventDestroy(cudaipc_remote_data[local_index + i].ipcEvent));        
                    CUDA_CHECK(cudaIpcCloseMemHandle(cudaipc_remote_data[local_index + i].buffer));
                }
            }
        }
    }
   
    /*TODO: this synchronization can be made local to the node*/
    MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);

    if (cudaipc_init) { 
        for (j=0; j < pg_size; j++) {
            if (j == my_rank) continue;
            MPIDI_PG_Get_vc(pg, j, &vc);
 
            if (vc->smp.can_access_peer == CUDA_IPC_ENABLED) {
                local_index = CUDAIPC_BUF_LOCAL_IDX(vc->smp.local_rank);
                for (i = 0; i < cudaipc_num_stage_buffers; i++) {
                    CUDA_CHECK(cudaEventDestroy(cudaipc_local_data[local_index + i].ipcEvent));
                    CUDA_CHECK(cudaFree(cudaipc_local_data[local_index + i].buffer));
                }
            }
        }
    }

    MPIU_Free(cudaipc_remote_data);
    MPIU_Free(cudaipc_local_data);

cleanup_shmem_file:
    cudaipc_shmem_cleanup();
}
#endif
#endif 

