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
#include <math.h>

#include "rdma_impl.h"
#include "mpiimpl.h"
#include "dreg.h"
#include "ibv_param.h"
#include "infiniband/verbs.h"
#include "mpidrma.h"
#include "upmi.h"
#include "mpiutil.h"

#if defined(_SMP_LIMIC_)
#include <fcntl.h>
#include <sys/mman.h>
#include "mpimem.h"
#endif /*_SMP_LIMIC_*/

#undef FUNCNAME
#define FUNCNAME 1SC_PUT_datav
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)

//#define DEBUG
#undef DEBUG_PRINT
#ifdef DEBUG
#define DEBUG_PRINT(args...) \
do {                                                          \
    int rank;                                                 \
    UPMI_GET_RANK(&rank);                                      \
    fprintf(stderr, "[%d][%s:%d] ", rank, __FILE__, __LINE__);\
    fprintf(stderr, args);                                    \
} while (0)
#else
#define DEBUG_PRINT(args...)
#endif

MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_vbuf_available);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_allocated);
MPIR_T_PVAR_ULONG_COUNTER_DECL_EXTERN(MV2, mv2_ud_vbuf_freed);
MPIR_T_PVAR_ULONG_LEVEL_DECL_EXTERN(MV2, mv2_ud_vbuf_available);

#ifdef _ENABLE_XRC_
#define IS_XRC_SEND_IDLE_UNSET(vc_ptr) (USE_XRC && VC_XST_ISUNSET (vc_ptr, XF_SEND_IDLE))
#define CHECK_HYBRID_XRC_CONN(vc_ptr) \
do {                                                                    \
    /* This is hack to force XRC connection in hybrid mode*/            \
    if (USE_XRC && VC_XST_ISSET (vc_ptr, XF_SEND_IDLE)                  \
            && !(vc_ptr->mrail.state & MRAILI_RC_CONNECTED)) {          \
        VC_XST_CLR (vc_ptr, XF_SEND_IDLE);                              \
    }                                                                   \
}while(0)                                                                   
#else
#define IS_XRC_SEND_IDLE_UNSET(vc_ptr) (0)
#define CHECK_HYBRID_XRC_CONN(vc_ptr) 
#endif

#define ONESIDED_RDMA_POST(_v, _vc_ptr, _save_vc, _rail)                    \
do {                                                                        \
    if (unlikely(!(IS_RC_CONN_ESTABLISHED((_vc_ptr)))                       \
            || (IS_XRC_SEND_IDLE_UNSET((_vc_ptr)))                          \
            || !MPIDI_CH3I_CM_One_Sided_SendQ_empty((_vc_ptr)))) {          \
        /* VC is not ready to be used. Wait till it is ready and send */    \
        MPIDI_CH3I_CM_One_Sided_SendQ_enqueue((_vc_ptr), (_v));             \
        if (!((_vc_ptr)->mrail.state & MRAILI_RC_CONNECTED) &&              \
                  (_vc_ptr)->ch.state == MPIDI_CH3I_VC_STATE_IDLE) {        \
            (_vc_ptr)->ch.state = MPIDI_CH3I_VC_STATE_UNCONNECTED;          \
        }                                                                   \
        CHECK_HYBRID_XRC_CONN((_vc_ptr));                                   \
        if ((_vc_ptr)->ch.state == MPIDI_CH3I_VC_STATE_UNCONNECTED) {       \
            /* VC is not connected, initiate connection */                  \
            MPIDI_CH3I_CM_Connect((_vc_ptr));                               \
        }                                                                   \
    } else {                                                                \
        XRC_FILL_SRQN_FIX_CONN ((_v), (_vc_ptr), (_rail));                  \
        if (MRAILI_Flush_wqe((_vc_ptr),(_v),(_rail)) != -1) { /* message not enqueued */\
            -- (_vc_ptr)->mrail.rails[(_rail)].send_wqes_avail;             \
            IBV_POST_SR((_v), (_vc_ptr), (_rail), "Failed to post rma put");\
        }                                                                   \
        if((_save_vc)) { (_vc_ptr) = (_save_vc);}                           \
    }                                                                       \
}while(0);

#define SHM_DIR "/"
#define PID_CHAR_LEN 22
/*20 is to hold rma_shmid, rank, mv2 and other punctuations */
#define SHM_FILENAME_LEN (sizeof(SHM_DIR) + PID_CHAR_LEN + 20)

unsigned short rma_shmid = 100;

typedef enum {
   SINGLE=0,
   STRIPE,
   REPLICATE
} rail_select_t;

typedef struct {
  uintptr_t win_ptr;
  uint32_t win_rkeys[MAX_NUM_HCAS];
  uint32_t completion_counter_rkeys[MAX_NUM_HCAS];
  uint32_t post_flag_rkeys[MAX_NUM_HCAS];
  uint32_t fall_back;
} win_info;

typedef struct {
     char filename[SHM_FILENAME_LEN];
     size_t size;
     size_t displacement;
     int    shm_fallback;
} file_info;

typedef struct shm_buffer {
  char filename[SHM_FILENAME_LEN];
  void *ptr;
  size_t size;
  int owner;
  int fd;
  int ref_count;
  struct shm_buffer *next;
} shm_buffer;
shm_buffer *shm_buffer_llist = NULL;
shm_buffer *shm_buffer_rlist = NULL;

#if defined(_SMP_LIMIC_)
char* shmem_file;
extern struct smpi_var g_smpi;
extern int limic_fd;
#endif /* _SMP_LIMIC_ */

extern int number_of_op;
static int Post_Get_Put_Get_List(MPID_Win *, MPIDI_msg_sz_t , dreg_entry * ,
        MPIDI_VC_t *, vbuf **, void *local_buf[], void *remote_buf[], void *user_buf[], 
        MPIDI_msg_sz_t length, uint32_t lkeys[], uint32_t rkeys[], 
        rail_select_t rail_select, int target);

static int Post_Put_Put_Get_List(MPID_Win *, MPIDI_msg_sz_t,  dreg_entry *, 
        MPIDI_VC_t *, vbuf **, void *local_buf[], void *remote_buf[], MPIDI_msg_sz_t length,
        uint32_t lkeys[], uint32_t rkeys[], rail_select_t rail_select, int target);


static int iba_put(MPIDI_RMA_Op_t *, MPID_Win *, MPIDI_msg_sz_t);
static int iba_get(MPIDI_RMA_Op_t *, MPID_Win *, MPIDI_msg_sz_t);
static int iba_fetch_and_add(MPIDI_RMA_Op_t *, MPID_Win *, MPIDI_msg_sz_t);
static int iba_compare_and_swap(MPIDI_RMA_Op_t *, MPID_Win *, MPIDI_msg_sz_t);

int     iba_lock(MPID_Win *, MPIDI_RMA_Op_t *, int);
int     iba_unlock(MPID_Win *, MPIDI_RMA_Op_t *, int);
int MRAILI_Handle_one_sided_completions(vbuf * v);                            

#ifdef INFINIBAND_VERBS_EXP_H
uint64_t ntohll(const uint64_t value)
{
   enum { TYP_INIT, TYP_SMLE, TYP_BIGE };

   union
   {
      uint64_t ull;
      uint8_t  c[8];
   } x;

   /* Test if on Big Endian system. */
   static int typ = TYP_INIT;

   if (typ == TYP_INIT)
   {
      x.ull = 0x01;
      typ = (x.c[7] == 0x01) ? TYP_BIGE : TYP_SMLE;
   }

   /* System is Big Endian; return value as is. */
   if (typ == TYP_BIGE)
   {
      return value;
   }

   /* Else convert value to Big Endian */
   x.ull = value;

   int8_t c = 0;
   c = x.c[0]; x.c[0] = x.c[7]; x.c[7] = c;
   c = x.c[1]; x.c[1] = x.c[6]; x.c[6] = c;
   c = x.c[2]; x.c[2] = x.c[5]; x.c[5] = c;
   c = x.c[3]; x.c[3] = x.c[4]; x.c[4] = c;

   return x.ull;
}

uint64_t htonll(const uint64_t value)
{
   return ntohll(value);
}
#endif  /* INFINIBAND_VERBS_EXP_H */

#undef FUNCNAME
#define FUNCNAME mv2_allocate_shm_local
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
int mv2_allocate_shm_local(int size, void **rnt_buf)
{
    int mpi_errno = MPI_SUCCESS;
    char *shm_file = NULL;
    void *mem_ptr = NULL;
    struct stat file_status;
    int fd;
    shm_buffer *shm_buffer_ptr, *prev_ptr, *curr_ptr;

    shm_file = (char *) MPIU_Malloc(SHM_FILENAME_LEN);
    if(shm_file == NULL) {
        MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_exit,
                   "**fail", "**fail %s","malloc failed");
    }

    sprintf(shm_file, "%smv2-%d-%d-%d.tmp",
                   SHM_DIR, MPIDI_Process.my_pg_rank, getpid(), rma_shmid);
    rma_shmid++;

    fd = shm_open(shm_file, O_CREAT | O_RDWR | O_EXCL, S_IRWXU);
    if(fd == -1) {
        MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_exit, 
                   "**fail", "**fail %s", strerror(errno));
    }

    if (ftruncate(fd, size) == -1)
    {
        MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto close_file, 
                   "**fail", "**fail %s", "ftruncate failed");
    }

    /*verify file creation*/
    do
    {
        if (fstat(fd, &file_status) != 0)
        {
            MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto close_file,
                   "**fail", "**fail %s", "fstat failed");
        }
    } while (file_status.st_size != size);

    mem_ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_LOCKED,
                         fd, 0);
    if (mem_ptr == MAP_FAILED)
    {
        MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto close_file,
                  "**fail", "**fail %s", "mmap failed");
    }

    MPIU_Memset(mem_ptr, 0, size);

    *rnt_buf =  mem_ptr;

    /*adding buffer to the list*/
    shm_buffer_ptr = (shm_buffer *) MPIU_Malloc(sizeof(shm_buffer));
    MPIU_Memcpy(shm_buffer_ptr->filename, shm_file, SHM_FILENAME_LEN);
    shm_buffer_ptr->ptr = mem_ptr;
    shm_buffer_ptr->owner = MPIDI_Process.my_pg_rank;
    shm_buffer_ptr->size = size;
    shm_buffer_ptr->fd = fd;
    shm_buffer_ptr->next = NULL;

    if(NULL == shm_buffer_llist) {
        shm_buffer_llist = shm_buffer_ptr;
        shm_buffer_ptr->next = NULL;
    } else {
        curr_ptr = shm_buffer_llist;
        prev_ptr = shm_buffer_llist;
        while(NULL != curr_ptr) {
          if ((size_t) curr_ptr->ptr > (size_t) shm_buffer_ptr->ptr) {
             break;
          }
          prev_ptr = curr_ptr;
          curr_ptr = curr_ptr->next;
        }
        if (prev_ptr == curr_ptr) { 
            shm_buffer_ptr->next = curr_ptr;
            shm_buffer_llist = shm_buffer_ptr;
        } else { 
            shm_buffer_ptr->next = prev_ptr->next;
            prev_ptr->next = shm_buffer_ptr;
        }
    }

fn_exit:
    if (shm_file) { 
        MPIU_Free(shm_file);
    }
    return mpi_errno;
close_file: 
    close(fd);
    shm_unlink(shm_file);
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME mv2_deallocate_shm_local
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
int mv2_deallocate_shm_local (void *ptr)
{
    int mpi_errno = MPI_SUCCESS;
    shm_buffer *curr_ptr, *prev_ptr;

    curr_ptr = shm_buffer_llist;
    prev_ptr = NULL;
    while (curr_ptr) {
       if (curr_ptr->ptr == ptr) {
           break;
       }
       prev_ptr = curr_ptr;
       curr_ptr = curr_ptr->next;
    }

    /*return if buffer not found in shm buffer list*/
    if (curr_ptr == NULL) {
        MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_exit,
                   "**fail", "**fail %s", "buffer not found in shm list");
    }

    /*delink the current pointer from the list*/
    if (prev_ptr != NULL) {
       prev_ptr->next = curr_ptr->next;
    } else {
       shm_buffer_llist = curr_ptr->next;
    }

    if (munmap(curr_ptr->ptr, curr_ptr->size)) {
        ibv_error_abort (GEN_EXIT_ERR, 
                 "rdma_iba_1sc: munmap failed in mv2_deallocate_shm_local");
    }
    close(curr_ptr->fd);
    shm_unlink(curr_ptr->filename);
    MPIU_Free(curr_ptr);

fn_exit:
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME mv2_find_and_deallocate_shm_local
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
int mv2_find_and_deallocate_shm (shm_buffer **list)
{
    int mpi_errno = MPI_SUCCESS;
    shm_buffer *curr_ptr, *prev_ptr;

    curr_ptr = *list;
    prev_ptr = NULL;
    while (curr_ptr) {
       if (curr_ptr->ref_count == 0) {
          /*delink pointer from the list*/
          if (prev_ptr != NULL) {
             prev_ptr->next = curr_ptr->next;
          } else {
             *list = curr_ptr->next;
          }

          if (munmap(curr_ptr->ptr, curr_ptr->size)) {
                ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc: \
                      mv2_find_and_deallocate_shm_local");
          }
          close(curr_ptr->fd);

          MPIU_Free(curr_ptr);
          if (prev_ptr != NULL) {
             curr_ptr = prev_ptr->next;
          } else {
             curr_ptr = *list;
          }
       } else {
          prev_ptr = curr_ptr;
          curr_ptr = curr_ptr->next;
       }
    }

    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME mv2_rma_allocate_shm
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
int mv2_rma_allocate_shm(int size, int g_rank, int *shmem_fd, 
                   void **rnt_buf, MPID_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno1 = MPI_SUCCESS;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    void* rma_shared_memory = NULL;
    const char *rma_shmem_dir="/";
    struct stat file_status;
    int length;
    char *rma_shmem_file = NULL;

    length = strlen(rma_shmem_dir)
             + 7 /*this is to hold MV2 and the unsigned short rma_shmid*/
             + PID_CHAR_LEN 
             + 6 /*this is for the hyphens and extension */;

    rma_shmem_file = (char *) MPIU_Malloc(length);
	
	if(g_rank == 0)
    {
       sprintf(rma_shmem_file, "%smv2-%d-%d.tmp",
                       rma_shmem_dir, getpid(), rma_shmid);
       rma_shmid++;
    }

    MPIR_Bcast_impl(rma_shmem_file, length, MPI_CHAR, 0, comm_ptr, &errflag); 

    *shmem_fd = shm_open(rma_shmem_file, O_CREAT | O_RDWR, S_IRWXU);
    if(*shmem_fd == -1){
        mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**nomem", 
                                 "**nomem %s", strerror(errno));
		goto fn_exit;
    }

    if (ftruncate(*shmem_fd, size) == -1)
    {
		   mpi_errno =
              MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**nomem", 0);
		   goto fn_exit;
    } 

    /*verify file creation*/
    do 
    {
        if (fstat(*shmem_fd, &file_status) != 0)
        {
            mpi_errno =
               MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME,
                                   __LINE__, MPI_ERR_OTHER, "**nomem", 0);
            goto fn_exit;
        }
    } while (file_status.st_size != size);

    rma_shared_memory = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         *shmem_fd, 0);
    if (rma_shared_memory == MAP_FAILED)
    {
         mpi_errno =
            MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME,
                                 __LINE__, MPI_ERR_OTHER, "**nomem", 0);
         goto fn_exit;
    }

    MPIU_Memset(rma_shared_memory, 0, size);
    *rnt_buf =  rma_shared_memory;

fn_exit:
    mpi_errno1 = MPIR_Barrier_impl(comm_ptr, &errflag);
    if (mpi_errno1 != MPI_SUCCESS) {
        ibv_error_abort (GEN_EXIT_ERR,
                    "rdma_iba_1sc: error calling barrier");
    }
    if (*shmem_fd != -1) { 
        shm_unlink(rma_shmem_file);
    }
    if (rma_shmem_file) {  
        MPIU_Free(rma_shmem_file);
    }
    return mpi_errno;
}

#undef FUNCNAME
#define FUNCNAME mv2_rma_deallocate_shm
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
void mv2_rma_deallocate_shm(void *addr, int size)
{
    if(munmap(addr, size))
    {
        DEBUG_PRINT("munmap failed in mv2_rma_deallocate_shm with error: %d \n", errno);
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc: mv2_rma_deallocate_shm");
    }
}

#ifdef MPIDI_CH3I_HAS_ALLOC_MEM
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Alloc_mem
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
void *MPIDI_CH3I_Alloc_mem (size_t size, MPID_Info *info)
{
   char value[10] = "";
   int flag = 0;
   void *ptr = NULL;

   if (info != NULL) { 
       MPIR_Info_get_impl(info, "alloc_shm", 10, value, &flag);
   }

   ptr = MPIU_Malloc(size);

   return ptr;
}
#endif

#ifdef MPIDI_CH3I_HAS_FREE_MEM
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_Free_mem
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
void MPIDI_CH3I_Free_mem (void *ptr)
{
   int mpi_errno = MPI_SUCCESS;

   if(SMP_INIT) {
      mpi_errno = mv2_deallocate_shm_local(ptr);
      if(mpi_errno != MPI_SUCCESS) {
           DEBUG_PRINT("this buffer was not allocated in shared memory, \
                        calling MPIU_Free \n");
           MPIU_Free(ptr);
      }

      mv2_find_and_deallocate_shm(&shm_buffer_rlist);
   } else {
      MPIU_Free(ptr);
   }
}
#endif

/* For active synchronization, it is a blocking call*/
void
MPIDI_CH3I_RDMA_start (MPID_Win* win_ptr, int start_grp_size, int* ranks_in_win_grp) 
{
    MPIDI_VC_t* vc = NULL;
    MPID_Comm* comm_ptr = NULL;
    int flag = 0;
    int src;
    int i;
    int counter = 0;

    if (SMP_INIT)
    {
        comm_ptr = win_ptr->comm_ptr;
    }

    while (flag == 0 && start_grp_size != 0)
    {
        /* Need to make sure we make some progress on
         * anything in the extended sendq or coalesced
         * or we can have a deadlock.
         */
        if (counter % 200 == 0)
        {
            MPIDI_CH3I_Progress_test();
        }

        ++counter;

        for (i = 0; i < start_grp_size; ++i)
        {
            flag = 1;
            src = ranks_in_win_grp[i];  /*src is the rank in comm*/

            if (SMP_INIT)
            {
                MPIDI_Comm_get_vc(comm_ptr, src, &vc);

                if (win_ptr->post_flag[src] == 0 && vc->smp.local_nodes == -1)
                {
                    /* Correspoding post has not been issued. */
                    flag = 0;
                    break;
                }
            }
            else if (win_ptr->post_flag[src] == 0)
            {
                /* Correspoding post has not been issued. */
                flag = 0;
                break;
            }
        }
    }
}

/* Waiting for all the completion signals and unregister buffers*/
int MPIDI_CH3I_RDMA_finish_rma(MPID_Win * win_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    if (win_ptr->put_get_list_size != 0 || 
        (win_ptr->enable_fast_path == 1 && win_ptr->rma_issued !=0)) {
            win_ptr->poll_flag = 1;
            while(win_ptr->poll_flag == 1){
                mpi_errno = MPIDI_CH3I_Progress_test();
                if(mpi_errno) MPIR_ERR_POP(mpi_errno);
	    }	
    }
    else return 0;

fn_fail:
    return mpi_errno;
}

int MPIDI_CH3I_RDMA_finish_rma_target(MPID_Win * win_ptr, int target_rank)
{
    int mpi_errno = MPI_SUCCESS;
    while (win_ptr->put_get_list_size_per_process[target_rank] != 0) {
        mpi_errno = MPIDI_CH3I_Progress_test();
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

fn_fail:
    return mpi_errno;
}

/*Directly post RDMA operations */
inline int MPIDI_CH3I_RDMA_try_rma_op_fast( int type, void *origin_addr, int origin_count,
        MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        int target_count, MPI_Datatype target_datatype, void *compare_addr,
        void *result_addr, MPID_Win *win_ptr)
{
    MPIDI_msg_sz_t complete = 0;
    MPIDI_VC_t* vc = NULL;
    MPID_Comm *comm_ptr;

    comm_ptr = win_ptr->comm_ptr;

    MPIDI_Comm_get_vc(comm_ptr, target_rank, &vc);

    if (SMP_INIT && vc->smp.local_nodes != -1
        && !mv2_MPIDI_CH3I_RDMA_Process.force_ib_atomic) {
        goto fn_exit;
    }

    MPIDI_msg_sz_t size, target_type_size, origin_type_size;
    char *local_addr = NULL, *remote_addr = NULL;
    uint32_t r_key, l_key;
    vbuf *v;
    int rail = 0;
    dreg_entry *tmp_dreg = NULL;
    switch (type)
    {
        case MPIDI_CH3_PKT_PUT:
            {
                MPID_Datatype_get_size_macro(origin_datatype, origin_type_size);
                size = origin_count * origin_type_size;
                ++win_ptr->rma_issued;

                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                tmp_dreg = dreg_register(origin_addr, size);
                l_key = tmp_dreg->memhandle[0]->lkey;
                local_addr = (char *)origin_addr;

                r_key = win_ptr->win_rkeys[target_rank *
                    rdma_num_hcas + rail];

                remote_addr = (char *) win_ptr->basic_info_table[target_rank].base_addr +
                    win_ptr->basic_info_table[target_rank].disp_unit * target_disp;

                v->vc = (void *) vc;

                ++(win_ptr->put_get_list_size_per_process[target_rank]);

                v->tmp_dreg = tmp_dreg;
                v->target_rank = target_rank;
                v->list = (void *) win_ptr;

                if(size<=rdma_max_inline_size){
                    v->desc.u.sr.send_flags |= IBV_SEND_INLINE;
                }

                vbuf_init_rma_put(v, local_addr, l_key, remote_addr, r_key, size, rail);
                ONESIDED_RDMA_POST(v, vc, NULL, rail);

                complete = 1;
                break;
            }
        case MPIDI_CH3_PKT_GET:
            {
                MPID_Datatype_get_size_macro(target_datatype, target_type_size);
                size = target_count * target_type_size;

                ++win_ptr->rma_issued;

                remote_addr = (char *) win_ptr->basic_info_table[target_rank].base_addr +
                    win_ptr->basic_info_table[target_rank].disp_unit * target_disp;

                GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                tmp_dreg = dreg_register(origin_addr, size);
                l_key = tmp_dreg->memhandle[0]->lkey;
                r_key = win_ptr->win_rkeys[target_rank *
                    rdma_num_hcas + rail];
                local_addr = (char *)origin_addr;

                v->vc = (void *) vc;

                ++(win_ptr->put_get_list_size_per_process[target_rank]);

                v->tmp_dreg = tmp_dreg;
                v->target_rank = target_rank;
                v->list = (void *) win_ptr;

                if(size<=rdma_max_inline_size){
                    v->desc.u.sr.send_flags |= IBV_SEND_INLINE;
                }

                vbuf_init_rma_get(v, local_addr, l_key, remote_addr, r_key, size, rail);
                ONESIDED_RDMA_POST(v, vc, NULL, rail);

                complete = 1;
                break;
            }
       case MPIDI_CH3_PKT_FOP:
            {
                MPID_Datatype_get_size_macro(origin_datatype, origin_type_size);
                uint64_t *fetch_addr;
                uint64_t add_value;
                int aligned;

                size = origin_type_size;
                add_value = *((uint64_t *) origin_addr);

                remote_addr = (char *) win_ptr->basic_info_table[target_rank].base_addr +
                    win_ptr->basic_info_table[target_rank].disp_unit * target_disp;

                aligned = !((intptr_t)remote_addr % 8);

                if (g_atomics_support && aligned && origin_type_size == 8) {
                    ++win_ptr->rma_issued;
                    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                    l_key = v->region->mem_handle[0]->lkey;
                    r_key = win_ptr->win_rkeys[target_rank *
                        rdma_num_hcas + rail];

                    v->vc = (void *) vc;

                    ++(win_ptr->put_get_list_size_per_process[target_rank]);

                    v->result_addr = result_addr;
                    v->target_rank = target_rank;
                    v->list = (void *) win_ptr;

                    if(size<=rdma_max_inline_size){
                        v->desc.u.sr.send_flags |= IBV_SEND_INLINE;
                    }
                    fetch_addr = (uint64_t *)v->buffer;
                    *fetch_addr = 0;
                    vbuf_init_rma_fetch_and_add(v, (void *) fetch_addr, l_key, remote_addr, r_key, add_value, rail);
                    ONESIDED_RDMA_POST(v, vc, NULL, rail);

                    complete = 1;
                }
                else 
                    complete = 0;
                break;
            }
       case MPIDI_CH3_PKT_CAS_IMMED:
            {
                MPID_Datatype_get_size_macro(origin_datatype, origin_type_size);
                char *return_addr;
                uint64_t compare_value, swap_value;
                int aligned;
                
                size = origin_type_size;
                swap_value = *((uint64_t *) origin_addr);
                compare_value = *((uint64_t *) compare_addr);

                remote_addr = (char *) win_ptr->basic_info_table[target_rank].base_addr +
                    win_ptr->basic_info_table[target_rank].disp_unit * target_disp;

                aligned = !((intptr_t)remote_addr % 8);

                if (g_atomics_support && aligned && origin_type_size == 8) {
                    ++win_ptr->rma_issued;
                    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
                    l_key = v->region->mem_handle[0]->lkey;
                    r_key = win_ptr->win_rkeys[target_rank *
                        rdma_num_hcas + rail];

                    v->vc = (void *) vc;

                    ++(win_ptr->put_get_list_size_per_process[target_rank]);

                    v->result_addr = result_addr;
                    v->target_rank = target_rank;
                    v->list = (void *) win_ptr;

                    if(size<=rdma_max_inline_size){
                        v->desc.u.sr.send_flags |= IBV_SEND_INLINE;
                    }
                    return_addr = (char *)v->buffer;
                    *((uint64_t *)return_addr) = 0;
                    vbuf_init_rma_compare_and_swap(v, return_addr, l_key, remote_addr, r_key, compare_value, swap_value, rail);
                    ONESIDED_RDMA_POST(v, vc, NULL, rail);

                    complete = 1;
                }
                else 
                    complete = 0;
                break;

            }
       default:
            DEBUG_PRINT("Unknown ONE SIDED OP\n");
            ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
            break;
    }

fn_exit:
    return complete;
}


void mv2_init_rank_for_barrier (MPID_Win ** win_ptr) 
{
    int             i, comm_size;
    MPIDI_VC_t*     vc=NULL;
    MPID_Comm       *comm_ptr = NULL;

    MPIU_Assert(win_ptr != NULL);
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr );
    comm_size = comm_ptr->local_size;

    (*win_ptr)->shm_l2g_rank = (int *)
                  MPIU_Malloc(g_smpi.num_local_nodes * sizeof(int));
    if((*win_ptr)->shm_l2g_rank == NULL) {
        ibv_error_abort (GEN_EXIT_ERR, 
               "rdma_iba_1sc: error allocating shm_l2g_rank");
    }

    for(i=0; i<comm_size; i++) {
        MPIDI_Comm_get_vc(comm_ptr, i, &vc);
        if(vc->smp.local_nodes != -1) {
            (*win_ptr)->shm_l2g_rank[vc->smp.local_nodes] = vc->pg_rank;
        }
    }
}

int MPIDI_CH3I_barrier_in_rma(MPID_Win **win_ptr, int rank, int node_size, int comm_size) 
{
    int lsrc, ldst, src, dst, mask, mpi_errno=MPI_SUCCESS;
    MPID_Comm *comm_ptr;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    int l_rank = g_smpi.my_local_id;

    MPIU_Assert(win_ptr != NULL);
    /* Trivial barriers return immediately */
    if (node_size == 1) goto fn_exit;

    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);

    mask = 0x1;
    while (mask < node_size) {
        ldst = (l_rank + mask) % node_size;
        lsrc = (l_rank - mask + node_size) % node_size;

        src = (*win_ptr)->shm_l2g_rank[lsrc];
        dst = (*win_ptr)->shm_l2g_rank[ldst];

        mpi_errno = MPIC_Sendrecv(NULL, 0, MPI_BYTE, dst,
                MPIR_BARRIER_TAG, NULL, 0, MPI_BYTE,
                src, MPIR_BARRIER_TAG, comm_ptr,
                MPI_STATUS_IGNORE, &errflag);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        mask <<= 1;
    }

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;

}

/* Go through RMA op list once, and start as many RMA ops as possible */
void
MPIDI_CH3I_RDMA_try_rma(MPID_Win * win_ptr, MPIDI_RMA_Target_t * target)
{
    MPIDI_RMA_Op_t *curr_ptr = NULL;
    MPIDI_RMA_Op_t *next_ptr = NULL;
    int mpi_errno = MPI_SUCCESS;
    int has_iwarp = 0;
    intptr_t aligned;
    MPIDI_VC_t* vc = NULL;
    MPID_Comm* comm_ptr = NULL;
#if defined(RDMA_CM)
    has_iwarp = mv2_MPIDI_CH3I_RDMA_Process.use_iwarp_mode;
#endif
    if (SMP_INIT)
    {
        comm_ptr = win_ptr->comm_ptr;
    }

    if (win_ptr->num_targets_with_pending_net_ops == 0 || target == NULL ||
            target->pending_net_ops_list_head == NULL)
        return;

    curr_ptr = target->next_op_to_issue;

    while (curr_ptr != NULL) {

        if (SMP_INIT) {
            MPIDI_Comm_get_vc(comm_ptr, curr_ptr->target_rank, &vc);

            if (vc->smp.local_nodes != -1) {
                curr_ptr = curr_ptr->next;
                target->issue_2s_sync = 1;
                continue;
            }
        }

        MPIDI_msg_sz_t size, origin_type_size, target_type_size;
        switch (curr_ptr->pkt.type)
        {
            case MPIDI_CH3_PKT_PUT_IMMED:
            case MPIDI_CH3_PKT_PUT:
                {
                    int origin_dt_derived;
                    int target_dt_derived;
                    MPI_Datatype target_datatype;

                    MPIDI_CH3_PKT_RMA_GET_TARGET_DATATYPE(curr_ptr->pkt, target_datatype, mpi_errno);
                    target_dt_derived = HANDLE_GET_KIND(target_datatype) != HANDLE_KIND_BUILTIN ? 1 : 0; 
                    origin_dt_derived = HANDLE_GET_KIND(curr_ptr->origin_datatype) != HANDLE_KIND_BUILTIN ? 1 : 0; 
                    MPID_Datatype_get_size_macro(target_datatype, target_type_size);  

                    MPID_Datatype_get_size_macro(curr_ptr->origin_datatype, origin_type_size);
                    size = curr_ptr->origin_count * origin_type_size;

                    if (!origin_dt_derived
                        && !target_dt_derived
                        && size > rdma_put_fallback_threshold
                        && curr_ptr->ureq == NULL)
                    {
                        next_ptr = curr_ptr->next;
                        iba_put(curr_ptr, win_ptr, size);
                        MPIDI_CH3I_RMA_Ops_free_elem(win_ptr, &(target->pending_net_ops_list_head), curr_ptr);
                        curr_ptr = next_ptr;
                        if (!target->issue_2s_sync) {
                            target->next_op_to_issue = next_ptr;
                        }
                    }                 
                    else
                    {
                        curr_ptr = curr_ptr->next;
                        target->issue_2s_sync = 1;
                    }

                    break;
                }
            case MPIDI_CH3_PKT_ACCUMULATE:
            case MPIDI_CH3_PKT_ACCUMULATE_IMMED:
            case MPIDI_CH3_PKT_GET_ACCUM:
            case MPIDI_CH3_PKT_GET_ACCUM_IMMED:
                {
                    curr_ptr = curr_ptr->next;
                    target->issue_2s_sync = 1;
                    break;
                }
            case MPIDI_CH3_PKT_FOP:
            case MPIDI_CH3_PKT_FOP_IMMED:
                {
                    aligned = !(((intptr_t)(curr_ptr->pkt.fop.addr)) % 8);
                    MPID_Datatype_get_size_macro(curr_ptr->origin_datatype, origin_type_size);
                    /* IB supports fetch_and_add operation for 8 bytes message
                     * size, so check the data size here 
                     * IB atomic operations require aligned address*/
                    if (g_atomics_support && origin_type_size == 8 && curr_ptr->origin_datatype != MPI_DOUBLE
                        && aligned && !has_iwarp && curr_ptr->pkt.fop.op == MPI_SUM && curr_ptr->ureq == NULL)
                    {
                        next_ptr = curr_ptr->next;
                        iba_fetch_and_add(curr_ptr, win_ptr, origin_type_size);
                        MPIDI_CH3I_RMA_Ops_free_elem(win_ptr, &(target->pending_net_ops_list_head), curr_ptr);
                        curr_ptr = next_ptr;
                        if (!target->issue_2s_sync) {
                            target->next_op_to_issue = next_ptr;
                        }
                    }
                    else 
                    {
                        curr_ptr = curr_ptr->next;
                        target->issue_2s_sync = 1;
                    }
                    break;
                }
            case MPIDI_CH3_PKT_CAS_IMMED:
                {
                    aligned = !(((intptr_t)(curr_ptr->pkt.cas.addr)) % 8);
                    MPID_Datatype_get_size_macro(curr_ptr->origin_datatype, origin_type_size);
                    /* IB supports compare_and_swap operation for 8 bytes
                     * message size, so check the data size here 
                     * IB atomic operations require aligned address*/
                    if (g_atomics_support && origin_type_size == 8 && aligned && !has_iwarp && curr_ptr->ureq == NULL)
                    {
                        next_ptr = curr_ptr->next;
                        iba_compare_and_swap(curr_ptr, win_ptr, origin_type_size);
                        MPIDI_CH3I_RMA_Ops_free_elem(win_ptr, &(target->pending_net_ops_list_head), curr_ptr);
                        curr_ptr = next_ptr;
                        if (!target->issue_2s_sync) {
                            target->next_op_to_issue = next_ptr;
                        }
                    } 
                    else
                    {
                        curr_ptr = curr_ptr->next;
                        target->issue_2s_sync = 1;
                    }
                    break;

                }
            case MPIDI_CH3_PKT_GET:
                {
                    int origin_dt_derived;
                    int target_dt_derived;
                    MPI_Datatype target_datatype;

                    MPIDI_CH3_PKT_RMA_GET_TARGET_DATATYPE(curr_ptr->pkt, target_datatype, mpi_errno);
                    target_dt_derived = HANDLE_GET_KIND(target_datatype) != HANDLE_KIND_BUILTIN ? 1 : 0;
                    origin_dt_derived = HANDLE_GET_KIND(curr_ptr->origin_datatype) != HANDLE_KIND_BUILTIN ? 1 : 0;
                    MPID_Datatype_get_size_macro(target_datatype, target_type_size); 

                    int target_count;
                    MPIDI_CH3_PKT_RMA_GET_TARGET_COUNT(curr_ptr->pkt, target_count, mpi_errno);
                    size = target_count * target_type_size;

                    if (!origin_dt_derived
                        && !target_dt_derived 
                        && size > rdma_get_fallback_threshold
                        && curr_ptr->ureq == NULL)
                    {
                        next_ptr = curr_ptr->next;
                        iba_get(curr_ptr, win_ptr, size);
                        MPIDI_CH3I_RMA_Ops_free_elem(win_ptr, &(target->pending_net_ops_list_head), curr_ptr);
                        curr_ptr = next_ptr;
                        if (!target->issue_2s_sync) {
                            target->next_op_to_issue = next_ptr;
                        }
                    }
                    else
                    {
                        curr_ptr = curr_ptr->next;
                        target->issue_2s_sync = 1;
                    }

                    break;
                }
            default:
                DEBUG_PRINT("Unknown ONE SIDED OP\n");
                ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
                break;
        }
    }

    if (target->next_op_to_issue == NULL && target->issue_2s_sync == 0) {
        if (target->sync.sync_flag == MPIDI_RMA_SYNC_UNLOCK && 
                target->pending_net_ops_list_head == NULL) {
            MPIDI_CH3_Pkt_flags_t flag = MPIDI_CH3_PKT_FLAG_RMA_UNLOCK_NO_ACK;
            mpi_errno = send_unlock_msg(target->target_rank, win_ptr, flag);
            if (mpi_errno != MPI_SUCCESS)
                MPIR_ERR_POP(mpi_errno);
        }

        if (target->sync.sync_flag == MPIDI_RMA_SYNC_FLUSH ||
            target->sync.sync_flag == MPIDI_RMA_SYNC_UNLOCK || 
            target->win_complete_flag) {
            if (target->win_complete_flag && target->issue_2s_sync != 1) {
                MPIDI_CH3I_RDMA_set_CC(win_ptr, target->target_rank);
            }
            target->sync.sync_flag = MPIDI_RMA_SYNC_NONE;
        }

    }

    if (target->pending_net_ops_list_head == NULL && target->issue_2s_sync == 0) {
        win_ptr->num_targets_with_pending_net_ops--;
        MPIU_Assert(win_ptr->num_targets_with_pending_net_ops >= 0);
        if (win_ptr->num_targets_with_pending_net_ops == 0) {
            MPIDI_CH3I_Win_set_inactive(win_ptr);
        }
    }
fn_fail:;
}

/* For active synchronization */
#undef FUNCNAME
#define FUNCNAME MPIDI_CH3I_RDMA_post
#undef FCNAME
#define FCNAME MPL_QUOTE(FCNAME)
int MPIDI_CH3I_RDMA_post(MPID_Win * win_ptr, int target_rank)
{
    MPIDI_STATE_DECL(MPID_STATE_MPIDI_CH3I_RDMA_POST);
    MPIDI_FUNC_ENTER(MPID_STATE_MPIDI_CH3I_RDMA_POST);
    int mpi_errno = MPI_SUCCESS;

    char                *origin_addr, *remote_addr;
    MPIDI_VC_t          *tmp_vc;
    MPID_Comm           *comm_ptr;
    uint32_t            i, size, hca_index, 
                        r_key[MAX_NUM_SUBRAILS],
                        l_key[MAX_NUM_SUBRAILS];
    vbuf                *v = NULL;

    /*part 1 prepare origin side buffer */
    size = sizeof(int);

    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    origin_addr = (char *) v->buffer;
    *((int *) origin_addr) = 1;
    remote_addr = (char *) win_ptr->remote_post_flags[target_rank];

    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, target_rank, &tmp_vc);

    for (i=0; i<rdma_num_rails; ++i) {
        hca_index = i/rdma_num_rails_per_hca;
        l_key[i] = v->region->mem_handle[hca_index]->lkey;
        r_key[i] = win_ptr->post_flag_rkeys[target_rank * rdma_num_hcas + hca_index];
    }

    Post_Put_Put_Get_List(win_ptr, -1, NULL, 
            tmp_vc, &v, (void *)&origin_addr, 
            (void*)&remote_addr, size, 
            l_key, r_key, SINGLE, target_rank);

    win_ptr->poll_flag = 1;
    while (win_ptr->poll_flag == 1)
    {
        if ((mpi_errno = MPIDI_CH3I_Progress_test()) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }

fn_fail:
    MPIDI_FUNC_EXIT(MPID_STATE_MPIDI_CH3I_RDMA_POST);
    return mpi_errno;
}

void
MPIDI_CH3I_RDMA_win_create (void *base,
                            MPI_Aint size,
                            int comm_size,
                            int my_rank,
                            MPID_Win ** win_ptr, MPID_Comm * comm_ptr)
{
 
    int             ret, i,j,arrIndex;
    MPIR_Errflag_t  errflag = MPIR_ERR_NONE;
    win_info        *win_info_exchange;
    uintptr_t       *cc_ptrs_exchange;
    uintptr_t       *post_flag_ptr_send, *post_flag_ptr_recv;
    int             fallback_trigger = 0;

    if (mv2_MPIDI_CH3I_RDMA_Process.enable_rma_fast_path == 1)
    {
        (*win_ptr)->enable_fast_path = 1;
    }

    if (!mv2_MPIDI_CH3I_RDMA_Process.has_one_sided)
    {
        (*win_ptr)->fall_back = 1;
        goto fn_exit;
    }

    if ((*win_ptr)->create_flavor == MPI_WIN_FLAVOR_DYNAMIC)
    {
        (*win_ptr)->fall_back = 1;
        goto fn_exit;
    }
    
    /*Allocate structure for window information exchange*/
    win_info_exchange = MPIU_Malloc(comm_size * sizeof(win_info));
    if (!win_info_exchange)
    {
        DEBUG_PRINT("Error malloc win_info_exchange when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /*Allocate memory for completion counter pointers exchange*/
    cc_ptrs_exchange =  MPIU_Malloc(comm_size * sizeof(uintptr_t) * rdma_num_rails);
    if (!cc_ptrs_exchange)
    {
        DEBUG_PRINT("Error malloc cc_ptrs_exchangee when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    (*win_ptr)->fall_back = 0;
    /*Register the exposed buffer for this window */
    if (base != NULL && size > 0) {

        (*win_ptr)->win_dreg_entry = dreg_register(base, size);
        if (NULL == (*win_ptr)->win_dreg_entry) {
            (*win_ptr)->fall_back = 1;
            goto err_base_register;
        }
        for (i=0; i < rdma_num_hcas; ++i) {
            win_info_exchange[my_rank].win_rkeys[i] = 
                 (uint32_t) (*win_ptr)->win_dreg_entry->memhandle[i]->rkey;
        }
    } else {
        (*win_ptr)->win_dreg_entry = NULL;
        for (i = 0; i < rdma_num_hcas; ++i) {
             win_info_exchange[my_rank].win_rkeys[i] = 0;
        }
    }

    /*Register buffer for completion counter */
    (*win_ptr)->completion_counter = MPIU_Malloc(sizeof(long long) * comm_size 
                * rdma_num_rails);
    if (NULL == (*win_ptr)->completion_counter) {
        /* FallBack case */
        (*win_ptr)->fall_back = 1;
        goto err_cc_buf;
    }

    MPIU_Memset((void *) (*win_ptr)->completion_counter, 0, sizeof(long long)   
            * comm_size * rdma_num_rails);

    (*win_ptr)->completion_counter_dreg_entry = dreg_register(
        (void*)(*win_ptr)->completion_counter, sizeof(long long) * comm_size 
               * rdma_num_rails);
    if (NULL == (*win_ptr)->completion_counter_dreg_entry) {
        /* FallBack case */
        (*win_ptr)->fall_back = 1;
        goto err_cc_register;
    }

    for (i = 0; i < rdma_num_rails; ++i){
        cc_ptrs_exchange[my_rank * rdma_num_rails + i] =
               (uintptr_t) ((*win_ptr)->completion_counter + i);
    }

    for (i = 0; i < rdma_num_hcas; ++i){
        win_info_exchange[my_rank].completion_counter_rkeys[i] =
               (uint32_t) (*win_ptr)->completion_counter_dreg_entry->
                            memhandle[i]->rkey;
    }

    /*Register buffer for post flags : from target to origin */
    (*win_ptr)->post_flag = (int *) MPIU_Malloc(comm_size * sizeof(int)); 
    if (!(*win_ptr)->post_flag) {
        (*win_ptr)->fall_back = 1;
        goto err_postflag_buf;
    }
    DEBUG_PRINT(
        "rank[%d] : post flag start before exchange is %p\n",
        my_rank,
        (*win_ptr)->post_flag
    ); 
  
    /* Register the post flag */
    (*win_ptr)->post_flag_dreg_entry = 
                dreg_register((void*)(*win_ptr)->post_flag, 
                               sizeof(int)*comm_size);
    if (NULL == (*win_ptr)->post_flag_dreg_entry) {
        /* Fallback case */
        (*win_ptr)->fall_back = 1;
        goto err_postflag_register;
    }

    for (i = 0; i < rdma_num_hcas; ++i)
    {
        win_info_exchange[my_rank].post_flag_rkeys[i] = 
              (uint32_t)((*win_ptr)->post_flag_dreg_entry->memhandle[i]->rkey);
        DEBUG_PRINT(
            "the rank [%d] post_flag rkey before exchange is %x\n", 
            my_rank,
            win_info_exchange[my_rank].post_flag_rkeys[i]
        );
    }

    win_info_exchange[my_rank].fall_back = (*win_ptr)->fall_back;    

    /*Exchange the information about rkeys and addresses */
    /* All processes will exchange the setup keys *
     * since each process has the same data for all other
     * processes, use allgather */

    ret = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, win_info_exchange,
               sizeof(win_info), MPI_BYTE, comm_ptr, &errflag);
    if (ret != MPI_SUCCESS) {
        DEBUG_PRINT("Error gather win_info  when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* check if any peers fail */
    for (i = 0; i < comm_size; ++i)
    {
            if (win_info_exchange[i].fall_back != 0)
            {
                fallback_trigger = 1;
            }
    } 
    
    if (fallback_trigger) {
        MPIU_Free(win_info_exchange);
        MPIU_Free(cc_ptrs_exchange);
        dreg_unregister((*win_ptr)->post_flag_dreg_entry);
        MPIU_Free((*win_ptr)->post_flag);
        dreg_unregister((*win_ptr)->completion_counter_dreg_entry);
        MPIU_Free((*win_ptr)->completion_counter);
        dreg_unregister((*win_ptr)->win_dreg_entry);
        (*win_ptr)->fall_back = 1;
        goto fn_exit;
    }

    win_elem_t *new_element = (win_elem_t *) MPIU_Malloc(sizeof(win_elem_t));
    new_element->next = NULL;

    if (mv2_win_list) {                                                                               
        new_element->prev = mv2_win_list->prev;                                                         
        mv2_win_list->prev->next = new_element;                                                     
        mv2_win_list->prev = new_element;                                                            
        new_element->next = NULL;                                                              
    } else {                                                                                  
        mv2_win_list = new_element;                                                                         
        mv2_win_list->prev = new_element;                                                           
        mv2_win_list->next = NULL;                                                             
    }                          

    new_element->win_base = (*win_ptr)->win_dreg_entry;
    new_element->complete_counter = (*win_ptr)->completion_counter_dreg_entry;
    new_element->post_flag = (*win_ptr)->post_flag_dreg_entry; 
    ret = MPIR_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cc_ptrs_exchange,
             rdma_num_rails*sizeof(uintptr_t), MPI_BYTE, comm_ptr, &errflag);
    if (ret != MPI_SUCCESS) {
        DEBUG_PRINT("Error cc pointer  when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* Now allocate the rkey array for all other processes */
    (*win_ptr)->win_rkeys = (uint32_t *) MPIU_Malloc(comm_size * sizeof(uint32_t) 
                             * rdma_num_hcas);
    if (!(*win_ptr)->win_rkeys) {
        DEBUG_PRINT("Error malloc win->win_rkeys when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* Now allocate the rkey2 array for all other processes */
    (*win_ptr)->completion_counter_rkeys = (uint32_t *) MPIU_Malloc(comm_size * 
                             sizeof(uint32_t) * rdma_num_hcas); 
    if (!(*win_ptr)->completion_counter_rkeys) {
        DEBUG_PRINT("Error malloc win->completion_counter_rkeys when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* Now allocate the completion counter array for all other processes */
    (*win_ptr)->all_completion_counter = (long long **) MPIU_Malloc(comm_size * 
                             sizeof(long long *) * rdma_num_rails);
    if (!(*win_ptr)->all_completion_counter) {
        DEBUG_PRINT
            ("error malloc win->all_completion_counter when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* Now allocate the post flag rkey array for all other processes */
    (*win_ptr)->post_flag_rkeys = (uint32_t *) MPIU_Malloc(comm_size *
                                                  sizeof(uint32_t) * rdma_num_hcas);
    if (!(*win_ptr)->post_flag_rkeys) {
        DEBUG_PRINT("error malloc win->post_flag_rkeys when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* Now allocate the post flag ptr array for all other processes */
    (*win_ptr)->remote_post_flags =
        (int **) MPIU_Malloc(comm_size * sizeof(int *));
    if (!(*win_ptr)->remote_post_flags) {
        DEBUG_PRINT
            ("error malloc win->remote_post_flags when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    for (i = 0; i < comm_size; ++i)
    {
        for (j = 0; j < rdma_num_hcas; ++j) 
        {
            arrIndex = rdma_num_hcas * i + j;
            (*win_ptr)->win_rkeys[arrIndex] = 
                    win_info_exchange[i].win_rkeys[j];
            (*win_ptr)->completion_counter_rkeys[arrIndex] = 
                    win_info_exchange[i].completion_counter_rkeys[j];
            (*win_ptr)->post_flag_rkeys[arrIndex] = 
                    win_info_exchange[i].post_flag_rkeys[j];
        }
    }

    for (i = 0; i < comm_size; ++i)
    {
        for (j = 0; j < rdma_num_rails; ++j)
        {
            arrIndex = rdma_num_rails * i + j;
            (*win_ptr)->all_completion_counter[arrIndex] = (long long *)
                    ((size_t)(cc_ptrs_exchange[arrIndex])
                    + sizeof(long long) * my_rank * rdma_num_rails);
        }
    }

    post_flag_ptr_send = (uintptr_t *) MPIU_Malloc(comm_size * 
            sizeof(uintptr_t));
    if (!post_flag_ptr_send) {
        DEBUG_PRINT("Error malloc post_flag_ptr_send when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    post_flag_ptr_recv = (uintptr_t *) MPIU_Malloc(comm_size * 
            sizeof(uintptr_t));
    if (!post_flag_ptr_recv) {
        DEBUG_PRINT("Error malloc post_flag_ptr_recv when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    /* use all to all to exchange rkey and address for post flag */
    for (i = 0; i < comm_size; ++i)
    {
        (*win_ptr)->post_flag[i] = i != my_rank ? 0 : 1;
        post_flag_ptr_send[i] = (uintptr_t) &((*win_ptr)->post_flag[i]);
    }

    /* use all to all to exchange the address of post flag */
    ret = MPIR_Alltoall_impl(post_flag_ptr_send, sizeof(uintptr_t), MPI_BYTE, 
            post_flag_ptr_recv, sizeof(uintptr_t), MPI_BYTE, comm_ptr, &errflag);
    if (ret != MPI_SUCCESS) {
        DEBUG_PRINT("Error gather post flag ptr  when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    for (i = 0; i < comm_size; ++i) {
        (*win_ptr)->remote_post_flags[i] = (int *) post_flag_ptr_recv[i];
        DEBUG_PRINT(" rank is %d remote rank %d,  post flag addr is %p\n",
                my_rank, i, (*win_ptr)->remote_post_flags[i]);
    }

    MPIU_Free(win_info_exchange);
    MPIU_Free(cc_ptrs_exchange);
    MPIU_Free(post_flag_ptr_send);
    MPIU_Free(post_flag_ptr_recv);
    (*win_ptr)->using_lock = 0;
    (*win_ptr)->using_start = 0;
    /* Initialize put/get queue */
    (*win_ptr)->put_get_list_size = 0;
    (*win_ptr)->put_get_list_tail = 0;
    (*win_ptr)->wait_for_complete = 0;
    (*win_ptr)->rma_issued = 0;

    (*win_ptr)->put_get_list =
        (MPIDI_CH3I_RDMA_put_get_list *) MPIU_Malloc( 
            rdma_default_put_get_list_size *
            sizeof(MPIDI_CH3I_RDMA_put_get_list));
    if (!(*win_ptr)->put_get_list) {
        DEBUG_PRINT("Fail to malloc space for window put get list\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }

    (*win_ptr)->put_get_list_size_per_process =
        (int *) MPIU_Malloc (sizeof(int) * comm_size);
    if (!(*win_ptr)->put_get_list_size_per_process) {
        DEBUG_PRINT("Fail to malloc space for window put get list per process\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }
    MPIU_Memset((*win_ptr)->put_get_list_size_per_process,
            0 , sizeof(int)*comm_size);
fn_exit:
    
    if (1 == (*win_ptr)->fall_back) {
        (*win_ptr)->using_lock = 0;
        (*win_ptr)->using_start = 0;
        /* Initialize put/get queue */
        (*win_ptr)->put_get_list_size = 0;
        (*win_ptr)->put_get_list_tail = 0;
        (*win_ptr)->wait_for_complete = 0;
        (*win_ptr)->rma_issued = 0;
    }

    return;

  err_postflag_register:
    MPIU_Free((*win_ptr)->post_flag);
  err_postflag_buf:
    dreg_unregister((*win_ptr)->completion_counter_dreg_entry);
  err_cc_register:
    MPIU_Free((*win_ptr)->completion_counter);
  err_cc_buf:
    dreg_unregister((*win_ptr)->win_dreg_entry);
  err_base_register:
    win_info_exchange[my_rank].fall_back = (*win_ptr)->fall_back;

    ret = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, win_info_exchange, 
                      sizeof(win_info), MPI_BYTE, comm_ptr, &errflag);
    if (ret != MPI_SUCCESS) {
        DEBUG_PRINT("Error gather window information when creating windows\n");
        ibv_error_abort (GEN_EXIT_ERR, "rdma_iba_1sc");
    }
 
    MPIU_Free(win_info_exchange);
    MPIU_Free(cc_ptrs_exchange);
    goto fn_exit;
     
}

void MPIDI_CH3I_RDMA_win_free(MPID_Win** win_ptr)
{
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

    MPIU_Free((*win_ptr)->win_rkeys);
    MPIU_Free((*win_ptr)->completion_counter_rkeys);
    MPIU_Free((*win_ptr)->post_flag);
    MPIU_Free((*win_ptr)->post_flag_rkeys);
    MPIU_Free((*win_ptr)->remote_post_flags);
    MPIU_Free((*win_ptr)->put_get_list);
    MPIU_Free((*win_ptr)->put_get_list_size_per_process);

    MPIU_Free((*win_ptr)->completion_counter);
    MPIU_Free((*win_ptr)->all_completion_counter);
}

int MPIDI_CH3I_RDMA_set_CC(MPID_Win * win_ptr, int target_rank)
{
    int                 i;
    int                 mpi_errno = MPI_SUCCESS;
    uint32_t            r_key2[MAX_NUM_SUBRAILS], l_key2[MAX_NUM_SUBRAILS];
    int hca_index;
    void * remote_addr[MAX_NUM_SUBRAILS], *local_addr[MAX_NUM_SUBRAILS];
    vbuf *v[MAX_NUM_SUBRAILS];

    MPIDI_VC_t *tmp_vc;
    MPID_Comm *comm_ptr;

    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, target_rank, &tmp_vc);
    
    for (i=0; i<rdma_num_rails; ++i) {
            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v[i], MV2_SMALL_DATA_VBUF_POOL_OFFSET);
            local_addr[i] = (void *) v[i]->buffer;
            hca_index = i/rdma_num_rails_per_hca;
            remote_addr[i]    = (void *)(uintptr_t)
                (win_ptr->all_completion_counter[target_rank*rdma_num_rails+i]);
	    *((long long *) local_addr[i]) = 1;
            l_key2[i] = v[i]->region->mem_handle[hca_index]->lkey;
            r_key2[i] = win_ptr->completion_counter_rkeys[target_rank*rdma_num_hcas + hca_index];
    }
 
    Post_Put_Put_Get_List(win_ptr, -1, NULL, tmp_vc, &v[0], 
            local_addr, remote_addr, sizeof (long long), l_key2, r_key2, REPLICATE, 
            target_rank);
    return mpi_errno;
    
}

static int Post_Put_Put_Get_List(  MPID_Win * winptr, 
                            MPIDI_msg_sz_t size, 
                            dreg_entry * dreg_tmp,
                            MPIDI_VC_t * vc_ptr, vbuf **allocated_v,
                            void *local_buf[], void *remote_buf[],
                            MPIDI_msg_sz_t length,
                            uint32_t lkeys[], uint32_t rkeys[],
                            rail_select_t rail_select, int target)
{
    int mpi_errno = MPI_SUCCESS;
    int rail, i, index;
    MPIDI_msg_sz_t count, bytes_per_rail, posting_length;
    void *local_address, *remote_address;
    vbuf *v;
    MPIDI_VC_t *save_vc = vc_ptr;

    while(winptr->put_get_list[winptr->put_get_list_tail].status == BUSY) {
        winptr->put_get_list_tail = (winptr->put_get_list_tail + 1 ) % rdma_default_put_get_list_size;
    }

    index = winptr->put_get_list_tail;
    winptr->put_get_list[index].op_type     = SIGNAL_FOR_PUT;
    winptr->put_get_list[index].mem_entry   = dreg_tmp;
    winptr->put_get_list[index].data_size   = size;
    winptr->put_get_list[index].win_ptr     = winptr;
    winptr->put_get_list[index].vc_ptr      = vc_ptr;
    winptr->put_get_list_tail = (winptr->put_get_list_tail + 1) %
                    rdma_default_put_get_list_size;
    winptr->put_get_list[index].completion = 0;
    winptr->put_get_list[index].target_rank = target;
    winptr->put_get_list[index].status      = BUSY;

    if (rail_select == STRIPE) { /* stripe the message across rails */
        /*post data in chunks of mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize as long as the 
          total size per rail is larger than that*/
        count = 0;
        bytes_per_rail = length/rdma_num_rails;
        while (bytes_per_rail > mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize) {
           for (i = 0; i < rdma_num_rails; ++i) {
              GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET); 
              if (NULL == v) {
                  MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
              }

              v->list = (void *)(&(winptr->put_get_list[index]));
              v->vc = (void *)vc_ptr;

              ++(winptr->put_get_list[index].completion);
              ++(winptr->put_get_list_size);
              ++(winptr->rma_issued);
              ++(winptr->put_get_list_size_per_process[target]);

              local_address = (void *)((char*)local_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
                         + i*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
              remote_address = (void *)((char *)remote_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize
                         + i*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);

              vbuf_init_rma_put(v, 
                                local_address, 
                                lkeys[i], 
                                remote_address,
                                rkeys[i], 
                                mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize, 
                                i);
              ONESIDED_RDMA_POST(v, vc_ptr, save_vc, i);

           }
          
           ++count;
           length -= rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize;
           bytes_per_rail = length/rdma_num_rails; 
        }

        /* Post remaining data as length < rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
         * Still stripe if length > rdma_large_msg_rail_sharing_threshold*/
        if (length < rdma_large_msg_rail_sharing_threshold) { 
           rail = MRAILI_Send_select_rail(vc_ptr);
           GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
           if (NULL == v) {
               MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
           }
 
           ++winptr->put_get_list[index].completion;
           ++(winptr->put_get_list_size);
           ++(winptr->rma_issued);
           ++(winptr->put_get_list_size_per_process[target]);

           local_address = (void *)((char*)local_buf[0]
                      + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
           remote_address = (void *)((char *)remote_buf[0]
                      + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
 
           vbuf_init_rma_put(v, 
                             local_address, 
                             lkeys[rail], 
                             remote_address,
                             rkeys[rail], 
                             length, 
                             rail);
           v->list = (void *)(&(winptr->put_get_list[index]));
           v->vc = (void *)vc_ptr;

            ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);
 
        } else {
           for (i = 0; i < rdma_num_rails; ++i) {
              GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET); 
              if (NULL == v) {
                  MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
              }

              v->list = (void *)(&(winptr->put_get_list[index]));
              v->vc = (void *)vc_ptr;

              ++(winptr->put_get_list[index].completion);
              ++(winptr->put_get_list_size);
              ++(winptr->rma_issued);
              ++(winptr->put_get_list_size_per_process[target]);

              local_address = (void *)((char*)local_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
                         + i*bytes_per_rail);
              remote_address = (void *)((char *)remote_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize
                         + i*bytes_per_rail);

              if (i < rdma_num_rails - 1) {
                 posting_length = bytes_per_rail;
              } else {
                 posting_length = length - (rdma_num_rails - 1)*bytes_per_rail;
              }

              vbuf_init_rma_put(v, 
                                local_address, 
                                lkeys[i], 
                                remote_address,
                                rkeys[i], 
                                posting_length, 
                                i);
              ONESIDED_RDMA_POST(v, vc_ptr, save_vc, i);

           }
        }
    } else if (rail_select == SINGLE) { /* send on a single rail */
        rail = MRAILI_Send_select_rail(vc_ptr);

        if (*allocated_v == NULL) { 
            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
        } else {
            v = *allocated_v;
        }
        if (NULL == v) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }

        v->list = (void *)(&(winptr->put_get_list[index]));
        v->vc = (void *)vc_ptr;

        ++(winptr->put_get_list[index].completion);
        ++(winptr->put_get_list_size);
        ++(winptr->rma_issued);
        ++(winptr->put_get_list_size_per_process[target]);

        vbuf_init_rma_put(v, 
                          local_buf[0], 
                          lkeys[rail], 
                          remote_buf[0],
                          rkeys[rail], 
                          length, 
                          rail);

        ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);

    } else if (rail_select == REPLICATE) { /* send on all rails */
        for (i = 0; i < rdma_num_rails; ++i) {
	    if (*allocated_v == NULL) {
               GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET); 
               if (NULL == v) {
                  MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
               }
            } else {
               v = allocated_v[i];
            }

            v->list = (void *)(&(winptr->put_get_list[index]));
            v->vc = (void *)vc_ptr;

            ++(winptr->put_get_list[index].completion);
            ++(winptr->put_get_list_size);
            ++(winptr->rma_issued);
            ++(winptr->put_get_list_size_per_process[target]);

            vbuf_init_rma_put(v, 
                              local_buf[i], 
                              lkeys[i], 
                              remote_buf[i],
                              rkeys[i], 
                              length, 
                              i);

            ONESIDED_RDMA_POST(v, vc_ptr, save_vc, i);
 
        }
    }
 
    while (winptr->put_get_list_size >= rdma_default_put_get_list_size)
    {
        if ((mpi_errno = MPIDI_CH3I_Progress_test()) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }
    
fn_fail:
    return mpi_errno;
}

static int Post_Get_Put_Get_List(  MPID_Win * winptr, 
                            MPIDI_msg_sz_t size, 
                            dreg_entry * dreg_tmp,
                            MPIDI_VC_t * vc_ptr, vbuf **allocated_v,
                            void *local_buf[], void *remote_buf[],
                            void *user_buf[], MPIDI_msg_sz_t length,
                            uint32_t lkeys[], uint32_t rkeys[],
                            rail_select_t rail_select, int target)
{
     int mpi_errno = MPI_SUCCESS;
     int i, rail, index;
     MPIDI_msg_sz_t posting_length, bytes_per_rail, count;
     void *local_address, *remote_address;
     vbuf *v;
     MPIDI_VC_t *save_vc = vc_ptr;

     while(winptr->put_get_list[winptr->put_get_list_tail].status == BUSY) {
         winptr->put_get_list_tail = (winptr->put_get_list_tail + 1 ) % rdma_default_put_get_list_size;
     }

     index = winptr->put_get_list_tail;
     if(size <= rdma_eagersize_1sc){    
         winptr->put_get_list[index].origin_addr = local_buf[0];
         winptr->put_get_list[index].target_addr = user_buf[0];
     } else {
         winptr->put_get_list[index].origin_addr = NULL;
         winptr->put_get_list[index].target_addr = NULL;
     }
     winptr->put_get_list[index].op_type     = SIGNAL_FOR_GET;
     winptr->put_get_list[index].mem_entry   = dreg_tmp;
     winptr->put_get_list[index].data_size   = size;
     winptr->put_get_list[index].win_ptr     = winptr;
     winptr->put_get_list[index].vc_ptr      = vc_ptr;
     winptr->put_get_list_tail = (winptr->put_get_list_tail + 1) %
                                   rdma_default_put_get_list_size;
     winptr->put_get_list[index].completion = 0;
     winptr->put_get_list[index].target_rank = target;
     winptr->put_get_list[index].status      = BUSY;

     if (rail_select == STRIPE) { /*stripe across the rails*/
        /*post data in chunks of mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize as long as the 
          total size per rail is larger than that*/
        count = 0;
        bytes_per_rail = length/rdma_num_rails;
        while (bytes_per_rail > mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize) {
           for (i = 0; i < rdma_num_rails; ++i) {
              GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET); 
              if (NULL == v) {
                  MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
              }

              v->list = (void *)(&(winptr->put_get_list[index]));
              v->vc = (void *)vc_ptr;

              ++(winptr->put_get_list[index].completion);
              ++(winptr->put_get_list_size);
              ++(winptr->rma_issued);
              ++(winptr->put_get_list_size_per_process[target]);

              local_address = (void *)((char*)local_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
                         + i*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
              remote_address = (void *)((char *)remote_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize
                         + i*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);

              vbuf_init_rma_get(v, 
                                local_address, 
                                lkeys[i], 
                                remote_address,
                                rkeys[i], 
                                mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize, 
                                i);

              ONESIDED_RDMA_POST(v, vc_ptr, save_vc, i);

           }
          
           ++count;
           length -= rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize;
           bytes_per_rail = length/rdma_num_rails; 
        }

        /* Post remaining data as length < rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
         * Still stripe if length > rdma_large_msg_rail_sharing_threshold*/
        if (length < rdma_large_msg_rail_sharing_threshold) { 
           rail = MRAILI_Send_select_rail(vc_ptr);
           GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
           if (NULL == v) {
               MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
           }
 
           ++winptr->put_get_list[index].completion;
           ++(winptr->put_get_list_size);
           ++(winptr->rma_issued);
           ++(winptr->put_get_list_size_per_process[target]);

           local_address = (void *)((char*)local_buf[0]
                      + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
           remote_address = (void *)((char *)remote_buf[0]
                      + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize);
 
           vbuf_init_rma_get(v, 
                             local_address, 
                             lkeys[rail], 
                             remote_address,
                             rkeys[rail], 
                             length, 
                             rail);
           v->list = (void *)(&(winptr->put_get_list[index]));
           v->vc = (void *)vc_ptr;

           ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);
 
        } else {
           for (i = 0; i < rdma_num_rails; ++i) {
              GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET); 
              if (NULL == v) {
                  MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
              }

              v->list = (void *)(&(winptr->put_get_list[index]));
              v->vc = (void *)vc_ptr;

              ++(winptr->put_get_list[index].completion);
              ++(winptr->put_get_list_size);
              ++(winptr->rma_issued);
              ++(winptr->put_get_list_size_per_process[target]);

              local_address = (void *)((char*)local_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize 
                         + i*bytes_per_rail);
              remote_address = (void *)((char *)remote_buf[0] 
                         + count*rdma_num_rails*mv2_MPIDI_CH3I_RDMA_Process.maxtransfersize
                         + i*bytes_per_rail);

              if (i < rdma_num_rails - 1) {
                 posting_length = bytes_per_rail;
              } else {
                 posting_length = length - (rdma_num_rails - 1)*bytes_per_rail;
              }

              vbuf_init_rma_get(v, 
                                local_address, 
                                lkeys[i], 
                                remote_address,
                                rkeys[i], 
                                posting_length, 
                                i);

              ONESIDED_RDMA_POST(v, vc_ptr, save_vc, i);

           }
        }
    } else if (rail_select == SINGLE) { /* send on a single rail */
        rail = MRAILI_Send_select_rail(vc_ptr);
 
        if (*allocated_v == NULL) { 
            GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
        } else {
            v = *allocated_v;
        }
        if (NULL == v) {
            MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
        }

        v->list = (void *)(&(winptr->put_get_list[index]));
        v->vc = (void *)vc_ptr;

        winptr->put_get_list[index].completion = 1;
        ++(winptr->put_get_list_size);
        ++(winptr->rma_issued);
        ++(winptr->put_get_list_size_per_process[target]);

        vbuf_init_rma_get(v, local_buf[0], lkeys[rail], remote_buf[0],
                          rkeys[rail], length, rail);

        ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);

    }

    while (winptr->put_get_list_size >= rdma_default_put_get_list_size)
    {
        if ((mpi_errno = MPIDI_CH3I_Progress_test()) != MPI_SUCCESS)
        {
            MPIR_ERR_POP(mpi_errno);
        }
    }

fn_fail:
    return mpi_errno;
}

int MRAILI_Handle_one_sided_completions(vbuf * v)                            
{
    dreg_entry      	          *dreg_tmp;
    MPIDI_msg_sz_t                size;
    int                           mpi_errno = MPI_SUCCESS;
    void                          *target_addr, *origin_addr;
    MPID_Win                      *list_win_ptr;
    if ( v->target_rank == -1) {
        MPIDI_CH3I_RDMA_put_get_list  *list_entry=NULL;
        list_entry = (MPIDI_CH3I_RDMA_put_get_list *)v->list;
        list_win_ptr = list_entry->win_ptr;

        switch (list_entry->op_type) {
            case (SIGNAL_FOR_PUT):
                {
                    dreg_tmp = list_entry->mem_entry;
                    size = list_entry->data_size;

                    if (size > (int)rdma_eagersize_1sc) {
                        --(list_entry->completion);
                        if (list_entry->completion == 0) {
                            dreg_unregister(dreg_tmp);
                        }
                    }
                    --(list_win_ptr->put_get_list_size);
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[list_entry->target_rank]);
                    list_entry->status = FREE;
                    break;
                }
            case (SIGNAL_FOR_GET):
                {
                    size = list_entry->data_size;
                    target_addr = list_entry->target_addr;
                    origin_addr = list_entry->origin_addr;
                    dreg_tmp = list_entry->mem_entry;

                    if (origin_addr == NULL) {
                        MPIU_Assert(size > rdma_eagersize_1sc); 
                        --(list_entry->completion);
                        if (list_entry->completion == 0){
                            dreg_unregister(dreg_tmp);
                        }
                    } else {
                        MPIU_Assert(size <= rdma_eagersize_1sc);
                        MPIU_Assert(target_addr != NULL);
                        MPIU_Memcpy(target_addr, origin_addr, size);
                    }
                    --(list_win_ptr->put_get_list_size);
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[list_entry->target_rank]);
                    list_entry->status = FREE;
                    break;
                }
            case (SIGNAL_FOR_COMPARE_AND_SWAP):
                {
                    target_addr = list_entry->target_addr;
                    origin_addr = list_entry->origin_addr;
#ifdef INFINIBAND_VERBS_EXP_H
                    if (g_atomics_support_be) {
                        *((uint64_t *) target_addr) = ntohll(*((uint64_t *) origin_addr));
                    } else
#endif  /* INFINIBAND_VERBS_EXP_H */
                    {
                        *((uint64_t *) target_addr) = *((uint64_t *) origin_addr);
                    }
                    --(list_entry->completion);
                    --(list_win_ptr->put_get_list_size);
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[list_entry->target_rank]);
                    break;
                }
            case (SIGNAL_FOR_FETCH_AND_ADD):
                {   
                    target_addr = list_entry->target_addr;
                    origin_addr = list_entry->origin_addr;
#ifdef INFINIBAND_VERBS_EXP_H
                    if (g_atomics_support_be) {
                        *((uint64_t *) target_addr) = ntohll(*((uint64_t *) origin_addr));
                    } else
#endif /* INFINIBAND_VERBS_EXP_H */
                    {
                        *((uint64_t *) target_addr) = *((uint64_t *) origin_addr);
                    }
                    --(list_entry->completion);
                    --(list_win_ptr->put_get_list_size);
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[list_entry->target_rank]);
                    break;
                }

            default:
                MPIR_ERR_SETSIMPLE(mpi_errno, MPI_ERR_OTHER, "**onesidedcomps");
                break;
        }
        if (list_win_ptr->put_get_list_size == 0) 
            list_win_ptr->put_get_list_tail = 0;

    } else {
        list_win_ptr = (MPID_Win *)v->list;

        switch (v->desc.u.sr.opcode) {
            case (IBV_WR_RDMA_WRITE):
                {
                    if (v->tmp_dreg != NULL) {
                        dreg_unregister((dreg_entry *) v->tmp_dreg);
                    }
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[v->target_rank]);
                    break;
                }
            case (IBV_WR_RDMA_READ):
                {
                    if ( v->tmp_dreg != NULL) {
                        dreg_unregister((dreg_entry *) v->tmp_dreg);
                    }
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[v->target_rank]);
                    break;
                }
            case (IBV_WR_ATOMIC_FETCH_AND_ADD):
                {
#ifdef INFINIBAND_VERBS_EXP_H
                    if (g_atomics_support_be) {
                        *((uint64_t *) v->result_addr) = ntohll(*((uint64_t *) v->buffer));
                    } else 
#endif /* INFINIBAND_VERBS_EXP_H */
                    {
                        *((uint64_t *) v->result_addr) = *((uint64_t *) v->buffer);
                    }
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[v->target_rank]);
                    break;
                }
            case (IBV_WR_ATOMIC_CMP_AND_SWP):
                {
#ifdef INFINIBAND_VERBS_EXP_H
                    if (g_atomics_support_be) {
                        *((uint64_t *) v->result_addr) = ntohll(*((uint64_t *) v->buffer));
                    } else 
#endif /* INFINIBAND_VERBS_EXP_H */
                    {
                        *((uint64_t *) v->result_addr) = *((uint64_t *) v->buffer);
                    }
                    --(list_win_ptr->rma_issued);
                    --(list_win_ptr->put_get_list_size_per_process[v->target_rank]);
                    break;
                }
            default:
                MPIR_ERR_SETSIMPLE(mpi_errno, MPI_ERR_OTHER, "**onesidedcomps");
                break;

        }
    }

    if (list_win_ptr->rma_issued == 0) {
        list_win_ptr->poll_flag = 0;
        MPIDI_CH3_Progress_signal_completion();
    }
#ifndef CHANNEL_MRAIL
fn_fail:
#endif
    return mpi_errno;
}

static int iba_put(MPIDI_RMA_Op_t * rma_op, MPID_Win * win_ptr, MPIDI_msg_sz_t size)
{
    char                *remote_address;
    int                 mpi_errno = MPI_SUCCESS;
    int                 hca_index;
    uint32_t            r_key[MAX_NUM_SUBRAILS],
                        l_key1[MAX_NUM_SUBRAILS], 
                        l_key[MAX_NUM_SUBRAILS];
    int                 i;
    dreg_entry          *tmp_dreg = NULL;
    char                *origin_addr;

    MPIDI_VC_t          *tmp_vc;
    MPID_Comm           *comm_ptr;
    vbuf                *v = NULL; 

    /*part 1 prepare origin side buffer target buffer and keys */
    MPIU_Assert(rma_op->pkt.type == MPIDI_CH3_PKT_PUT_IMMED || 
                rma_op->pkt.type == MPIDI_CH3_PKT_PUT);
    remote_address = rma_op->pkt.put.addr;

    if (likely(size <= rdma_eagersize_1sc)) {
        MV2_GET_RC_VBUF(v, size);
	    origin_addr = (char *)v->buffer;
        MPIU_Memcpy(origin_addr, (char *)rma_op->origin_addr, size);

        for(i = 0; i < rdma_num_hcas; ++i) {
            l_key[i] = v->region->mem_handle[i]->lkey;
        }
    } else {
        tmp_dreg = dreg_register(rma_op->origin_addr, size);

        for(i = 0; i < rdma_num_hcas; ++i) {
            l_key[i] = tmp_dreg->memhandle[i]->lkey;
        }        

        origin_addr = rma_op->origin_addr;
        win_ptr->wait_for_complete = 1;
    }
    
    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, rma_op->target_rank, &tmp_vc);

    for (i=0; i<rdma_num_rails; ++i) {
        hca_index = i/rdma_num_rails_per_hca;
        r_key[i] = win_ptr->win_rkeys[rma_op->target_rank*rdma_num_hcas + hca_index];
        l_key1[i] = l_key[hca_index];
    }

    if(size < rdma_large_msg_rail_sharing_threshold || size < rdma_eagersize_1sc) {
        Post_Put_Put_Get_List(win_ptr, size, tmp_dreg, tmp_vc, &v, 
                (void *)&origin_addr, (void *)&remote_address, 
                size, l_key1, r_key, SINGLE, rma_op->target_rank);
    } else {
        Post_Put_Put_Get_List(win_ptr, size, tmp_dreg, tmp_vc, &v,
                (void *)&origin_addr, (void *)&remote_address,
                size, l_key1, r_key, STRIPE, rma_op->target_rank);
    }

    return mpi_errno;
}

int iba_get(MPIDI_RMA_Op_t * rma_op, MPID_Win * win_ptr, MPIDI_msg_sz_t size)
{
    int                 mpi_errno = MPI_SUCCESS;
    int                 hca_index;
    uint32_t            r_key[MAX_NUM_SUBRAILS], 
                        l_key1[MAX_NUM_SUBRAILS], 
                        l_key[MAX_NUM_SUBRAILS];
    dreg_entry          *tmp_dreg = NULL;
    char                *target_addr, *user_addr;
    char                *remote_addr;
    int                 i;
    MPIDI_VC_t          *tmp_vc;
    MPID_Comm           *comm_ptr;
    vbuf                *v = NULL;

    /*part 1 prepare origin side buffer target address and keys  */
    MPIU_Assert(rma_op->pkt.type == MPIDI_CH3_PKT_GET);
    remote_addr = rma_op->pkt.get.addr;

    if (size <= rdma_eagersize_1sc) {
        MV2_GET_RC_VBUF(v, size);
        for(i = 0; i < rdma_num_hcas; ++i) {
            l_key[i] = v->region->mem_handle[i]->lkey;
        }
        target_addr = (char *)v->buffer;
        user_addr = (char *)rma_op->origin_addr;
    } else {
        tmp_dreg = dreg_register(rma_op->origin_addr, size);
        for(i = 0; i < rdma_num_hcas; ++i) {
            l_key[i] = tmp_dreg->memhandle[i]->lkey;
        }
        target_addr = user_addr = rma_op->origin_addr;
        win_ptr->wait_for_complete = 1;
    }

    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, rma_op->target_rank, &tmp_vc);

   for (i=0; i<rdma_num_rails; ++i)
   {
       hca_index = i/rdma_num_rails_per_hca;
       r_key[i] = win_ptr->win_rkeys[rma_op->target_rank*rdma_num_hcas + hca_index];
       l_key1[i] = l_key[hca_index];
   }

   if(size < rdma_large_msg_rail_sharing_threshold || size < rdma_eagersize_1sc) {
       Post_Get_Put_Get_List(win_ptr, size, tmp_dreg, 
                             tmp_vc, &v, (void *)&target_addr, 
                             (void *)&remote_addr, (void*)&user_addr, size, 
                             l_key1, r_key, SINGLE, rma_op->target_rank );
   } else {
       Post_Get_Put_Get_List(win_ptr, size, tmp_dreg,
                             tmp_vc, &v, (void *)&target_addr,
                             (void *)&remote_addr, (void*)&user_addr, size,
                             l_key1, r_key, STRIPE, rma_op->target_rank );
   }

   return mpi_errno;
}


int iba_compare_and_swap(MPIDI_RMA_Op_t * rma_op, MPID_Win * win_ptr, MPIDI_msg_sz_t size)
{
    char                *remote_addr = NULL, *local_addr;
    uint64_t            compare_value, swap_value;
    int                 mpi_errno = MPI_SUCCESS;
    int                 hca_index, rail, index;
    uint32_t            r_key, l_key;

    MPIDI_VC_t          *vc_ptr;
    MPID_Comm           *comm_ptr;
    vbuf                *v = NULL;

    /*prepare target buffer and key*/
    MPIU_Assert(rma_op->pkt.type == MPIDI_CH3_PKT_CAS_IMMED);
    remote_addr = rma_op->pkt.cas.addr;

    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, rma_op->target_rank, &vc_ptr);
    rail = 0;
    hca_index = vc_ptr->mrail.rails[rail].hca_index;
    r_key = win_ptr->win_rkeys[rma_op->target_rank * rdma_num_hcas + hca_index];
    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    if (NULL == v)
        MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");

    index = win_ptr->put_get_list_tail;
    win_ptr->put_get_list[index].op_type        = SIGNAL_FOR_COMPARE_AND_SWAP;
    win_ptr->put_get_list[index].win_ptr        = win_ptr;
    win_ptr->put_get_list[index].vc_ptr         = vc_ptr;
    win_ptr->put_get_list[index].target_addr    = rma_op->result_addr;
    win_ptr->put_get_list[index].origin_addr    = v->buffer;
    win_ptr->put_get_list_tail = (win_ptr->put_get_list_tail + 1)
                % rdma_default_put_get_list_size;
    win_ptr->put_get_list[index].completion = 0;
    win_ptr->put_get_list[index].target_rank = rma_op->target_rank;


    ++(win_ptr->put_get_list[index].completion);
    ++(win_ptr->put_get_list_size);
    ++(win_ptr->rma_issued);
    ++(win_ptr->put_get_list_size_per_process[rma_op->target_rank]);
    v->vc = (void *)vc_ptr;
    v->list = (void *)(&(win_ptr->put_get_list[index]));
    local_addr = (char *)v->buffer;
    *((uint64_t *)local_addr) = 0;
    l_key = v->region->mem_handle[hca_index]->lkey;

    compare_value = *((uint64_t *) rma_op->compare_addr);
    swap_value    = *((uint64_t *) rma_op->origin_addr);
    vbuf_init_rma_compare_and_swap(v, local_addr, l_key, remote_addr, r_key,
            compare_value, swap_value, rail);
    ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);

fn_fail:
    return mpi_errno;
}

int iba_fetch_and_add(MPIDI_RMA_Op_t *rma_op, MPID_Win *win_ptr, MPIDI_msg_sz_t size)
{
    char                *remote_addr = NULL;
    uint64_t            *fetch_addr;
    uint64_t            add_value;
    int                 mpi_errno = MPI_SUCCESS;
    int                 hca_index, rail, index;
    uint32_t            r_key, l_key;

    MPIDI_VC_t          *vc_ptr;
    MPID_Comm           *comm_ptr;
    vbuf                *v = NULL;

    hca_index = 0; rail = 0;

    /*prepaer target buffer and key, using HCA 0*/
    MPIU_Assert(rma_op->pkt.type == MPIDI_CH3_PKT_FOP_IMMED ||
                    rma_op->pkt.type == MPIDI_CH3_PKT_FOP);
    remote_addr = rma_op->pkt.fop.addr;

    r_key = win_ptr->win_rkeys[rma_op->target_rank*rdma_num_hcas + hca_index];

    comm_ptr = win_ptr->comm_ptr;
    MPIDI_Comm_get_vc(comm_ptr, rma_op->target_rank, &vc_ptr);

    GET_VBUF_BY_OFFSET_WITHOUT_LOCK(v, MV2_SMALL_DATA_VBUF_POOL_OFFSET);
    if(NULL == v){
        MPIR_ERR_SETFATALANDJUMP(mpi_errno, MPI_ERR_OTHER, "**nomem");
    }

    index = win_ptr->put_get_list_tail;
    win_ptr->put_get_list[index].op_type        = SIGNAL_FOR_FETCH_AND_ADD;
    win_ptr->put_get_list[index].win_ptr        = win_ptr;
    win_ptr->put_get_list[index].vc_ptr         = vc_ptr;
    win_ptr->put_get_list[index].target_addr    = rma_op->result_addr;
    win_ptr->put_get_list[index].origin_addr    = v->buffer;
    win_ptr->put_get_list_tail = (win_ptr->put_get_list_tail + 1 )
        % rdma_default_put_get_list_size;
    win_ptr->put_get_list[index].completion     = 0;
    win_ptr->put_get_list[index].target_rank    = rma_op->target_rank;
    

    ++(win_ptr->put_get_list[index].completion);
    ++(win_ptr->put_get_list_size);
    ++(win_ptr->rma_issued);
    ++(win_ptr->put_get_list_size_per_process[rma_op->target_rank]);

    v->vc   = (void *)vc_ptr;
    v->list = (void *)(&(win_ptr->put_get_list[index]));

    fetch_addr = (uint64_t *) v->buffer;
    *fetch_addr = 0;
    l_key = v->region->mem_handle[hca_index]->lkey;
    add_value = *((uint64_t *) rma_op->origin_addr);
    vbuf_init_rma_fetch_and_add(v, (void *) fetch_addr, l_key, remote_addr, r_key, add_value, rail);
    ONESIDED_RDMA_POST(v, vc_ptr, NULL, rail);

fn_fail:
    return mpi_errno;
}


