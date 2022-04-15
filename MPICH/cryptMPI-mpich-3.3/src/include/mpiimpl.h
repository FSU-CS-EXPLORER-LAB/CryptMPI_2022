/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * Portions of this code were written by Microsoft. Those portions are
 * Copyright (c) 2007 Microsoft Corporation. Microsoft grants
 * permission to use, reproduce, prepare derivative works, and to
 * redistribute to others. The code is licensed "as is." The User
 * bears the risk of using it. Microsoft gives no express warranties,
 * guarantees or conditions. To the extent permitted by law, Microsoft
 * excludes the implied warranties of merchantability, fitness for a
 * particular purpose and non-infringement.
 */

#ifndef MPIIMPL_H_INCLUDED
#define MPIIMPL_H_INCLUDED

#include "mpichconfconst.h"
#include "mpichconf.h"

#include <stdio.h>
#ifdef STDC_HEADERS
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#else
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#endif

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

/* for MAXHOSTNAMELEN under Linux and OSX */
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#if defined (HAVE_USLEEP)
#include <unistd.h>
#if defined (NEEDS_USLEEP_DECL)
int usleep(useconds_t usec);
#endif
#endif

#if defined(HAVE_LONG_LONG_INT)
/* Assume two's complement for determining LLONG_MAX (already assumed
 * elsewhere in MPICH). */
#ifndef LLONG_MAX
/* slightly tricky (values in binary):
 * - put a 1 in the second-to-msb digit                   (0100...0000)
 * - sub 1, giving all 1s starting at third-to-msb digit  (0011...1111)
 * - shift left 1                                         (0111...1110)
 * - add 1, yielding all 1s in positive space             (0111...1111) */
#define LLONG_MAX (((((long long) 1 << (sizeof(long long) * CHAR_BIT - 2)) - 1) << 1) + 1)
#endif
#endif /* defined(HAVE_LONG_LONG_INT) */

#if (!defined MAXHOSTNAMELEN) && (!defined MAX_HOSTNAME_LEN)
#define MAX_HOSTNAME_LEN 256
#elif !defined MAX_HOSTNAME_LEN
#define MAX_HOSTNAME_LEN MAXHOSTNAMELEN
#endif

/* This allows us to keep names local to a single file when we can use
   weak symbols */
#ifdef  USE_WEAK_SYMBOLS
#define PMPI_LOCAL static
#else
#define PMPI_LOCAL
#endif

/* Fix for universal endianess added in autoconf 2.62 */
#ifdef WORDS_UNIVERSAL_ENDIAN
#if defined(__BIG_ENDIAN__)
#elif defined(__LITTLE_ENDIAN__)
#define WORDS_LITTLEENDIAN
#else
#error 'Universal endianess defined without __BIG_ENDIAN__ or __LITTLE_ENDIAN__'
#endif
#endif

#if defined(HAVE_VSNPRINTF) && defined(NEEDS_VSNPRINTF_DECL) && !defined(vsnprintf)
int vsnprintf(char *str, size_t size, const char *format, va_list ap);
#endif


/*****************************************************************************
 * We use the following ordering of information in this file:
 *
 *   1. Start with independent headers that do not have any
 *      dependencies on the rest of the MPICH implementation (e.g.,
 *      mpl, opa, mpi.h).
 *
 *   2. Next is forward declarations of MPIR structures (MPIR_Comm,
 *      MPIR_Win, etc.).
 *
 *   3. After that we have device-independent headers (MPIR
 *      functionality that does not have any dependency on MPID).
 *
 *   4. Next is the device "pre" header that defines device-level
 *      initial objects that would be used by the MPIR structures.
 *
 *   5. Then comes the device-dependent MPIR functionality, with the
 *      actual definitions of structures, function prototypes, etc.
 *      This functionality can only rely on the device "pre"
 *      functionality.
 *
 *   6. Finally, we'll add the device "post" header that is allowed to
 *      use anything from the MPIR layer.
 *****************************************************************************/


/*****************************************************************************/
/*********************** PART 1: INDEPENDENT HEADERS *************************/
/*****************************************************************************/

/* if we are defining this, we must define it before including mpl.h */
#if defined(MPICH_DEBUG_MEMINIT)
#define MPL_VG_ENABLED 1
#endif

#include "mpl.h"
#include "opa_primitives.h"
#include "mpi.h"

/************ Mohsen *************/

#define COLL_NAIVE_AESGCM 0      // encypt and decrypt both intra and inter node comm: reduce, allreduce, bcast
#define COLL_NAIVE_AESGCM_PLUS 1 // encrypt and derypt only inter-node communication

extern int node_cnt;
extern int PRINT_FUN_NAME;
extern int ALLGATHER_PRINT_FUN_NAME;
extern int ENABLE_SECURE_DEBUG;
extern int CONCUR_INTER_METHOD;
extern int CONCUR_RS_METHOD;
extern int ALLGATHER_METHOD;
extern int SHOW_FINAL_RESULT;
extern int ALLREDUCE_CONCUR_ENABLE;
extern int SHMEM_BCAST;
extern int DEBUG_INIT_FILE;
extern int init_phase;
extern int PRINT_Ring_FUN_NAME;
extern int PRINT_RSA_FUN_NAME;
extern int PRINT_SUPER;
extern int leader_cnt;
extern int Allgather_Reduce;
extern int enc_choping_sz;
extern int super_node;
extern int UNSEC_ALLREDUCE_MULTI_LEADER;
extern int *comm_rank_list;
extern int *comm_rank_list_back;
extern int *node_sizes;
extern int init_rank;
extern int mv2_user_allgather_inter;
/*** Mehran Added ***/
extern int bcast_tuning;
/********************/
extern int inter_scatter_tuning;
// extern int is_uniform;
extern int inter_gather_tuning;
extern int inter_allreduce_tuning;

/* Collective debug function flag */
#define ALLREDUCE_PRINT_FUN  0		//Mohsen
#define Print_Ring  0		//Mohsen
#define ALLGATHER_PRINT_FUN  0
#define BCAST_PRINT_FUN  0
#define GATHER_PRINT_FUN 0
#define SCATTER_PRINT_FUN 0

/************ End *************/


/************************************/
/*  Added by Abu Naser              */
/************************************/
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/aead.h>
#include <openssl/rand.h>
#include <pthread.h>
#include <sys/ipc.h>
#include <sys/shm.h>

/* Debug Flag */
#define ENABLE_SECURE_MPI_DEBUG 0
#define ENABLE_SECURE_CHOPP_DEBUG 0
#define ENABLE_COUNTER_DEBUG 0
#define ENABLE_SECURE_STAT 0
#define ENABLE_MEASURE_INIT_TIME 0
#define UN_ENCRYPTED_BASE 0       /* set 1 for un-encrypted base */
#define OMP_DEBUG 0
#define OMP_DEBUG_DYNAMIC_THREAD_PIPELINE 0
#define OMP_DEBUG_V 0
#define PSC_THREAD_USE_DEBUG 0
#define NONBLOCK_RECV_DEBUG 0
#define NONBLOCK_SEND_DEBUG 0

/* Adaptive chopping */
#define CRYPTMPI_ADAPTIVE_CHOPP 1

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN "\x1b[36m"
#define COLOR_RESET "\x1b[0m"

/* Define sizes */
#define COMMON_LARGE_SEND_BUFFER_SIZE  67110656 //67108864 //150401232 //9437184 
#define COMMON_LARGE_RECV_BUFFER_SIZE  67110656 //67108864 //150401232 //9437184 
#define NON_BLOCKING_SND_RCV_1 64
#define NON_BLOCKING_SND_RCV_2 9437184
#define NON_BLOCK_SEND NON_BLOCKING_SND_RCV_1
#define NON_BLOCK_SEND2 NON_BLOCKING_SND_RCV_2
//#define SIZE_REQ_HANDLE_CHOPPING 5000

/* Define header sizes */
#define MSG_HEADER_SIZE 25 
#define ENC_MSG_TAG_SIZE 16
#define NONCE_HEADER 5
#define NONCE_SIZE 12
#define MAX_OMP_THREADS_LIMIT 20
#define SYMMETRIC_KEY_SIZE 16

/* Externs */
extern EVP_AEAD_CTX *ctx;
extern EVP_AEAD_CTX *local_ctx;
extern EVP_CIPHER_CTX *ctx_enc; 
extern unsigned char Ideciphertext[NON_BLOCKING_SND_RCV_1][NON_BLOCKING_SND_RCV_2];
extern unsigned char Iciphertext[NON_BLOCK_SEND][NON_BLOCK_SEND2];
extern int nonBlockCounter;

extern unsigned char symmetric_key[300];
extern int symmetric_key_size;
extern int choping_sz;
extern unsigned char large_send_buffer[COMMON_LARGE_SEND_BUFFER_SIZE];
extern unsigned char large_recv_buffer[COMMON_LARGE_RECV_BUFFER_SIZE];
//extern int security_approach;

#if CRYPTMPI_ADAPTIVE_CHOPP
extern int cryptMPI_adap_chop;
#endif

//typedef struct isend_req isend_req;
#define ISEND_REQ NON_BLOCKING_SND_RCV_1 //100001
extern int isendCounter;

struct isend_req{
  int source;
  int dest;
  int tag;
  int totaldata;
  unsigned char * buffer;
  int req_type;
  int segment_number;
  MPI_Request request[8200]; //300 //
  int msg_chop_size;
  int total_request;
  int test_flag;
};

extern struct isend_req nonblock_req_handler[ISEND_REQ+5];
#define FOUR_K 4096
#define SIXTEEN_K 16384
#define THIRTY_2K 32768
#define SIXTY_4K 65536
#define ONE_TWO_8K 131072
#define TWO_FIVE_6K 262144
#define FIVE_ONE_2K 524288
#define ONE_M 1048576
#define TWO_M  2097152

#define OPENMP_THREADS 1
#define MY_OPENMP_THREADS_NO openmp_active_thread_no
#define SKIP_INTRA_ENC 1 // 0 means Encrypting also intra node communication, 1 means skipping intra node encryption

#define PIPELINE_SIZE  FIVE_ONE_2K 
#define LARGE_SEGMENT_SIZE  1048575 // ONE_M -1 
#define PSC_BRIDGE_TUNE 0 // psc bridge specific tunning; also need to enable omp pipe line and omp dyn thread
#define SUBKEY_GEN_START 65535 // SIXTY_4K - 1
#define ENABLE_CRYPTMPI_RC_SWITCH 0 
#define BLOCKING_CHOP_SZ PIPELINE_SIZE
#define MULTI_THREAD_START_SIZE 16384

#define OMP_DYNAMIC_THREADS_PIPELINE 1
#define OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION 0
#define PSC_BRIDGE_TUNE 0
#define CRYPTMPI_SINGLE_THREAD_MODE 0
#define ONLY_ONE_THREAD_PIPELINE 0
#define OMP_DYNAMIC_THREADS 0
#define OMP_DYNAMIC_THREADS_INNOVATION 0
#define ALL_COMM_PLUS_ENC_TIME 0

#define MAX_PENDING_ISEND_LIMIT 64 
#define MAX_PROCESS_SIZE 2000 
#define MAX_RANKS_LIMIT 2000
extern int ranksIrecvS[MAX_RANKS_LIMIT]; // check the posted irecv requests
extern long pendingIsendRequestCount[MAX_RANKS_LIMIT];
extern  int no_of_max_omp_threads;
extern int cryptmpi_process_id;
extern char cryptmpi_process_name[MPI_MAX_PROCESSOR_NAME];
extern int cryptmpi_init_done;
extern int cyptmpi_series_thread;
extern  int cryptmpi_own_rank;            // rank
extern int cryptmpi_local_process;        // number of process in the node
extern int cryptmpi_local_rank;           // node id
extern char all_p_names[2048*MPI_MAX_PROCESSOR_NAME];
extern int sameNode[2048];
//extern int sendtag[2048];
//extern int recvtag[2048];
struct CryptHandleProbe{
  int no_tag;
  unsigned char header[200][MSG_HEADER_SIZE];
  int tags[200];
};

extern struct CryptHandleProbe Crypthandle_probe[2048]; 
extern int long_msg_flag ;

/*keys */
extern EVP_AEAD_CTX * global_openmp_ctx;
extern EVP_AEAD_CTX * global_small_msg_ctx;
extern EVP_AEAD_CTX * global_coll_msg_ctx;
extern EVP_AEAD_CTX * global_ctx;
extern int openmp_active_thread_no;

/******* counter related flags begin ***************/
/**************************************************/
#define COUNTER_HEADER_SIZE 26 

/* base counter flags*/
//#define BASE_COUNTER_MODE 0
#define BASE_COUNTER_NO_PIPELINE 1   // if 1 then there is no pipeline, 0 otherwise
#define BASE_COUNTER_LIBRARY_NONCE 0 // if 1 then generate nonce using library call rand_bytes, otherwise otpmized one (default)
//#define COLL_NAIVE_BASECOUNTER 0  // collective enc

/* pre-computation flag */
//#define PRE_COMPUTE_COUNTER_MODE 1
#define DYNAMIC_PIPELINE 1        // keep it as 1
//#define COLL_NAIVE_PRECOMPUTE 1  // collective enc
#define MAX_COMMON_COUNTER_SZ 8388608 // 8MB
#define INITIAL_COMMON_COUNTER_SZ 4096 // 4K
#define PRE_COM_DATA_RANGE SIXTY_4K

/* Debug flags for counter mode */
#define DEBUG_BASE_COUNTER 0
#define DEBUG_PRECOMPUTE_COUNTER 0

/* Externs */
extern EVP_CIPHER_CTX * global_counter_ctx;
extern EVP_CIPHER_CTX * local_counter_ctx[MAX_OMP_THREADS_LIMIT]; 
extern EVP_CIPHER_CTX * base_counter_ctx;
extern unsigned char Send_common_IV[32];
extern unsigned char Recv_common_IV[MAX_PROCESS_SIZE*32];
extern unsigned char enc_common_buffer[MAX_COMMON_COUNTER_SZ];
extern unsigned char  dec_common_buffer[MAX_COMMON_COUNTER_SZ];
extern unsigned char  zeros[MAX_COMMON_COUNTER_SZ];
extern unsigned int enc_common_start;
extern unsigned int enc_common_end;
extern unsigned long enc_common_counter;
extern unsigned long enc_common_counter_long_msg;
extern unsigned long base_global_counter;
extern int common_compute_size;
extern unsigned long counter_needto_send;
extern unsigned long counter_needto_send_large_msg;

extern void IV_Count(unsigned char *IV, unsigned long cter);
// extern inline void IV_Count_out(unsigned char *IV, unsigned long cter, unsigned char *in);
extern void IV_Count_out(unsigned char *IV, unsigned long cter, unsigned char *in);
extern void encryption_common_counter(const void *buf, int next, int enc_datasize, int pos, unsigned char * out_buf);
extern void decryption_common_counter(void *buf, int pre_com_dec_mask_sz, int datasz, int recv_pos, int dec_pos, int mask_pos,
                               unsigned long dec_counter, int source, unsigned char * in_buf);
extern void decryption_common_counter_ivflag(void *buf, int pre_com_dec_mask_sz, int datasz, int recv_pos, int dec_pos, int mask_pos,
                               unsigned long dec_counter, int source, unsigned char * in_buf, char pflag);                               
extern void multithread_decryption_common_counter_large_msg(void *buf, int datasz, int recv_pos, int dec_pos, unsigned long dec_counter, 
                            unsigned char * in, EVP_CIPHER_CTX * t_counter_ctx, int source);
extern void multithreads_encryption_common_counter_large_msg(const void *buf, int enc_datasize,  unsigned char * out,
                                             EVP_CIPHER_CTX * t_counter_ctx, long unsigned int counter, int tid);  

/************************************/
/*  End of add                      */
/************************************/

/****************************** Added by Mehran *****************************/

extern int security_approach;
int overlap_decryption, shmem_leaders, concurrent_comm;
int concurrent_bcast, o_bruck2;
key_t shmem_key, ciphertext_shmem_key;
int shmid, ciphertext_shmid, allocated_shmem;
void *shmem_buffer, *ciphertext_shmem_buffer;

extern int init_shmem();
extern int create_concurrent_comm (MPI_Comm comm, int size, int my_rank);
extern int host_checking_comm (MPI_Comm comm, int **node_sizes_p, int **comm_rank_list_p, int **comm_rank_list_back_p);
/**************************************************************************/


/*****************************************************************************/
/*********************** PART 2: FORWARD DECLARATION *************************/
/*****************************************************************************/

struct MPIR_Request;
typedef struct MPIR_Request MPIR_Request;

struct MPIR_Comm;
typedef struct MPIR_Comm MPIR_Comm;

struct MPIR_Datatype;
typedef struct MPIR_Datatype MPIR_Datatype;

struct MPIR_Win;
typedef struct MPIR_Win MPIR_Win;

struct MPIR_Info;
typedef struct MPIR_Info MPIR_Info;

struct MPIR_Group;
typedef struct MPIR_Group MPIR_Group;

struct MPIR_Topology;
typedef struct MPIR_Topology MPIR_Topology;


/*****************************************************************************/
/******************* PART 3: DEVICE INDEPENDENT HEADERS **********************/
/*****************************************************************************/

#include "mpir_misc.h"
#include "mpir_dbg.h"
#include "mpir_objects.h"
#include "mpir_strerror.h"
#include "mpir_type_defs.h"
#include "mpir_assert.h"
#include "mpir_pointers.h"
#include "mpir_refcount.h"
#include "mpir_mem.h"
#include "mpir_info.h"
#include "mpir_errhandler.h"
#include "mpir_attr_generic.h"
#include "mpir_contextid.h"
#include "mpir_status.h"
#include "mpir_debugger.h"
#include "mpir_op.h"
#include "mpir_topo.h"
#include "mpir_tags.h"
#include "mpir_pt2pt.h"
#include "mpir_ext.h"

#ifdef HAVE_CXX_BINDING
#include "mpii_cxxinterface.h"
#endif

#ifdef HAVE_FORTRAN_BINDING
#include "mpii_f77interface.h"
#endif

#include "coll_types.h"
#include "coll_impl.h"

/*****************************************************************************/
/********************** PART 4: DEVICE PRE DECLARATION ***********************/
/*****************************************************************************/

#include "mpidpre.h"


/*****************************************************************************/
/********************* PART 5: DEVICE DEPENDENT HEADERS **********************/
/*****************************************************************************/

#include "mpir_thread.h"
#include "mpir_attr.h"
#include "mpir_group.h"
#include "mpir_comm.h"
#include "mpir_request.h"
#include "mpir_win.h"
#include "mpir_coll.h"
#include "mpir_func.h"
#include "mpir_err.h"
#include "mpir_nbc.h"
#include "mpir_process.h"
#include "mpir_dataloop.h"
#include "mpir_datatype.h"
#include "mpir_cvars.h"
#include "mpir_misc_post.h"
#include "mpit.h"
#include "mpir_handlemem.h"

/*****************************************************************************/
/******************** PART 6: DEVICE "POST" FUNCTIONALITY ********************/
/*****************************************************************************/


/* Added by abu naser */
int MPIR_Naive_Sec_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, int recvcount, MPI_Datatype recvtype,
                      int root, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);                   
int MPIR_Scatter_MV2_Direct_unencrypted(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);     
int MPIR_Scatter_MV2_Direct_shmem(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_H2(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_CHS(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_no_shmem(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_no_shmem_pipeline(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_no_shmem_intra_RR(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_HS2_ML_Pipeline(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);  
int MPIR_Scatter_MV2_Direct_HS2_ML_UNENC(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag); 
int MPIR_Scatter_MV2_Direct_HBcast(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_Scatter_MV2_Direct_no_shmem_intra_RR_UNENC(const void *sendbuf,
                            int sendcnt,
                            MPI_Datatype sendtype,
                            void *recvbuf,
                            int recvcnt,
                            MPI_Datatype recvtype,
                            int root, MPIR_Comm * comm_ptr, MPIR_Errflag_t *errflag);
int MPIR_ENC_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                  MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);                                                                                    
// gather                                                       
/*int MPIR_Naive_Sec_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype,
                          int root, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);*/ 
int MPIR_Gather_MV2_Direct_CHS(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPIR_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag);
int MPIR_Gather_MV2_Direct_CHS_UNENCRYPTED(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPIR_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag);                    
int MPIR_Gather_MV2_Direct_CNoSM_RR(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                      int recvcount, MPI_Datatype recvtype, int root, MPIR_Comm *comm_ptr,
                      MPIR_Errflag_t *errflag);                                                                                                                                                                                                                                                                                              

/* end of add*/

#include "mpidpost.h"

/* avoid conflicts in source files with old-style "char FCNAME[]" vars */
#undef FUNCNAME
#undef FCNAME

#endif /* MPIIMPL_H_INCLUDED */
