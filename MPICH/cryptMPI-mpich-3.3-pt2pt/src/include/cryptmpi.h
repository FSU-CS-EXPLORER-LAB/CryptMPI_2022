/* Added by Abu Naser(an16e@my.fsu.edu) */
#define ENABLE_SECURE_MPI 1
#if ENABLE_SECURE_MPI  
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/aead.h>
#include <openssl/rand.h>
//#define _GNU_SOURCE 
#include <pthread.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <omp.h>



/* Enable/Disable Secure options */
#define DEFAULT_ENCRYPTION_SCHEME ENABLE_SECURE_MPI
#define SYMMETRIC_KEY_SIZE 16 //USER_KEY_SIZE
//#define AES_GCM ENABLE_AES_GCM
//#define AES_GCM_CHOPPING ENABLE_AES_GCM_CHOPPING
//#define MSG_CHOPPING_SIZE USER_MSG_CHOPP_SIZE



/* Debug Flag */
#define ENABLE_SECURE_MPI_DEBUG 0
#define ENABLE_SECURE_CHOPP_DEBUG 0
#define ENABLE_COUNTER_DEBUG 0
#define ENABLE_SECURE_STAT 0
#define ENABLE_MEASURE_INIT_TIME 0
#define UN_ENCRYPTED_BASE 0       /* set 1 for un-encrypted base */

/* Adaptive chopping */
#define CRYPTMPI_ADAPTIVE_CHOPP 1
#define NON_BLOCKING_CHOP_SZ  60000 //65500 //65536 //12000 //134217728 bt could hault

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN "\x1b[36m"
#define COLOR_RESET "\x1b[0m"

/* Define sizes */
#define MILLIONS 2000000
#define COMMON_LARGE_SEND_BUFFER_SIZE  50401232 //150401232 //9437184 //4200000 //4194304 //33554432 // 67108864 //134217728 //268435456+2000 //268435456 367001600=350MB 268435456 = 256MB
#define COMMON_LARGE_RECV_BUFFER_SIZE  50401232 //150401232 //9437184 //4200000 //4194304 //33554432 // 67108864 // 134217728 //268435456+2000
#define SIZE_REQ_HANDLE_CHOPPING 5000

/* Total data 4 + random 16 + comm type 1 + chop size 4 bytes */
#define MSG_HEADER_SIZE 25 //20
#define ENC_MSG_TAG_SIZE 16
#define NONCE_HEADER 5
#define NONCE_SIZE 12

//#define PRE_COUNTER_HEADER_SIZE 9

#define NON_BLOCKING_SND_RCV_1 64
#define NON_BLOCKING_SND_RCV_2 9437184 //4200000  //2100000 //1048576 //  //1226400 //1126400 //2100000

#define NON_BLOCK_SEND NON_BLOCKING_SND_RCV_1
#define NON_BLOCK_SEND2 NON_BLOCKING_SND_RCV_2

#if CRYPTMPI_ADAPTIVE_CHOPP
#define COMMON_CHOPPING_BEGIN  cryptMPI_adap_chop //8192 //4194304 //8192 //1048576+20//1048576//8192 //32 //8192 // 33554432
#define COMMON_CHOPPING_SIZE  cryptMPI_adap_chop
#define CRYPTMPI_CHOPPING_SIZE 8192
#else
#define COMMON_CHOPPING_BEGIN  8192 //8192 //4194304 //8192 //1048576+20//1048576//8192 //32 //8192 // 33554432
#define COMMON_CHOPPING_SIZE  7500 //7//7515//16 // 7500, 7510, 7511, 7512, 7513, 7514
#endif

/* Externs */
extern EVP_AEAD_CTX *ctx;
extern EVP_CIPHER_CTX *ctx_counter;
extern EVP_CIPHER_CTX *ctx_enc; 
extern EVP_CIPHER_CTX *ctx_global_enc; 
extern EVP_CIPHER_CTX *ctx_global_dec; 
extern unsigned char Ideciphertext[NON_BLOCKING_SND_RCV_1][NON_BLOCKING_SND_RCV_2];
extern unsigned char Iciphertext[NON_BLOCK_SEND][NON_BLOCK_SEND2];
extern int nonBlockCounter;

extern unsigned char symmetric_key[300];
extern int symmetric_key_size;
extern int choping_sz;
extern unsigned char large_send_buffer[COMMON_LARGE_SEND_BUFFER_SIZE];
extern unsigned char large_recv_buffer[COMMON_LARGE_RECV_BUFFER_SIZE];

#if CRYPTMPI_ADAPTIVE_CHOPP
extern int cryptMPI_adap_chop;
#endif

//typedef struct isend_req isend_req;
#define ISEND_REQ NON_BLOCKING_SND_RCV_1 //100001
extern int isendCounter;

//#define NON_BLOCK_IRECV_REQ_Q_SIZE 16000
//extern MPI_Request irecvRequestQ[NON_BLOCK_IRECV_REQ_Q_SIZE];
//extern int front_of_irecvRequestQ;
//extern int rear_of_irecvRequestQ;

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
//extern unsigned long requestIndex[100000][2];


/* Enable debug */
#define DYNAMIC_DEBUG_V 0
#define DYNAMIC_DEBUG 0
#define STATIC_DEBUG_V 0
#define STATIC_DEBUG 0
#define GUIDED_DEBUG_V 0
#define GUIDED_DEBUG 0
#define MULTI_THREAD_DEBUG 0
#define OMP_DEBUG 0
#define OMP_DEBUG_V 0
#define OMP_DEBUG_DYNAMIC_THREAD_PIPELINE 0
#define NONBLOCK_RECV_DEBUG 0
#define NONBLOCK_SEND_DEBUG 0
#define CRYPTMPI_THREAD_MONITOR_DEBUG 0

//#define NO_OF_CORE 32
#define MY_MAX_NO_THREADS 1 // 3 = 1 process + 2 thread

//#define CHOPPING_ONLY 0
//#define MULTI_THREAD_CHOPPING 0 
//#define MULTI_THREAD_BATCH_SEND 0
#define MULTI_THREAD_ONLY 0 // Encrypt data using multithread 
#define MULTI_THREAD_BLOCK_SEND 0  // dynamic /static / guided shceduling

/* Enable scheduling algorithm */
#define DYNAMIC_SCH 0
#define STATIC_SCH 0
#define GUIDED_SCH 0
#define HYBRID_SCH 0

// ==================================================================

#define FOUR_K 4096
#define SIXTEEN_K 16384
#define THIRTY_2K 32768
#define SIXTY_4K 65536
#define ONE_TWO_8K 131072
#define TWO_FIVE_6K 262144
#define FIVE_ONE_2K 524288
#define ONE_M 1048576
#define TWO_M  2097152


/* openmp flags */
#define OMP_COLLECTIVES 0
#define OPENMP_THREADS 1
#define MY_OPENMP_THREADS_NO openmp_active_thread_no

#define SKIP_INTRA_ENC 1 // 0 means Encrypting also intra node communication, 1 means skipping intra node encryption

/* Multi-Thread */
#define OPENMP_MULTI_THREAD_ONLY 1        // both cluster, require when use only multi-thread. 
#define OMP_DYNAMIC_THREADS 0             // Noleland
#define OMP_DYNAMIC_THREADS_INNOVATION 0  

#define OPENMP_PIPE_LINE 1
#define OMP_DYNAMIC_THREADS_PIPELINE 1        // only for noleland cluster, require pipeline+multi-thread
#define CRYPTMPI_SINGLE_THREAD_MODE 0        // 1 for using single thread pipeline, above 2 flags should 1 
#define OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION 0 // only for innovation cluster, reruire pipeline+multi-thread
#define PIPELINE_SIZE  FIVE_ONE_2K 
#define LARGE_SEGMENT_SIZE  1048575 // ONE_M -1 
#define PSC_BRIDGE_TUNE 0 // psc bridge specific tunning; also need to enable omp pipe line and omp dyn thread

#define SUBKEY_GEN_START 65535 // SIXTY_4K - 1

#define ENABLE_CRYPTMPI_RC_SWITCH 0 // If it is 1, CryptMPI dynamically decides in runtime  CryptMPI-R or CryptMPI-C
                                    // in that case CRYPTMPI_SINGLE_THREAD_MODE should set 0.
#define COLL_NAIVE_AESGCM 0       // encypt and decrypt both intra and inter node communication: reduce, allreduce, bcast
//#define COLL_NAIVE_PLUS 0 // encrypt and derypt only inter-node communication

/* Collective debug function flag */
#define ALLREDUCE_PRINT_FUN 0
#define ALLTOALL_PRINT_FUN  0
#define ALLTOALLv_PRINT_FUN 0
#define REDUCE_PRINT_FUN    0
#define BCAST_PRINT_FUN     0

/* Collective tuning function flag */
#define ALLTOALL_TUNE_FLAG 0


//#define OMP_DYNAMIC_THREADS_PIPELINE_OPTIMIZAION 1
//#define OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION_OPTIMIZAION 0

//#define PIPELINE_SIZE  exp_pipeline_size 
//#define LARGE_SEGMENT_SIZE  exp_pipeline_start 


/* Only reuires for openmp pipeline */
#define OPENMP_DYNAMIC_SCH 0
#define OPENMP_STATIC_SCH 1
#define OPENMP_GUIDED_SCH 0


#define ONLY_ONE_THREAD_PIPELINE 0
#define BLOCKING_CHOP_SZ  PIPELINE_SIZE //130072

#define MAX_PENDING_ISEND_LIMIT 64 //32 //64 //512
#define NONBLOCK_REQ_OPTIMIZATION 1
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


#define BOLT_OPENMP_AGBOTS 0
#define CRYPTMPI_MONITOR_THREAD_USE 0

#if CRYPTMPI_MONITOR_THREAD_USE
extern int cryptmpi_shmid;
extern key_t cryptmpi_key;                // shared memory key
extern int *cryptmpi_r, *cryptmpi_ranks;
extern int extra_threads;
extern long encdecCounter;
extern long sendThinc_1;
extern long sendThinc_2;
extern long recvThinc_1;
extern long recvThinc_2 ;
extern long ed_times;
#endif

#define CRYPTMPI_SHMSZ  512
#define NO_OF_CORE 32
#define ENCDECLIMIT 10
#define FREQ_THRESHOLD 1.10
#define MAX_OMP_THREADS_LIMIT 20

#if OPENMP_THREADS
#include <omp.h>
#endif

/* For profiling */

#define LOG_DATA 0
#define COLL_LOG_DATA 0
#define ALL_COMM_PLUS_ENC_TIME 0


#if COLL_LOG_DATA

extern int max_bcast_datasz;
extern int max_reduce_datasz;
extern int max_allreduce_datasz;
#endif

#if LOG_DATA
#include <time.h> 

struct CRYPT_LOG_DATA{
  int rank;
  int dest_freq;
  int datasize[5000];
  int datasize_freq[5000];
  int no_datasize;
};

extern struct CRYPT_LOG_DATA cryptlogdata[128];
#endif


#if ALL_COMM_PLUS_ENC_TIME
extern struct timeval comm_start_time, comm_end_time;
extern double  total_comm_plus_enc_time;
extern double total_comm_plus_enc_time_long_msg;
extern double total_comm_plus_enc_time_small_msg;

extern double bcast_total_time;
extern double reduce_total_time;
extern double allreduce_total_time;
extern double alltoall_total_time;
extern double alltoallv_total_time;

extern double send_total_time;
extern double isend_total_time;
extern double recv_total_time;
extern double irecv_total_time;
extern double wait_total_time;
extern double waitall_total_time;

extern double inter_less_than_4K;
extern double inter_less_than_16K;
extern double inter_less_than_32K;
extern double inter_less_than_64K;

extern double inter_less_than_128K;
extern double inter_128K_256K;
extern double inter_256K_512K;
extern double inter_512K_1M;
extern double inter_more_than_1M;

extern double intra_less_than_128K;
extern double intra_128K_256K;
extern double intra_256K_512K;
extern double intra_512K_1M;
extern double intra_more_than_1M;
#endif

#define PSC_THREAD_USE_DEBUG 0

#if PSC_THREAD_USE_DEBUG
extern int max_send_thread_used;
extern int max_wait_thread_used;    
#endif

extern int long_msg_flag ;

#if 1 //ENC_DEC_TIME_DEBUG
#include <sys/time.h>
extern struct timeval  omp_tv1, omp_tv2, omp_tv3, omp_tv4, prog_start_time, prog_end_time;
extern double omp_t1,omp_t2, prog_exec_time;
#endif


#define COMM_TIME_DEBUG 0
#define ENC_DEC_TIME_DEBUG 0

/* Lock with conditional variable */
#define COND_LOCK 0
extern int cond_variable[MY_MAX_NO_THREADS+10];
extern pthread_mutex_t myp_lock;
extern pthread_cond_t myp_cond;


#define RECV_CHUNK_SIZE 1

/* Fixed size */
#define MULTI_THREAD_START_SIZE 16384 //16384 // if data size is 16K, then split it in multi-threads
 //SIXTY_4K //32768 //16384 //32768 //32768 //65536 //8000 //16304 //16384 //16304
#define CHUNK_SIZE 1

/* test with arguments */
//#define BLOCKING_CHOP_SZ test_block_chop_sz 
//#define CHUNK_SIZE test_chunk_sz

#define MAX_OPENMP_THREAD_NO 32


extern   unsigned char testNonce[12];
//extern volatile unsigned char nonce[50];
//extern volatile struct threadInfo tinfo[MY_MAX_NO_THREADS+10];
//extern int myThreadId[MY_MAX_NO_THREADS+10];
//extern pthread_t thread_id[MY_MAX_NO_THREADS+10];
//extern void * my_thread_operation(void *arg);

extern EVP_AEAD_CTX * global_openmp_ctx;
extern EVP_AEAD_CTX * global_small_msg_ctx;
extern EVP_AEAD_CTX * global_coll_msg_ctx;
extern EVP_AEAD_CTX * global_ctx;
extern int openmp_active_thread_no;


/* test */
extern int test_block_chop_sz; 
extern int test_chunk_sz;

extern long exp_pipeline_size;
extern long exp_pipeline_start;
extern long exp_thread_no;

#define DATA_SIZE_DEBUG 0

#if DATA_SIZE_DEBUG
#define DB_DATA 130
struct data_size_debug{
  int process[DB_DATA];
  int pdata[DB_DATA][DB_DATA]; 
  int freq[DB_DATA]; // freq of each data
};

extern struct  data_size_debug datasizeDB[DB_DATA];
#endif


// unencypted pipeline 

#define UNENCRYPTED_PIPELINE 0
#define UNENCREPTED_HEADER_SZ 26

/* 
* ---- Define all counter related flags here ---- 
*/
/*
 To use the following disable 
  OPENMP_MULTI_THREAD_ONLY and OPENMP_PIPE_LINE
*/
#define CRYPTMPI_COUNTER_MODE 1 // 1 to enable counter mode


#if CRYPTMPI_COUNTER_MODE
#define COUNTER_HEADER_SIZE 26 

/* base counter flags*/
#define BASE_COUNTER_MODE 1
#define BASE_COUNTER_NO_PIPELINE 1   // if 1 then there is no pipeline, 0 otherwise
#define BASE_COUNTER_LIBRARY_NONCE 0 // if 1 then generate nonce using library call rand_bytes, otherwise otpmized one (default)
#define COLL_NAIVE_BASECOUNTER 0  // collective enc

/* pre-computation flag */
#define PRE_COMPUTE_COUNTER_MODE 1
#define DYNAMIC_PIPELINE 1        // keep it as 1
#define COLL_NAIVE_PRECOMPUTE 0  // collective enc
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
extern inline void IV_Count_out(unsigned char *IV, unsigned long cter, unsigned char *in);
extern void encryption_common_counter(const void *buf, int next, int enc_datasize, int pos, unsigned char * out_buf);
extern void decryption_common_counter(void *buf, int pre_com_dec_mask_sz, int datasz, int recv_pos, int dec_pos, int mask_pos,
                               unsigned long dec_counter, int source, unsigned char * in_buf);
extern void decryption_common_counter_ivflag(void *buf, int pre_com_dec_mask_sz, int datasz, int recv_pos, int dec_pos, int mask_pos,
                               unsigned long dec_counter, int source, unsigned char * in_buf, char pflag);                               
extern void multithread_decryption_common_counter_large_msg(void *buf, int datasz, int recv_pos, int dec_pos, unsigned long dec_counter, 
                            unsigned char * in, EVP_CIPHER_CTX * t_counter_ctx, int source);
extern void multithreads_encryption_common_counter_large_msg(const void *buf, int enc_datasize,  unsigned char * out,
                                             EVP_CIPHER_CTX * t_counter_ctx, long unsigned int counter, int tid);                                                           

#endif // endof counter mode
#endif
/* End of add by Abu Naser */
