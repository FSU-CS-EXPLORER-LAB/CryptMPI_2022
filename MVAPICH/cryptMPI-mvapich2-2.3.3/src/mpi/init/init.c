/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
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
#include <strings.h>

#include "mpiimpl.h"
#include "mpi_init.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
#include "coll_shmem.h"
#endif

// Add By Mohsen
int init_phase=1;
int super_node=0;
int PRINT_FUN_NAME=0;
int ALLGATHER_PRINT_FUN_NAME=0;
int ENABLE_SECURE_DEBUG=0;
int DEBUG_INIT_FILE=0;
int SHOW_FINAL_RESULT=0;
int PRINT_Ring_FUN_NAME=0;
int PRINT_RSA_FUN_NAME=0;
int UNSEC_ALLREDUCE_MULTI_LEADER=0;
int PRINT_SUPER=0;

int CONCUR_INTER_METHOD=3;
int CONCUR_AllGTHER_METHOD=1;
int CONCUR_RS_METHOD=1;
int leader_cnt=1;
int SHMEM_BCAST=0;
int Allgather_Reduce=0;
int enc_choping_sz=0;

// End of Add 

/* Added by Abu Naser */
#include <openssl/rsa.h>
#include <openssl/pem.h>

int choping_sz = PIPELINE_SIZE;

void MPI_SEC_Initial_Key_Aggrement();
void init_crypto();
unsigned char symmetric_key[300];
int symmetric_key_size = 16;

EVP_AEAD_CTX *ctx = NULL;
// EVP_AEAD_CTX *local_ctx = NULL;
EVP_CIPHER_CTX *ctx_enc = NULL;
EVP_AEAD_CTX * global_openmp_ctx = NULL;
EVP_AEAD_CTX * global_small_msg_ctx = NULL;
EVP_AEAD_CTX * global_coll_msg_ctx = NULL;
EVP_AEAD_CTX * global_ctx = NULL;
EVP_CIPHER_CTX * global_counter_ctx = NULL;
EVP_CIPHER_CTX * local_counter_ctx[MAX_OMP_THREADS_LIMIT]; 
EVP_CIPHER_CTX * base_counter_ctx = NULL;

int openmp_active_thread_no = 1;
int cryptmpi_process_id = 0;
char cryptmpi_process_name[MPI_MAX_PROCESSOR_NAME];
char all_p_names[2048*MPI_MAX_PROCESSOR_NAME];
int sameNode[2048];
int total_process_number = 0;
int no_of_max_omp_threads = 0;
int noCommThreads = 2;
int cyptmpi_series_thread=0;

int cryptmpi_local_process =0;
int cryptmpi_local_rank;
int cryptmpi_own_rank;
int cryptmpi_init_done = 0;
int security_approach=0;

unsigned char Send_common_IV[32], Recv_common_IV[MAX_PROCESS_SIZE*32];
unsigned char enc_common_buffer[MAX_COMMON_COUNTER_SZ], dec_common_buffer[MAX_COMMON_COUNTER_SZ];
unsigned int enc_common_start, enc_common_end;
unsigned long enc_common_counter = 0; 
unsigned long counter_needto_send = 0; 
unsigned long counter_needto_send_large_msg = 0;
unsigned long enc_common_counter_long_msg = 0;
unsigned long base_global_counter;
unsigned char  zeros[MAX_COMMON_COUNTER_SZ];
int common_compute_size =0;
int long_msg_flag = 0;

/* End of add */

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

categories:
    - name        : THREADS
      description : multi-threading cvars

cvars:
    - name        : MPIR_CVAR_ASYNC_PROGRESS
      category    : THREADS
      type        : boolean
      default     : false
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If set to true, MPICH will initiate an additional thread to
        make asynchronous progress on all communication operations
        including point-to-point, collective, one-sided operations and
        I/O.  Setting this variable will automatically increase the
        thread-safety level to MPI_THREAD_MULTIPLE.  While this
        improves the progress semantics, it might cause a small amount
        of performance overhead for regular MPI operations.  The user
        is encouraged to leave one or more hardware threads vacant in
        order to prevent contention between the application threads
        and the progress thread(s).  The impact of oversubscription is
        highly system dependent but may be substantial in some cases,
        hence this recommendation.

    - name        : MPIR_CVAR_DEFAULT_THREAD_LEVEL
      category    : THREADS
      type        : string
      default     : "MPI_THREAD_SINGLE"
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Sets the default thread level to use when using MPI_INIT. This variable
        is case-insensitive.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* -- Begin Profiling Symbol Block for routine MPI_Init */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Init = PMPI_Init
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Init  MPI_Init
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Init as PMPI_Init
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Init(int *argc, char ***argv) __attribute__((weak,alias("PMPI_Init")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Init
#define MPI_Init PMPI_Init

/* Fortran logical values. extern'd in mpiimpl.h */
/* MPI_Fint MPIR_F_TRUE, MPIR_F_FALSE; */

/* Any internal routines can go here.  Make them static if possible */

/* must go inside this #ifdef block to prevent duplicate storage on darwin */
int MPIR_async_thread_initialized = 0;
#endif

#undef FUNCNAME
#define FUNCNAME MPI_Init

/*@
   MPI_Init - Initialize the MPI execution environment

Input Parameters:
+  argc - Pointer to the number of arguments 
-  argv - Pointer to the argument vector

Thread and Signal Safety:
This routine must be called by one thread only.  That thread is called
the `main thread` and must be the thread that calls 'MPI_Finalize'.

Notes:
   The MPI standard does not say what a program can do before an 'MPI_INIT' or
   after an 'MPI_FINALIZE'.  In the MPICH implementation, you should do
   as little as possible.  In particular, avoid anything that changes the
   external state of the program, such as opening files, reading standard
   input or writing to standard output.

Notes for C:
    As of MPI-2, 'MPI_Init' will accept NULL as input parameters. Doing so
    will impact the values stored in 'MPI_INFO_ENV'.

Notes for Fortran:
The Fortran binding for 'MPI_Init' has only the error return
.vb
    subroutine MPI_INIT( ierr )
    integer ierr
.ve

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_INIT

.seealso: MPI_Init_thread, MPI_Finalize
@*/
int MPI_Init( int *argc, char ***argv )
{
    static const char FCNAME[] = "MPI_Init";
    int mpi_errno = MPI_SUCCESS;
    int rc ATTRIBUTE((unused));
    int threadLevel, provided;
	

    MPID_MPI_INIT_STATE_DECL(MPID_STATE_MPI_INIT);

    /* Handle mpich_state in case of Re-init */
    if (OPA_load_int(&MPIR_Process.mpich_state) == MPICH_POST_FINALIZED) {
        OPA_store_int(&MPIR_Process.mpich_state, MPICH_PRE_INIT);
    }
    rc = MPID_Wtime_init();
#ifdef USE_DBG_LOGGING
    MPIU_DBG_PreInit( argc, argv, rc );
#endif

    MPID_MPI_INIT_FUNC_ENTER(MPID_STATE_MPI_INIT);
#if defined(CHANNEL_PSM)
    MV2_Read_env_vars();
#endif /* defined(CHANNEL_PSM) */

#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            if (OPA_load_int(&MPIR_Process.mpich_state) != MPICH_PRE_INIT) {
                mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
						  "**inittwice", NULL );
	    }
            if (mpi_errno) goto fn_fail;
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */

    /* Temporarily disable thread-safety.  This is needed because the
     * mutexes are not initialized yet, and we don't want to
     * accidentally use them before they are initialized.  We will
     * reset this value once it is properly initialized. */
#if defined MPICH_IS_THREADED
    MPIR_ThreadInfo.isThreaded = 0;
#endif /* MPICH_IS_THREADED */

    MPIR_T_env_init();

    if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_MULTIPLE"))
        threadLevel = MPI_THREAD_MULTIPLE;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_SERIALIZED"))
        threadLevel = MPI_THREAD_SERIALIZED;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_FUNNELED"))
        threadLevel = MPI_THREAD_FUNNELED;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_SINGLE"))
        threadLevel = MPI_THREAD_SINGLE;
    else {
        MPL_error_printf("Unrecognized thread level %s\n", MPIR_CVAR_DEFAULT_THREAD_LEVEL);
        exit(1);
    }

    /* If the user requested for asynchronous progress, request for
     * THREAD_MULTIPLE. */
    if (MPIR_CVAR_ASYNC_PROGRESS)
        threadLevel = MPI_THREAD_MULTIPLE;

    mpi_errno = MPIR_Init_thread( argc, argv, threadLevel, &provided );
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    if (MPIR_CVAR_ASYNC_PROGRESS) {
        if (provided == MPI_THREAD_MULTIPLE) {
            mpi_errno = MPIR_Init_async_thread();
            if (mpi_errno) goto fn_fail;

            MPIR_async_thread_initialized = 1;
        }
        else {
            printf("WARNING: No MPI_THREAD_MULTIPLE support (needed for async progress)\n");
        }
    }

#if defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_PSM)
    /* initialize the two level communicator for MPI_COMM_WORLD  */
    if (mv2_use_osu_collectives && 
            mv2_enable_shmem_collectives) {

       MPID_Comm *comm_ptr = NULL;
       MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
       int flag=0; 
       PMPI_Comm_test_inter(comm_ptr->handle, &flag);

       if(flag == 0 && comm_ptr->dev.ch.shmem_coll_ok == 0 &&
               comm_ptr->local_size < mv2_two_level_comm_early_init_threshold &&
               check_split_comm(pthread_self())) { 

            disable_split_comm(pthread_self());
            mpi_errno = create_2level_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
            enable_split_comm(pthread_self());
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
       } 
    }
#endif /*defined(CHANNEL_MRAIL_GEN2) || defined(CHANNEL_PSM)*/


/* Added by Abu Naser */
    security_approach = 0;
    MPI_SEC_Initial_Key_Aggrement();
   
    char *s_value, *o_value, *t_value, *sml_value, a_value, *c_value, *cb_value, *b_value, *ob_value;
    char *rl_value;
    if ((s_value = getenv("MV2_SECURITY_APPROACH")) != NULL) {   // Mohsen: "MV2_" is appended to make Flags uniform
        security_approach = (atoi(s_value));
    }

   /****************************** Added by Mehran *****************************/
     MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);

     overlap_decryption = 0;
    if ((o_value = getenv("MV2_OVERLAP_DECRYPTION")) != NULL) {  // Mohsen: "MV2_" is appended to make Flags uniform
        overlap_decryption = (atoi(o_value));
    }
 
    allocated_shmem = 0;
    if ((t_value = getenv("MV2_INTER_ALLGATHER_TUNING")) != NULL) {
        int alg = (atoi(t_value));
        if(alg == 14 || alg == 18 || alg == 20){
            allocated_shmem = 1;
            if(security_approach!=0){
                allocated_shmem = 2;
            }
        }
    }

    int initialize_rank_list = 0;
    if ((a_value = getenv("MV2_ALLTOALL_TUNING")) != NULL) {
        int alg = (atoi(a_value));
        if(alg == 5){
            if(allocated_shmem == 0){
                allocated_shmem = 1;
            }
            initialize_rank_list = 1;
        }
    }
    /********************** Added by Mohsen **************************/
	
	char *tc;
	char *value;
	
	if ((value = getenv("MV2_LEADER_CNT")) != NULL) 
    leader_cnt = atoi(value);

	if ((value = getenv("MV2_Allgather_Reduce")) != NULL) 
    Allgather_Reduce = atoi(value);

	if ((value = getenv("MV2_CHOPING_SIZE")) != NULL) 
    choping_sz = atoi(value);

	if ((value = getenv("MV2_SUPER_NODE")) != NULL) 
    super_node = atoi(value);

	tc = getenv("MV2_CONCUR_RS_METHOD");   // Reduce-Scatter in 1st step of Ring

	if (tc && (strncmp(tc, "2",2) == 0))	{	
		CONCUR_RS_METHOD = 2;	// Pt2pt-method for 1  &&  ShMem-method for 2
	}
	
	tc = getenv("MV2_CONCUR_AllGTHER_METHOD"); // Reduce-Scatter in 3rf step of Ring

	if (tc && (strncmp(tc, "2",2) == 0))	{	
		CONCUR_AllGTHER_METHOD = 2;	  // Pt2pt-method for 1  &&  ShMem-method for 2
	}
	
	tc = getenv("MV2_SHMEM_BCAST");

	if (tc && (strncmp(tc, "1",1) == 0))	{	
		SHMEM_BCAST = 1;	
	}

    tc = getenv("MV2_UNSEC_ALLREDUCE_MULTI_LEADER");

	if (tc && (strncmp(tc, "1",1) == 0))	{	
		UNSEC_ALLREDUCE_MULTI_LEADER = 1;	
	}
	
	// RD-method for 1  &&  RSA-method for 2  &&  Ring for 3 && Alg for 4	
	if ((value = getenv("MV2_CONCUR_INTER_METHOD")) != NULL) 
    CONCUR_INTER_METHOD = atoi(value);

	if ((CONCUR_INTER_METHOD != 1) && (CONCUR_INTER_METHOD != 2) && (CONCUR_INTER_METHOD != 3) ){ 
		CONCUR_INTER_METHOD = 1;
		if (comm_ptr->rank==0) fprintf(stderr,COLOR_YELLOW"CryptoMPI Warning: CONCUR_INTER_METHOD is out of range. It has been set to 1."COLOR_RESET"\n");
	}

	if ((super_node == 1) || (CONCUR_AllGTHER_METHOD == 2) || (SHMEM_BCAST == 1) || (CONCUR_RS_METHOD == 2)) {
        allocated_shmem = 1;        
        if (comm_ptr->rank==0) fprintf(stderr,COLOR_GREEN"CryptoMPI: Allreduce shared memory creation..."COLOR_RESET"\n");
    }
	
	enc_choping_sz = choping_sz + ENC_MSG_TAG_SIZE;
	
    /****************************** End by Mohsen *****************************/

    /***********************Added by Cong******************************/
    if ((b_value = getenv("MV2_INTER_BCAST_TUNING")) != NULL) {
        int alg = (atoi(b_value));               
        if(alg == 13) {
            allocated_shmem = 2; 
            if (comm_ptr->rank==0) fprintf(stderr,COLOR_GREEN"CryptoMPI: Bcast shared memory creation..."COLOR_RESET"\n");
        }
    }  
    /****************************** End by Cong *****************************/

    if(allocated_shmem != 0){
        init_shmem();
    }

    shmem_leaders = 1;
    if ((sml_value = getenv("MV2_SHMEM_LEADERS")) != NULL) {    // Mohsen: "MV2_" is appended to make Flags uniform
        shmem_leaders = (atoi(sml_value));
    }

    concurrent_comm = 0;
    if ((c_value = getenv("MV2_CONCURRENT_COMM")) != NULL) {    // Mohsen: "MV2_" is appended to make Flags uniform
        concurrent_comm = (atoi(c_value));
    }
    concurrent_bcast = 0;
    if ((cb_value = getenv("MV2_CONCURRENT_BCAST")) != NULL) {  // Mohsen: "MV2_" is appended to make Flags uniform
        concurrent_bcast = (atoi(cb_value));
    }

    if (concurrent_comm == 1){
        // MPID_Comm *comm_ptr = NULL;
        // MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
        mpi_errno = create_concurrent_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    }

    if ((rl_value = getenv("INIT_RANK_LIST")) != NULL) {
        initialize_rank_list = (atoi(c_value));
    }
    if(initialize_rank_list == 1 && NULL == comm_ptr->dev.ch.rank_list){
        MPIR_Errflag_t errflag = MPIR_ERR_NONE;
        mpi_errno = create_allgather_comm(comm_ptr, &errflag);
        if(mpi_errno) {
            MPIR_ERR_POP(mpi_errno);
        }
    } 

   /****************************** End by Mehran ***************************************/

 

    if (allocated_shmem == 0)
    {
        if ((t_value = getenv("MV2_SCATTER_SHRD_MEM")) != NULL) // Mohsen: "MV2_" is appended to make Flags uniform
        {
            int alg = (atoi(t_value));
            if (alg == 1)
            {
                allocated_shmem = 2;
                init_shmem();
                //printf("shared memory creation done\n");fflush(stdout);
            }
        }
    }

      if ((t_value = getenv("MV2_PIPELINE_SIZE")) != NULL)  // Mohsen: "MV2_" is appended to make Flags uniform
    {
        choping_sz = (atoi(t_value));
    }
   
    init_crypto();
/* end of add */

    
    /* ... end of body of routine ... */
    MPID_MPI_INIT_FUNC_EXIT(MPID_STATE_MPI_INIT);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_REPORTING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, 
	    "**mpi_init", "**mpi_init %p %p", argc, argv);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( 0, FCNAME, mpi_errno );
    return mpi_errno;
    /* --END ERROR HANDLING-- */
}


/* Added by Abu Naser */

void init_openmp_key(){

    //if(OPENMP_MULTI_THREAD_ONLY)
    if(security_approach==600 || security_approach==601)
    {    
        global_openmp_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);
        global_coll_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);
        global_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);
        ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);                                                                          
    }
    //else if(OPENMP_PIPE_LINE)
    else if(security_approach==602)
    {
         global_small_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             &symmetric_key[symmetric_key_size*2],
                                             symmetric_key_size, 0);
         global_coll_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             &symmetric_key[symmetric_key_size*2],
                                             symmetric_key_size, 0);
        global_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             &symmetric_key[symmetric_key_size*2],
                                             symmetric_key_size, 0);
        ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             &symmetric_key[symmetric_key_size*2],
                                             symmetric_key_size, 0);                                                                          
    }
#if 0 //CRYPTMPI_COUNTER_MODE    
    else if(CRYPTMPI_COUNTER_MODE){
        global_counter_ctx = EVP_CIPHER_CTX_new();                                     
        EVP_EncryptInit_ex(global_counter_ctx, EVP_aes_128_ctr(), NULL, symmetric_key, NULL);                                     
    }
#endif

    
    
    /* Get the number of threads */
    openmp_active_thread_no = 1;
    int i;
//#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION
    if (security_approach == 602)
    {
        for (i = 0; i < MAX_RANKS_LIMIT; i++)
        {
            ranksIrecvS[i] = 0;
            pendingIsendRequestCount[i] = 0;
        }
    }
//#endif

    no_of_max_omp_threads = omp_get_num_procs();  
    MPI_Comm_size(MPI_COMM_WORLD, &total_process_number);
     MPID_Comm *cryptmpi_comm_ptr = NULL;
   
    MPID_Comm_get_ptr( MPI_COMM_WORLD, cryptmpi_comm_ptr );
    MPI_Comm_rank(MPI_COMM_WORLD, &cryptmpi_own_rank);

    cryptmpi_local_rank = cryptmpi_comm_ptr->intranode_table[cryptmpi_own_rank];
    int  num_local, local_rank, local_procs, num_external, external_rank, external_procs;
    int mpi_errno = MPIU_Find_local_and_external(cryptmpi_comm_ptr,
                                                 &num_local, &local_rank, &local_procs,
                                                 &num_external, &external_rank, &external_procs,
                                                 &cryptmpi_comm_ptr->intranode_table, &cryptmpi_comm_ptr->internode_table);
    
   
     cryptmpi_local_process = num_local;
     int len;
    MPI_Get_processor_name(cryptmpi_process_name, &len );
    for(i=0;i<total_process_number;i++){
         sameNode[i]= -1;
         Crypthandle_probe[i].no_tag = 0;
         //sendtag[i] = 0;
         //recvtag[i] = 0;
    }
     MPI_Allgather( cryptmpi_process_name, len, MPI_CHAR, all_p_names, len, MPI_CHAR, MPI_COMM_WORLD );
    
    for(i=0;i<total_process_number;i++){
        if(strncmp(cryptmpi_process_name, all_p_names+(i*len), len ) == 0 ){
            sameNode[i] = 1;
        }
    }



//#if OMP_DYNAMIC_THREADS_PIPELINE || OMP_DYNAMIC_THREADS_PIPELINE_INNOVATION || OMP_DYNAMIC_THREADS || OMP_DYNAMIC_THREADS_INNOVATION
    if (security_approach == 602)
    {
        int hybrid = 0;
        int spread = 0;
        char *t = getenv("MV2_CPU_BINDING_POLICY");
        if (t)
        {
            if ((strncmp(t, "hybrid", 6) == 0) || (strncmp(t, "HYBRID", 6) == 0))
                hybrid = 1;
        }

        t = getenv("MV2_HYBRID_BINDING_POLICY");
        if (t)
        {
            if ((strncmp(t, "spread", 6) == 0) || (strncmp(t, "SPREAD", 6) == 0))
                spread = 1;
        }

        //printf("hybrid =%d spread = %d\n",hybrid, spread);fflush(stdout);
        if (hybrid == 1 && spread == 1)
            cyptmpi_series_thread = no_of_max_omp_threads - noCommThreads; // liberal, in case of hybrid+spread
        else
            cyptmpi_series_thread = (no_of_max_omp_threads / 2) / cryptmpi_local_process; // liberal

        // printf("cyptmpi_series_thread =%d no_of_max_omp_threads = %d noCommThreads =%d cryptmpi_local_process=%d\n",cyptmpi_series_thread,no_of_max_omp_threads,noCommThreads,cryptmpi_local_process);fflush(stdout);
        //cyptmpi_series_thread = no_of_max_omp_threads - noCommThreads; // liberal, in case of hybrid+spread

        if (cyptmpi_series_thread < 1)
            cyptmpi_series_thread = 1;
    }
//#endif

    return;
}


/* Init counter mode keys */
void init_counter_mode_keys(){
   
    
    global_counter_ctx = EVP_CIPHER_CTX_new(); 
    base_counter_ctx =  EVP_CIPHER_CTX_new();                                    
    EVP_EncryptInit_ex(global_counter_ctx, EVP_aes_128_ctr(), NULL, symmetric_key, NULL);
                                      
    /* set init is done */
    cryptmpi_init_done = 1;
    
    /* Get the number of threads */ 
    openmp_active_thread_no = 1;
 int i;
//#if PRE_COMPUTE_COUNTER_MODE
 if (security_approach == 702)
 {
     for (i = 0; i < MAX_RANKS_LIMIT; i++)
     {
         ranksIrecvS[i] = 0;
         pendingIsendRequestCount[i] = 0;
     }
     memset(zeros, 0, MAX_COMMON_COUNTER_SZ);
 }
//#endif

 no_of_max_omp_threads = omp_get_num_procs();  
    MPI_Comm_size(MPI_COMM_WORLD, &total_process_number);
     MPID_Comm *cryptmpi_comm_ptr = NULL;
   
    MPID_Comm_get_ptr( MPI_COMM_WORLD, cryptmpi_comm_ptr );
    MPI_Comm_rank(MPI_COMM_WORLD, &cryptmpi_own_rank);

    cryptmpi_local_rank = cryptmpi_comm_ptr->intranode_table[cryptmpi_own_rank];
    int  num_local, local_rank, local_procs, num_external, external_rank, external_procs;
    int mpi_errno = MPIU_Find_local_and_external(cryptmpi_comm_ptr,
                                                 &num_local, &local_rank, &local_procs,
                                                 &num_external, &external_rank, &external_procs,
                                                 &cryptmpi_comm_ptr->intranode_table, &cryptmpi_comm_ptr->internode_table);
    
   
     cryptmpi_local_process = num_local;
     int len;
    MPI_Get_processor_name(cryptmpi_process_name, &len );
    for(i=0;i<total_process_number;i++){
         sameNode[i]= -1;
         Crypthandle_probe[i].no_tag = 0;
    }
     MPI_Allgather( cryptmpi_process_name, len, MPI_CHAR, all_p_names, len, MPI_CHAR, MPI_COMM_WORLD );
    
    for(i=0;i<total_process_number;i++){
        if(strncmp(cryptmpi_process_name, all_p_names+(i*len), len ) == 0 ){
            sameNode[i] = 1;
        }
    }

//#if PRE_COMPUTE_COUNTER_MODE
    if (security_approach == 702)
    {
        /* init common counter array */
        //RAND_bytes(Send_common_IV, 16);
        RAND_bytes(Send_common_IV, 32);
        if (INITIAL_COMMON_COUNTER_SZ)
        {
            EVP_EncryptInit_ex(global_counter_ctx, NULL, NULL, NULL, Send_common_IV);
            EVP_EncryptUpdate(global_counter_ctx, enc_common_buffer, &len, zeros, INITIAL_COMMON_COUNTER_SZ);
            enc_common_counter = INITIAL_COMMON_COUNTER_SZ / 16;
            common_compute_size = INITIAL_COMMON_COUNTER_SZ;
            enc_common_start = 0;
            enc_common_end = INITIAL_COMMON_COUNTER_SZ;
            counter_needto_send = 0;
        }
        else
        {
            enc_common_counter = 0;
            common_compute_size = 0;
            enc_common_start = 0;
            enc_common_end = 0;
            counter_needto_send = 0;
        }
        mpi_errno = MPI_Allgather(Send_common_IV, 32, MPI_UNSIGNED_CHAR, Recv_common_IV, 32, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
    }
//#elif BASE_COUNTER_MODE
    else // 700 and 701
    {
        RAND_bytes(Send_common_IV, 16);
        base_global_counter = 0;
        mpi_errno = MPI_Allgather(Send_common_IV, 16, MPI_UNSIGNED_CHAR, Recv_common_IV, 16, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
    }
//#endif

//#if PRE_COMPUTE_COUNTER_MODE
    if (security_approach == 702)
    {
        int hybrid = 0;
        int spread = 0;
        char *t = getenv("MV2_CPU_BINDING_POLICY");
        if (t)
        {
            if ((strncmp(t, "hybrid", 6) == 0) || (strncmp(t, "HYBRID", 6) == 0))
                hybrid = 1;
        }

        t = getenv("MV2_HYBRID_BINDING_POLICY");
        if (t)
        {
            if ((strncmp(t, "spread", 6) == 0) || (strncmp(t, "SPREAD", 6) == 0))
                spread = 1;
        }

        //printf("hybrid =%d spread = %d\n",hybrid, spread);fflush(stdout);
        if (hybrid == 1 && spread == 1)
            cyptmpi_series_thread = no_of_max_omp_threads - noCommThreads; // liberal, in case of hybrid+spread
        else
            cyptmpi_series_thread = (no_of_max_omp_threads / 2) / cryptmpi_local_process; // liberal

        if (cyptmpi_series_thread < 1)
            cyptmpi_series_thread = 1;

        for (i = 0; i < cyptmpi_series_thread; i++)
        {
            local_counter_ctx[i] = EVP_CIPHER_CTX_new();
            EVP_EncryptInit_ex(local_counter_ctx[i], EVP_aes_128_ctr(), NULL, symmetric_key, NULL);
        }
    }
//#endif
    return;
    }

//void init_crypto(unsigned char *key){
void init_crypto(){    
   
    ctx_enc = EVP_CIPHER_CTX_new();
	
    if(symmetric_key_size == 16){
	    EVP_EncryptInit_ex(ctx_enc, EVP_aes_128_ecb(), NULL, symmetric_key, NULL);
    }
	else{
        EVP_EncryptInit_ex(ctx_enc, EVP_aes_256_ecb(), NULL, symmetric_key, NULL); 
    }

/*#if CRYPTMPI_COUNTER_MODE
    init_counter_mode_keys();   // will initialize counter-mode
#else
     init_openmp_key();         // will initialize cryptMPI-AesGcm
#endif*/

/************ Add by Mohsen *************/

    if(security_approach==2005 || security_approach==2001)
    {    
        //local_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
          //                                   symmetric_key,
            //                                 symmetric_key_size, 0);
        global_coll_msg_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);                                                                               
    }

/************ End by Mohsen *************/    

if (security_approach == 600 || security_approach == 601 || security_approach == 602)
    {
         init_openmp_key();         // will initialize cryptMPI-AesGcm
    }
    else if(security_approach == 700 || security_approach == 701 || security_approach == 702)
    {
         init_counter_mode_keys();   // will initialize counter-mode
    }
    else 
    {
        global_ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);
        ctx = EVP_AEAD_CTX_new(EVP_aead_aes_128_gcm(),
                                             symmetric_key,
                                             symmetric_key_size, 0);                                     
        //printf("Valid security_approach values are: 600, 602, 700, and 702\n");fflush(stdout);
    }

    return;                          
}


void MPI_SEC_Initial_Key_Aggrement(){    
    int wrank, wsize;
    int mpi_errno = MPI_SUCCESS;   
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

   
    int keylen, i, ret;
    unsigned char  *root_public_key, *public_key, *all_process_public_key;
    unsigned char  *encrypted_text;
    unsigned char recv_buf[3000];
    int encrypted_len, decrypted_len, pub_key_size, next;
    MPI_Status status;
    BIGNUM *bn;
    BIGNUM *bnPublic = BN_new();
    BIGNUM *exponent = BN_new();
    BIGNUM *bnPrivate = BN_new();
 

    bn = BN_new();
    BN_set_word(bn, RSA_F4);

    RSA *rsaKey, *temprsa;
    rsaKey = RSA_new();
    temprsa = RSA_new();

    /* Generate rsa keypair */
    RSA_generate_key_ex(rsaKey,  2048, bn,  NULL);

    /* Get the public key and exponent */
    RSA_get0_key(rsaKey, &bnPublic, &exponent, &bnPrivate);

   
    all_process_public_key = (unsigned char *)MPIU_Malloc(wsize*256+10);
    encrypted_text = (unsigned char *)MPIU_Malloc(wsize*256+10);

    pub_key_size = BN_num_bytes(bnPublic);
    public_key = (unsigned char *) malloc(256+10);
    ret = BN_bn2bin(bnPublic, public_key);

    /* send the public key to root process */ 
    /*mpi_errno = MPI_Gather(public_key, 256, MPI_UNSIGNED_CHAR,
               all_process_public_key, 256, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);*/
    
    int *disp = (int *)malloc(wsize*sizeof(int));
    int *sendcnt = (int *)malloc(wsize*sizeof(int));
    int *recvcnt = (int *)malloc(wsize*sizeof(int));   
    
    for(i=0;i<wsize;i++)
    {
        disp[i] = i*256;
        recvcnt[i] = 256;
    }
    mpi_errno = MPI_Gatherv(public_key, 256, MPI_UNSIGNED_CHAR,
               all_process_public_key, recvcnt, disp, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);     


    /* set the key size */
    if( SYMMETRIC_KEY_SIZE == 32)
          symmetric_key_size = 32;

    if( wrank ==0 ){  
        BIGNUM *bnOthPubkey = BN_new();
        
        /* Generate a random key */
        RAND_bytes(symmetric_key, (symmetric_key_size*2));
        
        symmetric_key[symmetric_key_size*2] = '\0';
       

        int next;
        /* Encrypt random key with the public key of other process */
          for(i=1; i<wsize; i++){  
            next = (i*256);
            
            bnOthPubkey = BN_bin2bn((all_process_public_key+next), 256, NULL );
          
            temprsa = NULL;
            temprsa = RSA_new();
            if(RSA_set0_key(temprsa, bnOthPubkey, exponent, NULL)){  
                next = i* 256;
                ret = RSA_public_encrypt((symmetric_key_size*2), (unsigned char*)symmetric_key, (unsigned char*)(encrypted_text+next), 
                                        temprsa, RSA_PKCS1_OAEP_PADDING); 

                if(ret!=-1){
                #if ENABLE_SECURE_MPI_DEBUG                  
                    printf("[Rank %d] Encrypted %d bytes for %d\n",wrank, ret, i); fflush(stdout);
                #endif                    
                }
                else{
                     printf("[Rank %d] Encryption failed for for %d\n",wrank,  i); fflush(stdout);   
                }                         
            }
            else{
                printf("RSA_set0_key: Failed in %d for %d\n",wrank, i); fflush(stdout);
            }

        }
        
    }
   
    /* send/recv encrypted symmetric key from/to processes */
    /*mpi_errno = MPI_Scatter(encrypted_text, 256, MPI_UNSIGNED_CHAR, recv_buf, 256, 
                                    MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);*/

     for(i=0;i<wsize;i++)
    {
        disp[i] = i*256;
        sendcnt[i] = 256;
    } 

     mpi_errno = MPI_Scatterv(encrypted_text, sendcnt, disp, MPI_UNSIGNED_CHAR, recv_buf, 256, 
                                    MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);                               
                           

     if( wrank != 0 ){

        /* Now decrypt the key */
         ret = RSA_private_decrypt(256, (unsigned char*)recv_buf, (unsigned char*)symmetric_key,
                       rsaKey, RSA_PKCS1_OAEP_PADDING);

        if(ret!=-1){
            #if ENABLE_SECURE_MPI_DEBUG             
            printf("[Rank %d] Decrypted size is %d\n",wrank, ret); 
            //symmetric_key[ret] = '\0';
            //printf("[%d] symmetric key is: %s\n",wrank, symmetric_key);
            fflush(stdout);
            #endif            
        } 
        else{
                printf("RSA_private_decrypt: Failed in %d\n",wrank);
              
                fflush(stdout);
            }              
                                   

    }
    /* initialize with key */
    //init_crypto(symmetric_key);
    MPIU_Free(encrypted_text);
    MPIU_Free(all_process_public_key);
    free(sendcnt);
    free(recvcnt);
    free(disp);
    return;
}

int init_shmem(){
    static const char FCNAME[] = "init_shmem";
    int mpi_errno = MPI_SUCCESS;
    int security_approach, overlap_decryption;
    //printf("Hello from init_shmem\n");
    MPID_Comm *comm_ptr = NULL;
    MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
    MPID_Comm *shmem_comm_ptr = NULL;
    MPID_Comm_get_ptr(comm_ptr->dev.ch.shmem_comm, shmem_comm_ptr);

       //TODO: Allocate Shmem
    size_t shmem_size = (comm_ptr->local_size) * 4 * 1024 *1024;
    size_t ciphertext_shmem_size = (comm_ptr->local_size) * (1024 * 1024 * 4 + 16 + 12);
    shmem_key = 12345; //32984;
    ciphertext_shmem_key = 67890; //56982;

    if(shmem_comm_ptr->rank == 0){
        
        shmid = shmget(shmem_key, shmem_size, IPC_CREAT | 0666);

        if(allocated_shmem==2){
            ciphertext_shmid = shmget(ciphertext_shmem_key, ciphertext_shmem_size, IPC_CREAT | 0666);

        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(shmem_comm_ptr->rank > 0){
        shmid = shmget(shmem_key, shmem_size, 0666);

        if(allocated_shmem==2){
            ciphertext_shmid = shmget(ciphertext_shmem_key, ciphertext_shmem_size, 0666);

        }
    }
    
    if (shmid < 0) {
        printf("%s",strerror(errno));
        printf("ERROR 1\n");
        goto fn_fail;
    }
    
    // attach shared memory 

    shmem_buffer = (void *) shmat(shmid, NULL, 0);
    if (shmem_buffer == (void *) -1) {
        printf("ERROR 2\n");
        goto fn_fail;
    }
    if(allocated_shmem==2){
        ciphertext_shmem_buffer = (void *) shmat(ciphertext_shmid, NULL, 0);
        if (ciphertext_shmem_buffer == (void *) -1) {
            printf("ERROR 3\n");
            goto fn_fail;
        }

    }

    return mpi_errno;
    
    fn_fail:
        mpi_errno = MPIR_Err_return_comm( 0, FCNAME, mpi_errno );
    
    return mpi_errno;
    
}

int create_concurrent_comm (MPI_Comm comm, int size, int my_rank)
{
    static const char FCNAME[] = "create_concurrent_comm";
    int mpi_errno = MPI_SUCCESS;
    MPID_Comm* comm_ptr = NULL;

    MPID_Comm_get_ptr( comm, comm_ptr );
    if (size <= 1) {
        return mpi_errno;
    }

    
    comm_ptr->dev.ch.concurrent_comm =MPI_COMM_NULL;
    
    MPID_Comm* shmem_commptr;
    MPI_Comm shmem_comm = comm_ptr->dev.ch.shmem_comm;
    MPID_Comm_get_ptr(shmem_comm, shmem_commptr);
    
    /* get our rank and the size of this communicator */
    int local_rank = shmem_commptr->rank;


    mpi_errno = PMPI_Comm_split(comm, local_rank, my_rank, &(comm_ptr->dev.ch.concurrent_comm));
    if(mpi_errno) {
       MPIR_ERR_POP(mpi_errno);
    }

    return (mpi_errno);
    fn_fail:
        mpi_errno = MPIR_Err_return_comm( 0, FCNAME, mpi_errno );

    return (mpi_errno);
}




/* end of add */